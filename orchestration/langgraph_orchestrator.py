# orchestration/langgraph_orchestrator.py
"""Orchestrate LangGraph-based SAGA narrative generation.

This module defines the orchestration boundary around the LangGraph workflow:
Neo4j connectivity, workflow checkpoint lifecycle, and an explicit LLM HTTP-client
lifecycle that is shared across workflow nodes.

Error/cleanup policy at this boundary is intentionally mixed:

- Strict: Neo4j connection/setup failures and workflow construction/streaming
  errors propagate to the caller.
- Best-effort: restoring patched `llm_service` module attributes is attempted
  during cleanup and must not mask the original workflow failure.
"""

import importlib
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import structlog

import config
from core.db_manager import neo4j_manager
from core.langgraph.initialization.validation import validate_initialization_artifacts
from core.langgraph.state import NarrativeState, create_initial_state
from core.langgraph.workflow import create_checkpointer, create_full_workflow_graph
from core.llm_interface_refactored import async_llm_context
from data_access import chapter_queries
from ui.rich_display import RichDisplayManager

logger = structlog.get_logger(__name__)


# Modules that directly import `llm_service` via `from core.llm_interface_refactored import llm_service`.
#
# Because that pattern binds the object into each module namespace at import-time,
# orchestrator/workflow boundaries must explicitly override those module attributes
# for the duration of a workflow run to ensure the underlying HTTP client lifecycle
# is managed (opened/closed) deterministically.
_LLM_SERVICE_PATCH_MODULES: tuple[str, ...] = (
    # Root singleton module (defensive; some tests patch this directly)
    "core.llm_interface_refactored",
    # LangGraph generation + extraction + embedding + revision + summary
    "core.langgraph.nodes.generation_node",
    "core.langgraph.nodes.embedding_node",
    "core.langgraph.nodes.extraction_nodes",
    "core.langgraph.nodes.revision_node",
    "core.langgraph.nodes.summary_node",
    # LangGraph initialization nodes
    "core.langgraph.initialization.character_sheets_node",
    "core.langgraph.initialization.global_outline_node",
    "core.langgraph.initialization.act_outlines_node",
    "core.langgraph.initialization.chapter_outline_node",
    "core.langgraph.initialization.commit_init_node",
    # Validation subgraph (LLM-based quality eval + world rule checks)
    "core.langgraph.subgraphs.validation",
    # Services invoked by workflow nodes that also import `llm_service`
    "core.graph_healing_service",
    "core.relationship_normalization_service",
    # Defensive: other nodes occasionally used in graphs
    "core.langgraph.nodes.context_retrieval_node",
    "core.langgraph.nodes.scene_planning_node",
)


@asynccontextmanager
async def _managed_llm_lifecycle_for_workflow() -> Any:
    """Manage an explicit LLM HTTP-client lifecycle for a workflow run.

    This context manager creates a fresh `llm_service` instance via
    [`async_llm_context()`](core/llm_interface_refactored.py:37) and temporarily
    patches workflow-related modules that import `llm_service` into their module
    namespace at import time.

    The patching is scoped to the context manager and is intended to make the
    orchestrator/workflow boundary responsible for client cleanup, rather than
    individual nodes.

    Yields:
        The managed `llm_service` instance used by the workflow.

    Notes:
        - Missing or optional modules are ignored during patching.
        - Restoring prior module attributes is best-effort and must not mask an
          error raised by the workflow itself.
    """
    async with async_llm_context() as (managed_llm_service, _embedding_service):
        patched: list[tuple[ModuleType, Any]] = []

        for module_name in _LLM_SERVICE_PATCH_MODULES:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                # Defensive: missing/optional modules should not break orchestration.
                continue

            module_any = cast(Any, module)
            if hasattr(module_any, "llm_service"):
                patched.append((module, module_any.llm_service))
                module_any.llm_service = managed_llm_service

        try:
            yield managed_llm_service
        finally:
            # Restore in reverse order for sanity.
            for module, previous in reversed(patched):
                try:
                    cast(Any, module).llm_service = previous
                except Exception:
                    # Best-effort restore; failing to restore should not mask the
                    # original workflow error.
                    continue


class LangGraphOrchestrator:
    """Orchestrate a LangGraph-based narrative generation run.

    This class establishes the high-level lifecycle boundaries for a single run:

    - Connect to Neo4j and ensure schema exists.
    - Create a fresh [`NarrativeState`](core/langgraph/state.py:1) seed for the
      workflow, including initialization detection based on on-disk artifacts.
    - Run the LangGraph workflow under a checkpointer context and a managed LLM
      client context.
    - Stream workflow events to drive UI updates and structured logging.

    Notes:
        - Checkpoint persistence is owned by the workflow checkpointer context,
          not by the state creation step.
        - The per-chapter loop is best-effort: certain chapter-level failures
          stop generation early without raising to the caller.
    """

    def __init__(self) -> None:
        logger.info("Initializing LangGraph Orchestrator...")
        # Use settings.BASE_OUTPUT_DIR which is the Pydantic field
        self.project_dir = Path(config.settings.BASE_OUTPUT_DIR)
        self.checkpointer_path = self.project_dir / ".saga" / "checkpoints.db"

        # Initialize Rich display for progress tracking
        self.display = RichDisplayManager()
        self.run_start_time: float = 0.0

        logger.info("LangGraph Orchestrator initialized.")

    async def run_novel_generation_loop(self) -> None:
        """Run the end-to-end LangGraph novel generation loop.

        This method owns the orchestration boundary and associated cleanup:

        - Starts the Rich progress display (if enabled).
        - Establishes a Neo4j connection and ensures the database schema exists.
        - Creates a fresh workflow state seed for the run.
        - Runs the workflow under:
          - a managed LLM client lifecycle context, and
          - a checkpointer context bound to a persistent SQLite file.

        Error policy:
            - Neo4j connection/setup failures and workflow construction/streaming
              failures propagate to the caller.
            - Chapter generation is best-effort: chapter-level failures inside
              [`_run_chapter_generation_loop()`](orchestration/langgraph_orchestrator.py:273)
              are logged and stop the loop without raising.

        Cleanup:
            The Rich display is stopped in a `finally` block. If display shutdown
            raises, that exception propagates (and can mask a prior error).

        Side Effects:
            - Creates/updates a checkpoint database at `project_dir/.saga/checkpoints.db`.
            - Executes workflow nodes that may write files and mutate Neo4j state.
        """
        logger.info("=" * 60)
        logger.info("SAGA: LangGraph-based Novel Generation Starting")
        logger.info("=" * 60)

        # Start Rich display for progress tracking
        self.run_start_time = time.time()
        self.display.start()

        try:
            # Step 1: Connect to Neo4j
            await self._ensure_neo4j_connection()

            # Step 2: Load or create state
            state = await self._load_or_create_state()

            # Step 3: Create workflow with checkpointing
            # AsyncSqliteSaver.from_conn_string() returns an async context manager
            #
            # IMPORTANT: Establish an explicit LLM client lifecycle boundary at the
            # orchestrator/workflow boundary (not inside individual nodes).
            async with _managed_llm_lifecycle_for_workflow():
                async with create_checkpointer(str(self.checkpointer_path)) as checkpointer:
                    graph = create_full_workflow_graph(checkpointer=checkpointer)

                    # Step 4: Generate chapters
                    # The graph will automatically run initialization on first run
                    # via the conditional routing node
                    await self._run_chapter_generation_loop(graph, state)

            logger.info("=" * 60)
            logger.info("SAGA: LangGraph Generation Complete")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(
                "LangGraph orchestrator encountered error",
                error=str(e),
                exc_info=True,
            )
            raise
        finally:
            # Stop Rich display
            await self.display.stop()

    async def _ensure_neo4j_connection(self) -> None:
        """Connect to Neo4j and ensure the required schema exists.

        This is a strict boundary: connection failures or schema creation errors
        propagate to the caller.

        Side Effects:
            - Opens a Neo4j connection via the global manager.
            - Creates or updates the database schema (intended to be idempotent).
        """
        logger.info("Connecting to Neo4j...")
        await neo4j_manager.connect()
        await neo4j_manager.create_db_schema()
        logger.info("✓ Neo4j connected")

    async def _load_or_create_state(self) -> NarrativeState:
        """Create a fresh workflow state seed for this run.

        This method does not load state from the checkpointer. Instead, it derives
        the starting chapter from persisted chapters in Neo4j and constructs a new
        [`NarrativeState`](core/langgraph/state.py:1) via
        [`create_initial_state()`](core/langgraph/state.py:1).

        Initialization detection contract:
            - For chapter 1, initialization is artifact-driven: initialization is
              considered complete only if the expected on-disk initialization
              artifacts are present.
            - For continuation runs (when chapters already exist in Neo4j),
              initialization is treated as complete.

        Returns:
            A state dictionary seeded with project metadata plus:
            - `current_chapter` set to the next chapter number, and
            - `initialization_complete` set based on the rules above.

        Raises:
            Any exception raised by the underlying database query or filesystem
            validation helpers.
        """
        # Check if we have existing chapters to determine current chapter
        chapter_count = await chapter_queries.load_chapter_count_from_db()
        current_chapter = chapter_count + 1

        logger.info(f"Current chapter: {current_chapter} (existing: {chapter_count})")

        # Determine whether initialization should run.
        #
        # Contract:
        # - For chapter 1 (chapter_count == 0), initialization is artifact-driven. We must run init unless
        #   the expected initialization artifacts exist on disk. This avoids skipping init just because
        #   Neo4j contains leftover :Character nodes from a prior run.
        # - For continuation runs (chapter_count > 0), initialization is treated as complete.
        initialization_artifacts_ok = False
        missing_artifacts: list[str] = []
        if self.project_dir.exists():
            initialization_artifacts_ok, missing_artifacts = validate_initialization_artifacts(self.project_dir)

        if chapter_count == 0:
            initialization_complete = initialization_artifacts_ok
            logger.info(
                "Initialization detection (chapter 1): using filesystem artifacts",
                initialization_complete=initialization_complete,
                missing_artifacts=missing_artifacts,
            )
        else:
            initialization_complete = True
            logger.info(
                "Initialization detection (continuation): chapters already exist",
                initialization_complete=initialization_complete,
                existing_chapters=chapter_count,
            )

        # Create initial state
        state = create_initial_state(
            project_id="saga_novel",
            title=config.DEFAULT_PLOT_OUTLINE_TITLE,
            genre=config.CONFIGURED_GENRE,
            theme=config.CONFIGURED_THEME or "",
            setting=config.CONFIGURED_SETTING_DESCRIPTION or "",
            target_word_count=80000,  # Default, could be loaded from user configuration
            total_chapters=config.TOTAL_CHAPTERS or 12,  # Default, could be loaded from user configuration
            project_dir=str(self.project_dir),
            protagonist_name=config.DEFAULT_PROTAGONIST_NAME,
            # Model Mapping
            generation_model=config.NARRATIVE_MODEL,  # User override for generation
            extraction_model=config.MEDIUM_MODEL,
            revision_model=config.LARGE_MODEL,
            # Tiered models
            large_model=config.LARGE_MODEL,
            medium_model=config.MEDIUM_MODEL,
            small_model=config.SMALL_MODEL,
            narrative_model=config.NARRATIVE_MODEL,
        )

        # Update current chapter and initialization status
        state["current_chapter"] = current_chapter
        state["initialization_complete"] = initialization_complete

        # Advisory validation of initialization artifacts (non-breaking)
        if self.project_dir.exists():
            if initialization_artifacts_ok:
                logger.info(
                    "Initialization artifacts appear complete",
                    project_dir=str(self.project_dir),
                )
            else:
                logger.warning(
                    "Initialization artifacts incomplete for %s: %s",
                    str(self.project_dir),
                    "; ".join(missing_artifacts),
                )

        return state

    async def _run_chapter_generation_loop(self, graph: Any, state: NarrativeState) -> None:
        """Stream workflow events to generate up to `CHAPTERS_PER_RUN` chapters.

        For each chapter, this method updates `state["current_chapter"]`, runs the
        workflow using `astream()` (updates mode), merges per-node state updates
        into a single state dict, and determines completion based on the last
        user-visible node executed.

        Error policy:
            This loop is best-effort. If a chapter fails to complete, the failure
            is logged and the loop stops early without raising to the caller.

        Side Effects:
            - Executes workflow nodes, which may write files, update Neo4j, and
              record checkpoints through the provided checkpointer.
            - Mutates the provided `state` mapping in-place and also rebinds the
              local `state` variable to merged copies.

        Notes:
            - The thread identifier is fixed to `"saga_generation"` for checkpoint
              scoping, which means multiple concurrent orchestrator runs will
              contend for the same checkpoint stream.
        """
        chapters_per_run = config.CHAPTERS_PER_RUN
        total_chapters = state.get("total_chapters", 20)
        current_chapter = state.get("current_chapter", 1)

        logger.info(
            "Starting chapter generation loop",
            chapters_per_run=chapters_per_run,
            starting_chapter=current_chapter,
            total_chapters=total_chapters,
        )

        config_dict = {"configurable": {"thread_id": "saga_generation"}}

        chapters_generated = 0
        while chapters_generated < chapters_per_run and current_chapter <= total_chapters:
            logger.info("=" * 60 + f"\nGenerating Chapter {current_chapter} of {total_chapters}" + "\n" + "=" * 60)

            # Update state for this chapter
            state["current_chapter"] = current_chapter

            try:
                # Run workflow for this chapter using event streaming
                # The workflow will:
                # 1. Generate chapter outline (on-demand)
                # 2. Generate chapter text
                # 3. Extract entities
                # 4. Commit to graph
                # 5. Validate
                # 6. (Optional) Revise
                # 7. Summarize
                # 8. Finalize

                # Use astream() for event-based progress tracking
                # LangGraph's astream() yields events in format: {node_name: state_update}
                # We need to merge all updates to get the final state
                last_node = None
                async for event in graph.astream(state, config=config_dict):
                    # Handle workflow event for progress tracking
                    await self._handle_workflow_event(event, current_chapter)

                    # Merge state updates from event
                    # Event format: {node_name: state_update_dict}
                    if isinstance(event, dict) and len(event) > 0:
                        node_name = list(event.keys())[0]
                        if not node_name.startswith("__"):  # Skip internal nodes
                            state_update = event[node_name]
                            if isinstance(state_update, dict):
                                # Merge update into state
                                state = {**state, **state_update}
                                last_node = node_name

                # Check if chapter was successfully generated
                # After finalization, last executed node should be "finalize", "heal_graph", or "check_quality"
                if last_node in ["finalize", "heal_graph", "check_quality"]:
                    chapters_generated += 1
                    logger.info(
                        f"✓ Chapter {current_chapter} complete",
                        word_count=state.get("draft_word_count", 0),
                        node=last_node,
                    )

                    # Increment chapter for next iteration
                    current_chapter = state.get("current_chapter", current_chapter) + 1
                elif last_node:
                    # Generation ran but didn't complete successfully
                    error = state.get("last_error", "Unknown error")
                    logger.error(
                        f"Chapter {current_chapter} generation incomplete",
                        final_node=last_node,
                        error=error,
                    )
                    break
                else:
                    logger.error(f"Chapter {current_chapter} generation failed - no events received")
                    break

            except Exception as e:
                logger.error(
                    f"Error generating chapter {current_chapter}",
                    error=str(e),
                    exc_info=True,
                )
                break

        logger.info(
            "Chapter generation loop complete",
            chapters_generated=chapters_generated,
            final_chapter=current_chapter - 1,
        )

    async def _handle_workflow_event(self, event: dict[str, Any], chapter_number: int) -> None:
        """Update UI and structured logs for a workflow event.

        This method is called once per `astream()` event and is responsible for:
        - translating node identifiers into a human-readable step label, and
        - updating the Rich progress UI.

        Args:
            event: A single `astream()` event in updates-mode shape
                `{node_name: state_update}`.
            chapter_number: The chapter number currently being generated.

        Notes:
            - Internal LangGraph nodes (names starting with `"__"`) are ignored.
            - Empty or malformed events are treated as non-fatal and are logged
              as warnings.
        """
        # LangGraph's astream() yields events in "updates" mode format:
        # {node_name: state_update_dict}
        #
        # Example: {"generate": {"current_node": "generate", "draft_text": "...", ...}}
        #
        # The node name is the KEY, and the state update is the VALUE.

        if not isinstance(event, dict) or len(event) == 0:
            logger.warning("Received empty or invalid event", event_type=type(event).__name__)
            return

        # Extract node name and state update from event
        # Event format: {node_name: state_update}
        node_name = list(event.keys())[0]  # Get first (and usually only) key
        state_update = event[node_name]

        # Skip special internal nodes that don't represent user-visible progress
        if node_name.startswith("__"):
            logger.debug(f"Skipping internal node event: {node_name}")
            return

        initialization_step = state_update.get("initialization_step", "") if isinstance(state_update, dict) else ""

        # Determine human-readable step description
        step_description = self._get_step_description(node_name, initialization_step)

        # Update Rich display with current progress
        novel_title = None
        if isinstance(state_update, dict) and state_update.get("title"):
            novel_title = state_update.get("title", "Novel Generation")

        self.display.update(
            novel_title=novel_title,
            chapter_num=chapter_number,
            step=step_description,
            run_start_time=self.run_start_time,
        )

        # Log structured event information
        logger.info(
            f"[Chapter {chapter_number}] {step_description}",
            node=node_name,
            chapter=chapter_number,
            init_step=initialization_step if initialization_step else None,
        )

        # Handle specific node events with additional logging
        if not isinstance(state_update, dict):
            return

        if node_name == "validate" or node_name == "validate_consistency":
            contradictions = state_update.get("contradictions", [])
            if contradictions:
                severity_counts: dict[str, int] = {}
                for c in contradictions:
                    # Contradiction is a Pydantic model, not a dict
                    # Access severity as an attribute
                    if isinstance(c, dict):
                        severity = c.get("severity", "unknown")
                    else:
                        severity = getattr(c, "severity", "unknown")
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1

                logger.warning(
                    f"  ⚠️  Found {len(contradictions)} consistency issues",
                    total=len(contradictions),
                    severity_breakdown=severity_counts,
                )

        elif node_name == "revise":
            iteration = state_update.get("iteration_count", 0)
            max_iter = state_update.get("max_iterations", 3)
            logger.info(
                f"  🔄 Revision attempt {iteration}/{max_iter}",
                iteration=iteration,
                max_iterations=max_iter,
            )

        elif node_name == "finalize":
            word_count = state_update.get("draft_word_count", 0)
            logger.info(
                "  ✅ Chapter finalized",
                word_count=word_count,
                chapter=chapter_number,
            )

        elif node_name == "init_complete":
            character_count = len(state_update.get("character_sheets", {}))
            act_count = len(state_update.get("act_outlines", {}))
            logger.info(
                "  🎭 Initialization complete",
                characters=character_count,
                acts=act_count,
            )

    def _get_step_description(self, node_name: str, initialization_step: str = "") -> str:
        """Map workflow node identifiers to UI step descriptions.

        Args:
            node_name: The workflow node identifier emitted by LangGraph.
            initialization_step: Optional initialization phase indicator emitted
                by initialization nodes.

        Returns:
            A human-readable step description suitable for UI display. Unknown
            nodes fall back to `"Processing: {node_name}"`.
        """
        # Initialization phase descriptions
        # Only use initialization_step if it's a recognized initialization phase
        # Ignore chapter outline completion markers like "chapter_outline_2_complete"
        if initialization_step and not initialization_step.startswith("chapter_outline"):
            init_descriptions = {
                "character_sheets": "Generating Character Sheets",
                "global_outline": "Creating Global Story Outline",
                "act_outlines": "Detailing Act Structures",
                "committing": "Saving to Knowledge Graph",
                "files_persisted": "Writing Initialization Files",
                "complete": "Initialization Complete",
            }
            # Only return init description if it's a known init step
            if initialization_step in init_descriptions:
                return init_descriptions[initialization_step]

        # Generation phase descriptions (prioritized for all non-init nodes)
        node_descriptions = {
            "route": "Routing Workflow",
            "chapter_outline": "Generating Chapter Outline",
            "generate": "Generating Chapter Text",
            "generate_chapter": "Generating Chapter Text",
            "extract": "Extracting Entities & Relationships",
            "normalize_relationships": "Normalizing Relationship Types",
            "commit": "Committing to Knowledge Graph",
            "commit_to_graph": "Committing to Knowledge Graph",
            "validate": "Validating Consistency",
            "validate_consistency": "Validating Consistency",
            "revise": "Revising Chapter",
            "revise_chapter": "Revising Chapter",
            "summarize": "Creating Chapter Summary",
            "summarize_chapter": "Creating Chapter Summary",
            "finalize": "Finalizing Chapter",
            "finalize_chapter": "Finalizing Chapter",
            "init_character_sheets": "Creating Character Sheets",
            "init_global_outline": "Creating Story Outline",
            "init_act_outlines": "Detailing Acts",
            "init_commit_to_graph": "Saving Initialization Data",
            "init_persist_files": "Writing Files",
            "init_complete": "Completing Initialization",
            "init_error": "Initialization Error",
        }

        return node_descriptions.get(node_name, f"Processing: {node_name}")


__all__ = ["LangGraphOrchestrator"]
