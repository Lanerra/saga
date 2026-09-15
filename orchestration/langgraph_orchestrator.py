# orchestration/langgraph_orchestrator.py
"""Orchestrate LangGraph-based SAGA narrative generation.

This module defines the orchestration boundary around the LangGraph workflow:
Neo4j connectivity, workflow checkpoint lifecycle, and an explicit LLM HTTP-client
lifecycle that is shared across workflow nodes.

Error/cleanup policy at this boundary is intentionally mixed:

- Strict: Neo4j connection/setup failures and workflow construction/streaming
  errors propagate to the caller.
- Run-owned services close on success, failure and cancellation.
"""

import time
from pathlib import Path
from typing import Any, cast

import structlog
import yaml

import config
from core.exceptions import CheckpointResumeConflictError, WorkflowExecutionError
from core.graph_ownership import load_graph_project_id
from core.langgraph.chapter_lifecycle import ChapterLifecycle, reconcile_checkpoint
from core.langgraph.initialization.validation import validate_initialization_artifacts
from core.langgraph.state import NarrativeState, create_initial_state, validate_state_contract
from core.langgraph.workflow import create_checkpointer, create_full_workflow_graph
from core.project_config import NarrativeProjectConfig
from core.service_context import RunServices, get_services, service_lifetime
from data_access import chapter_queries
from ui.rich_display import RichDisplayManager
from utils.file_io import ContainedFiles

logger = structlog.get_logger(__name__)


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
        - State-signaled fatal completion raises with retained diagnostics.
    """

    def __init__(self, *, project_dir: Path | None = None, services: RunServices | None = None) -> None:
        logger.info("Initializing LangGraph Orchestrator...")
        self.services = services
        # Use settings.BASE_OUTPUT_DIR which is the Pydantic field
        self.project_dir = Path(project_dir) if project_dir is not None else Path(config.settings.BASE_OUTPUT_DIR)
        self.checkpointer_path = self.project_dir / "checkpoints" / "saga.db"

        # Initialize Rich display for progress tracking
        self.display = RichDisplayManager()
        self.run_start_time: float = 0.0

        logger.info("LangGraph Orchestrator initialized.")

    async def run_novel_generation_loop(self, narrative_config: NarrativeProjectConfig | None = None) -> None:
        async with service_lifetime(self.services):
            await self._run_novel_generation_loop(narrative_config)

    async def _run_novel_generation_loop(self, narrative_config: NarrativeProjectConfig | None = None) -> None:
        """Run the end-to-end LangGraph novel generation loop.

        This method owns the orchestration boundary and associated cleanup:

        - Starts the Rich progress display (if enabled).
        - Establishes a Neo4j connection and ensures the database schema exists.
        - Loads the latest checkpointed state when resuming (checkpoint-first).
        - Runs the workflow under:
          - a managed LLM client lifecycle context, and
          - a checkpointer context bound to a persistent SQLite file.

        Resume policy:
            Checkpoints are the single source of truth. Neo4j and filesystem artifacts are
            treated as persisted artifacts and are only used for conflict detection. When
            artifacts conflict with checkpoint state, orchestration fails fast with a clear,
            stable error.

        Error policy:
            - Neo4j connection/setup failures and workflow construction/streaming
              failures propagate to the caller.
            - State-signaled fatal completion raises `WorkflowExecutionError`.
              Cancellation propagates without a success event.

        Cleanup:
            The Rich display is stopped in a `finally` block. If display shutdown
            raises, that exception propagates (and can mask a prior error).

        Args:
            narrative_config: Optional narrative configuration to seed new runs. When
                provided, it overrides the default config.settings values for narrative
                metadata on new state creation.

        Side Effects:
            - Creates/updates a checkpoint database at `project_dir/checkpoints/saga.db`.
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

            requested_project_id = self._get_requested_project_id()
            thread_id = self._checkpoint_thread_id(requested_project_id)

            async with create_checkpointer(str(self.checkpointer_path)) as checkpointer:
                graph = create_full_workflow_graph(checkpointer=checkpointer)
                state = await self._load_state_for_run(
                    graph=graph,
                    requested_project_id=requested_project_id,
                    thread_id=thread_id,
                    narrative_config=narrative_config,
                )
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
            try:
                await self.display.stop()
            except Exception:
                # Log but don't let this mask original error from workflow
                logger.warning("Display shutdown failed", exc_info=True)

    async def _ensure_neo4j_connection(self) -> None:
        """Connect to Neo4j and ensure the required schema exists.

        This is a strict boundary: connection failures or schema creation errors
        propagate to the caller.

        Side Effects:
            - Opens a Neo4j connection via the run's manager.
            - Creates or updates the database schema (intended to be idempotent).
        """
        logger.info("Connecting to Neo4j...")
        database = get_services().database
        database.bind_project(load_graph_project_id(self.project_dir))
        await database.connect()
        await database.create_db_schema()
        logger.info("✓ Neo4j connected")

    async def _load_or_create_state(
        self,
        *,
        project_id: str,
        narrative_config: NarrativeProjectConfig | None,
    ) -> NarrativeState:
        """Create a fresh workflow state seed for this run (non-resume path).

        This method is used only when no checkpoint is present for the project's checkpoint
        thread. It derives the starting chapter from contiguous finalized chapters and
        constructs a new [`NarrativeState`](core/langgraph/state.py:1) via
        [`create_initial_state()`](core/langgraph/state.py:343).

        Initialization detection contract:
            - For chapter 1, initialization requires a retained plan and its exact
              graph receipt. Human-readable artifacts are not acceptance evidence.
            - For continuation runs (when a finalized prefix exists in Neo4j),
              initialization is treated as complete.

        Args:
            project_id: Project identifier to seed into state.
            narrative_config: Optional narrative configuration used for new state creation.

        Returns:
            A state dictionary seeded with project metadata plus:
            - `current_chapter` set to the next chapter number, and
            - `initialization_complete` set based on the rules above.

        Raises:
            Any exception raised by the underlying database query or filesystem
            validation helpers.
        """
        progress = await chapter_queries.load_chapter_progress_from_db()
        current_chapter = progress.last_finalized_chapter + 1

        logger.info("Chapter progress loaded", current_chapter=current_chapter, last_finalized_chapter=progress.last_finalized_chapter)

        initialization_artifacts_ok = False
        missing_artifacts: list[str] = []
        initialization_import_state: NarrativeState = {}
        if self.project_dir.exists():
            initialization_artifacts_ok, missing_artifacts = validate_initialization_artifacts(self.project_dir)

        if progress.last_finalized_chapter == 0:
            from core.langgraph.initialization.staged_import import InitializationImport

            importer = InitializationImport(str(self.project_dir))
            initialization_complete = False
            if importer.files.exists(f"{importer.root}/selected"):
                plan = importer.load()
                if plan.snapshot.project_id != load_graph_project_id(self.project_dir):
                    raise ValueError("Selected initialization receipt belongs to another graph project")
                initialization_import_state = importer.state(plan)
                if initialization_import_state["project_id"] != project_id:
                    raise ValueError("Selected initialization receipt belongs to another workflow project")
                if not await importer.receipt(plan):
                    raise ValueError("Missing accepted initialization receipt; accept the retained import before generation")
                initialization_complete = True
            elif initialization_artifacts_ok:
                raise ValueError("Missing initialization receipt; existing user files cannot authorize generation or reinitialization")
        else:
            initialization_complete = True
            logger.info(
                "Initialization detection (continuation): finalized prefix exists",
                initialization_complete=initialization_complete,
                last_finalized_chapter=progress.last_finalized_chapter,
            )

        if narrative_config is not None:
            logger.info(
                "Creating new state from narrative config",
                title=narrative_config.title,
                project_dir=str(self.project_dir),
            )
            state = create_initial_state(
                project_id=project_id,
                title=narrative_config.title,
                genre=narrative_config.genre,
                theme=narrative_config.theme,
                setting=narrative_config.setting,
                target_word_count=narrative_config.target_word_count,
                total_chapters=narrative_config.total_chapters,
                project_dir=str(self.project_dir),
                protagonist_name=narrative_config.protagonist_name,
                narrative_style=narrative_config.narrative_style,
                original_prompt=narrative_config.original_prompt,
                extraction_model=config.MEDIUM_MODEL,
                revision_model=config.LARGE_MODEL,
                large_model=config.LARGE_MODEL,
                medium_model=config.MEDIUM_MODEL,
                small_model=config.SMALL_MODEL,
                narrative_model=config.NARRATIVE_MODEL,
                max_iterations=config.MAX_REVISION_CYCLES_PER_CHAPTER,
            )
        else:
            logger.info(
                "Creating new state from config.settings",
                title=config.DEFAULT_PLOT_OUTLINE_TITLE,
                project_dir=str(self.project_dir),
            )
            state = create_initial_state(
                project_id=project_id,
                title=config.DEFAULT_PLOT_OUTLINE_TITLE,
                genre=config.CONFIGURED_GENRE,
                theme=config.CONFIGURED_THEME,
                setting=config.CONFIGURED_SETTING_DESCRIPTION,
                target_word_count=config.TARGET_WORD_COUNT,
                total_chapters=config.TOTAL_CHAPTERS,
                project_dir=str(self.project_dir),
                protagonist_name=config.DEFAULT_PROTAGONIST_NAME,
                narrative_style=config.DEFAULT_NARRATIVE_STYLE,
                extraction_model=config.MEDIUM_MODEL,
                revision_model=config.LARGE_MODEL,
                large_model=config.LARGE_MODEL,
                medium_model=config.MEDIUM_MODEL,
                small_model=config.SMALL_MODEL,
                narrative_model=config.NARRATIVE_MODEL,
                max_iterations=config.MAX_REVISION_CYCLES_PER_CHAPTER,
            )

        state.update(initialization_import_state)
        state["current_chapter"] = current_chapter
        state["initialization_complete"] = initialization_complete
        state["run_start_chapter"] = current_chapter

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

        validate_state_contract(state)
        return state

    async def _run_chapter_generation_loop(self, graph: Any, state: NarrativeState) -> None:
        """Stream workflow events for end-to-end chapter generation.

        This method runs the full LangGraph workflow, which handles initialization
        and multiple chapters internally. It tracks progress via `astream()` and
        updates the UI based on state changes across multiple chapters.

        Error policy:
            Workflow failures during streaming are logged and re-raised to the
            caller.

        Side Effects:
            - Executes workflow nodes, which may write files, update Neo4j, and
              record checkpoints.
            - Updates the provided `state` mapping as it receives updates from the
              workflow stream.
        """
        project_id = state.get("project_id")
        if not isinstance(project_id, str) or not project_id:
            raise ValueError("Workflow state must include a non-empty str 'project_id'")

        thread_id = self._checkpoint_thread_id(project_id)

        logger.info(
            "Starting multi-chapter generation stream",
            project_id=project_id,
            thread_id=thread_id,
            starting_chapter=state.get("current_chapter", 1),
            total_chapters=state.get("total_chapters"),
        )

        config_dict = {"configurable": {"thread_id": thread_id}, "recursion_limit": 500}
        last_node = None
        event_index = 0
        interrupted = False

        try:
            # Use astream() for event-based progress tracking across all chapters.
            # The workflow now handles the loop internally.
            workflow_input: NarrativeState | None = state
            if getattr(self, "_resume_checkpoint", False) and "lifecycle_version" in state:
                state = await reconcile_checkpoint(graph, state, config_dict)
                workflow_input = None
            async for event in graph.astream(workflow_input, config=config_dict):
                if not isinstance(event, dict) or not event:
                    continue

                if "__interrupt__" in event:
                    interrupted = True
                    continue

                node_name = list(event.keys())[0]
                if node_name.startswith("__"):
                    continue

                event_index += 1

                state_update = event[node_name]
                if not isinstance(state_update, dict):
                    continue

                # Merge updates into local state tracking
                state = {**state, **state_update}
                last_node = node_name

                # Track chapter transitions and completion
                current_chapter = state.get("current_chapter", 1)

                # Handle workflow event for progress tracking
                await self._handle_workflow_event(event, current_chapter, event_index)

                # Log chapter completion
                if node_name in ["finalize", "heal_graph", "check_quality"]:
                    # Check if we just completed a chapter
                    # In an internal loop, we might get multiple nodes for the same chapter.
                    # We rely on finalize/heal_graph/check_quality being the 'completion' markers.
                    if state.get("current_node") == node_name:
                        logger.info(
                            f"✓ Chapter {current_chapter} node reached: {node_name}",
                            word_count=state.get("draft_word_count", 0),
                        )

            native_end = False
            if "lifecycle_version" in state:
                snapshot = await graph.aget_state(config_dict)
                if snapshot.created_at is not None and isinstance(snapshot.values, dict) and snapshot.values.get("project_id") == project_id:
                    state = {**state, **snapshot.values}
                    native_end = not snapshot.next and not snapshot.tasks
                    interrupted = interrupted or bool(snapshot.interrupts)

            # Final summary of the run
            rollback_failure = state.get("revision_rollback_failure")
            if state.get("has_fatal_error") or rollback_failure is not None:
                raise WorkflowExecutionError(
                    "Workflow terminated with fatal error",
                    details={
                        "project_id": project_id,
                        "current_chapter": state.get("current_chapter"),
                        "last_error": state.get("last_error"),
                        "error_node": state.get("error_node"),
                        "revision_rollback_failure": rollback_failure,
                        "final_node": last_node,
                    },
                )

            final_node = last_node
            if last_node is None and workflow_input is None and native_end:
                final_node = state.get("current_node")
            completed = final_node in {"finalize", "heal_graph", "check_quality", "init_complete"}
            if native_end and final_node == "advance_chapter" and state.get("lifecycle_phase") == "advanced":
                completed = True
            if interrupted or ("lifecycle_version" in state and not native_end) or not completed:
                outcome = "interrupted" if interrupted else "incomplete"
                raise WorkflowExecutionError(
                    f"Workflow {outcome}; no completion claimed",
                    details={
                        "outcome": outcome,
                        "project_id": project_id,
                        "current_chapter": state.get("current_chapter"),
                        "last_error": state.get("last_error"),
                        "final_node": final_node,
                    },
                )

            logger.info(
                "Workflow stream finished successfully",
                final_chapter=state.get("current_chapter"),
                final_node=final_node,
            )

        except Exception as e:
            logger.error(
                "Error during chapter generation stream",
                error=str(e),
                exc_info=True,
            )
            raise  # Re-raise to let caller handle original error

        logger.info("Multi-chapter generation stream complete.")

    async def _handle_workflow_event(
        self,
        event: object,
        chapter_number: int,
        event_index: int = 0,
    ) -> None:
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
            event_index=event_index,
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

        elif node_name == "heal_graph":
            warnings = state_update.get("last_healing_warnings", [])
            apoc_available = state_update.get("last_apoc_available")
            if warnings:
                warning_message = f"⚠️  Healing warnings: {warnings}"
                logger.warning(warning_message)
                if self.display.live:
                    console = self.display.get_shared_console()
                    console.print(warning_message)
            if apoc_available is False:
                logger.warning("APOC unavailable during graph healing")

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

    def _checkpoint_thread_id(self, project_id: str) -> str:
        safe_project_id = project_id.replace("/", "_").replace("\\", "_").strip()
        return f"saga_{safe_project_id}"

    def _get_requested_project_id(self) -> str:
        files = ContainedFiles(self.project_dir)
        if files.exists("saga.yaml"):
            try:
                data = yaml.safe_load(files.read_bytes("saga.yaml"))
            except yaml.YAMLError as error:
                raise ValueError("Invalid saga.yaml project metadata") from error
            if not isinstance(data, dict):
                raise ValueError("saga.yaml must contain project metadata")
            value = data.get("project_id")
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError("saga.yaml project_id must be a nonempty string without surrounding whitespace")
            return value

        project_dir_name = self.project_dir.name.strip()
        if not project_dir_name:
            raise ValueError("Project directory name must be non-empty to derive project_id")

        return project_dir_name

    async def _load_state_for_run(
        self,
        *,
        graph: Any,
        requested_project_id: str,
        thread_id: str,
        narrative_config: NarrativeProjectConfig | None,
    ) -> NarrativeState:
        """Load checkpointed state when available; otherwise create a fresh seed state.

        Use the latest native snapshot, including successful pending task writes.
        Raw saver channel values can lag completed work. Do not specify a checkpoint
        ID here: LangGraph treats that as historical replay without pending writes.
        Narrative configuration only seeds a thread with no checkpoint.
        """
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        self._resume_checkpoint = snapshot.created_at is not None
        if not self._resume_checkpoint:
            state = await self._load_or_create_state(
                project_id=requested_project_id,
                narrative_config=narrative_config,
            )
            if state.get("current_chapter", 1) != 1:
                raise CheckpointResumeConflictError("Missing checkpoint for existing graph progress; restore the project checkpoint or use an explicit migration")
            return {**state, "lifecycle_version": 1, "graph_project_id": load_graph_project_id(self.project_dir), "attempt_id": None}

        if not isinstance(snapshot.values, dict):
            raise CheckpointResumeConflictError(
                "Resume conflict: native checkpoint is missing required state mapping",
            )

        checkpoint_state = cast(NarrativeState, snapshot.values)
        await self._validate_resume_state_or_raise_async(
            checkpoint_state=checkpoint_state,
            requested_project_id=requested_project_id,
        )

        if checkpoint_state.get("lifecycle_version") != 1:
            raise CheckpointResumeConflictError("Legacy checkpoint has no attempt lifecycle; explicit migration is required")
        if checkpoint_state.get("initialization_id") and not checkpoint_state.get("initialization_complete"):
            from core.langgraph.initialization.staged_import import InitializationImport

            checkpoint_state = await InitializationImport(str(self.project_dir)).reconcile_checkpoint(
                graph, checkpoint_state, {"configurable": {"thread_id": thread_id}},
            )
        return checkpoint_state

    def _validate_resume_state_or_raise(
        self,
        *,
        checkpoint_state: NarrativeState,
        requested_project_id: str,
    ) -> None:
        checkpoint_project_id = checkpoint_state.get("project_id")
        if checkpoint_project_id != requested_project_id:
            raise CheckpointResumeConflictError(f"Resume conflict: checkpoint project_id '{checkpoint_project_id}' does not match requested project_id '{requested_project_id}'")

        checkpoint_directory = checkpoint_state.get("project_dir")
        if not isinstance(checkpoint_directory, str) or not checkpoint_directory.strip() or Path(checkpoint_directory).absolute() != self.project_dir.absolute():
            raise CheckpointResumeConflictError("Resume conflict: checkpoint project_dir does not match the requested project directory")

        current_chapter = checkpoint_state.get("current_chapter")
        if not isinstance(current_chapter, int) or isinstance(current_chapter, bool) or current_chapter <= 0:
            raise CheckpointResumeConflictError(
                "Resume conflict: checkpoint current_chapter must be a positive integer",
                details={"current_chapter": current_chapter},
            )

    async def _validate_resume_state_or_raise_async(
        self,
        *,
        checkpoint_state: NarrativeState,
        requested_project_id: str,
    ) -> None:
        self._validate_resume_state_or_raise(
            checkpoint_state=checkpoint_state,
            requested_project_id=requested_project_id,
        )

        if "lifecycle_version" in checkpoint_state:
            try:
                validate_state_contract(checkpoint_state)
            except ValueError as error:
                raise CheckpointResumeConflictError(f"Resume conflict: state contract: {error}") from error

        current_chapter = cast(int, checkpoint_state.get("current_chapter"))
        progress = await chapter_queries.load_chapter_progress_from_db()
        current_accepted = False
        if "lifecycle_version" in checkpoint_state:
            lifecycle = ChapterLifecycle(checkpoint_state)
            if checkpoint_state.get("attempt_id") is not None or lifecycle.files.exists(lifecycle.selection_path()):
                lifecycle.stage()
                receipt = await lifecycle.graph_receipt()
                current_accepted = receipt is not None and receipt["phase"] == "accepted"
        if any(number > current_chapter or (number == current_chapter and not current_accepted) for number in progress.finalized_chapters):
            raise CheckpointResumeConflictError(f"Resume conflict: Neo4j has finalized chapters at or ahead of checkpoint current_chapter={current_chapter}")
        if progress.last_finalized_chapter + (0 if current_accepted else 1) != current_chapter:
            raise CheckpointResumeConflictError(
                f"Resume conflict: contiguous finalized prefix ends at {progress.last_finalized_chapter}, but checkpoint current_chapter={current_chapter}"
            )

        # Conflict: checkpoint references missing artifact files.
        for key, value in checkpoint_state.items():
            if not key.endswith("_ref"):
                continue
            if value is None:
                continue
            if not isinstance(value, dict):
                raise CheckpointResumeConflictError(f"Resume conflict: checkpoint field '{key}' must be a ContentRef dict")
            ref_path = value.get("path")
            if not isinstance(ref_path, str) or not ref_path:
                raise CheckpointResumeConflictError(f"Resume conflict: checkpoint field '{key}' must include ContentRef.path as non-empty str")
            full_path = self.project_dir / ref_path
            if not full_path.exists():
                raise CheckpointResumeConflictError(f"Resume conflict: checkpoint references missing artifact for field '{key}': path='{ref_path}'")

        return None


__all__ = ["LangGraphOrchestrator"]
