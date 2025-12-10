# orchestration/langgraph_orchestrator.py
"""
LangGraph-based orchestrator for SAGA narrative generation.

This orchestrator uses the LangGraph workflow system instead of the legacy
NANA pipeline, running initialization and then generating chapters using
the Phase 2 complete workflow.

Migration Reference: docs/langgraph-architecture.md - Section 10.2
"""

import time
from pathlib import Path
from typing import Any

import structlog

import config
from core.db_manager import neo4j_manager
from core.langgraph.initialization.validation import validate_initialization_artifacts
from core.langgraph.state import NarrativeState, create_initial_state
from core.langgraph.workflow import create_checkpointer, create_full_workflow_graph
from data_access import chapter_queries
from ui.rich_display import RichDisplayManager

logger = structlog.get_logger(__name__)


class LangGraphOrchestrator:
    """
    Orchestrator for LangGraph-based narrative generation.

    Handles:
    - Initialization (if not already complete)
    - Chapter generation loop (respecting CHAPTERS_PER_RUN)
    - State persistence via checkpointer
    - Resume capability
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
        """
        Main entry point for LangGraph-based generation.

        Flow:
        1. Connect to Neo4j
        2. Load or create initial state
        3. Create workflow with checkpointing
        4. Generate CHAPTERS_PER_RUN chapters (initialization handled automatically)
        5. Save state for resumption
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
        """Ensure Neo4j connection is established."""
        logger.info("Connecting to Neo4j...")
        await neo4j_manager.connect()
        await neo4j_manager.create_db_schema()
        logger.info("✓ Neo4j connected")

    async def _load_or_create_state(self) -> NarrativeState:
        """
        Load existing state from checkpointer or create new initial state.

        If a checkpoint exists with thread_id "saga_generation", resume from there.
        Otherwise, create fresh initial state.
        """
        # Check if we have existing chapters to determine current chapter
        chapter_count = await chapter_queries.load_chapter_count_from_db()
        current_chapter = chapter_count + 1

        logger.info(f"Current chapter: {current_chapter} (existing: {chapter_count})")

        # Check if initialization was already completed by checking for
        # initialization artifacts (character profiles in Neo4j or YAML files)
        from data_access.character_queries import get_character_profiles

        try:
            character_profiles = await get_character_profiles()
            initialization_complete = len(character_profiles) > 0
            logger.info(
                f"Initialization detection: {len(character_profiles)} characters found",
                initialization_complete=initialization_complete,
            )
        except Exception as e:
            logger.warning(
                f"Could not check for existing initialization: {e}",
                exc_info=True,
            )
            # Fallback: check for initialization files
            init_file = self.project_dir / "outline" / "structure.yaml"
            initialization_complete = init_file.exists()
            logger.info(
                f"Initialization detection (fallback): structure.yaml exists={initialization_complete}"
            )

        # Create initial state
        state = create_initial_state(
            project_id="saga_novel",
            title=config.DEFAULT_PLOT_OUTLINE_TITLE,
            genre=config.CONFIGURED_GENRE,
            theme=config.CONFIGURED_THEME or "",
            setting=config.CONFIGURED_SETTING_DESCRIPTION or "",
            target_word_count=80000,  # Default, could be loaded from user configuration
            total_chapters=20,  # Default, could be loaded from user configuration
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
            ok, missing = validate_initialization_artifacts(self.project_dir)
            if ok:
                logger.info(
                    "Initialization artifacts appear complete",
                    project_dir=str(self.project_dir),
                )
            else:
                logger.warning(
                    "Initialization artifacts incomplete for %s: %s",
                    str(self.project_dir),
                    "; ".join(missing),
                )

        return state

    async def _run_chapter_generation_loop(
        self, graph: Any, state: NarrativeState
    ) -> None:
        """
        Generate CHAPTERS_PER_RUN chapters through the LangGraph workflow.

        For each chapter:
        1. Update current_chapter in state
        2. Run workflow (chapter_outline → generate → extract → commit → validate → revise? → summarize → finalize)
        3. Update chapter count
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
        while (
            chapters_generated < chapters_per_run and current_chapter <= total_chapters
        ):
            logger.info(
                "=" * 60
                + f"\nGenerating Chapter {current_chapter} of {total_chapters}"
                + "\n"
                + "=" * 60
            )

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
                    logger.error(
                        f"Chapter {current_chapter} generation failed - no events received"
                    )
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

    async def _handle_workflow_event(
        self, event: dict[str, Any], chapter_number: int
    ) -> None:
        """
        Handle LangGraph workflow events for real-time progress tracking.

        This method is called for each node execution in the workflow,
        providing visibility into the generation process as outlined in
        docs/langgraph-architecture.md section 4.2.

        Args:
            event: Event dict from astream() in format {node_name: state_update}
            chapter_number: Current chapter being generated

        Migration Reference: docs/langgraph-architecture.md - Section 10.2.4
        """
        # LangGraph's astream() yields events in "updates" mode format:
        # {node_name: state_update_dict}
        #
        # Example: {"generate": {"current_node": "generate", "draft_text": "...", ...}}
        #
        # The node name is the KEY, and the state update is the VALUE.

        if not isinstance(event, dict) or len(event) == 0:
            logger.warning(
                "Received empty or invalid event", event_type=type(event).__name__
            )
            return

        # Extract node name and state update from event
        # Event format: {node_name: state_update}
        node_name = list(event.keys())[0]  # Get first (and usually only) key
        state_update = event[node_name]

        # Skip special internal nodes that don't represent user-visible progress
        if node_name.startswith("__"):
            logger.debug(f"Skipping internal node event: {node_name}")
            return

        initialization_step = (
            state_update.get("initialization_step", "")
            if isinstance(state_update, dict)
            else ""
        )

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

    def _get_step_description(
        self, node_name: str, initialization_step: str = ""
    ) -> str:
        """
        Convert node name to human-readable step description.

        Args:
            node_name: Internal node name from workflow
            initialization_step: Optional initialization phase indicator

        Returns:
            Human-readable description of current step
        """
        # Initialization phase descriptions
        # Only use initialization_step if it's a recognized initialization phase
        # Ignore chapter outline completion markers like "chapter_outline_2_complete"
        if initialization_step and not initialization_step.startswith(
            "chapter_outline"
        ):
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
