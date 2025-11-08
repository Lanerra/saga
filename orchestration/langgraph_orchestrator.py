"""
LangGraph-based orchestrator for SAGA narrative generation.

This orchestrator uses the LangGraph workflow system instead of the legacy
NANA pipeline, running initialization and then generating chapters using
the Phase 2 complete workflow.
"""

from pathlib import Path
from typing import Any

import structlog

import config
from core.db_manager import neo4j_manager
from core.langgraph.initialization.validation import validate_initialization_artifacts
from core.langgraph.state import NarrativeState, create_initial_state
from core.langgraph.workflow import create_checkpointer, create_full_workflow_graph
from data_access import chapter_queries

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

    def __init__(self):
        logger.info("Initializing LangGraph Orchestrator...")
        # Use settings.BASE_OUTPUT_DIR which is the Pydantic field
        self.project_dir = Path(config.settings.BASE_OUTPUT_DIR)
        self.checkpointer_path = self.project_dir / ".saga" / "checkpoints.db"
        logger.info("LangGraph Orchestrator initialized.")

    async def run_novel_generation_loop(self):
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

    async def _ensure_neo4j_connection(self):
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
            target_word_count=80000,  # Default, can be loaded from plot_outline
            total_chapters=20,  # Default, can be loaded from plot_outline
            project_dir=str(self.project_dir),
            protagonist_name=config.DEFAULT_PROTAGONIST_NAME,
            generation_model=config.NARRATIVE_MODEL,
            extraction_model=config.NARRATIVE_MODEL,
            revision_model=config.NARRATIVE_MODEL,
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
                # Run workflow for this chapter
                # The workflow will:
                # 1. Generate chapter outline (on-demand)
                # 2. Generate chapter text
                # 3. Extract entities
                # 4. Commit to graph
                # 5. Validate
                # 6. (Optional) Revise
                # 7. Summarize
                # 8. Finalize
                result = await graph.ainvoke(state, config=config_dict)

                # Check if chapter was successfully generated
                if result.get("draft_text"):
                    chapters_generated += 1
                    current_chapter += 1

                    logger.info(
                        f"✓ Chapter {result['current_chapter']} complete",
                        word_count=result.get("draft_word_count", 0),
                    )

                    # Update state for next iteration
                    state = result
                else:
                    logger.error(f"Chapter {current_chapter} generation failed")
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


__all__ = ["LangGraphOrchestrator"]
