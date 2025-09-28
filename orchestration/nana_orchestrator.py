# orchestration/nana_orchestrator.py
import asyncio
import logging as stdlib_logging
import logging.handlers
import os
import time  # For Rich display updates
from typing import Any

import structlog

import config
from config import settings, simple_formatter
import utils
from agents.knowledge_agent import KnowledgeAgent
from agents.narrative_agent import NarrativeAgent
from agents.revision_agent import RevisionAgent
from core.db_manager import neo4j_manager
from core.llm_interface_refactored import llm_service
from core.runtime_config_validator import (
    validate_runtime_configuration,
)
from data_access import (
    chapter_queries,
    plot_queries,
)

# Import native versions for performance optimization
from data_access.character_queries import (
    get_character_profiles,
)
from data_access.world_queries import (
    get_world_building,
)
from initialization.bootstrap_pipeline import run_bootstrap_pipeline
from initialization.bootstrap_validator import validate_bootstrap_results
from initialization.bootstrappers.plot_bootstrapper import bootstrap_plot_outline
from initialization.data_loader import (
    convert_model_to_objects,
    load_user_supplied_model,
)
from models import (
    CharacterProfile,
    EvaluationResult,
    ProblemDetail,
    SceneDetail,
    WorldItem,
)
from models.user_input_models import UserStoryInputModel, user_story_to_objects
from orchestration.chapter_flow import run_chapter_pipeline
from processing.revision_logic import revise_chapter_draft_logic
from processing.text_deduplicator import TextDeduplicator
from processing.zero_copy_context_generator import ZeroCopyContextGenerator
from ui.rich_display import RichDisplayManager
from utils.common import split_text_into_chapters

try:
    from rich.logging import RichHandler

    RICH_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when Rich is missing
    RICH_AVAILABLE = False

    class RichHandler(stdlib_logging.Handler):
        def emit(self, record: stdlib_logging.LogRecord) -> None:  # pragma: no cover
            stdlib_logging.getLogger(__name__).handle(record)


logger = structlog.get_logger(__name__)


class NANA_Orchestrator:
    def __init__(self):
        logger.info("Initializing SAGA Orchestrator...")
        self.narrative_agent = NarrativeAgent(config)
        self.revision_agent = RevisionAgent(config)
        self.knowledge_agent = KnowledgeAgent()

        self.plot_outline: dict[str, Any] = {}
        self.chapter_count: int = 0
        self.novel_props_cache: dict[str, Any] = {}
        self._user_story_prime_attempted: bool = False

        self.display = RichDisplayManager()
        self.run_start_time: float = 0.0
        utils.load_spacy_model_if_needed()
        logger.info("SAGA Orchestrator initialized.")

    def _update_rich_display(
        self, chapter_num: int | None = None, step: str | None = None
    ) -> None:
        self.display.update(
            plot_outline=self.plot_outline,
            chapter_num=chapter_num,
            step=step,
            run_start_time=self.run_start_time,
        )

    async def _generate_plot_points_from_kg(self, count: int) -> None:
        """Generate and persist additional plot points using the planner agent."""
        if count <= 0:
            return

        summaries: list[str] = []
        start = max(1, self.chapter_count - config.CONTEXT_CHAPTER_COUNT + 1)
        for i in range(start, self.chapter_count + 1):
            chap = await chapter_queries.get_chapter_data_from_db(i)
            if chap and (chap.get("summary") or chap.get("text")):
                summaries.append((chap.get("summary") or chap.get("text", "")).strip())

        combined_summary = "\n".join(summaries)
        if not combined_summary.strip():
            logger.warning("No summaries available for continuation planning.")
            return

        new_points, _ = await self.narrative_agent.plan_continuation(
            combined_summary, count
        )
        if not new_points:
            logger.error("Failed to generate continuation plot points.")
            return

        for desc in new_points:
            if await plot_queries.plot_point_exists(desc):
                logger.info(f"Plot point already exists, skipping: {desc}")
                continue
            prev_id = await plot_queries.get_last_plot_point_id()
            await self.knowledge_agent.add_plot_point(desc, prev_id or "")
            self.plot_outline.setdefault("plot_points", []).append(desc)
        self._update_novel_props_cache()

    def load_state_from_user_model(self, model: UserStoryInputModel) -> None:
        """Populate orchestrator state from a user-provided model."""
        plot_outline, _, _ = user_story_to_objects(model)
        self.plot_outline = plot_outline

    def _update_novel_props_cache(self):
        self.novel_props_cache = {
            "title": self.plot_outline.get("title", config.DEFAULT_PLOT_OUTLINE_TITLE),
            "genre": self.plot_outline.get("genre", config.CONFIGURED_GENRE),
            "theme": self.plot_outline.get("theme", config.CONFIGURED_THEME),
            "protagonist_name": self.plot_outline.get(
                "protagonist_name", config.DEFAULT_PROTAGONIST_NAME
            ),
            "character_arc": self.plot_outline.get("character_arc", "N/A"),
            "logline": self.plot_outline.get("logline", "N/A"),
            "setting": self.plot_outline.get(
                "setting", config.CONFIGURED_SETTING_DESCRIPTION
            ),
            "narrative_style": self.plot_outline.get("narrative_style", "N/A"),
            "tone": self.plot_outline.get("tone", "N/A"),
            "pacing": self.plot_outline.get("pacing", "N/A"),
            "plot_points": self.plot_outline.get("plot_points", []),
            "plot_outline_full": self.plot_outline,
        }
        self._update_rich_display()

    async def refresh_plot_outline(self) -> None:
        """Reload plot outline from the database."""
        result = await plot_queries.get_plot_outline_from_db()
        if isinstance(result, dict):
            self.plot_outline = result
            self._update_novel_props_cache()
        else:
            logger.error(f"Failed to refresh plot outline from DB: {result}")

    async def async_init_orchestrator(self):
        logger.info("SAGA Orchestrator async_init_orchestrator started...")
        self._update_rich_display(step="Initializing Orchestrator")
        self.chapter_count = await chapter_queries.load_chapter_count_from_db()
        logger.info(f"Loaded chapter count from Neo4j: {self.chapter_count}")
        await plot_queries.ensure_novel_info()
        result = await plot_queries.get_plot_outline_from_db()
        if isinstance(result, Exception):
            logger.error(
                "Error loading plot outline during orchestrator init: %s",
                result,
                exc_info=result,
            )
            self.plot_outline = {}
        else:
            self.plot_outline = result if isinstance(result, dict) else {}

        if not self.plot_outline.get("plot_points"):
            logger.warning(
                "Orchestrator init: Plot outline loaded from DB has no plot points. Initial setup might be needed or DB is empty/corrupt."
            )
        else:
            logger.info(
                f"Orchestrator init: Loaded {len(self.plot_outline.get('plot_points', []))} plot points from DB."
            )

        self._update_novel_props_cache()
        logger.info("SAGA Orchestrator async_init_orchestrator complete.")
        self._update_rich_display(step="Orchestrator Initialized")

        # Validate runtime configuration against bootstrap content
        if getattr(config, "ENABLE_RUNTIME_CONFIG_VALIDATION", True):
            await self._validate_runtime_configuration()

    async def _prime_from_user_story_elements(self) -> None:
        """Load user story file early and persist content if the KG is empty."""
        if self._user_story_prime_attempted:
            return

        self._user_story_prime_attempted = True

        model = load_user_supplied_model()
        if model is None:
            return

        plot_outline, character_profiles, world_building = convert_model_to_objects(
            model
        )
        plot_outline.setdefault("source", "user_story_elements")
        world_building.setdefault("source", "user_story_elements")

        validation = await validate_bootstrap_results(
            plot_outline,
            character_profiles,
            world_building,
        )
        if not validation.is_valid:
            logger.warning(
                "User story elements validation failed; falling back to bootstrap pipeline. Warnings: %s",
                validation.warnings,
            )
            return

        try:
            await plot_queries.ensure_novel_info()
            existing_outline = await plot_queries.get_plot_outline_from_db()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "Unable to inspect existing plot outline before priming from user story.",
                error=str(exc),
                exc_info=True,
            )
            existing_outline = {}

        existing_concrete_points: list[str] = []
        if isinstance(existing_outline, dict):
            raw_points = existing_outline.get("plot_points", [])
            if isinstance(raw_points, list):
                existing_concrete_points = [
                    point
                    for point in raw_points
                    if isinstance(point, str)
                    and point.strip()
                    and not utils._is_fill_in(point)
                ]

        if existing_concrete_points:
            logger.info(
                "Existing plot outline detected in Neo4j; skipping user story priming (plot points: %d).",
                len(existing_concrete_points),
            )
            self.plot_outline = existing_outline
            self._update_novel_props_cache()
            return

        concrete_points = [
            point
            for point in plot_outline.get("plot_points", [])
            if isinstance(point, str) and point.strip() and not utils._is_fill_in(point)
        ]
        if len(concrete_points) < config.TARGET_PLOT_POINTS_INITIAL_GENERATION:
            deficit = config.TARGET_PLOT_POINTS_INITIAL_GENERATION - len(
                concrete_points
            )
            logger.info(
                "Generating %d supplemental plot point(s) from user story context.",
                deficit,
            )
            plot_outline, _ = await bootstrap_plot_outline(
                plot_outline,
                character_profiles,
                world_building,
            )

        logger.info(
            "Priming knowledge graph from user story elements; bootstrap pipeline will be skipped. Title: '%s'; Protagonist: '%s'; Plot points: %d",
            plot_outline.get("title"),
            plot_outline.get("protagonist_name"),
            len(plot_outline.get("plot_points", [])),
        )

        await plot_queries.save_plot_outline_to_db(plot_outline)
        await self.knowledge_agent.persist_world(
            world_building,
            config.KG_PREPOPULATION_CHAPTER_NUM,
            full_sync=True,
        )
        await self.knowledge_agent.persist_profiles(
            character_profiles,
            config.KG_PREPOPULATION_CHAPTER_NUM,
            full_sync=True,
        )

        self.plot_outline = plot_outline
        self._update_novel_props_cache()

    async def perform_initial_setup(self):
        self._update_rich_display(step="Performing Initial Setup")
        logger.info("SAGA performing initial setup...")
        # SAGA standardized on the phased bootstrap pipeline (world -> characters -> plot)
        # Genesis path has been retired per bootstrap-exam recommendation #5.
        phase = "all"
        level = getattr(config, "BOOTSTRAP_HIGHER_SETTING", "enhanced")
        (
            plot_outline,
            character_profiles,
            world_building,
            _,
        ) = await run_bootstrap_pipeline(
            phase=phase,
            level=level,
            dry_run=False,
            kg_heal=getattr(config, "BOOTSTRAP_RUN_KG_HEAL", True),
        )
        self.plot_outline = plot_outline

        plot_source = self.plot_outline.get("source", "unknown")
        logger.info(
            f"   Plot Outline and Characters initialized/loaded (source: {plot_source}). "
            f"Title: '{self.plot_outline.get('title', 'N/A')}'. "
            f"Plot Points: {len(self.plot_outline.get('plot_points', []))}"
        )
        world_source = world_building.get("source", "unknown")
        logger.info(f"   World Building initialized/loaded (source: {world_source}).")
        self._update_rich_display(step="Bootstrap State Bootstrapped")

        self._update_novel_props_cache()
        logger.info("   Initial plot, character, and world data saved to Neo4j.")
        self._update_rich_display(step="Initial State Saved")

        return True

    async def _prepopulate_kg_if_needed(self):
        self._update_rich_display(step="Pre-populating KG (if needed)")
        logger.info("SAGA: Checking if KG pre-population is needed...")

        plot_source = self.plot_outline.get("source", "")
        logger.info(
            f"\n--- SAGA: Pre-populating Knowledge Graph from Initial Data (Plot Source: '{plot_source}') ---"
        )

        profile_objs: list[CharacterProfile] = await get_character_profiles()
        world_objs: list[WorldItem] = await get_world_building()

        await self.knowledge_agent.persist_profiles(
            profile_objs, config.KG_PREPOPULATION_CHAPTER_NUM
        )
        await self.knowledge_agent.persist_world(
            world_objs, config.KG_PREPOPULATION_CHAPTER_NUM
        )
        logger.info("   Knowledge Graph pre-population step complete.")
        self._update_rich_display(step="KG Pre-population Complete")

    def _get_plot_point_info_for_chapter(
        self, novel_chapter_number: int
    ) -> tuple[str | None, int]:
        plot_points_list = self.plot_outline.get("plot_points", [])
        if not isinstance(plot_points_list, list) or not plot_points_list:
            logger.error(
                f"No plot points available in orchestrator state for chapter {novel_chapter_number}."
            )
            return None, -1

        plot_point_index = novel_chapter_number - 1

        if 0 <= plot_point_index < len(plot_points_list):
            plot_point_item = plot_points_list[plot_point_index]
            plot_point_text = (
                plot_point_item.get("description")
                if isinstance(plot_point_item, dict)
                else str(plot_point_item)
            )
            if isinstance(plot_point_text, str) and plot_point_text.strip():
                return plot_point_text, plot_point_index
            logger.warning(
                f"Plot point at index {plot_point_index} for chapter {novel_chapter_number} is empty or invalid. Using placeholder."
            )
            return config.FILL_IN, plot_point_index
        else:
            logger.error(
                f"Plot point index {plot_point_index} is out of bounds for plot_points list (len: {len(plot_points_list)}) for chapter {novel_chapter_number}."
            )
            return None, -1

    async def _save_chapter_text_and_log(
        self, chapter_number: int, final_text: str, raw_llm_log: str | None
    ):
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                self._save_chapter_files_sync_io,
                chapter_number,
                final_text,
                raw_llm_log or "N/A",
            )
            logger.info(
                f"Saved chapter text and raw LLM log files for ch {chapter_number}."
            )
        except OSError as e:
            logger.error(
                f"Failed writing chapter text/log files for ch {chapter_number}: {e}",
                exc_info=True,
            )

    def _save_chapter_files_sync_io(
        self, chapter_number: int, final_text: str, raw_llm_log: str
    ):
        chapter_file_path = os.path.join(
            config.CHAPTERS_DIR, f"chapter_{chapter_number:04d}.txt"
        )
        log_file_path = os.path.join(
            config.CHAPTER_LOGS_DIR,
            f"chapter_{chapter_number:04d}_raw_llm_log.txt",
        )
        os.makedirs(os.path.dirname(chapter_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(chapter_file_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(raw_llm_log)

    async def _save_debug_output(
        self, chapter_number: int, stage_description: str, content: Any
    ):
        if content is None:
            return
        content_str = str(content) if not isinstance(content, str) else content
        if not content_str.strip():
            return
        loop = asyncio.get_running_loop()
        try:
            safe_stage_desc = "".join(
                c if c.isalnum() or c in ["_", "-"] else "_" for c in stage_description
            )
            file_name = f"chapter_{chapter_number:04d}_{safe_stage_desc}.txt"
            file_path = os.path.join(config.DEBUG_OUTPUTS_DIR, file_name)
            await loop.run_in_executor(
                None, self._save_debug_output_sync_io, file_path, content_str
            )
            logger.debug(
                f"Saved debug output for Ch {chapter_number}, Stage '{stage_description}' to {file_path}"
            )
        except Exception as e:
            logger.error(
                f"Failed to save debug output (Ch {chapter_number}, Stage '{stage_description}'): {e}",
                exc_info=True,
            )

    def _save_debug_output_sync_io(self, file_path: str, content_str: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content_str)

    async def perform_deduplication(
        self, text_to_dedup: str, chapter_number: int
    ) -> tuple[str, int]:
        logger.info(f"SAGA: Performing de-duplication for Chapter {chapter_number}...")
        if not text_to_dedup or not text_to_dedup.strip():
            logger.info(
                f"De-duplication for Chapter {chapter_number}: Input text is empty. No action taken."
            )
            return text_to_dedup, 0
        try:
            deduper = TextDeduplicator(
                similarity_threshold=config.DEDUPLICATION_SEMANTIC_THRESHOLD,
                use_semantic_comparison=config.DEDUPLICATION_USE_SEMANTIC,
                min_segment_length_chars=config.DEDUPLICATION_MIN_SEGMENT_LENGTH,
            )
            deduplicated_text, chars_removed = await deduper.deduplicate(
                text_to_dedup, segment_level="sentence"
            )
            if chars_removed > 0:
                method = (
                    "semantic"
                    if config.DEDUPLICATION_USE_SEMANTIC
                    else "normalized string"
                )
                logger.info(
                    f"De-duplication for Chapter {chapter_number} removed {chars_removed} text characters using {method} matching."
                )
            else:
                logger.info(
                    f"De-duplication for Chapter {chapter_number}: No significant duplicates found."
                )
            return deduplicated_text, chars_removed
        except Exception as e:
            logger.error(
                f"Error during de-duplication for Chapter {chapter_number}: {e}",
                exc_info=True,
            )
            return text_to_dedup, 0

    async def _run_evaluation_cycle(
        self,
        novel_chapter_number: int,
        attempt: int,
        current_text: str,
        plot_point_focus: str,
        plot_point_index: int,
        hybrid_context_for_draft: str,
        patched_spans: list[tuple[int, int]],
    ) -> tuple[
        EvaluationResult,
        list[ProblemDetail],
        dict[str, int] | None,
        dict[str, int] | None,
    ]:
        # Sequential by design for single-user determinism; no parallel tasks here.
        self._update_rich_display(
            step=f"Ch {novel_chapter_number} - Evaluation Cycle {attempt}"
        )

        revision_result: tuple[bool, list[str]] | None = None
        if (
            config.ENABLE_COMPREHENSIVE_EVALUATION
            or config.ENABLE_WORLD_CONTINUITY_CHECK
        ):
            # Use the consolidated RevisionAgent for both evaluation and continuity checks
            world_state = {
                "plot_outline": self.plot_outline,
                "chapter_number": novel_chapter_number,
                "previous_chapters_context": hybrid_context_for_draft,
            }
            is_valid, issues = await self.revision_agent.validate_revision(
                current_text,
                "",  # Previous chapter text (not available in this context)
                world_state,
            )
            revision_result = (is_valid, issues)

        # Create evaluation result object from revision result
        eval_result_obj = {
            "needs_revision": not revision_result[0] if revision_result else False,
            "reasons": revision_result[1]
            if revision_result and revision_result[1]
            else [],
            "problems_found": [],
            "coherence_score": None,
            "consistency_issues": None,
            "plot_deviation_reason": None,
            "thematic_issues": None,
            "narrative_depth_issues": None,
        }

        # Convert revision issues to ProblemDetail format
        continuity_problems = []
        if revision_result and revision_result[1]:
            for issue in revision_result[1]:
                continuity_problems.append(
                    {
                        "issue_category": "consistency",
                        "problem_description": issue,
                        "quote_from_original_text": "N/A - General Issue",
                        "quote_char_start": None,
                        "quote_char_end": None,
                        "sentence_char_start": None,
                        "sentence_char_end": None,
                        "suggested_fix_focus": "Address consistency issues identified by revision agent.",
                    }
                )

        return eval_result_obj, continuity_problems, None, None

    async def _perform_revisions(
        self,
        novel_chapter_number: int,
        attempt: int,
        current_text: str,
        evaluation_result: EvaluationResult,
        hybrid_context_for_draft: str,
        chapter_plan: list[SceneDetail] | None,
        patched_spans: list[tuple[int, int]],
        is_from_flawed_source_for_kg: bool,
    ) -> tuple[str | None, str | None, list[tuple[int, int]], dict[str, int] | None]:
        if attempt >= config.MAX_REVISION_CYCLES_PER_CHAPTER:
            logger.error(
                f"SAGA: Ch {novel_chapter_number} - Max revision attempts ({config.MAX_REVISION_CYCLES_PER_CHAPTER}) reached."
            )
            return current_text, None, patched_spans, None

        # Use native queries for optimal performance (Phase 3 optimization)
        characters_for_revision = await get_character_profiles()
        world_items_for_revision = await get_world_building()

        # Convert to dict format for existing revision logic (temporary)
        characters_dict = {char.name: char for char in characters_for_revision}
        world_dict = {}
        for item in world_items_for_revision:
            if item.category not in world_dict:
                world_dict[item.category] = {}
            world_dict[item.category][item.name] = item

        revision_tuple_result, revision_usage_data = await revise_chapter_draft_logic(
            self.plot_outline,
            characters_dict,
            world_dict,
            current_text,
            novel_chapter_number,
            evaluation_result,
            hybrid_context_for_draft,
            chapter_plan,
            is_from_flawed_source=is_from_flawed_source_for_kg,
            already_patched_spans=patched_spans,
        )

        if (
            revision_tuple_result
            and revision_tuple_result[0]
            and len(revision_tuple_result[0]) > 50
        ):
            new_text, rev_raw_output, new_spans = revision_tuple_result
            patched_spans = new_spans
            return new_text, rev_raw_output, patched_spans, revision_usage_data

        logger.error(
            f"SAGA: Ch {novel_chapter_number} - Revision attempt {attempt} failed to produce usable text."
        )
        return current_text, None, patched_spans, revision_usage_data

    async def _prepare_chapter_prerequisites(
        self, novel_chapter_number: int
    ) -> tuple[str | None, int, list[SceneDetail] | None, str | None]:
        """Gather planning and context needed before drafting a chapter."""
        self._update_rich_display(
            step=f"Ch {novel_chapter_number} - Preparing Prerequisites"
        )

        plot_point_focus, plot_point_index = self._get_plot_point_info_for_chapter(
            novel_chapter_number
        )
        if plot_point_focus is None:
            logger.error(
                f"SAGA: Ch {novel_chapter_number} prerequisite check failed: no concrete plot point focus (index {plot_point_index})."
            )
            return None, -1, None, None

        self._update_novel_props_cache()

        # Use scene planning for optimal performance
        characters_for_planning = await get_character_profiles()
        world_items_for_planning = await get_world_building()

        (
            chapter_plan,
            _,  # unused usage data
        ) = await self.narrative_agent._plan_chapter_scenes(
            self.plot_outline,
            characters_for_planning,  # List[CharacterProfile]
            world_items_for_planning,  # List[WorldItem]
            novel_chapter_number,
            plot_point_focus,
            plot_point_index,
        )

        if (
            config.ENABLE_SCENE_PLAN_VALIDATION
            and chapter_plan is not None
            and config.ENABLE_WORLD_CONTINUITY_CHECK
        ):
            # Use RevisionAgent for scene plan consistency check
            world_state = {
                "plot_outline": self.plot_outline,
                "chapter_number": novel_chapter_number,
                "previous_chapters_context": "",
            }

            # Create a draft text from the chapter plan for validation
            draft_text = "\n".join(
                [scene.get("description", "") for scene in chapter_plan]
            )

            is_valid, issues = await self.revision_agent.validate_revision(
                draft_text, "", world_state
            )

            plan_problems = []
            if not is_valid:
                for issue in issues:
                    plan_problems.append(
                        {
                            "issue_category": "consistency",
                            "problem_description": issue,
                            "quote_from_original_text": "N/A - Plan Issue",
                            "quote_char_start": None,
                            "quote_char_end": None,
                            "sentence_char_start": None,
                            "sentence_char_end": None,
                            "suggested_fix_focus": "Address scene plan consistency issues.",
                        }
                    )

            # Mock usage data for now

            await self._save_debug_output(
                novel_chapter_number,
                "scene_plan_consistency_problems",
                plan_problems,
            )
            if plan_problems:
                logger.warning(
                    f"SAGA: Ch {novel_chapter_number} scene plan has {len(plan_problems)} consistency issues."
                )

        hybrid_context_for_draft = (
            await ZeroCopyContextGenerator.generate_hybrid_context_native(
                self.plot_outline, novel_chapter_number, chapter_plan
            )
        )

        if config.ENABLE_AGENTIC_PLANNING and chapter_plan is None:
            logger.warning(
                f"SAGA: Ch {novel_chapter_number}: Planning Agent failed or plan invalid. Proceeding with plot point focus only."
            )
        await self._save_debug_output(
            novel_chapter_number,
            "scene_plan",
            chapter_plan if chapter_plan else "No plan generated.",
        )
        await self._save_debug_output(
            novel_chapter_number,
            "hybrid_context_for_draft",
            hybrid_context_for_draft,
        )

        return (
            plot_point_focus,
            plot_point_index,
            chapter_plan,
            hybrid_context_for_draft,
        )

    async def _draft_initial_chapter_text(
        self,
        novel_chapter_number: int,
        plot_point_focus: str,
        hybrid_context_for_draft: str,
        chapter_plan: list[SceneDetail] | None,
    ) -> tuple[str | None, str | None]:
        self._update_rich_display(
            step=f"Ch {novel_chapter_number} - Drafting Initial Text"
        )
        # Prefer using precomputed chapter_plan and hybrid context if available to avoid re-planning and ensure continuity.
        # Use native queries for optimal performance (Phase 3 optimization)
        characters = await get_character_profiles()
        world_items = await get_world_building()
        if chapter_plan is not None and hybrid_context_for_draft is not None:
            # Draft the chapter directly using the prepared scenes and context.
            # _draft_chapter expects: plot_outline, chapter_number, plot_point_focus, hybrid_context_for_draft, chapter_plan
            (
                draft_text,
                raw_output,
                _,  # unused usage data
            ) = await self.narrative_agent.draft_chapter(
                self.plot_outline,
                novel_chapter_number,
                plot_point_focus,
                hybrid_context_for_draft,
                chapter_plan,
            )
            if not draft_text:
                logger.error(
                    f"SAGA: Drafting Agent failed for Ch {novel_chapter_number}. No initial draft produced."
                )
                await self._save_debug_output(
                    novel_chapter_number,
                    "initial_draft_fail_raw_llm",
                    raw_output or "Drafting Agent returned None for raw output.",
                )
                return None, None
            await self._save_debug_output(
                novel_chapter_number, "initial_draft", draft_text
            )
            return draft_text, raw_output
        # Fallback: if no valid plan/context, use generate_chapter for optimal performance
        (
            initial_draft_text,
            initial_raw_llm_text,
            _,  # unused usage data
        ) = await self.narrative_agent.generate_chapter(
            self.plot_outline,
            characters,  # List[CharacterProfile]
            world_items,  # List[WorldItem]
            novel_chapter_number,
            plot_point_focus,
        )
        if not initial_draft_text:
            logger.error(
                f"SAGA: Drafting Agent failed for Ch {novel_chapter_number}. No initial draft produced."
            )
            await self._save_debug_output(
                novel_chapter_number,
                "initial_draft_fail_raw_llm",
                initial_raw_llm_text or "Drafting Agent returned None for raw output.",
            )
            return None, None
        await self._save_debug_output(
            novel_chapter_number, "initial_draft", initial_draft_text
        )
        return initial_draft_text, initial_raw_llm_text

    async def _process_and_revise_draft(
        self,
        novel_chapter_number: int,
        initial_draft_text: str,
        initial_raw_llm_text: str | None,
        plot_point_focus: str,
        plot_point_index: int,
        hybrid_context_for_draft: str,
        chapter_plan: list[SceneDetail] | None,
    ) -> tuple[str | None, str | None, bool]:
        # FAST PATH: If all evaluation agents are disabled, just de-duplicate and return.
        if (
            not config.ENABLE_COMPREHENSIVE_EVALUATION
            and not config.ENABLE_WORLD_CONTINUITY_CHECK
        ):
            logger.info(
                f"SAGA: Ch {novel_chapter_number} - All evaluation agents disabled. Applying de-duplication and finalizing draft."
            )
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} - Skipping Revisions (disabled)"
            )

            deduplicated_text, removed_char_count = await self.perform_deduplication(
                initial_draft_text, novel_chapter_number
            )
            is_from_flawed_source_for_kg = removed_char_count > 0
            if is_from_flawed_source_for_kg:
                logger.info(
                    f"SAGA: Ch {novel_chapter_number} - Text marked as flawed for KG due to de-duplication removing {removed_char_count} characters."
                )
                await self._save_debug_output(
                    novel_chapter_number,
                    "deduplicated_text_no_eval_path",
                    deduplicated_text,
                )

            return (
                deduplicated_text,
                initial_raw_llm_text,
                is_from_flawed_source_for_kg,
            )

        current_text_to_process: str | None = initial_draft_text
        current_raw_llm_output: str | None = initial_raw_llm_text
        is_from_flawed_source_for_kg = False
        patched_spans: list[tuple[int, int]] = []

        # The de-duplication step is a single, definitive cleaning step after drafting and before evaluation.
        self._update_rich_display(
            step=f"Ch {novel_chapter_number} - Post-Draft De-duplication"
        )
        logger.info(
            f"SAGA: Ch {novel_chapter_number} - Applying post-draft de-duplication."
        )
        (
            deduplicated_text,
            removed_char_count,
        ) = await self.perform_deduplication(
            current_text_to_process, novel_chapter_number
        )
        if removed_char_count > 0:
            is_from_flawed_source_for_kg = True
            logger.info(
                f"SAGA: Ch {novel_chapter_number} - De-duplication removed {removed_char_count} characters. Text marked as potentially flawed for KG."
            )
            current_text_to_process = deduplicated_text
            await self._save_debug_output(
                novel_chapter_number,
                "deduplicated_text_after_draft",
                current_text_to_process,
            )
        else:
            logger.info(
                f"SAGA: Ch {novel_chapter_number} - Post-draft de-duplication found no significant changes."
            )

        revisions_made = 0
        needs_revision = True
        while (
            needs_revision and revisions_made < config.MAX_REVISION_CYCLES_PER_CHAPTER
        ):
            attempt = revisions_made + 1
            if current_text_to_process is None:
                logger.error(
                    f"SAGA: Ch {novel_chapter_number} - Text became None before processing cycle {attempt}. Aborting chapter."
                )
                return None, None, True
            (
                eval_result_obj,
                continuity_problems,
                _,  # unused eval usage data
                _,  # unused continuity usage data
            ) = await self._run_evaluation_cycle(
                novel_chapter_number,
                attempt,
                current_text_to_process,
                plot_point_focus,
                plot_point_index,
                hybrid_context_for_draft,
                patched_spans,
            )

            evaluation_result: EvaluationResult = eval_result_obj
            await self._save_debug_output(
                novel_chapter_number,
                f"evaluation_result_attempt_{attempt}",
                evaluation_result,
            )
            await self._save_debug_output(
                novel_chapter_number,
                f"continuity_problems_attempt_{attempt}",
                continuity_problems,
            )

            if continuity_problems:
                logger.warning(
                    f"SAGA: Ch {novel_chapter_number} (Attempt {attempt}) - Revision Agent found {len(continuity_problems)} issues."
                )
                evaluation_result["problems_found"].extend(continuity_problems)
                if not evaluation_result["needs_revision"]:
                    evaluation_result["needs_revision"] = True
                unique_reasons = set(evaluation_result.get("reasons", []))
                unique_reasons.add("Issues identified by RevisionAgent.")
                evaluation_result["reasons"] = sorted(list(unique_reasons))

            needs_revision = evaluation_result["needs_revision"]
            if not needs_revision:
                logger.info(
                    f"SAGA: Ch {novel_chapter_number} draft passed evaluation (Attempt {attempt}). Text is considered good."
                )
                self._update_rich_display(
                    step=f"Ch {novel_chapter_number} - Passed Evaluation"
                )
                break
            else:
                is_from_flawed_source_for_kg = True
                logger.warning(
                    f"SAGA: Ch {novel_chapter_number} draft (Attempt {attempt}) needs revision. Reasons: {'; '.join(evaluation_result.get('reasons', []))}"
                )
                self._update_rich_display(
                    step=f"Ch {novel_chapter_number} - Revision Attempt {attempt}"
                )
                (
                    new_text,
                    rev_raw_output,
                    patched_spans,
                    _,  # unused revision usage data
                ) = await self._perform_revisions(
                    novel_chapter_number,
                    attempt,
                    current_text_to_process,
                    evaluation_result,
                    hybrid_context_for_draft,
                    chapter_plan,
                    patched_spans,
                    is_from_flawed_source_for_kg,
                )
                if new_text and new_text != current_text_to_process:
                    new_embedding, prev_embedding = await asyncio.gather(
                        llm_service.async_get_embedding(new_text),
                        llm_service.async_get_embedding(current_text_to_process),
                    )
                    if new_embedding is not None and prev_embedding is not None:
                        similarity = utils.numpy_cosine_similarity(
                            prev_embedding, new_embedding
                        )
                        if similarity > config.REVISION_SIMILARITY_ACCEPTANCE:
                            logger.warning(
                                f"SAGA: Ch {novel_chapter_number} revision attempt {attempt} produced text too similar to previous (score: {similarity:.4f}). Stopping revisions."
                            )
                            current_text_to_process = new_text
                            current_raw_llm_output = (
                                rev_raw_output or current_raw_llm_output
                            )
                            break
                    current_text_to_process = new_text
                    current_raw_llm_output = rev_raw_output or current_raw_llm_output
                    logger.info(
                        f"SAGA: Ch {novel_chapter_number} - Revision attempt {attempt} successful. New text length: {len(current_text_to_process)}. Re-evaluating."
                    )
                    await self._save_debug_output(
                        novel_chapter_number,
                        f"revised_text_attempt_{attempt}",
                        current_text_to_process,
                    )
                    revisions_made += 1
                else:
                    logger.error(
                        f"SAGA: Ch {novel_chapter_number} - Revision attempt {attempt} failed to produce usable text. Proceeding with previous draft, marked as flawed."
                    )
                    self._update_rich_display(
                        step=f"Ch {novel_chapter_number} - Revision Failed (Flawed)"
                    )
                    break
        if current_text_to_process is None:
            logger.critical(
                f"SAGA: Ch {novel_chapter_number} - current_text_to_process is None after revision loop. Aborting chapter."
            )
            return None, None, True

        dedup_text_after_rev, removed_after_rev = await self.perform_deduplication(
            current_text_to_process, novel_chapter_number
        )
        if removed_after_rev > 0:
            logger.info(
                f"SAGA: Ch {novel_chapter_number} - De-duplication after revisions removed {removed_after_rev} characters."
            )
            current_text_to_process = dedup_text_after_rev
            is_from_flawed_source_for_kg = True
            await self._save_debug_output(
                novel_chapter_number,
                "deduplicated_text_after_revision",
                current_text_to_process,
            )

        if len(current_text_to_process) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
            logger.warning(
                f"SAGA: Final chosen text for Ch {novel_chapter_number} is short ({len(current_text_to_process)} chars). Marked as flawed for KG."
            )
            is_from_flawed_source_for_kg = True

        return (
            current_text_to_process,
            current_raw_llm_output,
            is_from_flawed_source_for_kg,
        )

    async def _finalize_and_save_chapter(
        self,
        novel_chapter_number: int,
        final_text_to_process: str,
        final_raw_llm_output: str | None,
        is_from_flawed_source_for_kg: bool,
    ) -> str | None:
        self._update_rich_display(step=f"Ch {novel_chapter_number} - Finalization")

        # Generate chapter summary
        summary, _ = await self.knowledge_agent.summarize_chapter(
            final_text_to_process, novel_chapter_number
        )

        # Get text embedding
        embedding = await llm_service.async_get_embedding(final_text_to_process)

        # Extract and merge knowledge updates using native models for performance
        characters = await get_character_profiles()
        world_items = await get_world_building()

        _ = await self.knowledge_agent.extract_and_merge_knowledge(
            self.plot_outline,
            characters,  # List of CharacterProfile models
            world_items,  # List of WorldItem models
            novel_chapter_number,
            final_text_to_process,
            is_from_flawed_source_for_kg,
        )

        result = {
            "summary": summary,
            "embedding": embedding,
        }

        await self._save_debug_output(
            novel_chapter_number, "final_summary", result.get("summary")
        )

        # Save chapter data to Neo4j database
        try:
            await chapter_queries.save_chapter_data_to_db(
                chapter_number=novel_chapter_number,
                text=final_text_to_process,
                raw_llm_output=final_raw_llm_output or "",
                summary=result.get("summary"),
                embedding_array=result.get("embedding"),
                is_provisional=is_from_flawed_source_for_kg,
            )
            logger.info(
                f"Neo4j: Successfully saved chapter data for chapter {novel_chapter_number} to database."
            )
        except Exception as e:
            logger.error(
                f"Neo4j: Failed to save chapter data for chapter {novel_chapter_number} to database: {e}",
                exc_info=True,
            )

        if result.get("embedding") is None:
            logger.error(
                "SAGA CRITICAL: Failed to generate embedding for final text of Chapter %s. Text saved to file system only.",
                novel_chapter_number,
            )
            await self._save_chapter_text_and_log(
                novel_chapter_number,
                final_text_to_process,
                final_raw_llm_output,
            )
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} Failed - No Embedding"
            )
            return None

        await self._save_chapter_text_and_log(
            novel_chapter_number, final_text_to_process, final_raw_llm_output
        )

        self.chapter_count = max(self.chapter_count, novel_chapter_number)

        return final_text_to_process

    async def _validate_plot_outline(self, novel_chapter_number: int) -> bool:
        if (
            not self.plot_outline
            or not self.plot_outline.get("plot_points")
            or not self.plot_outline.get("protagonist_name")
        ):
            logger.error(
                f"SAGA: Cannot write Ch {novel_chapter_number}: Plot outline or critical plot data missing."
            )
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} Failed - Missing Plot Outline"
            )
            return False
        return True

    async def _process_prereq_result(
        self,
        novel_chapter_number: int,
        prereq_result: tuple[str | None, int, list[SceneDetail] | None, str | None],
    ) -> tuple[str, int, list[SceneDetail] | None, str] | None:
        (
            plot_point_focus,
            plot_point_index,
            chapter_plan,
            hybrid_context_for_draft,
        ) = prereq_result

        if plot_point_focus is None or hybrid_context_for_draft is None:
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} Failed - Prerequisites Incomplete"
            )
            return None
        return (
            plot_point_focus,
            plot_point_index,
            chapter_plan,
            hybrid_context_for_draft,
        )

    async def _process_initial_draft(
        self,
        novel_chapter_number: int,
        draft_result: tuple[str | None, str | None],
    ) -> tuple[str, str | None] | None:
        initial_draft_text, initial_raw_llm_text = draft_result
        if initial_draft_text is None:
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} Failed - No Initial Draft"
            )
            return None
        return initial_draft_text, initial_raw_llm_text

    async def _process_revision_result(
        self,
        novel_chapter_number: int,
        revision_result: tuple[str | None, str | None, bool],
    ) -> tuple[str, str | None, bool] | None:
        processed_text, processed_raw_llm, is_flawed = revision_result
        if processed_text is None:
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} Failed - Revision/Processing Error"
            )
            return None
        return processed_text, processed_raw_llm, is_flawed

    async def _finalize_and_log(
        self,
        novel_chapter_number: int,
        processed_text: str,
        processed_raw_llm: str | None,
        is_flawed: bool,
    ) -> str | None:
        final_text_result = await self._finalize_and_save_chapter(
            novel_chapter_number, processed_text, processed_raw_llm, is_flawed
        )

        if final_text_result:
            status_message = (
                "Successfully Generated"
                if not is_flawed
                else "Generated (Marked with Flaws)"
            )
            logger.info(
                f"=== SAGA: Finished Novel Chapter {novel_chapter_number} - {status_message} ==="
            )
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} - {status_message}"
            )
        else:
            logger.error(
                f"=== SAGA: Failed Novel Chapter {novel_chapter_number} - Finalization/Save Error ==="
            )
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} Failed - Finalization Error"
            )
        return final_text_result

    async def run_chapter_generation_process(
        self, novel_chapter_number: int
    ) -> str | None:
        return await run_chapter_pipeline(self, novel_chapter_number)

    def _validate_critical_configs(self) -> bool:
        critical_str_configs = {
            "EMBEDDING_API_BASE": config.EMBEDDING_API_BASE,
            "OPENAI_API_BASE": config.OPENAI_API_BASE,
            "EMBEDDING_MODEL": config.EMBEDDING_MODEL,
            "NEO4J_URI": config.NEO4J_URI,
            "LARGE_MODEL": config.Models.LARGE,
            "MEDIUM_MODEL": config.Models.MEDIUM,
            "SMALL_MODEL": config.Models.SMALL,
            "NARRATIVE_MODEL": config.Models.NARRATOR,
        }
        missing_or_empty_configs = []
        for name, value in critical_str_configs.items():
            if not value or not isinstance(value, str) or not value.strip():
                missing_or_empty_configs.append(name)

        if missing_or_empty_configs:
            logger.critical(
                f"SAGA CRITICAL CONFIGURATION ERROR: The following critical configuration(s) are missing or empty: {', '.join(missing_or_empty_configs)}. Please set them (e.g., in .env file or environment variables) and restart."
            )
            return False

        if config.EXPECTED_EMBEDDING_DIM <= 0:
            logger.critical(
                f"SAGA CRITICAL CONFIGURATION ERROR: EXPECTED_EMBEDDING_DIM must be a positive integer, but is {config.EXPECTED_EMBEDDING_DIM}."
            )
            return False

        logger.info("Critical configurations validated successfully.")
        return True

    async def run_novel_generation_loop(self):
        logger.info("--- SAGA: Starting Novel Generation Run ---")

        if not self._validate_critical_configs():
            self._update_rich_display(step="Critical Config Error - Halting")
            await self.display.stop()
            return

        self.run_start_time = time.time()
        self.display.start()
        try:
            await neo4j_manager.connect()
            await neo4j_manager.create_db_schema()
            logger.info("SAGA: Neo4j connection and schema verified.")

            await self.knowledge_agent.load_schema_from_db()
            logger.info("SAGA: KG schema loaded into knowledge agent.")

            await self._prime_from_user_story_elements()

            await self.async_init_orchestrator()

            plot_points_exist = (
                self.plot_outline
                and self.plot_outline.get("plot_points")
                and len(
                    [
                        pp
                        for pp in self.plot_outline.get("plot_points", [])
                        if not utils._is_fill_in(pp)
                    ]
                )
                > 0
            )

            if (
                not plot_points_exist
                or not self.plot_outline.get("title")
                or utils._is_fill_in(self.plot_outline.get("title"))
            ):
                logger.info(
                    "SAGA: Core plot data missing or insufficient (e.g., no title, no concrete plot points). Performing initial setup..."
                )
                if not await self.perform_initial_setup():
                    logger.critical("SAGA: Initial setup failed. Halting generation.")
                    self._update_rich_display(step="Initial Setup Failed - Halting")
                    return
                self._update_novel_props_cache()

            # KG pre-population is performed during bootstrap pipeline execution

            logger.info("\n--- SAGA: Starting Novel Writing Process ---")

            plot_points_raw = self.plot_outline.get("plot_points", [])
            if isinstance(plot_points_raw, list):
                plot_points_list = plot_points_raw
            elif isinstance(plot_points_raw, dict):
                plot_points_list = list(plot_points_raw.values())
            elif plot_points_raw:
                plot_points_list = [plot_points_raw]
            else:
                plot_points_list = []

            total_concrete_plot_points = len(
                [
                    pp
                    for pp in plot_points_list
                    if not utils._is_fill_in(pp) and isinstance(pp, str) and pp.strip()
                ]
            )

            remaining_plot_points_to_address_in_novel = (
                total_concrete_plot_points - self.chapter_count
            )

            logger.info(
                f"SAGA: Current Novel Chapter Count (State): {self.chapter_count}"
            )
            logger.info(
                f"SAGA: Total Concrete Plot Points in Outline: {total_concrete_plot_points}"
            )
            logger.info(
                f"SAGA: Remaining Concrete Plot Points to Cover in Novel: {remaining_plot_points_to_address_in_novel}"
            )

            if remaining_plot_points_to_address_in_novel <= 0:
                await self._generate_plot_points_from_kg(config.CHAPTERS_PER_RUN)
                await self.refresh_plot_outline()

            logger.info(
                f"SAGA: Starting dynamic chapter loop (max {config.CHAPTERS_PER_RUN} chapter(s) this run)."
            )

            chapters_successfully_written_this_run = 0
            attempts_this_run = 0
            while attempts_this_run < config.CHAPTERS_PER_RUN:
                plot_points_raw = self.plot_outline.get("plot_points", [])
                if isinstance(plot_points_raw, list):
                    plot_points_list = plot_points_raw
                elif isinstance(plot_points_raw, dict):
                    plot_points_list = list(plot_points_raw.values())
                elif plot_points_raw:
                    plot_points_list = [plot_points_raw]
                else:
                    plot_points_list = []

                total_concrete_plot_points = len(
                    [
                        pp
                        for pp in plot_points_list
                        if not utils._is_fill_in(pp)
                        and isinstance(pp, str)
                        and pp.strip()
                    ]
                )
                remaining_plot_points_to_address_in_novel = (
                    total_concrete_plot_points - self.chapter_count
                )

                if remaining_plot_points_to_address_in_novel <= 0:
                    await self._generate_plot_points_from_kg(
                        config.CHAPTERS_PER_RUN - attempts_this_run
                    )
                    await self.refresh_plot_outline()
                    plot_points_raw = self.plot_outline.get("plot_points", [])
                    if isinstance(plot_points_raw, list):
                        plot_points_list = plot_points_raw
                    elif isinstance(plot_points_raw, dict):
                        plot_points_list = list(plot_points_raw.values())
                    elif plot_points_raw:
                        plot_points_list = [plot_points_raw]
                    else:
                        plot_points_list = []

                    total_concrete_plot_points = len(
                        [
                            pp
                            for pp in plot_points_list
                            if not utils._is_fill_in(pp)
                            and isinstance(pp, str)
                            and pp.strip()
                        ]
                    )
                    remaining_plot_points_to_address_in_novel = (
                        total_concrete_plot_points - self.chapter_count
                    )
                    if remaining_plot_points_to_address_in_novel <= 0:
                        logger.info(
                            "SAGA: No plot points available after generation. Ending run early."
                        )
                        break

                current_novel_chapter_number = self.chapter_count + 1
                plot_point_index = current_novel_chapter_number - 1

                # Get the correct plot point focus for this chapter
                plot_points = self.plot_outline.get("plot_points", [])
                if plot_point_index < len(plot_points):
                    plot_point_focus = plot_points[plot_point_index]
                else:
                    # Fallback to last available plot point if chapter count exceeds plot points
                    plot_point_focus = (
                        plot_points[-1] if plot_points else "No plot point available"
                    )

                logger.info(
                    f"\n--- SAGA: Attempting Novel Chapter {current_novel_chapter_number} of {config.CHAPTERS_PER_RUN} ---"
                )
                self._update_rich_display(
                    chapter_num=current_novel_chapter_number,
                    step="Starting Chapter Loop",
                )

                try:
                    chapter_text_result = await self.run_chapter_generation_process(
                        current_novel_chapter_number
                    )
                    if chapter_text_result:
                        chapters_successfully_written_this_run += 1
                        logger.info(
                            f"SAGA: Novel Chapter {current_novel_chapter_number}: Processed. Final text length: {len(chapter_text_result)} chars."
                        )
                        logger.info(
                            f"   Snippet: {chapter_text_result[:200].replace(chr(10), ' ')}..."
                        )

                        if (
                            current_novel_chapter_number > 0
                            and current_novel_chapter_number
                            % config.KG_HEALING_INTERVAL
                            == 0
                        ):
                            logger.info(
                                f"\n--- SAGA: Triggering KG Healing/Enrichment after Chapter {current_novel_chapter_number} ---"
                            )
                            self._update_rich_display(
                                step=f"Ch {current_novel_chapter_number} - KG Maintenance"
                            )
                            await self.knowledge_agent.heal_and_enrich_kg()
                            await self.refresh_plot_outline()
                            logger.info(
                                "--- SAGA: KG Healing/Enrichment cycle complete. ---"
                            )
                    else:
                        logger.error(
                            f"SAGA: Novel Chapter {current_novel_chapter_number}: Failed to generate or save. Halting run."
                        )
                        self._update_rich_display(
                            step=f"Ch {current_novel_chapter_number} Failed - Halting Run"
                        )
                        break
                except Exception as e:
                    logger.critical(
                        f"SAGA: Critical unhandled error during Novel Chapter {current_novel_chapter_number} writing process: {e}",
                        exc_info=True,
                    )
                    self._update_rich_display(
                        step=f"Critical Error Ch {current_novel_chapter_number} - Halting Run"
                    )
                    break

                attempts_this_run += 1

            final_chapter_count_from_db = (
                await chapter_queries.load_chapter_count_from_db()
            )
            logger.info("\n--- SAGA: Novel writing process finished for this run ---")
            logger.info(
                f"SAGA: Successfully processed {chapters_successfully_written_this_run} chapter(s) in this run."
            )
            logger.info(
                f"SAGA: Current total chapters in database after this run: {final_chapter_count_from_db}"
            )

            self._update_rich_display(
                chapter_num=self.chapter_count, step="Run Finished"
            )

        except Exception as e:
            logger.critical(
                f"SAGA: Unhandled exception in orchestrator main loop: {e}",
                exc_info=True,
            )
            self._update_rich_display(step="Critical Error in Main Loop")
        finally:
            await self.display.stop()
            await neo4j_manager.close()
            logger.info("SAGA: Neo4j driver successfully closed on application exit.")

    async def run_ingestion_process(self, text_file: str) -> None:
        """Ingest existing text and populate the knowledge graph."""
        logger.info("--- SAGA: Starting Ingestion Process ---")

        if not self._validate_critical_configs():
            await self.display.stop()
            return

        self.display.start()
        self.run_start_time = time.time()
        await neo4j_manager.connect()
        await neo4j_manager.create_db_schema()
        if neo4j_manager.driver is not None:
            await plot_queries.ensure_novel_info()
        else:
            logger.warning("Neo4j driver not initialized. Skipping NovelInfo setup.")
        await self.knowledge_agent.load_schema_from_db()

        with open(text_file, encoding="utf-8") as f:
            raw_text = f.read()

        chunks = split_text_into_chapters(raw_text)
        plot_outline = {"title": "Ingested Narrative", "plot_points": []}
        characters: list[CharacterProfile] = []
        world_items: list[WorldItem] = []
        summaries: list[str] = []

        for idx, chunk in enumerate(chunks, 1):
            self._update_rich_display(chapter_num=idx, step="Ingesting Text")
            # Generate chapter summary
            summary, _ = await self.knowledge_agent.summarize_chapter(chunk, idx)

            # Get text embedding
            embedding = await llm_service.async_get_embedding(chunk)

            # Extract and merge knowledge updates using NATIVE implementation
            _ = await self.knowledge_agent.extract_and_merge_knowledge(
                plot_outline,
                characters,  # List of CharacterProfile models
                world_items,  # List of WorldItem models
                idx,
                chunk,
                False,  # from_flawed_draft
            )

            result = {
                "summary": summary,
                "embedding": embedding,
            }
            if result.get("summary"):
                summaries.append(str(result["summary"]))
                plot_outline["plot_points"].append(result["summary"])

            if idx % config.KG_HEALING_INTERVAL == 0:
                logger.info(
                    f"--- SAGA: Triggering KG Healing/Enrichment after Ingestion Chunk {idx} ---"
                )
                self._update_rich_display(step=f"Ch {idx} - KG Maintenance")
                await self.knowledge_agent.heal_and_enrich_kg()
                await self.refresh_plot_outline()

        await self.knowledge_agent.heal_and_enrich_kg()
        combined_summary = "\n".join(summaries)
        continuation, _ = await self.narrative_agent.plan_continuation(combined_summary)
        if continuation:
            plot_outline["plot_points"].extend(continuation)
        self.plot_outline = plot_outline
        self.chapter_count = len(chunks)
        await plot_queries.save_plot_outline_to_db(plot_outline)
        await self.display.stop()
        await neo4j_manager.close()
        logger.info("SAGA: Ingestion process completed.")

    async def _validate_runtime_configuration(self) -> None:
        """
        Validate runtime configuration against bootstrap content.

        This method ensures that the current runtime configuration is consistent
        with the bootstrap-generated content to prevent narrative inconsistencies.
        """
        self._update_rich_display(step="Validating Runtime Configuration")

        try:
            # Get current content from database/runtime
            characters = await get_character_profiles()
            world_items = await get_world_building()

            # Convert to dict format for validation
            character_profiles_dict = {char.name: char for char in characters}
            world_building_dict = {}
            for item in world_items:
                if item.category not in world_building_dict:
                    world_building_dict[item.category] = {}
                world_building_dict[item.category][item.name] = item

            # Validate runtime configuration
            is_valid, validation_errors = validate_runtime_configuration(
                self.plot_outline, character_profiles_dict, world_building_dict
            )

            if not is_valid:
                logger.warning(
                    f"Runtime configuration validation failed with {len(validation_errors)} errors: {[str(error) for error in validation_errors]}"
                )

                # Log specific validation errors
                for error in validation_errors:
                    logger.warning(
                        f"Configuration validation error - Field: {error.field}, Message: {error.message}, Bootstrap: {error.bootstrap_value}, Runtime: {error.runtime_value}"
                    )

                # Create validation report
                validation_report = {
                    "validation_timestamp": None,
                    "total_errors": len(validation_errors),
                    "is_valid": False,
                    "errors": [
                        {
                            "field": error.field,
                            "message": error.message,
                            "bootstrap_value": str(error.bootstrap_value),
                            "runtime_value": str(error.runtime_value),
                        }
                        for error in validation_errors
                    ],
                    "summary": f"Runtime validation failed with {len(validation_errors)} error(s)",
                }

                await self._save_debug_output(
                    0, "runtime_config_validation_report", validation_report
                )

            else:
                logger.info("Runtime configuration validation passed")

        except Exception as e:
            logger.error(
                "Runtime configuration validation failed with exception",
                error=str(e),
                exc_info=True,
            )

    # Dynamic schema refresh function removed - not needed for single-user deployment


def setup_logging_nana():
    # Keep logging simple for single-user unless advanced mode is desired.
    stdlib_logging.basicConfig(
        level=config.LOG_LEVEL_STR,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
        handlers=[],
    )
    root_logger = stdlib_logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Simple logging mode: console only, no rotation or Rich coupling
    if getattr(config, "SIMPLE_LOGGING_MODE", False):
        stream_handler = stdlib_logging.StreamHandler()
        stream_handler.setLevel(config.LOG_LEVEL_STR)
        stream_handler.setFormatter(simple_formatter)
        root_logger.addHandler(stream_handler)
        root_logger.info("Simple logging mode enabled: console only.")
    elif config.LOG_FILE:
        try:
            log_path = os.path.join(
                config.settings.BASE_OUTPUT_DIR, config.settings.LOG_FILE
            )
            # Ensure the parent directory exists, not the file path itself
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            file_handler = stdlib_logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                mode="a",
                encoding="utf-8",
            )
            file_handler.setLevel(config.LOG_LEVEL_STR)
            # Use the structlog formatter for human-readable output
            file_handler.setFormatter(simple_formatter)
            root_logger.addHandler(file_handler)
            root_logger.info(f"File logging enabled. Log file: {log_path}")
        except Exception as e:
            console_handler_fallback = stdlib_logging.StreamHandler()
            # Use the structlog formatter for human-readable output
            console_handler_fallback.setFormatter(simple_formatter)
            root_logger.addHandler(console_handler_fallback)
            root_logger.error(
                f"Failed to configure file logging: {e}. Logging to console instead.",
                exc_info=True,
            )

    if (
        not getattr(config, "SIMPLE_LOGGING_MODE", False)
        and RICH_AVAILABLE
        and config.ENABLE_RICH_PROGRESS
    ):
        existing_console = None
        if root_logger.handlers:
            for h_idx, h in enumerate(root_logger.handlers):
                if hasattr(h, "console") and not isinstance(
                    h, stdlib_logging.FileHandler
                ):
                    existing_console = h.console  # type: ignore
                    break

        # Ensure Rich logging uses the same Console as the Live display
        if existing_console is None:
            try:
                existing_console = RichDisplayManager.get_shared_console()
            except Exception:
                existing_console = None

        rich_handler = RichHandler(
            level=config.LOG_LEVEL_STR,
            rich_tracebacks=True,
            show_path=False,
            markup=True,
            show_time=True,
            show_level=True,
            console=existing_console,
        )
        root_logger.addHandler(rich_handler)
        root_logger.info("Rich logging handler enabled for console.")
    elif not any(
        isinstance(h, stdlib_logging.StreamHandler) for h in root_logger.handlers
    ):
        stream_handler = stdlib_logging.StreamHandler()
        stream_handler.setLevel(config.LOG_LEVEL_STR)
        # Use the structlog formatter for human-readable output
        stream_handler.setFormatter(simple_formatter)
        root_logger.addHandler(stream_handler)
        root_logger.info("Standard stream logging handler enabled for console.")

    stdlib_logging.getLogger("neo4j.notifications").setLevel(stdlib_logging.WARNING)
    stdlib_logging.getLogger("neo4j").setLevel(stdlib_logging.WARNING)
    stdlib_logging.getLogger("httpx").setLevel(stdlib_logging.WARNING)
    stdlib_logging.getLogger("httpcore").setLevel(stdlib_logging.WARNING)

    # Log the completion message using structlog to avoid the None logger issue
    import structlog

    structlog.get_logger().info(
        f"SAGA Logging setup complete. Application Log Level: {stdlib_logging.getLevelName(config.LOG_LEVEL_STR)}."
    )
