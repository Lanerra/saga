# nana_orchestrator.py
import asyncio
import logging
import logging.handlers
import os
import time  # For Rich display updates
from typing import Any, Dict, List, Optional, Tuple

import config
import utils
from agents.comprehensive_evaluator_agent import ComprehensiveEvaluatorAgent
from agents.drafting_agent import DraftingAgent
from agents.finalize_agent import FinalizeAgent
from agents.kg_maintainer_agent import KGMaintainerAgent
from agents.planner_agent import PlannerAgent
from agents.world_continuity_agent import WorldContinuityAgent
from core.db_manager import neo4j_manager
from core.llm_interface import llm_service
from data_access import (
    chapter_queries,
    character_queries,
    plot_queries,
    world_queries,
)
from initialization.genesis import run_genesis_phase
from kg_maintainer.models import (
    CharacterProfile,
    EvaluationResult,
    ProblemDetail,
    SceneDetail,
    WorldItem,
)
from models.user_input_models import UserStoryInputModel, user_story_to_objects
from orchestration.chapter_flow import run_chapter_pipeline
from orchestration.token_tracker import TokenTracker
from processing.context_generator import generate_hybrid_chapter_context_logic
from processing.revision_logic import revise_chapter_draft_logic
from processing.text_deduplicator import TextDeduplicator
from ui.rich_display import RichDisplayManager
from utils.ingestion_utils import split_text_into_chapters

try:
    from rich.logging import RichHandler

    RICH_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when Rich is missing
    RICH_AVAILABLE = False

    class RichHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
            logging.getLogger(__name__).handle(record)


logger = logging.getLogger(__name__)


class NANA_Orchestrator:
    def __init__(self):
        logger.info("Initializing NANA Orchestrator...")
        self.planner_agent = PlannerAgent()
        self.drafting_agent = DraftingAgent()
        self.evaluator_agent = ComprehensiveEvaluatorAgent()
        self.world_continuity_agent = WorldContinuityAgent()
        self.kg_maintainer_agent = KGMaintainerAgent()
        self.finalize_agent = FinalizeAgent(self.kg_maintainer_agent)

        self.plot_outline: Dict[str, Any] = {}
        self.chapter_count: int = 0
        self.novel_props_cache: Dict[str, Any] = {}
        self.token_tracker = TokenTracker()
        self.total_tokens_generated_this_run: int = 0

        self.display = RichDisplayManager()
        self.run_start_time: float = 0.0
        utils.load_spacy_model_if_needed()
        logger.info("NANA Orchestrator initialized.")

    def _update_rich_display(
        self, chapter_num: Optional[int] = None, step: Optional[str] = None
    ) -> None:
        self.display.update(
            plot_outline=self.plot_outline,
            chapter_num=chapter_num,
            step=step,
            total_tokens=self.total_tokens_generated_this_run,
            run_start_time=self.run_start_time,
        )

    def _accumulate_tokens(
        self, operation_name: str, usage_data: Optional[Dict[str, int]]
    ):
        self.token_tracker.add(operation_name, usage_data)
        self.total_tokens_generated_this_run = self.token_tracker.total
        self._update_rich_display()

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
            logger.error("Failed to refresh plot outline from DB: %s", result)

    async def async_init_orchestrator(self):
        logger.info("NANA Orchestrator async_init_orchestrator started...")
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
        logger.info("NANA Orchestrator async_init_orchestrator complete.")
        self._update_rich_display(step="Orchestrator Initialized")

    async def perform_initial_setup(self):
        self._update_rich_display(step="Performing Initial Setup")
        logger.info("NANA performing initial setup...")
        (
            self.plot_outline,
            character_profiles,
            world_building,
            usage,
        ) = await run_genesis_phase()
        self._accumulate_tokens("Genesis-Phase", usage)

        plot_source = self.plot_outline.get("source", "unknown")
        logger.info(
            f"   Plot Outline and Characters initialized/loaded (source: {plot_source}). "
            f"Title: '{self.plot_outline.get('title', 'N/A')}'. "
            f"Plot Points: {len(self.plot_outline.get('plot_points', []))}"
        )
        world_source = world_building.get("source", "unknown")
        logger.info(f"   World Building initialized/loaded (source: {world_source}).")
        self._update_rich_display(step="Genesis State Bootstrapped")

        self._update_novel_props_cache()
        logger.info("   Initial plot, character, and world data saved to Neo4j.")
        self._update_rich_display(step="Initial State Saved")

        return True

    async def _prepopulate_kg_if_needed(self):
        self._update_rich_display(step="Pre-populating KG (if needed)")
        logger.info("NANA: Checking if KG pre-population is needed...")

        plot_source = self.plot_outline.get("source", "")
        logger.info(
            f"\n--- NANA: Pre-populating Knowledge Graph from Initial Data (Plot Source: '{plot_source}') ---"
        )

        profile_objs: Dict[
            str, CharacterProfile
        ] = await character_queries.get_character_profiles_from_db()
        world_objs: Dict[
            str, Dict[str, WorldItem]
        ] = await world_queries.get_world_building_from_db()

        await self.kg_maintainer_agent.persist_profiles(
            profile_objs, config.KG_PREPOPULATION_CHAPTER_NUM
        )
        await self.kg_maintainer_agent.persist_world(
            world_objs, config.KG_PREPOPULATION_CHAPTER_NUM
        )
        logger.info("   Knowledge Graph pre-population step complete.")
        self._update_rich_display(step="KG Pre-population Complete")

    def _get_plot_point_info_for_chapter(
        self, novel_chapter_number: int
    ) -> Tuple[Optional[str], int]:
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
        self, chapter_number: int, final_text: str, raw_llm_log: Optional[str]
    ):
        loop = asyncio.get_event_loop()
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
        except IOError as e:
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
        loop = asyncio.get_event_loop()
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
    ) -> Tuple[str, int]:
        logger.info(f"NANA: Performing de-duplication for Chapter {chapter_number}...")
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
                    f"De-duplication for Chapter {chapter_number} removed {chars_removed} characters using {method} matching."
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
        patched_spans: List[Tuple[int, int]],
    ) -> Tuple[
        EvaluationResult,
        List[ProblemDetail],
        Optional[Dict[str, int]],
        Optional[Dict[str, int]],
    ]:
        self._update_rich_display(
            step=f"Ch {novel_chapter_number} - Evaluation Cycle {attempt} (Parallel)"
        )

        tasks_to_run = []
        task_names = []

        character_names = await character_queries.get_all_character_names()
        world_item_ids_by_category = (
            await world_queries.get_all_world_item_ids_by_category()
        )

        if config.ENABLE_COMPREHENSIVE_EVALUATION:
            tasks_to_run.append(
                self.evaluator_agent.evaluate_chapter_draft(
                    self.plot_outline,
                    character_names,
                    world_item_ids_by_category,
                    current_text,
                    novel_chapter_number,
                    plot_point_focus,
                    plot_point_index,
                    hybrid_context_for_draft,
                    ignore_spans=patched_spans,
                )
            )
            task_names.append("evaluation")

        if config.ENABLE_WORLD_CONTINUITY_CHECK:
            tasks_to_run.append(
                self.world_continuity_agent.check_consistency(
                    self.plot_outline,
                    current_text,
                    novel_chapter_number,
                    hybrid_context_for_draft,
                    ignore_spans=patched_spans,
                )
            )
            task_names.append("continuity")

        results = await asyncio.gather(*tasks_to_run)

        eval_result_obj = None
        eval_usage = None
        continuity_problems: List[ProblemDetail] = []
        continuity_usage = None

        result_idx = 0
        if "evaluation" in task_names:
            eval_result_obj, eval_usage = results[result_idx]
            result_idx += 1
        if "continuity" in task_names:
            continuity_problems, continuity_usage = results[result_idx]

        if eval_result_obj is None:
            eval_result_obj = {
                "needs_revision": False,
                "reasons": [],
                "problems_found": [],
                "coherence_score": None,
                "consistency_issues": None,
                "plot_deviation_reason": None,
                "thematic_issues": None,
                "narrative_depth_issues": None,
            }

        return eval_result_obj, continuity_problems, eval_usage, continuity_usage

    async def _perform_revisions(
        self,
        novel_chapter_number: int,
        attempt: int,
        current_text: str,
        evaluation_result: EvaluationResult,
        hybrid_context_for_draft: str,
        chapter_plan: Optional[List[SceneDetail]],
        patched_spans: List[Tuple[int, int]],
        is_from_flawed_source_for_kg: bool,
    ) -> Tuple[
        Optional[str], Optional[str], List[Tuple[int, int]], Optional[Dict[str, int]]
    ]:
        if attempt >= config.MAX_REVISION_CYCLES_PER_CHAPTER:
            logger.error(
                f"NANA: Ch {novel_chapter_number} - Max revision attempts ({config.MAX_REVISION_CYCLES_PER_CHAPTER}) reached."
            )
            return current_text, None, patched_spans, None

        revision_tuple_result, revision_usage = await revise_chapter_draft_logic(
            self.plot_outline,
            await character_queries.get_character_profiles_from_db(),
            await world_queries.get_world_building_from_db(),
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
            return new_text, rev_raw_output, patched_spans, revision_usage

        logger.error(
            f"NANA: Ch {novel_chapter_number} - Revision attempt {attempt} failed to produce usable text."
        )
        return current_text, None, patched_spans, revision_usage

    async def _prepare_chapter_prerequisites(
        self, novel_chapter_number: int
    ) -> Tuple[Optional[str], int, Optional[List[SceneDetail]], Optional[str]]:
        """Gather planning and context needed before drafting a chapter."""
        self._update_rich_display(
            step=f"Ch {novel_chapter_number} - Preparing Prerequisites"
        )

        plot_point_focus, plot_point_index = self._get_plot_point_info_for_chapter(
            novel_chapter_number
        )
        if plot_point_focus is None:
            logger.error(
                f"NANA: Ch {novel_chapter_number} prerequisite check failed: no concrete plot point focus (index {plot_point_index})."
            )
            return None, -1, None, None

        self._update_novel_props_cache()

        chapter_plan_result, plan_usage = await self.planner_agent.plan_chapter_scenes(
            self.plot_outline,
            await character_queries.get_character_profiles_from_db(),
            await world_queries.get_world_building_from_db(),
            novel_chapter_number,
            plot_point_focus,
            plot_point_index,
        )
        self._accumulate_tokens(f"Ch{novel_chapter_number}-Planning", plan_usage)

        chapter_plan: Optional[List[SceneDetail]] = chapter_plan_result

        hybrid_context_for_draft = await generate_hybrid_chapter_context_logic(
            self, novel_chapter_number, chapter_plan
        )

        if config.ENABLE_AGENTIC_PLANNING and chapter_plan is None:
            logger.warning(
                f"NANA: Ch {novel_chapter_number}: Planning Agent failed or plan invalid. Proceeding with plot point focus only."
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
        chapter_plan: Optional[List[SceneDetail]],
    ) -> Tuple[Optional[str], Optional[str]]:
        self._update_rich_display(
            step=f"Ch {novel_chapter_number} - Drafting Initial Text"
        )
        (
            initial_draft_text,
            initial_raw_llm_text,
            draft_usage,
        ) = await self.drafting_agent.draft_chapter(
            self.plot_outline,
            novel_chapter_number,
            plot_point_focus,
            hybrid_context_for_draft,
            chapter_plan,
        )
        self._accumulate_tokens(f"Ch{novel_chapter_number}-Drafting", draft_usage)

        if not initial_draft_text:
            logger.error(
                f"NANA: Drafting Agent failed for Ch {novel_chapter_number}. No initial draft produced."
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
        initial_raw_llm_text: Optional[str],
        plot_point_focus: str,
        plot_point_index: int,
        hybrid_context_for_draft: str,
        chapter_plan: Optional[List[SceneDetail]],
    ) -> Tuple[Optional[str], Optional[str], bool]:
        # FAST PATH: If all evaluation agents are disabled, just de-duplicate and return.
        if (
            not config.ENABLE_COMPREHENSIVE_EVALUATION
            and not config.ENABLE_WORLD_CONTINUITY_CHECK
        ):
            logger.info(
                f"NANA: Ch {novel_chapter_number} - All evaluation agents disabled. Applying de-duplication and finalizing draft."
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
                    f"NANA: Ch {novel_chapter_number} - Text marked as flawed for KG due to de-duplication removing {removed_char_count} characters."
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

        current_text_to_process: Optional[str] = initial_draft_text
        current_raw_llm_output: Optional[str] = initial_raw_llm_text
        is_from_flawed_source_for_kg = False
        patched_spans: List[Tuple[int, int]] = []

        # The de-duplication step is a single, definitive cleaning step after drafting and before evaluation.
        self._update_rich_display(
            step=f"Ch {novel_chapter_number} - Post-Draft De-duplication"
        )
        logger.info(
            f"NANA: Ch {novel_chapter_number} - Applying post-draft de-duplication."
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
                f"NANA: Ch {novel_chapter_number} - De-duplication removed {removed_char_count} characters. Text marked as potentially flawed for KG."
            )
            current_text_to_process = deduplicated_text
            await self._save_debug_output(
                novel_chapter_number,
                "deduplicated_text_after_draft",
                current_text_to_process,
            )
        else:
            logger.info(
                f"NANA: Ch {novel_chapter_number} - Post-draft de-duplication found no significant changes."
            )

        revisions_made = 0
        needs_revision = True
        while (
            needs_revision and revisions_made < config.MAX_REVISION_CYCLES_PER_CHAPTER
        ):
            attempt = revisions_made + 1
            if current_text_to_process is None:
                logger.error(
                    f"NANA: Ch {novel_chapter_number} - Text became None before processing cycle {attempt}. Aborting chapter."
                )
                return None, None, True
            (
                eval_result_obj,
                continuity_problems,
                eval_usage,
                continuity_usage,
            ) = await self._run_evaluation_cycle(
                novel_chapter_number,
                attempt,
                current_text_to_process,
                plot_point_focus,
                plot_point_index,
                hybrid_context_for_draft,
                patched_spans,
            )

            self._accumulate_tokens(
                f"Ch{novel_chapter_number}-Evaluation-Attempt{attempt}",
                eval_usage,
            )
            self._accumulate_tokens(
                f"Ch{novel_chapter_number}-ContinuityCheck-Attempt{attempt}",
                continuity_usage,
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
                    f"NANA: Ch {novel_chapter_number} (Attempt {attempt}) - World Continuity Agent found {len(continuity_problems)} issues."
                )
                evaluation_result["problems_found"].extend(continuity_problems)
                if not evaluation_result["needs_revision"]:
                    evaluation_result["needs_revision"] = True
                unique_reasons = set(evaluation_result.get("reasons", []))
                unique_reasons.add(
                    "Continuity issues identified by WorldContinuityAgent."
                )
                evaluation_result["reasons"] = sorted(list(unique_reasons))

            needs_revision = evaluation_result["needs_revision"]
            if not needs_revision:
                logger.info(
                    f"NANA: Ch {novel_chapter_number} draft passed evaluation (Attempt {attempt}). Text is considered good."
                )
                self._update_rich_display(
                    step=f"Ch {novel_chapter_number} - Passed Evaluation"
                )
                break
            else:
                is_from_flawed_source_for_kg = True
                logger.warning(
                    f"NANA: Ch {novel_chapter_number} draft (Attempt {attempt}) needs revision. Reasons: {'; '.join(evaluation_result.get('reasons', []))}"
                )
                self._update_rich_display(
                    step=f"Ch {novel_chapter_number} - Revision Attempt {attempt}"
                )
                (
                    new_text,
                    rev_raw_output,
                    patched_spans,
                    revision_usage,
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
                self._accumulate_tokens(
                    f"Ch{novel_chapter_number}-Revision-Attempt{attempt}",
                    revision_usage,
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
                                f"NANA: Ch {novel_chapter_number} revision attempt {attempt} produced text too similar to previous (score: {similarity:.4f}). Stopping revisions."
                            )
                            current_text_to_process = new_text
                            current_raw_llm_output = (
                                rev_raw_output or current_raw_llm_output
                            )
                            break
                    current_text_to_process = new_text
                    current_raw_llm_output = rev_raw_output or current_raw_llm_output
                    logger.info(
                        f"NANA: Ch {novel_chapter_number} - Revision attempt {attempt} successful. New text length: {len(current_text_to_process)}. Re-evaluating."
                    )
                    await self._save_debug_output(
                        novel_chapter_number,
                        f"revised_text_attempt_{attempt}",
                        current_text_to_process,
                    )
                    revisions_made += 1
                else:
                    logger.error(
                        f"NANA: Ch {novel_chapter_number} - Revision attempt {attempt} failed to produce usable text. Proceeding with previous draft, marked as flawed."
                    )
                    self._update_rich_display(
                        step=f"Ch {novel_chapter_number} - Revision Failed (Flawed)"
                    )
                    break
        if current_text_to_process is None:
            logger.critical(
                f"NANA: Ch {novel_chapter_number} - current_text_to_process is None after revision loop. Aborting chapter."
            )
            return None, None, True

        dedup_text_after_rev, removed_after_rev = await self.perform_deduplication(
            current_text_to_process, novel_chapter_number
        )
        if removed_after_rev > 0:
            logger.info(
                f"NANA: Ch {novel_chapter_number} - De-duplication after revisions removed {removed_after_rev} characters."
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
                f"NANA: Final chosen text for Ch {novel_chapter_number} is short ({len(current_text_to_process)} chars). Marked as flawed for KG."
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
        final_raw_llm_output: Optional[str],
        is_from_flawed_source_for_kg: bool,
    ) -> Optional[str]:
        self._update_rich_display(step=f"Ch {novel_chapter_number} - Finalization")

        result = await self.finalize_agent.finalize_chapter(
            self.plot_outline,
            await character_queries.get_character_profiles_from_db(),
            await world_queries.get_world_building_from_db(),
            novel_chapter_number,
            final_text_to_process,
            final_raw_llm_output,
            is_from_flawed_source_for_kg,
        )

        self._accumulate_tokens(
            f"Ch{novel_chapter_number}-Summarization", result.get("summary_usage")
        )
        self._accumulate_tokens(
            f"Ch{novel_chapter_number}-KGExtractionMerge", result.get("kg_usage")
        )
        await self._save_debug_output(
            novel_chapter_number, "final_summary", result.get("summary")
        )

        if result.get("embedding") is None:
            logger.error(
                "NANA CRITICAL: Failed to generate embedding for final text of Chapter %s. Text saved to file system only.",
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
                f"NANA: Cannot write Ch {novel_chapter_number}: Plot outline or critical plot data missing."
            )
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} Failed - Missing Plot Outline"
            )
            return False
        return True

    async def _process_prereq_result(
        self,
        novel_chapter_number: int,
        prereq_result: Tuple[
            Optional[str], int, Optional[List[SceneDetail]], Optional[str]
        ],
    ) -> Optional[Tuple[str, int, Optional[List[SceneDetail]], str]]:
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
        draft_result: Tuple[Optional[str], Optional[str]],
    ) -> Optional[Tuple[str, Optional[str]]]:
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
        revision_result: Tuple[Optional[str], Optional[str], bool],
    ) -> Optional[Tuple[str, Optional[str], bool]]:
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
        processed_raw_llm: Optional[str],
        is_flawed: bool,
    ) -> Optional[str]:
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
                f"=== NANA: Finished Novel Chapter {novel_chapter_number} - {status_message} ==="
            )
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} - {status_message}"
            )
        else:
            logger.error(
                f"=== NANA: Failed Novel Chapter {novel_chapter_number} - Finalization/Save Error ==="
            )
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} Failed - Finalization Error"
            )
        return final_text_result

    async def run_chapter_generation_process(
        self, novel_chapter_number: int
    ) -> Optional[str]:
        return await run_chapter_pipeline(self, novel_chapter_number)

    def _validate_critical_configs(self) -> bool:
        critical_str_configs = {
            "OLLAMA_EMBED_URL": config.OLLAMA_EMBED_URL,
            "OPENAI_API_BASE": config.OPENAI_API_BASE,
            "EMBEDDING_MODEL": config.EMBEDDING_MODEL,
            "NEO4J_URI": config.NEO4J_URI,
            "LARGE_MODEL": config.Models.LARGE,
            "MEDIUM_MODEL": config.Models.MEDIUM,
            "SMALL_MODEL": config.Models.SMALL,
            "NARRATOR_MODEL": config.Models.NARRATOR,
        }
        missing_or_empty_configs = []
        for name, value in critical_str_configs.items():
            if not value or not isinstance(value, str) or not value.strip():
                missing_or_empty_configs.append(name)

        if missing_or_empty_configs:
            logger.critical(
                f"NANA CRITICAL CONFIGURATION ERROR: The following critical configuration(s) are missing or empty: {', '.join(missing_or_empty_configs)}. Please set them (e.g., in .env file or environment variables) and restart."
            )
            return False

        if config.EXPECTED_EMBEDDING_DIM <= 0:
            logger.critical(
                f"NANA CRITICAL CONFIGURATION ERROR: EXPECTED_EMBEDDING_DIM must be a positive integer, but is {config.EXPECTED_EMBEDDING_DIM}."
            )
            return False

        logger.info("Critical configurations validated successfully.")
        return True

    async def run_novel_generation_loop(self):
        logger.info("--- NANA: Starting Novel Generation Run ---")

        if not self._validate_critical_configs():
            self._update_rich_display(step="Critical Config Error - Halting")
            await self.display.stop()
            return

        self.total_tokens_generated_this_run = 0
        self.run_start_time = time.time()
        self.display.start()
        try:
            await neo4j_manager.connect()
            await neo4j_manager.create_db_schema()
            logger.info("NANA: Neo4j connection and schema verified.")

            await self.kg_maintainer_agent.load_schema_from_db()
            logger.info("NANA: KG schema loaded into maintainer agent.")

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
                    "NANA: Core plot data missing or insufficient (e.g., no title, no concrete plot points). Performing initial setup..."
                )
                if not await self.perform_initial_setup():
                    logger.critical("NANA: Initial setup failed. Halting generation.")
                    self._update_rich_display(step="Initial Setup Failed - Halting")
                    return
                self._update_novel_props_cache()

            # KG pre-population handled within run_genesis_phase

            logger.info("\n--- NANA: Starting Novel Writing Process ---")

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
                f"NANA: Current Novel Chapter Count (State): {self.chapter_count}"
            )
            logger.info(
                f"NANA: Total Concrete Plot Points in Outline: {total_concrete_plot_points}"
            )
            logger.info(
                f"NANA: Remaining Concrete Plot Points to Cover in Novel: {remaining_plot_points_to_address_in_novel}"
            )

            if remaining_plot_points_to_address_in_novel <= 0:
                logger.info(
                    f"NANA: All {total_concrete_plot_points} concrete plot points appear to be covered by existing {self.chapter_count} chapters. No new chapters to generate."
                )
                self._update_rich_display(
                    chapter_num=self.chapter_count,
                    step="All Plot Points Covered",
                )
            else:
                logger.info(
                    f"NANA: Starting dynamic chapter loop (max {config.CHAPTERS_PER_RUN} chapter(s) this run)."
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
                        logger.info(
                            "NANA: No remaining plot points to cover. Ending run early."
                        )
                        break

                    current_novel_chapter_number = self.chapter_count + 1

                    logger.info(
                        f"\n--- NANA: Attempting Novel Chapter {current_novel_chapter_number} (attempt {attempts_this_run + 1}/{config.CHAPTERS_PER_RUN}) ---"
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
                                f"NANA: Novel Chapter {current_novel_chapter_number}: Processed. Final text length: {len(chapter_text_result)} chars."
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
                                    f"\n--- NANA: Triggering KG Healing/Enrichment after Chapter {current_novel_chapter_number} ---"
                                )
                                self._update_rich_display(
                                    step=f"Ch {current_novel_chapter_number} - KG Maintenance"
                                )
                                await self.kg_maintainer_agent.heal_and_enrich_kg()
                                await self.refresh_plot_outline()
                                logger.info(
                                    "--- NANA: KG Healing/Enrichment cycle complete. ---"
                                )

                        else:
                            logger.error(
                                f"NANA: Novel Chapter {current_novel_chapter_number}: Failed to generate or save. Halting run."
                            )
                            self._update_rich_display(
                                step=f"Ch {current_novel_chapter_number} Failed - Halting Run"
                            )
                            break
                    except Exception as e:
                        logger.critical(
                            f"NANA: Critical unhandled error during Novel Chapter {current_novel_chapter_number} writing process: {e}",
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
                logger.info(
                    "\n--- NANA: Novel writing process finished for this run ---"
                )
                logger.info(
                    f"NANA: Successfully processed {chapters_successfully_written_this_run} chapter(s) in this run."
                )
                logger.info(
                    f"NANA: Current total chapters in database after this run: {final_chapter_count_from_db}"
                )

            logger.info(
                f"NANA: Total LLM tokens generated this run: {self.total_tokens_generated_this_run}"
            )
            self._update_rich_display(
                chapter_num=self.chapter_count, step="Run Finished"
            )

        except Exception as e:
            logger.critical(
                f"NANA: Unhandled exception in orchestrator main loop: {e}",
                exc_info=True,
            )
            self._update_rich_display(step="Critical Error in Main Loop")
        finally:
            await self.display.stop()
            await neo4j_manager.close()
            logger.info("NANA: Neo4j driver successfully closed on application exit.")

    async def run_ingestion_process(self, text_file: str) -> None:
        """Ingest existing text and populate the knowledge graph."""
        logger.info("--- NANA: Starting Ingestion Process ---")

        if not self._validate_critical_configs():
            await self.display.stop()
            return

        self.display.start()
        self.run_start_time = time.time()
        await neo4j_manager.connect()
        await neo4j_manager.create_db_schema()
        await self.kg_maintainer_agent.load_schema_from_db()

        with open(text_file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        chunks = split_text_into_chapters(raw_text)
        plot_outline = {"title": "Ingested Narrative", "plot_points": []}
        character_profiles: Dict[str, CharacterProfile] = {}
        world_building: Dict[str, Dict[str, WorldItem]] = {}
        summaries: List[str] = []

        for idx, chunk in enumerate(chunks, 1):
            self._update_rich_display(chapter_num=idx, step="Ingesting Text")
            result = await self.finalize_agent.ingest_and_finalize_chunk(
                plot_outline,
                character_profiles,
                world_building,
                idx,
                chunk,
            )
            if result.get("summary"):
                summaries.append(str(result["summary"]))
                plot_outline["plot_points"].append(result["summary"])

            if idx % config.KG_HEALING_INTERVAL == 0:
                logger.info(
                    f"--- NANA: Triggering KG Healing/Enrichment after Ingestion Chunk {idx} ---"
                )
                self._update_rich_display(step=f"Ch {idx} - KG Maintenance")
                await self.kg_maintainer_agent.heal_and_enrich_kg()
                await self.refresh_plot_outline()

        await self.kg_maintainer_agent.heal_and_enrich_kg()
        combined_summary = "\n".join(summaries)
        continuation, _ = await self.planner_agent.plan_continuation(combined_summary)
        if continuation:
            plot_outline["plot_points"].extend(continuation)
        self.plot_outline = plot_outline
        self.chapter_count = len(chunks)
        await plot_queries.save_plot_outline_to_db(plot_outline)
        await self.display.stop()
        await neo4j_manager.close()
        logger.info("NANA: Ingestion process completed.")


def setup_logging_nana():
    logging.basicConfig(
        level=config.LOG_LEVEL_STR,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
        handlers=[],
    )
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if config.LOG_FILE:
        try:
            log_dir = os.path.dirname(config.LOG_FILE)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                config.LOG_FILE,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                mode="a",
                encoding="utf-8",
            )
            file_handler.setLevel(config.LOG_LEVEL_STR)
            formatter = logging.Formatter(
                config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            root_logger.info(f"File logging enabled. Log file: {config.LOG_FILE}")
        except Exception as e:
            console_handler_fallback = logging.StreamHandler()
            console_handler_fallback.setFormatter(
                logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
            )
            root_logger.addHandler(console_handler_fallback)
            root_logger.error(
                f"Failed to configure file logging to {config.LOG_FILE}: {e}. Logging to console instead.",
                exc_info=True,
            )

    if RICH_AVAILABLE and config.ENABLE_RICH_PROGRESS:
        existing_console = None
        if root_logger.handlers:
            for h_idx, h in enumerate(root_logger.handlers):
                if hasattr(h, "console") and not isinstance(h, logging.FileHandler):
                    existing_console = h.console  # type: ignore
                    break

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
    elif not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(config.LOG_LEVEL_STR)
        stream_formatter = logging.Formatter(
            config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT
        )
        stream_handler.setFormatter(stream_formatter)
        root_logger.addHandler(stream_handler)
        root_logger.info("Standard stream logging handler enabled for console.")

    logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    root_logger.info(
        f"NANA Logging setup complete. Application Log Level: {logging.getLevelName(config.LOG_LEVEL_STR)}."
    )
