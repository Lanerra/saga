# nana_orchestrator.py
import asyncio
import importlib
import time  # For Rich display updates
from dataclasses import dataclass, field
from typing import Any

import structlog
import utils
from agents.comprehensive_evaluator_agent import ComprehensiveEvaluatorAgent
from agents.drafting_agent import DraftingAgent
from agents.finalize_agent import FinalizationResult, FinalizeAgent
from agents.kg_maintainer_agent import KGMaintainerAgent
from agents.planner_agent import PlannerAgent
from agents.pre_flight_check_agent import PreFlightCheckAgent
from chapter_generation import (
    ContextOrchestrator,
    DraftResult,
    PrerequisiteData,
)
from config import settings
from core.db_manager import neo4j_manager
from core.llm_interface import llm_service
from core.usage import TokenUsage
from data_access import (
    chapter_queries,
    character_queries,
    plot_queries,
    world_queries,
)
from ingestion.ingestion_manager import IngestionManager
from initialization.data_loader import convert_model_to_objects
from initialization.genesis import run_genesis_phase
from initialization.models import PlotOutline
from kg_maintainer.models import (
    CharacterProfile,
    EvaluationResult,
    ProblemDetail,
    SceneDetail,
    WorldItem,
)
from processing.repetition_analyzer import RepetitionAnalyzer
from processing.repetition_tracker import RepetitionTracker
from processing.revision_manager import RevisionManager
from processing.text_deduplicator import TextDeduplicator
from storage.file_manager import FileManager
from ui.rich_display import RichDisplayManager
from utils.plot import get_plot_point_info

from models.agent_models import ChapterEndState
from models.user_input_models import UserStoryInputModel
from orchestration.chapter_flow import run_chapter_pipeline
from orchestration.chapter_generation_runner import ChapterGenerationRunner
from orchestration.token_accountant import Stage, TokenAccountant

logger = structlog.get_logger(__name__)


@dataclass
class RevisionOutcome:
    """Final text after processing and whether it is marked flawed."""

    text: str | None
    raw_llm_output: str | None
    is_flawed: bool


@dataclass
class KnowledgeCache:
    """In-memory cache for KG data used during chapter generation."""

    characters: dict[str, CharacterProfile] = field(default_factory=dict)
    world: dict[str, dict[str, WorldItem]] = field(default_factory=dict)


class NANA_Orchestrator:
    def __init__(self, file_manager: FileManager | None = None):
        logger.info("Initializing NANA Orchestrator...")
        self.file_manager = file_manager or FileManager()
        self.planner_agent = PlannerAgent()
        self.drafting_agent = DraftingAgent()
        self.evaluator_agent = ComprehensiveEvaluatorAgent()
        self.kg_maintainer_agent = KGMaintainerAgent()
        self.pre_flight_agent = PreFlightCheckAgent()
        self.finalize_agent = FinalizeAgent(self.kg_maintainer_agent)
        self.revision_manager = RevisionManager()
        self.repetition_tracker = RepetitionTracker()
        self.repetition_analyzer = RepetitionAnalyzer(tracker=self.repetition_tracker)

        provider_instances = []
        for dotted in settings.CONTEXT_PROVIDERS:
            module_name, class_name = dotted.rsplit(".", 1)
            module = importlib.import_module(module_name)
            provider_cls = getattr(module, class_name)
            provider_instances.append(provider_cls())
        self.context_service = ContextOrchestrator(provider_instances)

        self.plot_outline: PlotOutline = PlotOutline()
        self.chapter_count: int = 0
        self.novel_props_cache: dict[str, Any] = {}
        self.knowledge_cache = KnowledgeCache()
        self.token_accountant = TokenAccountant()
        self.total_tokens_generated_this_run: int = 0

        self.next_chapter_context: str | None = None
        self.chapter_zero_end_state: ChapterEndState | None = None

        self.display = RichDisplayManager()
        self.run_start_time: float = 0.0
        utils.load_spacy_model_if_needed()
        logger.info("NANA Orchestrator initialized.")

    def _update_rich_display(
        self, chapter_num: int | None = None, step: str | None = None
    ) -> None:
        self.display.update(
            plot_outline=self.plot_outline,
            chapter_num=chapter_num,
            step=step,
            total_tokens=self.total_tokens_generated_this_run,
            run_start_time=self.run_start_time,
        )

    def _accumulate_tokens(
        self, stage: str, usage_data: dict[str, int] | TokenUsage | None
    ) -> None:
        self.token_accountant.record_usage(stage, usage_data)
        self.total_tokens_generated_this_run = self.token_accountant.total
        self._update_rich_display()

    async def _generate_plot_points_from_kg(self, count: int) -> None:
        """Generate and persist additional plot points using the planner agent."""
        if count <= 0:
            return

        summaries: list[str] = []
        start = max(1, self.chapter_count - settings.CONTEXT_CHAPTER_COUNT + 1)
        for i in range(start, self.chapter_count + 1):
            chap = await chapter_queries.get_chapter_data_from_db(i)
            if chap and (chap.get("summary") or chap.get("text")):
                summaries.append((chap.get("summary") or chap.get("text", "")).strip())

        combined_summary = "\n".join(summaries)
        if not combined_summary.strip():
            logger.warning("No summaries available for continuation planning.")
            return

        new_points, usage = await self.planner_agent.plan_continuation(
            combined_summary, count
        )
        self._accumulate_tokens(Stage.PLAN_CONTINUATION.value, usage)
        if not new_points:
            logger.error("Failed to generate continuation plot points.")
            return

        for desc in new_points:
            if await plot_queries.plot_point_exists(desc):
                logger.info("Plot point already exists, skipping: %s", desc)
                continue
            prev_id = await plot_queries.get_last_plot_point_id()
            await self.kg_maintainer_agent.add_plot_point(desc, prev_id or "")
            self.plot_outline.setdefault("plot_points", []).append(desc)
        self._update_novel_props_cache()

    def load_state_from_user_model(self, model: UserStoryInputModel) -> None:
        """Populate orchestrator state from a user-provided model."""
        plot_outline, _, _ = convert_model_to_objects(model)
        self.plot_outline = plot_outline

    def _update_novel_props_cache(self):
        self.novel_props_cache = {
            "title": self.plot_outline.get(
                "title", settings.DEFAULT_PLOT_OUTLINE_TITLE
            ),
            "genre": self.plot_outline.get("genre", settings.CONFIGURED_GENRE),
            "theme": self.plot_outline.get("theme", settings.CONFIGURED_THEME),
            "protagonist_name": self.plot_outline.get(
                "protagonist_name", settings.DEFAULT_PROTAGONIST_NAME
            ),
            "character_arc": self.plot_outline.get("character_arc", "N/A"),
            "logline": self.plot_outline.get("logline", "N/A"),
            "setting": self.plot_outline.get(
                "setting", settings.CONFIGURED_SETTING_DESCRIPTION
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
            self.plot_outline = PlotOutline(**result)
            self._update_novel_props_cache()
        else:
            logger.error("Failed to refresh plot outline from DB: %s", result)

    async def refresh_knowledge_cache(self) -> None:
        """Reload character profiles and world building into the cache."""
        logger.info("Refreshing knowledge cache from Neo4j...")
        self.knowledge_cache.characters = (
            await character_queries.get_character_profiles_from_db()
        )
        self.knowledge_cache.world = await world_queries.get_world_building_from_db()
        logger.info(
            "Knowledge cache refreshed: %d characters, %d world categories.",
            len(self.knowledge_cache.characters),
            len(self.knowledge_cache.world),
        )

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
            self.plot_outline = PlotOutline()
        else:
            self.plot_outline = (
                PlotOutline(**result) if isinstance(result, dict) else PlotOutline()
            )

        if not self.plot_outline.get("plot_points"):
            logger.warning(
                "Orchestrator init: Plot outline loaded from DB has no plot points. Initial setup might be needed or DB is empty/corrupt."
            )
        else:
            logger.info(
                f"Orchestrator init: Loaded {len(self.plot_outline.get('plot_points', []))} plot points from DB."
            )

        self._update_novel_props_cache()
        await self.refresh_knowledge_cache()
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
        self._accumulate_tokens(Stage.GENESIS_PHASE.value, usage)

        plot_source = self.plot_outline.get("source", "unknown")
        logger.info(
            f"   Plot Outline and Characters initialized/loaded (source: {plot_source}). "
            f"Title: '{self.plot_outline.get('title', 'N/A')}'. "
            f"Plot Points: {len(self.plot_outline.get('plot_points', []))}"
        )
        world_source = world_building.get("source", "unknown")
        logger.info(f"   World Building initialized/loaded (source: {world_source}).")
        self._update_rich_display(step="Genesis State Bootstrapped")

        self.knowledge_cache.characters = character_profiles
        self.knowledge_cache.world = world_building
        await self.refresh_knowledge_cache()
        self._update_novel_props_cache()
        logger.info("   Initial plot, character, and world data saved to Neo4j.")
        self._update_rich_display(step="Initial State Saved")

        await self.refresh_plot_outline()
        if neo4j_manager.driver is not None:
            await self.refresh_knowledge_cache()
        else:
            logger.warning(
                "Neo4j driver not initialized. Skipping knowledge cache refresh."
            )
        try:
            data = await chapter_queries.get_chapter_data_from_db(0)
            if data and data.get("end_state_json"):
                self.chapter_zero_end_state = ChapterEndState.model_validate_json(
                    data["end_state_json"]
                )
        except Exception as exc:
            logger.error("Failed to load chapter 0 end state: %s", exc, exc_info=True)
        self.next_chapter_context = await self.context_service.build_hybrid_context(
            self,
            1,
            None,
            {"chapter_zero_end_state": self.chapter_zero_end_state}
            if self.chapter_zero_end_state
            else None,
        )

        return True

    async def _save_chapter_text_and_log(
        self, chapter_number: int, final_text: str, raw_llm_log: str | None
    ):
        try:
            await self.file_manager.save_chapter_and_log(
                chapter_number, final_text, raw_llm_log or "N/A"
            )
            logger.info(
                f"Saved chapter text and raw LLM log files for ch {chapter_number}."
            )
        except OSError as e:
            logger.error(
                f"Failed writing chapter text/log files for ch {chapter_number}: {e}",
                exc_info=True,
            )

    async def _save_debug_output(
        self, chapter_number: int, stage_description: str, content: Any
    ):
        if content is None:
            return
        content_str = str(content) if not isinstance(content, str) else content
        if not content_str.strip():
            return
        try:
            await self.file_manager.save_debug_output(
                chapter_number, stage_description, content_str
            )
            logger.debug(
                f"Saved debug output for Ch {chapter_number}, Stage '{stage_description}' to file system"
            )
        except Exception as e:
            logger.error(
                f"Failed to save debug output (Ch {chapter_number}, Stage '{stage_description}'): {e}",
                exc_info=True,
            )

    async def perform_deduplication(
        self, text_to_dedup: str, chapter_number: int
    ) -> tuple[str, int]:
        logger.info(f"NANA: Performing de-duplication for Chapter {chapter_number}...")
        if not text_to_dedup or not text_to_dedup.strip():
            logger.info(
                f"De-duplication for Chapter {chapter_number}: Input text is empty. No action taken."
            )
            return text_to_dedup, 0
        try:
            deduper = TextDeduplicator(
                similarity_threshold=settings.DEDUPLICATION_SEMANTIC_THRESHOLD,
                use_semantic_comparison=settings.DEDUPLICATION_USE_SEMANTIC,
                min_segment_length_chars=settings.DEDUPLICATION_MIN_SEGMENT_LENGTH,
            )
            deduplicated_text, chars_removed = await deduper.deduplicate(
                text_to_dedup, segment_level="sentence"
            )
            if chars_removed > 0:
                method = (
                    "semantic"
                    if settings.DEDUPLICATION_USE_SEMANTIC
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
        patched_spans: list[tuple[int, int]],
    ) -> tuple[
        EvaluationResult,
        list[ProblemDetail],
        dict[str, int] | None,
        dict[str, int] | None,
        list[ProblemDetail],
    ]:
        self._update_rich_display(
            step=f"Ch {novel_chapter_number} - Evaluation Cycle {attempt} (Parallel)"
        )

        tasks_to_run: list[asyncio.Future] = []
        task_names: list[str] = []

        ignore_spans = patched_spans

        if settings.ENABLE_COMPREHENSIVE_EVALUATION:
            tasks_to_run.append(
                self.evaluator_agent.evaluate_chapter_draft(
                    self.plot_outline,
                    current_text,
                    novel_chapter_number,
                    plot_point_focus,
                    plot_point_index,
                    hybrid_context_for_draft,
                    ignore_spans=ignore_spans,
                )
            )
            task_names.append("evaluation")

        results = await asyncio.gather(*tasks_to_run)

        eval_result_obj: EvaluationResult | None = None
        eval_usage = None
        continuity_problems: list[ProblemDetail] = []
        continuity_usage = None

        result_idx = 0
        if "evaluation" in task_names:
            eval_result_obj, eval_usage = results[result_idx]

        if isinstance(eval_result_obj, EvaluationResult):
            evaluation_result: EvaluationResult = eval_result_obj
        else:
            data = eval_result_obj or {}
            evaluation_result = EvaluationResult(
                needs_revision=data.get("needs_revision", False),
                reasons=data.get("reasons", []),
                problems_found=data.get("problems_found", []),
                coherence_score=data.get("coherence_score"),
                consistency_issues=data.get("consistency_issues"),
                plot_deviation_reason=data.get("plot_deviation_reason"),
                thematic_issues=data.get("thematic_issues"),
                narrative_depth_issues=data.get("narrative_depth_issues"),
            )

        repetition_probs = await self.repetition_analyzer.analyze(current_text)
        evaluation_result.problems_found.extend(repetition_probs)
        if repetition_probs:
            evaluation_result.needs_revision = True
            evaluation_result.reasons.append("Repetition issues detected")

        return (
            evaluation_result,
            continuity_problems,
            eval_usage,
            continuity_usage,
            repetition_probs,
        )

    async def _handle_no_evaluation_fast_path(
        self,
        novel_chapter_number: int,
        initial_text: str,
        initial_raw_llm_text: str | None,
    ) -> tuple[str, str | None, bool] | None:
        """Return early with deduplicated text when evaluation is disabled."""
        if (
            not settings.ENABLE_COMPREHENSIVE_EVALUATION
            and not settings.ENABLE_WORLD_CONTINUITY_CHECK
        ):
            logger.info(
                f"NANA: Ch {novel_chapter_number} - All evaluation agents disabled. Applying de-duplication and finalizing draft."
            )
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} - Skipping Revisions (disabled)"
            )
            deduplicated_text, removed_char_count = await self.perform_deduplication(
                initial_text,
                novel_chapter_number,
            )
            is_flawed = removed_char_count > 0
            if is_flawed:
                logger.info(
                    f"NANA: Ch {novel_chapter_number} - Text marked as flawed for KG due to de-duplication removing {removed_char_count} characters."
                )
                await self._save_debug_output(
                    novel_chapter_number,
                    "deduplicated_text_no_eval_path",
                    deduplicated_text,
                )
            return deduplicated_text, initial_raw_llm_text, is_flawed
        return None

    async def _deduplicate_post_draft(
        self,
        novel_chapter_number: int,
        text: str | None,
    ) -> tuple[str | None, bool]:
        """Deduplicate text after drafting and log results."""
        self._update_rich_display(
            step=f"Ch {novel_chapter_number} - Post-Draft De-duplication"
        )
        logger.info(
            f"NANA: Ch {novel_chapter_number} - Applying post-draft de-duplication."
        )
        if text is None:
            return None, False
        deduped, removed = await self.perform_deduplication(text, novel_chapter_number)
        if removed > 0:
            await self._save_debug_output(
                novel_chapter_number,
                "deduplicated_text_after_draft",
                deduped,
            )
            logger.info(
                f"NANA: Ch {novel_chapter_number} - De-duplication removed {removed} characters. Text marked as potentially flawed for KG."
            )
            return deduped, True
        logger.info(
            f"NANA: Ch {novel_chapter_number} - Post-draft de-duplication found no significant changes."
        )
        return text, False

    async def _run_revision_loop(
        self,
        novel_chapter_number: int,
        current_text: str | None,
        current_raw_llm_output: str | None,
        plot_point_focus: str,
        plot_point_index: int,
        hybrid_context_for_draft: str,
        chapter_plan: list[SceneDetail] | None,
        patched_spans: list[tuple[int, int]],
        is_from_flawed_source_for_kg: bool,
    ) -> tuple[str | None, str | None, bool, list[tuple[int, int]]]:
        """Iteratively revise a chapter draft until it passes evaluation.

        De-duplicates text at the start of each cycle, performs evaluation,
        and applies patches or rewrites as needed.

        Args:
            novel_chapter_number: Sequential chapter number being processed.
            current_text: Text to revise.
            current_raw_llm_output: Raw LLM output for the current text.
            plot_point_focus: Plot point focus for the chapter.
            plot_point_index: Index of the plot point within the outline.
            hybrid_context_for_draft: Combined contextual information for prompts.
            chapter_plan: Optional plan for scenes in the chapter.
            patched_spans: Already patched ranges to avoid re-editing.
            is_from_flawed_source_for_kg: Whether the text is marked flawed.

        Returns:
            Tuple containing the revised text, raw output, flaw flag, and
            patched spans.
        """
        revisions_made = 0
        needs_revision = True
        last_eval_result: EvaluationResult | None = None
        while (
            needs_revision and revisions_made < settings.MAX_REVISION_CYCLES_PER_CHAPTER
        ):
            attempt = revisions_made + 1
            if current_text is None:
                logger.error(
                    "NANA: Ch %s - Text became None before processing cycle %s. Aborting chapter.",
                    novel_chapter_number,
                    attempt,
                )
                return None, None, True, patched_spans

            (
                current_text,
                removed_chars,
            ) = await self.perform_deduplication(current_text, novel_chapter_number)

            if removed_chars > 0:
                logger.info(
                    "Ch %s Rev-Loop: De-duplication removed %s chars before evaluation.",
                    novel_chapter_number,
                    removed_chars,
                )
                is_from_flawed_source_for_kg = True

            (
                eval_result_obj,
                continuity_problems,
                eval_usage,
                continuity_usage,
                repetition_problems,
            ) = await self._run_evaluation_cycle(
                novel_chapter_number,
                attempt,
                current_text,
                plot_point_focus,
                plot_point_index,
                hybrid_context_for_draft,
                patched_spans,
            )

            self._accumulate_tokens(
                f"Ch{novel_chapter_number}-{Stage.EVALUATION.value}-Attempt{attempt}",
                eval_usage,
            )
            self._accumulate_tokens(
                f"Ch{novel_chapter_number}-{Stage.CONTINUITY_CHECK.value}-Attempt{attempt}",
                continuity_usage,
            )

            if isinstance(eval_result_obj, EvaluationResult):
                evaluation_result: EvaluationResult = eval_result_obj
            else:
                evaluation_result = EvaluationResult(**eval_result_obj)
            last_eval_result = evaluation_result
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
            await self._save_debug_output(
                novel_chapter_number,
                f"repetition_problems_attempt_{attempt}",
                repetition_problems,
            )

            if continuity_problems:
                logger.warning(
                    "NANA: Ch %s (Attempt %s) - Consistency checker found %s issues.",
                    novel_chapter_number,
                    attempt,
                    len(continuity_problems),
                )
                evaluation_result.problems_found.extend(continuity_problems)
                if not evaluation_result.needs_revision:
                    evaluation_result.needs_revision = True
                unique_reasons = set(evaluation_result.reasons)
                unique_reasons.add("Continuity issues detected")
                evaluation_result.reasons = sorted(list(unique_reasons))

            if repetition_problems:
                evaluation_result.problems_found.extend(repetition_problems)
                if not evaluation_result.needs_revision:
                    evaluation_result.needs_revision = True
                if "Repetition issues detected" not in evaluation_result.reasons:
                    evaluation_result.reasons.append("Repetition issues detected")

            needs_revision = evaluation_result.needs_revision
            if not needs_revision:
                logger.info(
                    "NANA: Ch %s draft passed evaluation (Attempt %s). Text is considered good.",
                    novel_chapter_number,
                    attempt,
                )
                self._update_rich_display(
                    step=f"Ch {novel_chapter_number} - Passed Evaluation"
                )
                break

            is_from_flawed_source_for_kg = True
            logger.warning(
                "NANA: Ch %s draft (Attempt %s) needs revision. Reasons: %s",
                novel_chapter_number,
                attempt,
                "; ".join(evaluation_result.reasons),
            )
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} - Revision Attempt {attempt}"
            )
            (
                revision_result,
                revision_usage,
            ) = await self.revision_manager.revise_chapter(
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
                continuity_problems=continuity_problems,
            )
            self._accumulate_tokens(
                f"Ch{novel_chapter_number}-{Stage.REVISION.value}-Attempt{attempt}",
                revision_usage,
            )
            if (
                revision_result
                and revision_result[0]
                and len(revision_result[0]) > 50
                and len(revision_result[0]) >= len(current_text) * 0.5
            ):
                new_text, rev_raw_output, patched_spans = revision_result
                if new_text and new_text != current_text:
                    new_embedding, prev_embedding = await asyncio.gather(
                        llm_service.async_get_embedding(new_text),
                        llm_service.async_get_embedding(current_text),
                    )
                    if new_embedding is not None and prev_embedding is not None:
                        similarity = utils.numpy_cosine_similarity(
                            prev_embedding,
                            new_embedding,
                        )
                        if similarity > settings.REVISION_SIMILARITY_ACCEPTANCE:
                            logger.warning(
                                "NANA: Ch %s revision attempt %s produced text too similar to previous (score: %.4f). Stopping revisions.",
                                novel_chapter_number,
                                attempt,
                                similarity,
                            )
                            current_text = new_text
                            current_raw_llm_output = (
                                rev_raw_output or current_raw_llm_output
                            )
                            break
                    current_text = new_text
                    current_raw_llm_output = rev_raw_output or current_raw_llm_output
                    logger.info(
                        "NANA: Ch %s - Revision attempt %s successful. New text length: %s. Re-evaluating.",
                        novel_chapter_number,
                        attempt,
                        len(current_text),
                    )
                    await self._save_debug_output(
                        novel_chapter_number,
                        f"revised_text_attempt_{attempt}",
                        current_text,
                    )
                    revisions_made += 1
                else:
                    logger.error(
                        "NANA: Ch %s - Revision attempt %s failed to produce usable text. Proceeding with previous draft, marked as flawed.",
                        novel_chapter_number,
                        attempt,
                    )
                    self._update_rich_display(
                        step=f"Ch {novel_chapter_number} - Revision Failed (Retrying)"
                    )
                    revisions_made += 1
                    needs_revision = True
                    continue
            else:
                logger.error(
                    "NANA: Ch %s - Revision attempt %s failed to produce usable text.",
                    novel_chapter_number,
                    attempt,
                )
                self._update_rich_display(
                    step=f"Ch {novel_chapter_number} - Revision Failed (Retrying)"
                )
                revisions_made += 1
                needs_revision = True
                continue

        if needs_revision and last_eval_result is not None:
            root_cause = self.revision_manager.identify_root_cause(
                [p.model_dump() for p in last_eval_result.problems_found],
                self.plot_outline,
                await character_queries.get_character_profiles_from_db(),
                await world_queries.get_world_building_from_db(),
            )
            if root_cause:
                logger.warning(
                    "NANA: Ch %s - Root cause analysis: %s",
                    novel_chapter_number,
                    root_cause,
                )

        return (
            current_text,
            current_raw_llm_output,
            is_from_flawed_source_for_kg,
            patched_spans,
        )

    async def _deduplicate_post_revision(
        self,
        novel_chapter_number: int,
        text: str,
        is_flawed: bool,
    ) -> tuple[str, bool]:
        (
            dedup_text_after_rev,
            removed_after_rev,
        ) = await self.perform_deduplication(
            text,
            novel_chapter_number,
        )
        if removed_after_rev > 0:
            logger.info(
                "NANA: Ch %s - De-duplication after revisions removed %s characters.",
                novel_chapter_number,
                removed_after_rev,
            )
            text = dedup_text_after_rev
            is_flawed = True
            await self._save_debug_output(
                novel_chapter_number,
                "deduplicated_text_after_revision",
                text,
            )
        if len(text) < settings.MIN_ACCEPTABLE_DRAFT_LENGTH:
            logger.warning(
                "NANA: Final chosen text for Ch %s is short (%s chars). Marked as flawed for KG.",
                novel_chapter_number,
                len(text),
            )
            is_flawed = True
        return text, is_flawed

    async def _prepare_chapter_prerequisites(
        self, novel_chapter_number: int
    ) -> PrerequisiteData:
        """Gather planning and context needed before drafting a chapter."""
        self._update_rich_display(
            step=f"Ch {novel_chapter_number} - Preparing Prerequisites"
        )

        plot_point_focus, plot_point_index = get_plot_point_info(
            self.plot_outline, novel_chapter_number
        )
        if plot_point_focus is None:
            logger.error(
                f"NANA: Ch {novel_chapter_number} prerequisite check failed: no concrete plot point focus (index {plot_point_index})."
            )
            return PrerequisiteData(None, -1, None, None)

        self._update_novel_props_cache()

        await self.pre_flight_agent.perform_core_checks(
            self.plot_outline,
            self.knowledge_cache.characters,
            self.knowledge_cache.world,
        )

        planning_context = self.next_chapter_context
        if planning_context is None:
            planning_context = await self.context_service.build_hybrid_context(
                self,
                novel_chapter_number,
                None,
                None,
            )
        chapter_plan_result, plan_usage = await self.planner_agent.plan_chapter_scenes(
            self.plot_outline,
            novel_chapter_number,
            plot_point_focus,
            plot_point_index,
            (novel_chapter_number - 1) % settings.PLOT_POINT_CHAPTER_SPAN + 1,
            planning_context,
        )
        self._accumulate_tokens(
            f"Ch{novel_chapter_number}-{Stage.CHAPTER_PLANNING.value}", plan_usage
        )

        chapter_plan: list[SceneDetail] | None = chapter_plan_result

        if (
            settings.ENABLE_SCENE_PLAN_VALIDATION
            and chapter_plan is not None
            and settings.ENABLE_WORLD_CONTINUITY_CHECK
        ):
            (
                plan_problems,
                usage,
            ) = await self.evaluator_agent.check_scene_plan_consistency(
                self.plot_outline,
                chapter_plan,
                novel_chapter_number,
                planning_context,
            )
            self._accumulate_tokens(
                f"Ch{novel_chapter_number}-{Stage.PLAN_CONSISTENCY.value}", usage
            )
            await self._save_debug_output(
                novel_chapter_number,
                "scene_plan_consistency_problems",
                plan_problems,
            )
            if plan_problems:
                logger.warning(
                    f"NANA: Ch {novel_chapter_number} scene plan has {len(plan_problems)} consistency issues."
                )

        hybrid_context_for_draft = self.next_chapter_context
        if hybrid_context_for_draft is None:
            await self.refresh_plot_outline()
            if neo4j_manager.driver is not None:
                await self.refresh_knowledge_cache()
            else:
                logger.warning(
                    "Neo4j driver not initialized. Skipping knowledge cache refresh."
                )
            hybrid_context_for_draft = await self.context_service.build_hybrid_context(
                self,
                novel_chapter_number,
                chapter_plan,
                None,
            )
        else:
            self.next_chapter_context = None

        if settings.ENABLE_AGENTIC_PLANNING and chapter_plan is None:
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

        return PrerequisiteData(
            plot_point_focus=plot_point_focus,
            plot_point_index=plot_point_index,
            chapter_plan=chapter_plan,
            hybrid_context_for_draft=hybrid_context_for_draft,
        )

    async def _draft_initial_chapter_text(
        self,
        novel_chapter_number: int,
        plot_point_focus: str,
        hybrid_context_for_draft: str,
        chapter_plan: list[SceneDetail] | None,
    ) -> DraftResult:
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
        self._accumulate_tokens(
            f"Ch{novel_chapter_number}-{Stage.DRAFTING.value}", draft_usage
        )

        if not initial_draft_text:
            logger.error(
                f"NANA: Drafting Agent failed for Ch {novel_chapter_number}. No initial draft produced."
            )
            await self._save_debug_output(
                novel_chapter_number,
                "initial_draft_fail_raw_llm",
                initial_raw_llm_text or "Drafting Agent returned None for raw output.",
            )
            return DraftResult(text=None, raw_llm_output=None)

        await self._save_debug_output(
            novel_chapter_number, "initial_draft", initial_draft_text
        )
        return DraftResult(text=initial_draft_text, raw_llm_output=initial_raw_llm_text)

    async def _process_and_revise_draft(
        self,
        novel_chapter_number: int,
        initial_draft_text: str,
        initial_raw_llm_text: str | None,
        plot_point_focus: str,
        plot_point_index: int,
        hybrid_context_for_draft: str,
        chapter_plan: list[SceneDetail] | None,
    ) -> RevisionOutcome:
        fast_path_result = await self._handle_no_evaluation_fast_path(
            novel_chapter_number,
            initial_draft_text,
            initial_raw_llm_text,
        )
        if fast_path_result is not None:
            text, raw, flawed = fast_path_result
            return RevisionOutcome(text=text, raw_llm_output=raw, is_flawed=flawed)

        current_text_to_process: str | None = initial_draft_text
        current_raw_llm_output: str | None = initial_raw_llm_text
        is_from_flawed_source_for_kg = False
        patched_spans: list[tuple[int, int]] = []

        (
            current_text_to_process,
            flawed_after_dedup,
        ) = await self._deduplicate_post_draft(
            novel_chapter_number,
            current_text_to_process,
        )
        if flawed_after_dedup:
            is_from_flawed_source_for_kg = True

        (
            current_text_to_process,
            current_raw_llm_output,
            is_from_flawed_source_for_kg,
            patched_spans,
        ) = await self._run_revision_loop(
            novel_chapter_number,
            current_text_to_process,
            current_raw_llm_output,
            plot_point_focus,
            plot_point_index,
            hybrid_context_for_draft,
            chapter_plan,
            patched_spans,
            is_from_flawed_source_for_kg,
        )
        if current_text_to_process is None:
            logger.critical(
                f"NANA: Ch {novel_chapter_number} - current_text_to_process is None after revision loop. Aborting chapter."
            )
            return RevisionOutcome(text=None, raw_llm_output=None, is_flawed=True)

        (
            current_text_to_process,
            is_from_flawed_source_for_kg,
        ) = await self._deduplicate_post_revision(
            novel_chapter_number,
            current_text_to_process,
            is_from_flawed_source_for_kg,
        )

        return RevisionOutcome(
            text=current_text_to_process,
            raw_llm_output=current_raw_llm_output,
            is_flawed=is_from_flawed_source_for_kg,
        )

    async def _finalize_and_save_chapter(
        self,
        novel_chapter_number: int,
        final_text_to_process: str,
        final_raw_llm_output: str | None,
        is_from_flawed_source_for_kg: bool,
    ) -> str | None:
        self._update_rich_display(step=f"Ch {novel_chapter_number} - Finalization")

        result: FinalizationResult = await self.finalize_agent.finalize_chapter(
            self.plot_outline,
            await character_queries.get_character_profiles_from_db(),
            await world_queries.get_world_building_from_db(),
            novel_chapter_number,
            final_text_to_process,
            final_raw_llm_output,
            is_from_flawed_source_for_kg,
        )

        self._accumulate_tokens(
            f"Ch{novel_chapter_number}-{Stage.SUMMARIZATION.value}",
            result.get("summary_usage"),
        )
        self._accumulate_tokens(
            f"Ch{novel_chapter_number}-{Stage.KG_EXTRACTION_MERGE.value}",
            result.get("kg_usage"),
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

        self.repetition_tracker.update_from_text(final_text_to_process)

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
        prereq_result: PrerequisiteData,
    ) -> PrerequisiteData | None:
        plot_point_focus = prereq_result.plot_point_focus
        plot_point_index = prereq_result.plot_point_index
        chapter_plan = prereq_result.chapter_plan
        hybrid_context_for_draft = prereq_result.hybrid_context_for_draft

        if plot_point_focus is None or hybrid_context_for_draft is None:
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} Failed - Prerequisites Incomplete"
            )
            return None
        return PrerequisiteData(
            plot_point_focus=plot_point_focus,
            plot_point_index=plot_point_index,
            chapter_plan=chapter_plan,
            hybrid_context_for_draft=hybrid_context_for_draft,
        )

    async def _process_initial_draft(
        self,
        novel_chapter_number: int,
        draft_result: DraftResult,
    ) -> DraftResult | None:
        initial_draft_text = draft_result.text
        initial_raw_llm_text = draft_result.raw_llm_output
        if initial_draft_text is None:
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} Failed - No Initial Draft"
            )
            return None
        return DraftResult(text=initial_draft_text, raw_llm_output=initial_raw_llm_text)

    async def _process_revision_result(
        self,
        novel_chapter_number: int,
        revision_result: RevisionOutcome,
    ) -> RevisionOutcome | None:
        processed_text = revision_result.text
        processed_raw_llm = revision_result.raw_llm_output
        is_flawed = revision_result.is_flawed
        if processed_text is None:
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} Failed - Revision/Processing Error"
            )
            return None
        return RevisionOutcome(
            text=processed_text,
            raw_llm_output=processed_raw_llm,
            is_flawed=is_flawed,
        )

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

        await self.refresh_plot_outline()
        if neo4j_manager.driver is not None:
            await self.refresh_knowledge_cache()
        else:
            logger.warning(
                "Neo4j driver not initialized. Skipping knowledge cache refresh."
            )
        self.next_chapter_context = await self.context_service.build_hybrid_context(
            self,
            novel_chapter_number + 1,
            None,
            None,
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
    ) -> str | None:
        return await run_chapter_pipeline(self, novel_chapter_number)

    def _validate_critical_configs(self) -> bool:
        critical_str_configs = {
            "OLLAMA_EMBED_URL": settings.OLLAMA_EMBED_URL,
            "OPENAI_API_BASE": settings.OPENAI_API_BASE,
            "EMBEDDING_MODEL": settings.EMBEDDING_MODEL,
            "NEO4J_URI": settings.NEO4J_URI,
            "LARGE_MODEL": settings.LARGE_MODEL,
            "MEDIUM_MODEL": settings.MEDIUM_MODEL,
            "SMALL_MODEL": settings.SMALL_MODEL,
            "NARRATOR_MODEL": settings.NARRATOR_MODEL,
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

        if settings.EXPECTED_EMBEDDING_DIM <= 0:
            logger.critical(
                f"NANA CRITICAL CONFIGURATION ERROR: EXPECTED_EMBEDDING_DIM must be a positive integer, but is {settings.EXPECTED_EMBEDDING_DIM}."
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
            async with neo4j_manager:
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

            runner = ChapterGenerationRunner(self)
            await runner.run()

            final_chapter_count_from_db = (
                await chapter_queries.load_chapter_count_from_db()
            )
            logger.info("\n--- NANA: Novel writing process finished for this run ---")
            logger.info(
                f"NANA: Successfully processed {runner.chapters_written} chapter(s) in this run."
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
            await self.shutdown()

    async def run_ingestion_process(self, text_file: str) -> None:
        """Ingest existing text and populate the knowledge graph."""
        logger.info("--- NANA: Starting Ingestion Process ---")

        if not self._validate_critical_configs():
            await self.display.stop()
            return

        self.display.start()
        self.run_start_time = time.time()

        manager = IngestionManager(
            finalize_agent=self.finalize_agent,
            planner_agent=self.planner_agent,
            kg_maintainer=self.kg_maintainer_agent,
            file_manager=self.file_manager,
        )

        await manager.ingest(text_file)
        await self.refresh_plot_outline()
        if neo4j_manager.driver is not None:
            await self.refresh_knowledge_cache()
        else:
            logger.warning(
                "Neo4j driver not initialized. Skipping knowledge cache refresh."
            )

        self.chapter_count = await chapter_queries.load_chapter_count_from_db()
        await self.display.stop()
        await self.shutdown()
        logger.info("NANA: Ingestion process completed.")

    async def shutdown(self) -> None:
        """Close the Neo4j driver and the LLM service."""
        if neo4j_manager.driver is not None:
            await neo4j_manager.close()
        await llm_service.aclose()
