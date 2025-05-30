# nana_orchestrator.py
import logging
import logging.handlers
import os
import random
import asyncio
import time # For Rich display updates
from typing import Dict, Any, Optional, List, Tuple

import config
import llm_interface
import utils # ADDED utils for deduplication and _is_fill_in
from core_db.base_db_manager import neo4j_manager
from data_access import (
    plot_queries,
    character_queries,
    world_queries,
    chapter_queries,
    kg_queries
)
from type import EvaluationResult, SceneDetail, ProblemDetail, AgentStateData

from comprehensive_evaluator_agent import ComprehensiveEvaluatorAgent
from planner_agent import PlannerAgent
from drafting_agent import DraftingAgent
from kg_maintainer_agent import KGMaintainerAgent
from world_continuity_agent import WorldContinuityAgent

from initial_setup_logic import generate_plot_outline_logic, generate_world_building_logic
from context_generation_logic import generate_hybrid_chapter_context_logic
from chapter_revision_logic import revise_chapter_draft_logic

try:
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich.console import Group
    # from rich.table import Table # Not used yet
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    class Live:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def update(self, *args, **kwargs): pass
        def stop(self): pass
    class Text: # type: ignore
        def __init__(self, initial_text: str = ""): self.plain = initial_text 
        def plain_setter(self, value: str): self.plain = value # type: ignore
    class Group: # type: ignore
        def __init__(self, *args, **kwargs): pass
    class Panel: # type: ignore
        def __init__(self, *args, **kwargs): pass
    class RichHandler: # type: ignore
        def __init__(self, *args, **kwargs): pass


logger = logging.getLogger(__name__)

class NANA_Orchestrator:
    def __init__(self):
        logger.info("Initializing NANA Orchestrator...")
        self.planner_agent = PlannerAgent()
        self.drafting_agent = DraftingAgent()
        self.evaluator_agent = ComprehensiveEvaluatorAgent()
        self.world_continuity_agent = WorldContinuityAgent()
        self.kg_maintainer_agent = KGMaintainerAgent()

        self.plot_outline: Dict[str, Any] = {}
        self.character_profiles: Dict[str, Any] = {}
        self.world_building: Dict[str, Any] = {}
        self.chapter_count: int = 0 # Number of chapters *successfully written and saved*
        self.novel_props_cache: Dict[str, Any] = {}
        self.total_tokens_generated_this_run: int = 0

        self.rich_live: Optional[Live] = None
        self.rich_status_group: Optional[Group] = None
        self.status_text_novel_title: Text = Text("Novel: N/A")
        self.status_text_current_chapter: Text = Text("Current Chapter: N/A")
        self.status_text_current_step: Text = Text("Current Step: Initializing...")
        self.status_text_tokens_generated: Text = Text("Tokens Generated (this run): 0")
        self.status_text_elapsed_time: Text = Text("Elapsed Time: 0s")
        self.run_start_time: float = 0.0

        if RICH_AVAILABLE and config.ENABLE_RICH_PROGRESS:
            self.rich_status_group = Group(
                self.status_text_novel_title,
                self.status_text_current_chapter,
                self.status_text_current_step,
                self.status_text_tokens_generated,
                self.status_text_elapsed_time
            )
            self.rich_live = Live(
                Panel(self.rich_status_group, title="SAGA NANA Progress", border_style="blue", expand=True),
                refresh_per_second=config.RICH_REFRESH_PER_SECOND,
                transient=False, 
                redirect_stdout=False, 
                redirect_stderr=False
            )
        else:
            logger.info("Rich library not available or ENABLE_RICH_PROGRESS is False. Progress will be shown via standard logs.")
        utils.load_spacy_model_if_needed() 
        logger.info("NANA Orchestrator initialized.")

    def _update_rich_display(self, chapter_num: Optional[int] = None, step: Optional[str] = None):
        if not (RICH_AVAILABLE and config.ENABLE_RICH_PROGRESS and self.rich_live and self.rich_status_group):
            return
        
        # Rich Text plain attribute needs to be set directly
        if chapter_num is not None:
            self.status_text_current_chapter.plain = f"Current Chapter: {chapter_num}" # type: ignore
        if step is not None:
            self.status_text_current_step.plain = f"Current Step: {step}" # type: ignore
        self.status_text_novel_title.plain = f"Novel: {self.plot_outline.get('title', 'N/A')}" # type: ignore
        self.status_text_tokens_generated.plain = f"Tokens Generated (this run): {self.total_tokens_generated_this_run:,}" # type: ignore
        elapsed_seconds = time.time() - self.run_start_time
        self.status_text_elapsed_time.plain = f"Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_seconds))}" # type: ignore

    def _accumulate_tokens(self, operation_name: str, usage_data: Optional[Dict[str, int]]):
        if usage_data and isinstance(usage_data.get("completion_tokens"), int):
            completed_tokens = usage_data["completion_tokens"]
            self.total_tokens_generated_this_run += completed_tokens
            logger.info(
                f"NANA Activity: Tokens from '{operation_name}': {completed_tokens}. "
                f"Total generated this run: {self.total_tokens_generated_this_run}"
            )
            self._update_rich_display()
        elif usage_data and isinstance(usage_data.get("total_tokens"), int) and not isinstance(usage_data.get("completion_tokens"), int):
            logger.warning(f"NANA Activity: '{operation_name}' - 'completion_tokens' missing or not int in usage_data. Tokens not added. Usage: {usage_data}")

    def _update_novel_props_cache(self):
        self.novel_props_cache = {
            "title": self.plot_outline.get("title", config.DEFAULT_PLOT_OUTLINE_TITLE),
            "genre": self.plot_outline.get("genre", config.CONFIGURED_GENRE),
            "theme": self.plot_outline.get("theme", config.CONFIGURED_THEME),
            "protagonist_name": self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME),
            "character_arc": self.plot_outline.get("character_arc", "N/A"),
            "logline": self.plot_outline.get("logline", "N/A"),
            "plot_points": self.plot_outline.get("plot_points", []), 
            "character_profiles": self.character_profiles,
            "world_building": self.world_building,
            "plot_outline_full": self.plot_outline 
        }
        self._update_rich_display()

    async def async_init_orchestrator(self):
        logger.info("NANA Orchestrator async_init_orchestrator started...")
        self._update_rich_display(step="Initializing Orchestrator")
        self.chapter_count = await chapter_queries.load_chapter_count_from_db()
        logger.info(f"Loaded chapter count from Neo4j: {self.chapter_count}")
        load_tasks = {
            "plot": plot_queries.get_plot_outline_from_db(),
            "chars": character_queries.get_character_profiles_from_db(),
            "world": world_queries.get_world_building_from_db()
        }
        results = await asyncio.gather(*load_tasks.values(), return_exceptions=True)
        loaded_data = dict(zip(load_tasks.keys(), results))
        for key, value in loaded_data.items():
            if isinstance(value, Exception):
                logger.error(f"Error loading {key} during orchestrator init: {value}", exc_info=value)
                if key == "plot": self.plot_outline = {}
                elif key == "chars": self.character_profiles = {}
                elif key == "world": self.world_building = {}
            else:
                if key == "plot": self.plot_outline = value if isinstance(value, dict) else {}
                elif key == "chars": self.character_profiles = value if isinstance(value, dict) else {}
                elif key == "world": self.world_building = value if isinstance(value, dict) else {}
        
        if not self.plot_outline.get("plot_points"):
            logger.warning("Orchestrator init: Plot outline loaded from DB has no plot points. Initial setup might be needed or DB is empty/corrupt.")
        else:
            logger.info(f"Orchestrator init: Loaded {len(self.plot_outline.get('plot_points',[]))} plot points from DB.")

        self._update_novel_props_cache()
        logger.info("NANA Orchestrator async_init_orchestrator complete.")
        self._update_rich_display(step="Orchestrator Initialized")

    async def _save_core_novel_state_to_neo4j(self):
        logger.info("NANA: Saving core novel state (plot, characters, world) to Neo4j...")
        tasks = [
            plot_queries.save_plot_outline_to_db(self.plot_outline),
            character_queries.save_character_profiles_to_db(self.character_profiles),
            world_queries.save_world_building_to_db(self.world_building)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = 0
        for i, res in enumerate(results):
            item_name = ["plot_outline", "character_profiles", "world_building"][i]
            if isinstance(res, Exception): logger.error(f"Failed to save {item_name} to Neo4j: {res}", exc_info=res)
            elif res is True: success_count += 1
            else: logger.warning(f"Unexpected return value from save_{item_name}: {res}")
        if success_count == len(tasks): logger.info("All core state objects saved to Neo4j successfully.")
        else: logger.warning(f"Only {success_count}/{len(tasks)} core state objects saved successfully.")

    async def perform_initial_setup(self):
        self._update_rich_display(step="Performing Initial Setup")
        logger.info("NANA performing initial setup...")
        logger.info("\n--- NANA: Initializing Plot, Characters, and World ---")
        
        generation_params: Dict[str, Any] = {} 
        if config.UNHINGED_PLOT_MODE and not os.path.exists(config.USER_STORY_ELEMENTS_FILE_PATH):
            generation_params.update({
                "genre": random.choice(config.UNHINGED_GENRES), "theme": random.choice(config.UNHINGED_THEMES),
                "setting_archetype": random.choice(config.UNHINGED_SETTINGS_ARCHETYPES),
                "protagonist_archetype": random.choice(config.UNHINGED_PROTAGONIST_ARCHETYPES),
                "conflict_archetype": random.choice(config.UNHINGED_CONFLICT_TYPES)
            })
        
        _, plot_usage = await generate_plot_outline_logic(self, config.DEFAULT_PROTAGONIST_NAME,
                                          config.UNHINGED_PLOT_MODE, 
                                          **generation_params)
        self._accumulate_tokens("InitialSetup-PlotOutline", plot_usage)
        plot_source = self.plot_outline.get("source", "unknown")
        logger.info(f"   Plot Outline initialized/loaded (source: {plot_source}). Title: '{self.plot_outline.get('title', 'N/A')}'. Number of plot points: {len(self.plot_outline.get('plot_points',[]))}")
        self._update_rich_display(step="Plot Outline Generated/Loaded")
        
        _, world_usage = await generate_world_building_logic(self)
        self._accumulate_tokens("InitialSetup-WorldBuilding", world_usage)
        world_source = self.world_building.get("source", "unknown")
        logger.info(f"   World Building initialized/loaded (source: {world_source}).")
        self._update_rich_display(step="World Building Generated/Loaded")
        
        self._update_novel_props_cache() 
        await self._save_core_novel_state_to_neo4j()
        logger.info("   Initial plot, character, and world data saved to Neo4j.")
        self._update_rich_display(step="Initial State Saved")
        
        if not self.plot_outline or not self.plot_outline.get("plot_points") or self.plot_outline.get("is_default"):
            logger.warning("Initial setup resulted in a default or empty/short plot outline. This might impact generation quality.")
        
        return True

    async def _prepopulate_kg_if_needed(self):
        self._update_rich_display(step="Pre-populating KG (if needed)")
        logger.info("NANA: Checking if KG pre-population is needed...")
        
        plot_source = self.plot_outline.get("source", "")
        is_user_or_llm_plot = plot_source == "user_supplied_markdown" or \
                              plot_source.startswith("llm_generated") or \
                              plot_source == "default_fallback" 
                              
        if not is_user_or_llm_plot and not self.plot_outline.get("is_default", False):
            logger.info(f"Skipping KG pre-population: Plot outline source '{plot_source}' suggests it might not be ready for KG.")
            return

        pp_check_query = f"MATCH (ni:NovelInfo {{id: '{config.MAIN_NOVEL_INFO_NODE_ID}'}}) RETURN ni LIMIT 1"
        novel_info_exists = await neo4j_manager.execute_read_query(pp_check_query)

        if novel_info_exists and novel_info_exists[0]:
            logger.info(f"Found existing NovelInfo node. Assuming KG already pre-populated or managed elsewhere. Skipping explicit pre-population call.")
            return
        
        logger.info("\n--- NANA: Pre-populating Knowledge Graph from Initial Data ---")
        await self.kg_maintainer_agent.prepopulate_kg_from_initial_data(
            self.plot_outline, self.character_profiles, self.world_building
        )
        logger.info("   Knowledge Graph pre-population step complete.")
        self._update_rich_display(step="KG Pre-population Complete")

    def _get_plot_point_info_for_chapter(self, novel_chapter_number: int) -> Tuple[Optional[str], int]:
        """
        Gets the plot point focus and its index for a given novel_chapter_number.
        The plot_point_index is 0-based.
        """
        plot_points_list = self.plot_outline.get("plot_points", [])
        if not isinstance(plot_points_list, list) or not plot_points_list:
            logger.error(f"No plot points available in orchestrator state for chapter {novel_chapter_number}.")
            return None, -1
        
        plot_point_index = novel_chapter_number - 1

        if 0 <= plot_point_index < len(plot_points_list):
            plot_point_text = plot_points_list[plot_point_index]
            if isinstance(plot_point_text, str) and plot_point_text.strip() and not utils._is_fill_in(plot_point_text): # Ensure not [Fill-in]
                return plot_point_text, plot_point_index
            else:
                logger.error(f"Plot point at index {plot_point_index} for chapter {novel_chapter_number} is invalid or '[Fill-in]': {plot_point_text}")
                return None, -1 # Treat [Fill-in] as not available for focus
        else:
            logger.error(f"Plot point index {plot_point_index} is out of bounds for plot_points list (len: {len(plot_points_list)}) for chapter {novel_chapter_number}.")
            return None, -1

    async def _save_chapter_text_and_log(self, chapter_number: int, final_text: str, raw_llm_log: Optional[str]):
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._save_chapter_files_sync_io, chapter_number, final_text, raw_llm_log or "N/A")
            logger.info(f"Saved chapter text and raw LLM log files for ch {chapter_number}.")
        except IOError as e:
            logger.error(f"Failed writing chapter text/log files for ch {chapter_number}: {e}", exc_info=True)

    def _save_chapter_files_sync_io(self, chapter_number: int, final_text: str, raw_llm_log: str):
        chapter_file_path = os.path.join(config.CHAPTERS_DIR, f"chapter_{chapter_number:04d}.txt")
        log_file_path = os.path.join(config.CHAPTER_LOGS_DIR, f"chapter_{chapter_number:04d}_raw_llm_log.txt")
        os.makedirs(os.path.dirname(chapter_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(chapter_file_path, 'w', encoding='utf-8') as f: f.write(final_text)
        with open(log_file_path, 'w', encoding='utf-8') as f: f.write(raw_llm_log)

    async def _save_debug_output(self, chapter_number: int, stage_description: str, content: Any):
        if content is None: return
        content_str = str(content) if not isinstance(content, str) else content
        if not content_str.strip(): return
        loop = asyncio.get_event_loop()
        try:
            safe_stage_desc = "".join(c if c.isalnum() or c in ['_', '-'] else "_" for c in stage_description)
            file_name = f"chapter_{chapter_number:04d}_{safe_stage_desc}.txt"
            file_path = os.path.join(config.DEBUG_OUTPUTS_DIR, file_name)
            await loop.run_in_executor(None, self._save_debug_output_sync_io, file_path, content_str)
            logger.debug(f"Saved debug output for Ch {chapter_number}, Stage '{stage_description}' to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save debug output (Ch {chapter_number}, Stage '{stage_description}'): {e}", exc_info=True)

    def _save_debug_output_sync_io(self, file_path: str, content_str: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f: f.write(content_str)

    async def perform_deduplication(self, text_to_dedup: str, chapter_number: int) -> Tuple[str, int]:
        logger.info(f"NANA: Performing de-duplication for Chapter {chapter_number}...")
        segment_level_dedup = "paragraph" 
        use_semantic_dedup = False 
        semantic_similarity_thresh_dedup = 0.97 
        min_segment_len_dedup = 80
        if not text_to_dedup or not text_to_dedup.strip():
            logger.info(f"De-duplication for Chapter {chapter_number}: Input text is empty. No action taken.")
            return text_to_dedup, 0
        try:
            deduplicated_text, chars_removed = await utils.deduplicate_text_segments(
                original_text=text_to_dedup, segment_level=segment_level_dedup,
                similarity_threshold=semantic_similarity_thresh_dedup,
                use_semantic_comparison=use_semantic_dedup,
                min_segment_length_chars=min_segment_len_dedup
            )
            if chars_removed > 0:
                logger.info(f"De-duplication for Chapter {chapter_number} removed {chars_removed} characters using {segment_level_dedup} matching (normalized string).")
            else:
                logger.info(f"De-duplication for Chapter {chapter_number}: No significant duplicates found at {segment_level_dedup} level (normalized string).")
            return deduplicated_text, chars_removed
        except Exception as e:
            logger.error(f"Error during de-duplication for Chapter {chapter_number}: {e}", exc_info=True)
            return text_to_dedup, 0

    async def run_chapter_generation_process(self, novel_chapter_number: int) -> Optional[str]:
        logger.info(f"=== NANA: Starting Novel Chapter {novel_chapter_number} Generation ===")
        self._update_rich_display(chapter_num=novel_chapter_number, step="Starting Chapter")
        
        if not self.plot_outline or not self.plot_outline.get("plot_points") or not self.plot_outline.get("protagonist_name"):
            logger.error(f"NANA: Cannot write Ch {novel_chapter_number}: Plot outline or critical plot data missing from orchestrator state.")
            self._update_rich_display(step=f"Ch {novel_chapter_number} Failed - Missing Plot Outline")
            return None

        plot_point_focus, plot_point_index = self._get_plot_point_info_for_chapter(novel_chapter_number)
        if plot_point_focus is None: # This now also checks for [Fill-in]
            logger.error(f"NANA: Ch {novel_chapter_number} generation halted: no concrete plot point focus could be determined (index {plot_point_index}).")
            self._update_rich_display(step=f"Ch {novel_chapter_number} Failed - No Plot Point Focus")
            return None

        self._update_rich_display(step=f"Ch {novel_chapter_number} - Planning Scenes")
        chapter_plan_result, plan_usage = await self.planner_agent.plan_chapter_scenes(
            self.novel_props_cache, novel_chapter_number, plot_point_focus, plot_point_index
        )
        self._accumulate_tokens(f"Ch{novel_chapter_number}-Planning", plan_usage)
        chapter_plan: Optional[List[SceneDetail]] = chapter_plan_result
        if config.ENABLE_AGENTIC_PLANNING and chapter_plan is None:
            logger.warning(f"NANA: Ch {novel_chapter_number}: Planning Agent failed or plan invalid. Proceeding with plot point focus only for drafting.")
        await self._save_debug_output(novel_chapter_number, "scene_plan", chapter_plan if chapter_plan else "No plan generated.")
        
        self._update_rich_display(step=f"Ch {novel_chapter_number} - Generating Hybrid Context")
        hybrid_context_for_draft = await generate_hybrid_chapter_context_logic(self.novel_props_cache, novel_chapter_number, chapter_plan)
        await self._save_debug_output(novel_chapter_number, "hybrid_context_for_draft", hybrid_context_for_draft)
        
        self._update_rich_display(step=f"Ch {novel_chapter_number} - Drafting Initial Text")
        initial_draft_text, initial_raw_llm_text, draft_usage = await self.drafting_agent.draft_chapter(
            self.novel_props_cache, novel_chapter_number, plot_point_focus, hybrid_context_for_draft, chapter_plan 
        )
        self._accumulate_tokens(f"Ch{novel_chapter_number}-Drafting", draft_usage)
        if not initial_draft_text:
            logger.error(f"NANA: Drafting Agent failed for Ch {novel_chapter_number}. No initial draft produced.")
            await self._save_debug_output(novel_chapter_number, "initial_draft_fail_raw_llm", initial_raw_llm_text or "Drafting Agent returned None for raw output.")
            self._update_rich_display(step=f"Ch {novel_chapter_number} Failed - No Initial Draft")
            return None
        await self._save_debug_output(novel_chapter_number, "initial_draft", initial_draft_text)
        
        current_text_to_process: Optional[str] = initial_draft_text
        current_raw_llm_output: Optional[str] = initial_raw_llm_text
        is_from_flawed_source_for_kg = False 

        max_revision_attempts = 1
        for attempt in range(max_revision_attempts + 1): 
            if current_text_to_process is None:
                 logger.error(f"NANA: Ch {novel_chapter_number} - Text became None before processing cycle {attempt + 1}. Aborting chapter.")
                 self._update_rich_display(step=f"Ch {novel_chapter_number} Failed - Text lost before Cycle {attempt + 1}")
                 return None

            self._update_rich_display(step=f"Ch {novel_chapter_number} - De-duplication Attempt {attempt + 1}")
            logger.info(f"NANA: Ch {novel_chapter_number} - Pre-Evaluation De-duplication, Cycle Attempt {attempt + 1}")
            deduplicated_text, removed_char_count = await self.perform_deduplication(
                current_text_to_process, novel_chapter_number
            )
            if removed_char_count > 0:
                is_from_flawed_source_for_kg = True 
                logger.info(f"NANA: Ch {novel_chapter_number} - De-duplication (Attempt {attempt+1}) removed {removed_char_count} characters. Text marked as potentially flawed for KG.")
                current_text_to_process = deduplicated_text
                await self._save_debug_output(novel_chapter_number, f"deduplicated_text_attempt_{attempt+1}", current_text_to_process)
            else:
                logger.info(f"NANA: Ch {novel_chapter_number} - De-duplication (Attempt {attempt+1}) found no significant changes.")

            logger.info(f"NANA: Ch {novel_chapter_number} - Evaluation Cycle, Attempt {attempt + 1}")
            self._update_rich_display(step=f"Ch {novel_chapter_number} - Evaluation Cycle {attempt + 1}")
            
            eval_result_obj, eval_usage = await self.evaluator_agent.evaluate_chapter_draft(
                self.novel_props_cache, current_text_to_process, novel_chapter_number,
                plot_point_focus, plot_point_index, hybrid_context_for_draft 
            )
            self._accumulate_tokens(f"Ch{novel_chapter_number}-Evaluation-Attempt{attempt+1}", eval_usage)
            evaluation_result: EvaluationResult = eval_result_obj
            await self._save_debug_output(novel_chapter_number, f"evaluation_result_attempt_{attempt+1}", evaluation_result)

            self._update_rich_display(step=f"Ch {novel_chapter_number} - Continuity Check {attempt + 1}")
            continuity_problems, continuity_usage = await self.world_continuity_agent.check_consistency(
                 self.novel_props_cache, current_text_to_process, novel_chapter_number, hybrid_context_for_draft
            )
            self._accumulate_tokens(f"Ch{novel_chapter_number}-ContinuityCheck-Attempt{attempt+1}", continuity_usage)
            await self._save_debug_output(novel_chapter_number, f"continuity_problems_attempt_{attempt+1}", continuity_problems)

            if continuity_problems:
                logger.warning(f"NANA: Ch {novel_chapter_number} (Attempt {attempt+1}) - World Continuity Agent found {len(continuity_problems)} issues.")
                evaluation_result["problems_found"].extend(continuity_problems) 
                if not evaluation_result["needs_revision"]: 
                    evaluation_result["needs_revision"] = True
                unique_reasons = set(evaluation_result.get("reasons", []))
                unique_reasons.add("Continuity issues identified by WorldContinuityAgent.")
                evaluation_result["reasons"] = sorted(list(unique_reasons))
            
            if not evaluation_result["needs_revision"]:
                logger.info(f"NANA: Ch {novel_chapter_number} draft passed evaluation (Attempt {attempt+1}). Text is considered good.")
                self._update_rich_display(step=f"Ch {novel_chapter_number} - Passed Evaluation")
                break 
            else: 
                logger.warning(f"NANA: Ch {novel_chapter_number} draft (Attempt {attempt+1}) needs revision. Reasons: {'; '.join(evaluation_result.get('reasons',[]))}")
                if attempt >= max_revision_attempts:
                    logger.error(f"NANA: Ch {novel_chapter_number} - Max revision attempts ({max_revision_attempts+1}) reached. Proceeding with current draft, marked as flawed.")
                    is_from_flawed_source_for_kg = True 
                    self._update_rich_display(step=f"Ch {novel_chapter_number} - Max Revisions Reached (Flawed)")
                    break 

                self._update_rich_display(step=f"Ch {novel_chapter_number} - Revision Attempt {attempt + 1}")
                revision_tuple_result, revision_usage = await revise_chapter_draft_logic(
                    self, current_text_to_process, novel_chapter_number, 
                    evaluation_result, hybrid_context_for_draft, chapter_plan
                )
                self._accumulate_tokens(f"Ch{novel_chapter_number}-Revision-Attempt{attempt+1}", revision_usage)
                
                if revision_tuple_result and revision_tuple_result[0] and len(revision_tuple_result[0]) > 50:
                    current_text_to_process, rev_raw_output = revision_tuple_result
                    current_raw_llm_output = rev_raw_output if rev_raw_output else current_raw_llm_output
                    logger.info(f"NANA: Ch {novel_chapter_number} - Revision attempt {attempt+1} successful. New text length: {len(current_text_to_process)}. Re-processing (de-dup & eval).")
                    await self._save_debug_output(novel_chapter_number, f"revised_text_attempt_{attempt+1}", current_text_to_process)
                else:
                    logger.error(f"NANA: Ch {novel_chapter_number} - Revision attempt {attempt+1} failed to produce usable text. Proceeding with previous draft, marked as flawed.")
                    is_from_flawed_source_for_kg = True 
                    self._update_rich_display(step=f"Ch {novel_chapter_number} - Revision Failed (Flawed)")
                    break 
        
        if current_text_to_process is None: 
            logger.critical(f"NANA: Ch {novel_chapter_number} - current_text_to_process is None after revision loop. Aborting chapter.")
            return None
        
        if len(current_text_to_process) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
             logger.warning(f"NANA: Final chosen text for Ch {novel_chapter_number} is short ({len(current_text_to_process)} chars). Marked as flawed for KG.")
             is_from_flawed_source_for_kg = True

        self._update_rich_display(step=f"Ch {novel_chapter_number} - Summarizing")
        chapter_summary, summary_usage = await self.kg_maintainer_agent.summarize_chapter(current_text_to_process, novel_chapter_number)
        self._accumulate_tokens(f"Ch{novel_chapter_number}-Summarization", summary_usage)
        await self._save_debug_output(novel_chapter_number, "final_summary", chapter_summary)
        
        final_embedding = await llm_interface.async_get_embedding(current_text_to_process)
        if final_embedding is None:
            logger.error(f"NANA CRITICAL: Failed to generate embedding for final text of Chapter {novel_chapter_number}. Text saved to file system only.")
            await self._save_chapter_text_and_log(novel_chapter_number, current_text_to_process, current_raw_llm_output)
            self._update_rich_display(step=f"Ch {novel_chapter_number} Failed - No Embedding")
            return None

        self._update_rich_display(step=f"Ch {novel_chapter_number} - Saving to DB")
        await chapter_queries.save_chapter_data_to_db(
            novel_chapter_number, current_text_to_process, current_raw_llm_output or "N/A",
            chapter_summary, final_embedding, is_from_flawed_source_for_kg 
        )
        await self._save_chapter_text_and_log(novel_chapter_number, current_text_to_process, current_raw_llm_output)

        self._update_rich_display(step=f"Ch {novel_chapter_number} - Updating KG")
        kg_merge_usage = await self.kg_maintainer_agent.extract_and_merge_knowledge(
            self.novel_props_cache, novel_chapter_number, current_text_to_process, is_from_flawed_source_for_kg
        )
        self._accumulate_tokens(f"Ch{novel_chapter_number}-KGExtractionMerge", kg_merge_usage)
        self.character_profiles = self.novel_props_cache['character_profiles'] 
        self.world_building = self.novel_props_cache['world_building']   
        self._update_novel_props_cache()

        self.chapter_count = max(self.chapter_count, novel_chapter_number) 
        
        await self._save_core_novel_state_to_neo4j() 

        status_message = "Successfully Generated" if not is_from_flawed_source_for_kg else "Generated (Marked with Flaws)"
        logger.info(f"=== NANA: Finished Novel Chapter {novel_chapter_number} - {status_message} ===")
        self._update_rich_display(step=f"Ch {novel_chapter_number} - {status_message}")
        return current_text_to_process

    def _validate_critical_configs(self) -> bool:
        """
        Performs basic validation of critical string configurations.
        Returns True if all checks pass, False otherwise.
        """
        critical_str_configs = {
            "OLLAMA_EMBED_URL": config.OLLAMA_EMBED_URL,
            "OPENAI_API_BASE": config.OPENAI_API_BASE,
            "EMBEDDING_MODEL": config.EMBEDDING_MODEL,
            "NEO4J_URI": config.NEO4J_URI,
            "LARGE_MODEL": config.LARGE_MODEL,
            "MEDIUM_MODEL": config.MEDIUM_MODEL,
            "SMALL_MODEL": config.SMALL_MODEL,
            "NARRATOR_MODEL": config.NARRATOR_MODEL,
        }
        missing_or_empty_configs = []
        for name, value in critical_str_configs.items():
            if not value or not isinstance(value, str) or not value.strip():
                missing_or_empty_configs.append(name)
        
        if missing_or_empty_configs:
            logger.critical(f"NANA CRITICAL CONFIGURATION ERROR: The following critical configuration(s) are missing or empty: {', '.join(missing_or_empty_configs)}. Please set them (e.g., in .env file or environment variables) and restart.")
            return False
        
        if config.EXPECTED_EMBEDDING_DIM <= 0:
            logger.critical(f"NANA CRITICAL CONFIGURATION ERROR: EXPECTED_EMBEDDING_DIM must be a positive integer, but is {config.EXPECTED_EMBEDDING_DIM}.")
            return False
            
        logger.info("Critical configurations validated successfully.")
        return True

    async def run_novel_generation_loop(self):
        logger.info("--- NANA: Starting Novel Generation Run ---")
        
        if not self._validate_critical_configs(): # MODIFICATION: Added validation call
            self._update_rich_display(step="Critical Config Error - Halting")
            if self.rich_live and self.rich_live.is_started: self.rich_live.stop() # type: ignore
            return # Halt if critical configs are missing

        self.total_tokens_generated_this_run = 0
        self.run_start_time = time.time()
        if self.rich_live: self.rich_live.start()
        try:
            await neo4j_manager.connect()
            await neo4j_manager.create_db_schema()
            logger.info("NANA: Neo4j connection and schema verified.")
            await self.async_init_orchestrator() # Loads existing state from DB

            plot_points_exist = self.plot_outline and self.plot_outline.get("plot_points") and \
                                len([pp for pp in self.plot_outline.get("plot_points", []) if not utils._is_fill_in(pp)]) > 0 # Use utils._is_fill_in
            
            if not plot_points_exist or not self.plot_outline.get("title") or utils._is_fill_in(self.plot_outline.get("title")): # Use utils._is_fill_in
                logger.info("NANA: Core plot data missing or insufficient (e.g., no title, no concrete plot points). Performing initial setup...")
                if not await self.perform_initial_setup(): 
                    logger.critical("NANA: Initial setup failed. Halting generation.")
                    self._update_rich_display(step="Initial Setup Failed - Halting")
                    return
                await self.async_init_orchestrator() 
            
            await self._prepopulate_kg_if_needed() 
            self._update_novel_props_cache() 

            logger.info("\n--- NANA: Starting Novel Writing Process ---")
            
            start_novel_chapter_to_write = self.chapter_count + 1
            
            available_plot_points = [pp for pp in self.plot_outline.get("plot_points", []) if not utils._is_fill_in(pp)] # Use utils._is_fill_in
            total_concrete_plot_points = len(available_plot_points)
            
            plot_points_covered_count = self.chapter_count 
            remaining_plot_points_to_address_in_novel = total_concrete_plot_points - plot_points_covered_count

            logger.info(f"NANA: Current Novel Chapter Count (State): {self.chapter_count}")
            logger.info(f"NANA: Total Concrete Plot Points in Outline: {total_concrete_plot_points}")
            logger.info(f"NANA: Remaining Concrete Plot Points to Cover in Novel: {remaining_plot_points_to_address_in_novel}")

            if remaining_plot_points_to_address_in_novel <= 0:
                logger.info(f"NANA: All {total_concrete_plot_points} concrete plot points appear to be covered by existing {self.chapter_count} chapters. No new chapters to generate.")
                self._update_rich_display(chapter_num=self.chapter_count, step="All Plot Points Covered")
            else:
                chapters_to_attempt_this_run = min(config.CHAPTERS_PER_RUN, remaining_plot_points_to_address_in_novel)
                logger.info(f"NANA: Targeting up to {chapters_to_attempt_this_run} new chapter(s) in this run, starting with Novel Chapter {start_novel_chapter_to_write}.")

                chapters_successfully_written_this_run = 0
                for k_th_chapter_this_run in range(chapters_to_attempt_this_run):
                    current_novel_chapter_number = start_novel_chapter_to_write + k_th_chapter_this_run
                    
                    logger.info(f"\n--- NANA: Attempting Novel Chapter {current_novel_chapter_number} ({k_th_chapter_this_run + 1}/{chapters_to_attempt_this_run} in this run) ---")
                    self._update_rich_display(chapter_num=current_novel_chapter_number, step="Starting Chapter Loop")
                    
                    try:
                        chapter_text_result = await self.run_chapter_generation_process(current_novel_chapter_number)
                        if chapter_text_result:
                            chapters_successfully_written_this_run += 1
                            logger.info(f"NANA: Novel Chapter {current_novel_chapter_number}: Processed. Final text length: {len(chapter_text_result)} chars.")
                            logger.info(f"   Snippet: {chapter_text_result[:200].replace(chr(10), ' ')}...")
                        else:
                            logger.error(f"NANA: Novel Chapter {current_novel_chapter_number}: Failed to generate or save. Halting run.")
                            self._update_rich_display(step=f"Ch {current_novel_chapter_number} Failed - Halting Run")
                            break 
                    except Exception as e:
                        logger.critical(f"NANA: Critical unhandled error during Novel Chapter {current_novel_chapter_number} writing process: {e}", exc_info=True)
                        self._update_rich_display(step=f"Critical Error Ch {current_novel_chapter_number} - Halting Run")
                        break
                
                final_chapter_count_from_db = await chapter_queries.load_chapter_count_from_db()
                logger.info(f"\n--- NANA: Novel writing process finished for this run ---")
                logger.info(f"NANA: Successfully processed {chapters_successfully_written_this_run} chapter(s) in this run.")
                logger.info(f"NANA: Current total chapters in database after this run: {final_chapter_count_from_db}")
            
            logger.info(f"NANA: Total LLM tokens generated this run: {self.total_tokens_generated_this_run}")
            self._update_rich_display(chapter_num=self.chapter_count, step="Run Finished")

        except Exception as e:
            logger.critical(f"NANA: Unhandled exception in orchestrator main loop: {e}", exc_info=True)
            if self.rich_live and self.rich_live.is_started: # type: ignore
                 self._update_rich_display(step="Critical Error in Main Loop")
        finally:
            if self.rich_live and self.rich_live.is_started: # type: ignore
                await asyncio.sleep(0.1) 
                self.rich_live.stop()
            await neo4j_manager.close()
            logger.info("NANA: Neo4j driver successfully closed on application exit.")

def setup_logging_nana():
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
        handlers=[] 
    )
    root_logger = logging.getLogger()
    # Remove any default handlers that basicConfig might have added
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if config.LOG_FILE:
        try:
            log_dir = os.path.dirname(config.LOG_FILE)
            if log_dir: os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                config.LOG_FILE, maxBytes=10*1024*1024, backupCount=5, mode='a', encoding='utf-8'
            )
            file_handler.setLevel(config.LOG_LEVEL)
            formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            root_logger.info(f"File logging enabled. Log file: {config.LOG_FILE}")
        except Exception as e:
            # Fallback to console if file logging fails
            console_handler_fallback = logging.StreamHandler()
            console_handler_fallback.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
            root_logger.addHandler(console_handler_fallback)
            root_logger.error(f"Failed to configure file logging to {config.LOG_FILE}: {e}. Logging to console instead.", exc_info=True)


    if RICH_AVAILABLE and config.ENABLE_RICH_PROGRESS:
        existing_console = None
        if root_logger.handlers:
             for h_idx, h in enumerate(root_logger.handlers): # Use enumerate if you need index
                  if hasattr(h, 'console') and not isinstance(h, logging.FileHandler): 
                       existing_console = h.console # type: ignore
                       break
        
        rich_handler = RichHandler(
            level=config.LOG_LEVEL, rich_tracebacks=True, show_path=False,
            markup=True, show_time=True, show_level=True,
            console=existing_console 
        )
        root_logger.addHandler(rich_handler)
        root_logger.info("Rich logging handler enabled for console.")
    elif not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers): 
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(config.LOG_LEVEL)
        stream_formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
        stream_handler.setFormatter(stream_formatter)
        root_logger.addHandler(stream_handler)
        root_logger.info("Standard stream logging handler enabled for console.")


    logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    root_logger.info(f"NANA Logging setup complete. Application Log Level: {logging.getLevelName(config.LOG_LEVEL)}.")

if __name__ == "__main__":
    setup_logging_nana()
    orchestrator = NANA_Orchestrator()
    try:
        asyncio.run(orchestrator.run_novel_generation_loop())
    except KeyboardInterrupt:
        logger.info("NANA Orchestrator shutting down gracefully due to KeyboardInterrupt...")
        if orchestrator.rich_live and orchestrator.rich_live.is_started: # type: ignore
            orchestrator._update_rich_display(step="Shutdown (KeyboardInterrupt)")
            orchestrator.rich_live.stop() # type: ignore
    except Exception as main_err:
        logger.critical(f"NANA Orchestrator encountered an unhandled main exception: {main_err}", exc_info=True)
        if orchestrator.rich_live and orchestrator.rich_live.is_started: # type: ignore
            orchestrator._update_rich_display(step=f"FATAL ERROR: {str(main_err)[:50]}...")
            orchestrator.rich_live.stop() # type: ignore
    finally:
        if neo4j_manager.driver is not None:
            logger.info("Ensuring Neo4j driver is closed from main entry point.")
            async def _close_driver_main():
                 await neo4j_manager.close()
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running() and not loop.is_closed():
                    asyncio.ensure_future(_close_driver_main()) 
                elif not loop.is_running() and not loop.is_closed(): 
                    loop.run_until_complete(_close_driver_main())
                else: 
                    asyncio.run(_close_driver_main()) 
            except RuntimeError as e:
                 logger.warning(f"Could not explicitly close driver from main (event loop might be closed or other issue): {e}")