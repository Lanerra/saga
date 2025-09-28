# agents/narrative_agent.py
import json
import re
from typing import Any

import structlog

import config
from core.llm_interface_refactored import llm_service
from core.text_processing_service import truncate_text_by_tokens
from data_access import chapter_queries
from models import CharacterProfile, SceneDetail, WorldItem
from models.narrative_state import NarrativeState
from prompts.prompt_data_getters import (
    get_reliable_kg_facts_for_drafting_prompt,
)
from prompts.prompt_renderer import get_system_prompt, render_prompt
from utils.common import extract_json_from_text, safe_json_loads, truncate_for_log

logger = structlog.get_logger()

SCENE_PLAN_KEY_MAP = {
    "scene": "scene_number",
    "scene_number": "scene_number",
    "summary": "summary",
    "characters_involved": "characters_involved",
    "key_dialogue_points": "key_dialogue_points",
    "setting_details": "setting_details",
    "scene_focus_elements": "scene_focus_elements",
    "contribution": "contribution",
    # NEW: Add the directorial fields to the key map
    "scene_type": "scene_type",
    "pacing": "pacing",
    "character_arc_focus": "character_arc_focus",
    "relationship_development": "relationship_development",
}
SCENE_PLAN_LIST_INTERNAL_KEYS = [
    "key_dialogue_points",
    "scene_focus_elements",
    "characters_involved",
]


class NarrativeAgent:
    def __init__(self, config: dict):
        self.logger = structlog.get_logger()
        self.config = config
        self.model = config.NARRATIVE_MODEL
        self._token_cache: dict[tuple[str, str], int] = {}

    def _cached_count_tokens(self, text: str, model: str) -> int:
        """Cache token counting to avoid redundant calculations."""
        cache_key = (text, model)
        if cache_key not in self._token_cache:
            self._token_cache[cache_key] = llm_service.count_tokens(text, model)
        return self._token_cache[cache_key]

    

    def _parse_llm_scene_plan_output(
        self, json_text: str, chapter_number: int
    ) -> list[SceneDetail] | None:
        """
        Parses JSON scene plan output from LLM.
        Expects a JSON array of scene objects.
        """
        if not json_text or not json_text.strip():
            logger.warning(
                f"JSON scene plan for Ch {chapter_number} is empty. No scenes parsed."
            )
            return None

        # Enhanced JSON cleaning and extraction
        cleaned_json_text = self._clean_llm_json_response(json_text)

        try:
            parsed_data = json.loads(cleaned_json_text)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode JSON scene plan for Ch {chapter_number}: {e}. Text: {truncate_for_log(cleaned_json_text)}"
            )
            parsed_data = safe_json_loads(cleaned_json_text, expected=list)
            if parsed_data is None:
                if config.settings.NARRATIVE_JSON_DEBUG_SAVE:
                    self._save_malformed_json_for_debugging(
                        cleaned_json_text, chapter_number
                    )
                return None

        if not isinstance(parsed_data, list):
            logger.warning(
                f"Parsed scene plan for Ch {chapter_number} is not a list as expected. Type: {type(parsed_data)}. Data: {str(parsed_data)[:300]}"
            )
            return None

        if not parsed_data:
            logger.warning(
                f"Parsed scene plan for Ch {chapter_number} is an empty list."
            )
            return None

        scenes_data: list[SceneDetail] = []
        for i, scene_item in enumerate(parsed_data):
            if not isinstance(scene_item, dict):
                logger.warning(
                    f"Scene item {i + 1} in Ch {chapter_number} is not a dictionary. Skipping. Item: {str(scene_item)[:100]}"
                )
                continue

            processed_scene_dict: dict[str, Any] = {}
            for llm_key, value in scene_item.items():
                internal_key = SCENE_PLAN_KEY_MAP.get(
                    llm_key.lower().replace(" ", "_"), llm_key
                )
                processed_scene_dict[internal_key] = value

            scene_num = processed_scene_dict.get("scene_number")
            if not isinstance(scene_num, int):
                logger.warning(
                    f"Scene {i + 1} in Ch {chapter_number} has invalid or missing 'scene_number'. Assigning {i + 1}. Value: {scene_num}"
                )
                processed_scene_dict["scene_number"] = i + 1

            summary = processed_scene_dict.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                logger.warning(
                    f"Scene {scene_num} in Ch {chapter_number} has invalid or missing 'summary'. Skipping."
                )
                continue

            for list_key in SCENE_PLAN_LIST_INTERNAL_KEYS:
                val = processed_scene_dict.get(list_key)
                if isinstance(val, str):
                    processed_scene_dict[list_key] = [
                        v.strip() for v in val.split(",") if v.strip()
                    ]
                elif not isinstance(val, list):
                    processed_scene_dict[list_key] = (
                        [str(val)] if val is not None else []
                    )

            for key_internal_name in SCENE_PLAN_KEY_MAP.values():
                if key_internal_name not in processed_scene_dict:
                    if key_internal_name in SCENE_PLAN_LIST_INTERNAL_KEYS:
                        processed_scene_dict[key_internal_name] = []
                    else:
                        # For optional fields, this will be None.
                        processed_scene_dict[key_internal_name] = None

            scenes_data.append(processed_scene_dict)  # type: ignore

        if not scenes_data:
            logger.warning(f"No valid scenes parsed from JSON for Ch {chapter_number}.")
            return None

        scenes_data.sort(key=lambda x: x.get("scene_number", float("inf")))
        return scenes_data

    def _clean_llm_json_response(self, text: str) -> str:
        """
        Clean common LLM JSON response issues.
        This is a conservative approach that focuses on extracting valid JSON.
        """
        if not text or not isinstance(text, str):
            return text or ""

        # First, try to extract JSON structure from the text
        extracted = extract_json_from_text(text)
        if extracted:
            return extracted

        # If no JSON structure found, clean the text conservatively
        # Remove any text before the first [ or {
        first_bracket = min(
            [text.find(char) for char in ["[", "{"] if text.find(char) != -1]
            or [len(text)]
        )
        if first_bracket < len(text):
            text = text[first_bracket:]

        # Remove any text after the last ] or }
        last_bracket = max(
            [text.rfind(char) for char in ["]", "}"] if text.rfind(char) != -1] or [-1]
        )
        if last_bracket != -1:
            text = text[: last_bracket + 1]

        return text.strip()

    # JSON extraction moved to utils/json_utils.py for reuse

    def _save_malformed_json_for_debugging(
        self, malformed_json: str, chapter_number: int
    ) -> None:
        """
        Save malformed JSON to a file for debugging purposes.
        """
        try:
            import os

            import config

            # Use configured debug outputs directory under BASE_OUTPUT_DIR
            debug_dir = config.DEBUG_OUTPUTS_DIR
            os.makedirs(debug_dir, exist_ok=True)

            filename = f"chapter_{chapter_number:04d}_malformed_scene_plan.json"
            filepath = os.path.join(debug_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(malformed_json)

            logger.info(f"Saved malformed JSON for debugging: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save malformed JSON for debugging: {e}")

    async def _plan_chapter_scenes(
        self,
        plot_outline: dict[str, Any],
        character_profiles: list[CharacterProfile],  # List instead of dict
        world_building: list[WorldItem],  # List instead of nested dict
        chapter_number: int,
        plot_point_focus: str | None,
        plot_point_index: int,
    ) -> tuple[list[SceneDetail] | None, dict[str, int] | None]:
        """
        Generates a detailed scene plan for the chapter using native model lists.
        Returns the plan and LLM usage data.
        """
        if not self.config.ENABLE_AGENTIC_PLANNING:
            logger.info(
                f"Agentic planning disabled. Skipping detailed planning for Chapter {chapter_number}."
            )
            return None, None

        logger.info(
            f"NarrativeAgent planning Chapter {chapter_number} with detailed scenes..."
        )
        if plot_point_focus is None:
            logger.error(
                f"Cannot plan chapter {chapter_number}: No plot point focus available."
            )
            return None, None

        # Snapshot provides continuation context during drafting; not used for planning here
        context_summary_str = ""

        protagonist_name = plot_outline.get(
            "protagonist_name", self.config.DEFAULT_PROTAGONIST_NAME
        )

        # Use native prompt data getters
        from prompts.prompt_data_getters import (
            get_character_state_snippet_for_prompt,
            get_world_state_snippet_for_prompt,
        )

        kg_context_section = await get_reliable_kg_facts_for_drafting_prompt(
            plot_outline, chapter_number, None
        )
        character_state_snippet_plain_text = (
            await get_character_state_snippet_for_prompt(
                character_profiles, plot_outline, chapter_number
            )
        )
        world_state_snippet_plain_text = await get_world_state_snippet_for_prompt(
            world_building, chapter_number
        )

        # Build future plot context
        future_plot_context_parts: list[str] = []
        all_plot_points = plot_outline.get("plot_points", [])
        total_plot_points_in_novel = len(all_plot_points)

        if plot_point_index + 1 < total_plot_points_in_novel:
            next_pp_text = all_plot_points[plot_point_index + 1]
            if isinstance(next_pp_text, str) and next_pp_text.strip():
                future_plot_context_parts.append(
                    f"\n**Anticipated Next Major Plot Point (PP {plot_point_index + 2}/{total_plot_points_in_novel} - for context, not this chapter's focus):**\n{next_pp_text.strip()}\n"
                )
            if plot_point_index + 2 < total_plot_points_in_novel:
                next_next_pp_text = all_plot_points[plot_point_index + 2]
                if isinstance(next_next_pp_text, str) and next_next_pp_text.strip():
                    future_plot_context_parts.append(
                        f"**And Then (PP {plot_point_index + 3}/{total_plot_points_in_novel} - distant context):**\n{next_next_pp_text.strip()}\n"
                    )
        future_plot_context_str = "".join(future_plot_context_parts)

        prompt = render_prompt(
            "narrative_agent/scene_plan.j2",
            {
                "no_think": self.config.ENABLE_LLM_NO_THINK_DIRECTIVE,
                "target_scenes_min": self.config.TARGET_SCENES_MIN,
                "target_scenes_max": self.config.TARGET_SCENES_MAX,
                "chapter_number": chapter_number,
                "novel_title": plot_outline.get("title", "Untitled"),
                "novel_genre": plot_outline.get("genre", "N/A"),
                "novel_theme": plot_outline.get("theme", "N/A"),
                "protagonist_name": protagonist_name,
                "protagonist_arc": plot_outline.get("character_arc", "N/A"),
                "plot_point_index_plus1": plot_point_index + 1,
                "total_plot_points_in_novel": total_plot_points_in_novel,
                "plot_point_focus": plot_point_focus,
                "future_plot_context_str": future_plot_context_str,
                "context_summary_str": context_summary_str,
                "kg_context_section": kg_context_section,
                "character_state_snippet_plain_text": character_state_snippet_plain_text,
                "world_state_snippet_plain_text": world_state_snippet_plain_text,
            },
        )
        logger.info(
            f"Calling LLM ({self.model}) for detailed scene plan for chapter {chapter_number} (target scenes: {self.config.TARGET_SCENES_MIN}-{self.config.TARGET_SCENES_MAX}, expecting JSON). Plot Point {plot_point_index + 1}/{total_plot_points_in_novel}."
        )

        (
            cleaned_plan_text_from_llm,
            usage_data,
        ) = await llm_service.async_call_llm(
            model_name=self.model,
            prompt=prompt,
            temperature=self.config.Temperatures.PLANNING,
            max_tokens=self.config.MAX_PLANNING_TOKENS,
            allow_fallback=True,
            stream_to_disk=True,
            frequency_penalty=self.config.FREQUENCY_PENALTY_PLANNING,
            presence_penalty=self.config.PRESENCE_PENALTY_PLANNING,
            auto_clean_response=True,
            system_prompt=get_system_prompt("narrative_agent"),
        )

        parsed_scenes_list_of_dicts = self._parse_llm_scene_plan_output(
            cleaned_plan_text_from_llm, chapter_number
        )

        if parsed_scenes_list_of_dicts:
            final_scenes_typed: list[SceneDetail] = []
            for i, scene_dict in enumerate(parsed_scenes_list_of_dicts):
                if not isinstance(scene_dict, dict):
                    logger.warning(
                        f"Parsed scene item {i + 1} for ch {chapter_number} is not a dict. Skipping. Item: {scene_dict}"
                    )
                    continue

                # Basic validation for required fields
                if not scene_dict.get("summary"):
                    logger.warning(
                        f"Scene {scene_dict.get('scene_number', i + 1)} from parser for ch {chapter_number} has a missing summary. Skipping."
                    )
                    continue

                final_scenes_typed.append(scene_dict)  # type: ignore

            if final_scenes_typed:
                logger.info(
                    f"Generated valid detailed scene plan for chapter {chapter_number} with {len(final_scenes_typed)} scenes."
                )
                return final_scenes_typed, usage_data

        logger.warning(
            f"Failed to generate valid scene plan for chapter {chapter_number}. No parsable scenes."
        )
        return None, usage_data

    async def _draft_chapter(
        self,
        plot_outline: dict[str, Any],
        chapter_number: int,
        plot_point_focus: str,
        hybrid_context_for_draft: str,
        chapter_plan: list[SceneDetail],
    ) -> tuple[str | None, str | None, dict[str, int] | None]:
        """
        Generates the initial draft for a chapter.

        If ``chapter_plan`` is provided, each scene is drafted sequentially and
        concatenated. When ``chapter_plan`` is ``None`` the entire chapter is
        drafted in a single LLM call using the provided ``plot_point_focus``.

        Returns:
            Tuple of the draft text, the raw LLM output, and token usage data.
        """
        logger.info(
            f"NarrativeAgent: Starting scene-by-scene draft for Chapter {chapter_number}..."
        )

        all_scenes_prose: list[str] = []
        all_raw_outputs: list[str] = []
        total_usage_data: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        novel_title = plot_outline.get("title", "Untitled Novel")
        novel_genre = plot_outline.get("genre", "Unknown Genre")

        for scene_index, scene_detail in enumerate(chapter_plan):
            scene_number = scene_detail.get("scene_number", scene_index + 1)
            logger.info(f"Drafting Scene {scene_number} of Chapter {chapter_number}...")

            previous_scenes_prose = "\n\n".join(all_scenes_prose)
            max_tokens_for_prev_scenes = self.config.MAX_GENERATION_TOKENS
            previous_scenes_prose_for_prompt = truncate_text_by_tokens(
                previous_scenes_prose,
                self.model,
                max_tokens_for_prev_scenes,
                truncation_marker="\n\n... (prose from earlier scenes in this chapter has been truncated)\n\n",
            )

            prompt = render_prompt(
                "narrative_agent/draft_scene.j2",
                {
                    "no_think": self.config.ENABLE_LLM_NO_THINK_DIRECTIVE,
                    "chapter_number": chapter_number,
                    "novel_title": novel_title,
                    "novel_genre": novel_genre,
                    "scene_detail": scene_detail,
                    "hybrid_context_for_draft": hybrid_context_for_draft,
                    "previous_scenes_prose": previous_scenes_prose_for_prompt,
                    "min_length_per_scene": self.config.MIN_ACCEPTABLE_DRAFT_LENGTH
                    // len(chapter_plan),
                },
            )

            prompt_tokens = self._cached_count_tokens(prompt, self.model)
            available_for_generation = (
                self.config.MAX_CONTEXT_TOKENS
                - prompt_tokens
                - config.settings.NARRATIVE_TOKEN_BUFFER
            )  # Safety buffer
            max_gen_tokens = min(
                self.config.MAX_GENERATION_TOKENS // 2, available_for_generation
            )

            if max_gen_tokens < 300:
                error_msg = f"Insufficient token space for generating Scene {scene_number} in Ch {chapter_number}. Prompt tokens: {prompt_tokens}."
                logger.error(error_msg)
                return (
                    "\n\n".join(all_scenes_prose) or None,
                    "\n\n---\n\n".join(all_raw_outputs),
                    total_usage_data,
                )

            scene_prose, scene_usage_data = await llm_service.async_call_llm(
                model_name=self.model,
                prompt=prompt,
                temperature=self.config.Temperatures.DRAFTING,
                max_tokens=max_gen_tokens,
                allow_fallback=True,
                stream_to_disk=False,
                frequency_penalty=self.config.FREQUENCY_PENALTY_DRAFTING,
                presence_penalty=self.config.PRESENCE_PENALTY_DRAFTING,
                auto_clean_response=True,
                system_prompt=get_system_prompt("narrative_agent"),
            )

            if not scene_prose or not scene_prose.strip():
                logger.warning(
                    f"Drafting model failed to write prose for Scene {scene_number} of Chapter {chapter_number}. Skipping scene."
                )
                all_raw_outputs.append(
                    f"--- SCENE {scene_number} FAILED TO GENERATE ---"
                )
                continue

            all_scenes_prose.append(scene_prose)
            all_raw_outputs.append(scene_prose)

            if scene_usage_data:
                for key, value in scene_usage_data.items():
                    total_usage_data[key] = total_usage_data.get(key, 0) + value

        final_draft_text = "\n\n".join(all_scenes_prose)
        final_raw_output = "\n\n---\n\n".join(all_raw_outputs)

        if not final_draft_text.strip():
            logger.error(
                f"Drafting failed for Chapter {chapter_number}: no scenes were successfully generated."
            )
            return None, final_raw_output, total_usage_data

        logger.info(
            f"NarrativeAgent: Successfully generated draft for Chapter {chapter_number} from {len(all_scenes_prose)} scenes. Total Length: {len(final_draft_text)} characters."
        )

        return final_draft_text, final_raw_output, total_usage_data

    # Public wrapper to expose draft_chapter functionality for tests and external callers
    async def draft_chapter(
        self,
        plot_outline: dict[str, Any],
        chapter_number: int,
        plot_point_focus: str,
        hybrid_context_for_draft: str,
        chapter_plan: list[SceneDetail] | None,
        *,
        state: NarrativeState | None = None,
    ) -> tuple[str | None, str | None, dict[str, int] | None]:
        """
        Draft a chapter using either a provided scene plan or by drafting the entire chapter at once.

        This is a convenience wrapper around the internal `_draft_chapter` method. It mirrors the
        signature used in tests and orchestrator pathways. If `chapter_plan` is None, it drafts the
        entire chapter in a single call; otherwise, it drafts each scene sequentially.

        Args:
            plot_outline: The plot outline dictionary for the novel.
            chapter_number: The chapter number being drafted.
            plot_point_focus: The primary plot point or focus for this chapter.
            hybrid_context_for_draft: Contextual information combining semantic and KG facts.
            chapter_plan: A list of SceneDetail objects (or dicts) defining the scenes to draft. If None,
                the chapter is drafted in one LLM call.

        Returns:
            A tuple of the draft text, the raw LLM output, and token usage data, or (None, raw_output, usage)
            if drafting fails.
        """
        # Prefer snapshot-provided context if available
        if state and getattr(state, "snapshot", None) and state.snapshot.hybrid_context:
            hybrid_context_for_draft = state.snapshot.hybrid_context

        if chapter_plan is None:
            # When no scene plan is provided, call the LLM once for the whole chapter.
            # Use the same logic as in `_draft_chapter` but without scene iteration.
            novel_title = plot_outline.get("title", "Untitled Novel")
            novel_genre = plot_outline.get("genre", "Unknown Genre")
            prompt = render_prompt(
                "narrative_agent/draft_chapter_from_plot_point.j2",
                {
                    "no_think": self.config.ENABLE_LLM_NO_THINK_DIRECTIVE,
                    "chapter_number": chapter_number,
                    "novel_title": novel_title,
                    "novel_genre": novel_genre,
                    "plot_point_focus": plot_point_focus,
                    "hybrid_context_for_draft": hybrid_context_for_draft,
                    "min_length": self.config.MIN_ACCEPTABLE_DRAFT_LENGTH,
                },
            )
            prompt_tokens = self._cached_count_tokens(prompt, self.model)
            max_gen_tokens = min(
                self.config.MAX_GENERATION_TOKENS,
                self.config.MAX_CONTEXT_TOKENS
                - prompt_tokens
                - config.settings.NARRATIVE_TOKEN_BUFFER // 2,
            )
            chapter_text, usage_data = await llm_service.async_call_llm(
                model_name=self.model,
                prompt=prompt,
                temperature=self.config.Temperatures.DRAFTING,
                max_tokens=max_gen_tokens,
                allow_fallback=True,
                stream_to_disk=False,
                frequency_penalty=self.config.FREQUENCY_PENALTY_DRAFTING,
                presence_penalty=self.config.PRESENCE_PENALTY_DRAFTING,
                auto_clean_response=True,
                system_prompt=get_system_prompt("narrative_agent"),
            )
            # Mirror return structure of `_draft_chapter`
            return chapter_text, chapter_text, usage_data
        # Use the internal scene-by-scene drafting method
        return await self._draft_chapter(
            plot_outline,
            chapter_number,
            plot_point_focus,
            hybrid_context_for_draft,
            chapter_plan,
        )

    async def plan_continuation(
        self, summary_text: str, num_points: int = 5
    ) -> tuple[list[str] | None, dict[str, int] | None]:
        """Generate future plot points from a story summary."""
        if not self.config.ENABLE_AGENTIC_PLANNING:
            logger.info("Agentic planning disabled. Skipping continuation planning.")
            return None, None

        prompt = render_prompt(
            "narrative_agent/plan_continuation.j2",
            {"summary": summary_text, "num_points": num_points},
        )
        cleaned, usage = await llm_service.async_call_llm(
            model_name=self.model,
            prompt=prompt,
            temperature=self.config.Temperatures.PLANNING,
            max_tokens=self.config.MAX_PLANNING_TOKENS,
            allow_fallback=True,
            stream_to_disk=False,
            frequency_penalty=self.config.FREQUENCY_PENALTY_PLANNING,
            presence_penalty=self.config.PRESENCE_PENALTY_PLANNING,
            auto_clean_response=True,
            system_prompt=get_system_prompt("narrative_agent"),
        )
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [str(p) for p in parsed if str(p).strip()], usage
        except json.JSONDecodeError:
            logger.error(f"Failed to parse continuation plan JSON: {cleaned}")
        return None, usage

    async def _check_quality(self, draft_text: str) -> bool:
        """
        Perform initial quality checks on the generated draft.

        Returns:
            True if quality checks pass, False otherwise.
        """
        # Check minimum length
        if len(draft_text.strip()) < self.config.MIN_ACCEPTABLE_DRAFT_LENGTH:
            logger.warning(
                f"Draft too short: {len(draft_text)} characters. Minimum required: {self.config.MIN_ACCEPTABLE_DRAFT_LENGTH}"
            )
            return False

        # Check for common coherence issues
        # Look for repetitive patterns that might indicate poor quality

        sentences = re.split(r"[.!?]+", draft_text)
        sentence_count = len([s.strip() for s in sentences if s.strip()])

        if sentence_count < 5:
            logger.warning(f"Draft has too few sentences: {sentence_count}")
            return False

        # Check for excessive use of passive voice (common in poor quality writing)
        passive_patterns = [
            r"\b(?:was|were|is|are|being|been)\s+\w+ed\b",
            r"\b(?:has been|have been|had been)\s+\w+ed\b",
        ]
        for pattern in passive_patterns:
            if re.search(pattern, draft_text, re.IGNORECASE):
                logger.warning(f"Found potential passive voice: {pattern}")
                # We'll allow some passive voice but flag it
                break

        # Check for excessive use of filler words
        filler_words = ["very", "really", "quite", "rather", "somewhat", "basically"]
        filler_count = sum(
            1 for word in draft_text.lower().split() if word in filler_words
        )

        if filler_count > 5:
            logger.warning(f"Found excessive filler words: {filler_count} instances")
            return False

        # Check for repetition of key phrases
        words = draft_text.lower().split()
        word_freq = {}
        for word in words:
            word = re.sub(r"[^\w]", "", word)
            if len(word) > 3:  # Only count meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1

        repeated_words = [word for word, freq in word_freq.items() if freq > 3]
        if len(repeated_words) > 2:
            logger.warning(f"Found repeated words: {repeated_words}")
            return False

        logger.info("Quality checks passed")
        return True

    
