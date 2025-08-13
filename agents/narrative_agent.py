from typing import Dict, List, Optional, Tuple, Any
import json  # Added for JSON parsing
import re  # Added for regex operations
import structlog
from config import NARRATIVE_MODEL
from core.llm_interface import llm_service, count_tokens, truncate_text_by_tokens
from data_access import chapter_queries
from kg_maintainer.models import CharacterProfile, SceneDetail, WorldItem
from prompt_data_getters import (
    get_character_state_snippet_for_prompt,
    get_reliable_kg_facts_for_drafting_prompt,
    get_world_state_snippet_for_prompt,
)
from prompt_renderer import render_prompt

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
    def __init__(self, config: Dict):
        self.logger = structlog.get_logger()
        self.config = config
        self.model = NARRATIVE_MODEL

    async def _plan_chapter_scenes(
        self,
        plot_outline: Dict[str, Any],
        character_profiles: Dict[str, CharacterProfile],
        world_building: Dict[str, Dict[str, WorldItem]],
        chapter_number: int,
        plot_point_focus: Optional[str],
        plot_point_index: int,
    ) -> Tuple[Optional[List[SceneDetail]], Optional[Dict[str, int]]]:
        """
        Generates a detailed scene plan for the chapter.
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

        context_summary_parts: List[str] = []
        if chapter_number > 1:
            prev_chap_data = await chapter_queries.get_chapter_data_from_db(
                chapter_number - 1
            )
            if prev_chap_data:
                prev_summary = prev_chap_data.get("summary")
                prev_is_provisional = prev_chap_data.get("is_provisional", False)
                summary_prefix = (
                    "[Provisional Summary from Prev Ch] "
                    if prev_is_provisional and prev_summary
                    else "[Summary from Prev Ch] "
                )
                if prev_summary:
                    context_summary_parts.append(
                        f"{summary_prefix}({chapter_number - 1}):\n{prev_summary[:1000].strip()}...\n"
                    )
                else:
                    prev_text = prev_chap_data.get("text", "")
                    text_prefix = (
                        "[Provisional Text Snippet from Prev Ch] "
                        if prev_is_provisional and prev_text
                        else "[Text Snippet from Prev Ch] "
                    )
                    if prev_text:
                        context_summary_parts.append(
                            f"{text_prefix}({chapter_number - 1}):\n...{prev_text[-1000:].strip()}\n"
                        )

        context_summary_str = "".join(context_summary_parts)

        protagonist_name = plot_outline.get(
            "protagonist_name", self.config.DEFAULT_PROTAGONIST_NAME
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

        future_plot_context_parts: List[str] = []
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

        few_shot_scene_plan_example_str = """
**Ignore the narrative details in this example. It shows the required format only.**
[
  {
    "scene_number": 1,
    "summary": "Elara arrives at the Sunken Library, finding its entrance hidden and guarded by an ancient riddle.",
    "characters_involved": ["Elara Vance"],
    "key_dialogue_points": [
      "Elara (internal): \\"This riddle... it speaks of starlight and shadow. What reflects both?\\"",
      "Elara (to herself, solving): \\"The water! The entrance must be beneath the lake's surface.\\""
    ],
    "setting_details": "A mist-shrouded, unnaturally still lake. Crumbling, moss-covered ruins of a tower are visible on a small island.",
    "scene_focus_elements": ["Elara's deductive reasoning", "Building atmosphere of mystery and ancient magic"],
    "contribution": "Introduces the challenge of accessing the Sunken Library and showcases Elara's intellect.",
    "scene_type": "ATMOSPHERE_BUILDING",
    "pacing": "SLOW",
    "character_arc_focus": "Establishes Elara's scholarly and determined nature when faced with a puzzle.",
    "relationship_development": null
  },
  {
    "scene_number": 2,
    "summary": "Elara meets Master Kael, the library's ancient archivist, who tests her worthiness before revealing information about the Starfall Map.",
    "characters_involved": ["Elara Vance", "Master Kael"],
    "key_dialogue_points": [
      "Kael: \\"Many seek what is lost. Few understand its price. Why do you search, child of the shifting stars?\\"",
      "Elara: \\"I seek knowledge not for power, but to mend what was broken.\\""
    ],
    "setting_details": "Inside the Sunken Library: vast, circular, dimly lit by glowing runes and bioluminescent moss.",
    "scene_focus_elements": ["The cryptic nature and wisdom of Master Kael", "The initial reveal of a clue for the Starfall Map"],
    "contribution": "Elara gains a crucial piece of information and a potential ally/gatekeeper in Kael, advancing the plot.",
    "scene_type": "DIALOGUE",
    "pacing": "MEDIUM",
    "character_arc_focus": "Elara must articulate her noble motivations, reinforcing her core identity.",
    "relationship_development": "The relationship between Elara and Kael is established as one of a student and a gatekeeper/mentor."
  },
  {
    "scene_number": 3,
    "summary": "As Elara leaves, she is ambushed by rival Seekers who try to steal the clue from her. She uses her wits and a minor magical artifact to escape.",
    "characters_involved": ["Elara Vance", "Rival Seeker (Thane)"],
    "key_dialogue_points": [
      "Thane: \\"The old man was a fool to trust you. The map belongs to the Crimson Hand!\\"",
      "Elara (activating artifact): \\"It belongs to those who would protect it!\\""
    ],
    "setting_details": "The narrow, crumbling causeway leading away from the library island.",
    "scene_focus_elements": ["Sudden danger and threat", "Elara's quick thinking under pressure", "First use of her 'Silvershard' artifact"],
    "contribution": "Introduces the antagonist faction and demonstrates that Elara is capable of defending herself.",
    "scene_type": "ACTION",
    "pacing": "FAST",
    "character_arc_focus": "Elara is forced from a purely intellectual challenge to a physical one, showing her resilience.",
    "relationship_development": "An antagonistic relationship with Thane and the Crimson Hand is established."
  }
]
"""
        prompt = render_prompt(
            "planner_agent/scene_plan.j2",
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
                "few_shot_scene_plan_example_str": few_shot_scene_plan_example_str,
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
        )

        parsed_scenes_list_of_dicts = self._parse_llm_scene_plan_output(
            cleaned_plan_text_from_llm, chapter_number
        )

        if parsed_scenes_list_of_dicts:
            final_scenes_typed: List[SceneDetail] = []
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
            else:
                logger.error(
                    f"Parsed list was empty or all scenes were invalid after parsing for chapter {chapter_number}. Cleaned LLM output: '{cleaned_plan_text_from_llm[:500]}...'"
                )
                return None, usage_data
        else:
            logger.error(
                f"Failed to parse a valid list of scenes for chapter {chapter_number}. Cleaned LLM output: '{cleaned_plan_text_from_llm[:500]}...'"
            )
            return None, usage_data

    def _parse_llm_scene_plan_output(
        self, json_text: str, chapter_number: int
    ) -> Optional[List[SceneDetail]]:
        """
        Parses JSON scene plan output from LLM.
        Expects a JSON array of scene objects.
        """
        if not json_text or not json_text.strip():
            logger.warning(
                f"JSON scene plan for Ch {chapter_number} is empty. No scenes parsed."
            )
            return None

        try:
            parsed_data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode JSON scene plan for Ch {chapter_number}: {e}. Text: {json_text[:500]}..."
            )
            match = re.search(r"\[\s*\{.*\}\s*\]", json_text, re.DOTALL)
            if match:
                logger.info(
                    "Found a JSON array within the malformed JSON string. Attempting to parse that."
                )
                try:
                    parsed_data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    logger.error(
                        f"Still failed to parse extracted JSON array for Ch {chapter_number}."
                    )
                    return None
            else:
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

        scenes_data: List[SceneDetail] = []
        for i, scene_item in enumerate(parsed_data):
            if not isinstance(scene_item, dict):
                logger.warning(
                    f"Scene item {i + 1} in Ch {chapter_number} is not a dictionary. Skipping. Item: {str(scene_item)[:100]}"
                )
                continue

            processed_scene_dict: Dict[str, Any] = {}
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

    async def _draft_chapter(
        self,
        plot_outline: Dict[str, Any],
        chapter_number: int,
        plot_point_focus: str,
        hybrid_context_for_draft: str,
        chapter_plan: List[SceneDetail],
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, int]]]:
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

        all_scenes_prose: List[str] = []
        all_raw_outputs: List[str] = []
        total_usage_data: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        novel_title = plot_outline.get("title", "Untitled Novel")
        novel_genre = plot_outline.get("genre", "Unknown Genre")

        for scene_index, scene_detail in enumerate(chapter_plan):
            scene_number = scene_detail.get("scene_number", scene_index + 1)
            logger.info(
                f"Drafting Scene {scene_number} of Chapter {chapter_number}..."
            )

            previous_scenes_prose = "\n\n".join(all_scenes_prose)
            max_tokens_for_prev_scenes = self.config.MAX_GENERATION_TOKENS // 2
            previous_scenes_prose_for_prompt = truncate_text_by_tokens(
                previous_scenes_prose,
                self.model,
                max_tokens_for_prev_scenes,
                truncation_marker="\n\n... (prose from earlier scenes in this chapter has been truncated)\n\n",
            )

            prompt = render_prompt(
                "drafting_agent/draft_scene.j2",
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

            prompt_tokens = count_tokens(prompt, self.model)
            available_for_generation = (
                self.config.MAX_CONTEXT_TOKENS - prompt_tokens - 200
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

    async def plan_continuation(
        self,
        summary_text: str,
        num_points: int = 5
    ) -> Tuple[Optional[List[str]], Optional[Dict[str, int]]]:
        """Generate future plot points from a story summary."""
        if not self.config.ENABLE_AGENTIC_PLANNING:
            logger.info("Agentic planning disabled. Skipping continuation planning.")
            return None, None

        prompt = render_prompt(
            "planner_agent/plan_continuation.j2",
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
        )
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [str(p) for p in parsed if str(p).strip()], usage
        except json.JSONDecodeError:
            logger.error("Failed to parse continuation plan JSON: %s", cleaned)
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
        import re
        sentences = re.split(r'[.!?]+', draft_text)
        sentence_count = len([s.strip() for s in sentences if s.strip()])
        
        if sentence_count < 5:
            logger.warning(f"Draft has too few sentences: {sentence_count}")
            return False

        # Check for excessive use of passive voice (common in poor quality writing)
        passive_patterns = [
            r'\b(?:was|were|is|are|being|been)\s+\w+ed\b',
            r'\b(?:has been|have been|had been)\s+\w+ed\b'
        ]
        for pattern in passive_patterns:
            if re.search(pattern, draft_text, re.IGNORECASE):
                logger.warning(f"Found potential passive voice: {pattern}")
                # We'll allow some passive voice but flag it
                break

        # Check for excessive use of filler words
        filler_words = ['very', 'really', 'quite', 'rather', 'somewhat', 'basically']
        filler_count = sum(1 for word in draft_text.lower().split() if word in filler_words)
        
        if filler_count > 5:
            logger.warning(f"Found excessive filler words: {filler_count} instances")
            return False

        # Check for repetition of key phrases
        words = draft_text.lower().split()
        word_freq = {}
        for word in words:
            word = re.sub(r'[^\w]', '', word)
            if len(word) > 3:  # Only count meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        repeated_words = [word for word, freq in word_freq.items() if freq > 3]
        if len(repeated_words) > 2:
            logger.warning(f"Found repeated words: {repeated_words}")
            return False

        logger.info("Quality checks passed")
        return True

    async def generate_chapter(
        self,
        plot_outline: Dict,
        character_profiles: Dict,
        world_building: Dict,
        chapter_number: int,
        plot_point_focus: str
    ) -> Tuple[str, Dict]:
        """Generate a new chapter based on plot outline and
        character/world context."""
        self.logger.info("Generating chapter", chapter=chapter_number)
        
        # Get the current plot point index
        all_plot_points = plot_outline.get("plot_points", [])
        plot_point_index = -1
        for i, pp in enumerate(all_plot_points):
            if isinstance(pp, str) and pp.strip() == plot_point_focus:
                plot_point_index = i
                break
                
        if plot_point_index == -1:
            logger.error(f"Could not find plot point focus: {plot_point_focus}")
            return "Failed to generate chapter: plot point focus not found", {"summary": "Error"}

        # Plan the chapter scenes
        scenes, usage_data = await self._plan_chapter_scenes(
            plot_outline,
            character_profiles,
            world_building,
            chapter_number,
            plot_point_focus,
            plot_point_index
        )
        
        if not scenes:
            logger.error(f"Failed to plan scenes for chapter {chapter_number}")
            return "Failed to generate chapter: scene planning failed", {"summary": "Error"}

        # Generate hybrid context for drafting
        hybrid_context_for_draft = (
            f"Protagonist: {plot_outline.get('protagonist_name', 'Unknown')}\n"
            f"Genre: {plot_outline.get('genre', 'Unknown')}\n"
            f"Theme: {plot_outline.get('theme', 'Unknown')}\n"
            f"Character Arc: {plot_outline.get('character_arc', 'Unknown')}"
        )

        # Draft the chapter
        draft_text, raw_output, draft_usage = await self._draft_chapter(
            plot_outline,
            chapter_number,
            plot_point_focus,
            hybrid_context_for_draft,
            scenes
        )
        
        if not draft_text:
            logger.error(f"Failed to draft chapter {chapter_number}")
            return "Failed to generate chapter: drafting failed", {"summary": "Error"}

        # Perform quality checks
        if not await self._check_quality(draft_text):
            logger.warning(f"Chapter {chapter_number} failed quality checks")
            # Return the draft anyway but mark it as low quality
            return draft_text, {"summary": "Low quality - needs revision"}

        return draft_text, {"summary": "Chapter generated successfully"}
