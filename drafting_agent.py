# drafting_agent.py
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import config
from llm_interface import count_tokens, llm_service
from prompt_renderer import render_prompt
from kg_maintainer.models import CharacterProfile, SceneDetail, WorldItem
from utils import format_scene_plan_for_prompt

logger = logging.getLogger(__name__)


class DraftingAgent:
    def __init__(self, model_name: str = config.DRAFTING_MODEL):
        self.model_name = model_name
        logger.info(f"DraftingAgent initialized with model: {self.model_name}")

    async def draft_chapter(
        self,
        plot_outline: Dict[str, Any],
        character_profiles: Dict[str, CharacterProfile],
        world_building: Dict[str, Dict[str, WorldItem]],
        chapter_number: int,
        plot_point_focus: str,
        hybrid_context_for_draft: str,
        chapter_plan: Optional[List[SceneDetail]],
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, int]]]:
        """
        Generates the initial draft for a chapter based on plot focus, context, and scene plan.
        Returns: (draft_text, raw_llm_output, usage_data)
        """
        logger.info(f"DraftingAgent: Generating draft for Chapter {chapter_number}...")

        protagonist_name = plot_outline.get(
            "protagonist_name", config.DEFAULT_PROTAGONIST_NAME
        )
        novel_title = plot_outline.get("title", "Untitled Novel")
        novel_genre = plot_outline.get("genre", "Unknown Genre")
        novel_theme = plot_outline.get("theme", "Unknown Theme")

        # Prepare the scene plan part of the prompt
        # Max tokens for scene plan in prompt can be a fraction of total context, e.g., 1/4th
        max_tokens_for_plan_prompt = config.MAX_CONTEXT_TOKENS // 4
        scene_plan_prompt_text = (
            "No detailed scene plan available. Focus on the plot point."
        )
        if chapter_plan and config.ENABLE_AGENTIC_PLANNING:
            scene_plan_prompt_text = format_scene_plan_for_prompt(
                chapter_plan, self.model_name, max_tokens_for_plan_prompt
            )
        elif not config.ENABLE_AGENTIC_PLANNING:
            scene_plan_prompt_text = "Agentic scene planning is disabled. Focus on the plot point for this chapter."

        prompt = render_prompt(
            "drafting_agent/draft_chapter.j2",
            {
                "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
                "chapter_number": chapter_number,
                "novel_title": novel_title,
                "novel_genre": novel_genre,
                "novel_theme": novel_theme,
                "protagonist_name": protagonist_name,
                "plot_point_focus": plot_point_focus,
                "scene_plan_prompt_text": scene_plan_prompt_text,
                "hybrid_context_for_draft": hybrid_context_for_draft,
                "min_length": config.MIN_ACCEPTABLE_DRAFT_LENGTH,
            },
        )

        # Estimate prompt tokens and adjust max_generation_tokens if necessary
        # This is a rough way to ensure the prompt itself doesn't consume the entire window
        # A more sophisticated approach would be needed for very large context + small total model window
        prompt_tokens = count_tokens(prompt, self.model_name)
        available_for_generation = (
            config.MAX_CONTEXT_TOKENS - prompt_tokens - 200
        )  # 200 as buffer

        # Ensure max_generation_tokens is positive and not excessively large
        # MAX_GENERATION_TOKENS is an upper cap defined in config
        max_gen_tokens = config.MAX_GENERATION_TOKENS
        if available_for_generation < max_gen_tokens:
            if available_for_generation > 500:  # Min reasonable generation
                max_gen_tokens = available_for_generation
                logger.info(
                    f"Drafting Ch {chapter_number}: Adjusted max_tokens for generation to {max_gen_tokens} due to prompt size."
                )
            else:
                logger.error(
                    f"Drafting Ch {chapter_number}: Insufficient token space for generation after prompt. Prompt tokens: {prompt_tokens}, Model max context: {config.MAX_CONTEXT_TOKENS}. Cannot draft."
                )
                return None, "Prompt too large for generation window.", None

        logger.info(
            f"Calling LLM ({self.model_name}) to draft Chapter {chapter_number}. Est. prompt tokens: {prompt_tokens}. Max generation tokens: {max_gen_tokens}."
        )
        draft_text, usage_data = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=config.Temperatures.DRAFTING,
            max_tokens=max_gen_tokens,  # Use potentially adjusted max_gen_tokens
            allow_fallback=True,  # Allow fallback for critical drafting step
            stream_to_disk=True,  # Streaming is good for long generations
            frequency_penalty=config.FREQUENCY_PENALTY_DRAFTING,
            presence_penalty=config.PRESENCE_PENALTY_DRAFTING,
            auto_clean_response=True,  # Clean preamble/postamble from LLM
        )

        chapter_text = ""
        if draft_text.strip():
            try:
                parsed = json.loads(draft_text)
                chapter_text = (
                    parsed.get("chapter_text", "") if isinstance(parsed, dict) else ""
                )
            except json.JSONDecodeError:
                logger.error(
                    f"DraftingAgent: Failed to parse JSON chapter text for Chapter {chapter_number}."
                )
                chapter_text = draft_text.strip()
        if not chapter_text:
            logger.error(
                f"DraftingAgent: LLM returned empty or whitespace-only draft for Chapter {chapter_number}."
            )
            return (
                None,
                draft_text,
                usage_data,
            )  # Return raw LLM output even if it's bad

        logger.info(
            f"DraftingAgent: Successfully generated draft for Chapter {chapter_number}. Length: {len(chapter_text)} characters."
        )
        return (
            chapter_text,
            draft_text,
            usage_data,
        )
