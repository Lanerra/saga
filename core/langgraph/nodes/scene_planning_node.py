# core/langgraph/nodes/scene_planning_node.py
import json

import structlog

from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)


async def plan_scenes(state: NarrativeState) -> NarrativeState:
    """
    Break the chapter into scenes based on the outline.
    """
    logger.info(
        "plan_scenes: planning scenes for chapter", chapter=state["current_chapter"]
    )

    chapter_number = state["current_chapter"]
    chapter_outlines = state.get("chapter_outlines", {})
    outline = chapter_outlines.get(chapter_number)

    if not outline:
        logger.error(
            "plan_scenes: no outline found for chapter", chapter=chapter_number
        )
        return {
            **state,
            "last_error": f"No outline found for chapter {chapter_number}",
            "current_node": "plan_scenes",
        }

    # Determine number of scenes (heuristic or config)
    # For now, we'll ask for 3-5 scenes depending on complexity, or just default to 3
    num_scenes = 3

    prompt = render_prompt(
        "narrative_agent/plan_scenes.j2",
        {
            "novel_title": state["title"],
            "novel_genre": state["genre"],
            "novel_theme": state["theme"],
            "chapter_number": chapter_number,
            "outline": outline,
            "num_scenes": num_scenes,
        },
    )

    try:
        response, _ = await llm_service.async_call_llm(
            model_name=state["generation_model"],
            prompt=prompt,
            temperature=0.7,
            max_tokens=2000,
            system_prompt=get_system_prompt("narrative_agent"),
        )

        # Parse JSON response
        # The LLM might wrap it in markdown code blocks, so we need to clean it
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]

        scenes = json.loads(cleaned_response)

        if not isinstance(scenes, list):
            raise ValueError("LLM response is not a list of scenes")

        logger.info("plan_scenes: successfully planned scenes", count=len(scenes))

        return {
            "chapter_plan": scenes,
            "current_scene_index": 0,
            "scene_drafts": [],
            "current_node": "plan_scenes",
        }

    except Exception as e:
        logger.error("plan_scenes: error planning scenes", error=str(e))
        return {
            **state,
            "last_error": f"Error planning scenes: {str(e)}",
            "current_node": "plan_scenes",
        }
