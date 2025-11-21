# core/langgraph/nodes/scene_generation_node.py
import structlog

from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)


async def draft_scene(state: NarrativeState) -> NarrativeState:
    """
    Draft a single scene.
    """
    logger.info("draft_scene: generating text")

    chapter_number = state["current_chapter"]
    scene_index = state["current_scene_index"]
    chapter_plan = state.get("chapter_plan")

    if not chapter_plan or scene_index >= len(chapter_plan):
        logger.error("draft_scene: invalid scene index", index=scene_index)
        return state

    current_scene = chapter_plan[scene_index]

    # Calculate target word count for this scene
    total_target = 3000  # Default chapter length
    if state.get("target_word_count") and state.get("total_chapters"):
        total_target = state["target_word_count"] // state["total_chapters"]

    scene_target = total_target // len(chapter_plan)

    prompt = render_prompt(
        "narrative_agent/draft_scene.j2",
        {
            "chapter_number": chapter_number,
            "novel_title": state["title"],
            "novel_genre": state["genre"],
            "novel_theme": state["theme"],
            "scene": current_scene,
            "hybrid_context": state.get("hybrid_context", ""),
            "target_word_count": scene_target,
        },
    )

    try:
        draft_text, _ = await llm_service.async_call_llm(
            model_name=state["narrative_model"],
            prompt=prompt,
            temperature=0.7,
            max_tokens=4000,  # Allow enough for a full scene
            system_prompt=get_system_prompt("narrative_agent"),
        )

        # Update state
        current_drafts = state.get("scene_drafts", [])
        new_drafts = current_drafts + [draft_text]

        return {
            "scene_drafts": new_drafts,
            "current_scene_index": scene_index + 1,  # Increment for next loop
            "current_node": "draft_scene",
        }

    except Exception as e:
        logger.error("draft_scene: error generating scene", error=str(e))
        return {
            **state,
            "last_error": f"Error generating scene: {str(e)}",
            "current_node": "draft_scene",
        }
