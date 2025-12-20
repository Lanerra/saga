# core/langgraph/nodes/scene_generation_node.py
import structlog

from core.langgraph.content_manager import (
    ContentManager,
    get_chapter_plan,
    get_hybrid_context,
    get_scene_drafts,
)
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)


async def draft_scene(state: NarrativeState) -> NarrativeState:
    """
    Draft a single scene.
    """
    logger.info("draft_scene: generating text")

    # Initialize content manager
    content_manager = ContentManager(state.get("project_dir", ""))

    chapter_number = state.get("current_chapter", 1)
    scene_index = state.get("current_scene_index", 0)

    # Get chapter plan from externalized content
    chapter_plan = get_chapter_plan(state, content_manager)

    if not chapter_plan or scene_index >= len(chapter_plan):
        logger.error("draft_scene: invalid scene index", index=scene_index)
        return state

    current_scene = chapter_plan[scene_index]

    # Calculate target word count for this scene
    total_target = 3000  # Default chapter length
    if state.get("target_word_count") and state.get("total_chapters"):
        total_target = state.get("target_word_count", 0) // state.get("total_chapters", 0)

    scene_target = total_target // len(chapter_plan)

    # Get hybrid context from content manager
    hybrid_context = get_hybrid_context(state, content_manager) or ""

    prompt = render_prompt(
        "narrative_agent/draft_scene.j2",
        {
            "chapter_number": chapter_number,
            "novel_title": state.get("title", ""),
            "novel_genre": state.get("genre", ""),
            "novel_theme": state.get("theme", ""),
            "scene": current_scene,
            "hybrid_context": hybrid_context,
            "target_word_count": scene_target,
        },
    )

    try:
        draft_text, _ = await llm_service.async_call_llm(
            model_name=state.get("narrative_model", ""),
            prompt=prompt,
            temperature=0.7,
            max_tokens=16384,  # Allow enough for a full scene
            system_prompt=get_system_prompt("narrative_agent"),
        )

        # Update state
        current_drafts = get_scene_drafts(state, content_manager)
        new_drafts = current_drafts + [draft_text]

        # Save updated scene drafts
        current_version = content_manager.get_latest_version("scenes", f"chapter_{chapter_number}") + 1
        scene_drafts_ref = content_manager.save_list_of_texts(
            new_drafts,
            "scenes",
            f"chapter_{chapter_number}",
            version=current_version,
        )

        return {
            "scene_drafts_ref": scene_drafts_ref,
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
