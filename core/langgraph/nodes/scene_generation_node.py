# core/langgraph/nodes/scene_generation_node.py
"""Draft individual scenes for a chapter.

This module defines the scene drafting node used by the scene-based generation
workflow. Each call drafts one scene, appends it to the chapter's scene drafts,
and advances the `current_scene_index` for the drafting loop.
"""

import structlog

import config
from core.langgraph.content_manager import (
    ContentManager,
    get_chapter_plan,
    get_hybrid_context,
    get_scene_drafts,
    require_project_dir,
)
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)


async def draft_scene(state: NarrativeState) -> NarrativeState:
    """Draft the next scene and append it to the chapter's scene drafts.

    Args:
        state: Workflow state. Requires a valid chapter plan and `current_scene_index`.
            Reads `hybrid_context` (preferring `hybrid_context_ref`) to condition the
            draft.

    Returns:
        Partial state update containing:
        - scene_drafts_ref: Externalized list of drafted scene texts so far.
        - current_scene_index: Incremented for the next drafting iteration.
        - current_node: `"draft_scene"`.

        If the scene index is invalid, returns only `current_node`.
        On drafting errors, returns an update with `last_error` populated.

    Notes:
        This node performs LLM I/O and filesystem I/O (externalizing scene drafts).
    """
    logger.info("draft_scene: generating text")

    # Initialize content manager
    content_manager = ContentManager(require_project_dir(state))

    chapter_number = state.get("current_chapter", 1)
    scene_index = state.get("current_scene_index", 0)

    # Get chapter plan from externalized content
    chapter_plan = get_chapter_plan(state, content_manager)

    if not chapter_plan or scene_index >= len(chapter_plan):
        logger.error("draft_scene: invalid scene index", index=scene_index)
        return {"current_node": "draft_scene"}

    current_scene = chapter_plan[scene_index]

    # Inject scene_number for template access (scenes are 1-indexed)
    current_scene["scene_number"] = scene_index + 1

    # Calculate target word count for this scene
    total_target = 3000  # Default chapter length
    if state.get("target_word_count") and state.get("total_chapters"):
        total_target = state.get("target_word_count", 0) // state.get("total_chapters", 0)

    scene_target = total_target // len(chapter_plan)

    # Get hybrid context from content manager
    hybrid_context = get_hybrid_context(state, content_manager) or ""

    revision_guidance_text = ""
    revision_guidance_ref = state.get("revision_guidance_ref")
    if revision_guidance_ref:
        revision_guidance_text = content_manager.load_text(revision_guidance_ref)

    # Collect outcomes from previous scenes in this chapter to prevent re-dramatization
    previous_scenes = []
    if scene_index > 0:
        for i in range(scene_index):
            prev_scene_plan = chapter_plan[i]
            previous_scenes.append(
                {
                    "scene_number": i + 1,
                    "outcome": prev_scene_plan.get("outcome", "Scene completed."),
                }
            )

    prompt = render_prompt(
        "narrative_agent/draft_scene.j2",
        {
            "chapter_number": chapter_number,
            "novel_title": state.get("title", ""),
            "novel_genre": state.get("genre", ""),
            "novel_theme": state.get("theme", ""),
            "narrative_style": state.get("narrative_style", config.DEFAULT_NARRATIVE_STYLE),
            "total_scenes": len(chapter_plan),  # Total scenes in this chapter
            "previous_scenes": previous_scenes,
            "scene": current_scene,
            "hybrid_context": hybrid_context,
            "revision_guidance": revision_guidance_text,
            "target_word_count": scene_target,
        },
    )

    try:
        draft_text, _ = await llm_service.async_call_llm(
            model_name=state.get("narrative_model", config.NARRATIVE_MODEL),
            prompt=prompt,
            temperature=0.7,
            max_tokens=config.MAX_GENERATION_TOKENS,
            system_prompt=get_system_prompt("narrative_agent"),
        )

        draft_word_count = len(draft_text.split())
        if draft_word_count > scene_target * 2:
            logger.warning(
                "draft_scene: scene output exceeded target",
                scene_index=scene_index,
                target_words=scene_target,
                actual_words=draft_word_count,
                ratio=round(draft_word_count / scene_target, 1) if scene_target > 0 else 0,
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
            "last_error": f"Error generating scene: {str(e)}",
            "current_node": "draft_scene",
        }
