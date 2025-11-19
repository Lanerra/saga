# core/langgraph/nodes/context_retrieval_node.py
import structlog

from core.langgraph.state import NarrativeState
from prompts.prompt_data_getters import get_reliable_kg_facts_for_drafting_prompt

logger = structlog.get_logger(__name__)


async def retrieve_context(state: NarrativeState) -> NarrativeState:
    """
    Retrieve context for the current scene/chapter.
    """
    logger.info("retrieve_context: fetching context")

    chapter_number = state["current_chapter"]
    scene_index = state["current_scene_index"]
    chapter_plan = state.get("chapter_plan")

    if not chapter_plan or scene_index >= len(chapter_plan):
        logger.error("retrieve_context: invalid scene index", index=scene_index)
        return state

    current_scene = chapter_plan[scene_index]

    # In a full implementation, we would query Neo4j for the specific characters
    # and location mentioned in current_scene.
    # For now, we'll reuse the chapter-level context getter but ideally filter it.

    # TODO: Refine this to be scene-specific using current_scene['characters']
    kg_facts_block = await get_reliable_kg_facts_for_drafting_prompt(
        state.get(
            "plot_outline"
        ),  # Fallback if needed, but ideally we use chapter_outlines
        chapter_number,
        None,  # No specific focus character for the whole chapter, but we have one for the scene
    )

    # Build hybrid context string
    hybrid_context_parts = []

    if kg_facts_block:
        hybrid_context_parts.append(kg_facts_block)

    # Add previous chapter summaries
    if state.get("previous_chapter_summaries"):
        summaries_text = "\n\n**Recent Chapter Summaries:**\n"
        for summary in state["previous_chapter_summaries"][-3:]:
            summaries_text += f"\n{summary}"
        hybrid_context_parts.append(summaries_text)

    # Add context from previous scenes in THIS chapter
    if state.get("scene_drafts"):
        previous_scenes_text = "\n\n**Previous Scenes in This Chapter:**\n"
        for i, draft in enumerate(state["scene_drafts"]):
            scene_title = chapter_plan[i].get("title", f"Scene {i+1}")
            # Summarize or truncate if too long? For now, just last few paragraphs might be better
            # but let's put the whole thing if it fits context window.
            # To be safe, let's just take the last 500 chars of the previous scene.
            previous_scenes_text += f"\n--- {scene_title} ---\n...{draft[-1000:]}\n"
        hybrid_context_parts.append(previous_scenes_text)

    hybrid_context = "\n\n".join(hybrid_context_parts)

    return {
        "hybrid_context": hybrid_context,
        "current_node": "retrieve_context",
    }
