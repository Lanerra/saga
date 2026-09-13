# core/langgraph/nodes/context_retrieval_node.py
"""Orchestrate scene-specific context retrieval from multiple sources.

This module is the entry point for the context retrieval workflow. It coordinates
character profiles, KG facts, scene events, relationships, items, location context,
previous scenes, and semantic search results into a single hybrid context block.

The retrieval logic for each category is delegated to specialized modules:
- context_character_retrieval: character profile retrieval
- context_world_retrieval: KG facts and world knowledge
- context_plot_retrieval: scene events, relationships, character items, act structure
- context_scene_retrieval: scene items, previous scenes, location, semantic search
"""

import structlog

import config
from core.langgraph.content_manager import (
    ContentManager,
    get_chapter_outlines,
    get_chapter_plan,
    get_previous_summaries,
    get_scene_drafts,
    require_project_dir,
)
from core.langgraph.state import NarrativeState
from core.text_processing_service import count_tokens
from processing.scene_plan_parser import extract_scene_characters

from .context_character_retrieval import get_scene_character_context
from .context_plot_retrieval import (
    get_act_events_context,
    get_character_items_context,
    get_character_relationships_context,
    get_scene_events_context,
)
from .context_scene_retrieval import (
    get_location_context,
    get_previous_scenes_context,
    get_scene_items_context,
    get_semantic_context,
)
from .context_world_retrieval import get_scene_kg_facts

logger = structlog.get_logger(__name__)

DEFAULT_CONTEXT_BUDGET_TOKENS = config.settings.MAX_CONTEXT_TOKENS // 2


async def retrieve_context(state: NarrativeState) -> NarrativeState:
    """Build and externalize hybrid context for the current scene.

    The hybrid context is a single text block assembled from multiple sources and
    truncated to respect a token budget.

    Args:
        state: Workflow state. Requires a valid chapter plan and `current_scene_index`.

    Returns:
        Partial state update containing:
        - hybrid_context_ref: Externalized hybrid context for the current scene.
        - current_node: `"retrieve_context"`.

        On fatal retrieval failures (for example, character profile retrieval or KG
        facts retrieval), returns an update with `has_fatal_error` set and `last_error`
        populated.

    Notes:
        This node performs I/O (Neo4j reads and LLM calls for summarization/semantic
        context) and writes externalized context to disk.
    """
    logger.info("retrieve_context: fetching scene-specific context")

    content_manager = ContentManager(require_project_dir(state))

    chapter_number = state.get("current_chapter", 1)
    scene_index = state.get("current_scene_index", 0)

    chapter_plan = get_chapter_plan(state, content_manager)

    if not chapter_plan or type(scene_index) is not int or scene_index < 0 or scene_index >= len(chapter_plan):
        logger.error("retrieve_context: invalid scene index", index=scene_index)
        return {
            "has_fatal_error": True,
            "last_error": "Invalid scene index for context retrieval",
            "error_node": "retrieve_context",
            "hybrid_context_ref": None,
            "current_node": "retrieve_context",
        }

    current_scene = chapter_plan[scene_index]
    model_name = state.get("narrative_model", config.NARRATIVE_MODEL)

    chapter_outlines = get_chapter_outlines(state, content_manager)

    hybrid_context_parts = []

    # 1. Scene-Specific Character Context
    try:
        character_context = await get_scene_character_context(
            state=state,
            current_scene=current_scene,
            chapter_number=chapter_number,
            model_name=model_name,
            content_manager=content_manager,
        )
        if character_context:
            hybrid_context_parts.append(character_context)
    except Exception as e:
        logger.error(
            "retrieve_context: fatal error getting character profiles",
            error=str(e),
            exc_info=True,
        )
        return {
            "has_fatal_error": True,
            "last_error": f"Failed to retrieve character profiles: {str(e)}",
            "error_node": "retrieve_context",
            "current_node": "retrieve_context",
        }

    # 2. Scene-Specific KG Facts
    try:
        kg_facts_block = await get_scene_kg_facts(
            state=state,
            current_scene=current_scene,
            chapter_number=chapter_number,
            chapter_outlines=chapter_outlines,
            chapter_plan=chapter_plan,
            model_name=model_name,
            content_manager=content_manager,
        )
        if kg_facts_block:
            hybrid_context_parts.append(kg_facts_block)
    except Exception as e:
        logger.error(
            "retrieve_context: fatal error getting KG facts",
            error=str(e),
            exc_info=True,
        )
        return {
            "has_fatal_error": True,
            "last_error": f"Failed to retrieve KG facts: {str(e)}",
            "error_node": "retrieve_context",
            "current_node": "retrieve_context",
        }

    # 3. Scene Events
    scene_events_context = await get_scene_events_context(
        state=state,
        chapter_number=chapter_number,
        scene_index=scene_index,
        content_manager=content_manager,
    )
    if scene_events_context:
        hybrid_context_parts.append(scene_events_context)

    # 4. Character Relationships
    scene_characters = extract_scene_characters(current_scene)
    if scene_characters:
        relationships_context = await get_character_relationships_context(
            state=state,
            character_names=scene_characters,
            chapter_number=chapter_number,
            content_manager=content_manager,
        )
        if relationships_context:
            hybrid_context_parts.append(relationships_context)

    # 5. Character Items
    if scene_characters:
        character_items_context = await get_character_items_context(
            state=state,
            character_names=scene_characters,
            chapter_number=chapter_number,
            content_manager=content_manager,
        )
        if character_items_context:
            hybrid_context_parts.append(character_items_context)

    # 6. Scene Items
    scene_items_context = await get_scene_items_context(
        state=state,
        chapter_number=chapter_number,
        scene_index=scene_index,
        content_manager=content_manager,
    )
    if scene_items_context:
        hybrid_context_parts.append(scene_items_context)

    # 7. Act Events (Plot Structure)
    act_number = chapter_outlines.get(chapter_number, {}).get("act_number")
    if act_number:
        act_events_context = await get_act_events_context(
            state=state,
            act_number=act_number,
            content_manager=content_manager,
        )
        if act_events_context:
            hybrid_context_parts.append(act_events_context)

    # 8. Previous Chapter Summaries
    previous_summaries = get_previous_summaries(state, content_manager)
    if previous_summaries:
        summaries_text = "\n\n**Recent Chapter Summaries:**\n"
        for summary in previous_summaries[-3:]:
            summaries_text += f"\n{summary}"
        hybrid_context_parts.append(summaries_text)

    # 9. Previous Scenes in This Chapter (Token-Aware)
    scene_drafts = get_scene_drafts(state, content_manager)
    if scene_drafts:
        previous_scenes_context = await get_previous_scenes_context(
            state=state,
            scene_drafts=scene_drafts,
            chapter_plan=chapter_plan,
            scene_index=scene_index,
            model_name=model_name,
            extraction_model=state.get("small_model", model_name),
            content_manager=content_manager,
        )
        if previous_scenes_context:
            hybrid_context_parts.append(previous_scenes_context)

    # 10. Location Context (if specified in scene)
    location_context = await get_location_context(
        state=state,
        current_scene=current_scene,
        chapter_number=chapter_number,
        content_manager=content_manager,
    )
    if location_context:
        hybrid_context_parts.append(location_context)

    # 11. Semantic Context (Vector Search)
    scene_query = (
        f"{current_scene.get('title', '')} "
        f"{current_scene.get('plot_point', '')} "
        f"{current_scene.get('conflict', '')} "
        f"{current_scene.get('setting', '')}"
    )
    semantic_context = await get_semantic_context(
        state=state,
        query_text=scene_query,
        chapter_number=chapter_number,
        model_name=model_name,
        content_manager=content_manager,
    )
    if semantic_context:
        hybrid_context_parts.append(semantic_context)

    hybrid_context = "\n\n".join(hybrid_context_parts)

    context_tokens = count_tokens(hybrid_context, model_name)
    logger.info(
        "retrieve_context: context built",
        scene_index=scene_index,
        context_length_chars=len(hybrid_context),
        context_length_tokens=context_tokens,
        components=len(hybrid_context_parts),
    )

    identifier = f"chapter_{chapter_number}_scene_{scene_index}"
    version = content_manager.get_latest_version("hybrid_context", identifier) + 1

    hybrid_context_ref = content_manager.save_text(
        hybrid_context,
        "hybrid_context",
        identifier,
        version=version,
    )

    return {
        "hybrid_context_ref": hybrid_context_ref,
        "current_node": "retrieve_context",
    }


__all__ = ["retrieve_context"]
