# core/langgraph/nodes/context_retrieval_node.py
"""Retrieve scene-specific context for scene-based drafting.

This module defines the context retrieval node used by the scene-based generation
workflow. It builds a token-budgeted "hybrid context" composed of character profiles,
knowledge-graph facts, recent chapter summaries, prior scene context, location context,
and semantic search results.
"""

from typing import Any, cast

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
from core.llm_interface_refactored import llm_service
from core.text_processing_service import count_tokens, truncate_text_by_tokens
from data_access import chapter_queries, kg_queries, scene_queries
from models.agent_models import SceneDetail
from prompts.prompt_data_getters import (
    get_filtered_character_profiles_for_prompt_plain_text,
    get_reliable_kg_facts_for_drafting_prompt,
)
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)

# Context budget configuration (in tokens)
# These can be tuned based on model context window
DEFAULT_CONTEXT_BUDGET_TOKENS = config.settings.MAX_CONTEXT_TOKENS // 2  # Reserve half for generation
PREVIOUS_SCENES_TOKEN_BUDGET = config.MAX_GENERATION_TOKENS  # Max tokens for previous scene context
SUMMARY_MAX_TOKENS = config.MAX_GENERATION_TOKENS  # Target tokens per scene summary
CHARACTER_PROFILES_TOKEN_BUDGET = config.MAX_GENERATION_TOKENS  # Max tokens for character profiles
KG_FACTS_TOKEN_BUDGET = config.MAX_GENERATION_TOKENS  # Max tokens for KG facts
SEMANTIC_CONTEXT_TOKEN_BUDGET = config.MAX_GENERATION_TOKENS  # Max tokens for semantic search results


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

    # Initialize content manager for reading externalized content
    content_manager = ContentManager(require_project_dir(state))

    chapter_number = state.get("current_chapter", 1)
    scene_index = state.get("current_scene_index", 0)

    # Get chapter plan from externalized content
    chapter_plan = get_chapter_plan(state, content_manager)

    if not chapter_plan or scene_index >= len(chapter_plan):
        logger.error("retrieve_context: invalid scene index", index=scene_index)
        return {"current_node": "retrieve_context"}

    current_scene = chapter_plan[scene_index]
    model_name = state.get("narrative_model", config.NARRATIVE_MODEL)

    # Get chapter outlines from content manager
    chapter_outlines = get_chapter_outlines(state, content_manager)

    # Build context components
    hybrid_context_parts = []

    # =========================================================================
    # 1. Scene-Specific Character Context
    # =========================================================================
    try:
        character_context = await _get_scene_character_context(
            current_scene=current_scene,
            chapter_number=chapter_number,
            chapter_outlines=chapter_outlines,
            model_name=model_name,
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

    # =========================================================================
    # 2. Scene-Specific KG Facts
    # =========================================================================
    try:
        kg_facts_block = await _get_scene_specific_kg_facts(
            current_scene=current_scene,
            chapter_number=chapter_number,
            chapter_outlines=chapter_outlines,
            chapter_plan=chapter_plan,
            model_name=model_name,
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

    # =========================================================================
    # 3. Scene Events
    # =========================================================================
    scene_events_context = await _get_scene_events_context(
        chapter_number=chapter_number,
        scene_index=scene_index,
    )
    if scene_events_context:
        hybrid_context_parts.append(scene_events_context)

    # =========================================================================
    # 4. Character Relationships
    # =========================================================================
    scene_characters = _extract_scene_characters(current_scene)
    if scene_characters:
        relationships_context = await _get_character_relationships_context(
            character_names=scene_characters,
            chapter_number=chapter_number,
        )
        if relationships_context:
            hybrid_context_parts.append(relationships_context)

    # =========================================================================
    # 5. Character Items
    # =========================================================================
    if scene_characters:
        character_items_context = await _get_character_items_context(
            character_names=scene_characters,
            chapter_number=chapter_number,
        )
        if character_items_context:
            hybrid_context_parts.append(character_items_context)

    # =========================================================================
    # 6. Scene Items
    # =========================================================================
    scene_items_context = await _get_scene_items_context(
        chapter_number=chapter_number,
        scene_index=scene_index,
    )
    if scene_items_context:
        hybrid_context_parts.append(scene_items_context)

    # =========================================================================
    # 7. Act Events (Plot Structure)
    # =========================================================================
    act_number = chapter_outlines.get(chapter_number, {}).get("act_number")
    if act_number:
        act_events_context = await _get_act_events_context(act_number=act_number)
        if act_events_context:
            hybrid_context_parts.append(act_events_context)

    # =========================================================================
    # 8. Previous Chapter Summaries
    # =========================================================================
    # Get summaries (prefers externalized content, falls back to in-state)
    previous_summaries = get_previous_summaries(state, content_manager)
    if previous_summaries:
        summaries_text = "\n\n**Recent Chapter Summaries:**\n"
        for summary in previous_summaries[-3:]:
            summaries_text += f"\n{summary}"
        hybrid_context_parts.append(summaries_text)

    # =========================================================================
    # 9. Previous Scenes in This Chapter (Token-Aware)
    # =========================================================================
    scene_drafts = get_scene_drafts(state, content_manager)
    if scene_drafts:
        previous_scenes_context = await _get_previous_scenes_context(
            scene_drafts=scene_drafts,
            chapter_plan=chapter_plan,
            scene_index=scene_index,
            model_name=model_name,
            extraction_model=state.get("small_model", model_name),
        )
        if previous_scenes_context:
            hybrid_context_parts.append(previous_scenes_context)

    # =========================================================================
    # 10. Location Context (if specified in scene)
    # =========================================================================
    location_context = await _get_scene_location_context(
        current_scene=current_scene,
        chapter_number=chapter_number,
    )
    if location_context:
        hybrid_context_parts.append(location_context)

    # =========================================================================
    # 11. Semantic Context (Vector Search)
    # =========================================================================
    # Generate query from current scene description
    scene_query = f"{current_scene.get('title', '')} {current_scene.get('scene_description', '')}"
    semantic_context = await _get_semantic_context(
        query_text=scene_query,
        chapter_number=chapter_number,
        model_name=model_name,
    )
    if semantic_context:
        hybrid_context_parts.append(semantic_context)

    hybrid_context = "\n\n".join(hybrid_context_parts)

    # Log context size for monitoring
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


async def _get_scene_character_context(
    current_scene: dict,
    chapter_number: int,
    chapter_outlines: dict[int, dict],
    model_name: str,
) -> str | None:
    """Build a token-budgeted character profile block for the current scene.

    Args:
        current_scene: Scene plan entry that may include a character list.
        chapter_number: Current chapter number for limiting profile retrieval.
        chapter_outlines: Chapter outlines used by profile retrieval helpers.
        model_name: Model name used for token counting.

    Returns:
        Formatted character profiles block, or `None` when the scene does not specify
        characters or no profiles are available.
    """
    # Extract character names from scene
    scene_characters = _extract_scene_characters(current_scene)

    if not scene_characters:
        logger.debug("retrieve_context: no characters specified in scene")
        return None

    logger.debug(
        "retrieve_context: filtering context for scene characters",
        characters=scene_characters,
    )

    # Get filtered character profiles - let exceptions propagate
    character_profiles_text = await get_filtered_character_profiles_for_prompt_plain_text(
        character_names=scene_characters,
        up_to_chapter_inclusive=chapter_number - 1 if chapter_number > 1 else config.KG_PREPOPULATION_CHAPTER_NUM,
    )

    if character_profiles_text and character_profiles_text != "No character profiles available.":
        # Truncate if exceeds budget
        truncated = truncate_text_by_tokens(
            text=character_profiles_text,
            model_name=model_name,
            max_tokens=CHARACTER_PROFILES_TOKEN_BUDGET,
            truncation_marker="\n... (character profiles truncated for context budget)",
        )
        return f"**Scene Character Profiles:**\n{truncated}"

    return None


def _extract_scene_characters(scene: dict) -> list[str]:
    """
    Extract character names from a scene definition.

    Handles multiple possible field names for character lists.

    Args:
        scene: Scene dictionary

    Returns:
        List of character names
    """
    characters = []

    # Check various possible field names
    for field in ["characters", "characters_involved", "character_list", "cast"]:
        scene_chars = scene.get(field)
        if scene_chars:
            if isinstance(scene_chars, list):
                for char in scene_chars:
                    if isinstance(char, str) and char.strip():
                        characters.append(char.strip())
                    elif isinstance(char, dict) and char.get("name"):
                        characters.append(char["name"].strip())
            elif isinstance(scene_chars, str):
                # Comma-separated list
                characters.extend([c.strip() for c in scene_chars.split(",") if c.strip()])
            break

    # Deduplicate while preserving order
    seen = set()
    unique_characters = []
    for char in characters:
        if char not in seen:
            seen.add(char)
            unique_characters.append(char)

    return unique_characters


async def _get_scene_specific_kg_facts(
    current_scene: dict,
    chapter_number: int,
    chapter_outlines: dict[int, dict],
    chapter_plan: list,
    model_name: str,
) -> str | None:
    """
    Get KG facts filtered by scene-specific entities.

    Uses targeted queries focusing on:
    - Characters in the scene
    - Location of the scene
    - Related events and relationships

    Args:
        current_scene: Current scene definition
        chapter_number: Current chapter number
        chapter_outlines: Chapter outlines dictionary from content manager
        chapter_plan: Full chapter plan (list of scenes)
        model_name: Model name for token counting

    Returns:
        Formatted KG facts string or None
    """
    scene_characters = _extract_scene_characters(current_scene)

    # Build scene detail for the KG facts function.
    # The function expects SceneDetail-like dicts.
    scene_detail_untyped: dict[str, Any] = {
        "characters_involved": scene_characters,
        **current_scene,
    }
    scene_detail = cast(SceneDetail, scene_detail_untyped)

    # Get KG facts with scene-specific filtering - let exceptions propagate
    # Import config to get protagonist_name
    protagonist_name = getattr(config, "DEFAULT_PROTAGONIST_NAME", "Protagonist")

    kg_facts_block = await get_reliable_kg_facts_for_drafting_prompt(
        chapter_outlines=chapter_outlines,
        chapter_number=chapter_number,
        chapter_plan=[scene_detail],  # Pass only current scene for focused filtering
        protagonist_name=protagonist_name,
    )

    if kg_facts_block and "No specific reliable KG facts" not in kg_facts_block:
        # Truncate if exceeds budget
        truncated = truncate_text_by_tokens(
            text=kg_facts_block,
            model_name=model_name,
            max_tokens=KG_FACTS_TOKEN_BUDGET,
            truncation_marker="\n... (KG facts truncated for context budget)",
        )
        return truncated

    return None


async def _get_previous_scenes_context(
    scene_drafts: list[str],
    chapter_plan: list[dict],
    scene_index: int,
    model_name: str,
    extraction_model: str,
) -> str | None:
    """
    Build context from previous scenes with intelligent token management.

    Strategy:
    1. Calculate available token budget for previous scenes
    2. For scenes that fit within budget, include full text (tail portion)
    3. For scenes that would exceed budget, generate/use summaries
    4. Use sliding window approach - most recent scenes get more tokens

    Args:
        scene_drafts: List of previous scene draft texts
        chapter_plan: Full chapter plan for scene titles
        scene_index: Current scene index
        model_name: Model for token counting
        extraction_model: Model for summarization

    Returns:
        Formatted previous scenes context or None
    """
    if not scene_drafts:
        return None

    previous_scenes_text = "\n\n**Previous Scenes in This Chapter:**\n"

    # Calculate tokens per scene based on budget and number of scenes
    num_scenes = len(scene_drafts)

    # Sliding window: allocate more tokens to recent scenes
    # Weight distribution: most recent gets most tokens
    total_weight = sum(range(1, num_scenes + 1))

    context_parts = []
    total_tokens_used = 0

    for i, draft in enumerate(scene_drafts):
        if i >= scene_index:
            break

        scene_title = chapter_plan[i].get("title", f"Scene {i + 1}")

        # Calculate token budget for this scene (more for recent scenes)
        scene_weight = i + 1  # Earlier scenes get less weight
        scene_token_budget = int(PREVIOUS_SCENES_TOKEN_BUDGET * (scene_weight / total_weight))

        # Ensure minimum budget
        scene_token_budget = max(scene_token_budget, SUMMARY_MAX_TOKENS)

        # Check if full context fits
        draft_tokens = count_tokens(draft, model_name)

        if draft_tokens <= scene_token_budget:
            # Full context fits within budget
            scene_context = f"\n--- {scene_title} ---\n{draft}\n"
            context_parts.append(scene_context)
            total_tokens_used += draft_tokens
        else:
            # Need to summarize or truncate
            if draft_tokens > scene_token_budget * 2:
                # Scene is significantly over budget - use LLM summarization
                summary = await _summarize_scene_text(
                    scene_text=draft,
                    scene_title=scene_title,
                    extraction_model=extraction_model,
                    max_tokens=SUMMARY_MAX_TOKENS,
                )
                scene_context = f"\n--- {scene_title} (Summary) ---\n{summary}\n"
            else:
                # Scene is slightly over budget - intelligent truncation
                # Keep the tail (most relevant for continuity) with some head context
                truncated = _smart_truncate_scene(
                    text=draft,
                    model_name=model_name,
                    max_tokens=scene_token_budget,
                )
                scene_context = f"\n--- {scene_title} ---\n...{truncated}\n"

            context_parts.append(scene_context)
            total_tokens_used += count_tokens(scene_context, model_name)

    if not context_parts:
        return None

    previous_scenes_text += "".join(context_parts)

    logger.debug(
        "retrieve_context: previous scenes context built",
        num_scenes=len(context_parts),
        total_tokens=total_tokens_used,
        budget=PREVIOUS_SCENES_TOKEN_BUDGET,
    )

    return previous_scenes_text


async def _summarize_scene_text(
    scene_text: str,
    scene_title: str,
    extraction_model: str,
    max_tokens: int,
) -> str:
    """
    Generate a concise summary of a scene using LLM.

    Args:
        scene_text: Full scene text to summarize
        scene_title: Scene title for context
        extraction_model: Model to use for summarization
        max_tokens: Target max tokens for summary

    Returns:
        Summary text
    """
    try:
        prompt = render_prompt(
            "knowledge_agent/summarize_scene_for_continuity.j2",
            {
                "scene_title": scene_title,
                "scene_text": scene_text,
            },
        )

        summary_text, _ = await llm_service.async_call_llm(
            model_name=extraction_model,
            prompt=prompt,
            temperature=config.TEMPERATURE_SUMMARY,
            max_tokens=max_tokens,
            allow_fallback=True,
            auto_clean_response=True,
            system_prompt=get_system_prompt("knowledge_agent"),
        )

        if summary_text and summary_text.strip():
            return summary_text.strip()

    except Exception as e:
        logger.warning(
            "retrieve_context: failed to summarize scene, falling back to truncation",
            scene_title=scene_title,
            error=str(e),
        )

    # Fallback to simple truncation
    return truncate_text_by_tokens(
        text=scene_text,
        model_name=extraction_model,
        max_tokens=max_tokens,
        truncation_marker="...",
    )


def _smart_truncate_scene(
    text: str,
    model_name: str,
    max_tokens: int,
) -> str:
    """
    Intelligently truncate scene text keeping the most relevant parts.

    Strategy:
    - Keep the last 70% of tokens (most relevant for continuity)
    - Include a brief head section (10%) for context
    - Middle section (20%) gets truncated

    Args:
        text: Scene text to truncate
        model_name: Model for token counting
        max_tokens: Target max tokens

    Returns:
        Truncated text
    """
    total_tokens = count_tokens(text, model_name)

    if total_tokens <= max_tokens:
        return text

    # Calculate split points
    head_tokens = int(max_tokens * 0.1)  # 10% for beginning context
    tail_tokens = max_tokens - head_tokens - 10  # Rest for tail, minus some for ellipsis

    # Split text approximately
    words = text.split()
    total_words = len(words)

    # Estimate tokens per word ratio
    tokens_per_word = total_tokens / total_words if total_words > 0 else 1

    head_words = int(head_tokens / tokens_per_word)
    tail_words = int(tail_tokens / tokens_per_word)

    if head_words + tail_words >= total_words:
        # Just do simple tail truncation if the math doesn't work out
        return truncate_text_by_tokens(
            text=text,
            model_name=model_name,
            max_tokens=max_tokens,
            truncation_marker="",
        )

    head_section = " ".join(words[:head_words])
    tail_section = " ".join(words[-tail_words:])

    return f"{head_section}\n[...]\n{tail_section}"


async def _get_scene_location_context(
    current_scene: dict,
    chapter_number: int,
) -> str | None:
    """
    Get location details for the current scene from Neo4j.

    Args:
        current_scene: Scene definition with location info
        chapter_number: Current chapter number

    Returns:
        Formatted location context or None
    """
    location_name = current_scene.get("location") or current_scene.get("setting")

    if not location_name:
        return None

    try:
        # Query Neo4j for location details
        kg_chapter_limit = config.KG_PREPOPULATION_CHAPTER_NUM if chapter_number == 1 else chapter_number - 1

        # Get location status and description
        results = await kg_queries.query_kg_from_db(
            subject=location_name,
            chapter_limit=kg_chapter_limit,
            limit_results=10,
        )

        if results:
            location_facts = []
            for fact in results:
                predicate = fact.get("predicate", "").replace("_", " ").lower()
                obj = fact.get("object", "")
                if obj:
                    location_facts.append(f"- {location_name} {predicate}: {obj}")

            if location_facts:
                return f"**Current Location - {location_name}:**\n" + "\n".join(location_facts[:3])

    except Exception as e:
        logger.warning(
            "retrieve_context: non-fatal error getting location context, continuing without it",
            location=location_name,
            error=str(e),
        )

    return None


async def _get_semantic_context(
    query_text: str,
    chapter_number: int,
    model_name: str,
) -> str | None:
    """
    Get semantically similar context from previous chapters.

    Args:
        query_text: Text to generate embedding for
        chapter_number: Current chapter number (to exclude)
        model_name: Model for token counting

    Returns:
        Formatted semantic context string or None
    """
    if not query_text.strip():
        return None

    try:
        # Generate embedding for query
        query_embedding = await llm_service.async_get_embedding(query_text)

        if query_embedding is None:
            logger.warning("retrieve_context: failed to generate query embedding")
            return None

        # Find similar context using native query
        # This gets both similar chapters and the immediate previous chapter
        context_chapters = await chapter_queries.find_semantic_context_native(
            query_embedding=query_embedding,
            current_chapter_number=chapter_number,
            limit=3,  # Get top 3 similar + previous
        )

        if not context_chapters:
            return None

        formatted_parts = []
        formatted_parts.append("**Relevant Past Context (Semantic Search):**")

        for chapter in context_chapters:
            chap_num = chapter.get("chapter_number")
            summary = chapter.get("summary")
            score = chapter.get("score", 0)
            context_type = chapter.get("context_type", "similarity")

            # Label context type clearly
            label = "Previous Chapter" if context_type == "immediate_previous" else f"Similar Chapter (Score: {score:.2f})"

            if summary:
                formatted_parts.append(f"\n--- Chapter {chap_num} ({label}) ---\n{summary}")

        result_text = "\n".join(formatted_parts)

        # Truncate if needed
        return truncate_text_by_tokens(
            text=result_text,
            model_name=model_name,
            max_tokens=SEMANTIC_CONTEXT_TOKEN_BUDGET,
            truncation_marker="\n... (semantic context truncated)",
        )

    except Exception as e:
        logger.warning(
            "retrieve_context: non-fatal error getting semantic context, continuing without it",
            error=str(e),
            exc_info=True,
        )
        return None


async def _get_scene_events_context(
    chapter_number: int,
    scene_index: int,
) -> str | None:
    """Get scene events context from Neo4j.

    Args:
        chapter_number: Current chapter number
        scene_index: Current scene index

    Returns:
        Formatted scene events context or None
    """
    try:
        events = await scene_queries.get_scene_events(
            chapter_number=chapter_number,
            scene_index=scene_index,
        )

        if not events:
            return None

        events_text = "**Scene Events:**\n"
        for event in events:
            events_text += f"\n- **{event.get('name', 'Unnamed Event')}**: {event.get('description', '')}"
            if event.get("conflict"):
                events_text += f"\n  - Conflict: {event['conflict']}"
            if event.get("outcome"):
                events_text += f"\n  - Outcome: {event['outcome']}"
            if event.get("characters_involved"):
                chars = ", ".join(event["characters_involved"])
                events_text += f"\n  - Characters: {chars}"

        return events_text

    except Exception as e:
        logger.warning(
            "retrieve_context: non-fatal error getting scene events, continuing without them",
            chapter=chapter_number,
            scene_index=scene_index,
            error=str(e),
        )
        return None


async def _get_character_relationships_context(
    character_names: list[str],
    chapter_number: int,
) -> str | None:
    """Get character relationships context from Neo4j.

    Args:
        character_names: List of character names in the scene
        chapter_number: Current chapter number

    Returns:
        Formatted character relationships context or None
    """
    try:
        relationships = await scene_queries.get_character_relationships_for_scene(
            character_names=character_names,
            chapter_limit=chapter_number - 1 if chapter_number > 1 else 0,
        )

        if not relationships:
            return None

        relationships_text = "**Character Relationships:**\n"
        for rel in relationships:
            rel_type = rel.get("relationship_type", "").replace("_", " ").lower()
            source = rel.get("source", "")
            target = rel.get("target", "")
            description = rel.get("description", "")

            relationships_text += f"\n- {source} {rel_type} {target}"
            if description:
                relationships_text += f": {description}"

        return relationships_text

    except Exception as e:
        logger.warning(
            "retrieve_context: non-fatal error getting character relationships, continuing without them",
            characters=character_names,
            error=str(e),
        )
        return None


async def _get_character_items_context(
    character_names: list[str],
    chapter_number: int,
) -> str | None:
    """Get character items context from Neo4j.

    Args:
        character_names: List of character names in the scene
        chapter_number: Current chapter number

    Returns:
        Formatted character items context or None
    """
    try:
        items = await scene_queries.get_character_items(
            character_names=character_names,
            chapter_limit=chapter_number - 1 if chapter_number > 1 else 0,
        )

        if not items:
            return None

        items_text = "**Character Possessions:**\n"
        by_character: dict[str, list[dict]] = {}
        for item in items:
            char_name = item.get("character_name", "")
            if char_name not in by_character:
                by_character[char_name] = []
            by_character[char_name].append(item)

        for char_name, char_items in by_character.items():
            items_text += f"\n- {char_name}:"
            for item in char_items:
                item_name = item.get("item_name", "")
                item_desc = item.get("item_description", "")
                items_text += f"\n  - {item_name}"
                if item_desc:
                    items_text += f": {item_desc}"

        return items_text

    except Exception as e:
        logger.warning(
            "retrieve_context: non-fatal error getting character items, continuing without them",
            characters=character_names,
            error=str(e),
        )
        return None


async def _get_scene_items_context(
    chapter_number: int,
    scene_index: int,
) -> str | None:
    """Get scene items context from Neo4j.

    Args:
        chapter_number: Current chapter number
        scene_index: Current scene index

    Returns:
        Formatted scene items context or None
    """
    try:
        items = await scene_queries.get_scene_items(
            chapter_number=chapter_number,
            scene_index=scene_index,
        )

        if not items:
            return None

        items_text = "**Items Featured in Scene:**\n"
        for item in items:
            item_name = item.get("item_name", "")
            item_desc = item.get("item_description", "")
            items_text += f"\n- {item_name}"
            if item_desc:
                items_text += f": {item_desc}"

        return items_text

    except Exception as e:
        logger.warning(
            "retrieve_context: non-fatal error getting scene items, continuing without them",
            chapter=chapter_number,
            scene_index=scene_index,
            error=str(e),
        )
        return None


async def _get_act_events_context(
    act_number: int,
) -> str | None:
    """Get act events context from Neo4j.

    Args:
        act_number: Act number (1, 2, or 3)

    Returns:
        Formatted act events context or None
    """
    try:
        events_data = await scene_queries.get_act_events(act_number=act_number)

        major_points = events_data.get("major_plot_points", [])
        act_events = events_data.get("act_key_events", [])

        if not major_points and not act_events:
            return None

        context_text = f"**Act {act_number} Plot Structure:**\n"

        if major_points:
            context_text += "\nMajor Plot Points:\n"
            for point in sorted(major_points, key=lambda x: x.get("sequence_order", 0)):
                context_text += f"- {point.get('name', '')}: {point.get('description', '')}\n"

        if act_events:
            context_text += f"\nKey Events in Act {act_number}:\n"
            for event in sorted(act_events, key=lambda x: x.get("sequence_in_act", 0)):
                context_text += f"- {event.get('name', '')}: {event.get('description', '')}\n"
                if event.get("cause"):
                    context_text += f"  - Cause: {event['cause']}\n"
                if event.get("effect"):
                    context_text += f"  - Effect: {event['effect']}\n"

        return context_text

    except Exception as e:
        logger.warning(
            "retrieve_context: non-fatal error getting act events, continuing without them",
            act_number=act_number,
            error=str(e),
        )
        return None


__all__ = ["retrieve_context"]
