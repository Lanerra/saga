"""Scene-level retrieval for context building.

Provides:
- Scene items context
- Previous scenes context (token-aware with summarization)
- Location context
"""

import structlog

import config
from core.langgraph.content_manager import ContentManager
from core.langgraph.state import NarrativeState
from core.service_context import get_services
from core.text_processing_service import count_tokens, truncate_text_by_tokens
from data_access import chapter_queries, kg_queries, scene_queries
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)

PREVIOUS_SCENES_TOKEN_BUDGET = config.MAX_GENERATION_TOKENS
SUMMARY_MAX_TOKENS = config.MAX_SUMMARY_TOKENS
SEMANTIC_CONTEXT_TOKEN_BUDGET = config.MAX_GENERATION_TOKENS


async def get_scene_items_context(
    state: NarrativeState,
    chapter_number: int,
    scene_index: int,
    content_manager: ContentManager,
) -> str | None:
    """Get scene items context from Neo4j.

    Args:
        state: Workflow state.
        chapter_number: Current chapter number.
        scene_index: Current scene index.
        content_manager: Content manager instance.

    Returns:
        Formatted scene items context or None.
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
            "context_scene: non-fatal error getting scene items, continuing without them",
            chapter=chapter_number,
            scene_index=scene_index,
            error=str(e),
        )
        return None


async def get_previous_scenes_context(
    state: NarrativeState,
    scene_drafts: list[str],
    chapter_plan: list[dict],
    scene_index: int,
    model_name: str,
    extraction_model: str,
    content_manager: ContentManager,
) -> str | None:
    """Build previous-scene context within one rendered-string token budget.

    Strategy:
    1. Prioritize recent scenes, displaying retained scenes chronologically.
    2. Include full text when it fits; otherwise summarize or retain a tail.
    3. Measure each complete candidate, including all headings and markers,
       using count_tokens for model_name (which may use a configured fallback).
       The cap is not a narrative-quality guarantee.

    Args:
        state: Workflow state.
        scene_drafts: List of previous scene draft texts.
        chapter_plan: Full chapter plan for scene titles.
        scene_index: Current scene index.
        model_name: Model for token counting.
        extraction_model: Model for summarization.
        content_manager: Content manager instance.

    Returns:
        Formatted previous scenes context or None.
    """
    if PREVIOUS_SCENES_TOKEN_BUDGET < 0:
        raise ValueError("Previous-scenes token budget must be non-negative")
    if not scene_drafts or scene_index <= 0 or PREVIOUS_SCENES_TOKEN_BUDGET == 0:
        return None

    previous_scenes_text = "\n\n**Previous Scenes in This Chapter:**\n"
    context_parts: list[str] = []

    for index in reversed(range(min(len(scene_drafts), scene_index))):
        draft = scene_drafts[index]
        if not draft.strip():
            continue
        scene_title = chapter_plan[index].get("title", f"Scene {index + 1}")
        scene_header = f"\n--- {scene_title} ---\n"
        later_scenes = "".join(context_parts)
        scene_context = f"{scene_header}{draft}\n"
        if count_tokens(previous_scenes_text + scene_context + later_scenes, model_name) <= PREVIOUS_SCENES_TOKEN_BUDGET:
            context_parts.insert(0, scene_context)
            continue

        available_tokens = PREVIOUS_SCENES_TOKEN_BUDGET - count_tokens(previous_scenes_text + scene_header + "\n" + later_scenes, model_name)
        if available_tokens <= 0:
            continue

        if count_tokens(draft, model_name) > available_tokens * 2 and SUMMARY_MAX_TOKENS > 0:
            scene_text = await _summarize_scene_text(
                scene_text=draft,
                scene_title=scene_title,
                extraction_model=extraction_model,
                max_tokens=min(SUMMARY_MAX_TOKENS, available_tokens),
            )
            scene_header = f"\n--- {scene_title} (Summary) ---\n"
        else:
            scene_text = _smart_truncate_scene(draft, model_name, available_tokens)

        fitted_context = _fit_previous_scene_context(
            previous_scenes_text, scene_header, scene_text, later_scenes,
            model_name, PREVIOUS_SCENES_TOKEN_BUDGET,
        )
        if fitted_context is not None:
            context_parts.insert(0, fitted_context)

    if not context_parts:
        return None

    previous_scenes_text += "".join(context_parts)

    logger.debug(
        "context_scene: previous scenes context built",
        num_scenes=len(context_parts),
        total_tokens=count_tokens(previous_scenes_text, model_name),
        budget=PREVIOUS_SCENES_TOKEN_BUDGET,
    )

    return previous_scenes_text


def _fit_previous_scene_context(
    section_header: str,
    scene_header: str,
    scene_text: str,
    later_scenes: str,
    model_name: str,
    token_budget: int,
) -> str | None:
    """Retain a measured candidate, not an assumed additive token allocation.

    Search Unicode character boundaries to avoid partial UTF-8 decoding. Token
    counts need not be monotonic in suffix length: only measured feasible
    candidates are retained, without promising a maximally filled budget.
    """
    if not scene_text.strip():
        return None
    complete = f"{scene_header}{scene_text}\n"
    if count_tokens(section_header + complete + later_scenes, model_name) <= token_budget:
        return complete

    scene_text = scene_text.rstrip()
    fitted: str | None = None
    minimum_characters = 1
    maximum_characters = len(scene_text)
    while minimum_characters <= maximum_characters:
        retained_characters = (minimum_characters + maximum_characters) // 2
        candidate = f"{scene_header}[...]\n{scene_text[-retained_characters:]}\n"
        if count_tokens(section_header + candidate + later_scenes, model_name) <= token_budget:
            fitted = candidate
            minimum_characters = retained_characters + 1
        else:
            maximum_characters = retained_characters - 1
    return fitted


async def _summarize_scene_text(
    scene_text: str,
    scene_title: str,
    extraction_model: str,
    max_tokens: int,
) -> str:
    """Generate a concise summary of a scene using LLM.

    Args:
        scene_text: Full scene text to summarize.
        scene_title: Scene title for context.
        extraction_model: Model to use for summarization.
        max_tokens: Retained context budget for the truncation fallback, not reasoning.

    Returns:
        Summary text.
    """
    try:
        prompt = render_prompt(
            "knowledge_agent/summarize_scene_for_continuity.j2",
            {
                "scene_title": scene_title,
                "scene_text": scene_text,
            },
        )

        summary_text, _ = await get_services().language_model.async_call_llm(
            model_name=extraction_model,
            prompt=prompt,
            temperature=config.TEMPERATURE_SUMMARY,
            max_tokens=config.MAX_SUMMARY_TOKENS,
            allow_fallback=True,
            auto_clean_response=True,
            system_prompt=get_system_prompt("knowledge_agent"),
        )

        if summary_text and summary_text.strip():
            return summary_text.strip()

    except TimeoutError:
        raise
    except Exception as e:
        logger.warning(
            "context_scene: failed to summarize scene, falling back to truncation",
            scene_title=scene_title,
            error=str(e),
        )

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
    """Intelligently truncate scene text keeping the most relevant parts.

    Strategy:
    - Keep the last 70% of tokens (most relevant for continuity)
    - Include a brief head section (10%) for context
    - Middle section (20%) gets truncated

    Args:
        text: Scene text to truncate.
        model_name: Model for token counting.
        max_tokens: Target max tokens.

    Returns:
        Truncated text.
    """
    total_tokens = count_tokens(text, model_name)

    if total_tokens <= max_tokens:
        return text

    head_tokens = int(max_tokens * 0.1)
    tail_tokens = max_tokens - head_tokens - 10

    words = text.split()
    total_words = len(words)

    tokens_per_word = total_tokens / total_words if total_words > 0 else 1

    head_words = int(head_tokens / tokens_per_word)
    tail_words = int(tail_tokens / tokens_per_word)

    if head_words + tail_words >= total_words:
        return truncate_text_by_tokens(
            text=text,
            model_name=model_name,
            max_tokens=max_tokens,
            truncation_marker="",
        )

    head_section = " ".join(words[:head_words])
    tail_section = " ".join(words[-tail_words:])

    return f"{head_section}\n[...]\n{tail_section}"


async def get_location_context(
    state: NarrativeState,
    current_scene: dict,
    chapter_number: int,
    content_manager: ContentManager,
) -> str | None:
    """Get location details for the current scene from Neo4j.

    Args:
        state: Workflow state.
        current_scene: Scene definition with location info.
        chapter_number: Current chapter number.
        content_manager: Content manager instance.

    Returns:
        Formatted location context or None.
    """
    location_name = current_scene.get("location") or current_scene.get("setting")

    if not location_name:
        return None

    try:
        kg_chapter_limit = config.KG_PREPOPULATION_CHAPTER_NUM if chapter_number == 1 else chapter_number - 1

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
            "context_scene: non-fatal error getting location context, continuing without it",
            location=location_name,
            error=str(e),
        )

    return None


async def get_semantic_context(
    state: NarrativeState,
    query_text: str,
    chapter_number: int,
    model_name: str,
    content_manager: ContentManager,
) -> str | None:
    """Get semantically similar context from previous chapters via vector search.

    Args:
        state: Workflow state.
        query_text: Text to generate embedding for.
        chapter_number: Current chapter number (to exclude).
        model_name: Model for token counting.
        content_manager: Content manager instance.

    Returns:
        Formatted semantic context string or None.
    """
    if not query_text.strip():
        return None

    try:
        query_embedding = await get_services().language_model.async_get_embedding(query_text)

        if query_embedding is None:
            logger.warning("context_scene: failed to generate query embedding")
            return None

        context_chapters = await chapter_queries.find_semantic_context_native(
            query_embedding=query_embedding,
            embedding_model=config.EMBEDDING_MODEL,
            current_chapter_number=chapter_number,
            limit=3,
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

            label = "Previous Chapter" if context_type == "immediate_previous" else f"Similar Chapter (Score: {score:.2f})"

            if summary:
                formatted_parts.append(f"\n--- Chapter {chap_num} ({label}) ---\n{summary}")

        result_text = "\n".join(formatted_parts)

        return truncate_text_by_tokens(
            text=result_text,
            model_name=model_name,
            max_tokens=SEMANTIC_CONTEXT_TOKEN_BUDGET,
            truncation_marker="\n... (semantic context truncated)",
        )

    except TimeoutError:
        raise
    except Exception as e:
        logger.warning(
            "context_scene: non-fatal error getting semantic context, continuing without it",
            error=str(e),
            exc_info=True,
        )
        return None
