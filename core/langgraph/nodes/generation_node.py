# core/langgraph/nodes/generation_node.py
"""
Generation node for LangGraph workflow.

This module contains the chapter generation logic for the LangGraph-based
narrative generation workflow.

Migration Reference: docs/phase2_migration_plan.md - Step 2.1

Source Code Ported From:
- agents/narrative_agent.py:
  - draft_chapter() (lines 486-564)
  - _draft_chapter() (lines 361-483)
- prompts/prompt_data_getters.py:
  - get_reliable_kg_facts_for_drafting_prompt()
"""

from __future__ import annotations

import structlog

import config
from core.langgraph.content_manager import (
    ContentManager,
    get_chapter_outlines,
    get_previous_summaries,
)
from core.langgraph.graph_context import get_key_events
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from processing.text_deduplicator import TextDeduplicator
from prompts.prompt_data_getters import get_reliable_kg_facts_for_drafting_prompt
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)


async def generate_chapter(state: NarrativeState) -> NarrativeState:
    """
    Generate chapter prose from outline and Neo4j context.

    This is the main LangGraph node function for chapter generation.
    It builds context from the knowledge graph, constructs a generation prompt,
    calls the LLM, and updates the state with the generated chapter text.

    PORTED FROM: NarrativeAgent.draft_chapter()

    Process Flow:
    1. Build context from knowledge graph (characters, locations, events, etc.)
    2. Construct generation prompt with all relevant context
    3. Generate chapter text via LLM
    4. Post-process and update state

    Args:
        state: Current narrative state containing outline, chapter number, etc.

    Returns:
        Updated state with draft_text and draft_word_count populated
    """
    logger.info(
        "generate_chapter: starting generation",
        chapter=state["current_chapter"],
        model=state["generation_model"],
    )

    # Initialize content manager for reading externalized content
    content_manager = ContentManager(state["project_dir"])

    # Get chapter outlines (prefers externalized content, falls back to in-state)
    chapter_outlines = get_chapter_outlines(state, content_manager)

    # Check for deprecated plot_outline usage as ultimate fallback
    if not chapter_outlines:
        plot_outline = state.get("plot_outline")
        if plot_outline:
            logger.warning(
                "generate_chapter: using deprecated plot_outline field. "
                "Please migrate to chapter_outlines. "
                "plot_outline will be removed in SAGA v3.0",
                deprecation=True,
            )
            chapter_outlines = plot_outline

    if not chapter_outlines:
        error_msg = "No chapter outlines available for generation"
        logger.error("generate_chapter: fatal error", error=error_msg)
        return {
            **state,
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "generate",
            "current_node": "generate",
        }

    chapter_number = state["current_chapter"]

    # Validate chapter exists in outline
    if chapter_number not in chapter_outlines:
        error_msg = f"Chapter {chapter_number} not found in chapter outlines"
        logger.error(
            "generate_chapter: fatal error",
            error=error_msg,
            chapter=chapter_number,
            available_chapters=list(chapter_outlines.keys()),
        )
        return {
            **state,
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "generate",
            "current_node": "generate",
        }

    # Get plot point focus for this chapter
    plot_point_focus = state.get("plot_point_focus")
    if not plot_point_focus:
        # Try to extract from outline
        chapter_outline = chapter_outlines.get(chapter_number)
        if chapter_outline and isinstance(chapter_outline, dict):
            plot_point_focus = chapter_outline.get("plot_point", "")

        if not plot_point_focus:
            logger.warning(
                "generate_chapter: no plot point focus found, using generic focus",
                chapter=chapter_number,
            )
            plot_point_focus = f"Continue the story in Chapter {chapter_number}"

    # Get key events for additional context
    key_events = await get_key_events(
        current_chapter=chapter_number,
        lookback_chapters=10,
        max_events=20,
    )

    # Step 2: Construct hybrid context for the prompt
    # Use existing prompt data getter for KG facts
    kg_facts_block = await get_reliable_kg_facts_for_drafting_prompt(
        chapter_outlines, chapter_number, None
    )

    # Build hybrid context string combining KG facts and summaries
    hybrid_context_parts = []

    # Add KG facts
    if kg_facts_block:
        hybrid_context_parts.append(kg_facts_block)

    # Add previous chapter summaries (uses externalized content with fallback)
    previous_summaries = get_previous_summaries(state, content_manager)
    if previous_summaries:
        summaries_text = "\n\n**Recent Chapter Summaries:**\n"
        for summary in previous_summaries[-3:]:
            summaries_text += f"\n{summary}"
        hybrid_context_parts.append(summaries_text)

    # Add key events if available
    if key_events:
        events_text = "\n\n**Key Recent Events:**\n"
        for event in key_events[:5]:  # Top 5 most important
            events_text += f"- {event['description']} (Chapter {event['chapter']})\n"
        hybrid_context_parts.append(events_text)

    hybrid_context_for_draft = "\n\n".join(hybrid_context_parts)

    # Step 3: Construct generation prompt
    prompt = await _construct_generation_prompt(
        chapter_number=chapter_number,
        plot_point_focus=plot_point_focus,
        hybrid_context=hybrid_context_for_draft,
        novel_title=state["title"],
        novel_genre=state["genre"],
        novel_theme=state.get("theme", ""),
        protagonist_name=state.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME),
    )

    # Step 4: Calculate token budget
    model_name = state["narrative_model"]
    prompt_tokens = llm_service.count_tokens(prompt, model_name)

    # Calculate max tokens for generation
    max_context = getattr(config, "MAX_CONTEXT_TOKENS", 128000)
    token_buffer = getattr(config.settings, "NARRATIVE_TOKEN_BUFFER", 2000)
    max_generation = getattr(config, "MAX_GENERATION_TOKENS", 8000)

    available_tokens = max_context - prompt_tokens - token_buffer
    max_gen_tokens = min(max_generation, available_tokens)

    if max_gen_tokens < 500:
        error_msg = (
            f"Insufficient token space for generation. "
            f"Prompt tokens: {prompt_tokens}, available: {available_tokens}"
        )
        logger.error("generate_chapter: token budget exceeded", error=error_msg)
        return {
            **state,
            "last_error": error_msg,
            "current_node": "generate",
        }

    # Step 5: Generate via LLM
    logger.info(
        "generate_chapter: calling LLM",
        chapter=chapter_number,
        model=model_name,
        max_tokens=max_gen_tokens,
        prompt_tokens=prompt_tokens,
    )

    try:
        draft_text, usage = await llm_service.async_call_llm(
            model_name=model_name,
            prompt=prompt,
            temperature=getattr(config.Temperatures, "CHAPTER_GENERATION", 0.7),
            max_tokens=max_gen_tokens,
            allow_fallback=True,
            stream_to_disk=False,
            frequency_penalty=getattr(config, "FREQUENCY_PENALTY_DRAFTING", 0.3),
            presence_penalty=getattr(config, "PRESENCE_PENALTY_DRAFTING", 0.3),
            auto_clean_response=True,
            system_prompt=get_system_prompt("narrative_agent"),
        )

        if not draft_text or not draft_text.strip():
            logger.error(
                "generate_chapter: LLM returned empty text",
                chapter=chapter_number,
            )
            return {
                **state,
                "last_error": "LLM generation returned empty text",
                "current_node": "generate",
            }

        # Step 6: Post-process and update state
        word_count = len(draft_text.split())

        logger.info(
            "generate_chapter: generation complete",
            chapter=chapter_number,
            word_count=word_count,
            tokens_used=usage.get("total_tokens", 0) if usage else 0,
        )

        # Step 7: Deduplicate text to remove repetitive segments
        deduplicator = TextDeduplicator()
        deduplicated_text, removed_chars = await deduplicator.deduplicate(
            draft_text, segment_level="paragraph"
        )

        # Track if deduplication modified text (signals potentially flawed extraction)
        is_from_flawed_draft = removed_chars > 0

        if removed_chars > 0:
            final_word_count = len(deduplicated_text.split())
            logger.info(
                "generate_chapter: deduplication applied",
                chapter=chapter_number,
                chars_removed=removed_chars,
                original_words=word_count,
                final_words=final_word_count,
                is_from_flawed_draft=True,
            )
        else:
            deduplicated_text = draft_text
            final_word_count = word_count
            logger.info(
                "generate_chapter: no duplicates detected",
                chapter=chapter_number,
                is_from_flawed_draft=False,
            )

        # Initialize content manager for external storage
        content_manager = ContentManager(state["project_dir"])

        # Get current version (for revision tracking)
        current_version = (
            content_manager.get_latest_version("draft", f"chapter_{chapter_number}") + 1
        )

        # Externalize draft_text to reduce state bloat
        draft_ref = content_manager.save_text(
            deduplicated_text,
            "draft",
            f"chapter_{chapter_number}",
            current_version,
        )

        # Externalize hybrid_context
        hybrid_context_ref = (
            content_manager.save_text(
                hybrid_context_for_draft,
                "hybrid_context",
                f"chapter_{chapter_number}",
                current_version,
            )
            if hybrid_context_for_draft
            else None
        )

        # Externalize kg_facts_block
        kg_facts_ref = (
            content_manager.save_text(
                kg_facts_block,
                "kg_facts",
                f"chapter_{chapter_number}",
                current_version,
            )
            if kg_facts_block
            else None
        )

        logger.info(
            "generate_chapter: content externalized",
            chapter=chapter_number,
            version=current_version,
            draft_size=draft_ref["size_bytes"],
        )

        return {
            **state,
            "draft_ref": draft_ref,
            "draft_word_count": final_word_count,
            "is_from_flawed_draft": is_from_flawed_draft,
            "current_node": "generate",
            "last_error": None,
            "hybrid_context_ref": hybrid_context_ref,
            "kg_facts_ref": kg_facts_ref,
        }

    except Exception as e:
        error_msg = f"Error during LLM generation: {str(e)}"
        logger.error(
            "generate_chapter: exception during generation",
            chapter=chapter_number,
            error=str(e),
            exc_info=True,
        )
        return {
            **state,
            "last_error": error_msg,
            "current_node": "generate",
        }


async def _construct_generation_prompt(
    *,
    chapter_number: int,
    plot_point_focus: str,
    hybrid_context: str,
    novel_title: str,
    novel_genre: str,
    novel_theme: str,
    protagonist_name: str,
) -> str:
    """
    Construct the generation prompt for chapter drafting.

    This uses the existing Jinja2 template system to build a comprehensive
    prompt with all necessary context.

    REUSES: prompts/narrative_agent/draft_chapter_from_plot_point.j2

    Args:
        chapter_number: Chapter number being generated
        plot_point_focus: Main plot point or focus for this chapter
        hybrid_context: Combined context from KG and previous chapters
        novel_title: Title of the novel
        novel_genre: Genre of the novel
        novel_theme: Central theme
        protagonist_name: Name of the protagonist

    Returns:
        Rendered prompt string ready for LLM
    """
    # Get minimum chapter length from config
    min_length = getattr(config.settings, "MIN_CHAPTER_LENGTH_CHARS", 12000)

    prompt = render_prompt(
        "narrative_agent/draft_chapter_from_plot_point.j2",
        {
            "chapter_number": chapter_number,
            "novel_title": novel_title,
            "novel_genre": novel_genre,
            "novel_theme": novel_theme,
            "protagonist_name": protagonist_name,
            "plot_point_focus": plot_point_focus,
            "hybrid_context_for_draft": hybrid_context,
            "min_length": min_length,
        },
    )

    return prompt


__all__ = ["generate_chapter"]
