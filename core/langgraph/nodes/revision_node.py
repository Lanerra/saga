"""
Revision node for LangGraph workflow.

This module contains the chapter revision logic for the LangGraph-based
narrative generation workflow.

Migration Reference: docs/phase2_migration_plan.md - Step 2.2

Source Code Ported From:
- agents/revision_agent.py:
  - Various revision methods
- prompts/revision_agent/full_chapter_rewrite.j2
"""

from __future__ import annotations

from typing import Any

import structlog

import config
from core.langgraph.state import Contradiction, NarrativeState
from core.llm_interface_refactored import llm_service
from processing.text_deduplicator import TextDeduplicator
from prompts.prompt_data_getters import get_reliable_kg_facts_for_drafting_prompt
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)


async def revise_chapter(state: NarrativeState) -> NarrativeState:
    """
    Revise chapter based on validation feedback.

    This is the main LangGraph node function for chapter revision.
    It takes contradictions from validation, constructs a revision prompt,
    calls the LLM to generate a revised version, and updates the state.

    PORTED FROM: RevisionAgent methods

    Process Flow:
    1. Check iteration limits (prevent infinite loops)
    2. Build revision prompt with contradictions and feedback
    3. Call LLM with revision model (lower temperature for consistency)
    4. Update state with revised text
    5. Clear contradictions for re-validation
    6. Increment iteration count

    Args:
        state: Current narrative state containing draft_text and contradictions

    Returns:
        Updated state with revised draft_text and incremented iteration_count
    """
    logger.info(
        "revise_chapter: starting revision",
        chapter=state["current_chapter"],
        iteration=state["iteration_count"],
        contradictions=len(state.get("contradictions", [])),
    )

    # Step 1: Check iteration limit
    if state["iteration_count"] >= state["max_iterations"]:
        error_msg = f"Max revision attempts ({state['max_iterations']}) reached"
        logger.error(
            "revise_chapter: iteration limit exceeded",
            iterations=state["iteration_count"],
            max_iterations=state["max_iterations"],
        )
        return {
            **state,
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "revise",
            "needs_revision": False,
            "current_node": "revise",
        }

    # Validate we have text to revise
    if not state.get("draft_text"):
        error_msg = "No draft text available for revision"
        logger.error("revise_chapter: fatal error", error=error_msg)
        return {
            **state,
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "revise",
            "current_node": "revise",
        }

    # Step 2: Build revision prompt
    try:
        prompt = await _construct_revision_prompt(
            draft_text=state["draft_text"],
            contradictions=state.get("contradictions", []),
            chapter_number=state["current_chapter"],
            plot_outline=state.get("plot_outline", {}),
            hybrid_context=state.get("hybrid_context"),
            novel_title=state["title"],
            novel_genre=state["genre"],
            protagonist_name=state.get(
                "protagonist_name", config.DEFAULT_PROTAGONIST_NAME
            ),
        )
    except Exception as e:
        error_msg = f"Revision prompt construction failed: {str(e)}"
        logger.error(
            "revise_chapter: fatal error",
            error=error_msg,
            exc_info=True,
        )
        return {
            **state,
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "revise",
            "current_node": "revise",
        }

    # Step 3: Calculate token budget
    model_name = state["revision_model"]
    prompt_tokens = llm_service.count_tokens(prompt, model_name)

    max_context = getattr(config, "MAX_CONTEXT_TOKENS", 128000)
    token_buffer = getattr(config.settings, "NARRATIVE_TOKEN_BUFFER", 2000)
    max_generation = getattr(config, "MAX_GENERATION_TOKENS", 8000)

    available_tokens = max_context - prompt_tokens - token_buffer
    max_gen_tokens = min(max_generation, available_tokens)

    if max_gen_tokens < 500:
        error_msg = (
            f"Insufficient token space for revision. "
            f"Prompt tokens: {prompt_tokens}, available: {available_tokens}"
        )
        logger.error("revise_chapter: fatal error - token budget exceeded", error=error_msg)
        return {
            **state,
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "revise",
            "current_node": "revise",
        }

    # Step 4: Call revision model (lower temperature for consistency)
    logger.info(
        "revise_chapter: calling LLM for revision",
        chapter=state["current_chapter"],
        model=model_name,
        iteration=state["iteration_count"] + 1,
        max_tokens=max_gen_tokens,
    )

    try:
        revised_text, usage = await llm_service.async_call_llm(
            model_name=model_name,
            prompt=prompt,
            temperature=getattr(
                config.Temperatures, "REVISION", 0.5
            ),  # Lower temp for consistency
            max_tokens=max_gen_tokens,
            allow_fallback=True,
            stream_to_disk=False,
            frequency_penalty=getattr(config, "FREQUENCY_PENALTY_DRAFTING", 0.3),
            presence_penalty=getattr(config, "PRESENCE_PENALTY_DRAFTING", 0.3),
            auto_clean_response=True,
            system_prompt=get_system_prompt("revision_agent"),
        )

        if not revised_text or not revised_text.strip():
            logger.error(
                "revise_chapter: LLM returned empty text",
                chapter=state["current_chapter"],
            )
            return {
                **state,
                "last_error": "Revision LLM returned empty text",
                "current_node": "revise",
            }

        # Step 5: Update state
        word_count = len(revised_text.split())

        logger.info(
            "revise_chapter: revision complete",
            chapter=state["current_chapter"],
            iteration=state["iteration_count"] + 1,
            word_count=word_count,
            tokens_used=usage.get("total_tokens", 0) if usage else 0,
        )

        # Step 6: Deduplicate text to remove repetitive segments
        deduplicator = TextDeduplicator()
        deduplicated_text, removed_chars = await deduplicator.deduplicate(
            revised_text, segment_level="paragraph"
        )

        # Track if deduplication modified text (signals potentially flawed extraction)
        # Preserve existing flag if already set, or set based on this revision
        is_from_flawed_draft = state.get("is_from_flawed_draft", False) or (
            removed_chars > 0
        )

        if removed_chars > 0:
            final_word_count = len(deduplicated_text.split())
            logger.info(
                "revise_chapter: deduplication applied",
                chapter=state["current_chapter"],
                iteration=state["iteration_count"] + 1,
                chars_removed=removed_chars,
                original_words=word_count,
                final_words=final_word_count,
                is_from_flawed_draft=True,
            )
        else:
            deduplicated_text = revised_text
            final_word_count = word_count
            logger.info(
                "revise_chapter: no duplicates detected",
                chapter=state["current_chapter"],
                iteration=state["iteration_count"] + 1,
                is_from_flawed_draft=is_from_flawed_draft,
            )

        return {
            **state,
            "draft_text": deduplicated_text,
            "draft_word_count": final_word_count,
            "is_from_flawed_draft": is_from_flawed_draft,
            "iteration_count": state["iteration_count"] + 1,
            "contradictions": [],  # Will be re-validated
            "needs_revision": False,  # Reset flag, will be set again by validation if needed
            "current_node": "revise",
            "last_error": None,
        }

    except Exception as e:
        error_msg = f"Revision failed: {str(e)}"
        logger.error(
            "revise_chapter: fatal error",
            error=error_msg,
            chapter=state["current_chapter"],
            exc_info=True,
        )
        return {
            **state,
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "revise",
            "current_node": "revise",
        }


async def _construct_revision_prompt(
    *,
    draft_text: str,
    contradictions: list[Contradiction],
    chapter_number: int,
    plot_outline: dict[str, Any],
    hybrid_context: str | None,
    novel_title: str,
    novel_genre: str,
    protagonist_name: str,
) -> str:
    """
    Construct the revision prompt for chapter rewriting.

    This uses the existing Jinja2 template system to build a comprehensive
    revision prompt with all contradictions and context.

    REUSES: prompts/revision_agent/full_chapter_rewrite.j2

    Args:
        draft_text: Current chapter text to revise
        contradictions: List of Contradiction objects from validation
        chapter_number: Chapter number being revised
        plot_outline: Plot outline dictionary
        hybrid_context: Combined context from KG and previous chapters
        novel_title: Title of the novel
        novel_genre: Genre of the novel
        protagonist_name: Name of the protagonist

    Returns:
        Rendered prompt string ready for LLM
    """
    # Build revision reason from contradictions
    revision_reason = _format_contradictions_for_prompt(contradictions)

    # Get chapter focus/plot point if available
    chapter_outline = plot_outline.get(chapter_number, {})
    if isinstance(chapter_outline, dict):
        plot_point_focus = chapter_outline.get("plot_point", "")
    else:
        plot_point_focus = ""

    # Build plan focus section if we have a plot point
    plan_focus_section = ""
    if plot_point_focus:
        plan_focus_section = f"""**Original Chapter Focus:**
Plot Point: {plot_point_focus}

Please ensure your revision stays true to this plot point while addressing all issues."""

    # Get KG facts for context (if not already in hybrid_context)
    if not hybrid_context:
        try:
            kg_facts = await get_reliable_kg_facts_for_drafting_prompt(
                plot_outline, chapter_number, None
            )
            hybrid_context = kg_facts if kg_facts else "No previous context available."
        except Exception as e:
            logger.warning(
                "revise_chapter: failed to fetch KG facts",
                error=str(e),
            )
            hybrid_context = "Context unavailable."

    # Build detailed problem descriptions
    all_problem_descriptions = ""
    if contradictions:
        all_problem_descriptions = """**Detailed Issues to Address:**
"""
        for i, contradiction in enumerate(contradictions, 1):
            severity_label = contradiction.severity.upper()
            all_problem_descriptions += (
                f"{i}. [{severity_label}] {contradiction.description}\n"
            )
            if contradiction.suggested_fix:
                all_problem_descriptions += (
                    f"   Suggested fix: {contradiction.suggested_fix}\n"
                )

    # Note about length requirements
    min_length = getattr(config, "MIN_ACCEPTABLE_DRAFT_LENGTH", 8000)
    length_issue_explicit_instruction = ""
    if len(draft_text) < min_length:
        length_issue_explicit_instruction = f"""**LENGTH REQUIREMENT:** The original draft was too short ({len(draft_text)} chars).
Your revision MUST be at least {min_length} characters of narrative text."""

    # Render using the full chapter rewrite template
    prompt = render_prompt(
        "revision_agent/full_chapter_rewrite.j2",
        {
            "chapter_number": chapter_number,
            "protagonist_name": protagonist_name,
            "revision_reason": revision_reason,
            "all_problem_descriptions": all_problem_descriptions,
            "deduplication_note": "",  # Not applicable in this context
            "length_issue_explicit_instruction": length_issue_explicit_instruction,
            "plan_focus_section": plan_focus_section,
            "hybrid_context_for_revision": hybrid_context,
            "original_snippet": draft_text,  # Full text, not just snippet
            "genre": novel_genre,
            "min_acceptable_draft_length": min_length,
        },
    )

    return prompt


def _format_contradictions_for_prompt(contradictions: list[Contradiction]) -> str:
    """
    Format contradictions into a human-readable revision reason.

    Args:
        contradictions: List of Contradiction objects

    Returns:
        Formatted string describing all contradictions
    """
    if not contradictions:
        return "General quality improvements needed."

    # Group by severity
    critical = [c for c in contradictions if c.severity == "critical"]
    major = [c for c in contradictions if c.severity == "major"]
    minor = [c for c in contradictions if c.severity == "minor"]

    parts = []

    if critical:
        parts.append("**CRITICAL ISSUES:**")
        for c in critical:
            parts.append(f"- {c.description}")

    if major:
        parts.append("\n**MAJOR ISSUES:**")
        for c in major:
            parts.append(f"- {c.description}")

    if minor:
        parts.append("\n**MINOR ISSUES:**")
        for c in minor:
            parts.append(f"- {c.description}")

    return "\n".join(parts)


__all__ = ["revise_chapter"]
