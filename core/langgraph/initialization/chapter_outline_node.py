# core/langgraph/initialization/chapter_outline_node.py
"""Generate chapter outlines on demand.

This module defines the on-demand chapter outline node. It produces a structured
outline for a single chapter using the global outline, act outlines, character
sheets, and recent chapter summaries, then externalizes the updated
`chapter_outlines` mapping to keep workflow state small.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

import config
from core.langgraph.content_manager import (
    ContentManager,
    get_act_outlines,
    get_chapter_outlines,
    get_character_sheets,
    get_global_outline,
    get_previous_summaries,
    require_project_dir,
)
from core.langgraph.initialization.chapter_allocation import (
    choose_act_ranges,
)
from core.langgraph.initialization.chapter_allocation import (
    determine_act_for_chapter as determine_act_for_chapter_from_outline,
)
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)


async def generate_chapter_outline(state: NarrativeState) -> NarrativeState:
    """Generate and externalize a detailed outline for the current chapter.

    Args:
        state: Workflow state. Requires `current_chapter` and uses initialization
            artifacts (global outline, act outlines, character sheets) when available.

    Returns:
        Updated state containing:
        - chapter_outlines_ref: Externalized outline mapping including the new chapter.
        - current_node: `"chapter_outline"`.
        - initialization_step: Chapter-specific progress marker.

        If the outline already exists, returns a no-op update.
        If generation fails, returns an error update without setting `has_fatal_error`.

    Notes:
        This node performs LLM I/O and writes the updated outline mapping to disk via
        `ContentManager.save_json()`.
    """
    chapter_number = state.get("current_chapter", 1)

    logger.info(
        "generate_chapter_outline: starting chapter outline generation",
        chapter=chapter_number,
        title=state.get("title", ""),
    )

    # Initialize content manager for reading externalized content
    content_manager = ContentManager(require_project_dir(state))

    # Get chapter outlines (prefers externalized content, falls back to in-state)
    existing_outlines = get_chapter_outlines(state, content_manager)

    # Check if outline already exists
    if chapter_number in existing_outlines:
        logger.info(
            "generate_chapter_outline: outline already exists",
            chapter=chapter_number,
        )
        return {
            **state,
            "current_node": "chapter_outline",
            "initialization_step": f"chapter_outline_{chapter_number}_exists",
        }

    # Get global outline and act outlines for validation
    global_outline = get_global_outline(state, content_manager)
    act_outlines = get_act_outlines(state, content_manager)

    # Validate inputs
    if not global_outline:
        logger.warning("generate_chapter_outline: no global outline available, " "generating with limited context")

    if not act_outlines:
        logger.warning("generate_chapter_outline: no act outlines available, " "generating with limited context")

    # Determine which act this chapter belongs to
    act_number = _determine_act_for_chapter(state, chapter_number)

    # Generate the chapter outline
    chapter_outline = await _generate_single_chapter_outline(
        state=state,
        chapter_number=chapter_number,
        act_number=act_number,
    )

    if not chapter_outline:
        error_msg = f"Failed to generate outline for chapter {chapter_number}"
        logger.error(
            "generate_chapter_outline: generation failed",
            chapter=chapter_number,
        )
        return {
            **state,
            "last_error": error_msg,
            "current_node": "chapter_outline",
            "initialization_step": f"chapter_outline_{chapter_number}_failed",
        }

    # Update chapter_outlines dict (canonical source of truth)
    updated_outlines = {**existing_outlines, chapter_number: chapter_outline}

    # ContentManager.save_json expects JSON-serializable dicts (JSON object keys are strings).
    outlines_for_storage: dict[str, Any] = {str(chapter): outline for chapter, outline in updated_outlines.items()}

    logger.info(
        "generate_chapter_outline: generation complete",
        chapter=chapter_number,
        act=act_number,
    )

    # Get current version (for revision tracking)
    current_version = content_manager.get_latest_version("chapter_outlines", "all") + 1

    # Externalize chapter_outlines to reduce state bloat
    chapter_outlines_ref = content_manager.save_json(
        outlines_for_storage,
        "chapter_outlines",
        "all",
        current_version,
    )

    logger.info(
        "generate_chapter_outline: content externalized",
        chapter=chapter_number,
        version=current_version,
        size=chapter_outlines_ref["size_bytes"],
    )

    return {
        **state,
        "chapter_outlines_ref": chapter_outlines_ref,
        "current_node": "chapter_outline",
        "last_error": None,
        "initialization_step": f"chapter_outline_{chapter_number}_complete",
    }


async def _generate_single_chapter_outline(
    state: NarrativeState,
    chapter_number: int,
    act_number: int,
) -> dict[str, Any] | None:
    """Generate an outline payload for a single chapter.

    Args:
        state: Workflow state.
        chapter_number: Chapter number to outline.
        act_number: Act number the chapter belongs to.

    Returns:
        Chapter outline payload dictionary, or `None` when generation fails.

    Notes:
        This helper performs LLM I/O and may return `None` for best-effort failure
        handling in the parent node.
    """
    # Initialize content manager for reading externalized content
    content_manager = ContentManager(require_project_dir(state))

    # Gather context (prefers externalized content, falls back to in-state)
    global_outline = get_global_outline(state, content_manager) or {}
    act_outlines = get_act_outlines(state, content_manager)
    character_sheets = get_character_sheets(state, content_manager)
    previous_summaries = get_previous_summaries(state, content_manager)

    # Get act outline if available
    act_outline = act_outlines.get(act_number, {})
    act_outline_text = act_outline.get("raw_text", "")

    # Build character context
    character_context = _build_character_summary(character_sheets)

    # Build previous context
    previous_context = "\n".join(previous_summaries[-3:]) if previous_summaries else "This is the beginning of the story."

    # Determine chapter position in act.
    #
    # LANGGRAPH-016: avoid modulo/div-by-zero when act_count > total_chapters
    # (which makes chapters_per_act == 0 under integer division).
    #
    # Prefer explicit per-act ranges from the global outline when present; otherwise
    # fall back to a balanced allocation that covers all chapters exactly once.
    total_chapters = state.get("total_chapters", 20)
    act_ranges = choose_act_ranges(global_outline=global_outline, total_chapters=total_chapters)
    act_range = act_ranges.get(act_number)

    if act_range and act_range.contains(chapter_number):
        chapter_in_act = (chapter_number - act_range.chapters_start) + 1
    else:
        # If the act range is empty or the chapter is out-of-range, don't crash.
        # We use 0 to make the "Chapter X of this act" text visibly suspicious
        # rather than silently misleading.
        chapter_in_act = 0

    prompt = render_prompt(
        "initialization/generate_chapter_outline.j2",
        {
            "title": state.get("title", ""),
            "genre": state.get("genre", ""),
            "theme": state.get("theme", ""),
            "setting": state.get("setting", ""),
            "chapter_number": chapter_number,
            "act_number": act_number,
            "chapter_in_act": chapter_in_act,
            "total_chapters": total_chapters,
            "global_outline": global_outline.get("raw_text", ""),
            "act_outline": act_outline_text,
            "character_context": character_context,
            "previous_context": previous_context,
            "protagonist_name": state.get("protagonist_name", ""),
        },
    )

    try:
        response, usage = await llm_service.async_call_llm(
            model_name=state.get("large_model", config.LARGE_MODEL),
            prompt=prompt,
            temperature=0.7,
            max_tokens=16384,
            allow_fallback=True,
            auto_clean_response=True,
            system_prompt=get_system_prompt("initialization"),
        )

        if not response or not response.strip():
            logger.error(
                "_generate_single_chapter_outline: empty response",
                chapter=chapter_number,
            )
            return None

        # Parse the response to extract key components
        chapter_outline = _parse_chapter_outline(response, chapter_number, act_number)

        logger.debug(
            "_generate_single_chapter_outline: chapter generated",
            chapter=chapter_number,
            length=len(response),
        )

        return chapter_outline

    except Exception as e:
        logger.error(
            "_generate_single_chapter_outline: exception during generation",
            chapter=chapter_number,
            error=str(e),
            exc_info=True,
        )
        return None


def _determine_act_for_chapter(state: NarrativeState, chapter_number: int) -> int:
    """
    Determine which act a chapter belongs to.

    Args:
        state: Current narrative state
        chapter_number: Chapter number

    Returns:
        Act number (1-indexed)
    """
    total_chapters = state.get("total_chapters", 20)

    # Initialize content manager and get global outline
    content_manager = ContentManager(require_project_dir(state))
    global_outline = get_global_outline(state, content_manager) or {}

    # Prefer explicit act chapter ranges from the global outline when present,
    # otherwise fall back to a balanced allocation.
    return determine_act_for_chapter_from_outline(
        global_outline=global_outline,
        total_chapters=total_chapters,
        chapter_number=chapter_number,
    )


def _build_character_summary(character_sheets: dict[str, dict]) -> str:
    """
    Build a concise summary of characters for chapter outline generation.

    Args:
        character_sheets: Dictionary of character sheets

    Returns:
        Formatted string summarizing characters
    """
    if not character_sheets:
        return "No characters defined."

    summaries = []
    for name, sheet in list(character_sheets.items())[:3]:  # Top 3 characters
        is_protag = sheet.get("is_protagonist", False)
        role = "Protagonist" if is_protag else "Character"
        summaries.append(f"- **{name}** ({role})")

    return "\n".join(summaries)


def _parse_chapter_outline(
    response: str,
    chapter_number: int,
    act_number: int,
) -> dict[str, Any]:
    """
    Parse the LLM response into a structured chapter outline.

    Args:
        response: LLM-generated chapter outline text
        chapter_number: Chapter number
        act_number: Act number

    Returns:
        Dictionary containing structured chapter outline
    """
    scene_description = ""
    key_beats = []
    plot_point = ""

    try:
        # Clean potential markdown
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]

        # Try to parse as JSON first
        data = json.loads(cleaned_response)

        scene_description = data.get("scene_description", "")
        key_beats = data.get("key_beats", [])
        plot_point = data.get("plot_point", "")

    except json.JSONDecodeError:
        logger.warning(
            "_parse_chapter_outline: JSON parsing failed, falling back to text parsing",
            chapter=chapter_number,
        )

        # Fallback to text parsing
        lines = response.split("\n")

        # Try to extract sections
        current_section = None
        for line in lines:
            line_lower = line.lower().strip()

            if "scene" in line_lower or "summary" in line_lower:
                current_section = "scene"
            elif "beat" in line_lower or "event" in line_lower:
                current_section = "beats"
            elif "plot point" in line_lower or "focus" in line_lower:
                current_section = "plot"
            elif line.strip():
                if current_section == "scene" and not scene_description:
                    scene_description = line.strip()
                elif current_section == "beats" and line.strip().startswith(("-", "*", "•")):
                    key_beats.append(line.strip().lstrip("-*• "))
                elif current_section == "plot" and not plot_point:
                    plot_point = line.strip()

        # Fallback: use full response as scene description
        if not scene_description:
            scene_description = response
    chapter_outline = {
        "chapter_number": chapter_number,
        "act_number": act_number,
        "raw_text": response,
        "scene_description": scene_description,
        "key_beats": key_beats[:10],  # Limit to 10 beats
        "plot_point": plot_point or f"Chapter {chapter_number} events",
        "generated_at": "on_demand",
    }

    return chapter_outline


__all__ = ["generate_chapter_outline"]
