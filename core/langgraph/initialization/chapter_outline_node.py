# core/langgraph/initialization/chapter_outline_node.py
"""
Chapter Outline Generation Node (On-Demand).

This node generates a detailed outline for a specific chapter based on the
global outline, act outlines, and character sheets. It can be called on-demand
before generating each chapter.
"""

from __future__ import annotations

import structlog

from core.langgraph.content_manager import (
    ContentManager,
    get_act_outlines,
    get_character_sheets,
    get_chapter_outlines,
    get_global_outline,
    get_previous_summaries,
)
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)


async def generate_chapter_outline(state: NarrativeState) -> NarrativeState:
    """
    Generate a detailed outline for the current chapter.

    This node is designed to be called on-demand before chapter generation,
    creating a specific chapter-level outline that includes:
    - Scene breakdown
    - Key events and beats
    - Character interactions
    - Pacing notes
    - Connection to previous and next chapters

    The chapter outline uses all available context (character sheets, global
    outline, act outlines) to maintain narrative coherence.

    Args:
        state: Current narrative state with current_chapter set

    Returns:
        Updated state with chapter outline added to chapter_outlines dict
    """
    chapter_number = state.get("current_chapter", 1)

    logger.info(
        "generate_chapter_outline: starting chapter outline generation",
        chapter=chapter_number,
        title=state["title"],
    )

    # Initialize content manager for reading externalized content
    content_manager = ContentManager(state["project_dir"])

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
        logger.warning(
            "generate_chapter_outline: no global outline available, "
            "generating with limited context"
        )

    if not act_outlines:
        logger.warning(
            "generate_chapter_outline: no act outlines available, "
            "generating with limited context"
        )

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

    logger.info(
        "generate_chapter_outline: generation complete",
        chapter=chapter_number,
        act=act_number,
    )

    # Get current version (for revision tracking)
    current_version = content_manager.get_latest_version("chapter_outlines", "all") + 1

    # Externalize chapter_outlines to reduce state bloat
    chapter_outlines_ref = content_manager.save_json(
        updated_outlines,
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
) -> dict[str, any] | None:
    """
    Generate an outline for a single chapter.

    Args:
        state: Current narrative state
        chapter_number: Chapter number to generate outline for
        act_number: Act number this chapter belongs to

    Returns:
        Dictionary containing chapter outline details or None on failure
    """
    # Initialize content manager for reading externalized content
    content_manager = ContentManager(state["project_dir"])

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
    previous_context = (
        "\n".join(previous_summaries[-3:])
        if previous_summaries
        else "This is the beginning of the story."
    )

    # Determine chapter position in act
    total_chapters = state.get("total_chapters", 20)
    act_count = global_outline.get("act_count", 3)
    chapters_per_act = total_chapters // act_count
    chapter_in_act = ((chapter_number - 1) % chapters_per_act) + 1

    prompt = render_prompt(
        "initialization/generate_chapter_outline.j2",
        {
            "title": state["title"],
            "genre": state["genre"],
            "theme": state.get("theme", ""),
            "setting": state.get("setting", ""),
            "chapter_number": chapter_number,
            "act_number": act_number,
            "chapter_in_act": chapter_in_act,
            "total_chapters": total_chapters,
            "global_outline": global_outline.get("raw_text", "")[:1000],
            "act_outline": act_outline_text[:1500],
            "character_context": character_context,
            "previous_context": previous_context,
            "protagonist_name": state.get("protagonist_name", ""),
        },
    )

    try:
        response, usage = await llm_service.async_call_llm(
            model_name=state["large_model"],
            prompt=prompt,
            temperature=0.7,
            max_tokens=2000,
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
    content_manager = ContentManager(state["project_dir"])
    global_outline = get_global_outline(state, content_manager) or {}
    act_count = global_outline.get("act_count", 3)

    chapters_per_act = total_chapters / act_count
    act_number = int((chapter_number - 1) / chapters_per_act) + 1

    # Ensure we don't exceed act count
    act_number = min(act_number, act_count)

    return act_number


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
    for name, sheet in list(character_sheets.items())[:5]:  # Top 5 characters
        is_protag = sheet.get("is_protagonist", False)
        role = "Protagonist" if is_protag else "Character"
        summaries.append(f"- **{name}** ({role})")

    return "\n".join(summaries)


def _parse_chapter_outline(
    response: str,
    chapter_number: int,
    act_number: int,
) -> dict[str, any]:
    """
    Parse the LLM response into a structured chapter outline.

    Args:
        response: LLM-generated chapter outline text
        chapter_number: Chapter number
        act_number: Act number

    Returns:
        Dictionary containing structured chapter outline
    """
    # Simple parsing - extract key sections
    # In production, could use more sophisticated parsing or JSON mode

    lines = response.split("\n")
    scene_description = ""
    key_beats = []
    plot_point = ""

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
            elif current_section == "beats" and line.strip().startswith(
                ("-", "*", "•")
            ):
                key_beats.append(line.strip().lstrip("-*• "))
            elif current_section == "plot" and not plot_point:
                plot_point = line.strip()

    # Fallback: use full response as scene description
    if not scene_description:
        scene_description = response[:500]

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
