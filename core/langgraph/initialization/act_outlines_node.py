# core/langgraph/initialization/act_outlines_node.py
"""
Act Outlines Generation Node for Initialization Phase.

This node generates detailed outlines for each act based on the global outline,
providing more granular structure for the narrative.
"""

from __future__ import annotations

import structlog

from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)


async def generate_act_outlines(state: NarrativeState) -> NarrativeState:
    """
    Generate detailed outlines for each act in the story.

    This node takes the global outline and expands it into detailed act-level
    outlines that specify:
    - Key events and plot points within each act
    - Character development within the act
    - Pacing and tension progression
    - Act-specific goals and resolutions

    Each act outline references the global outline and character sheets to
    maintain coherence.

    Args:
        state: Current narrative state with global outline

    Returns:
        Updated state with act_outlines populated
    """
    logger.info(
        "generate_act_outlines: starting act outline generation",
        title=state["title"],
    )

    # Validate inputs
    global_outline = state.get("global_outline")
    if not global_outline:
        error_msg = "No global outline available for act outline generation"
        logger.error("generate_act_outlines: missing global outline")
        return {
            **state,
            "last_error": error_msg,
            "current_node": "act_outlines",
            "initialization_step": "act_outlines_failed",
        }

    act_count = global_outline.get("act_count", 3)
    total_chapters = state.get("total_chapters", 20)
    chapters_per_act = total_chapters // act_count

    logger.info(
        "generate_act_outlines: generating outlines",
        act_count=act_count,
        chapters_per_act=chapters_per_act,
    )

    # Generate outline for each act
    act_outlines = {}
    for act_num in range(1, act_count + 1):
        logger.info("generate_act_outlines: generating act", act_number=act_num)

        act_outline = await _generate_single_act_outline(
            state=state,
            act_number=act_num,
            total_acts=act_count,
            chapters_in_act=chapters_per_act,
        )

        if act_outline:
            act_outlines[act_num] = act_outline
        else:
            logger.warning(
                "generate_act_outlines: failed to generate act",
                act_number=act_num,
            )

    if not act_outlines:
        error_msg = "Failed to generate any act outlines"
        logger.error("generate_act_outlines: no outlines generated")
        return {
            **state,
            "last_error": error_msg,
            "current_node": "act_outlines",
            "initialization_step": "act_outlines_failed",
        }

    logger.info(
        "generate_act_outlines: generation complete",
        act_count=len(act_outlines),
    )

    return {
        **state,
        "act_outlines": act_outlines,
        "current_node": "act_outlines",
        "last_error": None,
        "initialization_step": "act_outlines_complete",
    }


async def _generate_single_act_outline(
    state: NarrativeState,
    act_number: int,
    total_acts: int,
    chapters_in_act: int,
) -> dict[str, any] | None:
    """
    Generate an outline for a single act.

    Args:
        state: Current narrative state
        act_number: Act number to generate (1-indexed)
        total_acts: Total number of acts in the story
        chapters_in_act: Approximate number of chapters in this act

    Returns:
        Dictionary containing act outline details or None on failure
    """
    global_outline = state.get("global_outline", {})
    character_sheets = state.get("character_sheets", {})

    # Determine act role in story structure
    act_role = _get_act_role(act_number, total_acts)

    # Build context strings
    character_context = _build_character_summary(character_sheets)
    global_outline_text = global_outline.get("raw_text", "")

    prompt = render_prompt(
        "initialization/generate_act_outline.j2",
        {
            "title": state["title"],
            "genre": state["genre"],
            "theme": state.get("theme", ""),
            "setting": state.get("setting", ""),
            "act_number": act_number,
            "total_acts": total_acts,
            "act_role": act_role,
            "chapters_in_act": chapters_in_act,
            "global_outline": global_outline_text[:2000],  # Truncate if too long
            "character_context": character_context,
            "protagonist_name": state.get("protagonist_name", ""),
        },
    )

    try:
        response, usage = await llm_service.async_call_llm(
            model_name=state["generation_model"],
            prompt=prompt,
            temperature=0.7,
            max_tokens=3000,
            allow_fallback=True,
            auto_clean_response=True,
            system_prompt=get_system_prompt("initialization"),
        )

        if not response or not response.strip():
            logger.error(
                "_generate_single_act_outline: empty response",
                act_number=act_number,
            )
            return None

        act_outline = {
            "act_number": act_number,
            "raw_text": response,
            "chapters_in_act": chapters_in_act,
            "act_role": act_role,
            "generated_at": "initialization",
        }

        logger.debug(
            "_generate_single_act_outline: act generated",
            act_number=act_number,
            length=len(response),
        )

        return act_outline

    except Exception as e:
        logger.error(
            "_generate_single_act_outline: exception during generation",
            act_number=act_number,
            error=str(e),
            exc_info=True,
        )
        return None


def _get_act_role(act_number: int, total_acts: int) -> str:
    """
    Determine the narrative role of an act.

    Args:
        act_number: Act number (1-indexed)
        total_acts: Total number of acts

    Returns:
        String describing the act's role in the story structure
    """
    if act_number == 1:
        return "Setup/Introduction"
    elif act_number == total_acts:
        return "Resolution/Climax"
    elif total_acts == 3 and act_number == 2:
        return "Confrontation/Rising Action"
    elif total_acts == 5:
        if act_number == 2:
            return "Rising Action"
        elif act_number == 3:
            return "Midpoint/Crisis"
        elif act_number == 4:
            return "Falling Action"
    else:
        return "Development"


def _build_character_summary(character_sheets: dict[str, dict]) -> str:
    """
    Build a concise summary of characters for act outline generation.

    Args:
        character_sheets: Dictionary of character sheets

    Returns:
        Formatted string summarizing characters
    """
    if not character_sheets:
        return "No characters defined."

    summaries = []
    for name, sheet in character_sheets.items():
        is_protag = sheet.get("is_protagonist", False)
        role = "Protagonist" if is_protag else "Character"
        # Get first sentence or 100 chars of description
        desc = sheet.get("description", "")
        first_sentence = desc.split(".")[0] if desc else "No description"
        summaries.append(f"- **{name}** ({role}): {first_sentence[:100]}")

    return "\n".join(summaries)


__all__ = ["generate_act_outlines"]
