# core/langgraph/initialization/global_outline_node.py
"""
Global Outline Generation Node for Initialization Phase.

This node generates a high-level story outline that defines the overall
narrative arc, key plot points, and story structure.
"""

from __future__ import annotations

import structlog

from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)


async def generate_global_outline(state: NarrativeState) -> NarrativeState:
    """
    Generate a global story outline covering the entire narrative arc.

    This node creates a high-level outline that defines:
    - Overall story structure (acts/parts)
    - Major plot points and turning points
    - Character arcs at a high level
    - Thematic progression
    - Beginning, middle, and end structure

    The global outline uses character sheets and story metadata to ensure
    coherence with established characters and themes.

    Args:
        state: Current narrative state with character sheets

    Returns:
        Updated state with global_outline populated
    """
    logger.info(
        "generate_global_outline: starting outline generation",
        title=state["title"],
        total_chapters=state.get("total_chapters", 0),
    )

    # Validate inputs
    if not state.get("character_sheets"):
        logger.warning(
            "generate_global_outline: no character sheets available, "
            "generating without character context"
        )

    # Build character context for outline generation
    character_context = _build_character_context(state)

    # Step 1: Generate global outline
    prompt = render_prompt(
        "initialization/generate_global_outline.j2",
        {
            "title": state["title"],
            "genre": state["genre"],
            "theme": state.get("theme", ""),
            "setting": state.get("setting", ""),
            "target_word_count": state.get("target_word_count", 80000),
            "total_chapters": state.get("total_chapters", 20),
            "protagonist_name": state.get("protagonist_name", ""),
            "character_context": character_context,
            "character_names": list(state.get("character_sheets", {}).keys()),
        },
    )

    try:
        response, usage = await llm_service.async_call_llm(
            model_name=state["generation_model"],
            prompt=prompt,
            temperature=0.7,
            max_tokens=4000,
            allow_fallback=True,
            auto_clean_response=True,
            system_prompt=get_system_prompt("initialization"),
        )

        if not response or not response.strip():
            error_msg = "LLM returned empty global outline"
            logger.error("generate_global_outline: empty response")
            return {
                **state,
                "last_error": error_msg,
                "current_node": "global_outline",
                "initialization_step": "global_outline_failed",
            }

        # Parse outline structure
        global_outline = _parse_global_outline(response, state)

        logger.info(
            "generate_global_outline: generation complete",
            outline_length=len(response),
            acts=global_outline.get("act_count", 0),
        )

        return {
            **state,
            "global_outline": global_outline,
            "current_node": "global_outline",
            "last_error": None,
            "initialization_step": "global_outline_complete",
        }

    except Exception as e:
        error_msg = f"Error generating global outline: {str(e)}"
        logger.error(
            "generate_global_outline: exception during generation",
            error=str(e),
            exc_info=True,
        )
        return {
            **state,
            "last_error": error_msg,
            "current_node": "global_outline",
            "initialization_step": "global_outline_failed",
        }


def _build_character_context(state: NarrativeState) -> str:
    """
    Build a concise character context string for outline generation.

    Args:
        state: Current narrative state with character sheets

    Returns:
        Formatted string summarizing main characters
    """
    character_sheets = state.get("character_sheets", {})
    if not character_sheets:
        return "No characters defined yet."

    context_parts = []
    for name, sheet in character_sheets.items():
        is_protag = sheet.get("is_protagonist", False)
        role = "Protagonist" if is_protag else "Main Character"
        # Truncate description to first 200 chars for context
        desc = sheet.get("description", "")[:200]
        context_parts.append(f"**{name}** ({role}): {desc}...")

    return "\n\n".join(context_parts)


def _parse_global_outline(response: str, state: NarrativeState) -> dict[str, any]:
    """
    Parse the LLM response into a structured global outline.

    This is a simple parser. In production, you might want to use
    structured output or JSON mode for more reliable parsing.

    Args:
        response: LLM-generated outline text
        state: Current narrative state

    Returns:
        Dictionary containing structured outline data
    """
    # For now, store the raw outline text and infer structure
    # In future iterations, enhance with JSON parsing

    # Try to detect act structure from the response
    act_count = 3  # Default three-act structure
    if "act 1" in response.lower() or "act i" in response.lower():
        if "act 5" in response.lower() or "act v" in response.lower():
            act_count = 5
        elif "act 4" in response.lower() or "act iv" in response.lower():
            act_count = 4

    global_outline = {
        "raw_text": response,
        "act_count": act_count,
        "total_chapters": state.get("total_chapters", 20),
        "structure_type": f"{act_count}-act",
        "generated_at": "initialization",
    }

    logger.debug(
        "_parse_global_outline: parsed outline",
        act_count=act_count,
        length=len(response),
    )

    return global_outline


__all__ = ["generate_global_outline"]
