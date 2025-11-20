# core/langgraph/initialization/character_sheets_node.py
"""
Character Sheets Generation Node for Initialization Phase.

This node generates detailed character sheets during the initialization phase,
creating comprehensive profiles for main characters that will be used throughout
the narrative generation process.
"""

from __future__ import annotations

import structlog

from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)


async def generate_character_sheets(state: NarrativeState) -> NarrativeState:
    """
    Generate detailed character sheets for main characters.

    This node creates comprehensive character profiles based on the story's
    genre, theme, setting, and protagonist. These sheets will inform character
    behavior and development throughout the narrative.

    Process Flow:
    1. Generate character list based on story parameters
    2. For each character, generate a detailed character sheet
    3. Store character sheets in state for later use

    Args:
        state: Current narrative state with story metadata

    Returns:
        Updated state with character_sheets populated
    """
    logger.info(
        "generate_character_sheets: starting character sheet generation",
        title=state["title"],
        genre=state["genre"],
    )

    # Validate inputs
    if not state.get("title") or not state.get("genre"):
        error_msg = "Missing required fields: title and genre"
        logger.error("generate_character_sheets: validation failed", error=error_msg)
        return {
            **state,
            "last_error": error_msg,
            "current_node": "character_sheets",
            "initialization_step": "character_sheets_failed",
        }

    # Step 1: Generate list of main characters
    logger.info("generate_character_sheets: generating character list")
    character_list = await _generate_character_list(state)

    if not character_list:
        error_msg = "Failed to generate character list"
        logger.error("generate_character_sheets: character list generation failed")
        return {
            **state,
            "last_error": error_msg,
            "current_node": "character_sheets",
            "initialization_step": "character_sheets_failed",
        }

    # Step 2: Generate detailed sheet for each character
    character_sheets = {}
    for character_name in character_list:
        logger.info(
            "generate_character_sheets: generating sheet",
            character=character_name,
        )

        sheet = await _generate_character_sheet(
            state=state,
            character_name=character_name,
            other_characters=character_list,
        )

        if sheet:
            character_sheets[character_name] = sheet
        else:
            logger.warning(
                "generate_character_sheets: failed to generate sheet",
                character=character_name,
            )

    if not character_sheets:
        error_msg = "Failed to generate any character sheets"
        logger.error("generate_character_sheets: no sheets generated")
        return {
            **state,
            "last_error": error_msg,
            "current_node": "character_sheets",
            "initialization_step": "character_sheets_failed",
        }

    logger.info(
        "generate_character_sheets: generation complete",
        character_count=len(character_sheets),
        characters=list(character_sheets.keys()),
    )

    return {
        **state,
        "character_sheets": character_sheets,
        "current_node": "character_sheets",
        "last_error": None,
        "initialization_step": "character_sheets_complete",
    }


async def _generate_character_list(state: NarrativeState) -> list[str]:
    """
    Generate a list of main character names for the story.

    Args:
        state: Current narrative state

    Returns:
        List of character names (including protagonist)
    """
    prompt = render_prompt(
        "initialization/generate_character_list.j2",
        {
            "title": state["title"],
            "genre": state["genre"],
            "theme": state.get("theme", ""),
            "setting": state.get("setting", ""),
            "protagonist_name": state.get("protagonist_name", ""),
        },
    )

    try:
        response, _ = await llm_service.async_call_llm(
            model_name=state["generation_model"],
            prompt=prompt,
            temperature=0.7,
            max_tokens=1000,
            allow_fallback=True,
            auto_clean_response=True,
            system_prompt=get_system_prompt("initialization"),
        )

        # Parse response to extract character names
        # Expected format: one character per line or comma-separated
        if not response:
            logger.error("_generate_character_list: empty response from LLM")
            return []

        # Try to parse as list
        character_names = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # Remove common prefixes and numbers
            line = line.lstrip("-*•123456789. ")
            if line and len(line) < 100:  # Sanity check
                character_names.append(line)

        # Ensure protagonist is included
        protagonist = state.get("protagonist_name", "")
        if protagonist and protagonist not in character_names:
            character_names.insert(0, protagonist)

        logger.info(
            "_generate_character_list: generated list",
            count=len(character_names),
            names=character_names,
        )

        return character_names[:10]  # Limit to 10 main characters

    except Exception as e:
        logger.error(
            "_generate_character_list: exception during generation",
            error=str(e),
            exc_info=True,
        )
        # Fallback: return just protagonist
        return [state.get("protagonist_name", "Protagonist")]


async def _generate_character_sheet(
    state: NarrativeState,
    character_name: str,
    other_characters: list[str],
) -> dict[str, any] | None:
    """
    Generate a detailed character sheet for a specific character.

    Args:
        state: Current narrative state
        character_name: Name of the character to generate sheet for
        other_characters: List of other main characters for context

    Returns:
        Dictionary containing character sheet details or None on failure
    """
    is_protagonist = character_name == state.get("protagonist_name", "")

    prompt = render_prompt(
        "initialization/generate_character_sheet.j2",
        {
            "title": state["title"],
            "genre": state["genre"],
            "theme": state.get("theme", ""),
            "setting": state.get("setting", ""),
            "character_name": character_name,
            "is_protagonist": is_protagonist,
            "other_characters": [c for c in other_characters if c != character_name],
        },
    )

    try:
        response, usage = await llm_service.async_call_llm(
            model_name=state["generation_model"],
            prompt=prompt,
            temperature=0.7,
            max_tokens=2000,
            allow_fallback=True,
            auto_clean_response=True,
            system_prompt=get_system_prompt("initialization"),
        )

        if not response:
            logger.error(
                "_generate_character_sheet: empty response",
                character=character_name,
            )
            return None

        # For now, store as text. In future, parse into structured format
        sheet = {
            "name": character_name,
            "description": response,
            "is_protagonist": is_protagonist,
            "generated_at": "initialization",
        }

        logger.debug(
            "_generate_character_sheet: sheet generated",
            character=character_name,
            length=len(response),
        )

        return sheet

    except Exception as e:
        logger.error(
            "_generate_character_sheet: exception during generation",
            character=character_name,
            error=str(e),
            exc_info=True,
        )
        return None


__all__ = ["generate_character_sheets"]
