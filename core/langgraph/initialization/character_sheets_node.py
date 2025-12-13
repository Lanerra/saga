# core/langgraph/initialization/character_sheets_node.py
"""
Character Sheets Generation Node for Initialization Phase.

This node generates detailed character sheets during the initialization phase,
creating comprehensive profiles for main characters that will be used throughout
the narrative generation process.
"""

from __future__ import annotations

import re

import structlog

from core.db_manager import neo4j_manager
from core.langgraph.content_manager import ContentManager
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from core.schema_validator import schema_validator
from prompts.grammar_loader import load_grammar
from prompts.prompt_renderer import get_system_prompt, render_prompt
from utils.text_processing import validate_and_filter_traits

logger = structlog.get_logger(__name__)


async def _get_existing_traits() -> list[str]:
    """
    Fetch existing trait names from the database to encourage reuse.

    Returns:
        List of existing trait names (normalized)
    """
    try:
        query = """
        MATCH (t:Trait)
        RETURN DISTINCT t.name AS trait_name
        ORDER BY trait_name
        LIMIT 100
        """
        results = await neo4j_manager.execute_read_query(query)
        if results:
            traits = [r["trait_name"] for r in results if r.get("trait_name")]
            logger.info(
                "_get_existing_traits: fetched existing traits",
                count=len(traits),
            )
            return traits
        return []
    except Exception as e:
        logger.warning(
            "_get_existing_traits: failed to fetch existing traits",
            error=str(e),
        )
        return []


def _parse_character_sheet_response(response: str, character_name: str) -> dict[str, any]:
    """
    Parse the structured character sheet response into CharacterProfile-compatible format.
    Refactored to handle JSON response enforced by GBNF grammar.

    Args:
        response: Raw LLM response with structured character data (JSON)
        character_name: Name of the character

    Returns:
        Dictionary with CharacterProfile-compatible fields
    """
    import json

    # Defaults
    parsed = {
        "name": character_name,
        "description": "",
        "traits": [],
        "type": "Character",  # Force type to "Character"
        "status": "Active",
        "motivations": "",
        "background": "",
        "skills": [],
        "relationships": {},
        "internal_conflict": "",
    }

    try:
        # Clean potential markdown
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]

        data = json.loads(cleaned_response)

        # Merge data into defaults
        parsed.update(data)

        # Ensure name matches requested character if not provided or empty
        if not parsed.get("name"):
            parsed["name"] = character_name

        # Validate and filter traits to ensure single-word format
        raw_traits = parsed.get("traits", [])
        parsed["traits"] = validate_and_filter_traits(raw_traits)

        if len(parsed["traits"]) != len(raw_traits):
            logger.warning(
                "_parse_character_sheet_response: filtered invalid traits",
                character=character_name,
                original_count=len(raw_traits),
                filtered_count=len(parsed["traits"]),
                removed=set(raw_traits) - set(parsed["traits"]),
            )

        # Transform relationships if needed to internal structure
        structured_relationships = {}
        if isinstance(parsed.get("relationships"), dict):
            for target, desc in parsed["relationships"].items():
                # Try to extract type from description if possible, or default
                structured_relationships[target] = {
                    "type": "ASSOCIATE",  # Default
                    "description": desc,
                }
            parsed["relationships"] = structured_relationships

        # Double check that we are using a valid type (should be 'Character')
        is_valid, normalized, err = schema_validator.validate_entity_type("Character")
        parsed["type"] = normalized if is_valid else "Character"

    except json.JSONDecodeError as e:
        logger.warning(
            "_parse_character_sheet_response: JSON parsing failed",
            character=character_name,
            error=str(e),
        )
        # Fallback to description = raw response
        parsed["description"] = response

    return parsed


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
        title=state.get("title", ""),
        genre=state.get("genre", ""),
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

    # Step 0: Fetch existing traits to encourage reuse
    logger.info("generate_character_sheets: fetching existing traits")
    existing_traits = await _get_existing_traits()

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
            existing_traits=existing_traits,
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

    # Initialize content manager for external storage
    content_manager = ContentManager(state.get("project_dir", ""))

    # Externalize character_sheets to reduce state bloat
    character_sheets_ref = content_manager.save_json(
        character_sheets,
        "character_sheets",
        "all",
        version=1,
    )

    logger.info(
        "generate_character_sheets: content externalized",
        size=character_sheets_ref["size_bytes"],
    )

    return {
        **state,
        "character_sheets_ref": character_sheets_ref,
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
            "title": state.get("title", ""),
            "genre": state.get("genre", ""),
            "theme": state.get("theme", ""),
            "setting": state.get("setting", ""),
            "protagonist_name": state.get("protagonist_name", ""),
        },
    )

    try:
        response, _ = await llm_service.async_call_llm(
            model_name=state.get("large_model", ""),
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
    existing_traits: list[str] | None = None,
) -> dict[str, any] | None:
    """
    Generate a detailed character sheet for a specific character.

    Args:
        state: Current narrative state
        character_name: Name of the character to generate sheet for
        other_characters: List of other main characters for context
        existing_traits: List of existing traits in the database to encourage reuse

    Returns:
        Dictionary containing character sheet details or None on failure
    """
    is_protagonist = character_name == state.get("protagonist_name", "")

    # Prepare existing traits hint for the prompt
    existing_traits_hint = ""
    if existing_traits:
        traits_sample = existing_traits[:40]  # Limit to 20 examples
        existing_traits_hint = f"\n\nExisting traits in the story (consider reusing to create interconnectedness): " f"{', '.join(traits_sample)}"

    prompt = render_prompt(
        "initialization/generate_character_sheet.j2",
        {
            "title": state.get("title", ""),
            "genre": state.get("genre", ""),
            "theme": state.get("theme", ""),
            "setting": state.get("setting", ""),
            "character_name": character_name,
            "is_protagonist": is_protagonist,
            "other_characters": [c for c in other_characters if c != character_name],
            "existing_traits_hint": existing_traits_hint,
        },
    )

    # Load and configure grammar
    grammar = load_grammar("initialization")
    # Enforce character_sheet as root by replacing the default root
    grammar = re.sub(r"^root ::= .*$", "", grammar, flags=re.MULTILINE)
    # Be sure to include the replaced root at the top
    grammar = f"root ::= character-sheet\n{grammar}"

    try:
        response, usage = await llm_service.async_call_llm(
            model_name=state.get("large_model", ""),
            prompt=prompt,
            temperature=0.7,
            max_tokens=2000,
            allow_fallback=True,
            auto_clean_response=True,
            system_prompt=get_system_prompt("initialization"),
            grammar=grammar,
        )

        if not response:
            logger.error(
                "_generate_character_sheet: empty response",
                character=character_name,
            )
            return None

        # Parse the structured response into CharacterProfile-compatible format
        sheet = _parse_character_sheet_response(response, character_name)

        # Add metadata
        sheet["is_protagonist"] = is_protagonist
        sheet["generated_at"] = "initialization"
        sheet["raw_response"] = response  # Keep raw response for reference

        logger.debug(
            "_generate_character_sheet: sheet generated",
            character=character_name,
            traits_count=len(sheet.get("traits", [])),
            has_relationships=bool(sheet.get("relationships")),
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
