# core/langgraph/initialization/character_sheets_node.py
"""Generate character sheets during initialization.

This module defines the initialization node that generates structured character
sheets for the protagonist and other main characters. The resulting sheets are
externalized to keep workflow state small and to provide durable initialization
artifacts for later steps.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import structlog

from core.db_manager import neo4j_manager
from core.langgraph.content_manager import ContentManager
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from core.schema_validator import schema_validator
from prompts.prompt_renderer import get_system_prompt, render_prompt
from utils.common import try_load_json_from_response
from utils.text_processing import validate_and_filter_traits

logger = structlog.get_logger(__name__)


async def _get_existing_traits() -> list[str]:
    """Fetch existing trait names to encourage reuse.

    Returns:
        Trait names from Neo4j, in display form.

    Notes:
        This helper performs Neo4j I/O. Failures are treated as non-fatal and
        return an empty list so initialization can proceed.
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


def _parse_character_sheet_response(response: str, character_name: str) -> dict[str, Any]:
    """Parse a character sheet JSON response into an internal sheet dictionary.

    Args:
        response: Raw LLM response expected to contain a JSON object.
        character_name: Character name used as a fallback when the response omits it.

    Returns:
        Parsed character sheet dictionary. Traits are filtered to single-word values
        via [`validate_and_filter_traits()`](utils/text_processing.py:1), and relationship
        descriptions are normalized into the internal relationship structure.

    Raises:
        ValueError: When no valid JSON object can be parsed from the response.
    """
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

    cleaned_response = response.strip()
    parsed_json, candidates, parse_errors = try_load_json_from_response(
        response,
        expected_root=dict,
    )

    if parsed_json is None:
        try:
            json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            try:
                response_sha1 = hashlib.sha1(response.encode("utf-8")).hexdigest()[:12]
                response_len = len(response)
                cleaned_sha1 = hashlib.sha1(cleaned_response.encode("utf-8")).hexdigest()[:12]
                cleaned_len = len(cleaned_response)
                response_last_codepoint = ord(response[-1]) if response else None
                cleaned_last_codepoint = ord(cleaned_response[-1]) if cleaned_response else None
            except Exception:  # pragma: no cover
                response_sha1 = None
                response_len = None
                cleaned_sha1 = None
                cleaned_len = None
                response_last_codepoint = None
                cleaned_last_codepoint = None

            logger.warning(
                "_parse_character_sheet_response: JSON parsing failed",
                character=character_name,
                error=str(e),
                error_pos=getattr(e, "pos", None),
                error_lineno=getattr(e, "lineno", None),
                error_colno=getattr(e, "colno", None),
                response_sha1=response_sha1,
                response_len=response_len,
                cleaned_sha1=cleaned_sha1,
                cleaned_len=cleaned_len,
                has_json_fence=("```json" in response),
                has_fence=("```" in response),
                cleaned_startswith=cleaned_response[:1] if cleaned_response else None,
                cleaned_endswith=cleaned_response[-1:] if cleaned_response else None,
                response_last_codepoint=response_last_codepoint,
                cleaned_last_codepoint=cleaned_last_codepoint,
                tried_sources=[source for source, _candidate in candidates[:5]],
                parse_errors=parse_errors[:5],
            )

            raise ValueError(
                "Character sheet JSON parsing failed "
                f"(character={character_name}, response_sha1={response_sha1}, response_len={response_len})"
            ) from e

        raise ValueError(f"Character sheet JSON parsing failed (character={character_name})")

    # Merge data into defaults
    parsed.update(parsed_json)

    # Ensure name matches requested character if not provided or empty
    if not parsed.get("name"):
        parsed["name"] = character_name

    # Validate and filter traits to ensure single-word format
    raw_traits_value = parsed.get("traits", [])
    if not isinstance(raw_traits_value, list):
        raw_traits: list[str] = []
    else:
        raw_traits = [t for t in raw_traits_value if isinstance(t, str)]
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
    relationships_value = parsed.get("relationships")
    structured_relationships: dict[str, dict[str, str]] = {}
    if isinstance(relationships_value, dict):
        for target, desc in relationships_value.items():
            if not isinstance(target, str):
                continue
            if not isinstance(desc, str):
                continue
            structured_relationships[target] = {
                "type": "ASSOCIATE",
                "description": desc,
            }
        parsed["relationships"] = structured_relationships

    # Double check that we are using a valid type (should be 'Character')
    is_valid, normalized, err = schema_validator.validate_entity_type("Character")
    parsed["type"] = normalized if is_valid else "Character"

    return parsed


async def generate_character_sheets(state: NarrativeState) -> NarrativeState:
    """Generate and externalize character sheets for initialization.

    Args:
        state: Workflow state. Requires core story metadata such as `title` and `genre`.

    Returns:
        Updated state containing character sheet artifacts (typically via an
        externalized reference) and initialization progress fields.

        If required metadata is missing, returns an error update without performing
        LLM calls.

    Notes:
        This node performs Neo4j I/O (trait reuse hints) and LLM I/O (sheet generation).
        JSON parsing failures in character sheet responses are treated as fatal for
        that character and surface as initialization errors.
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

    prompt_sha1 = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    logger.debug(
        "_generate_character_list: prompt prepared",
        prompt_sha1=prompt_sha1,
        prompt_len=len(prompt),
        has_protagonist=bool(state.get("protagonist_name", "")),
        title=state.get("title", ""),
        genre=state.get("genre", ""),
    )

    temperatures = [0.7, 0.3, 0.1]
    last_exception = None

    for attempt_index, temperature in enumerate(temperatures, start=1):
        try:
            response, _ = await llm_service.async_call_llm(
                model_name=state.get("large_model", ""),
                prompt=prompt,
                temperature=temperature,
                max_tokens=16384,
                allow_fallback=False,
                auto_clean_response=True,
                system_prompt=get_system_prompt("initialization"),
            )

            if not response:
                raise ValueError("LLM returned empty response for character list")

            response_sha1 = hashlib.sha1(response.encode("utf-8")).hexdigest()[:12]
            logger.debug(
                "_generate_character_list: llm response received",
                prompt_sha1=prompt_sha1,
                attempt=attempt_index,
                response_sha1=response_sha1,
                response_len=len(response),
            )

            data = json.loads(response)

            if not isinstance(data, list):
                raise ValueError("Character list must be a JSON array")

            protagonist_name = state.get("protagonist_name", "")
            minimum_count = 3 if protagonist_name else 1

            if len(data) < minimum_count:
                raise ValueError("Character list is too short")

            if len(data) > 10:
                raise ValueError("Character list is too long")

            validated_names: list[str] = []
            seen: set[str] = set()

            for element in data:
                if not isinstance(element, str):
                    raise ValueError("Character list elements must be strings")

                if element != element.strip():
                    raise ValueError("Character names must not have leading/trailing whitespace")

                if not element:
                    raise ValueError("Character names must be non-empty")

                if len(element) > 80:
                    raise ValueError("Character names must be a reasonable length")

                if "\n" in element or "\r" in element:
                    raise ValueError("Character names must be a single line")

                if "(" in element or ")" in element:
                    raise ValueError("Character names must not include parenthetical text")

                if element in seen:
                    logger.error(
                        "_generate_character_list: duplicate character name detected",
                        prompt_sha1=prompt_sha1,
                        attempt=attempt_index,
                        duplicate_name=element,
                        validated_count=len(validated_names),
                    )
                    raise ValueError("Character names must be unique")

                validated_names.append(element)
                seen.add(element)

            if protagonist_name and protagonist_name not in validated_names:
                raise ValueError("Character list must include the protagonist name exactly")

            placeholder_names = [name for name in validated_names if re.fullmatch(r"Name (One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|\d+)", name)]
            if placeholder_names:
                logger.error(
                    "_generate_character_list: placeholder names detected",
                    prompt_sha1=prompt_sha1,
                    placeholders=placeholder_names,
                    names=validated_names,
                )

            logger.info(
                "_generate_character_list: generated list",
                count=len(validated_names),
                names=validated_names,
            )

            return validated_names

        except Exception as exception:
            last_exception = exception
            logger.error(
                "_generate_character_list: attempt failed",
                prompt_sha1=prompt_sha1,
                attempt=attempt_index,
                error=str(exception),
                exc_info=True,
            )

    logger.error(
        "_generate_character_list: all attempts failed",
        prompt_sha1=prompt_sha1,
        error=str(last_exception) if last_exception else None,
    )
    return []


async def _generate_character_sheet(
    state: NarrativeState,
    character_name: str,
    other_characters: list[str],
    existing_traits: list[str] | None = None,
) -> dict[str, Any] | None:
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

    temperatures = [0.7, 0.3, 0.1]
    last_exception = None

    for attempt_index, temperature in enumerate(temperatures, start=1):
        try:
            response, usage = await llm_service.async_call_llm(
                model_name=state.get("large_model", ""),
                prompt=prompt,
                temperature=temperature,
                max_tokens=16384,
                allow_fallback=False,
                auto_clean_response=True,
                system_prompt=get_system_prompt("initialization"),
            )

            if not response:
                raise ValueError("LLM returned empty response for character sheet")

            response_sha1 = hashlib.sha1(response.encode("utf-8")).hexdigest()[:12]
            logger.debug(
                "_generate_character_sheet: llm response received",
                character=character_name,
                attempt=attempt_index,
                response_sha1=response_sha1,
                response_len=len(response),
                usage=usage,
            )

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

        except Exception as exception:
            last_exception = exception
            logger.error(
                "_generate_character_sheet: attempt failed",
                character=character_name,
                attempt=attempt_index,
                error=str(exception),
                exc_info=True,
            )

    logger.error(
        "_generate_character_sheet: all attempts failed",
        character=character_name,
        error=str(last_exception) if last_exception else None,
    )
    return None


__all__ = ["generate_character_sheets"]
