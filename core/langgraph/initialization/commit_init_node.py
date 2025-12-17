# core/langgraph/initialization/commit_init_node.py
"""
Commit Initialization Data to Knowledge Graph Node.

This node bridges the initialization phase to the generation loop by converting
initialization data (character sheets, outlines) into Neo4j-compatible models
and committing them to the knowledge graph.
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

import config
from core.db_manager import neo4j_manager
from core.langgraph.content_manager import (
    ContentManager,
    get_character_sheets,
    get_global_outline,
)
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from data_access.cypher_builders.native_builders import NativeCypherBuilder
from models.kg_models import CharacterProfile, WorldItem
from prompts.grammar_loader import load_grammar
from prompts.prompt_renderer import get_system_prompt, render_prompt
from utils.text_processing import validate_and_filter_traits

logger = structlog.get_logger(__name__)


async def commit_initialization_to_graph(state: NarrativeState) -> NarrativeState:
    """
    Convert initialization data to Neo4j models and commit to knowledge graph.

    This node acts as a bridge between the initialization phase (which generates
    rich text descriptions) and the generation loop (which needs structured
    CharacterProfile and WorldItem models in Neo4j).

    Process Flow:
    1. Parse character sheets to extract structured traits
    2. Convert to CharacterProfile models
    3. Extract world elements from outlines
    4. Convert to WorldItem models
    5. Commit all to Neo4j using existing persistence layer

    Args:
        state: Current narrative state with character_sheets and outlines

    Returns:
        Updated state with initialization data committed to graph
    """
    # Initialize content manager for reading externalized content
    content_manager = ContentManager(state.get("project_dir", ""))

    # Get character sheets and global outline (from external files)
    character_sheets = get_character_sheets(state, content_manager)
    global_outline = get_global_outline(state, content_manager)

    logger.info(
        "commit_initialization_to_graph: starting initialization data commit",
        characters=len(character_sheets),
        has_global_outline=bool(global_outline),
    )

    if not character_sheets:
        logger.warning("commit_initialization_to_graph: no character sheets to commit")

    try:
        # Step 1: Parse character sheets into CharacterProfile models
        character_profiles = []
        if character_sheets:
            character_profiles = await _parse_character_sheets_to_profiles(
                character_sheets,
                model_name=state.get("medium_model", ""),
            )

        # Step 2: Extract world items from outlines
        world_items = []
        if global_outline:
            world_items = await _extract_world_items_from_outline(
                global_outline,
                state.get("setting", ""),
                model_name=state.get("medium_model", ""),
            )

        # Step 3: Commit to Neo4j using direct batch approach
        if character_profiles or world_items:
            statements = _build_entity_persistence_statements(
                character_profiles,
                world_items,
                chapter_number=0,  # Initialization entities exist before any chapters
            )

            if statements:
                await neo4j_manager.execute_cypher_batch(statements)

                # P0-1: Cache invalidation after Neo4j writes
                # Local import avoids eager import side effects / circular deps.
                from data_access.cache_coordinator import (
                    clear_character_read_caches,
                    clear_world_read_caches,
                )

                cleared_character = clear_character_read_caches()
                cleared_world = clear_world_read_caches()

                logger.debug(
                    "commit_initialization_to_graph: executed batch and invalidated caches",
                    total_statements=len(statements),
                    cache_cleared={
                        "character": cleared_character,
                        "world": cleared_world,
                    },
                )

        logger.info(
            "commit_initialization_to_graph: successfully committed initialization data",
            characters=len(character_profiles),
            world_items=len(world_items),
        )

        # Step 4: Update active_characters with committed profiles
        # This makes characters immediately available to the generation loop
        updated_state = {
            **state,
            "active_characters": character_profiles[:5],  # Top 5 for initial context
            "world_items": world_items,
            "current_node": "commit_initialization",
            "last_error": None,
            "initialization_step": "committed_to_graph",
        }

        return updated_state

    except Exception as e:
        error_msg = f"Failed to commit initialization data: {e}"
        logger.error(
            "commit_initialization_to_graph: fatal error during commit",
            error=str(e),
            exc_info=True,
        )
        return {
            **state,
            "current_node": "commit_initialization",
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "commit_initialization",
            "initialization_step": "commit_failed",
        }


async def _parse_character_sheets_to_profiles(
    character_sheets: dict[str, dict],
    model_name: str | None = None,
) -> list[CharacterProfile]:
    """
    Convert pre-parsed character sheets to CharacterProfile models.

    The character sheets are now pre-parsed during generation, containing
    structured data (traits, motivations, relationships, etc.) that can be
    directly used to create CharacterProfile models without additional LLM calls.

    Args:
        character_sheets: Dict of character_name -> character_sheet (pre-parsed)

    Returns:
        List of CharacterProfile models ready for Neo4j persistence
    """
    profiles = []

    for name, sheet in character_sheets.items():
        # Check if we have pre-parsed structured data
        raw_traits = sheet.get("traits", [])
        traits = validate_and_filter_traits(raw_traits)

        if len(traits) != len(raw_traits):
            logger.warning(
                "_parse_character_sheets_to_profiles: filtered invalid traits",
                character=name,
                original_count=len(raw_traits),
                filtered_count=len(traits),
            )

        status = sheet.get("status", "Active")
        motivations = sheet.get("motivations", "")
        background = sheet.get("background", "")
        skills = sheet.get("skills", [])
        relationships = sheet.get("relationships", {})
        is_protagonist = sheet.get("is_protagonist", False)
        description = sheet.get("description", "")
        internal_conflict = sheet.get("internal_conflict", "")

        # If no pre-parsed traits, fall back to LLM extraction (backward compatibility)
        if not traits and description:
            logger.info(
                "_parse_character_sheets_to_profiles: no pre-parsed traits, using LLM extraction",
                character=name,
            )
            structured_data = await _extract_structured_character_data(name, description)
            raw_extracted_traits = structured_data.get("traits", [])
            traits = validate_and_filter_traits(raw_extracted_traits)
            status = structured_data.get("status", status)
            motivations = structured_data.get("motivations", motivations)
            background = structured_data.get("background", background)

        # Create CharacterProfile model
        profile = CharacterProfile(
            name=name,
            description=description,
            traits=traits,
            status=status,
            relationships=relationships,  # Now populated from pre-parsed data
            created_chapter=0,  # Initialization entities created before chapters
            is_provisional=False,  # Initialization characters are canonical
            updates={
                "is_protagonist": is_protagonist,
                "motivations": motivations,
                "background": background,
                "skills": skills,
                "internal_conflict": internal_conflict,
            },
        )

        profiles.append(profile)
        logger.debug(
            "_parse_character_sheets_to_profiles: created profile",
            name=name,
            traits=len(profile.traits),
            has_relationships=bool(relationships),
        )

    return profiles


async def _extract_structured_character_data(name: str, description: str, model_name: str | None = None) -> dict[str, Any]:
    """
    Use LLM to extract structured data from character sheet description.

    This is a grammar-enforced, JSON-only contract.
    Contract violations are fatal to initialization.

    Args:
        name: Character name
        description: Free-form character description
        model_name: Name of the LLM to use

    Returns:
        Dictionary with extracted structured data (traits, status, motivations, etc.)

    Raises:
        ValueError: If the LLM output violates the JSON/schema contract.
    """
    prompt = render_prompt(
        "knowledge_agent/extract_character_structured_lines.j2",
        {
            "name": name,
            "description": description,
        },
    )

    model = model_name or config.NARRATIVE_MODEL

    grammar_content = load_grammar("initialization")
    grammar = re.sub(r"^root ::= .*$", "", grammar_content, flags=re.MULTILINE)
    grammar = f"root ::= structured-character-extraction\n{grammar}"

    response, _ = await llm_service.async_call_llm(
        model_name=model,
        prompt=prompt,
        temperature=0.3,  # Low temp for extraction
        max_tokens=1024,
        allow_fallback=True,
        auto_clean_response=True,
        system_prompt=get_system_prompt("knowledge_agent"),
        grammar=grammar,
    )

    return _parse_character_extraction_response(response)


def _parse_character_extraction_response(response: str) -> dict[str, Any]:
    """
    Parse the LLM's structured character extraction response (strict JSON contract).

    Args:
        response: LLM response (JSON object)

    Returns:
        Dictionary with parsed structured data

    Raises:
        ValueError: If the output violates the JSON/schema contract.
    """
    raw_text = response.strip()
    data = json.loads(raw_text)

    if not isinstance(data, dict):
        raise ValueError("Structured character extraction must be a JSON object")

    required_keys = {"traits", "status", "motivations", "background"}
    data_keys = set(data.keys())
    if data_keys != required_keys:
        raise ValueError("Structured character extraction must contain exactly keys " f"{sorted(required_keys)} (got {sorted(data_keys)})")

    traits = data["traits"]
    if not isinstance(traits, list) or any(not isinstance(t, str) for t in traits):
        raise ValueError("Structured character extraction 'traits' must be a JSON array of strings")

    validated_traits = validate_and_filter_traits(traits)
    if validated_traits != traits:
        raise ValueError("Structured character extraction 'traits' must be single-word traits only")

    if not (3 <= len(traits) <= 7):
        raise ValueError("Structured character extraction 'traits' must contain 3-7 items")

    status = data["status"]
    motivations = data["motivations"]
    background = data["background"]

    if not isinstance(status, str) or not status.strip():
        raise ValueError("Structured character extraction 'status' must be a non-empty string")
    if not isinstance(motivations, str):
        raise ValueError("Structured character extraction 'motivations' must be a string")
    if not isinstance(background, str):
        raise ValueError("Structured character extraction 'background' must be a string")

    return {
        "traits": traits,
        "status": status.strip(),
        "motivations": motivations.strip(),
        "background": background.strip(),
    }


async def _extract_world_items_from_outline(global_outline: dict, setting: str, model_name: str | None = None) -> list[WorldItem]:
    """
    Extract world-building elements from the global outline.

    Uses LLM to identify key locations, objects, and world elements mentioned
    in the outline that should be added to the knowledge graph.

    Args:
        global_outline: Global outline with story structure
        setting: Story setting description
        model_name: Name of the LLM to use

    Returns:
        List of WorldItem models ready for Neo4j persistence
    """
    outline_text = global_outline.get("raw_text", "")
    if not outline_text:
        return []

    prompt = render_prompt(
        "knowledge_agent/extract_world_items_lines.j2",
        {
            "setting": setting,
            "outline_text": outline_text,
        },
    )

    model = model_name or config.NARRATIVE_MODEL

    grammar_content = load_grammar("initialization")
    grammar = re.sub(r"^root ::= .*$", "", grammar_content, flags=re.MULTILINE)
    grammar = f"root ::= world-items-extraction\n{grammar}"

    response, _ = await llm_service.async_call_llm(
        model_name=model,
        prompt=prompt,
        temperature=0.5,
        max_tokens=2000,
        allow_fallback=True,
        auto_clean_response=True,
        system_prompt=get_system_prompt("knowledge_agent"),
        grammar=grammar,
    )

    return _parse_world_items_extraction(response)


def _parse_world_items_extraction(response: str) -> list[WorldItem]:
    """
    Parse LLM response into WorldItem models (strict JSON contract).

    Args:
        response: LLM response (JSON array)

    Returns:
        List of WorldItem models

    Raises:
        ValueError: If the output violates the JSON/schema contract.
    """
    raw_text = response.strip()
    data = json.loads(raw_text)

    if not isinstance(data, list):
        raise ValueError("World items extraction must be a JSON array")

    items: list[WorldItem] = []

    allowed_categories = {"location", "object", "concept"}

    from processing.entity_deduplication import generate_entity_id

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"World item at index {index} must be a JSON object")

        required_keys = {"name", "category", "description"}
        item_keys = set(item.keys())
        if item_keys != required_keys:
            raise ValueError(f"World item at index {index} must contain exactly keys {sorted(required_keys)} " f"(got {sorted(item_keys)})")

        name = item["name"]
        category = item["category"]
        description = item["description"]

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"World item at index {index} 'name' must be a non-empty string")
        if not isinstance(category, str) or category not in allowed_categories:
            raise ValueError(f"World item at index {index} 'category' must be one of {sorted(allowed_categories)}")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"World item at index {index} 'description' must be a non-empty string")

        items.append(
            WorldItem(
                id=generate_entity_id(name.strip(), category, chapter=0),
                name=name.strip(),
                description=description.strip(),
                category=category,
                created_chapter=0,
                is_provisional=False,
            )
        )

    logger.info(
        "_parse_world_items_extraction: extracted world items",
        count=len(items),
    )

    return items


def _build_entity_persistence_statements(
    characters: list[CharacterProfile],
    world_items: list[WorldItem],
    chapter_number: int,
) -> list[tuple[str, dict]]:
    """
    Build Cypher statements for entity persistence without executing them.

    This uses the same approach as commit_node.py, building statements
    that will be executed in a single batch transaction for atomicity.

    Args:
        characters: List of CharacterProfile models
        world_items: List of WorldItem models
        chapter_number: Current chapter for tracking (0 for initialization)

    Returns:
        List of (cypher_query, parameters) tuples
    """
    statements: list[tuple[str, dict]] = []

    cypher_builder = NativeCypherBuilder()

    for char in characters:
        cypher, params = cypher_builder.character_upsert_cypher(char, chapter_number)
        statements.append((cypher, params))

    for item in world_items:
        cypher, params = cypher_builder.world_item_upsert_cypher(item, chapter_number)
        statements.append((cypher, params))

    logger.info(
        "_build_entity_persistence_statements: built statements",
        characters=len(characters),
        world_items=len(world_items),
        total_statements=len(statements),
    )

    return statements


__all__ = ["commit_initialization_to_graph"]
