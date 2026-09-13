# core/langgraph/initialization/commit_init_node.py
"""Commit initialization artifacts to Neo4j.

This module defines the initialization persistence boundary. It converts
initialization artifacts (character sheets and global outline) into Neo4j models
and writes them in a single batched transaction, then clears relevant read caches.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

import config
from core.langgraph.content_manager import (
    require_project_dir,
)
from core.langgraph.state import NarrativeState
from core.service_context import get_services
from data_access.cypher_builders.native_builders import NativeCypherBuilder
from models.kg_models import CharacterProfile, WorldItem
from prompts.prompt_renderer import get_system_prompt, render_prompt
from utils.common import ensure_exact_keys, try_load_json_from_response
from utils.text_processing import validate_and_filter_traits

logger = structlog.get_logger(__name__)


async def commit_initialization_to_graph(state: NarrativeState) -> NarrativeState:
    """Prepare the complete immutable import; graph acceptance belongs to run_parsers."""
    from core.langgraph.initialization.staged_import import InitializationImport

    try:
        plan = await InitializationImport(require_project_dir(state)).prepare(state)
        return {
            "current_node": "commit_initialization", "last_error": None,
            "initialization_step": "initialization_prepared", "initialization_id": plan.identity,
        }
    except Exception as error:
        return {
            "current_node": "commit_initialization", "last_error": f"Initialization admission failed: {error}",
            "initialization_step": "commit_failed", "has_fatal_error": True, "error_node": "commit_initialization",
        }


async def _parse_character_sheets_to_profiles(
    character_sheets: dict[str, dict],
    model_name: str | None = None,
) -> list[CharacterProfile]:
    """Convert character sheet dictionaries into `CharacterProfile` models.

    Args:
        character_sheets: Mapping of character name to sheet data (ideally pre-parsed).
        model_name: Optional model name used when falling back to LLM extraction.

    Returns:
        Character profiles ready for persistence.

    Notes:
        When a sheet lacks structured traits, this helper falls back to
        [`_extract_structured_character_data()`](core/langgraph/initialization/commit_init_node.py:239).
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
            personality_description=description,
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
    """Extract structured character fields from a prose description.

    This function enforces a strict JSON contract. Contract violations are treated
    as fatal to initialization because the results are persisted as canonical facts.

    Args:
        name: Character name.
        description: Free-form character description.
        model_name: LLM model name override.

    Returns:
        Structured character fields.

    Raises:
        ValueError: When the response violates the JSON/schema contract.

    Notes:
        This function performs LLM I/O.
    """
    prompt = render_prompt(
        "knowledge_agent/extract_character_structured_lines.j2",
        {
            "name": name,
            "description": description,
        },
    )

    model = model_name or config.NARRATIVE_MODEL

    for attempt in range(1, config.JSON_PARSE_RETRY_ATTEMPTS + 1):
        response, _ = await get_services().language_model.async_call_llm(
            model_name=model,
            prompt=prompt,
            temperature=0.3,
            max_tokens=config.MAX_GENERATION_TOKENS,
            allow_fallback=True,
            auto_clean_response=True,
            system_prompt=get_system_prompt("knowledge_agent"),
        )

        try:
            return _parse_character_extraction_response(response)
        except json.JSONDecodeError:
            if attempt == 2:
                raise

    raise RuntimeError("Structured character extraction exceeded max attempts")


def _log_json_decode_error(
    *,
    context: str,
    raw_text: str,
    error: json.JSONDecodeError,
) -> None:
    start = max(error.pos - 160, 0)
    end = min(error.pos + 160, len(raw_text))
    snippet = raw_text[start:end]

    logger.error(
        "initialization JSON decode failed",
        context=context,
        error=str(error),
        line=error.lineno,
        column=error.colno,
        position=error.pos,
        raw_text_length=len(raw_text),
        contains_code_fence="```" in raw_text,
        snippet_range={"start": start, "end": end},
        snippet=repr(snippet),
    )


def _load_json_with_contract_then_salvage(
    *,
    context: str,
    raw_text: str,
    expected_root: type,
) -> Any:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as error:
        _log_json_decode_error(
            context=context,
            raw_text=raw_text,
            error=error,
        )

        salvaged, _candidates, _parse_errors = try_load_json_from_response(
            raw_text,
            expected_root=expected_root,
        )
        if salvaged is None:
            raise

        logger.warning(
            "initialization JSON contract violated; salvaged embedded JSON",
            context=context,
            raw_text_length=len(raw_text),
            contains_code_fence="```" in raw_text,
        )
        return salvaged


def _parse_character_extraction_response(response: str) -> dict[str, Any]:
    """Parse a structured character extraction response.

    Args:
        response: LLM response expected to contain a JSON object.

    Returns:
        Parsed structured character fields.

    Raises:
        ValueError: When required keys are missing or values violate invariants
            (for example, traits not being single-word values).
    """
    raw_text = response.strip()

    data = _load_json_with_contract_then_salvage(
        context="structured_character_extraction",
        raw_text=raw_text,
        expected_root=dict,
    )

    if not isinstance(data, dict):
        raise ValueError("Structured character extraction must be a JSON object")

    required_keys = {"traits", "status", "motivations", "background"}
    ensure_exact_keys(
        value=data,
        required_keys=required_keys,
        context="Structured character extraction",
    )

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
    """Extract world items from the global outline for persistence.

    Args:
        global_outline: Parsed global outline data.
        setting: Story setting description used for prompt context.
        model_name: LLM model name override.

    Returns:
        World items to persist as initialization facts.

    Notes:
        This function performs LLM I/O when outline text is present.
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

    for attempt in range(1, config.JSON_PARSE_RETRY_ATTEMPTS + 1):
        response, _ = await get_services().language_model.async_call_llm(
            model_name=model,
            prompt=prompt,
            temperature=0.5,
            max_tokens=config.MAX_GENERATION_TOKENS,
            allow_fallback=True,
            auto_clean_response=True,
            system_prompt=get_system_prompt("knowledge_agent"),
        )

        try:
            return _parse_world_items_extraction(response)
        except json.JSONDecodeError:
            if attempt == 2:
                raise

    raise RuntimeError("World item extraction exceeded max attempts")


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

    from core.langgraph.initialization.snapshot import strict_json

    data = strict_json(raw_text)

    if not isinstance(data, list):
        raise ValueError("World items extraction must be a JSON array")

    items: list[WorldItem] = []

    allowed_categories = {"location", "object"}

    from utils.text_processing import generate_entity_id

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"World item at index {index} must be a JSON object")

        required_keys = {"name", "category", "description"}
        ensure_exact_keys(
            value=item,
            required_keys=required_keys,
            context=f"World item at index {index}",
        )

        name = item["name"]
        category = item["category"]
        description = item["description"]

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"World item at index {index} 'name' must be a non-empty string")
        if name != name.strip():
            raise ValueError(f"World item at index {index} 'name' must not have surrounding whitespace")
        if not isinstance(category, str) or category not in allowed_categories:
            raise ValueError(f"World item at index {index} 'category' must be one of {sorted(allowed_categories)}")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"World item at index {index} 'description' must be a non-empty string")

        items.append(
            WorldItem(
                id=generate_entity_id(name, category),
                name=name,
                description=description,
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


async def _build_outline_relationship_statements(
    relationships: list[dict[str, Any]],
    chapter_number: int,
) -> list[tuple[str, dict]]:
    """Build Cypher statements to persist outline relationships.

    Args:
        relationships: List of relationship dicts with keys: source_name, target_name, relationship_type, description.
        chapter_number: Current chapter for tracking (0 for initialization).

    Returns:
        List of (cypher_query, parameters) tuples.
    """
    import hashlib

    apoc_available = await get_services().database.is_apoc_available()
    assert apoc_available, "APOC procedures required for outline relationship persistence"

    statements: list[tuple[str, dict]] = []

    for rel in relationships:
        source_name = rel.get("source_name", "")
        target_name = rel.get("target_name", "")
        relationship_type = rel.get("relationship_type", "")
        description = rel.get("description", "")
        confidence = rel.get("confidence", 0.8)

        if not source_name or not target_name or not relationship_type:
            logger.warning(
                "_build_outline_relationship_statements: skipping incomplete relationship",
                source=source_name,
                target=target_name,
                type=relationship_type,
            )
            continue

        rel_id_source = f"{relationship_type}|{source_name.strip().lower()}|{target_name.strip().lower()}"
        rel_id = hashlib.sha1(rel_id_source.encode("utf-8")).hexdigest()[:16]

        cypher = """
        MATCH (s {name: $source_name})
        WHERE s:Character OR s:Location OR s:Event OR s:Item
        MATCH (t {name: $target_name})
        WHERE t:Character OR t:Location OR t:Event OR t:Item
        WITH s, t
        CALL apoc.merge.relationship(
            s,
            $relationship_type,
            {id: $rel_id},
            {
                chapter_added: $chapter,
                is_provisional: false,
                confidence: $confidence,
                description: $description,
                created_ts: timestamp(),
                updated_ts: timestamp()
            },
            t,
            {}
        ) YIELD rel
        RETURN rel
        """

        params = {
            "source_name": source_name,
            "target_name": target_name,
            "relationship_type": relationship_type,
            "rel_id": rel_id,
            "chapter": chapter_number,
            "confidence": confidence,
            "description": description,
        }

        statements.append((cypher, params))

    logger.info(
        "_build_outline_relationship_statements: built relationship statements",
        count=len(statements),
    )

    return statements


async def _build_entity_persistence_statements(
    characters: list[CharacterProfile],
    world_items: list[WorldItem],
    outline_relationships: list[dict[str, Any]],
    chapter_number: int,
) -> list[tuple[str, dict]]:
    """
    Build Cypher statements for entity persistence without executing them.

    This uses the same approach as commit_node.py, building statements
    that will be executed in a single batch transaction for atomicity.

    Args:
        characters: List of CharacterProfile models
        world_items: List of WorldItem models
        outline_relationships: List of relationship dicts from outline extraction
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

    if outline_relationships:
        relationship_statements = await _build_outline_relationship_statements(
            outline_relationships,
            chapter_number,
        )
        statements.extend(relationship_statements)

    embedding_statements_count = 0
    if config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE:
        from core.entity_embedding_service import build_entity_embedding_update_statements

        embedding_statements = await build_entity_embedding_update_statements(
            characters=characters,
            world_items=world_items,
        )
        embedding_statements_count = len(embedding_statements)
        statements.extend(embedding_statements)

    logger.info(
        "_build_entity_persistence_statements: built statements",
        characters=len(characters),
        world_items=len(world_items),
        outline_relationships=len(outline_relationships),
        embedding_statements=embedding_statements_count,
        total_statements=len(statements),
    )

    return statements


__all__ = ["commit_initialization_to_graph"]
