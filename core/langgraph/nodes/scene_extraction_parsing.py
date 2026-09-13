# core/langgraph/nodes/scene_extraction_parsing.py
"""LLM output parsing helpers for scene extraction.

This module is separated to keep the per-type extraction functions in scene_extraction.py
focused on the extraction flow while the parsing logic lives here.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

import config
from core.langgraph.nodes.scene_extraction_validation import validate_lexical_entity_name
from models.kg_constants import RELATIONSHIP_TYPES


class SceneRelationship(BaseModel):
    """Name-based relationship evidence, not outline catalog identifiers."""

    model_config = ConfigDict(extra="forbid", strict=True)
    subject: str = Field(pattern=r"\S")
    predicate: str = Field(json_schema_extra={"enum": [name for name in sorted(RELATIONSHIP_TYPES)]})
    object_entity: str = Field(pattern=r"\S")
    description: str

    @field_validator("subject", "object_entity")
    @classmethod
    def named_endpoint(cls, value: str) -> str:
        return validate_lexical_entity_name(value)

    @field_validator("predicate")
    @classmethod
    def canonical_predicate(cls, value: str) -> str:
        if value not in RELATIONSHIP_TYPES:
            raise ValueError("Scene relationship predicate must be canonical")
        return value


class SceneRelationships(BaseModel):
    """Complete scene relationship response; invalid rows are never dropped."""

    model_config = ConfigDict(extra="forbid", strict=True)
    kg_triples: list[SceneRelationship] = Field(max_length=15)

    @classmethod
    def response_format(cls, eligible_names: Collection[str] = ()) -> dict[str, Any]:
        schema = cls.model_json_schema()
        properties = schema["$defs"]["SceneRelationship"]["properties"]
        for endpoint in ("subject", "object_entity"):
            properties[endpoint]["enum"] = sorted(eligible_names)
        if not eligible_names:
            schema["properties"]["kg_triples"]["maxItems"] = 0
            for endpoint in ("subject", "object_entity"):
                del properties[endpoint]["enum"]
        return {"type": "json_schema", "json_schema": {
            "name": "extract_scene_relationships", "strict": config.STRUCTURED_OUTPUT_STRICT, "schema": schema,
        }}


def _require_named_updates(value: Any, field: str) -> list[tuple[str, dict[str, Any]]]:
    """Reject malformed entries rather than silently accepting a partial mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    for name, information in value.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(information, dict):
            raise ValueError(f"{field} entries must map nonblank names to objects")
        validate_lexical_entity_name(name)
        for key in ("description", "status", "category"):
            if key in information and not isinstance(information[key], str):
                raise ValueError(f"{field}.{name}.{key} must be a string")
        for key in ("traits", "goals", "rules", "key_elements"):
            if key in information and (
                not isinstance(information[key], list)
                or any(not isinstance(item, str) for item in information[key])
            ):
                raise ValueError(f"{field}.{name}.{key} must be an array of strings")
        if "relationships" in information and not isinstance(information["relationships"], dict):
            raise ValueError(f"{field}.{name}.relationships must be an object")
        for target_name, relationship in information.get("relationships", {}).items():
            validate_lexical_entity_name(target_name)
            if not isinstance(relationship, dict) or relationship.get("type") not in RELATIONSHIP_TYPES:
                raise ValueError(f"{field}.{name}.relationships must use canonical relationship types")
    return list(value.items())


def parse_character_updates(
    data: dict[str, Any],
    scene_index: int,
    chapter_number: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Parse character_updates from an LLM response dict.

    Args:
        data: The parsed JSON response from the LLM.
        scene_index: For logging context.
        chapter_number: For logging context.

    Returns:
        List of (name, info) tuples from a complete character_updates mapping.
    """
    return _require_named_updates(data.get("character_updates"), "character_updates")


def parse_world_updates(
    data: dict[str, Any],
    world_type: str,
    scene_index: int,
    chapter_number: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Parse a world entity type (Location/Event) from an LLM response dict.

    Args:
        data: The parsed JSON response from the LLM.
        world_type: "Location" or "Event".
        scene_index: For logging context.
        chapter_number: For logging context.

    Returns:
        List of (name, info) tuples from the world_updates[world_type] dict.
    """
    world_updates = data.get("world_updates")
    if not isinstance(world_updates, dict):
        raise ValueError("world_updates must be an object")
    return _require_named_updates(world_updates.get(world_type), f"world_updates.{world_type}")


def parse_kg_triples(
    data: dict[str, Any],
    scene_index: int,
    chapter_number: int,
) -> list[dict[str, Any]]:
    """Parse kg_triples from an LLM response dict.

    Args:
        data: The parsed JSON response from the LLM.
        scene_index: For logging context.
        chapter_number: For logging context.

    Returns:
        List of triple dicts from a complete kg_triples array.
    """
    return [row.model_dump() for row in SceneRelationships.model_validate(data).kg_triples]


def normalize_triple_entities(triple: dict[str, Any]) -> tuple[str, str, str, str]:
    """Convert the strict producer contract without defaults or identity repair."""
    row = SceneRelationship.model_validate(triple)
    return row.subject, row.object_entity, row.predicate, row.description


def normalize_dict_items(
    items: Any,
    *,
    item_kind: str,
) -> list[dict[str, Any]]:
    """Normalize a list of items (which may contain Pydantic models or dicts) to plain dicts.

    Args:
        items: The items to normalize.
        item_kind: Human-readable name for the item type, used in error messages.

    Returns:
        List of plain dicts.

    Raises:
        TypeError: If an item is neither a Pydantic BaseModel nor a dict.
    """
    from pydantic import BaseModel

    if items is None:
        return []
    if not isinstance(items, list):
        raise TypeError(f"normalize_dict_items: expected {item_kind} to be a list; got {type(items)}")

    normalized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, BaseModel):
            normalized_value = item.model_dump(mode="json")
            if not isinstance(normalized_value, dict):
                raise TypeError(
                    f"normalize_dict_items: expected {item_kind} model_dump to produce dict; got {type(normalized_value)}"
                )
            normalized.append(normalized_value)
            continue

        if isinstance(item, dict):
            normalized.append(item)
            continue

        raise TypeError(
            f"normalize_dict_items: expected {item_kind} item to be dict-like; got {type(item)}"
        )

    return normalized
