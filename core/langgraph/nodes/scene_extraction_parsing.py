# core/langgraph/nodes/scene_extraction_parsing.py
"""LLM output parsing helpers for scene extraction.

This module is separated to keep the per-type extraction functions in scene_extraction.py
focused on the extraction flow while the parsing logic lives here.
"""

from __future__ import annotations

from typing import Any


def _require_named_updates(value: Any, field: str) -> list[tuple[str, dict[str, Any]]]:
    """Reject malformed entries rather than silently accepting a partial mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    for name, information in value.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(information, dict):
            raise ValueError(f"{field} entries must map nonblank names to objects")
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
    kg_triples_list = data.get("kg_triples")
    if not isinstance(kg_triples_list, list):
        raise ValueError("kg_triples must be an array")
    for triple in kg_triples_list:
        if not isinstance(triple, dict):
            raise ValueError("kg_triples entries must be objects")
        for key in ("subject", "predicate", "object_entity"):
            if not isinstance(triple.get(key), str) or not triple[key].strip():
                raise ValueError(f"kg_triples.{key} must be a nonblank string")
        if "description" in triple and not isinstance(triple["description"], str):
            raise ValueError("kg_triples.description must be a string")
    return kg_triples_list


def normalize_triple_entities(triple: dict[str, Any]) -> tuple[str, str, str, str]:
    """Normalize subject/predicate/object from a kg_triple dict.

    Handles cases where subject or object_entity are nested dicts with a "name" field.

    Args:
        triple: A kg_triple dict with subject, predicate, object_entity keys.

    Returns:
        Tuple of (subject_text, target_text, predicate_text, description).
    """
    subject = triple.get("subject", "")
    predicate = triple.get("predicate", "RELATES_TO")
    object_entity = triple.get("object_entity", "")
    description = triple.get("description", "")

    if isinstance(subject, dict):
        subject = subject.get("name", str(subject))
    if isinstance(object_entity, dict):
        object_entity = object_entity.get("name", str(object_entity))

    subject_text = str(subject) if subject else ""
    target_text = str(object_entity) if object_entity else ""
    predicate_text = str(predicate) if predicate else ""

    return subject_text, target_text, predicate_text, str(description)


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
