# core/langgraph/nodes/scene_extraction_normalization.py
"""Entity deduplication and consolidation for scene extraction.

This module handles merging results from multiple scenes using exact names.
"""

from __future__ import annotations

from typing import Any

import structlog

from core.langgraph.nodes.scene_extraction_validation import (
    _get_normalized_entity_key,
)
from utils import classify_category_label

logger = structlog.get_logger(__name__)


def _merge_entity_identity(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Select descriptive prose without losing or conflicting with explicit identity."""
    identifiers: set[str] = set()
    for record in (existing, incoming):
        attributes = record.get("attributes", {})
        if "id" in attributes:
            identifier = attributes["id"]
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError("Invalid explicit entity ID during consolidation")
            identifiers.add(identifier)
    if len(identifiers) > 1 or existing.get("type") != incoming.get("type"):
        raise ValueError("Conflicting entity identity during consolidation")
    selected = incoming if len(incoming.get("description", "")) > len(existing.get("description", "")) else existing
    if not identifiers:
        return selected
    return {**selected, "attributes": {**selected.get("attributes", {}), "id": next(iter(identifiers))}}


def consolidate_scene_extractions(
    scene_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge and deduplicate extraction results from multiple scenes.

    Deduplication strategy:
    - Characters: Dedupe by exact name, retain explicit identity and longest description
    - World items: Dedupe by canonical category and exact name, retain explicit identity
    - Relationships: Dedupe by exact (source, target, type) tuple

    Args:
        scene_results: List of extraction results from individual scenes.

    Returns:
        Consolidated dict with deduplicated characters, world_items, relationships.
    """
    characters_map: dict[str, dict[str, Any]] = {}
    world_items_map: dict[tuple[str, str], dict[str, Any]] = {}
    relationships_set: set[tuple[str, str, str]] = set()
    relationships: list[dict[str, Any]] = []

    for scene_result in scene_results:
        for character in scene_result.get("characters", []):
            name = character["name"]
            name_key = _get_normalized_entity_key(name)

            if name_key in characters_map:
                characters_map[name_key] = _merge_entity_identity(characters_map[name_key], character)
            else:
                characters_map[name_key] = character

        for world_item in scene_result.get("world_items", []):
            name = world_item["name"]
            name_key = _get_normalized_entity_key(name)
            category = world_item.get("attributes", {}).get("category", world_item.get("type", ""))
            world_item_key = (classify_category_label(category), name_key)

            if world_item_key in world_items_map:
                world_items_map[world_item_key] = _merge_entity_identity(world_items_map[world_item_key], world_item)
            else:
                world_items_map[world_item_key] = world_item

        for relationship in scene_result.get("relationships", []):
            source = relationship.get("source_name", "")
            target = relationship.get("target_name", "")
            rel_type = relationship.get("relationship_type", "")

            # Endpoint spelling is part of identity, not a linguistic alias.
            source_key = _get_normalized_entity_key(source)
            target_key = _get_normalized_entity_key(target)
            rel_type_key = rel_type.upper()

            relationship_key = (source_key, target_key, rel_type_key)

            if relationship_key not in relationships_set:
                relationships_set.add(relationship_key)
                relationships.append(relationship)

    return {
        "characters": list(characters_map.values()),
        "world_items": list(world_items_map.values()),
        "relationships": relationships,
    }
