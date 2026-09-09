# core/langgraph/nodes/scene_extraction_normalization.py
"""Entity deduplication and consolidation for scene extraction.

This module handles merging results from multiple scenes, deduplicating entities
by name using spaCy-based normalization when available.
"""

from __future__ import annotations

from typing import Any

import structlog

from core.langgraph.nodes.scene_extraction_validation import (
    _get_normalized_entity_key,
)

logger = structlog.get_logger(__name__)


def consolidate_scene_extractions(
    scene_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge and deduplicate extraction results from multiple scenes.

    Deduplication strategy:
    - Characters: Dedupe by name (spaCy-based normalization when available), keep longest description
    - World items: Dedupe by name (spaCy-based normalization when available), keep longest description
    - Relationships: Dedupe by (source, target, type) tuple with spaCy normalization

    Args:
        scene_results: List of extraction results from individual scenes.

    Returns:
        Consolidated dict with deduplicated characters, world_items, relationships.
    """
    characters_map: dict[str, dict[str, Any]] = {}
    world_items_map: dict[str, dict[str, Any]] = {}
    relationships_set: set[tuple[str, str, str]] = set()
    relationships: list[dict[str, Any]] = []

    for scene_result in scene_results:
        for character in scene_result.get("characters", []):
            name = character["name"]
            name_key = _get_normalized_entity_key(name)

            if name_key in characters_map:
                existing = characters_map[name_key]
                existing_desc_len = len(existing.get("description", ""))
                new_desc_len = len(character.get("description", ""))

                if new_desc_len > existing_desc_len:
                    characters_map[name_key] = character
            else:
                characters_map[name_key] = character

        for world_item in scene_result.get("world_items", []):
            name = world_item["name"]
            name_key = _get_normalized_entity_key(name)

            if name_key in world_items_map:
                existing = world_items_map[name_key]
                existing_desc_len = len(existing.get("description", ""))
                new_desc_len = len(world_item.get("description", ""))

                if new_desc_len > existing_desc_len:
                    world_items_map[name_key] = world_item
            else:
                world_items_map[name_key] = world_item

        for relationship in scene_result.get("relationships", []):
            source = relationship.get("source_name", "")
            target = relationship.get("target_name", "")
            rel_type = relationship.get("relationship_type", "")

            # Use spaCy normalization for relationship deduplication
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
