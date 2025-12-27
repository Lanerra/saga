# core/langgraph/nodes/extraction_nodes.py
"""Consolidate extracted entities from scene-level extraction.

This module provides the consolidate_extraction node used by both chapter-level
and scene-level extraction subgraphs to externalize extraction results.

Notes:
    The consolidate_extraction node externalizes merged extraction results to
    reduce state bloat during the LangGraph workflow.
"""

from __future__ import annotations

from typing import Any

import structlog

from core.langgraph.content_manager import ContentManager
from core.langgraph.state import (
    ExtractedEntity,
    ExtractedRelationship,
    NarrativeState,
)

logger = structlog.get_logger(__name__)


def consolidate_extraction(state: NarrativeState) -> NarrativeState:
    """Externalize merged extraction results and mark extraction as complete.

    LangGraph reducers merge parallel extraction outputs into the in-memory state.
    This node persists the merged `extracted_entities` and `extracted_relationships`
    to external files to reduce state size for downstream nodes.

    Args:
        state: Workflow state.

    Returns:
        Partial state update containing:
        - extracted_entities: {} (cleared to prevent checkpoint bloat).
        - extracted_relationships: [] (cleared to prevent checkpoint bloat).
        - extracted_entities_ref: Content reference for persisted entities.
        - extracted_relationships_ref: Content reference for persisted relationships.
        - current_node: `"consolidate_extraction"`.

    Notes:
        This node performs filesystem I/O via [`ContentManager`](core/langgraph/content_manager.py:70).
    """
    logger.info(
        "consolidate_extraction: extraction complete",
        characters=len(state.get("extracted_entities", {}).get("characters", [])),
        world_items=len(state.get("extracted_entities", {}).get("world_items", [])),
        relationships=len(state.get("extracted_relationships", [])),
    )

    content_manager = ContentManager(state.get("project_dir", ""))
    chapter_number = state.get("current_chapter", 1)

    current_version = content_manager.get_latest_version("extracted_entities", f"chapter_{chapter_number}") + 1

    extracted_entities = state.get("extracted_entities", {})

    characters_as_dicts: list[dict[str, Any]] = []
    for entity in extracted_entities.get("characters", []):
        if isinstance(entity, ExtractedEntity):
            characters_as_dicts.append(entity.model_dump())
        else:
            if not isinstance(entity, dict):
                raise TypeError("consolidate_extraction: expected extracted character entity to be dict-like")
            characters_as_dicts.append(entity)

    world_items_as_dicts: list[dict[str, Any]] = []
    for entity in extracted_entities.get("world_items", []):
        if isinstance(entity, ExtractedEntity):
            world_items_as_dicts.append(entity.model_dump())
        else:
            if not isinstance(entity, dict):
                raise TypeError("consolidate_extraction: expected extracted world item entity to be dict-like")
            world_items_as_dicts.append(entity)

    entities_dict: dict[str, list[dict[str, Any]]] = {
        "characters": characters_as_dicts,
        "world_items": world_items_as_dicts,
    }

    relationships = state.get("extracted_relationships", [])
    relationships_list: list[dict[str, Any]] = []
    for rel in relationships:
        if isinstance(rel, ExtractedRelationship):
            relationships_list.append(rel.model_dump())
        else:
            if not isinstance(rel, dict):
                raise TypeError("consolidate_extraction: expected extracted relationship to be dict-like")
            relationships_list.append(rel)

    from core.langgraph.content_manager import (
        save_extracted_entities,
        save_extracted_relationships,
    )

    entities_ref = save_extracted_entities(
        content_manager,
        entities_dict,
        chapter_number,
        current_version,
    )

    relationships_ref = save_extracted_relationships(
        content_manager,
        relationships_list,
        chapter_number,
        current_version,
    )

    logger.info(
        "consolidate_extraction: content externalized",
        chapter=chapter_number,
        version=current_version,
        entities_size=entities_ref["size_bytes"],
        relationships_size=relationships_ref["size_bytes"],
    )

    return {
        "extracted_entities": {},
        "extracted_relationships": [],
        "extracted_entities_ref": entities_ref,
        "extracted_relationships_ref": relationships_ref,
        "current_node": "consolidate_extraction",
    }
