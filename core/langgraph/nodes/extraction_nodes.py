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

from core.langgraph.content_manager import ContentManager, require_project_dir
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
    from core.langgraph.content_manager import (
        get_extracted_entities,
        get_extracted_relationships,
    )

    content_manager = ContentManager(require_project_dir(state))
    chapter_number = state.get("current_chapter", 1)

    # Check if extraction results are already externalized (from extract_from_scenes)
    entities_ref = state.get("extracted_entities_ref")
    relationships_ref = state.get("extracted_relationships_ref")

    if entities_ref and relationships_ref:
        # Data is already externalized, just verify it exists
        logger.info(
            "consolidate_extraction: using pre-externalized content",
            chapter=chapter_number,
            entities_size=entities_ref["size_bytes"],
            relationships_size=relationships_ref["size_bytes"],
        )

        if not content_manager.exists(entities_ref):
            raise FileNotFoundError(f"Externalized entities file not found: {entities_ref['path']}")

        if not content_manager.exists(relationships_ref):
            raise FileNotFoundError(f"Externalized relationships file not found: {relationships_ref['path']}")

        logger.info(
            "consolidate_extraction: content externalized",
            chapter=chapter_number,
            version=entities_ref["version"],
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
    else:
        # Legacy path: externalize from in-memory state
        logger.info(
            "consolidate_extraction: extraction complete",
            characters=len(get_extracted_entities(state, content_manager).get("characters", [])),
            world_items=len(get_extracted_entities(state, content_manager).get("world_items", [])),
            relationships=len(get_extracted_relationships(state, content_manager)),
        )

        current_version = content_manager.get_latest_version("extracted_entities", f"chapter_{chapter_number}") + 1

        extracted_entities = get_extracted_entities(state, content_manager)

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

        relationships = get_extracted_relationships(state, content_manager)
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
