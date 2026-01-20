# core/langgraph/nodes/extraction_nodes.py
"""Consolidate extracted entities from scene-level extraction.

This module provides the consolidate_extraction node used by both chapter-level
and scene-level extraction subgraphs to externalize extraction results.

Notes:
    The consolidate_extraction node externalizes merged extraction results to
    reduce state bloat during the LangGraph workflow.
"""

from __future__ import annotations

import structlog

from core.langgraph.content_manager import ContentManager, require_project_dir
from core.langgraph.state import NarrativeState

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
        - extracted_entities_ref: Content reference for persisted entities.
        - extracted_relationships_ref: Content reference for persisted relationships.
        - current_node: `"consolidate_extraction"`.

    Notes:
        This node performs filesystem I/O via [`ContentManager`](core/langgraph/content_manager.py:70).
    """
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
            "extracted_entities_ref": entities_ref,
            "extracted_relationships_ref": relationships_ref,
            "current_node": "consolidate_extraction",
        }
    else:
        # Backward compatibility: externalize in-memory data
        from core.langgraph.content_manager import (
            get_extracted_entities,
            get_extracted_relationships,
            save_extracted_entities,
            save_extracted_relationships,
        )

        logger.info(
            "consolidate_extraction: externalizing in-memory data (backward compatibility)",
            chapter=chapter_number,
        )

        # Get entities and relationships from state (with fallback to in-state data)
        extracted_entities = get_extracted_entities(state, content_manager)
        extracted_relationships = get_extracted_relationships(state, content_manager)

        # Externalize the data
        current_version = content_manager.get_latest_version("extracted_entities", f"chapter_{chapter_number}") + 1

        entities_ref = save_extracted_entities(
            extracted_entities,
            content_manager,
            chapter_number=chapter_number,
            version=current_version,
        )

        relationships_ref = save_extracted_relationships(
            extracted_relationships,
            content_manager,
            chapter_number=chapter_number,
            version=current_version,
        )

        logger.info(
            "consolidate_extraction: in-memory data externalized",
            chapter=chapter_number,
            version=current_version,
            entities_count=len(extracted_entities.get("characters", [])) + len(extracted_entities.get("world_items", [])),
            relationships_count=len(extracted_relationships),
        )

        return {
            "extracted_entities_ref": entities_ref,
            "extracted_relationships_ref": relationships_ref,
            "current_node": "consolidate_extraction",
        }
