# core/langgraph/nodes/relationship_normalization_node.py
"""
Relationship normalization node for LangGraph workflow.

This node normalizes extracted relationships against accumulated vocabulary,
reducing type explosion while maintaining creative flexibility.
"""

from __future__ import annotations

from typing import Any

import structlog

import config
from core.langgraph.content_manager import (
    ContentManager,
    get_extracted_relationships,
    set_extracted_relationships,
)
from core.langgraph.state import ExtractedRelationship, NarrativeState
from core.relationship_normalization_service import normalization_service

logger = structlog.get_logger(__name__)


async def normalize_relationships(state: NarrativeState) -> dict[str, Any]:
    """
    Normalize extracted relationship types against accumulated vocabulary.

    This node:
    1. Compares each extracted relationship against existing vocabulary
    2. Normalizes semantically similar relationships to canonical forms
    3. Adds genuinely novel relationships to vocabulary
    4. Updates usage statistics for monitoring

    Args:
        state: Current narrative state with extracted relationships

    Returns:
        Updated state with normalized relationships and updated vocabulary
    """
    # Early exit if normalization disabled
    if not config.ENABLE_RELATIONSHIP_NORMALIZATION:
        logger.info("Relationship normalization disabled, skipping")
        return {
            "current_node": "normalize_relationships",
        }

    # Initialize content manager
    content_manager = ContentManager(state.get("project_dir", ""))

    # Get current relationships and vocabulary
    extracted_rels_dicts = get_extracted_relationships(state, content_manager)
    vocabulary = state.get("relationship_vocabulary", {})
    current_chapter = state.get("current_chapter", 1)

    logger.info(
        "normalize_relationships: starting",
        chapter=current_chapter,
        relationships=len(extracted_rels_dicts),
        vocabulary_size=len(vocabulary),
    )

    if not extracted_rels_dicts:
        logger.info("No relationships to normalize")
        return {
            "current_node": "normalize_relationships",
        }

    # Convert dicts to ExtractedRelationship objects
    extracted_rels = []
    for rel_dict in extracted_rels_dicts:
        try:
            rel = ExtractedRelationship(**rel_dict)
            extracted_rels.append(rel)
        except Exception as e:
            logger.warning(
                "Failed to parse extracted relationship",
                rel_dict=rel_dict,
                error=str(e),
            )
            continue

    # Process each relationship
    normalized_rels = []
    normalized_count = 0
    novel_count = 0

    for rel in extracted_rels:
        original_type = rel.relationship_type

        # Normalize relationship type
        normalized_type, was_normalized, similarity = (
            await normalization_service.normalize_relationship_type(
                rel_type=original_type,
                rel_description=rel.description,
                vocabulary=vocabulary,
                current_chapter=current_chapter,
            )
        )

        # Update usage in vocabulary
        vocabulary = normalization_service.update_vocabulary_usage(
            vocabulary=vocabulary,
            rel_type=normalized_type,
            rel_description=rel.description,
            current_chapter=current_chapter,
            was_normalized=was_normalized,
            original_type=original_type if was_normalized else None,
        )

        # Track metrics
        if was_normalized:
            normalized_count += 1
        else:
            # Check if truly novel (not just first occurrence of canonical form)
            if normalized_type == original_type and normalized_type not in state.get("relationship_vocabulary", {}):
                novel_count += 1

        # Create normalized relationship
        normalized_rel = ExtractedRelationship(
            source_name=rel.source_name,
            target_name=rel.target_name,
            relationship_type=normalized_type,  # Use normalized type
            description=rel.description,
            chapter=rel.chapter,
            confidence=rel.confidence,
            source_type=getattr(rel, "source_type", None),
            target_type=getattr(rel, "target_type", None),
        )

        normalized_rels.append(normalized_rel)

    # Prune vocabulary if needed (every 5 chapters)
    if current_chapter % 5 == 0:
        vocabulary = normalization_service.prune_vocabulary(vocabulary, current_chapter)

    # Log statistics
    logger.info(
        "normalize_relationships: complete",
        chapter=current_chapter,
        total_relationships=len(extracted_rels),
        normalized=normalized_count,
        novel=novel_count,
        vocabulary_size=len(vocabulary),
        top_relationships=_get_top_relationships(vocabulary, limit=5),
    )

    # Save normalized relationships back to content manager
    set_extracted_relationships(content_manager, normalized_rels, state)

    # Update state
    return {
        "extracted_relationships": normalized_rels,
        "relationship_vocabulary": vocabulary,
        "relationship_vocabulary_size": len(vocabulary),
        "relationships_normalized_this_chapter": normalized_count,
        "relationships_novel_this_chapter": novel_count,
        "current_node": "normalize_relationships",
    }


def _get_top_relationships(vocabulary: dict, limit: int = 5) -> list[tuple[str, int]]:
    """
    Get top N most-used relationships for logging.

    Returns:
        List of (relationship_type, usage_count) tuples
    """
    if not vocabulary:
        return []

    sorted_rels = sorted(
        vocabulary.items(),
        key=lambda x: x[1].get("usage_count", 0),
        reverse=True,
    )

    return [(rel_type, usage["usage_count"]) for rel_type, usage in sorted_rels[:limit]]


__all__ = ["normalize_relationships"]
