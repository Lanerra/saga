# core/langgraph/nodes/relationship_normalization_node.py
"""Normalize extracted relationship types to a stable vocabulary.

This module defines the relationship normalization node used by the LangGraph
workflow. It maps noisy, free-form relationship types produced by extraction to
a canonicalized set of types, reducing type explosion while preserving novel
relationships when appropriate.

Notes:
    This node updates `extracted_relationships_ref` so downstream nodes that prefer
    externalized relationships (for example, commit) observe normalized results.
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
    """Normalize extracted relationship types against accumulated vocabulary.

    Args:
        state: Workflow state. Reads extracted relationships (preferring externalized
            refs) and the current `relationship_vocabulary`.

    Returns:
        Partial state update containing:
        - extracted_relationships: [] (cleared to prevent checkpoint bloat).
        - extracted_relationships_ref: Content reference for normalized relationships.
        - relationship_vocabulary: Updated vocabulary (including usage stats).
        - relationships_normalized_this_chapter / relationships_novel_this_chapter: Metrics.
        - current_node: `"normalize_relationships"`.

        If normalization is disabled or there are no relationships to process,
        returns a minimal no-op update.

    Notes:
        This node performs model-assisted normalization via
        `normalization_service.normalize_relationship_type()`.
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
    # NOTE: [`get_extracted_relationships()`](core/langgraph/content_manager.py:783) prefers the ref when present.
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
        (
            normalized_type,
            was_normalized,
            similarity,
        ) = await normalization_service.normalize_relationship_type(
            rel_type=original_type,
            rel_description=rel.description,
            vocabulary=vocabulary,
            current_chapter=current_chapter,
        )

        # Check if novel (before updating vocabulary)
        is_novel = False
        if not was_normalized and normalized_type not in vocabulary:
            is_novel = True

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
        elif is_novel:
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

    # Prune vocabulary if needed (every 5 chapters) and not already done for this chapter
    last_pruned_chapter = state.get("last_pruned_chapter", 0)
    if current_chapter % 5 == 0 and current_chapter > last_pruned_chapter:
        vocabulary = normalization_service.prune_vocabulary(vocabulary, current_chapter)
        last_pruned_chapter = current_chapter

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

    # Save normalized relationships back to content manager and update the ref.
    #
    # This fixes the "normalization bypass" where commit prefers reading via
    # [`get_extracted_relationships()`](core/langgraph/content_manager.py:783) and would therefore
    # ignore normalized relationships unless `extracted_relationships_ref` is updated.
    normalized_ref = set_extracted_relationships(content_manager, normalized_rels, state)

    # Update state
    #
    # Source-of-truth contract:
    # - `extracted_relationships_ref` is authoritative for downstream nodes (including commit).
    # - `extracted_relationships` is cleared to prevent checkpoint bloat.
    return {
        "extracted_relationships": [],
        "extracted_relationships_ref": normalized_ref,
        "relationship_vocabulary": vocabulary,
        "relationship_vocabulary_size": len(vocabulary),
        "relationships_normalized_this_chapter": normalized_count,
        "relationships_novel_this_chapter": novel_count,
        "last_pruned_chapter": last_pruned_chapter,
        "current_node": "normalize_relationships",
    }


def _get_top_relationships(vocabulary: dict, limit: int = 5) -> list[tuple[str, int]]:
    """Select the most frequently used relationship types for logging.

    Args:
        vocabulary: Relationship vocabulary mapping types to metadata.
        limit: Maximum number of relationship types to return.

    Returns:
        List of `(relationship_type, usage_count)` tuples ordered by usage descending.
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
