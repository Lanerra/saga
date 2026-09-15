# core/langgraph/nodes/commit_validation.py
"""
Validation and filtering helpers for entity/relationship commit operations.

Extracted from commit_node.py as part of the module-split refactor (I4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from core.langgraph.nodes.scene_extraction_validation import validate_lexical_entity_name
from core.langgraph.state import ExtractedEntity

if TYPE_CHECKING:
    from core.langgraph.state import ExtractedRelationship


logger = structlog.get_logger(__name__)


def _filter_invalid_relationships(
    relationships: list[ExtractedRelationship],
) -> list[ExtractedRelationship]:
    """Validate the complete batch; never manufacture success by dropping rows.

    The historical function name remains for callers. Source grounding belongs
    to scene extraction; this boundary enforces the same lexical exclusions.
    """
    for relationship in relationships:
        validate_lexical_entity_name(relationship.source_name)
        validate_lexical_entity_name(relationship.target_name)
    return relationships


def _deduplicate_entity_list(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    """Remove within-batch duplicate entities by name.

    Args:
        entities: Extracted entities for a single commit batch.

    Returns:
        A list with duplicate names removed (keeping the first occurrence).
    """
    seen_names: dict[str, ExtractedEntity] = {}
    unique_entities: list[ExtractedEntity] = []

    for entity in entities:
        if entity.name not in seen_names:
            unique_entities.append(entity)
            seen_names[entity.name] = entity
        else:
            if entity != seen_names[entity.name]:
                raise ValueError("Conflicting same-name entity inputs")
            logger.debug(
                "_deduplicate_entity_list: skipping duplicate",
                name=entity.name,
                type=entity.type,
            )

    if len(unique_entities) < len(entities):
        logger.info(
            "_deduplicate_entity_list: removed duplicates",
            original_count=len(entities),
            unique_count=len(unique_entities),
            duplicates_removed=len(entities) - len(unique_entities),
        )

    return unique_entities
