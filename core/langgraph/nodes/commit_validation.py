# core/langgraph/nodes/commit_validation.py
"""
Validation and filtering helpers for entity/relationship commit operations.

Extracted from commit_node.py as part of the module-split refactor (I4).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

import config
from core.langgraph.state import ExtractedEntity

if TYPE_CHECKING:
    from core.langgraph.state import ExtractedRelationship


logger = structlog.get_logger(__name__)


def _filter_invalid_relationships(
    relationships: list[ExtractedRelationship],
) -> list[ExtractedRelationship]:
    """Filter out relationships with abstract/invalid entities.

    Rejects relationships where the target is:
    - Generic concepts (truth, knowledge, secrets, understanding)
    - Descriptive phrases (unknown dangers, the bloom)
    - Relationship pairs (Elias and Caleb)
    - Goals/actions (to understand the bloom)

    Args:
        relationships: Extracted relationships to validate.

    Returns:
        Filtered list containing only relationships with valid, concrete entities.
    """
    invalid_patterns = [
        r"^(the|a|an)\s",  # Articles: "the bloom", "a secret"
        r"\s+and\s+",  # Pairs: "Elias and Caleb"
        r"^to\s",  # Goals: "to understand"
        r"(truth|knowledge|secrets?|understanding|consequences)",  # Abstract concepts
        r"(unknown|mysterious)\s",  # Descriptive adjectives
        r"('s|')\s",  # Possessives: "Elias's decision"
    ]

    combined_pattern = "|".join(f"({p})" for p in invalid_patterns)
    pattern = re.compile(combined_pattern, re.IGNORECASE)

    valid = []
    filtered_count = 0

    for rel in relationships:
        target = rel.target_name.strip()
        source = rel.source_name.strip()

        # Reject if target matches invalid pattern
        if pattern.search(target):
            logger.debug(
                "_filter_invalid_relationships: rejected abstract target",
                source=source,
                predicate=rel.relationship_type,
                target=target,
            )
            filtered_count += 1
            continue

        # Reject if source matches invalid pattern
        if pattern.search(source):
            logger.debug(
                "_filter_invalid_relationships: rejected abstract source",
                source=source,
                predicate=rel.relationship_type,
                target=target,
            )
            filtered_count += 1
            continue

        # Reject single-word lowercase concepts (except proper names)
        if " " not in target and target.islower() and target not in config.settings.RELATIONSHIP_LOWERCASE_TARGET_ALLOWLIST:
            logger.debug(
                "_filter_invalid_relationships: rejected lowercase concept",
                target=target,
            )
            filtered_count += 1
            continue

        valid.append(rel)

    if filtered_count > 0:
        logger.info(
            "_filter_invalid_relationships: filtered abstract concepts",
            original_count=len(relationships),
            valid_count=len(valid),
            filtered_count=filtered_count,
        )

    return valid


def _deduplicate_entity_list(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    """Remove within-batch duplicate entities by name.

    Args:
        entities: Extracted entities for a single commit batch.

    Returns:
        A list with duplicate names removed (keeping the first occurrence).
    """
    seen_names: set[str] = set()
    unique_entities: list[ExtractedEntity] = []

    for entity in entities:
        if entity.name not in seen_names:
            unique_entities.append(entity)
            seen_names.add(entity.name)
        else:
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
