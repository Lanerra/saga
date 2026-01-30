# processing/entity_deduplication.py
"""Deprecated: Entity deduplication algorithms have been removed per Phase 4 requirements.

This module previously provided:
- Pre-insert duplicate checks using name similarity and optional embedding similarity.
- Post-extraction (phase 2) duplicate discovery using relationship-pattern overlap.
- Merge utilities that transfer relationships from a duplicate node into a canonical node.

NOTICE: All deduplication functionality has been removed as per schema-design.md Phase 4.
Entities are now canonical from Stage 1, so deduplication is no longer needed.

Only the generate_entity_id() function is kept for backward compatibility.
"""

import hashlib
import re

import structlog

logger = structlog.get_logger(__name__)


def generate_entity_id(name: str, category: str, chapter: int) -> str:
    """Generate a deterministic entity ID.

    Args:
        name: Entity name.
        category: Entity category (for world items) or `"character"` for characters.
        chapter: Chapter number associated with the entity.

    Returns:
        A stable identifier derived from a normalized form of `name` and `category`.

    Notes:
        `chapter` is currently not incorporated into the ID computation. Callers may still pass a
        chapter value for API compatibility or future extensibility.

        ID generation includes the category to ensure global uniqueness across entity types,
        preventing constraint violations when the same name is used for different entity types.
    """
    # Normalize name for consistent ID generation
    normalized_name = re.sub(r"[^\w\s]", "", name.lower().strip())
    normalized_name = re.sub(r"\s+", "_", normalized_name)

    # Use content-based hashing with extended hash space to reduce collision probability
    # Include both name AND category to ensure different entity types with the same name
    # generate different IDs (e.g., Character "Mars" vs Location "Mars")
    content_hash = hashlib.md5(f"{normalized_name}_{category}".encode()).hexdigest()[:12]

    return f"entity_{content_hash}"


# DEPRECATED: The following functions have been removed as per Phase 4 requirements
# They are no longer needed since entities are canonical from Stage 1 initialization
# and relationships are canonical from Stage 1 as well.

# Removed functions:
# - check_entity_similarity()
# - should_merge_entities()
# - prevent_character_duplication()
# - prevent_world_item_duplication()
# - check_relationship_pattern_similarity()
# - find_relationship_based_duplicates()
# - merge_duplicate_entities()

__all__ = ["generate_entity_id"]
