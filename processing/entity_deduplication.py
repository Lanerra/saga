# processing/entity_deduplication.py
"""
Proactive entity duplicate prevention system.
Prevents creation of duplicate or semantically similar entities during knowledge extraction.
"""

import hashlib
import re
from typing import Any

import structlog

from core.db_manager import neo4j_manager

logger = structlog.get_logger(__name__)


def generate_entity_id(name: str, category: str, chapter: int) -> str:
    """Generate deterministic entity IDs to prevent duplicates.

    Args:
        name: Entity name
        category: Entity category (for world items) or "character" for characters
        chapter: Current chapter number for context

    Returns:
        Deterministic entity ID
    """
    # Normalize name for consistent ID generation
    normalized_name = re.sub(r"[^\w\s]", "", name.lower().strip())
    normalized_name = re.sub(r"\s+", "_", normalized_name)

    # Use content-based hashing instead of category prefixes
    content_hash = hashlib.md5(f"{normalized_name}_{category}".encode()).hexdigest()[:8]

    return f"entity_{content_hash}"


async def check_entity_similarity(
    name: str, entity_type: str, category: str = ""
) -> dict[str, Any] | None:
    """Check for existing entities with same semantic content.

    Args:
        name: Name of the entity to check
        entity_type: Either "character" or "world_element"
        category: For world elements, the category

    Returns:
        Dict with existing entity info if similar entity found, None otherwise
    """
    try:
        import config

        threshold = config.DUPLICATE_PREVENTION_SIMILARITY_THRESHOLD

        if entity_type == "character":
            similarity_query = """
            MATCH (c:Character:Entity)
            WHERE c.name = $name OR
                  toLower(c.name) = toLower($name) OR
                  apoc.text.levenshteinSimilarity(toLower(c.name), toLower($name)) > $threshold
            RETURN c.name as existing_name,
                   labels(c) as existing_labels,
                   c.description as existing_description,
                   apoc.text.levenshteinSimilarity(toLower(c.name), toLower($name)) as similarity
            ORDER BY similarity DESC
            LIMIT 1
            """
            params = {"name": name, "threshold": threshold}
        else:
            similarity_query = """
            MATCH (w:Entity)
            WHERE (w:Object OR w:Artifact OR w:Location OR w:Document OR w:Item OR w:Relic)
              AND (w.name = $name OR
                   toLower(w.name) = toLower($name) OR
                   apoc.text.levenshteinSimilarity(toLower(w.name), toLower($name)) > $threshold)
              AND ($category = '' OR w.category = $category)
            RETURN w.id as existing_id,
                   w.name as existing_name,
                   w.category as existing_category,
                   labels(w) as existing_labels,
                   w.description as existing_description,
                   apoc.text.levenshteinSimilarity(toLower(w.name), toLower($name)) as similarity
            ORDER BY similarity DESC
            LIMIT 1
            """
            params = {"name": name, "category": category, "threshold": threshold}

        result = await neo4j_manager.execute_read_query(similarity_query, params)

        if result:
            similar_entity = result[0]
            similarity_score = similar_entity.get("similarity", 0.0)

            # Log the similarity check
            logger.debug(
                f"Entity similarity check for '{name}' (type: {entity_type}): "
                f"Found similar entity '{similar_entity['existing_name']}' "
                f"with similarity {similarity_score:.2f}"
            )

            # Return the similar entity info
            return {
                "existing_id": similar_entity.get("existing_id"),
                "existing_name": similar_entity["existing_name"],
                "existing_category": similar_entity.get("existing_category"),
                "existing_labels": similar_entity["existing_labels"],
                "existing_description": similar_entity.get("existing_description", ""),
                "similarity": similarity_score,
            }

        return None

    except Exception as e:
        logger.error(
            f"Error checking entity similarity for '{name}': {e}", exc_info=True
        )
        # On error, don't block creation - return None to allow normal processing
        return None


async def should_merge_entities(
    new_name: str,
    new_description: str,
    existing_entity: dict[str, Any],
    similarity_threshold: float = 0.4,
) -> bool:
    """Determine if entities should be merged based on similarity.

    Args:
        new_name: Name of new entity
        new_description: Description of new entity
        existing_entity: Dict with existing entity info
        similarity_threshold: Minimum similarity to consider merging

    Returns:
        True if entities should be merged, False otherwise
    """
    similarity = existing_entity.get("similarity", 0.0)

    # High name similarity - likely the same entity
    if similarity >= similarity_threshold:
        logger.info(
            f"High similarity detected: '{new_name}' vs '{existing_entity['existing_name']}' "
            f"(similarity: {similarity:.2f}). Recommending merge."
        )
        return True

    # Medium similarity - check descriptions if available
    if (
        similarity >= 0.7
        and new_description
        and existing_entity.get("existing_description")
    ):
        # Could add more sophisticated description similarity checking here
        # For now, use name similarity as primary indicator
        logger.info(
            f"Medium similarity detected: '{new_name}' vs '{existing_entity['existing_name']}' "
            f"(similarity: {similarity:.2f}). Recommending merge based on name similarity."
        )
        return True

    return False


async def prevent_character_duplication(
    name: str, description: str = "", similarity_threshold: float | None = None
) -> str | None:
    """Check for character duplicates and return existing name if found.

    Args:
        name: Character name to check
        description: Character description (for additional context)
        similarity_threshold: Similarity threshold for merging (defaults to config value)

    Returns:
        Existing character name if duplicate found, None otherwise
    """
    import config

    # Check if duplicate prevention is enabled
    if (
        not config.ENABLE_DUPLICATE_PREVENTION
        or not config.DUPLICATE_PREVENTION_CHARACTER_ENABLED
    ):
        return None

    # Use config default if not specified
    if similarity_threshold is None:
        similarity_threshold = config.DUPLICATE_PREVENTION_SIMILARITY_THRESHOLD

    similar_entity = await check_entity_similarity(name, "character")

    if similar_entity and await should_merge_entities(
        name, description, similar_entity, similarity_threshold
    ):
        logger.info(
            f"Preventing duplicate character creation: '{name}' is similar to existing "
            f"character '{similar_entity['existing_name']}'. Using existing character."
        )
        return similar_entity["existing_name"]

    return None


async def prevent_world_item_duplication(
    name: str,
    category: str,
    description: str = "",
    similarity_threshold: float | None = None,
) -> str | None:
    """Check for world item duplicates and return existing ID if found.

    Args:
        name: World item name to check
        category: World item category
        description: World item description (for additional context)
        similarity_threshold: Similarity threshold for merging (defaults to config value)

    Returns:
        Existing world item ID if duplicate found, None otherwise
    """
    import config

    # Check if duplicate prevention is enabled
    if (
        not config.ENABLE_DUPLICATE_PREVENTION
        or not config.DUPLICATE_PREVENTION_WORLD_ITEM_ENABLED
    ):
        return None

    # Use config default if not specified
    if similarity_threshold is None:
        similarity_threshold = config.DUPLICATE_PREVENTION_SIMILARITY_THRESHOLD

    similar_entity = await check_entity_similarity(name, "world_element", category)

    if similar_entity and await should_merge_entities(
        name, description, similar_entity, similarity_threshold
    ):
        logger.info(
            f"Preventing duplicate world item creation: '{name}' (category: {category}) "
            f"is similar to existing item '{similar_entity['existing_name']}'. Using existing item."
        )
        return similar_entity.get("existing_id")

    return None
