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
from utils import classify_category_label

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


async def check_entity_similarity(name: str, entity_type: str, category: str = "") -> dict[str, Any] | None:
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
            # Enhanced query with first name matching
            # Extracts first name from existing character names and checks for match
            similarity_query = """
            MATCH (c:Character)
            WITH c,
                 // Extract first name (everything before first space, or full name if no space)
                 CASE
                   WHEN c.name CONTAINS ' '
                   THEN split(c.name, ' ')[0]
                   ELSE c.name
                 END as first_name
            WITH c, first_name,
                 // Check if query is exactly the first name (case-insensitive)
                 // Give first name matches high similarity (0.95) to ensure deduplication
                 CASE
                   WHEN toLower(first_name) = toLower($name)
                   THEN 0.95
                   ELSE apoc.text.levenshteinSimilarity(toLower(c.name), toLower($name))
                 END as computed_similarity
            WHERE c.name = $name OR
                  toLower(c.name) = toLower($name) OR
                  toLower(first_name) = toLower($name) OR
                  computed_similarity > $threshold
            RETURN c.name as existing_name,
                   labels(c) as existing_labels,
                   c.description as existing_description,
                   computed_similarity as similarity
            ORDER BY similarity DESC
            LIMIT 1
            """
            params = {"name": name, "threshold": threshold}
        else:
            # Relaxed query: Remove strict category check, fetch top matches to filter in Python
            similarity_query = """
            MATCH (w)
            WHERE (w:Location OR w:Item OR w:Event OR w:Organization OR w:Concept)
              AND (w.name = $name OR
                   toLower(w.name) = toLower($name) OR
                   apoc.text.levenshteinSimilarity(toLower(w.name), toLower($name)) > $threshold)
            RETURN w.id as existing_id,
                   w.name as existing_name,
                   w.category as existing_category,
                   labels(w) as existing_labels,
                   w.description as existing_description,
                   apoc.text.levenshteinSimilarity(toLower(w.name), toLower($name)) as similarity
            ORDER BY similarity DESC
            LIMIT 5
            """
            params = {"name": name, "threshold": threshold}

        results = await neo4j_manager.execute_read_query(similarity_query, params)

        if not results:
            return None

        # Filter results for compatibility
        target_label = classify_category_label(category) if entity_type != "character" else "Character"

        for similar_entity in results:
            # Check compatibility for world elements
            if entity_type != "character":
                existing_cat = similar_entity.get("existing_category", "")
                existing_labels = similar_entity.get("existing_labels", [])

                existing_label = classify_category_label(existing_cat)

                # Check if labels match OR if existing node has the target label
                # This handles cases where 'Region' maps to 'Location'
                is_compatible = (existing_label == target_label) or (target_label in existing_labels)

                if not is_compatible:
                    logger.debug(f"Skipping incompatible match: '{similar_entity['existing_name']}' " f"(Category: {existing_cat}, Label: {existing_label}) != Target: {target_label}")
                    continue

            similarity_score = similar_entity.get("similarity", 0.0)

            # Check if this is a first name match (similarity = 0.95)
            is_first_name_match = entity_type == "character" and similarity_score == 0.95 and similar_entity["existing_name"].split()[0].lower() == name.lower()

            # Log the similarity check
            match_type = "first name match" if is_first_name_match else "similarity match"
            logger.info(f"Entity similarity check for '{name}' (type: {entity_type}): " f"Found {match_type} with '{similar_entity['existing_name']}' " f"(similarity: {similarity_score:.2f})")

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
        logger.error(f"Error checking entity similarity for '{name}': {e}", exc_info=True)
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
        logger.info(f"High similarity detected: '{new_name}' vs '{existing_entity['existing_name']}' " f"(similarity: {similarity:.2f}). Recommending merge.")
        return True

    # Medium similarity - check descriptions if available
    if similarity >= 0.7 and new_description and existing_entity.get("existing_description"):
        # Could add more sophisticated description similarity checking here
        # For now, use name similarity as primary indicator
        logger.info(f"Medium similarity detected: '{new_name}' vs '{existing_entity['existing_name']}' " f"(similarity: {similarity:.2f}). Recommending merge based on name similarity.")
        return True

    return False


async def prevent_character_duplication(name: str, description: str = "", similarity_threshold: float | None = None) -> str | None:
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
    if not config.ENABLE_DUPLICATE_PREVENTION or not config.DUPLICATE_PREVENTION_CHARACTER_ENABLED:
        return None

    # Use config default if not specified
    if similarity_threshold is None:
        similarity_threshold = config.DUPLICATE_PREVENTION_SIMILARITY_THRESHOLD

    similar_entity = await check_entity_similarity(name, "character")

    if similar_entity and await should_merge_entities(name, description, similar_entity, similarity_threshold):
        logger.info(f"Preventing duplicate character creation: '{name}' is similar to existing " f"character '{similar_entity['existing_name']}'. Using existing character.")
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
    if not config.ENABLE_DUPLICATE_PREVENTION or not config.DUPLICATE_PREVENTION_WORLD_ITEM_ENABLED:
        return None

    # Use config default if not specified
    if similarity_threshold is None:
        similarity_threshold = config.DUPLICATE_PREVENTION_SIMILARITY_THRESHOLD

    similar_entity = await check_entity_similarity(name, "world_element", category)

    if similar_entity and await should_merge_entities(name, description, similar_entity, similarity_threshold):
        logger.info(f"Preventing duplicate world item creation: '{name}' (category: {category}) " f"is similar to existing item '{similar_entity['existing_name']}'. Using existing item.")
        return similar_entity.get("existing_id")

    return None


async def check_relationship_pattern_similarity(entity1_name: str, entity2_name: str, entity_type: str = "character") -> float:
    """Check if two entities have similar relationship patterns.

    This function compares the relationships of two entities to determine
    if they likely represent the same real-world entity based on shared
    relationship patterns.

    Args:
        entity1_name: Name of first entity
        entity2_name: Name of second entity
        entity_type: Type of entity ("character" or "world_element")

    Returns:
        Similarity score (0.0-1.0) based on relationship pattern overlap
    """
    try:
        # Query relationships for both entities
        if entity_type == "character":
            query = """
            MATCH (e1:Character {name: $name1})
            OPTIONAL MATCH (e1)-[r1]->(target1)
            WITH e1, collect(DISTINCT {type: type(r1), target: target1.name}) as rels1

            MATCH (e2:Character {name: $name2})
            OPTIONAL MATCH (e2)-[r2]->(target2)
            WITH e1, rels1, e2, collect(DISTINCT {type: type(r2), target: target2.name}) as rels2

            RETURN rels1, rels2
            """
        else:
            query = """
            MATCH (e1)
            WHERE (e1:Location OR e1:Item OR e1:Event OR e1:Organization OR e1:Concept)
              AND e1.name = $name1
            OPTIONAL MATCH (e1)-[r1]->(target1)
            WITH e1, collect(DISTINCT {type: type(r1), target: target1.name}) as rels1

            MATCH (e2)
            WHERE (e2:Location OR e2:Item OR e2:Event OR e2:Organization OR e2:Concept)
              AND e2.name = $name2
            OPTIONAL MATCH (e2)-[r2]->(target2)
            WITH e1, rels1, e2, collect(DISTINCT {type: type(r2), target: target2.name}) as rels2

            RETURN rels1, rels2
            """

        params = {"name1": entity1_name, "name2": entity2_name}
        result = await neo4j_manager.execute_read_query(query, params)

        if not result:
            return 0.0

        rels1 = result[0].get("rels1", [])
        rels2 = result[0].get("rels2", [])

        # Filter out None relationships (from OPTIONAL MATCH with no match)
        rels1 = [r for r in rels1 if r.get("type") is not None]
        rels2 = [r for r in rels2 if r.get("type") is not None]

        # If either entity has no relationships, can't determine similarity
        if not rels1 or not rels2:
            return 0.0

        # Calculate relationship pattern similarity
        # Compare both relationship types and targets
        rel1_patterns = {f"{r['type']}:{r['target']}" for r in rels1}
        rel2_patterns = {f"{r['type']}:{r['target']}" for r in rels2}

        # Jaccard similarity: intersection / union
        if not rel1_patterns and not rel2_patterns:
            return 0.0

        intersection = len(rel1_patterns & rel2_patterns)
        union = len(rel1_patterns | rel2_patterns)

        similarity = intersection / union if union > 0 else 0.0

        logger.debug(f"Relationship pattern similarity for '{entity1_name}' vs '{entity2_name}': " f"{similarity:.2f} ({intersection} common / {union} total patterns)")

        return similarity

    except Exception as e:
        logger.error(
            f"Error checking relationship pattern similarity for '{entity1_name}' vs '{entity2_name}': {e}",
            exc_info=True,
        )
        return 0.0


async def find_relationship_based_duplicates(
    entity_type: str = "character",
    name_similarity_threshold: float = 0.6,
    relationship_similarity_threshold: float = 0.7,
) -> list[tuple[str, str, float, float]]:
    """Find potential duplicate entities based on relationship patterns.

    This function identifies entities that:
    1. Have moderately similar names (didn't merge in Phase 1)
    2. Have highly similar relationship patterns

    This is Phase 2 deduplication that runs AFTER relationships are extracted.

    Args:
        entity_type: Type of entity to check ("character" or "world_element")
        name_similarity_threshold: Minimum name similarity to consider (should be lower than Phase 1)
        relationship_similarity_threshold: Minimum relationship pattern similarity

    Returns:
        List of (entity1_name, entity2_name, name_similarity, relationship_similarity) tuples
    """
    try:
        import config

        # Check if Phase 2 deduplication is enabled
        if not config.ENABLE_DUPLICATE_PREVENTION:
            return []

        # Query for pairs of entities with similar names
        if entity_type == "character":
            query = """
            MATCH (e1:Character)
            MATCH (e2:Character)
            WHERE e1.name < e2.name  // Prevent duplicate pairs
              AND e1.name <> e2.name  // Not exactly the same
            WITH e1, e2,
                 apoc.text.levenshteinSimilarity(toLower(e1.name), toLower(e2.name)) as name_sim
            WHERE name_sim >= $name_threshold
              AND name_sim < 0.8  // Exclude pairs that should have merged in Phase 1
            RETURN e1.name as name1, e2.name as name2, name_sim
            ORDER BY name_sim DESC
            LIMIT 50  // Limit to prevent excessive processing
            """
        else:
            query = """
            MATCH (e1)
            WHERE (e1:Location OR e1:Item OR e1:Event OR e1:Organization OR e1:Concept)
            MATCH (e2)
            WHERE (e2:Location OR e2:Item OR e2:Event OR e2:Organization OR e2:Concept)
              AND e1.name < e2.name  // Prevent duplicate pairs
              AND e1.name <> e2.name  // Not exactly the same
            WITH e1, e2,
                 apoc.text.levenshteinSimilarity(toLower(e1.name), toLower(e2.name)) as name_sim
            WHERE name_sim >= $name_threshold
              AND name_sim < 0.8  // Exclude pairs that should have merged in Phase 1
            RETURN e1.name as name1, e2.name as name2, name_sim
            ORDER BY name_sim DESC
            LIMIT 50  // Limit to prevent excessive processing
            """

        params = {"name_threshold": name_similarity_threshold}
        results = await neo4j_manager.execute_read_query(query, params)

        if not results:
            return []

        # Check relationship patterns for each candidate pair
        duplicate_pairs = []

        for result in results:
            name1 = result["name1"]
            name2 = result["name2"]
            name_sim = result["name_sim"]

            # Check relationship pattern similarity
            rel_sim = await check_relationship_pattern_similarity(name1, name2, entity_type)

            if rel_sim >= relationship_similarity_threshold:
                duplicate_pairs.append((name1, name2, name_sim, rel_sim))
                logger.info(f"Found potential duplicate based on relationships: '{name1}' vs '{name2}' " f"(name_sim: {name_sim:.2f}, rel_sim: {rel_sim:.2f})")

        return duplicate_pairs

    except Exception as e:
        logger.error(
            f"Error finding relationship-based duplicates: {e}",
            exc_info=True,
        )
        return []


async def merge_duplicate_entities(
    entity1_name: str,
    entity2_name: str,
    entity_type: str = "character",
    keep_entity: str | None = None,
) -> bool:
    """Merge two duplicate entities into one.

    This function:
    1. Determines which entity to keep (canonical)
    2. Transfers all relationships from duplicate to canonical
    3. Merges properties
    4. Deletes the duplicate entity

    Args:
        entity1_name: Name of first entity
        entity2_name: Name of second entity
        entity_type: Type of entity ("character" or "world_element")
        keep_entity: Which entity to keep (entity1_name or entity2_name).
                     If None, keeps the one that appears earlier (lower created_chapter)

    Returns:
        True if merge was successful, False otherwise
    """
    try:
        # Determine which entity to keep
        if keep_entity is None:
            # Query to find which entity was created first
            if entity_type == "character":
                query = """
                MATCH (e1:Character {name: $name1})
                MATCH (e2:Character {name: $name2})
                RETURN e1.name as name1,
                       e2.name as name2,
                       coalesce(e1.created_chapter, 999) as created1,
                       coalesce(e2.created_chapter, 999) as created2
                """
            else:
                query = """
                MATCH (e1)
                WHERE (e1:Location OR e1:Item OR e1:Event OR e1:Organization OR e1:Concept)
                  AND e1.name = $name1
                MATCH (e2)
                WHERE (e2:Location OR e2:Item OR e2:Event OR e2:Organization OR e2:Concept)
                  AND e2.name = $name2
                RETURN e1.name as name1,
                       e2.name as name2,
                       coalesce(e1.created_chapter, 999) as created1,
                       coalesce(e2.created_chapter, 999) as created2
                """

            params = {"name1": entity1_name, "name2": entity2_name}
            result = await neo4j_manager.execute_read_query(query, params)

            if not result:
                logger.warning(f"Cannot merge entities '{entity1_name}' and '{entity2_name}': entities not found")
                return False

            # Keep the entity that was created first
            canonical = entity1_name if result[0]["created1"] <= result[0]["created2"] else entity2_name
            duplicate = entity2_name if canonical == entity1_name else entity1_name
        else:
            canonical = keep_entity
            duplicate = entity2_name if keep_entity == entity1_name else entity1_name

        # Merge the entities
        if entity_type == "character":
            merge_query = """
            MATCH (canonical:Character {name: $canonical})
            MATCH (duplicate:Character {name: $duplicate})

            // Transfer all relationships from duplicate to canonical
            // Incoming relationships
            OPTIONAL MATCH (other)-[r_in]->(duplicate)
            WHERE other <> canonical
            WITH canonical, duplicate, collect({other: other, rel: r_in, type: type(r_in), props: properties(r_in)}) as incoming

            // Outgoing relationships
            OPTIONAL MATCH (duplicate)-[r_out]->(other)
            WHERE other <> canonical
            WITH canonical, duplicate, incoming, collect({other: other, rel: r_out, type: type(r_out), props: properties(r_out)}) as outgoing

            // Create new relationships
            FOREACH (rel IN incoming |
                FOREACH (o IN CASE WHEN rel.other IS NOT NULL THEN [rel.other] ELSE [] END |
                    FOREACH (t IN CASE WHEN rel.type IS NOT NULL THEN [rel.type] ELSE [] END |
                        MERGE (o)-[new_rel:GENERIC_REL]->(canonical)
                        SET new_rel = rel.props
                    )
                )
            )

            FOREACH (rel IN outgoing |
                FOREACH (o IN CASE WHEN rel.other IS NOT NULL THEN [rel.other] ELSE [] END |
                    FOREACH (t IN CASE WHEN rel.type IS NOT NULL THEN [rel.type] ELSE [] END |
                        MERGE (canonical)-[new_rel:GENERIC_REL]->(o)
                        SET new_rel = rel.props
                    )
                )
            )

            // Merge properties (keep canonical's properties, add any missing from duplicate)
            SET canonical.deduplication_merged_from = coalesce(canonical.deduplication_merged_from, []) + [$duplicate],
                canonical.last_updated = timestamp()

            // Delete the duplicate and its relationships
            DETACH DELETE duplicate

            RETURN canonical.name as merged_name
            """
        else:
            merge_query = """
            MATCH (canonical)
            WHERE (canonical:Location OR canonical:Item OR canonical:Event OR canonical:Organization OR canonical:Concept)
              AND canonical.name = $canonical
            MATCH (duplicate)
            WHERE (duplicate:Location OR duplicate:Item OR duplicate:Event OR duplicate:Organization OR duplicate:Concept)
              AND duplicate.name = $duplicate

            // Transfer all relationships from duplicate to canonical
            // Incoming relationships
            OPTIONAL MATCH (other)-[r_in]->(duplicate)
            WHERE other <> canonical
            WITH canonical, duplicate, collect({other: other, rel: r_in, type: type(r_in), props: properties(r_in)}) as incoming

            // Outgoing relationships
            OPTIONAL MATCH (duplicate)-[r_out]->(other)
            WHERE other <> canonical
            WITH canonical, duplicate, incoming, collect({other: other, rel: r_out, type: type(r_out), props: properties(r_out)}) as outgoing

            // Create new relationships
            FOREACH (rel IN incoming |
                FOREACH (o IN CASE WHEN rel.other IS NOT NULL THEN [rel.other] ELSE [] END |
                    FOREACH (t IN CASE WHEN rel.type IS NOT NULL THEN [rel.type] ELSE [] END |
                        MERGE (o)-[new_rel:GENERIC_REL]->(canonical)
                        SET new_rel = rel.props
                    )
                )
            )

            FOREACH (rel IN outgoing |
                FOREACH (o IN CASE WHEN rel.other IS NOT NULL THEN [rel.other] ELSE [] END |
                    FOREACH (t IN CASE WHEN rel.type IS NOT NULL THEN [rel.type] ELSE [] END |
                        MERGE (canonical)-[new_rel:GENERIC_REL]->(o)
                        SET new_rel = rel.props
                    )
                )
            )

            // Merge properties
            SET canonical.deduplication_merged_from = coalesce(canonical.deduplication_merged_from, []) + [$duplicate],
                canonical.last_updated = timestamp()

            // Delete the duplicate and its relationships
            DETACH DELETE duplicate

            RETURN canonical.name as merged_name
            """

        params = {"canonical": canonical, "duplicate": duplicate}
        await neo4j_manager.execute_write_query(merge_query, params)

        logger.info(f"Successfully merged duplicate entities: '{duplicate}' merged into '{canonical}'")
        return True

    except Exception as e:
        logger.error(
            f"Error merging duplicate entities '{entity1_name}' and '{entity2_name}': {e}",
            exc_info=True,
        )
        return False
