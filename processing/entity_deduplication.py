# processing/entity_deduplication.py
"""Prevent creation of duplicate entities during knowledge extraction.

This module provides:
- Pre-insert duplicate checks using name similarity and optional embedding similarity.
- Post-extraction (phase 2) duplicate discovery using relationship-pattern overlap.
- Merge utilities that transfer relationships from a duplicate node into a canonical node.

Notes:
    - Name similarity is computed in Neo4j using `apoc.text.levenshteinSimilarity()` and is
      compared against `config.DUPLICATE_PREVENTION_SIMILARITY_THRESHOLD` when searching for
      candidates.
    - Character matching has a special-case: if an existing character's first token matches the
      provided name case-insensitively, the similarity is treated as `0.95`.
    - Embedding similarity checks are gated by `config.ENABLE_ENTITY_EMBEDDING_DEDUPLICATION` and
      require the configured Neo4j vector indexes to exist. Embedding calls depend on the LLM
      embedding provider and may be non-deterministic.
"""

import hashlib
import re
from typing import Any

import structlog

from core.db_manager import neo4j_manager
from utils import classify_category_label

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


async def check_entity_similarity(
    name: str,
    entity_type: str,
    category: str = "",
    description: str = "",
) -> dict[str, Any] | None:
    """Check whether an entity already exists that should be treated as the same node.

    The lookup uses a tiered strategy:
    1) Characters: exact/case-insensitive match, first-name match, and Levenshtein similarity.
    2) Optional: embedding similarity via Neo4j vector indexes (if enabled by config).
    3) World elements: Levenshtein similarity with a category/label compatibility check.

    Args:
        name: Proposed entity name.
        entity_type: Entity domain; must be `"character"` or `"world_element"`.
        category: World-element category used to select a target label/index.
        description: Optional descriptive context used only for the embedding query text.

    Returns:
        An entity match payload when a candidate exceeds the configured threshold; otherwise `None`.
        The payload contains `existing_id`, `existing_name`, `existing_category`, `existing_labels`,
        `existing_description`, `similarity`, and `similarity_source`.

    Raises:
        ValueError: If `entity_type` is not `"character"` or `"world_element"`.

    Notes:
        - Character first-name matches are returned with similarity `0.95`.
        - Name-based thresholds are controlled by `config.DUPLICATE_PREVENTION_SIMILARITY_THRESHOLD`.
        - Embedding matches require `similarity >= config.ENTITY_EMBEDDING_DEDUPLICATION_SIMILARITY_THRESHOLD`.
    """
    try:
        import config

        if entity_type not in ["character", "world_element"]:
            raise ValueError("entity_type must be 'character' or 'world_element'")

        name_threshold = config.DUPLICATE_PREVENTION_SIMILARITY_THRESHOLD

        # Keep the existing first-name matching behavior for Characters.
        if entity_type == "character":
            similarity_query = """
            MATCH (c:Character)
            WITH c,
                 CASE
                   WHEN c.name CONTAINS ' '
                   THEN split(c.name, ' ')[0]
                   ELSE c.name
                 END as first_name
            WITH c, first_name,
                 CASE
                   WHEN toLower(first_name) = toLower($name)
                   THEN 0.95
                   ELSE apoc.text.levenshteinSimilarity(toLower(c.name), toLower($name))
                 END as computed_similarity
            WHERE c.name = $name OR
                  toLower(c.name) = toLower($name) OR
                  toLower(first_name) = toLower($name) OR
                  computed_similarity > $threshold
            RETURN c.id as existing_id,
                   c.name as existing_name,
                   labels(c) as existing_labels,
                   c.description as existing_description,
                   computed_similarity as similarity
            ORDER BY similarity DESC
            LIMIT 1
            """
            params = {"name": name, "threshold": name_threshold}
            results = await neo4j_manager.execute_read_query(similarity_query, params)
            if results:
                similar_entity = results[0]
                similarity_score = similar_entity.get("similarity", 0.0)
                is_first_name_match = similarity_score == 0.95 and isinstance(similar_entity.get("existing_name"), str) and similar_entity["existing_name"].split()[0].lower() == name.lower()
                match_type = "first name match" if is_first_name_match else "similarity match"
                logger.info(f"Entity similarity check for '{name}' (type: {entity_type}): " f"Found {match_type} with '{similar_entity['existing_name']}' " f"(similarity: {similarity_score:.2f})")
                return {
                    "existing_id": similar_entity.get("existing_id"),
                    "existing_name": similar_entity["existing_name"],
                    "existing_category": None,
                    "existing_labels": similar_entity.get("existing_labels", []),
                    "existing_description": similar_entity.get("existing_description", ""),
                    "similarity": similarity_score,
                    "similarity_source": "name",
                }

        # Semantic (embedding) similarity path, gated behind config so unit tests remain deterministic.
        if config.ENABLE_ENTITY_EMBEDDING_DEDUPLICATION:
            from core.entity_embedding_service import compute_entity_embedding_text
            from core.llm_interface_refactored import llm_service

            if entity_type == "character":
                index_name = config.NEO4J_CHARACTER_ENTITY_VECTOR_INDEX_NAME
                embedding_text = compute_entity_embedding_text(
                    name=name,
                    category="",
                    description=description,
                )
                cypher_query = """
                CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
                YIELD node, score
                RETURN
                    node.id AS existing_id,
                    node.name AS existing_name,
                    node.category AS existing_category,
                    labels(node) AS existing_labels,
                    node.description AS existing_description,
                    score AS similarity
                ORDER BY similarity DESC
                LIMIT 5
                """
                params = {
                    "index_name": index_name,
                    "top_k": int(config.ENTITY_EMBEDDING_DEDUPLICATION_TOP_K),
                }
            else:
                target_label = classify_category_label(category)
                if target_label not in ["Location", "Item", "Event"]:
                    target_label = "Item"

                if target_label == "Location":
                    index_name = config.NEO4J_LOCATION_ENTITY_VECTOR_INDEX_NAME
                elif target_label == "Event":
                    index_name = config.NEO4J_EVENT_ENTITY_VECTOR_INDEX_NAME
                else:
                    index_name = config.NEO4J_ITEM_ENTITY_VECTOR_INDEX_NAME

                embedding_text = compute_entity_embedding_text(
                    name=name,
                    category=category,
                    description=description,
                )
                cypher_query = """
                CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
                YIELD node, score
                RETURN
                    node.id AS existing_id,
                    node.name AS existing_name,
                    node.category AS existing_category,
                    labels(node) AS existing_labels,
                    node.description AS existing_description,
                    score AS similarity
                ORDER BY similarity DESC
                LIMIT 5
                """
                params = {
                    "index_name": index_name,
                    "top_k": int(config.ENTITY_EMBEDDING_DEDUPLICATION_TOP_K),
                }

            query_embedding = await llm_service.async_get_embedding(embedding_text)
            if query_embedding is not None:
                query_vector = neo4j_manager.embedding_to_list(query_embedding)
                if query_vector:
                    vector_results = await neo4j_manager.execute_read_query(
                        cypher_query,
                        {
                            **params,
                            "query_vector": query_vector,
                        },
                    )

                    if vector_results:
                        best = vector_results[0]
                        similarity_score = float(best.get("similarity", 0.0) or 0.0)
                        if similarity_score >= float(config.ENTITY_EMBEDDING_DEDUPLICATION_SIMILARITY_THRESHOLD):
                            logger.info(
                                "Entity similarity check via embeddings",
                                name=name,
                                entity_type=entity_type,
                                similarity=similarity_score,
                                existing_name=best.get("existing_name"),
                            )
                            return {
                                "existing_id": best.get("existing_id"),
                                "existing_name": best.get("existing_name"),
                                "existing_category": best.get("existing_category"),
                                "existing_labels": best.get("existing_labels", []),
                                "existing_description": best.get("existing_description", ""),
                                "similarity": similarity_score,
                                "similarity_source": "embedding",
                            }

        # Fallback to legacy name similarity behavior for world elements (and for characters when no match found).
        if entity_type != "character":
            similarity_query = """
            MATCH (w)
            WHERE (w:Location OR w:Item OR w:Event)
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
            params = {"name": name, "threshold": name_threshold}

            results = await neo4j_manager.execute_read_query(similarity_query, params)
            if not results:
                return None

            target_label = classify_category_label(category)

            for similar_entity in results:
                existing_cat = similar_entity.get("existing_category", "")
                existing_labels = similar_entity.get("existing_labels", [])
                existing_label = classify_category_label(existing_cat)
                is_compatible = (existing_label == target_label) or (target_label in existing_labels)
                if not is_compatible:
                    continue

                similarity_score = similar_entity.get("similarity", 0.0)
                logger.info(f"Entity similarity check for '{name}' (type: {entity_type}): " f"Found similarity match with '{similar_entity['existing_name']}' " f"(similarity: {similarity_score:.2f})")
                return {
                    "existing_id": similar_entity.get("existing_id"),
                    "existing_name": similar_entity["existing_name"],
                    "existing_category": similar_entity.get("existing_category"),
                    "existing_labels": similar_entity.get("existing_labels", []),
                    "existing_description": similar_entity.get("existing_description", ""),
                    "similarity": similarity_score,
                    "similarity_source": "name",
                }

        return None

    except Exception as e:
        logger.error(f"Error checking entity similarity for '{name}': {e}", exc_info=True)
        return None


async def should_merge_entities(
    new_name: str,
    new_description: str,
    existing_entity: dict[str, Any],
    similarity_threshold: float = 0.51,
) -> bool:
    """Decide whether a candidate entity should be merged into an existing entity.

    Args:
        new_name: Proposed entity name.
        new_description: Proposed entity description.
        existing_entity: Match payload returned by `check_entity_similarity()`.
        similarity_threshold: Minimum similarity required to recommend a merge.

    Returns:
        `True` if a merge is recommended; otherwise `False`.

    Notes:
        - This decision uses `existing_entity["similarity"]` as the primary signal.
        - If `similarity >= similarity_threshold`, a merge is always recommended.
        - A secondary rule recommends a merge when `similarity >= 0.7` and both descriptions are
          present, even if `similarity_threshold` is higher than the observed similarity.
        - No description-to-description similarity is computed here.
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
    """Prevent duplicate character creation by reusing an existing canonical name.

    Args:
        name: Proposed character name.
        description: Proposed character description (used only in the merge decision).
        similarity_threshold: Minimum similarity required to reuse an existing character name. If
            unset, uses `config.DUPLICATE_PREVENTION_SIMILARITY_THRESHOLD`.

    Returns:
        The existing character name to use when a duplicate is detected; otherwise `None`.

    Notes:
        - This function is gated by `config.ENABLE_DUPLICATE_PREVENTION` and
          `config.DUPLICATE_PREVENTION_CHARACTER_ENABLED`.
        - The similarity lookup itself is name-driven; `description` does not affect the initial
          candidate search.
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
    """Prevent duplicate world-element creation by reusing an existing node ID.

    Args:
        name: Proposed world-element name.
        category: World-element category used for label/index selection and compatibility checks.
        description: Proposed description (used only for embedding text and merge decision).
        similarity_threshold: Minimum similarity required to reuse an existing node. If unset, uses
            `config.DUPLICATE_PREVENTION_SIMILARITY_THRESHOLD`.

    Returns:
        The existing node ID to use when a duplicate is detected; otherwise `None`.

    Notes:
        - This function is gated by `config.ENABLE_DUPLICATE_PREVENTION` and
          `config.DUPLICATE_PREVENTION_WORLD_ITEM_ENABLED`.
        - For non-character entities, name candidates are filtered for category/label compatibility
          before returning a match.
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
    """Compute relationship-pattern similarity between two entities.

    The similarity is computed as a Jaccard similarity between sets of
    `"RELATIONSHIP_TYPE:target_name"` patterns derived from each entity's outgoing relationships.

    Args:
        entity1_name: Name of the first entity.
        entity2_name: Name of the second entity.
        entity_type: Entity domain; `"character"` or `"world_element"`.

    Returns:
        A score in `[0.0, 1.0]` representing relationship pattern overlap.

    Notes:
        - If either entity has no outgoing relationships, this returns `0.0`.
        - The score is direction-sensitive: only outgoing relationships are considered.
        - Target identity is based on `target.name`, so unnamed targets reduce signal.
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
            WHERE (e1:Location OR e1:Item OR e1:Event)
              AND e1.name = $name1
            OPTIONAL MATCH (e1)-[r1]->(target1)
            WITH e1, collect(DISTINCT {type: type(r1), target: target1.name}) as rels1

            MATCH (e2)
            WHERE (e2:Location OR e2:Item OR e2:Event)
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
    name_similarity_threshold: float = 0.55,
    relationship_similarity_threshold: float = 0.55,
) -> list[tuple[str, str, float, float]]:
    """Find likely duplicates using name similarity plus relationship-pattern overlap.

    This is phase 2 deduplication intended to run after relationship extraction. It first finds
    candidate pairs with moderately similar names, then retains only those with high relationship
    similarity.

    Args:
        entity_type: Entity domain; `"character"` or `"world_element"`.
        name_similarity_threshold: Minimum Levenshtein name similarity used to generate candidates.
        relationship_similarity_threshold: Minimum relationship Jaccard similarity required to
            return a pair.

    Returns:
        A list of `(entity1_name, entity2_name, name_similarity, relationship_similarity)` tuples.

    Notes:
        - Candidates are restricted to `name_similarity < 0.8` to exclude pairs that should have
          been merged during phase 1.
        - The initial candidate search is capped (currently `LIMIT 50`) to bound database work and
          follow-on similarity computation.
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
            WHERE (e1:Location OR e1:Item OR e1:Event)
            MATCH (e2)
            WHERE (e2:Location OR e2:Item OR e2:Event)
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
    """Merge two duplicate nodes by transferring relationships to a canonical node.

    Args:
        entity1_name: Name of the first entity.
        entity2_name: Name of the second entity.
        entity_type: Entity domain; `"character"` or `"world_element"`.
        keep_entity: Name of the node to keep as canonical. If unset, the canonical node is chosen
            by the lowest `created_chapter` value (missing values default to a large sentinel).

    Returns:
        `True` if the merge completed without raising; otherwise `False`.

    Notes:
        - The merge is implemented in Cypher using APOC procedures and will `DETACH DELETE` the
          duplicate node after relationship transfer.
        - Relationship transfer attempts to avoid duplicating relationships by matching existing
          relationships by relationship type and endpoint.
        - If multiple relationships of the same type exist between the same endpoints, this merge
          does not consolidate them into a single relationship; it only avoids creating an
          additional relationship for the transferred edge.
        - Only a small set of node properties is updated here (`deduplication_merged_from` and
          `last_updated`). Other node properties (e.g., `description`) are not explicitly merged in
          this function.
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
                WHERE (e1:Location OR e1:Item OR e1:Event)
                  AND e1.name = $name1
                MATCH (e2)
                WHERE (e2:Location OR e2:Item OR e2:Event)
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

            // Transfer all relationships from duplicate to canonical while preserving relationship types.
            //
            // Deduplication rule:
            // - If the canonical node already has a relationship of the same type to the same other node,
            //   keep the canonical relationship as-is and only add missing properties from the duplicate.
            // - Otherwise, recreate the relationship with the original type and full properties.
            CALL () {
              WITH canonical, duplicate
                OPTIONAL MATCH (other)-[r_in]->(duplicate)
                WHERE other <> canonical
                WITH canonical, collect({other: other, rel_type: type(r_in), rel_props: properties(r_in)}) AS incoming
                UNWIND incoming AS rel
                WITH canonical, rel.other AS other, rel.rel_type AS rel_type, rel.rel_props AS rel_props
                WHERE other IS NOT NULL AND rel_type IS NOT NULL
                OPTIONAL MATCH (other)-[existing]->(canonical)
                WHERE type(existing) = rel_type
                CALL apoc.do.when(
                    existing IS NULL,
                    'CALL apoc.create.relationship($other, $rel_type, $rel_props, $canonical) YIELD rel RETURN rel',
                    'SET existing = apoc.map.merge($rel_props, properties(existing)) RETURN existing AS rel',
                    {other: other, rel_type: rel_type, rel_props: rel_props, canonical: canonical, existing: existing}
                ) YIELD value
                RETURN count(*) AS incoming_processed
            }

            CALL () {
              WITH canonical, duplicate
                OPTIONAL MATCH (duplicate)-[r_out]->(other)
                WHERE other <> canonical
                WITH canonical, collect({other: other, rel_type: type(r_out), rel_props: properties(r_out)}) AS outgoing
                UNWIND outgoing AS rel
                WITH canonical, rel.other AS other, rel.rel_type AS rel_type, rel.rel_props AS rel_props
                WHERE other IS NOT NULL AND rel_type IS NOT NULL
                OPTIONAL MATCH (canonical)-[existing]->(other)
                WHERE type(existing) = rel_type
                CALL apoc.do.when(
                    existing IS NULL,
                    'CALL apoc.create.relationship($canonical, $rel_type, $rel_props, $other) YIELD rel RETURN rel',
                    'SET existing = apoc.map.merge($rel_props, properties(existing)) RETURN existing AS rel',
                    {other: other, rel_type: rel_type, rel_props: rel_props, canonical: canonical, existing: existing}
                ) YIELD value
                RETURN count(*) AS outgoing_processed
            }

            SET canonical.deduplication_merged_from = coalesce(canonical.deduplication_merged_from, []) + [$duplicate],
                canonical.last_updated = timestamp()

            DETACH DELETE duplicate

            RETURN canonical.name as merged_name
            """
        else:
            merge_query = """
            MATCH (canonical)
            WHERE (canonical:Location OR canonical:Item OR canonical:Event)
              AND canonical.name = $canonical
            MATCH (duplicate)
            WHERE (duplicate:Location OR duplicate:Item OR duplicate:Event)
              AND duplicate.name = $duplicate

            // Transfer all relationships from duplicate to canonical while preserving relationship types.
            //
            // Deduplication rule:
            // - If the canonical node already has a relationship of the same type to the same other node,
            //   keep the canonical relationship as-is and only add missing properties from the duplicate.
            // - Otherwise, recreate the relationship with the original type and full properties.
            CALL () {
              WITH canonical, duplicate
                OPTIONAL MATCH (other)-[r_in]->(duplicate)
                WHERE other <> canonical
                WITH canonical, collect({other: other, rel_type: type(r_in), rel_props: properties(r_in)}) AS incoming
                UNWIND incoming AS rel
                WITH canonical, rel.other AS other, rel.rel_type AS rel_type, rel.rel_props AS rel_props
                WHERE other IS NOT NULL AND rel_type IS NOT NULL
                OPTIONAL MATCH (other)-[existing]->(canonical)
                WHERE type(existing) = rel_type
                CALL apoc.do.when(
                    existing IS NULL,
                    'CALL apoc.create.relationship($other, $rel_type, $rel_props, $canonical) YIELD rel RETURN rel',
                    'SET existing = apoc.map.merge($rel_props, properties(existing)) RETURN existing AS rel',
                    {other: other, rel_type: rel_type, rel_props: rel_props, canonical: canonical, existing: existing}
                ) YIELD value
                RETURN count(*) AS incoming_processed
            }

            CALL () {
              WITH canonical, duplicate
                OPTIONAL MATCH (duplicate)-[r_out]->(other)
                WHERE other <> canonical
                WITH canonical, collect({other: other, rel_type: type(r_out), rel_props: properties(r_out)}) AS outgoing
                UNWIND outgoing AS rel
                WITH canonical, rel.other AS other, rel.rel_type AS rel_type, rel.rel_props AS rel_props
                WHERE other IS NOT NULL AND rel_type IS NOT NULL
                OPTIONAL MATCH (canonical)-[existing]->(other)
                WHERE type(existing) = rel_type
                CALL apoc.do.when(
                    existing IS NULL,
                    'CALL apoc.create.relationship($canonical, $rel_type, $rel_props, $other) YIELD rel RETURN rel',
                    'SET existing = apoc.map.merge($rel_props, properties(existing)) RETURN existing AS rel',
                    {other: other, rel_type: rel_type, rel_props: rel_props, canonical: canonical, existing: existing}
                ) YIELD value
                RETURN count(*) AS outgoing_processed
            }

            SET canonical.deduplication_merged_from = coalesce(canonical.deduplication_merged_from, []) + [$duplicate],
                canonical.last_updated = timestamp()

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
