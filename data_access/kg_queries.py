# data_access/kg_queries.py
import re
from typing import Any

import structlog
from async_lru import alru_cache
from config import settings  # MODIFIED
from core.db_manager import neo4j_manager
from kg_constants import (
    KG_IS_PROVISIONAL,
    KG_REL_CHAPTER_ADDED,
    NODE_LABELS,
    RELATIONSHIP_TYPES,
)

logger = structlog.get_logger(__name__)

# Lookup table for canonical node labels to ensure consistent casing
_CANONICAL_NODE_LABEL_MAP: dict[str, str] = {lbl.lower(): lbl for lbl in NODE_LABELS}


def _to_pascal_case(text: str) -> str:
    """Convert underscore or space separated text to PascalCase."""
    parts = re.split(r"[_\s]+", text.strip())
    return "".join(part[:1].upper() + part[1:] for part in parts if part)


def normalize_relationship_type(rel_type: str) -> str:
    """Return a canonical representation of a relationship type."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", rel_type.strip())
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.upper()


async def normalize_existing_relationship_types() -> None:
    """Normalize all stored relationship types to canonical form."""
    query = "MATCH ()-[r:DYNAMIC_REL]->() RETURN DISTINCT r.type AS t"
    try:
        results = await neo4j_manager.execute_read_query(query)
    except Exception as exc:  # pragma: no cover - narrow DB errors
        logger.error("Error reading existing relationship types: %s", exc)
        return

    statements: list[tuple[str, dict[str, Any]]] = []
    for record in results:
        current = record.get("t")
        if not current:
            continue
        normalized = normalize_relationship_type(str(current))
        if normalized != current:
            statements.append(
                (
                    "MATCH ()-[r:DYNAMIC_REL {type: $old}]->() SET r.type = $new",
                    {"old": current, "new": normalized},
                )
            )
    if statements:
        try:
            await neo4j_manager.execute_cypher_batch(statements)
            logger.info("Normalized %d relationship type variations", len(statements))
        except Exception as exc:  # pragma: no cover - log but continue
            logger.error(
                "Failed to update some relationship types: %s", exc, exc_info=True
            )


def _get_cypher_labels(entity_type: str | None) -> str:
    """Helper to create a Cypher label string (e.g., :Character:Entity or :Person:Character:Entity)."""

    entity_label_suffix = ":Entity"  # All nodes get this
    specific_labels_parts: list[str] = []

    if entity_type and entity_type.strip():
        cleaned = re.sub(r"[^a-zA-Z0-9_\s]+", "", entity_type)
        normalized_key = re.sub(r"[_\s]+", "", cleaned).lower()

        canonical = _CANONICAL_NODE_LABEL_MAP.get(normalized_key)
        if canonical is None:
            pascal = _to_pascal_case(cleaned)
            canonical = _CANONICAL_NODE_LABEL_MAP.get(pascal.lower(), pascal)

        if canonical and canonical != "Entity":
            if canonical != "Character":
                specific_labels_parts.append(f":{canonical}")

            if cleaned.strip().lower() == "person" or canonical == "Character":
                if ":Character" not in specific_labels_parts:
                    specific_labels_parts.append(":Character")

    # Order: :Character (if present), then other specific labels (e.g., :Person), then :Entity
    # Remove duplicates and establish order
    final_ordered_labels = []
    if ":Character" in specific_labels_parts:
        final_ordered_labels.append(":Character")

    for label in specific_labels_parts:
        if label not in final_ordered_labels:
            final_ordered_labels.append(label)

    if not final_ordered_labels:
        return entity_label_suffix  # Just ":Entity"

    return "".join(final_ordered_labels) + entity_label_suffix


def _labels_to_list(labels_cypher: str) -> list[str]:
    """Convert a Cypher label string to a list of labels."""
    return [lbl for lbl in labels_cypher.split(":") if lbl]


async def add_kg_triples_batch_to_db(
    structured_triples_data: list[dict[str, Any]],
    chapter_number: int,
    is_from_flawed_draft: bool,
):
    """Insert a batch of knowledge graph triples into Neo4j."""

    if not structured_triples_data:
        logger.info("Neo4j: add_kg_triples_batch_to_db: No structured triples to add.")
        return

    triple_payload: list[dict[str, Any]] = []

    for triple_dict in structured_triples_data:
        subject_info = triple_dict.get("subject")
        predicate_str = triple_dict.get("predicate")
        object_entity_info = triple_dict.get("object_entity")
        object_literal_val = triple_dict.get("object_literal")
        is_literal_object = triple_dict.get("is_literal_object", False)

        if not (
            subject_info
            and isinstance(subject_info, dict)
            and subject_info.get("name")
            and predicate_str
        ):
            logger.warning(
                "Neo4j (Batch): Invalid subject or predicate in triple dict: %s",
                triple_dict,
            )
            continue

        subject_name = str(subject_info["name"]).strip()
        subject_type = subject_info.get("type")
        predicate_clean = str(predicate_str).strip().upper().replace(" ", "_")

        if not subject_name or not predicate_clean:
            logger.warning(
                "Neo4j (Batch): Empty subject name or predicate after stripping: %s",
                triple_dict,
            )
            continue

        subject_labels_list = _labels_to_list(_get_cypher_labels(subject_type))

        rel_props = {
            "type": predicate_clean,
            KG_REL_CHAPTER_ADDED: chapter_number,
            KG_IS_PROVISIONAL: is_from_flawed_draft,
            "confidence": 1.0,
        }

        if is_literal_object:
            if object_literal_val is None:
                logger.warning(
                    "Neo4j (Batch): Literal object is None for triple: %s",
                    triple_dict,
                )
                continue
            triple_payload.append(
                {
                    "subject_labels": subject_labels_list,
                    "subject_name": subject_name,
                    "object_labels": ["Entity", "ValueNode"],
                    "object_props": {
                        "value": str(object_literal_val),
                        "type": "Literal",
                    },
                    "rel_props": rel_props,
                }
            )
        elif (
            object_entity_info
            and isinstance(object_entity_info, dict)
            and object_entity_info.get("name")
        ):
            object_name = str(object_entity_info["name"]).strip()
            object_type = object_entity_info.get("type")
            if not object_name:
                logger.warning(
                    "Neo4j (Batch): Empty object name for entity object in triple: %s",
                    triple_dict,
                )
                continue

            object_labels_list = _labels_to_list(_get_cypher_labels(object_type))
            triple_payload.append(
                {
                    "subject_labels": subject_labels_list,
                    "subject_name": subject_name,
                    "object_labels": object_labels_list,
                    "object_props": {"name": object_name},
                    "rel_props": rel_props,
                }
            )
        else:
            logger.warning(
                "Neo4j (Batch): Invalid or missing object information in triple dict: %s",
                triple_dict,
            )
            continue

    if not triple_payload:
        logger.info("Neo4j: add_kg_triples_batch_to_db: No valid triples to process.")
        return

    cypher = """
    UNWIND $triples AS t
    CALL apoc.merge.node(t.subject_labels, {name: t.subject_name}) YIELD node AS s
    SET s.created_ts = coalesce(s.created_ts, timestamp())
    WITH s, t
    CALL apoc.merge.node(t.object_labels, t.object_props) YIELD node AS o
    SET o.created_ts = coalesce(o.created_ts, timestamp())
    WITH s, o, t
    MERGE (s)-[r:DYNAMIC_REL {type: t.rel_props.type}]->(o)
    ON CREATE SET r += t.rel_props, r.created_ts = timestamp()
    ON MATCH SET r += t.rel_props, r.updated_ts = timestamp()
    """

    try:
        await neo4j_manager.execute_write_query(cypher, {"triples": triple_payload})
        logger.info(
            "Neo4j: Batch processed %d KG triples via UNWIND.",
            len(triple_payload),
        )
    except Exception as e:
        logger.error(
            "Neo4j: Error in batch adding KG triples with UNWIND: %s",
            e,
            exc_info=True,
        )
        raise


async def query_kg_from_db(
    subject: str | None = None,
    predicate: str | None = None,
    obj_val: str | None = None,
    chapter_limit: int | None = None,
    include_provisional: bool = True,
    limit_results: int | None = None,
) -> list[dict[str, Any]]:
    """Query dynamic relationships from the graph with optional filters.

    Args:
        subject: Subject entity name to match.
        predicate: Relationship type to match.
        obj_val: Object entity or literal value to match.
        chapter_limit: Only consider relationships up to this chapter.
        include_provisional: Include provisional relationships when ``True``.
        limit_results: Maximum number of records to return.

    Returns:
        A list of matching relationship dictionaries.
    """
    conditions = []
    parameters: dict[str, str] = {}
    match_clause = "MATCH (s:Entity)-[r:DYNAMIC_REL]->(o) "

    if subject is not None:
        conditions.append("s.name = $subject_param")
        parameters["subject_param"] = subject.strip()
    if predicate is not None:
        conditions.append("r.type = $predicate_param")
        parameters["predicate_param"] = predicate.strip().upper().replace(" ", "_")
    if obj_val is not None:
        obj_val_stripped = obj_val.strip()
        conditions.append(
            """
            ( (o:ValueNode AND o.value = $object_param ) OR
              (NOT o:ValueNode AND o.name = $object_param)
            )
        """
        )
        parameters["object_param"] = obj_val_stripped
    if chapter_limit is not None:
        conditions.append(f"r.{KG_REL_CHAPTER_ADDED} <= $chapter_limit_param")
        parameters["chapter_limit_param"] = str(chapter_limit)
    if not include_provisional:
        conditions.append(
            f"(r.{KG_IS_PROVISIONAL} = FALSE OR r.{KG_IS_PROVISIONAL} IS NULL)"
        )

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

    return_clause = f"""
    RETURN s.name AS subject,
           r.type AS predicate,
           CASE WHEN o:ValueNode THEN o.value ELSE o.name END AS object,
           CASE WHEN o:ValueNode THEN 'Literal' ELSE labels(o)[0] END AS object_type, // Primary label or 'Literal'
           r.{KG_REL_CHAPTER_ADDED} AS {KG_REL_CHAPTER_ADDED},
           r.confidence AS confidence,
           r.{KG_IS_PROVISIONAL} AS {KG_IS_PROVISIONAL}
    """
    order_clause = f" ORDER BY r.{KG_REL_CHAPTER_ADDED} DESC, r.confidence DESC"
    limit_clause_str = (
        f" LIMIT {int(limit_results)}"
        if limit_results is not None and limit_results > 0
        else ""
    )

    parameters = {k: v for k, v in parameters.items() if v is not None}

    full_query = (
        match_clause + where_clause + return_clause + order_clause + limit_clause_str
    )
    try:
        results = await neo4j_manager.execute_read_query(full_query, parameters)
        triples_list: list[dict[str, Any]] = (
            [dict(record) for record in results] if results else []
        )
        logger.debug(
            f"Neo4j: KG query returned {len(triples_list)} results. Query: '{full_query[:200]}...' Params: {parameters}"
        )
        return triples_list
    except Exception as e:
        logger.error(
            f"Neo4j: Error querying KG. Query: '{full_query[:200]}...', Params: {parameters}, Error: {e}",
            exc_info=True,
        )
        return []


async def get_most_recent_value_from_db(
    subject: str,
    predicate: str,
    chapter_limit: int | None = None,
    include_provisional: bool = False,
) -> Any | None:
    """Get the newest relationship value for ``subject`` and ``predicate``."""

    if not subject.strip() or not predicate.strip():
        logger.warning(
            f"Neo4j: get_most_recent_value_from_db: empty subject or predicate. S='{subject}', P='{predicate}'"
        )
        return None

    results = await query_kg_from_db(
        subject=subject,
        predicate=predicate,
        chapter_limit=chapter_limit,
        include_provisional=include_provisional,
        limit_results=1,
    )
    if results and results[0] and "object" in results[0]:
        value = results[0]["object"]
        # Attempt to convert to number if it looks like one, as ValueNode.value stores as string from current triple parsing
        if isinstance(value, str):
            if re.match(r"^-?\d+$", value):
                value = int(value)
            elif re.match(r"^-?\d*\.\d+$", value):
                value = float(value)
            elif value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False

        logger.debug(
            f"Neo4j: Found most recent value for ('{subject}', '{predicate}'): '{value}' (type: {type(value)}) from Ch {results[0].get(KG_REL_CHAPTER_ADDED, 'N/A')}, Prov: {results[0].get(KG_IS_PROVISIONAL)}"
        )
        return value

    logger.debug(
        f"Neo4j: No value found for ({subject}, {predicate}) up to Ch {chapter_limit}, include_provisional={include_provisional}."
    )
    return None


async def get_novel_info_property_from_db(property_key: str) -> Any | None:
    """Return a property value from the NovelInfo node."""
    if not property_key.strip():
        logger.warning("Neo4j: empty property key for NovelInfo query")
        return None

    novel_id_param = settings.MAIN_NOVEL_INFO_NODE_ID  # MODIFIED
    query = f"MATCH (ni:NovelInfo:Entity {{id: $novel_id_param}}) RETURN ni.{property_key} AS value"
    try:
        results = await neo4j_manager.execute_read_query(
            query, {"novel_id_param": novel_id_param}
        )
        if results and results[0] and "value" in results[0]:
            return results[0]["value"]
    except Exception as e:  # pragma: no cover - narrow DB errors
        logger.error(
            f"Neo4j: Error retrieving NovelInfo property '{property_key}': {e}",
            exc_info=True,
        )
    return None


async def get_chapter_context_for_entity(
    entity_name: str | None = None, entity_id: str | None = None
) -> list[dict[str, Any]]:
    """
    Finds chapters where an entity was mentioned or involved to provide context for enrichment.
    Searches by name for Characters/ValueNodes or by ID for WorldElements.
    """
    if not entity_name and not entity_id:
        return []

    match_clause = (
        "MATCH (e {id: $id_param})" if entity_id else "MATCH (e {name: $name_param})"
    )

    params: dict[str, str] = {}
    if entity_id is not None:
        params["id_param"] = entity_id
    elif entity_name is not None:
        params["name_param"] = entity_name

    query = f"""
    {match_clause}

    // Get all paths to potential chapter number sources
    OPTIONAL MATCH (e)-[]->(event) WHERE (event:DevelopmentEvent OR event:WorldElaborationEvent) AND event.chapter_updated IS NOT NULL
    OPTIONAL MATCH (e)-[r:DYNAMIC_REL]-() WHERE r.chapter_added IS NOT NULL

    // Collect all numbers into one list, then process
    WITH
      CASE WHEN e.created_chapter IS NOT NULL THEN [e.created_chapter] ELSE [] END as created_chapter_list,
      COLLECT(DISTINCT event.chapter_updated) as event_chapters,
      COLLECT(DISTINCT r.chapter_added) as rel_chapters

    // Combine, filter out nulls, unwind, get distinct
    WITH created_chapter_list + event_chapters + rel_chapters as all_chapters
    UNWIND all_chapters as chapter_num
    WITH DISTINCT chapter_num
    WHERE chapter_num IS NOT NULL AND chapter_num > 0

    // Now fetch the chapter data
    MATCH (c:{settings.NEO4J_VECTOR_NODE_LABEL} {{number: chapter_num}})
    RETURN c.number as chapter_number, c.summary as summary, c.text as text
    ORDER BY c.number DESC
    LIMIT 5 // Limit context to most recent 5 chapters
    """
    try:
        results = await neo4j_manager.execute_read_query(query, params)
        return results if results else []
    except Exception as e:
        logger.error(
            f"Error getting chapter context for entity '{entity_name or entity_id}': {e}",
            exc_info=True,
        )
        return []


async def find_contradictory_trait_characters(
    contradictory_trait_pairs: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """
    Finds characters who have contradictory traits based on a provided list of pairs.
    e.g. [('Brave', 'Cowardly'), ('Honest', 'Deceitful')]
    """
    if not contradictory_trait_pairs:
        return []

    all_findings = []
    for trait1, trait2 in contradictory_trait_pairs:
        query = """
        MATCH (c:Character)-[:HAS_TRAIT]->(t1:Trait {name: $trait1_param}),
              (c)-[:HAS_TRAIT]->(t2:Trait {name: $trait2_param})
        RETURN c.name AS character_name, t1.name AS trait1, t2.name AS trait2
        """
        params = {"trait1_param": trait1, "trait2_param": trait2}
        try:
            results = await neo4j_manager.execute_read_query(query, params)
            if results:
                all_findings.extend(results)
        except Exception as e:
            logger.error(
                f"Error checking for contradictory traits '{trait1}' vs '{trait2}': {e}",
                exc_info=True,
            )

    return all_findings


async def find_post_mortem_activity() -> list[dict[str, Any]]:
    """
    Finds characters who have relationships or activities recorded in chapters
    after they were marked as dead.
    """
    query = """
    MATCH (c:Character)-[death_rel:DYNAMIC_REL {type: 'IS_DEAD'}]->()
    WHERE death_rel.is_provisional = false OR death_rel.is_provisional IS NULL
    WITH c, death_rel.chapter_added AS death_chapter

    MATCH (c)-[activity_rel:DYNAMIC_REL]->()
    WHERE activity_rel.chapter_added > death_chapter
      AND NOT activity_rel.type IN ['IS_REMEMBERED_AS', 'WAS_FRIEND_OF'] // Exclude retrospective rels
    RETURN DISTINCT c.name as character_name,
           death_chapter,
           collect(
             {
               activity_type: activity_rel.type,
               activity_chapter: activity_rel.chapter_added
             }
           ) AS post_mortem_activities
    LIMIT 20
    """
    try:
        results = await neo4j_manager.execute_read_query(query)
        return results if results else []
    except Exception as e:
        logger.error(f"Error checking for post-mortem activity: {e}", exc_info=True)
        return []


async def find_candidate_duplicate_entities(
    similarity_threshold: float = 0.85, limit: int = 50
) -> list[dict[str, Any]]:
    """
    Finds pairs of entities with similar names using APOC's Levenshtein distance.
    This requires the APOC plugin to be installed in Neo4j.
    """
    query = """
    MATCH (e1:Entity), (e2:Entity)
    WHERE id(e1) < id(e2)
      AND e1.name IS NOT NULL AND e2.name IS NOT NULL
      AND NOT e1:ValueNode AND NOT e2:ValueNode
    WITH e1, e2, apoc.text.distance(e1.name, e2.name) AS distance
    WITH e1, e2, distance, apoc.coll.max([size(e1.name), size(e2.name)]) as max_len
    WHERE max_len > 0 AND (1 - (distance / toFloat(max_len))) >= $threshold
    
    RETURN
      e1.id AS id1, e1.name AS name1, labels(e1) AS labels1,
      e2.id AS id2, e2.name AS name2, labels(e2) AS labels2,
      (1 - (distance / toFloat(max_len))) as similarity
    ORDER BY similarity DESC
    LIMIT $limit
    """
    params = {"threshold": similarity_threshold, "limit": limit}
    try:
        results = await neo4j_manager.execute_read_query(query, params)
        return results if results else []
    except Exception as e:
        if "apoc.text.distance" in str(e) or "apoc.coll.max" in str(e):
            logger.error(
                "A required APOC function was not found. "
                "Please ensure the full APOC Extended plugin is installed."
            )
        else:
            logger.error(
                f"Error finding candidate duplicate entities: {e}", exc_info=True
            )
        return []


async def get_entity_context_for_resolution(
    entity_id: str,
) -> dict[str, Any] | None:
    """
    Gathers comprehensive context for an entity to help an LLM decide on a merge.
    """
    query = """
    MATCH (e:Entity {id: $entity_id})
    OPTIONAL MATCH (e)-[r]-(o:Entity)
    WITH e,
         COUNT(r) as degree,
         COLLECT({
           rel_type: r.type,
           rel_props: properties(r),
           other_node_name: o.name,
           other_node_labels: labels(o)
         })[..10] AS relationships // Limit relationships for context brevity
    RETURN
      e.id AS id,
      e.name AS name,
      labels(e) AS labels,
      properties(e) AS properties,
      degree,
      relationships
    """
    params = {"entity_id": entity_id}
    try:
        results = await neo4j_manager.execute_read_query(query, params)
        return results[0] if results else None
    except Exception as e:
        logger.error(
            f"Error getting context for entity resolution (id: {entity_id}): {e}",
            exc_info=True,
        )
        return None


async def merge_entities(target_id: str, source_id: str) -> bool:
    """
    Merges one entity (source) into another (target) using APOC procedures.
    The source node will be deleted after its relationships are moved.
    """
    query = """
    MATCH (target:Entity {id: $target_id}), (source:Entity {id: $source_id})
    CALL apoc.refactor.mergeNodes([source], target, {
      properties: 'combine',
      mergeRels: true
    }) YIELD node
    RETURN node
    """
    params = {"target_id": target_id, "source_id": source_id}
    try:
        await neo4j_manager.execute_write_query(query, params)
        logger.info(f"Successfully merged node {source_id} into {target_id}.")
        return True
    except Exception as e:
        if "apoc.refactor.mergeNodes" in str(e):
            logger.error(
                "APOC Library not found or configured in Neo4j. "
                "Cannot merge entities. Please install the APOC plugin."
            )
        else:
            logger.error(
                f"Error merging entities ({source_id} -> {target_id}): {e}",
                exc_info=True,
            )
        return False


@alru_cache(maxsize=1)
async def get_defined_node_labels() -> list[str]:
    """Queries the database for all defined node labels and caches the result."""
    try:
        results = await neo4j_manager.execute_read_query("CALL db.labels() YIELD label")
        # Filter out internal labels
        return [
            r["label"]
            for r in results
            if r.get("label") and not r["label"].startswith("_")
        ]
    except Exception:
        logger.error("Failed to query defined node labels from Neo4j.", exc_info=True)
        # Fallback to constants if DB query fails
        return list(
            NODE_LABELS
        )  # Reverted: This should use the imported constant, not settings


@alru_cache(maxsize=1)
async def get_defined_relationship_types() -> list[str]:
    """Queries the database for all defined relationship types and caches the result."""
    try:
        results = await neo4j_manager.execute_read_query(
            "CALL db.relationshipTypes() YIELD relationshipType"
        )
        return [r["relationshipType"] for r in results if r.get("relationshipType")]
    except Exception:
        logger.error(
            "Failed to query defined relationship types from Neo4j.", exc_info=True
        )
        # Fallback to constants if DB query fails
        return list(
            RELATIONSHIP_TYPES
        )  # Reverted: This should use the imported constant, not settings


async def promote_dynamic_relationships() -> int:
    """Convert dynamic relationships to defined relationship types."""
    valid_types = await get_defined_relationship_types()
    query = """
    MATCH (s)-[r:DYNAMIC_REL]->(o)
    WHERE r.type IN $valid_types
    WITH s, r, o
    CALL apoc.create.relationship(
        s,
        r.type,
        r {.*, type: null},
        o
    ) YIELD rel
    DELETE r
    RETURN count(rel) AS promoted
    """
    try:
        results = await neo4j_manager.execute_write_query(
            query, {"valid_types": valid_types}
        )
        return results[0].get("promoted", 0) if results else 0
    except Exception as exc:  # pragma: no cover - narrow DB errors
        logger.error("Failed to promote dynamic relationships: %s", exc, exc_info=True)
        return 0


async def deduplicate_relationships() -> int:
    """Merge duplicate relationships of the same type between nodes."""
    query = """
    MATCH (s)-[r]->(o)
    WITH s, type(r) AS t, o, collect(r) AS rels, count(r) AS cnt
    WHERE cnt > 1
    CALL apoc.refactor.mergeRelationships(rels, {properties: 'combine'}) YIELD rel
    RETURN sum(cnt - 1) AS removed
    """
    try:
        results = await neo4j_manager.execute_write_query(query)
        return results[0].get("removed", 0) if results else 0
    except Exception as exc:  # pragma: no cover - narrow DB errors
        logger.error("Failed to deduplicate relationships: %s", exc, exc_info=True)
        return 0


async def fetch_unresolved_dynamic_relationships(
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Fetch dynamic relationships lacking a specific type."""
    query = """
    MATCH (s:Entity)-[r:DYNAMIC_REL]->(o:Entity)
    WHERE r.type IS NULL OR r.type = 'UNKNOWN'
    RETURN id(r) AS rel_id,
           s.name AS subject,
           labels(s) AS subject_labels,
           coalesce(s.description, '') AS subject_desc,
           o.name AS object,
           labels(o) AS object_labels,
           coalesce(o.description, '') AS object_desc,
           coalesce(r.type, 'UNKNOWN') AS type
    LIMIT $limit
    """
    try:
        results = await neo4j_manager.execute_read_query(query, {"limit": limit})
        return [dict(record) for record in results] if results else []
    except Exception as exc:  # pragma: no cover - narrow DB errors
        logger.error(
            "Failed to fetch unresolved dynamic relationships: %s", exc, exc_info=True
        )
        return []


async def update_dynamic_relationship_type(rel_id: int, new_type: str) -> None:
    """Update a dynamic relationship's type."""
    query = "MATCH ()-[r:DYNAMIC_REL]->() WHERE id(r) = $id SET r.type = $type"
    try:
        await neo4j_manager.execute_write_query(query, {"id": rel_id, "type": new_type})
    except Exception as exc:  # pragma: no cover - narrow DB errors
        logger.error(
            "Failed to update dynamic relationship %s: %s", rel_id, exc, exc_info=True
        )


async def get_shortest_path_length_between_entities(
    name1: str, name2: str, max_depth: int = 4
) -> int | None:
    """Return the shortest path length between two entities if it exists."""
    if max_depth <= 0:
        return None

    query = f"""
    MATCH (a:Entity {{name: $name1}}), (b:Entity {{name: $name2}})
    MATCH p = shortestPath((a)-[*..{max_depth}]-(b))
    RETURN length(p) AS len
    """
    try:
        results = await neo4j_manager.execute_read_query(
            query, {"name1": name1, "name2": name2}
        )
        if results:
            return results[0].get("len")
    except Exception as exc:  # pragma: no cover - narrow DB errors
        logger.error("Failed to compute shortest path length: %s", exc, exc_info=True)
    return None
