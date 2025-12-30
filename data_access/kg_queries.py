# data_access/kg_queries.py
import copy
import hashlib
import re
from typing import Any

import structlog
from async_lru import alru_cache

import config
from core.db_manager import neo4j_manager
from core.exceptions import handle_database_error
from core.schema_validator import schema_validator
from models.kg_constants import (
    KG_IS_PROVISIONAL,
    KG_REL_CHAPTER_ADDED,
    NOVEL_INFO_ALLOWED_PROPERTY_KEYS,
    RELATIONSHIP_TYPES,
    VALID_NODE_LABELS,
)
from utils import classify_category_label

logger = structlog.get_logger(__name__)

# Cache to prevent repeated type upgrade logging for the same entity
_upgrade_logged = set()

# Valid relationship types for narrative knowledge graphs - canonical reference set.
#
# Contract:
# - Relationship type *membership* is permissive (novel types may exist).
# - Relationship types that are interpolated into Cypher MUST pass strict safety checks
#   via validate_relationship_type_for_cypher_interpolation().
#
# This constant is used for reference/analytics/logging and (in some maintenance queries)
# as a whitelist when promoting DYNAMIC_REL -> typed relationships.
VALID_RELATIONSHIP_TYPES = RELATIONSHIP_TYPES

# Lookup table for canonical node labels to ensure consistent casing.
#
# Contract:
# - Domain node labels are strictly limited to [`VALID_NODE_LABELS`](models/kg_constants.py:64).
# - A small allowlist of *internal* labels is supported for infrastructure nodes only.
# - Subtypes (PlotPoint/DevelopmentEvent/etc.) must not be treated as labels; they are represented
#   via properties (e.g., `category`) and normalized by [`SchemaValidationService.validate_entity_type()`](core/schema_validator.py:41).
_CANONICAL_NODE_LABEL_MAP: dict[str, str] = {lbl.lower(): lbl for lbl in VALID_NODE_LABELS}
_CANONICAL_NODE_LABEL_MAP["novelinfo"] = "NovelInfo"
_CANONICAL_NODE_LABEL_MAP["worldcontainer"] = "WorldContainer"
_CANONICAL_NODE_LABEL_MAP["valuenode"] = "ValueNode"


def _infer_from_category_simple(category: str, name: str) -> str:
    """Infer a canonical node label from a category string.

    Args:
        category: A free-form category string used as the primary inference signal.
        name: Entity name used only for logging/debugging.

    Returns:
        A canonical node label from `VALID_NODE_LABELS`. Falls back to `"Item"` when the
        classifier returns an unknown label.

    Notes:
        This helper is not a Cypher interpolation boundary. It exists to normalize model/
        extraction output to application labels before any Cypher label selection occurs.
    """
    result = classify_category_label(category)
    logger.debug(f"Mapped category '{category}' to '{result}' for entity '{name}'")
    return result


def _infer_specific_node_type(name: str, category: str = "", fallback_type: str = "WorldElement") -> str:
    """Select a canonical node label for an entity.

    Args:
        name: Entity name. Must be non-empty to infer from category.
        category: Free-form category signal used for inference when `fallback_type` does not
            specify a canonical label.
        fallback_type: Caller-supplied type hint. When it matches a canonical label in
            `VALID_NODE_LABELS`, it is used as-is. `"WorldElement"` is treated as a sentinel
            meaning "infer from category".

    Returns:
        A canonical node label from `VALID_NODE_LABELS`.

    Notes:
        This function normalizes domain labels to the application schema. It is not itself a
        Cypher interpolation site; downstream Cypher builders must still enforce strict label
        allowlists before interpolating labels.
    """
    if not name or not name.strip():
        # No name signal -> use a safe canonical world label.
        return "Item"

    # Respect explicit canonical label provided by caller.
    if fallback_type in VALID_NODE_LABELS:
        return fallback_type

    # If caller provided a non-sentinel fallback, try to validate/normalize it.
    # (This catches legacy aliases/subtypes via schema_validator normalization.)
    if fallback_type and fallback_type not in {"WorldElement"}:
        is_valid, corrected_label, _ = schema_validator.validate_entity_type(fallback_type)
        if is_valid and corrected_label:
            return corrected_label

    # Default path: infer from category.
    inferred = _infer_from_category_simple(category, name)

    if inferred in VALID_NODE_LABELS:
        return inferred

    # Defensive fallback.
    logger.warning(f"Type inference returned invalid label '{inferred}' for entity '{name}'. " "Falling back to 'Item'.")
    return "Item"


def _to_pascal_case(text: str) -> str:
    """Convert underscore or space separated text to PascalCase."""
    parts = re.split(r"[_\s]+", text.strip())
    return "".join(part[:1].upper() + part[1:] for part in parts if part)


_SAFE_RELATIONSHIP_TYPE_RE = re.compile(r"^[A-Z0-9_]+$")


def validate_relationship_type(proposed_type: str) -> str:
    """Normalize a relationship type for storage and display.

    Args:
        proposed_type: A relationship type string that may be mixed-case and/or contain
            spaces.

    Returns:
        A normalized relationship type string (uppercase, spaces replaced with underscores).

    Notes:
        This function is intentionally lenient. It MUST NOT be used at Cypher relationship
        type interpolation boundaries. Use
        `validate_relationship_type_for_cypher_interpolation()` when a type is placed into
        a query string.
    """
    if not proposed_type or not proposed_type.strip():
        return "RELATES_TO"

    # Clean and normalize: strip whitespace, uppercase, replace spaces with underscores
    return proposed_type.strip().upper().replace(" ", "_")


def validate_relationship_type_for_cypher_interpolation(proposed_type: str) -> str:
    """Validate a relationship type for direct interpolation into Cypher.

    Neo4j does not support parameterizing relationship types. Any relationship type that is
    inserted into a query string (even inside backticks) must be strictly validated to
    prevent Cypher injection.

    Args:
        proposed_type: A relationship type that will be interpolated into the query string.

    Returns:
        The validated relationship type string, unchanged.

    Raises:
        ValueError: If the relationship type is empty, not already uppercase, or contains
            characters outside `[A-Z0-9_]`.

    Notes:
        This function enforces a strict allow-pattern and rejects unsafe inputs instead of
        silently normalizing them. Callers should treat `proposed_type` as untrusted unless
        it is sourced from application-controlled constants.
    """
    raw = str(proposed_type).strip() if proposed_type is not None else ""
    if not raw:
        raise ValueError("Relationship type must be a non-empty string")

    if raw != raw.upper():
        raise ValueError(f"Unsafe relationship type '{raw}': must be uppercase and match ^[A-Z0-9_]+$")

    if not _SAFE_RELATIONSHIP_TYPE_RE.fullmatch(raw):
        raise ValueError(f"Unsafe relationship type '{raw}': must match ^[A-Z0-9_]+$")

    return raw


async def _promote_dynamic_relationships_to_typed_relationships(*, valid_types: list[str]) -> int:
    """
    Promote ``DYNAMIC_REL`` relationships into typed relationships (write operation).

    Determinism/contract notes:
    - This only promotes when ``r.type`` is already a safe, allowlisted relationship type.
    - It preserves all properties except the ``type`` property (which is used only as a marker).
    """
    promotion_query = """
    MATCH (s)-[r:DYNAMIC_REL]->(o)
    WHERE r.type IS NOT NULL
      AND r.type <> 'UNKNOWN'
      AND r.type <> 'DYNAMIC_REL'
      AND r.type IN $valid_types
    WITH s, r, o, r.type as rel_type, properties(r) as rel_props

    CALL apoc.create.relationship(
        s,
        rel_type,
        apoc.map.removeKey(rel_props, 'type'),
        o
    ) YIELD rel

    DELETE r
    RETURN count(rel) AS promoted
    """

    try:
        results = await neo4j_manager.execute_write_query(promotion_query, {"valid_types": valid_types})
        return results[0].get("promoted", 0) if results else 0
    except Exception as exc:  # pragma: no cover - narrow DB errors
        logger.error(
            f"Failed to promote dynamic relationships to typed relationships: {exc}",
            exc_info=True,
        )
        return 0


async def normalize_and_deduplicate_relationships(
    *,
    run_validation: bool = True,
    run_type_consolidation: bool = True,
    run_deduplication: bool = True,
    run_dynamic_promotion: bool = True,
) -> dict[str, int]:
    """
    Canonical KG relationship maintenance pipeline (authoritative entrypoint).

    This is the ONE function callers should use for relationship-type maintenance in Neo4j.
    Other helpers (including ``promote_dynamic_relationships``) delegate to this to avoid
    divergent behavior over time.

    Deterministic contract / step ordering:
    1) Validation / normalization of ``DYNAMIC_REL.type`` strings
       - Fixes format variants (e.g. spaces/lowercase) using ``validate_relationship_type``.
    2) Type consolidation (format normalization of relationship TYPE names in the graph)
       - Renames non-canonical relationship types to the normalized canonical form.
       - NOTE: Despite historical naming, this does NOT do semantic similarity.
    3) Deduplication
       - Removes exact duplicate relationships of the same type between the same nodes.
    4) Optional promotion of ``DYNAMIC_REL`` -> typed relationships
       - Only promotes when ``r.type`` is allowlisted (``VALID_RELATIONSHIP_TYPES``).

    Args:
        run_validation: Whether to validate/correct ``DYNAMIC_REL.type`` strings.
        run_type_consolidation: Whether to consolidate relationship type format variants.
        run_deduplication: Whether to deduplicate identical relationships.
        run_dynamic_promotion: Whether to promote ``DYNAMIC_REL`` into typed relationships.

    Returns:
        Dict with counts (always present, deterministic keys):
            {
              "validated": int,
              "consolidated": int,
              "deduplicated": int,
              "promoted": int,
              "total": int
            }
    """

    counts: dict[str, int] = {
        "validated": 0,
        "consolidated": 0,
        "deduplicated": 0,
        "promoted": 0,
        "total": 0,
    }

    try:
        logger.info(
            "Starting relationship maintenance pipeline",
            run_validation=run_validation,
            run_type_consolidation=run_type_consolidation,
            run_deduplication=run_deduplication,
            run_dynamic_promotion=run_dynamic_promotion,
        )

        if run_validation:
            counts["validated"] = await _validate_and_correct_relationship_types()
            logger.info(f"✓ Validated {counts['validated']} relationship types")

        if run_type_consolidation:
            counts["consolidated"] = await consolidate_similar_relationships()
            logger.info(f"✓ Consolidated {counts['consolidated']} relationships to canonical forms")

        if run_deduplication:
            counts["deduplicated"] = await deduplicate_relationships()
            logger.info(f"✓ Deduplicated {counts['deduplicated']} duplicate relationships")

        if run_dynamic_promotion:
            counts["promoted"] = await _promote_dynamic_relationships_to_typed_relationships(valid_types=list(VALID_RELATIONSHIP_TYPES))
            logger.info(f"✓ Promoted {counts['promoted']} DYNAMIC_REL relationships to typed relationships")

        counts["total"] = counts["validated"] + counts["consolidated"] + counts["deduplicated"] + counts["promoted"]

        logger.info(
            "Relationship maintenance pipeline complete",
            total=counts["total"],
            validated=counts["validated"],
            consolidated=counts["consolidated"],
            deduplicated=counts["deduplicated"],
            promoted=counts["promoted"],
        )

        return counts

    except Exception as exc:
        logger.error(
            f"Relationship maintenance pipeline failed: {exc}",
            exc_info=True,
        )
        return counts


def _get_cypher_labels(entity_type: str | None) -> str:
    """Return a schema-allowlisted Cypher label clause for query interpolation.

    This helper is a Cypher interpolation boundary: the returned value is inserted into
    query strings as a label (for example, `:Character`). Labels are not parameterizable in
    Neo4j, so callers MUST NOT accept arbitrary user input here.

    Args:
        entity_type: A domain label input to normalize. Must be non-empty.

    Returns:
        A string of the form `":<CanonicalLabel>"`.

    Raises:
        ValueError: If `entity_type` is empty/None or cannot be normalized to a label in the
            application allowlist.

    Notes:
        Allowed values include:
        - canonical domain labels from `VALID_NODE_LABELS`
        - a small allowlist of infrastructure labels (for example, `NovelInfo`, `ValueNode`)
        Subtypes should be represented as properties (for example, `category`), not labels.
    """
    if not entity_type or not entity_type.strip():
        raise ValueError("Entity type must be provided")

    cleaned = re.sub(r"[^a-zA-Z0-9_\s]+", "", entity_type)
    normalized_key = re.sub(r"[_\s]+", "", cleaned).lower()

    canonical = _CANONICAL_NODE_LABEL_MAP.get(normalized_key)

    # If not found in map, try PascalCase conversion and check again
    if canonical is None:
        pascal = _to_pascal_case(cleaned)
        canonical = _CANONICAL_NODE_LABEL_MAP.get(pascal.lower())

    # Critical Validation: If still not found or not in allowed set, raise error.
    # We allow only true internal/support labels here (not domain subtypes).
    supporting_types = {
        "NovelInfo",
        "WorldContainer",
        "ValueNode",
    }

    if not canonical:
        # Attempt strict validation via schema_validator as last resort
        is_valid, corrected, _ = schema_validator.validate_entity_type(entity_type)
        if is_valid and corrected:
            canonical = corrected
        else:
            # STRICT: do not allow arbitrary labels at Cypher interpolation sites.
            raise ValueError(f"Invalid node label '{entity_type}'. " f"Must be one of {sorted(VALID_NODE_LABELS)} (or a supported internal label).")

    if canonical not in VALID_NODE_LABELS and canonical not in supporting_types:
        raise ValueError(f"Invalid node label '{canonical}'. " f"Must be one of {sorted(VALID_NODE_LABELS)} (or a supported internal label).")

    # Removed implicit inheritance of Entity label
    return f":{canonical}"


def _get_constraint_safe_merge(
    labels_cypher: str,
    name_param: str,
    create_ts_var: str = "s",
    id_param: str | None = None,
) -> tuple[str, list[str]]:
    """Build a constraint-safe `MERGE` clause for multi-label nodes.

    Args:
        labels_cypher: A colon-delimited label string (for example, `":Character:Item"`).
            This value is expected to be constructed from allowlisted labels upstream.
        name_param: The Cypher parameter name used for the node's `name` match key when
            `id_param` is not provided.
        create_ts_var: The variable name used for the merged node in the returned Cypher.
        id_param: Optional Cypher parameter name used for a stable identity merge key.

    Returns:
        A tuple of:
        - merge_query: A `MERGE (...)` clause that uses a single constraint-sensitive primary
          label to avoid violating Neo4j schema constraints.
        - additional_set_labels: Any remaining labels that should be applied via `SET` after
          the merge.

    Notes:
        This function does not validate labels; it assumes `labels_cypher` was produced from
        schema-allowlisted label sources. Do not pass untrusted labels to this helper.
    """
    # Parse the labels to identify constraint-sensitive ones
    labels = [label.strip() for label in labels_cypher.split(":") if label.strip()]
    # All VALID_NODE_LABELS now have constraints
    constraint_labels = list(VALID_NODE_LABELS) + [
        "NovelInfo",
        "WorldContainer",
        "PlotPoint",
        "ValueNode",
        "DevelopmentEvent",
        "WorldElaborationEvent",
    ]

    # Find the first constraint-sensitive label to use in MERGE
    primary_label = None
    additional_labels: list[str] = []

    # If we have a stable id to merge on, search for a constraint label
    if id_param:
        # Find valid primary label from constraints
        for label in labels:
            if label in constraint_labels:
                primary_label = label
                break

        if primary_label:
            additional_labels = [l for l in labels if l != primary_label]
            merge_query = f"MERGE ({create_ts_var}:{primary_label} {{id: ${id_param}}})"
            return merge_query, additional_labels
        else:
            # Fallback for when no known constraint label matches (should be rare)
            # Use the first label as primary
            primary_label = labels[0] if labels else "Item"
            additional_labels = labels[1:] if len(labels) > 1 else []
            merge_query = f"MERGE ({create_ts_var}:{primary_label} {{id: ${id_param}}})"
            return merge_query, additional_labels

    for label in labels:
        if label in constraint_labels:
            if primary_label is None:
                primary_label = label
            else:
                additional_labels.append(label)
        else:
            additional_labels.append(label)

    # If no constraint-sensitive label found, use first label or a canonical fallback.
    if primary_label is None:
        primary_label = labels[0] if labels else "Item"
        additional_labels = labels[1:] if len(labels) > 1 else []

    # Build the MERGE query with only the primary constraint label
    merge_query = f"MERGE ({create_ts_var}:{primary_label} {{name: ${name_param}}})"

    return merge_query, additional_labels


async def add_kg_triples_batch_to_db(
    structured_triples_data: list[dict[str, Any]],
    chapter_number: int,
    is_from_flawed_draft: bool,
) -> None:
    """Persist extracted KG triples to Neo4j in a single batch.

    Args:
        structured_triples_data: A list of extracted triple dictionaries. Each item must
            contain a subject dict and predicate, plus either an object entity dict or a
            literal object value (represented as a `ValueNode` in the graph).
        chapter_number: Chapter number used for `chapter_added` provenance.
        is_from_flawed_draft: Whether persisted relationships should be marked provisional.

    Returns:
        None.

    Raises:
        Exception: Propagates batch execution failures from the Neo4j driver so callers can
            distinguish operational failures from "no data" cases.

    Notes:
        Security and Cypher interpolation:
        - Relationship types are interpolated (Neo4j cannot parameterize them). This function
          enforces `validate_relationship_type_for_cypher_interpolation()` for every predicate
          before it is placed into Cypher.
        - Node labels are selected via `_get_cypher_labels()` which enforces a schema allowlist.
          Do not widen this allowlist without reviewing Cypher injection implications.

        Cache semantics:
        - This is a write path. On successful completion it triggers KG read cache invalidation
          via [`clear_kg_read_caches()`](data_access/cache_coordinator.py:1).
    """
    if not structured_triples_data:
        logger.info("Neo4j: add_kg_triples_batch_to_db: No structured triples to add.")
        return

    statements_with_params: list[tuple[str, dict[str, Any]]] = []

    for _i, triple_dict in enumerate(structured_triples_data):
        subject_info = triple_dict.get("subject")
        predicate_str = triple_dict.get("predicate")

        object_entity_info = triple_dict.get("object_entity")
        object_literal_val = triple_dict.get("object_literal")  # This will be a string from parsing
        is_literal_object = triple_dict.get("is_literal_object", False)

        if not (subject_info and isinstance(subject_info, dict) and subject_info.get("name") and predicate_str):
            logger.warning(f"Neo4j (Batch): Invalid subject or predicate in triple dict: {triple_dict}")
            continue

        subject_name = str(subject_info["name"]).strip()
        subject_type = subject_info.get("type")  # This is a string like "Character", "WorldElement", etc.
        predicate_clean = validate_relationship_type_for_cypher_interpolation(predicate_str)

        # Strict Type Validation & Inference
        # We must ensure subject_type is one of VALID_NODE_LABELS
        if subject_name:
            subject_category = subject_info.get("category", "")
            # Always run inference/validation to ensure compliance
            validated_type = _infer_specific_node_type(subject_name, subject_category, subject_type)

            if validated_type != subject_type:
                upgrade_key = f"{subject_name}:{subject_type}->{validated_type}"
                if upgrade_key not in _upgrade_logged:
                    logger.info(f"Type validation updated {subject_type} -> {validated_type} for '{subject_name}'")
                    _upgrade_logged.add(upgrade_key)
                subject_type = validated_type

        if not all([subject_name, predicate_clean]):
            logger.warning(f"Neo4j (Batch): Empty subject name or predicate after stripping: {triple_dict}")
            continue

        subject_label = _get_cypher_labels(subject_type).lstrip(":")

        # Base parameters for the relationship
        rel_props = {
            "type": predicate_clean,
            KG_REL_CHAPTER_ADDED: chapter_number,
            KG_IS_PROVISIONAL: is_from_flawed_draft,
            "confidence": 1.0,
        }

        params: dict[str, Any] = {
            "subject_label": subject_label,
            "subject_name_param": subject_name,
            "rel_props_param": rel_props,
            "predicate_clean_param": predicate_clean,
            "chapter_number_param": int(chapter_number),
        }

        subject_id: str | None = None
        if subject_type == "Character":
            try:
                from processing.entity_deduplication import generate_entity_id

                subject_id = generate_entity_id(subject_name, "character", int(chapter_number))
            except Exception:
                subject_id = None
        params["subject_id_param"] = subject_id

        if is_literal_object:
            if object_literal_val is None:
                logger.warning(f"Neo4j (Batch): Literal object is None for triple: {triple_dict}")
                continue

            params["object_literal_value_param"] = str(object_literal_val)
            params["value_node_type_param"] = "Literal"

            rel_id_source = f"{predicate_clean}|{subject_name.strip().lower()}|{str(object_literal_val).strip()}|{chapter_number}"
            rel_id = hashlib.sha1(rel_id_source.encode("utf-8")).hexdigest()[:16]
            params["rel_id_param"] = rel_id

            query = """
            // Handle subject node - first try to find by name, then by ID if name lookup fails
            OPTIONAL MATCH (s:{$subject_label} {name: $subject_name_param})
            WITH s
            WHERE s IS NULL AND $subject_id_param IS NOT NULL AND toString($subject_id_param) <> ''
            CALL apoc.merge.node(
                [$subject_label],
                {id: $subject_id_param},
                {
                    created_ts: timestamp(),
                    updated_ts: timestamp(),
                    type: $subject_label,
                    name: $subject_name_param,
                    id: $subject_id_param
                },
                {updated_ts: timestamp()}
            ) YIELD node AS s_new
            WITH CASE WHEN s IS NOT NULL THEN s ELSE s_new END AS s

            MERGE (o:ValueNode {value: $object_literal_value_param, type: $value_node_type_param})
            ON CREATE SET o.created_ts = timestamp(), o.updated_ts = timestamp()
            ON MATCH SET o.updated_ts = timestamp()

            WITH s, o
            CALL apoc.merge.relationship(
                s,
                $predicate_clean_param,
                {id: $rel_id_param},
                apoc.map.merge($rel_props_param, {created_ts: timestamp(), updated_ts: timestamp()}),
                o,
                apoc.map.merge($rel_props_param, {updated_ts: timestamp()})
            ) YIELD rel
            """
            statements_with_params.append((query, params))

        elif object_entity_info and isinstance(object_entity_info, dict) and object_entity_info.get("name"):
            object_name = str(object_entity_info["name"]).strip()
            object_type = object_entity_info.get("type")
            if not object_name:
                logger.warning(f"Neo4j (Batch): Empty object name for entity object in triple: {triple_dict}")
                continue

            if object_name:
                object_category = object_entity_info.get("category", "")
                validated_object_type = _infer_specific_node_type(object_name, object_category, object_type)

                if validated_object_type != object_type:
                    upgrade_key = f"object:{object_name}:{object_type}->{validated_object_type}"
                    if upgrade_key not in _upgrade_logged:
                        logger.info(f"Type validation updated object {object_type} -> {validated_object_type} for '{object_name}'")
                        _upgrade_logged.add(upgrade_key)
                    object_type = validated_object_type

            object_label = _get_cypher_labels(object_type).lstrip(":")
            params["object_label"] = object_label
            params["object_name_param"] = object_name

            object_id: str | None = None
            if object_type == "Character":
                try:
                    from processing.entity_deduplication import generate_entity_id

                    object_id = generate_entity_id(object_name, "character", int(chapter_number))
                except Exception:
                    object_id = None
            params["object_id_param"] = object_id

            rel_id_source = f"{predicate_clean}|{subject_name.strip().lower()}|{object_name.strip().lower()}|{chapter_number}"
            rel_id = hashlib.sha1(rel_id_source.encode("utf-8")).hexdigest()[:16]
            params["rel_id_param"] = rel_id

            query = """
            // Handle subject node - first try to find by name, then by ID if name lookup fails
            OPTIONAL MATCH (s:{$subject_label} {name: $subject_name_param})
            WITH s
            WHERE s IS NULL AND $subject_id_param IS NOT NULL AND toString($subject_id_param) <> ''
            CALL apoc.merge.node(
                [$subject_label],
                {id: $subject_id_param},
                {
                    created_ts: timestamp(),
                    updated_ts: timestamp(),
                    type: $subject_label,
                    name: $subject_name_param,
                    id: $subject_id_param
                },
                {updated_ts: timestamp()}
            ) YIELD node AS s_new
            WITH CASE WHEN s IS NOT NULL THEN s ELSE s_new END AS s

            // Handle object node - first try to find by name, then by ID if name lookup fails
            OPTIONAL MATCH (o:{$object_label} {name: $object_name_param})
            WITH s, o
            WHERE o IS NULL AND $object_id_param IS NOT NULL AND toString($object_id_param) <> ''
            CALL apoc.merge.node(
                [$object_label],
                {id: $object_id_param},
                {
                    created_ts: timestamp(),
                    updated_ts: timestamp(),
                    type: $object_label,
                    name: $object_name_param,
                    id: $object_id_param
                },
                {updated_ts: timestamp()}
            ) YIELD node AS o_new
            WITH s, o, o_new

            // Use the found node (by name) or the merged node (by ID)
            WITH s, CASE WHEN o IS NOT NULL THEN o ELSE o_new END AS o

            WITH s, o
            CALL apoc.merge.relationship(
                s,
                $predicate_clean_param,
                {id: $rel_id_param},
                apoc.map.merge($rel_props_param, {created_ts: timestamp(), updated_ts: timestamp()}),
                o,
                apoc.map.merge($rel_props_param, {updated_ts: timestamp()})
            ) YIELD rel
            """
            statements_with_params.append((query, params))
        else:
            logger.warning(f"Neo4j (Batch): Invalid or missing object information in triple dict: {triple_dict}")
            continue

    if not statements_with_params:
        logger.info("Neo4j: add_kg_triples_batch_to_db: No valid statements generated from triples.")
        return

    try:
        await neo4j_manager.execute_cypher_batch(statements_with_params)

        logger.info(f"Neo4j: Batch processed {len(statements_with_params)} KG triple statements.")

        # P1.6: Post-write cache invalidation (KG reads are cached via async_lru).
        # Local import avoids circular import / eager import side effects.
        from data_access.cache_coordinator import clear_kg_read_caches

        clear_kg_read_caches()

    except Exception as e:
        # Log first few problematic params for debugging, if any
        first_few_params_str = str([p_tuple[1] for p_tuple in statements_with_params[:2]]) if statements_with_params else "N/A"
        logger.error(
            f"Neo4j: Error in batch adding KG triples. First few params: {first_few_params_str}. Error: {e}",
            exc_info=True,
        )
        raise


@alru_cache(maxsize=256, ttl=300)  # Cache for 5 minutes with 256 entries max
async def _query_kg_from_db_cached(
    subject: str | None = None,
    predicate: str | None = None,
    obj_val: str | None = None,
    chapter_limit: int | None = None,
    include_provisional: bool = False,
    limit_results: int | None = None,
    *,
    allow_unbounded_scan: bool = False,
) -> list[dict[str, Any]]:
    """
    Query KG triples from Neo4j with explicit guardrails against accidental full-graph scans.

    Guardrail contract:
    - By default, this function requires at least one *caller-provided* filter that meaningfully
      scopes the traversal (e.g., `subject`, `predicate`, `obj_val`, or `chapter_limit`).
    - Calling this with no such filters will raise a ValueError, unless
      `allow_unbounded_scan=True` is provided explicitly by the caller.

    Rationale:
    - The underlying Cypher uses a broad pattern `MATCH (s)-[r]->(o)` and an ORDER BY.
      Without filters this can force expensive scans/sorts across the entire graph.
    """
    # Normalize empty-string inputs to "not provided" so callers don't accidentally bypass the guardrail.
    subject_clean = subject.strip() if isinstance(subject, str) else subject
    if subject_clean == "":
        subject_clean = None

    predicate_clean = predicate.strip() if isinstance(predicate, str) else predicate
    if predicate_clean == "":
        predicate_clean = None

    obj_val_clean = obj_val.strip() if isinstance(obj_val, str) else obj_val
    if obj_val_clean == "":
        obj_val_clean = None

    has_scoping_filter = any(x is not None for x in (subject_clean, predicate_clean, obj_val_clean, chapter_limit))
    if not has_scoping_filter and not allow_unbounded_scan:
        raise ValueError(
            "query_kg_from_db() requires at least one filter (subject, predicate, obj_val, "
            "or chapter_limit) to avoid accidental full-graph scans. "
            "Pass allow_unbounded_scan=True to explicitly opt in to an unbounded scan."
        )

    conditions = []
    parameters: dict[str, Any] = {}
    # Removed generic :Entity constraint, now matching on any node
    match_clause = "MATCH (s)-[r]->(o) "

    if subject_clean is not None:
        conditions.append("s.name = $subject_param")
        parameters["subject_param"] = subject_clean
    if predicate_clean is not None:
        normalized_predicate = validate_relationship_type_for_cypher_interpolation(predicate_clean)
        match_clause = f"MATCH (s)-[r:`{normalized_predicate}`]->(o) "
    if obj_val_clean is not None:
        conditions.append(
            """
            ( (o:ValueNode AND o.value = $object_param ) OR
              (NOT o:ValueNode AND o.name = $object_param)
            )
        """
        )
        parameters["object_param"] = obj_val_clean
    if chapter_limit is not None:
        conditions.append(f"coalesce(r.{KG_REL_CHAPTER_ADDED}, -1) <= $chapter_limit_param")
        parameters["chapter_limit_param"] = chapter_limit
    if not include_provisional:
        conditions.append(f"coalesce(r.{KG_IS_PROVISIONAL}, FALSE) = FALSE")

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

    return_clause = f"""
    RETURN s.name AS subject,
           type(r) AS predicate,
           CASE WHEN o:ValueNode THEN o.value ELSE o.name END AS object,
           CASE WHEN o:ValueNode THEN 'Literal' ELSE labels(o)[0] END AS object_type,
           coalesce(r.{KG_REL_CHAPTER_ADDED}, -1) AS {KG_REL_CHAPTER_ADDED},
           coalesce(r.confidence, 0.0) AS confidence,
           coalesce(r.{KG_IS_PROVISIONAL}, FALSE) AS {KG_IS_PROVISIONAL}
    """
    order_clause = f" ORDER BY coalesce(r.{KG_REL_CHAPTER_ADDED}, -1) DESC, coalesce(r.confidence, 0.0) DESC"
    limit_clause_str = f" LIMIT {int(limit_results)}" if limit_results is not None and limit_results > 0 else ""

    full_query = match_clause + where_clause + return_clause + order_clause + limit_clause_str
    try:
        results = await neo4j_manager.execute_read_query(full_query, parameters)
        triples_list: list[dict[str, Any]] = [dict(record) for record in results] if results else []
        logger.debug(f"Neo4j: KG query returned {len(triples_list)} results. Query: '{full_query[:200]}...' Params: {parameters}")
        return triples_list
    except Exception as e:
        # P1.9: Standardize error handling.
        # Return empty only for legitimate "no rows" cases; DB/runtime errors should be raised
        # as standardized core exceptions so callers can distinguish "no data" from "DB down".
        logger.error(
            f"Neo4j: Error querying KG. Query: '{full_query[:200]}...', Params: {parameters}, Error: {e}",
            exc_info=True,
        )
        raise handle_database_error(
            "query_kg_from_db",
            e,
            query_preview=full_query[:200],
            params=parameters,
        ) from e


async def query_kg_from_db(
    subject: str | None = None,
    predicate: str | None = None,
    obj_val: str | None = None,
    chapter_limit: int | None = None,
    include_provisional: bool = False,
    limit_results: int | None = None,
    *,
    allow_unbounded_scan: bool = False,
) -> list[dict[str, Any]]:
    """Return KG triples.

    This is a thin wrapper around the cached implementation that ensures callers always
    receive a *fresh* structure they can safely mutate without contaminating future cache hits.

    Rationale:
    - async_lru caches and returns the *same object instance* on repeated calls.
    - This function returns a deep copy of the cached value to prevent mutation leakage.
    """
    cached = await _query_kg_from_db_cached(
        subject=subject,
        predicate=predicate,
        obj_val=obj_val,
        chapter_limit=chapter_limit,
        include_provisional=include_provisional,
        limit_results=limit_results,
        allow_unbounded_scan=allow_unbounded_scan,
    )
    return copy.deepcopy(cached)


# Preserve legacy cache management hooks for tests / tooling.
# (e.g., callers historically did `query_kg_from_db.cache_clear()`.)
query_kg_from_db.cache_clear = _query_kg_from_db_cached.cache_clear  # type: ignore[attr-defined]
query_kg_from_db.cache_info = _query_kg_from_db_cached.cache_info  # type: ignore[attr-defined]


async def get_most_recent_value_from_db(
    subject: str,
    predicate: str,
    chapter_limit: int | None = None,
    include_provisional: bool = False,
) -> Any | None:
    """Return the most recent object value for a `(subject, predicate)` pair.

    Args:
        subject: Subject entity name. Must be non-empty.
        predicate: Relationship type to match. This value is interpolated into Cypher (inside
            backticks) after strict validation.
        chapter_limit: Optional upper bound on `chapter_added` used to scope "most recent".
        include_provisional: Whether to consider provisional relationships when selecting the
            most recent value.

    Returns:
        The object value for the most recent matching relationship. The value may be a string
        or a best-effort conversion to `int`/`float`/`bool` when the stored value is a string.
        Returns None when inputs are invalid or when no matching value exists.

    Notes:
        Security:
            Neo4j does not parameterize relationship types. This function uses
            `validate_relationship_type_for_cypher_interpolation()` and rejects unsafe
            predicates. Do not pass raw user input as `predicate`.
    """
    if not subject.strip() or not predicate.strip():
        logger.warning(f"Neo4j: get_most_recent_value_from_db: empty subject or predicate. S='{subject}', P='{predicate}'")
        return None

    conditions = []
    parameters: dict[str, Any] = {}
    normalized_predicate = validate_relationship_type_for_cypher_interpolation(predicate)
    match_clause = f"MATCH (s)-[r:`{normalized_predicate}`]->(o) "

    conditions.append("s.name = $subject_param")
    parameters["subject_param"] = subject.strip()

    if chapter_limit is not None:
        conditions.append(f"coalesce(r.{KG_REL_CHAPTER_ADDED}, -1) <= $chapter_limit_param")
        parameters["chapter_limit_param"] = chapter_limit
    if not include_provisional:
        conditions.append(f"(r.{KG_IS_PROVISIONAL} = FALSE OR r.{KG_IS_PROVISIONAL} IS NULL)")

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

    return_clause = f"""
    RETURN
           CASE WHEN o:ValueNode THEN o.value ELSE o.name END AS object,
           coalesce(r.{KG_REL_CHAPTER_ADDED}, -1) AS {KG_REL_CHAPTER_ADDED},
           coalesce(r.confidence, 0.0) AS confidence,
           coalesce(r.{KG_IS_PROVISIONAL}, FALSE) AS {KG_IS_PROVISIONAL}
    """
    order_clause = f" ORDER BY coalesce(r.{KG_REL_CHAPTER_ADDED}, -1) DESC, coalesce(r.confidence, 0.0) DESC"
    limit_clause_str = " LIMIT 1"

    full_query = match_clause + where_clause + return_clause + order_clause + limit_clause_str
    try:
        results = await neo4j_manager.execute_read_query(full_query, parameters)
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
    except Exception as e:
        logger.error(
            f"Neo4j: Error querying KG. Query: '{full_query[:200]}...', Params: {parameters}, Error: {e}",
            exc_info=True,
        )
    logger.debug(f"Neo4j: No value found for ({subject}, {predicate}) up to Ch {chapter_limit}, include_provisional={include_provisional}.")
    return None


@alru_cache(maxsize=64, ttl=600)  # Cache novel info properties for 10 minutes
async def _get_novel_info_property_from_db_cached(property_key: str) -> Any | None:
    """Return a property value from the NovelInfo node (cached).

    Args:
        property_key: A NovelInfo property key. This key is interpolated into the query
            string and therefore MUST be allowlisted.

    Returns:
        The property value when present. Returns None when `property_key` is empty, when the
        NovelInfo node is missing, or when the property is unset.

    Raises:
        ValueError: If `property_key` is not in `NOVEL_INFO_ALLOWED_PROPERTY_KEYS`.

    Notes:
        Security:
            Neo4j does not parameterize property access in a `RETURN ni.<property>` clause.
            This function enforces a strict allowlist and must not accept arbitrary keys.
    """
    key = property_key.strip() if property_key is not None else ""
    if not key:
        logger.warning("Neo4j: empty property key for NovelInfo query")
        return None

    if key not in NOVEL_INFO_ALLOWED_PROPERTY_KEYS:
        raise ValueError("Unsafe NovelInfo property key " f"'{key}'. Allowed keys: {sorted(NOVEL_INFO_ALLOWED_PROPERTY_KEYS)}")

    novel_id_param = config.MAIN_NOVEL_INFO_NODE_ID
    query = "MATCH (ni:NovelInfo {id: $novel_id_param}) " f"RETURN ni.{key} AS value"
    try:
        results = await neo4j_manager.execute_read_query(query, {"novel_id_param": novel_id_param})
        if results and results[0] and "value" in results[0]:
            return results[0]["value"]
    except Exception as e:  # pragma: no cover - narrow DB errors
        logger.error(
            f"Neo4j: Error retrieving NovelInfo property '{property_key}': {e}",
            exc_info=True,
        )
    return None


async def get_novel_info_property_from_db(property_key: str) -> Any | None:
    """Return a NovelInfo property value.

    Args:
        property_key: A NovelInfo property key. This value is validated against a strict
            allowlist because it is interpolated into Cypher.

    Returns:
        The NovelInfo property value. Returns a defensive deep copy so callers can safely
        mutate the returned structure without contaminating future cache hits.

    Raises:
        ValueError: If `property_key` is not in `NOVEL_INFO_ALLOWED_PROPERTY_KEYS`.

    Notes:
        Cache semantics:
            This is a read-through cache via `async_lru`. The wrapper returns a deep copy
            because `async_lru` returns the same object instance on repeated calls.

        Security:
            Do not pass untrusted input as `property_key`. The function rejects keys that
            are not explicitly allowlisted.
    """
    cached = await _get_novel_info_property_from_db_cached(property_key)
    return copy.deepcopy(cached)


# Preserve legacy cache management hooks for tests / tooling.
get_novel_info_property_from_db.cache_clear = (  # type: ignore[attr-defined]
    _get_novel_info_property_from_db_cached.cache_clear
)
get_novel_info_property_from_db.cache_info = (  # type: ignore[attr-defined]
    _get_novel_info_property_from_db_cached.cache_info
)


async def get_chapter_context_for_entity(
    entity_name: str | None = None,
    entity_id: str | None = None,
    *,
    chapter_context_limit: int = 5,
    max_event_chapters: int = 50,
    max_rel_chapters: int = 200,
) -> list[dict[str, Any]]:
    """Return recent chapter context for a single entity.

    Args:
        entity_name: Entity name lookup key. Exactly one of `entity_name` or `entity_id`
            must be provided.
        entity_id: Entity id lookup key. Exactly one of `entity_name` or `entity_id`
            must be provided.
        chapter_context_limit: Maximum number of chapters to return. Hard-capped to keep the
            query bounded and deterministic.
        max_event_chapters: Maximum number of event-derived chapter numbers to consider
            before truncation.
        max_rel_chapters: Maximum number of relationship-derived chapter numbers to consider
            before truncation.

    Returns:
        A list of dictionaries with keys:
        - `chapter_number`
        - `summary`
        - `text`

        Returns an empty list when the entity is not provided or no context exists.

    Raises:
        ValueError: If limits are non-positive or if `chapter_context_limit` exceeds the hard
            cap.

    Notes:
        Query guardrails:
            This query avoids OPTIONAL MATCH cartesian row explosion by collecting candidate
            chapter numbers in isolated subqueries and combining the results. The output is
            sorted by chapter descending and limited to `chapter_context_limit`.

        Error behavior:
            This function is best-effort and returns an empty list on Neo4j/query failures
            (it logs the exception). Callers should treat an empty list as "no context or
            not available" rather than a definitive absence.
    """
    if not entity_name and not entity_id:
        return []

    if chapter_context_limit <= 0:
        raise ValueError("chapter_context_limit must be a positive integer")
    if chapter_context_limit > 20:
        raise ValueError("chapter_context_limit is capped at 20 to keep context retrieval bounded")

    if max_event_chapters <= 0 or max_rel_chapters <= 0:
        raise ValueError("max_event_chapters and max_rel_chapters must be positive integers")

    match_clause = "MATCH (e {id: $id_param})" if entity_id else "MATCH (e {name: $name_param})"
    params: dict[str, Any] = {"id_param": entity_id} if entity_id else {"name_param": entity_name}
    params.update(
        {
            "chapter_context_limit": int(chapter_context_limit),
            "max_event_chapters": int(max_event_chapters),
            "max_rel_chapters": int(max_rel_chapters),
        }
    )

    query = f"""
    {match_clause}

    // Collect chapter numbers from independent sources without row explosion.
    WITH e

    CALL (e) {{
      WITH e
      OPTIONAL MATCH (e)-[]->(event)
      WHERE (event:DevelopmentEvent OR event:WorldElaborationEvent)
        AND event.chapter_updated IS NOT NULL
      WITH collect(DISTINCT event.chapter_updated) AS chapters
      RETURN chapters[..$max_event_chapters] AS event_chapters
    }}

    CALL (e) {{
      WITH e
      OPTIONAL MATCH (e)-[r]->()
      WHERE r.chapter_added IS NOT NULL
      WITH collect(DISTINCT r.chapter_added) AS chapters
      RETURN chapters[..$max_rel_chapters] AS rel_chapters
    }}

    WITH
      CASE WHEN e.created_chapter IS NOT NULL THEN [e.created_chapter] ELSE [] END AS created_chapter_list,
      event_chapters,
      rel_chapters

    WITH created_chapter_list + event_chapters + rel_chapters AS all_chapters
    UNWIND all_chapters AS chapter_num
    WITH DISTINCT chapter_num
    WHERE chapter_num IS NOT NULL AND chapter_num > 0
    ORDER BY chapter_num DESC
    LIMIT $chapter_context_limit

    MATCH (c:Chapter {{number: chapter_num}})
    RETURN c.number AS chapter_number, c.summary AS summary, c.text AS text
    ORDER BY c.number DESC
    """
    try:
        query_hash = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
        logger.debug(
            "Executing get_chapter_context_for_entity query",
            query_hash=query_hash,
            query_preview=query.strip()[:250],
            params=params,
        )
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
    """Find characters that have both traits in any of the provided contradictory pairs.

    Args:
        contradictory_trait_pairs: A list of `(trait_a, trait_b)` pairs.

    Returns:
        A list of dictionaries containing:
        - `character_name`
        - `trait1`
        - `trait2`

        Returns an empty list when no pairs are provided or no matches exist.

    Notes:
        This is a diagnostic query intended for validation/QA flows rather than core
        runtime reads.
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
    """Find characters with relationship activity after an `IS_DEAD` chapter.

    Returns:
        A list of dictionaries including:
        - `character_name`
        - `death_chapter`
        - `post_mortem_activities` (a list of `{activity_type, activity_chapter}`)

    Notes:
        This is a diagnostic query. It filters out retrospective relationships (for example,
        remembrance links) so the result focuses on potentially inconsistent timeline facts.
    """
    query = """
    MATCH (c:Character)-[death_rel:`IS_DEAD`]->()
    WHERE death_rel.is_provisional = false OR death_rel.is_provisional IS NULL
    WITH c, death_rel.chapter_added AS death_chapter

    MATCH (c)-[activity_rel]->()
    WHERE activity_rel.chapter_added > death_chapter
      AND NOT type(activity_rel) IN ['IS_REMEMBERED_AS', 'WAS_FRIEND_OF'] // Exclude retrospective rels
    RETURN DISTINCT c.name as character_name,
           death_chapter,
           collect(
             {
               activity_type: type(activity_rel),
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
    similarity_threshold: float = 0.45,
    limit: int = 50,
    *,
    desc_threshold: float = 0.30,
    per_label_limit: int | None = None,
    candidate_pool_size: int | None = None,
    max_candidate_pool_size: int = 500,
    allow_large_candidate_pool: bool = False,
) -> list[dict[str, Any]]:
    """
    Find candidate duplicate entity pairs with bounded candidate selection.

    Guardrail contract:
    - This query can be quadratic in the number of candidate nodes.
      To prevent worst-case blowups, we first select a bounded candidate pool per label
      (Characters; and a set of non-Character entity labels), then only compare pairs
      within that pool.
    - `max_candidate_pool_size` is a hard cap (unless `allow_large_candidate_pool=True`).

    API compatibility notes:
    - `similarity_threshold` and `limit` are preserved.
    - Results are "best effort" when the graph is very large: only the bounded pool is compared.
    """
    if limit <= 0:
        raise ValueError("limit must be a positive integer")

    # Use a per-label limit when provided; default to overall limit otherwise
    label_limit = per_label_limit or limit
    if label_limit <= 0:
        raise ValueError("per_label_limit must be a positive integer when provided")

    if max_candidate_pool_size <= 0:
        raise ValueError("max_candidate_pool_size must be a positive integer")

    # Choose a conservative default pool size based on requested output size.
    # Keep it bounded to prevent O(N^2) blowups.
    if candidate_pool_size is None:
        candidate_pool_size = min(max(label_limit * 10, 200), max_candidate_pool_size)

    candidate_pool_size = int(candidate_pool_size)
    if candidate_pool_size < label_limit:
        candidate_pool_size = label_limit

    if candidate_pool_size > max_candidate_pool_size and not allow_large_candidate_pool:
        raise ValueError(
            f"candidate_pool_size={candidate_pool_size} exceeds max_candidate_pool_size={max_candidate_pool_size}. "
            "Pass allow_large_candidate_pool=True to explicitly opt in to larger (potentially expensive) comparisons."
        )

    # The query runs two bounded label-specific searches and unions results.
    # Description similarity is a simple overlap on tokenized, cleaned text.
    query = """
    // ---------- Character pairs (bounded candidate pool) ----------
    CALL () {
      MATCH (e:Character)
      WHERE e.name IS NOT NULL AND e.id IS NOT NULL
      RETURN e
      ORDER BY toLower(toString(e.name)), toString(e.id)
      LIMIT $candidate_pool_size
    }
    WITH collect(e) AS candidates
    UNWIND candidates AS e1
    UNWIND candidates AS e2
    WITH e1, e2
    WHERE elementId(e1) < elementId(e2)
    WITH e1, e2,
         toLower(toString(e1.name)) AS n1,
         toLower(toString(e2.name)) AS n2,
         coalesce(toLower(toString(e1.description)), '') AS d1,
         coalesce(toLower(toString(e2.description)), '') AS d2,
         e1.category AS cat1,
         e2.category AS cat2
    WITH e1, e2, n1, n2, d1, d2,
         apoc.text.levenshteinSimilarity(n1, n2) AS name_sim,
         abs(size(n1) - size(n2)) AS len_diff,
         substring(n1, 0, 1) AS c1,
         substring(n2, 0, 1) AS c2,
         [w IN split(apoc.text.replace(n1, '[^a-z0-9 ]', ''), ' ') WHERE size(w) > 2] AS n1_tokens,
         [w IN split(apoc.text.replace(n2, '[^a-z0-9 ]', ''), ' ') WHERE size(w) > 2] AS n2_tokens,
         [w IN split(apoc.text.replace(d1, '[^a-z0-9 ]', ''), ' ') WHERE size(w) > 2] AS d1_tokens,
         [w IN split(apoc.text.replace(d2, '[^a-z0-9 ]', ''), ' ') WHERE size(w) > 2] AS d2_tokens,
         cat1, cat2
    WITH e1, e2, name_sim, len_diff,
         size([x IN n1_tokens WHERE x IN n2_tokens]) AS name_overlap,
         size([x IN d1_tokens WHERE x IN d2_tokens]) AS desc_overlap,
         CASE WHEN size(d1_tokens) >= size(d2_tokens) THEN size(d1_tokens) ELSE size(d2_tokens) END AS d_denom,
         c1, c2,
         cat1, cat2
    WITH e1, e2, name_sim, len_diff, name_overlap,
         CASE WHEN d_denom = 0 THEN 0.0 ELSE toFloat(desc_overlap) / toFloat(d_denom) END AS desc_sim,
         c1, c2,
         cat1, cat2
    WHERE len_diff <= 10
      AND c1 = c2
      AND name_overlap >= 1
      AND (cat1 IS NULL OR cat2 IS NULL OR cat1 = cat2)
      AND (
            name_sim >= $name_threshold
         OR (name_sim >= ($name_threshold * 0.8) AND desc_sim >= $desc_threshold)
      )
    RETURN e1.id AS id1, e1.name AS name1, labels(e1) AS labels1,
           e2.id AS id2, e2.name AS name2, labels(e2) AS labels2,
           name_sim AS similarity
    ORDER BY similarity DESC
    LIMIT $label_limit

    UNION

    // ---------- Typed Entity pairs (non-Character; bounded candidate pool) ----------
    CALL () {
      MATCH (e)
      WHERE (e:Location OR e:Item OR e:Event)
        AND e.name IS NOT NULL AND e.id IS NOT NULL
      RETURN e
      ORDER BY toLower(toString(e.name)), toString(e.id)
      LIMIT $candidate_pool_size
    }
    WITH collect(e) AS candidates
    UNWIND candidates AS e1
    UNWIND candidates AS e2
    WITH e1, e2
    WHERE elementId(e1) < elementId(e2)
    WITH e1, e2,
         toLower(toString(e1.name)) AS n1,
         toLower(toString(e2.name)) AS n2,
         coalesce(toLower(toString(e1.description)), '') AS d1,
         coalesce(toLower(toString(e2.description)), '') AS d2,
         e1.category AS cat1,
         e2.category AS cat2
    WITH e1, e2, n1, n2, d1, d2,
         apoc.text.levenshteinSimilarity(n1, n2) AS name_sim,
         abs(size(n1) - size(n2)) AS len_diff,
         substring(n1, 0, 1) AS c1,
         substring(n2, 0, 1) AS c2,
         [w IN split(apoc.text.replace(n1, '[^a-z0-9 ]', ''), ' ') WHERE size(w) > 2] AS n1_tokens,
         [w IN split(apoc.text.replace(n2, '[^a-z0-9 ]', ''), ' ') WHERE size(w) > 2] AS n2_tokens,
         [w IN split(apoc.text.replace(d1, '[^a-z0-9 ]', ''), ' ') WHERE size(w) > 2] AS d1_tokens,
         [w IN split(apoc.text.replace(d2, '[^a-z0-9 ]', ''), ' ') WHERE size(w) > 2] AS d2_tokens,
         cat1, cat2
    WITH e1, e2, name_sim, len_diff,
         size([x IN n1_tokens WHERE x IN n2_tokens]) AS name_overlap,
         size([x IN d1_tokens WHERE x IN d2_tokens]) AS desc_overlap,
         CASE WHEN size(d1_tokens) >= size(d2_tokens) THEN size(d1_tokens) ELSE size(d2_tokens) END AS d_denom,
         c1, c2,
         cat1, cat2
    WITH e1, e2, name_sim, len_diff, name_overlap,
         CASE WHEN d_denom = 0 THEN 0.0 ELSE toFloat(desc_overlap) / toFloat(d_denom) END AS desc_sim,
         c1, c2,
         cat1, cat2
    WHERE len_diff <= 10
      AND c1 = c2
      AND name_overlap >= 1
      AND (cat1 IS NULL OR cat2 IS NULL OR cat1 = cat2)
      AND (
            name_sim >= $name_threshold
         OR (name_sim >= ($name_threshold * 0.8) AND desc_sim >= $desc_threshold)
      )
    RETURN e1.id AS id1, e1.name AS name1, labels(e1) AS labels1,
           e2.id AS id2, e2.name AS name2, labels(e2) AS labels2,
           name_sim AS similarity
    ORDER BY similarity DESC
    LIMIT $label_limit
    """

    params = {
        "name_threshold": similarity_threshold,
        "desc_threshold": desc_threshold,
        "label_limit": label_limit,
        "candidate_pool_size": candidate_pool_size,
    }
    try:
        query_hash = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
        logger.debug(
            "Executing find_candidate_duplicate_entities query",
            query_hash=query_hash,
            query_preview=query.strip()[:250],
            params=params,
        )
        results = await neo4j_manager.execute_read_query(query, params)
        return results if results else []
    except Exception as e:
        logger.error(f"Error finding candidate duplicate entities: {e}", exc_info=True)
        return []


async def get_entity_context_for_resolution(
    entity_id: str,
) -> dict[str, Any] | None:
    """
    Gathers comprehensive context for an entity to help an LLM decide on a merge.
    """
    query = """
    MATCH (e)
    WHERE e.id = $entity_id OR e.name = $entity_id
    OPTIONAL MATCH (e)-[r]-(o)
    WHERE o.id IS NOT NULL
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
        if results:
            return results[0]
        else:
            # Debug: Check if entity exists with any labels
            debug_query = "MATCH (e) WHERE e.id = $entity_id OR e.name = $entity_id " "RETURN e.name AS name, labels(e) AS labels"
            debug_results = await neo4j_manager.execute_read_query(debug_query, params)
            if debug_results:
                logger.debug(f"Entity {entity_id} exists with name '{debug_results[0]['name']}' " f"and labels {debug_results[0]['labels']} but returned no context")
            else:
                logger.debug(f"Entity {entity_id} does not exist in database")
            return None
    except Exception as e:
        logger.error(
            f"Error getting context for entity resolution (id: {entity_id}): {e}",
            exc_info=True,
        )
        return None


async def merge_entities(source_id: str, target_id: str, reason: str, max_retries: int = 3) -> bool:
    """
    Merges one entity (source) into another (target) using atomic Neo4j operations with retry logic.
    The source node will be deleted after its relationships are moved.
    """
    import asyncio

    for attempt in range(max_retries):
        try:
            logger.info(f"Merge attempt {attempt + 1}/{max_retries} for {source_id} -> {target_id}")
            return await _execute_atomic_merge(source_id, target_id, reason)
        except Exception as e:
            logger.error(f"Merge attempt {attempt + 1}/{max_retries} failed: {e}", exc_info=True)
            error_msg = str(e).lower()
            if ("entitynotfound" in error_msg or "transaction" in error_msg or "locked" in error_msg or "deadlock" in error_msg) and attempt < max_retries - 1:
                logger.warning(f"Entity merge attempt {attempt + 1}/{max_retries} failed, retrying: {e}")
                await asyncio.sleep(0.1 * (2**attempt))  # Exponential backoff
                continue
            else:
                logger.error(
                    f"Entity merge failed after {attempt + 1} attempts: {e}",
                    exc_info=True,
                )
                return False

    return False


async def _execute_atomic_merge(source_id: str, target_id: str, reason: str) -> bool:
    merge_query = """
    MATCH (source {id: $source_id}), (target {id: $target_id})
    WITH
        target,
        source,
        coalesce(toString(source.description), '') AS source_description
    CALL apoc.refactor.mergeNodes(
        [target, source],
        {
            properties: 'discard',
            mergeRels: true,
            produceSelfRel: false,
            preserveExistingSelfRels: true,
            countMerge: true
        }
    ) YIELD node

    SET node.merge_reason = $reason,
        node.merge_timestamp = timestamp(),
        node.last_updated = timestamp(),
        node.description =
            CASE
                WHEN source_description = '' THEN node.description
                WHEN node.description IS NULL OR toString(node.description) = '' THEN source_description
                WHEN toString(node.description) CONTAINS source_description THEN node.description
                ELSE toString(node.description) + ', ' + source_description
            END

    RETURN node.id AS id
    """

    results = await neo4j_manager.execute_write_query(
        merge_query,
        {
            "source_id": source_id,
            "target_id": target_id,
            "reason": reason,
        },
    )
    return bool(results and results[0] and results[0].get("id"))


async def promote_dynamic_relationships() -> int:
    """
    Deprecated entrypoint: use ``normalize_and_deduplicate_relationships`` instead
    This function historically performed:
    - validation/correction of ``DYNAMIC_REL.type`` strings, then
    - promotion of ``DYNAMIC_REL`` -> typed relationships.

    To prevent drift, it is now a thin wrapper around the canonical pipeline.
    Return value is preserved for backwards compatibility: the number of relationships
    processed for validation + promotion (not including consolidation/deduplication).
    """
    counts = await normalize_and_deduplicate_relationships(
        run_validation=True,
        run_type_consolidation=False,
        run_deduplication=False,
        run_dynamic_promotion=True,
    )
    return int(counts.get("validated", 0) + counts.get("promoted", 0))


async def _validate_and_correct_relationship_types() -> int:
    """Validate and correct existing relationship types."""
    # Find all DYNAMIC_REL relationships with type properties
    validation_query = """
    MATCH (s)-[r:DYNAMIC_REL]->(o)
    WHERE r.type IS NOT NULL
      AND r.type <> 'UNKNOWN'
      AND r.type <> 'DYNAMIC_REL'
    RETURN elementId(r) as rel_id, r.type as current_type
    """

    try:
        results = await neo4j_manager.execute_read_query(validation_query)
        if not results:
            return 0

        corrected_count = 0

        for record in results:
            current_type = record["current_type"]
            validated_type = validate_relationship_type(current_type)

            if validated_type != current_type:
                # Update to validated type
                update_query = """
                MATCH ()-[r:DYNAMIC_REL]->()
                WHERE elementId(r) = $rel_id
                SET r.type = $new_type
                RETURN count(*) as updated
                """

                await neo4j_manager.execute_write_query(
                    update_query,
                    {"rel_id": record["rel_id"], "new_type": validated_type},
                )
                corrected_count += 1
                logger.debug(f"Corrected relationship type: '{current_type}' -> '{validated_type}'")

        return corrected_count

    except Exception as exc:
        logger.error(f"Failed to validate relationship types: {exc}", exc_info=True)
        return 0


async def deduplicate_relationships() -> int:
    query = """
    MATCH (s)-[r]->(o)
    WITH s, type(r) AS rel_type, o, collect(r) AS rels
    WHERE size(rels) > 1
    CALL apoc.refactor.mergeRelationships(rels, {properties: 'combine'}) YIELD rel
    RETURN count(rel) AS deduplicated
    """

    try:
        results = await neo4j_manager.execute_write_query(query)
        return results[0].get("deduplicated", 0) if results else 0
    except Exception as exc:  # pragma: no cover - narrow DB errors
        logger.error(f"Failed to deduplicate relationships: {exc}", exc_info=True)
        return 0


async def consolidate_similar_relationships() -> int:
    """
    Consolidate relationship type *format* variants into a canonical type name.

    Important: despite the historical name, this function does **not** perform semantic
    similarity consolidation. It only normalizes relationship TYPE names via
    ``validate_relationship_type`` (uppercase + underscores) and rewrites the graph to
    use the normalized type.

    Determinism:
    - We iterate types in a stable order (count desc, then rel_type asc) to make results
      reproducible across runs when counts tie.
    """
    # Get all relationship types currently in the database (stable ordering)
    query_current = """
    MATCH ()-[r]->()
    RETURN DISTINCT type(r) AS rel_type, count(r) AS count
    ORDER BY count DESC, rel_type ASC
    """

    try:
        current_results = await neo4j_manager.execute_read_query(query_current)
        current_types = [r["rel_type"] for r in current_results if r.get("rel_type")]

        consolidation_count = 0

        # Process each current relationship type
        for current_type in current_types:
            # Skip if already canonical
            if current_type in VALID_RELATIONSHIP_TYPES:
                continue

            # Find canonical version
            canonical_type = validate_relationship_type(current_type)

            # Skip if no change needed
            if current_type == canonical_type:
                continue

            consolidate_query = """
            CALL apoc.periodic.iterate(
                "MATCH (s)-[r]-(o) WHERE type(r) = $current_type RETURN s, r, o",
                "WITH s, r, o CALL apoc.create.relationship(s, $canonical_type, properties(r), o) YIELD rel DELETE r RETURN count(*) AS changed",
                {batchSize: 1000, parallel: false, params: {current_type: $current_type, canonical_type: $canonical_type}}
            ) YIELD total
            RETURN total AS consolidated
            """

            try:
                consolidate_results = await neo4j_manager.execute_write_query(
                    consolidate_query,
                    {"current_type": current_type, "canonical_type": canonical_type},
                )
                count = consolidate_results[0].get("consolidated", 0) if consolidate_results else 0
                consolidation_count += count

                if count > 0:
                    logger.info(f"Consolidated {count} relationships: {current_type} -> {canonical_type}")

            except Exception as exc:
                logger.warning(f"Failed to consolidate {current_type} -> {canonical_type}: {exc}")

        return consolidation_count

    except Exception as exc:
        logger.error("Failed to consolidate similar relationships: %s", exc, exc_info=True)
        return 0


async def get_shortest_path_length_between_entities(name1: str, name2: str, max_depth: int = 4) -> int | None:
    """Return the shortest path length between two entities if it exists."""
    if max_depth <= 0:
        return None

    query = f"""
    MATCH (a {{name: $name1}}), (b {{name: $name2}})
    MATCH p = shortestPath((a)-[*..{max_depth}]-(b))
    RETURN length(p) AS len
    """
    try:
        results = await neo4j_manager.execute_read_query(query, {"name1": name1, "name2": name2})
        if results:
            return results[0].get("len")
    except Exception as exc:
        logger.error(f"Failed to compute shortest path length: {exc}", exc_info=True)
    return None
