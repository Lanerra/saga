# data_access/cypher_builders/native_builders.py
"""Build Cypher statements directly from Pydantic models.

Notes:
    Security and Cypher identifier selection:
        - Neo4j labels are not parameterized. This module interpolates a small, allowlisted
          set of labels (for example, the primary world-item label) into query strings.
          Do not widen label selection without reviewing Cypher injection risk.
        - Relationship types for writes are supplied to APOC procedures (for example,
          `apoc.merge.relationship`). Although this is not string interpolation into raw
          Cypher, it still allows callers to create arbitrary relationship type names.
          Callers must treat relationship type inputs as untrusted and enforce a strict
          allowlist/pattern (for example, `[A-Z0-9_]+`) upstream to prevent schema drift.
"""

from typing import TYPE_CHECKING, Any

from models.kg_constants import WORLD_ITEM_CANONICAL_LABELS
from utils import classify_category_label
from utils.common import flatten_dict

if TYPE_CHECKING:
    from models.kg_models import CharacterProfile, WorldItem


def canonical_entity_cypher(variable: str, label: str, name: str, identifier: str, chapter: str, *, scope: str = "") -> str:
    """Resolve an allowlisted endpoint by supplied ID, otherwise unambiguous normalized name.

    Arguments are internal Cypher expressions, never interpolated user values.
    The owned database manager remains the project boundary.
    """
    return f"""
    CALL ({scope}) {{
        WITH {label} AS entity_label, {name} AS entity_name, {identifier} AS supplied_id
        CALL apoc.util.validate(
            NOT entity_label IN ['Character', 'Location', 'Item', 'Event']
            OR entity_name IS NULL OR trim(entity_name) = ''
            OR (supplied_id IS NOT NULL AND trim(supplied_id) = ''),
            'Invalid canonical entity', [])
        WITH entity_label, entity_name, supplied_id
        OPTIONAL MATCH (candidate)
        WHERE entity_label IN labels(candidate)
          AND CASE WHEN supplied_id IS NOT NULL THEN candidate.id = supplied_id
                   ELSE toLower(trim(candidate.name)) = toLower(trim(entity_name)) END
        WITH entity_label, entity_name, supplied_id, collect(candidate) AS candidates
        CALL apoc.util.validate(size(candidates) > 1, 'Ambiguous canonical entity', [])
        WITH entity_label, entity_name, supplied_id, head(candidates) AS found
        CALL apoc.util.validate(found IS NOT NULL AND (found.id IS NULL OR found.id = ''),
                                'Canonical entity has no stable ID', [])
        WITH entity_label, entity_name,
             coalesce(supplied_id, found.id, apoc.util.sha256([entity_label, toLower(trim(entity_name))])) AS entity_id
        CALL apoc.merge.node([entity_label], {{id: entity_id}},
            {{name: entity_name, created_chapter: {chapter}, is_provisional: true,
              created_ts: timestamp(), updated_ts: timestamp()}},
            {{updated_ts: timestamp()}}) YIELD node
        RETURN node AS {variable}
    }}
    """


def assertion_cypher(source: str, target: str, predicate: str, chapter: str, origin: str, properties: str) -> str:
    """Write one occurrence per endpoint/type/chapter/owner, retaining a semantic fact ID."""
    return f"""
    CALL apoc.merge.relationship(
        {source}, {predicate}, {{chapter_added: {chapter}, assertion_origin: {origin}}},
        apoc.map.merge({properties}, {{created_ts: timestamp(), updated_ts: timestamp()}}),
        {target}, apoc.map.merge({properties}, {{updated_ts: timestamp()}})
    ) YIELD rel
    SET rel.fact_id = apoc.util.sha256([
            apoc.text.join(apoc.coll.sort(labels({source})), ':'), {source}.id, {predicate},
            apoc.text.join(apoc.coll.sort(labels({target})), ':'), coalesce({target}.id, {target}.value)]),
        rel.id = apoc.util.sha256([rel.fact_id, toString({chapter}), {origin}])
    """


def chapter_assertion_delete_statement(chapter: int) -> tuple[str, dict[str, Any]]:
    """Remove only chapter-owned projections, never profile/import or unclassified legacy edges."""
    return (
        """
        MATCH ()-[r]->()
        WHERE r.chapter_added = $chapter
          AND r.assertion_origin IN ['chapter_extraction', 'chapter_profile']
        DELETE r
        """,
        {"chapter": chapter},
    )


def relationship_statement(
    subject: dict[str, Any], predicate: str, target: dict[str, Any], chapter: int,
    *, origin: str, provisional: bool, description: str = "", confidence: float = 1.0,
    literal: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Shared extracted/imported relationship writer; values remain query parameters."""
    from data_access.kg_queries import validate_relationship_type_for_cypher_interpolation

    predicate = validate_relationship_type_for_cypher_interpolation(predicate)
    for entity in [subject] + ([] if literal else [target]):
        if entity.get("type") not in {"Character", "Location", "Item", "Event"}:
            raise ValueError("Invalid canonical entity label")
        if not isinstance(entity.get("name"), str) or not entity["name"].strip():
            raise ValueError("Invalid canonical entity name")
        identifier = entity.get("id")
        if identifier is not None and (not isinstance(identifier, str) or not identifier.strip()):
            raise ValueError("Invalid canonical entity ID")
    if origin not in {"chapter_extraction", "chapter_profile", "import", "profile"}:
        raise ValueError("Invalid assertion origin")
    if type(chapter) is not int or chapter < 0:
        raise ValueError("Invalid assertion chapter")
    query = canonical_entity_cypher("s", "$subject_label", "$subject_name", "$subject_id", "$chapter")
    if literal:
        query += """
        MERGE (o:ValueNode {value: $object_value, type: 'Literal'})
        ON CREATE SET o.created_ts = timestamp()
        WITH s, o
        """
    else:
        query += canonical_entity_cypher("o", "$object_label", "$object_name", "$object_id", "$chapter")
    query += assertion_cypher("s", "o", "$predicate_clean", "$chapter", "$assertion_origin", "$relationship_properties")
    query += "RETURN rel"
    return query, {
        "subject_label": subject["type"], "subject_name": subject["name"], "subject_id": subject.get("id"),
        "object_label": target.get("type"), "object_name": target.get("name"), "object_id": target.get("id"),
        "object_value": target.get("value"), "predicate_clean": predicate, "chapter": chapter, "assertion_origin": origin,
        "relationship_properties": {"type": predicate, "is_provisional": provisional, "description": description,
                                    "confidence": confidence, "source_profile_managed": origin in {"profile", "chapter_profile"}},
    }


class NativeCypherBuilder:
    """Generate Cypher directly from Pydantic models without dict conversion."""

    @staticmethod
    def character_physical_description_cypher(char: "CharacterProfile", chapter_number: int) -> tuple[str, dict[str, Any]]:
        """Update one existing stable identity without replaying profile properties or assertions."""
        if not char.id.strip():
            raise ValueError("Physical-description updates require a stable ID")
        if char.physical_description is None:
            raise ValueError("Physical-description updates require an explicit value")
        if chapter_number < 0:
            raise ValueError("Physical-description updates require a nonnegative chapter")
        return """
            OPTIONAL MATCH (candidate:Character {id: $id})
            WITH collect(candidate) AS candidates
            CALL apoc.util.validate(size(candidates) <> 1, 'Physical-description target must exist uniquely', [])
            WITH head(candidates) AS c
            FOREACH (ignored IN CASE WHEN c.physical_description = $physical_description THEN [] ELSE [1] END |
                SET c.physical_description = $physical_description,
                    c.chapter_last_updated = $chapter_number,
                    c.updated_ts = timestamp())
            RETURN c.id AS updated_character
        """, {"id": char.id, "physical_description": char.physical_description, "chapter_number": chapter_number}

    @staticmethod
    def character_upsert_cypher(char: "CharacterProfile", chapter_number: int, *, assertion_origin: str = "profile") -> tuple[str, dict[str, Any]]:
        """Build a character upsert statement.

        Args:
            char: Character profile model.
            chapter_number: Chapter number used for provenance and update tracking.

        Returns:
            A `(cypher_query, parameters)` tuple.

        Notes:
            Relationship typing:
                Relationships are written via `apoc.merge.relationship` using a dynamic
                relationship type (`rel_data.rel_type`). This is not parameterizable in
                plain Cypher and enables creation of arbitrary relationship types.

                Upstream callers must ensure relationship types are constrained (for example,
                already-uppercase and matching `[A-Z0-9_]+`) and come from an application
                allowlist, not from raw user text.

            Provisional node creation:
                Relationship targets are merged as `:Character` by name. When the target does
                not exist, a provisional stub node is created with `is_provisional=true`.
        """
        cypher = canonical_entity_cypher("c", "'Character'", "$name", "$id", "$chapter_number")
        cypher += """
        SET c.personality_description = $description,
            c.status = $status,
            c.created_chapter = CASE
                WHEN c.created_chapter IS NULL THEN $created_chapter
                ELSE c.created_chapter
            END,
            c.is_provisional = $is_provisional,
            c.chapter_last_updated = $chapter_number,
            c.updated_ts = timestamp()

        // Handle traits as a node property
        SET c.traits = $trait_data
        SET c += $domain_properties

        WITH c
        CALL (c) {
            UNWIND $relationship_data AS rel_data
        """
        cypher += canonical_entity_cypher("other", "'Character'", "rel_data.target_name", "rel_data.target_id", "$chapter_number", scope="rel_data")
        cypher += assertion_cypher("c", "other", "rel_data.rel_type", "$chapter_number", "$assertion_origin",
                                   "{description: rel_data.description, source_profile_managed: true, type: rel_data.rel_type}")
        cypher += "} RETURN c.name as updated_character"

        # Process relationships for batch operations
        relationship_data = []
        for target_name, rel_info in char.relationships.items():
            if not isinstance(rel_info, dict):
                raise ValueError(f"Relationship data for {char.name} -> {target_name} must be a dict " f"with 'type' and 'description' keys, got {type(rel_info).__name__}")
            rel_type_raw = rel_info.get("type", "")
            rel_desc = rel_info.get("description", "")

            rel_type = str(rel_type_raw).strip().upper().replace(" ", "_") if rel_type_raw else ""
            if not rel_type:
                continue

            relationship_data.append(
                {
                    "target_name": target_name,
                    "rel_type": rel_type,
                    "target_id": rel_info.get("target_id"),
                    "description": rel_desc,
                }
            )

        # Process traits - filter out empty strings
        trait_data = [t.strip() for t in char.traits if t and t.strip()]

        params = {
            "name": char.name,
            "domain_properties": char.model_dump(
                include={"motivations", "background", "skills", "internal_conflict", "is_protagonist", "physical_description"},
                exclude_unset=True, exclude_none=True,
            ),
            "description": char.personality_description,
            "trait_data": trait_data,  # List of trait names for UNWIND
            "status": char.status,
            "id": char.id or None,
            "assertion_origin": assertion_origin,
            "created_chapter": char.created_chapter or chapter_number,
            "is_provisional": char.is_provisional,
            "chapter_number": chapter_number,
            "relationship_data": relationship_data,
        }

        return cypher, params

    @staticmethod
    def world_item_upsert_cypher(item: "WorldItem", chapter_number: int, *, assertion_origin: str = "profile") -> tuple[str, dict[str, Any]]:
        """Build a world item upsert statement.

        Args:
            item: World item model.
            chapter_number: Chapter number used for provenance and update tracking.

        Returns:
            A `(cypher_query, parameters)` tuple.

        Notes:
            Label selection:
                The primary node label is interpolated into the query string (labels are not
                parameterized by Neo4j). This builder constrains the label to
                `WORLD_ITEM_CANONICAL_LABELS` and falls back to `Item`.

                Do not pass untrusted label values into this builder.

            Relationship targets:
                Relationship target labels are selected inside Cypher via an allowlist
                (`world_item_target_label_allowlist`), defaulting to `Item` when invalid.
        """
        # Flatten nested dictionaries in additional_properties to ensure
        # all values are primitive types that Neo4j can store
        flattened_additional_props = flatten_dict(item.additional_properties)
        if {"id", "name"} & flattened_additional_props.keys():
            raise ValueError("Additional properties cannot override canonical identity")

        primary_label = classify_category_label(item.category)

        # P0.2: Ensure world upserts always select a canonical "world item" label.
        # This prevents accidentally writing :Trait/:Character/etc. from ambiguous categories,
        # which would then be invisible to world reads/fetches.
        if primary_label not in WORLD_ITEM_CANONICAL_LABELS:
            primary_label = "Item"

        # Build a safe labels clause. In Cypher, labels are colon-separated with no commas.
        # Removed implicit Entity label inheritance

        cypher = canonical_entity_cypher("w", "$primary_label", "$name", "$id", "$chapter_number")
        cypher += """
        SET
            w.category = $category,
            w.description = $description,
            w.goals = $goals,
            w.rules = $rules,
            w.key_elements = $key_elements,
            w.created_chapter = coalesce(w.created_chapter, $created_chapter),
            w.is_provisional = $is_provisional,
            w.chapter_last_updated = $chapter_number,
            w.updated_ts = timestamp(),
            w.created_at = coalesce(w.created_at, timestamp())
        WITH w
        SET w += $additional_props

        // Handle traits as a node property
        SET w.traits = $trait_data

        WITH w
        CALL (w) {
            UNWIND $relationship_data AS rel_data
        """
        cypher += canonical_entity_cypher("other", "coalesce(rel_data.target_label, 'Item')", "rel_data.target_name", "rel_data.target_id", "$chapter_number", scope="rel_data")
        cypher += assertion_cypher("w", "other", "rel_data.rel_type", "$chapter_number", "$assertion_origin",
                                   "{description: rel_data.description, source_profile_managed: true, type: rel_data.rel_type}")
        cypher += "} RETURN w.id as updated_world_item"

        # Process relationships for batch operations
        relationship_data = []
        for target_name, rel_info in item.relationships.items():
            target_label: str | None = None
            target_id: str | None = None

            if isinstance(rel_info, dict):
                rel_type_raw = rel_info.get("type", "RELATED_TO")
                rel_desc = rel_info.get("description", "")

                # Optional per-relationship target typing / identity (Option A).
                # These are validated/allowlisted at query time.
                target_label_raw = rel_info.get("target_label")
                target_id_raw = rel_info.get("target_id")

                if target_label_raw:
                    target_label = str(target_label_raw).strip().capitalize() or None
                if target_id_raw:
                    target_id = str(target_id_raw).strip() or None
            else:
                rel_type_raw = "RELATED_TO"
                rel_desc = str(rel_info) if rel_info else ""

            # Normalize relationship type for consistent storage and querying.
            rel_type = str(rel_type_raw).strip().upper().replace(" ", "_") if rel_type_raw else ""
            if not rel_type:
                rel_type = "RELATED_TO"

            relationship_data.append(
                {
                    "target_name": target_name,
                    "target_label": target_label,
                    "target_id": target_id,
                    "rel_type": rel_type,
                    "description": rel_desc,
                }
            )

        # Process traits - filter out empty strings
        trait_data = [t.strip() for t in item.traits if t and t.strip()]

        params = {
            "id": item.id or None,
            "primary_label": primary_label,
            "assertion_origin": assertion_origin,
            "name": item.name,
            "category": item.category,
            "description": item.description,
            "goals": item.goals,  # Direct field access
            "rules": item.rules,
            "key_elements": item.key_elements,
            "trait_data": trait_data,  # List of trait names for FOREACH
            "created_chapter": item.created_chapter or chapter_number,
            "is_provisional": item.is_provisional,
            "chapter_number": chapter_number,
            "additional_props": flattened_additional_props,  # Flattened to ensure primitive types
            "relationship_data": relationship_data,
            # Allowlist for safe label selection in apoc.merge.node
            "world_item_target_label_allowlist": list(WORLD_ITEM_CANONICAL_LABELS),
        }

        return cypher, params

    @staticmethod
    def character_fetch_cypher(
        filters: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Build a character fetch query with optional filters.

        Args:
            filters: Optional filter mapping. Supported keys:
                - `chapter_range`: `(min_chapter, max_chapter)` applied to `chapter_last_updated`
                - `is_provisional`: boolean applied to `c.is_provisional`

        Returns:
            A `(cypher_query, parameters)` tuple.

        Notes:
            Query safety:
                Filter values are passed as parameters. This builder does not accept dynamic
                labels or relationship types from `filters`.
        """
        where_clauses: list[str] = []
        params = {}

        if filters:
            if "chapter_range" in filters:
                where_clauses.append("c.chapter_last_updated >= $min_chapter")
                where_clauses.append("c.chapter_last_updated <= $max_chapter")
                params["min_chapter"] = filters["chapter_range"][0]
                params["max_chapter"] = filters["chapter_range"][1]

            if "is_provisional" in filters:
                where_clauses.append("c.is_provisional = $is_provisional")
                params["is_provisional"] = filters["is_provisional"]

        where_line = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        cypher = f"""
        MATCH (c:Character)
        {where_line}

        // Optionally collect relationships (use actual relationship type)
        OPTIONAL MATCH (c)-[r]->(other)

        RETURN c,
               collect(DISTINCT {{
                   target_name: other.name,
                   // Use actual relationship type; fallback to r.type property for legacy RELATIONSHIP types
                   type: CASE WHEN type(r) = 'RELATIONSHIP' THEN coalesce(r.type, type(r)) ELSE type(r) END,
                   description: coalesce(r.description, '')
               }}) as relationships
        ORDER BY c.name
        """

        return cypher, params

    @staticmethod
    def world_item_fetch_cypher(
        filters: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Build a world item fetch query with optional filters.

        Args:
            filters: Optional filter mapping. Supported keys:
                - `category`: world item category (property filter)
                - `chapter_range`: `(min_chapter, max_chapter)` applied to `chapter_last_updated`

        Returns:
            A `(cypher_query, parameters)` tuple.

        Notes:
            Label safety:
                The query restricts candidate nodes to `WORLD_ITEM_CANONICAL_LABELS` via a
                label predicate derived from an application constant. It must not accept
                arbitrary labels from callers.
        """
        where_clauses: list[str] = []
        params = {}

        if filters:
            if "category" in filters:
                where_clauses.append("w.category = $category")
                params["category"] = filters["category"]

            if "chapter_range" in filters:
                where_clauses.append("w.chapter_last_updated >= $min_chapter")
                where_clauses.append("w.chapter_last_updated <= $max_chapter")
                params["min_chapter"] = filters["chapter_range"][0]
                params["max_chapter"] = filters["chapter_range"][1]

        where_line = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Canonical labeling contract:
        # - World item nodes are labeled with canonical world labels only
        #   (Location/Item/Event).
        # - Legacy labels (Object/Artifact/Relic/Document) are handled via explicit migration,
        #   not by widening read predicates indefinitely.
        world_item_labels = WORLD_ITEM_CANONICAL_LABELS
        label_predicate = "(" + " OR ".join([f"w:{label}" for label in world_item_labels]) + ")"

        # Note: Character is handled by character_fetch_cypher
        additional_filter = f"AND {' AND '.join(where_clauses)}" if where_clauses else ""

        cypher = f"""
        MATCH (w)
        WHERE {label_predicate}
          {additional_filter}

        RETURN w
        ORDER BY w.category, w.name
        """

        return cypher, params

    @staticmethod
    def batch_character_upsert_cypher(characters: list["CharacterProfile"], chapter_number: int) -> list[tuple[str, dict[str, Any]]]:
        """Build batch character upsert statements.

        Args:
            characters: Character profiles to upsert.
            chapter_number: Chapter number used for provenance and update tracking.

        Returns:
            A list of `(cypher_query, parameters)` tuples.
        """
        statements = []
        for char in characters:
            cypher, params = NativeCypherBuilder.character_upsert_cypher(char, chapter_number)
            statements.append((cypher, params))
        return statements

    @staticmethod
    def batch_world_item_upsert_cypher(world_items: list["WorldItem"], chapter_number: int) -> list[tuple[str, dict[str, Any]]]:
        """Build batch world item upsert statements.

        Args:
            world_items: World items to upsert.
            chapter_number: Chapter number used for provenance and update tracking.

        Returns:
            A list of `(cypher_query, parameters)` tuples.
        """
        statements = []
        for item in world_items:
            cypher, params = NativeCypherBuilder.world_item_upsert_cypher(item, chapter_number)
            statements.append((cypher, params))
        return statements
