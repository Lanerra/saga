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
from utils.text_processing import generate_entity_id
from utils import classify_category_label
from utils.common import flatten_dict

if TYPE_CHECKING:
    from models.kg_models import CharacterProfile, WorldItem


class NativeCypherBuilder:
    """Generate Cypher directly from Pydantic models without dict conversion."""

    @staticmethod
    def character_upsert_cypher(char: "CharacterProfile", chapter_number: int) -> tuple[str, dict[str, Any]]:
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
        cypher = """
        MERGE (c:Character {name: $name})
        SET c.personality_description = $description,
            c.status = $status,
            c.id = CASE WHEN c.id IS NULL OR c.id = '' THEN $id ELSE c.id END,
            c.created_chapter = CASE
                WHEN c.created_chapter IS NULL THEN $created_chapter
                ELSE c.created_chapter
            END,
            c.is_provisional = $is_provisional,
            c.chapter_last_updated = $chapter_number,
            c.updated_ts = timestamp()

        // Handle traits as a node property
        SET c.traits = $trait_data

        // Handle relationships as separate merge operations
        WITH c
        UNWIND $relationship_data AS rel_data
        CALL (c, rel_data) {
            WITH c, rel_data
            // Use MERGE instead of MATCH to create provisional nodes if they don't exist
            // Mark them as Character since they're related to a character
            MERGE (other:Character {name: rel_data.target_name})
            ON CREATE SET
                other.is_provisional = true,
                other.created_chapter = $chapter_number,
                other.id = randomUUID(),
                other.description = 'Character created from relationship. Details to be developed.',
                other.status = 'Unknown',
                other.updated_ts = timestamp()

            // Use apoc.merge.relationship to create relationships with dynamic types
            // This allows proper semantic relationship types (KNOWS, LOVES, etc.) instead of generic RELATIONSHIP
            WITH c, other, rel_data
            CALL apoc.merge.relationship(
                c,
                rel_data.rel_type,
                {},
                {
                    description: rel_data.description,
                    updated_ts: timestamp(),
                    chapter_added: $chapter_number,

                    // P1.7: Ensure builder-created relationships are visible to profile reads
                    // (profile reads filter on r.source_profile_managed).
                    source_profile_managed: true,

                    // P1.7: Transitional compatibility for any readers still using r.type.
                    type: rel_data.rel_type
                },
                other
            ) YIELD rel
            SET rel.description = rel_data.description,
                rel.updated_ts = timestamp(),
                rel.source_profile_managed = true,
                rel.type = rel_data.rel_type
        }

        RETURN c.name as updated_character
        """

        # Process relationships for batch operations
        relationship_data = []
        for target_name, rel_info in char.relationships.items():
            if isinstance(rel_info, dict):
                rel_type_raw = rel_info.get("type", "")
                rel_desc = rel_info.get("description", "")
            else:
                rel_type_raw = ""
                rel_desc = str(rel_info) if rel_info else ""

            # P1.7: Normalize relationship type for consistent storage and querying.
            # Canonical representation is the Neo4j relationship type itself.
            rel_type = str(rel_type_raw).strip().upper().replace(" ", "_") if rel_type_raw else ""
            if not rel_type:
                # If no relationship type is provided, do not create a relationship.
                # This is a change from previous behavior which defaulted to "KNOWS".
                continue

            relationship_data.append(
                {
                    "target_name": target_name,
                    "rel_type": rel_type,
                    "description": rel_desc,
                }
            )

        # Process traits - filter out empty strings
        trait_data = [t.strip() for t in char.traits if t and t.strip()]

        params = {
            "name": char.name,
            "description": char.personality_description,
            "trait_data": trait_data,  # List of trait names for UNWIND
            "status": char.status,
            # Stable deterministic ID for characters (assigned once)
            "id": generate_entity_id(
                char.name,
                "character",
                int(char.created_chapter or chapter_number),
            ),
            "created_chapter": char.created_chapter or chapter_number,
            "is_provisional": char.is_provisional,
            "chapter_number": chapter_number,
            "relationship_data": relationship_data,
        }

        return cypher, params

    @staticmethod
    def world_item_upsert_cypher(item: "WorldItem", chapter_number: int) -> tuple[str, dict[str, Any]]:
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

        primary_label = classify_category_label(item.category)

        # P0.2: Ensure world upserts always select a canonical "world item" label.
        # This prevents accidentally writing :Trait/:Character/etc. from ambiguous categories,
        # which would then be invisible to world reads/fetches.
        if primary_label not in WORLD_ITEM_CANONICAL_LABELS:
            primary_label = "Item"

        # Build a safe labels clause. In Cypher, labels are colon-separated with no commas.
        # Removed implicit Entity label inheritance

        # MERGE by ID to ensure we match existing entities even if renamed
        cypher = f"""
        MERGE (w:{primary_label} {{id: $id}})
        ON CREATE SET
            w.name = $name,
            w.category = $category,
            w.description = $description,
            w.goals = $goals,
            w.rules = $rules,
            w.key_elements = $key_elements,
            w.created_chapter = $created_chapter,
            w.is_provisional = $is_provisional,
            w.chapter_last_updated = $chapter_number,
            w.updated_ts = timestamp(),
            w.created_at = timestamp()
        ON MATCH SET
            w.name = $name,
            w.category = $category,
            w.description = $description,
            w.goals = $goals,
            w.rules = $rules,
            w.key_elements = $key_elements,
            w.is_provisional = $is_provisional,
            w.chapter_last_updated = $chapter_number,
            w.updated_ts = timestamp()
        WITH w
        SET w += $additional_props

        // Handle traits as a node property
        SET w.traits = $trait_data

        // Handle relationships as separate merge operations
        WITH w
        UNWIND $relationship_data AS rel_data
        CALL (w, rel_data) {{
            // Use apoc.merge.node to create/merge targets with a SAFE allowlisted label.
            // Default remains :Item if target_label is missing/invalid.
            WITH
                w,
                rel_data,
                CASE
                    WHEN rel_data.target_label IN $world_item_target_label_allowlist THEN rel_data.target_label
                    ELSE 'Item'
                END AS target_label,
                CASE
                    WHEN rel_data.target_id IS NOT NULL AND rel_data.target_id <> '' THEN {{id: rel_data.target_id}}
                    ELSE {{name: rel_data.target_name}}
                END AS merge_key_props

            CALL apoc.merge.node(
                [target_label],
                // Use ID as primary key if available, otherwise use name
                CASE
                    WHEN rel_data.target_id IS NOT NULL AND rel_data.target_id <> ''
                    THEN {{id: rel_data.target_id}}
                    ELSE {{name: rel_data.target_name}}
                END,
                {{
                    id: coalesce(rel_data.target_id, randomUUID()),
                    name: rel_data.target_name,
                    is_provisional: true,
                    created_chapter: $chapter_number,
                    description: 'Entity created from world item relationship. Details to be developed.'
                }},
                {{}}
            ) YIELD node AS other

            // Use apoc.merge.relationship to create relationships with dynamic types
            // This allows proper semantic relationship types (LOCATED_IN, PART_OF, etc.) instead of generic RELATIONSHIP
            WITH w, other, rel_data
            CALL apoc.merge.relationship(
                w,
                rel_data.rel_type,
                {{}},
                {{description: rel_data.description, updated_ts: timestamp(), chapter_added: $chapter_number}},
                other
            ) YIELD rel
            SET rel.description = rel_data.description,
                rel.updated_ts = timestamp()
        }}

        RETURN w.id as updated_world_item
        """

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
            "id": item.id,
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
        where_clauses = ["(c.is_deleted IS NULL OR c.is_deleted = FALSE)"]
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

        where_clause = " AND ".join(where_clauses)

        cypher = f"""
        MATCH (c:Character)
        WHERE {where_clause}

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
        where_clauses = ["(w.is_deleted IS NULL OR w.is_deleted = FALSE)"]
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

        where_clause = " AND ".join(where_clauses)

        # Canonical labeling contract:
        # - World item nodes are labeled with canonical world labels only
        #   (Location/Item/Event).
        # - Legacy labels (Object/Artifact/Relic/Document) are handled via explicit migration,
        #   not by widening read predicates indefinitely.
        world_item_labels = WORLD_ITEM_CANONICAL_LABELS
        label_predicate = "(" + " OR ".join([f"w:{label}" for label in world_item_labels]) + ")"

        # Note: Character is handled by character_fetch_cypher
        cypher = f"""
        MATCH (w)
        WHERE {label_predicate}
          AND {where_clause}

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
