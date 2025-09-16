# data_access/cypher_builders/native_builders.py
"""
Native Cypher builders that generate Cypher directly from Pydantic models.
Eliminates the intermediate dict serialization layer for performance optimization.
"""

from typing import TYPE_CHECKING, Any

from utils.helpers import flatten_dict

if TYPE_CHECKING:
    from models.kg_models import CharacterProfile, WorldItem


class NativeCypherBuilder:
    """Generate Cypher directly from Pydantic models without dict conversion"""

    @staticmethod
    def character_upsert_cypher(
        char: "CharacterProfile", chapter_number: int
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate Cypher for character upsert directly from CharacterProfile model.

        Args:
            char: CharacterProfile model instance
            chapter_number: Current chapter for tracking updates

        Returns:
            Tuple of (cypher_query, parameters)
        """
        cypher = """
        MERGE (c:Character:Entity {name: $name})
        SET c.description = $description,
            c.traits = $traits,
            c.status = $status,
            c.created_chapter = CASE 
                WHEN c.created_chapter IS NULL THEN $created_chapter 
                ELSE c.created_chapter 
            END,
            c.is_provisional = $is_provisional,
            c.chapter_last_updated = $chapter_number,
            c.last_updated = timestamp()
        
        // Handle relationships as separate merge operations
        WITH c
        UNWIND $relationship_data AS rel_data
        CALL (c, rel_data) {
            WITH c, rel_data
            MATCH (other:Entity {name: rel_data.target_name})
            MERGE (c)-[r:RELATIONSHIP {type: rel_data.rel_type}]->(other)
            SET r.description = rel_data.description,
                r.last_updated = timestamp()
        }
        
        RETURN c.name as updated_character
        """

        # Process relationships for batch operations
        relationship_data = []
        for target_name, rel_info in char.relationships.items():
            if isinstance(rel_info, dict):
                rel_type = rel_info.get("type", "KNOWS")
                rel_desc = rel_info.get("description", "")
            else:
                rel_type = "KNOWS"
                rel_desc = str(rel_info) if rel_info else ""

            relationship_data.append(
                {
                    "target_name": target_name,
                    "rel_type": rel_type,
                    "description": rel_desc,
                }
            )

        params = {
            "name": char.name,
            "description": char.description,
            "traits": char.traits,  # Direct field access - no dict conversion
            "status": char.status,
            "created_chapter": char.created_chapter or chapter_number,
            "is_provisional": char.is_provisional,
            "chapter_number": chapter_number,
            "relationship_data": relationship_data,
        }

        return cypher, params

    @staticmethod
    def world_item_upsert_cypher(
        item: "WorldItem", chapter_number: int
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate Cypher for world item upsert directly from WorldItem model.

        Args:
            item: WorldItem model instance
            chapter_number: Current chapter for tracking updates

        Returns:
            Tuple of (cypher_query, parameters)
        """
        # Flatten nested dictionaries in additional_properties to ensure
        # all values are primitive types that Neo4j can store
        flattened_additional_props = flatten_dict(item.additional_properties)

        cypher = """
        MERGE (w:WorldElement:Entity {id: $id})
        SET w.name = $name,
            w.category = $category,
            w.description = $description,
            w.goals = $goals,
            w.rules = $rules,
            w.key_elements = $key_elements,
            w.traits = $traits,
            w.created_chapter = CASE 
                WHEN w.created_chapter IS NULL THEN $created_chapter 
                ELSE w.created_chapter 
            END,
            w.is_provisional = $is_provisional,
            w.chapter_last_updated = $chapter_number,
            w.last_updated = timestamp(),
            w += $additional_props
        
        RETURN w.id as updated_world_item
        """

        params = {
            "id": item.id,
            "name": item.name,
            "category": item.category,
            "description": item.description,
            "goals": item.goals,  # Direct field access
            "rules": item.rules,
            "key_elements": item.key_elements,
            "traits": item.traits,
            "created_chapter": item.created_chapter or chapter_number,
            "is_provisional": item.is_provisional,
            "chapter_number": chapter_number,
            "additional_props": flattened_additional_props,  # Flattened to ensure primitive types
        }

        return cypher, params

    @staticmethod
    def character_fetch_cypher(
        filters: dict[str, Any] = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate optimized Cypher for fetching characters with relationships.

        Args:
            filters: Optional filters for the query

        Returns:
            Tuple of (cypher_query, parameters)
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
        MATCH (c:Character:Entity)
        WHERE {where_clause}
        
        // Optionally collect relationships (match any type; use r.type property for semantics)
        OPTIONAL MATCH (c)-[r]->(other:Entity)
        
        RETURN c, 
               collect({{
                   target_name: other.name,
                   type: r.type,
                   description: r.description
               }}) as relationships
        ORDER BY c.name
        """

        return cypher, params

    @staticmethod
    def world_item_fetch_cypher(
        filters: dict[str, Any] = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate optimized Cypher for fetching world items.

        Args:
            filters: Optional filters for the query

        Returns:
            Tuple of (cypher_query, parameters)
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

        cypher = f"""
        MATCH (w:WorldElement:Entity)
        WHERE {where_clause}
        RETURN w
        ORDER BY w.category, w.name
        """

        return cypher, params

    @staticmethod
    def batch_character_upsert_cypher(
        characters: list["CharacterProfile"], chapter_number: int
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Generate batch Cypher statements for multiple characters.

        Args:
            characters: List of CharacterProfile models
            chapter_number: Current chapter for tracking

        Returns:
            List of (cypher_query, parameters) tuples
        """
        statements = []
        for char in characters:
            cypher, params = NativeCypherBuilder.character_upsert_cypher(
                char, chapter_number
            )
            statements.append((cypher, params))
        return statements

    @staticmethod
    def batch_world_item_upsert_cypher(
        world_items: list["WorldItem"], chapter_number: int
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Generate batch Cypher statements for multiple world items.

        Args:
            world_items: List of WorldItem models
            chapter_number: Current chapter for tracking

        Returns:
            List of (cypher_query, parameters) tuples
        """
        statements = []
        for item in world_items:
            cypher, params = NativeCypherBuilder.world_item_upsert_cypher(
                item, chapter_number
            )
            statements.append((cypher, params))
        return statements
