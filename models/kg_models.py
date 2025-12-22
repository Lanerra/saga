# models/kg_models.py
"""Define core knowledge-graph data models used across SAGA.

This module provides the in-memory representations used when moving data between:
- user-facing inputs,
- Neo4j record/node shapes, and
- Cypher parameter dictionaries.

Notes:
- These models intentionally preserve a flexible "overflow" area for unknown fields:
  [`CharacterProfile`](models/kg_models.py:19) uses `updates`, and
  [`WorldItem`](models/kg_models.py:123) uses `additional_properties`.
- `relationships` fields represent relationship *payloads* (type/description) keyed by a
  target entity identifier (typically the target entity's name as returned by queries).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from models.db_extraction_utils import Neo4jExtractor
from models.kg_constants import KG_IS_PROVISIONAL, KG_NODE_CREATED_CHAPTER
from utils.text_processing import validate_world_item_fields

if TYPE_CHECKING:
    import neo4j


class CharacterProfile(BaseModel):
    """Represent a character node and its narrative-relevant attributes.

    Notes:
        - `name` is treated as the stable identifier for character lookups in most
          in-memory maps and query outputs.
        - Unknown/extra fields from upstream sources are stored in `updates` by
          [`from_dict()`](models/kg_models.py:32) and flattened by [`to_dict()`](models/kg_models.py:44).
        - `relationships` stores relationship payloads keyed by target character name.
    """

    name: str
    type: str = "Character"
    description: str = ""
    traits: list[str] = Field(default_factory=list)
    relationships: dict[str, Any] = Field(default_factory=dict)
    status: str = "Unknown"
    updates: dict[str, Any] = Field(default_factory=dict)
    created_chapter: int = 0
    is_provisional: bool = False

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> CharacterProfile:
        """Create a profile from a raw dictionary.

        Fields matching model fields are applied directly. Remaining keys are merged into
        `updates` (including any nested `updates` keys already present in `data`).

        Args:
            name: Character name to use as the profile identifier.
            data: Source mapping with known fields and optional extra keys.

        Returns:
            A populated character profile.
        """

        known_fields = cls.model_fields.keys()
        profile_data = {k: v for k, v in data.items() if k in known_fields}
        updates_data = {k: v for k, v in data.items() if k not in known_fields}
        if "updates" in profile_data:
            updates_data.update(profile_data["updates"])
        profile_data["updates"] = updates_data
        return cls(name=name, **profile_data)

    def to_dict(self) -> dict[str, Any]:
        """Convert the profile to a flat dictionary.

        This merges `updates` into the top-level dictionary and excludes `name`.

        Returns:
            A flat dictionary suitable for JSON serialization or prompt injection.
        """

        data = self.model_dump(exclude={"name"})
        updates_data = data.pop("updates", {})
        data.update(updates_data)
        return data

    @classmethod
    def from_dict_record(cls, record: Mapping[str, Any]) -> CharacterProfile:
        """Construct a profile from a dictionary-shaped query result.

        This expects the character node to be available under the `c` alias and may
        optionally include a `relationships` collection and a `traits` collection.

        Args:
            record: Dictionary-like query result mapping.

        Returns:
            A populated character profile.
        """
        node = record["c"]  # Assuming 'c' is the character node alias

        # Extract relationships if available
        relationships = {}
        rels = record.get("relationships")
        if rels:
            for rel in rels:
                if rel and rel.get("target_name"):
                    relationships[rel["target_name"]] = {
                        "type": rel.get("type", "KNOWS"),
                        "description": rel.get("description", ""),
                    }

        # Extract traits from query result (HAS_TRAIT relationships)
        # Fallback to node property for backward compatibility
        traits = record.get("traits", [])
        if not traits:
            node_dict = node if isinstance(node, dict) else dict(node)
            traits = node_dict.get("traits", [])
        # Filter out None/empty values
        traits = [t for t in traits if t]

        # Node is already a dict if coming from db_manager
        node_dict = node if isinstance(node, dict) else dict(node)
        return cls(
            name=node_dict.get("name", ""),
            description=node_dict.get("description", ""),
            traits=traits,
            status=node_dict.get("status", "Unknown"),
            relationships=relationships,
            created_chapter=node_dict.get("created_chapter", 0),
            is_provisional=node_dict.get("is_provisional", False),
            updates={},  # Will be populated as needed
        )

    @classmethod
    def from_db_record(cls, record: neo4j.Record) -> CharacterProfile:
        """Construct a profile from a Neo4j record.

        This delegates to [`from_dict_record()`](models/kg_models.py:52) and relies on
        compatible field access on the record object.

        Args:
            record: Neo4j record returned by the driver.

        Returns:
            A populated character profile.
        """
        return cls.from_dict_record(record)

    @classmethod
    def from_db_node(cls, node: Any) -> CharacterProfile:
        """Construct a profile from a Neo4j node or node-like dict.

        Args:
            node: Neo4j node instance or a dictionary with character properties.

        Returns:
            A populated character profile.
        """
        node_dict = node if isinstance(node, dict) else dict(node)
        return cls(
            name=node_dict.get("name", ""),
            description=node_dict.get("description", ""),
            traits=node_dict.get("traits", []),
            status=node_dict.get("status", "Unknown"),
            relationships={},  # Relationships handled separately
            created_chapter=node_dict.get("created_chapter", 0),
            is_provisional=node_dict.get("is_provisional", False),
            updates={},
        )

    def to_cypher_params(self) -> dict[str, Any]:
        """Build a parameter dictionary for Cypher writes.

        This excludes `relationships` and `updates`, which are handled via separate
        write paths.

        Returns:
            A dictionary suitable for use as a Cypher parameter map.
        """
        return {
            "name": self.name,
            "description": self.description,
            "traits": self.traits,
            "status": self.status,
            "created_chapter": self.created_chapter,
            "is_provisional": self.is_provisional,
            # Note: relationships and updates handled separately
        }


class WorldItem(BaseModel):
    """Represent a non-character entity in the knowledge graph.

    Notes:
        - `id` is the stable identifier used for read-by-id operations and upserts.
        - `category` is a free-form subtype classifier (e.g., "City", "Weapon") and is
          intentionally not modeled as a Neo4j label.
        - Unknown/extra fields from upstream sources are stored in `additional_properties`
          by [`from_dict()`](models/kg_models.py:141) and flattened by [`to_dict()`](models/kg_models.py:208).
    """

    id: str
    category: str
    name: str
    type: str = "Item"
    created_chapter: int = 0
    is_provisional: bool = False
    description: str = ""
    goals: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    key_elements: list[str] = Field(default_factory=list)
    traits: list[str] = Field(default_factory=list)
    relationships: dict[str, Any] = Field(default_factory=dict)
    # Additional properties can still be stored in a dictionary for flexibility
    additional_properties: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        category: str,
        name: str,
        data: dict[str, Any],
        allow_empty_name: bool = False,
    ) -> WorldItem:
        """Create an item from a raw dictionary.

        Core fields are normalized/validated via
        [`validate_world_item_fields()`](models/kg_models.py:13). Remaining keys are
        preserved in `additional_properties`.

        Args:
            category: Category/subtype for the item (not a Neo4j label).
            name: Human-readable name for the item.
            data: Source mapping containing structured fields and optional extra keys.
            allow_empty_name: Whether validation may accept an empty `name`.

        Returns:
            A populated world item.
        """

        # Get ID from data if present, otherwise generate one
        item_id = data.get("id", "")

        # Validate and normalize all core fields
        category, name, item_id = validate_world_item_fields(category, name, item_id, allow_empty_name)

        # Extract and validate created_chapter using shared utility
        created_chapter = Neo4jExtractor.safe_int_extract(data.get(KG_NODE_CREATED_CHAPTER, 0))

        # Extract and validate is_provisional
        is_provisional = bool(data.get(KG_IS_PROVISIONAL, False))

        # Extract structured fields
        description = data.get("description", "")
        goals = data.get("goals", [])
        rules = data.get("rules", [])
        key_elements = data.get("key_elements", [])
        traits = data.get("traits", [])

        # Extract relationships if present
        relationships = data.get("relationships", {})

        # Collect remaining properties
        additional_properties = {
            k: v
            for k, v in data.items()
            if k
            not in {
                "id",
                "category",
                "name",
                KG_NODE_CREATED_CHAPTER,
                KG_IS_PROVISIONAL,
                "description",
                "goals",
                "rules",
                "key_elements",
                "traits",
                "relationships",
            }
        }

        return cls(
            id=item_id,
            category=category,
            name=name,
            created_chapter=created_chapter,
            is_provisional=is_provisional,
            description=description,
            goals=goals,
            rules=rules,
            key_elements=key_elements,
            traits=traits,
            relationships=relationships,
            additional_properties=additional_properties,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the item to a flat dictionary.

        This merges `additional_properties` into the top-level dictionary and excludes
        `id`, `category`, and `name`.

        Returns:
            A flat dictionary suitable for JSON serialization or prompt injection.
        """

        data = self.model_dump(exclude={"id", "category", "name"})
        additional_properties = data.pop("additional_properties", {})
        data.update(additional_properties)
        return data

    @classmethod
    def from_dict_record(cls, record: Mapping[str, Any]) -> WorldItem:
        """Construct an item from a dictionary-shaped query result.

        The world-element node is expected under the `w` or `we` alias. The record may
        optionally include a `relationships` collection and a `traits` collection.

        Args:
            record: Dictionary-like query result mapping.

        Returns:
            A populated world item.

        Raises:
            ValueError: If the record does not contain a world-element node under `w` or `we`.
        """
        # Try both 'w' and 'we' node aliases for compatibility
        node = record.get("w") or record.get("we")
        if not node:
            raise ValueError("No world element node found in record")

        # Define core fields that shouldn't go into additional_properties
        core_fields = {
            "id",
            "name",
            "category",
            "description",
            "goals",
            "rules",
            "key_elements",
            "traits",
            "created_chapter",
            "is_provisional",
            "chapter_last_updated",
            "last_updated",
        }

        # Extract additional properties using shared utility
        additional_props = Neo4jExtractor.extract_core_fields_from_node(node, core_fields)

        # Extract relationships if available
        relationships = {}
        rels = record.get("relationships")
        if rels:
            for rel in rels:
                if rel and rel.get("target_name"):
                    relationships[rel["target_name"]] = {
                        "type": rel.get("type", "RELATED_TO"),
                        "description": rel.get("description", ""),
                    }

        # Extract traits from query result (HAS_TRAIT relationships)
        # Fallback to node property for backward compatibility
        traits = record.get("traits", [])
        if not traits:
            node_dict = node if isinstance(node, dict) else dict(node)
            traits = Neo4jExtractor.safe_list_extract(node_dict.get("traits", []))
        # Filter out None/empty values
        traits = [t for t in traits if t]

        node_dict = node if isinstance(node, dict) else dict(node)
        return cls(
            id=Neo4jExtractor.safe_string_extract(node_dict.get("id", "")),
            category=Neo4jExtractor.safe_string_extract(node_dict.get("category", "")),
            name=Neo4jExtractor.safe_string_extract(node_dict.get("name", "")),
            description=Neo4jExtractor.safe_string_extract(node_dict.get("description", "")),
            goals=Neo4jExtractor.safe_list_extract(node_dict.get("goals", [])),
            rules=Neo4jExtractor.safe_list_extract(node_dict.get("rules", [])),
            key_elements=Neo4jExtractor.safe_list_extract(node_dict.get("key_elements", [])),
            traits=traits,
            created_chapter=Neo4jExtractor.safe_int_extract(node_dict.get("created_chapter", 0)),
            is_provisional=bool(node_dict.get("is_provisional", False)),
            relationships=relationships,
            additional_properties=additional_props,
        )

    @classmethod
    def from_db_record(cls, record: neo4j.Record) -> WorldItem:
        """Construct an item from a Neo4j record.

        This delegates to [`from_dict_record()`](models/kg_models.py:216) and relies on
        compatible field access on the record object.

        Args:
            record: Neo4j record returned by the driver.

        Returns:
            A populated world item.
        """
        return cls.from_dict_record(record)

    @classmethod
    def from_db_node(cls, node: Any) -> WorldItem:
        """Construct an item from a Neo4j node or node-like dict.

        Args:
            node: Neo4j node instance or a dictionary with world-item properties.

        Returns:
            A populated world item.
        """

        # Define core fields that shouldn't go into additional_properties
        core_fields = {
            "id",
            "name",
            "category",
            "description",
            "goals",
            "rules",
            "key_elements",
            "traits",
            "created_chapter",
            "is_provisional",
            "chapter_last_updated",
            "last_updated",
        }

        # Extract additional properties using shared utility
        additional_props = Neo4jExtractor.extract_core_fields_from_node(node, core_fields)

        node_dict = node if isinstance(node, dict) else dict(node)
        return cls(
            id=Neo4jExtractor.safe_string_extract(node_dict.get("id", "")),
            category=Neo4jExtractor.safe_string_extract(node_dict.get("category", "")),
            name=Neo4jExtractor.safe_string_extract(node_dict.get("name", "")),
            description=Neo4jExtractor.safe_string_extract(node_dict.get("description", "")),
            goals=Neo4jExtractor.safe_list_extract(node_dict.get("goals", [])),
            rules=Neo4jExtractor.safe_list_extract(node_dict.get("rules", [])),
            key_elements=Neo4jExtractor.safe_list_extract(node_dict.get("key_elements", [])),
            traits=Neo4jExtractor.safe_list_extract(node_dict.get("traits", [])),
            created_chapter=Neo4jExtractor.safe_int_extract(node_dict.get("created_chapter", 0)),
            is_provisional=bool(node_dict.get("is_provisional", False)),
            relationships={},  # Relationships handled separately
            additional_properties=additional_props,
        )

    def to_cypher_params(self) -> dict[str, Any]:
        """Build a parameter dictionary for Cypher writes.

        This excludes `relationships`, which are handled via separate write paths.

        Returns:
            A dictionary suitable for use as a Cypher parameter map.
        """
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "goals": self.goals,
            "rules": self.rules,
            "key_elements": self.key_elements,
            "traits": self.traits,
            "created_chapter": self.created_chapter,
            "is_provisional": self.is_provisional,
            "additional_props": self.additional_properties,
            # Note: relationships handled separately
        }


from dataclasses import dataclass, field


@dataclass
class RelationshipUsage:
    """Track narrative usage statistics for a relationship type.

    This is used by relationship normalization to keep vocabulary consistent while
    still allowing novel relationship types to appear over time.

    Notes:
        - `canonical_type` is the normalized relationship type name (for example, "WORKS_WITH").
        - `synonyms` captures observed variants that should normalize to `canonical_type`.
        - `embedding` may be populated to support similarity comparisons; it is optional.
    """

    canonical_type: str  # The normalized form (e.g., "WORKS_WITH")
    first_used_chapter: int  # When first introduced
    usage_count: int  # How many times used across narrative
    example_descriptions: list[str] = field(default_factory=list)  # Sample usage contexts
    embedding: list[float] | None = None  # Cached embedding for fast comparison
    synonyms: list[str] = field(default_factory=list)  # Variant forms normalized to this
    last_used_chapter: int = 0  # Most recent usage

    class Config:
        """Configure dataclass validation behavior."""

        frozen = False
        validate_assignment = True

    def to_dict(self) -> dict[str, Any]:
        """Convert usage tracking to a JSON-serializable dictionary.

        Returns:
            A dictionary with the tracked counters and metadata.
        """
        return {
            "canonical_type": self.canonical_type,
            "first_used_chapter": self.first_used_chapter,
            "usage_count": self.usage_count,
            "example_descriptions": self.example_descriptions,
            "embedding": self.embedding,
            "synonyms": self.synonyms,
            "last_used_chapter": self.last_used_chapter,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationshipUsage:
        """Create usage tracking from a dictionary.

        Args:
            data: Serialized representation produced by [`to_dict()`](models/kg_models.py:367).

        Returns:
            A relationship usage instance.
        """
        return cls(
            canonical_type=data["canonical_type"],
            first_used_chapter=data["first_used_chapter"],
            usage_count=data["usage_count"],
            example_descriptions=data.get("example_descriptions", []),
            embedding=data.get("embedding"),
            synonyms=data.get("synonyms", []),
            last_used_chapter=data.get("last_used_chapter", 0),
        )


__all__ = [
    "CharacterProfile",
    "WorldItem",
    "RelationshipUsage",
]
