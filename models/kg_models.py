# models/kg_models.py
"""Core data models for characters and world elements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from models.kg_constants import KG_IS_PROVISIONAL, KG_NODE_CREATED_CHAPTER
from pydantic import BaseModel, Field

import utils

if TYPE_CHECKING:
    import neo4j


class CharacterProfile(BaseModel):
    """Structured information about a character."""

    name: str
    description: str = ""
    traits: list[str] = Field(default_factory=list)
    relationships: dict[str, Any] = Field(default_factory=dict)
    status: str = "Unknown"
    updates: dict[str, Any] = Field(default_factory=dict)
    created_chapter: int = 0
    is_provisional: bool = False

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> CharacterProfile:
        """Create a ``CharacterProfile`` from a raw dictionary."""

        known_fields = cls.model_fields.keys()
        profile_data = {k: v for k, v in data.items() if k in known_fields}
        updates_data = {k: v for k, v in data.items() if k not in known_fields}
        if "updates" in profile_data:
            updates_data.update(profile_data["updates"])
        profile_data["updates"] = updates_data
        return cls(name=name, **profile_data)

    def to_dict(self) -> dict[str, Any]:
        """Convert the profile to a flat dictionary."""

        data = self.model_dump(exclude={"name"})
        updates_data = data.pop("updates", {})
        data.update(updates_data)
        return data

    @classmethod
    def from_db_record(cls, record: neo4j.Record) -> CharacterProfile:
        """Construct directly from Neo4j record - no dict conversion."""
        node = record["c"]  # Assuming 'c' is the character node alias

        # Extract relationships if available
        relationships = {}
        if "relationships" in record and record["relationships"]:
            for rel in record["relationships"]:
                if rel and rel.get("target_name"):
                    relationships[rel["target_name"]] = {
                        "type": rel.get("type", "KNOWS"),
                        "description": rel.get("description", ""),
                    }

        return cls(
            name=node.get("name", ""),
            description=node.get("description", ""),
            traits=node.get("traits", []),
            status=node.get("status", "Unknown"),
            relationships=relationships,
            created_chapter=node.get("created_chapter", 0),
            is_provisional=node.get("is_provisional", False),
            updates={},  # Will be populated as needed
        )

    @classmethod
    def from_db_node(cls, node: neo4j.Node) -> CharacterProfile:
        """Construct directly from Neo4j node - no dict conversion."""
        return cls(
            name=node.get("name", ""),
            description=node.get("description", ""),
            traits=node.get("traits", []),
            status=node.get("status", "Unknown"),
            relationships={},  # Relationships handled separately
            created_chapter=node.get("created_chapter", 0),
            is_provisional=node.get("is_provisional", False),
            updates={},
        )

    def to_cypher_params(self) -> dict[str, Any]:
        """Direct conversion to Cypher parameters without dict serialization."""
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
    """Structured information about a world element."""

    id: str
    category: str
    name: str
    created_chapter: int = 0
    is_provisional: bool = False
    description: str = ""
    goals: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    key_elements: list[str] = Field(default_factory=list)
    traits: list[str] = Field(default_factory=list)
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
        """Create a ``WorldItem`` from a raw dictionary."""

        # Get ID from data if present, otherwise generate one
        item_id = data.get("id", "")

        # Validate and normalize all core fields
        category, name, item_id = utils.validate_world_item_fields(
            category, name, item_id, allow_empty_name
        )

        # Extract and validate created_chapter (handle various formats)
        value = data.get(KG_NODE_CREATED_CHAPTER, "0")
        if isinstance(value, list):
            created_chapter = int(value[0]) if value else 0
        elif isinstance(value, str):
            cleaned_value = (
                value.replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .replace('"', "")
            )
            num_str = cleaned_value.split(",")[0].strip()
            created_chapter = int(num_str) if num_str else 0
        else:
            created_chapter = int(value) if value is not None else 0

        # Extract and validate is_provisional
        is_provisional = bool(data.get(KG_IS_PROVISIONAL, False))

        # Extract structured fields
        description = data.get("description", "")
        goals = data.get("goals", [])
        rules = data.get("rules", [])
        key_elements = data.get("key_elements", [])
        traits = data.get("traits", [])

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
            additional_properties=additional_properties,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the item to a flat dictionary."""

        data = self.model_dump(exclude={"id", "category", "name"})
        additional_properties = data.pop("additional_properties", {})
        data.update(additional_properties)
        return data

    @classmethod
    def from_db_record(cls, record: neo4j.Record) -> WorldItem:
        """Construct directly from Neo4j record - no dict conversion."""
        # Try both 'w' and 'we' node aliases for compatibility
        node = record.get("w") or record.get("we")
        if not node:
            raise ValueError("No world element node found in record")

        def _safe_string_extract(value) -> str:
            """Safely extract string from potentially array value."""
            if isinstance(value, list):
                return value[0] if value else ""
            return str(value) if value is not None else ""

        def _safe_int_extract(value) -> int:
            """Safely extract int from potentially array value."""
            if isinstance(value, list):
                return int(value[0]) if value else 0
            elif isinstance(value, str):
                # Handle comma-separated values by taking first part
                # Clean potential list representation before splitting
                cleaned_value = (
                    value.replace("[", "")
                    .replace("]", "")
                    .replace("'", "")
                    .replace('"', "")
                )
                num_str = cleaned_value.split(",")[0].strip()
                return int(num_str) if num_str else 0
            return int(value) if value is not None else 0

        def _safe_timestamp_extract(value) -> int:
            """Safely extract timestamp from potentially array or comma-separated value."""
            if isinstance(value, list):
                return int(value[0]) if value and value[0] else 0
            elif isinstance(value, str):
                # Handle comma-separated values by taking first part
                cleaned_value = (
                    value.replace("[", "")
                    .replace("]", "")
                    .replace("'", "")
                    .replace('"', "")
                )
                timestamp_str = cleaned_value.split(",")[0].strip()
                return int(timestamp_str) if timestamp_str else 0
            return int(value) if value is not None else 0

        def _safe_list_extract(value) -> list[str]:
            """Safely extract list from potentially mixed value."""
            if isinstance(value, list):
                return [str(item) for item in value if item is not None]
            return [str(value)] if value is not None else []

        # Extract additional properties that aren't core fields
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
        additional_props = {k: v for k, v in dict(node).items() if k not in core_fields}

        # Extract core fields with proper type handling
        created_chapter_val = node.get("created_chapter", 0)
        chapter_last_updated_val = node.get("chapter_last_updated", None)
        last_updated_val = node.get("last_updated", None)

        return cls(
            id=_safe_string_extract(node.get("id", "")),
            category=_safe_string_extract(node.get("category", "")),
            name=_safe_string_extract(node.get("name", "")),
            description=_safe_string_extract(node.get("description", "")),
            goals=_safe_list_extract(node.get("goals", [])),
            rules=_safe_list_extract(node.get("rules", [])),
            key_elements=_safe_list_extract(node.get("key_elements", [])),
            traits=_safe_list_extract(node.get("traits", [])),
            created_chapter=_safe_int_extract(created_chapter_val),
            is_provisional=bool(node.get("is_provisional", False)),
            additional_properties=additional_props,
        )

    @classmethod
    def from_db_node(cls, node: neo4j.Node) -> WorldItem:
        """Construct directly from Neo4j node - no dict conversion."""

        def _safe_string_extract(value) -> str:
            """Safely extract string from potentially array value."""
            if isinstance(value, list):
                return value[0] if value else ""
            return str(value) if value is not None else ""

        def _safe_int_extract(value) -> int:
            """Safely extract int from potentially array value."""
            if isinstance(value, list):
                return int(value[0]) if value else 0
            elif isinstance(value, str):
                # Handle comma-separated values by taking first part
                # Clean potential list representation before splitting
                cleaned_value = (
                    value.replace("[", "")
                    .replace("]", "")
                    .replace("'", "")
                    .replace('"', "")
                )
                num_str = cleaned_value.split(",")[0].strip()
                return int(num_str) if num_str else 0
            return int(value) if value is not None else 0

        def _safe_list_extract(value) -> list[str]:
            """Safely extract list from potentially mixed value."""
            if isinstance(value, list):
                return [str(item) for item in value if item is not None]
            return [str(value)] if value is not None else []

        # Extract additional properties that aren't core fields
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
        additional_props = {k: v for k, v in dict(node).items() if k not in core_fields}

        return cls(
            id=_safe_string_extract(node.get("id", "")),
            category=_safe_string_extract(node.get("category", "")),
            name=_safe_string_extract(node.get("name", "")),
            description=_safe_string_extract(node.get("description", "")),
            goals=_safe_list_extract(node.get("goals", [])),
            rules=_safe_list_extract(node.get("rules", [])),
            key_elements=_safe_list_extract(node.get("key_elements", [])),
            traits=_safe_list_extract(node.get("traits", [])),
            created_chapter=_safe_int_extract(node.get("created_chapter", 0)),
            is_provisional=bool(node.get("is_provisional", False)),
            additional_properties=additional_props,
        )

    def to_cypher_params(self) -> dict[str, Any]:
        """Direct conversion to Cypher parameters without dict serialization."""
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
        }
