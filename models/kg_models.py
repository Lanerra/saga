"""Core data models for characters and world elements."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

import utils
from kg_constants import KG_IS_PROVISIONAL, KG_NODE_CREATED_CHAPTER


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
    def from_dict(cls, category: str, name: str, data: dict[str, Any], allow_empty_name: bool = False) -> WorldItem:
        """Create a ``WorldItem`` from a raw dictionary."""

        # Get ID from data if present, otherwise generate one
        item_id = data.get("id", "")

        # Validate and normalize all core fields
        category, name, item_id = utils.validate_world_item_fields(
            category, name, item_id, allow_empty_name
        )

        created_chapter = int(data.get(KG_NODE_CREATED_CHAPTER, 0))
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
