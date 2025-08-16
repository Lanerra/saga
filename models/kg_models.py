"""Core data models for characters and world elements."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

import utils
from kg_constants import KG_IS_PROVISIONAL, KG_NODE_CREATED_CHAPTER


class CharacterProfile(BaseModel):
    """Structured information about a character."""

    name: str
    description: str = ""
    traits: List[str] = Field(default_factory=list)
    relationships: Dict[str, Any] = Field(default_factory=dict)
    status: str = "Unknown"
    updates: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "CharacterProfile":
        """Create a ``CharacterProfile`` from a raw dictionary."""

        known_fields = cls.model_fields.keys()
        profile_data = {k: v for k, v in data.items() if k in known_fields}
        updates_data = {k: v for k, v in data.items() if k not in known_fields}
        if "updates" in profile_data:
            updates_data.update(profile_data["updates"])
        profile_data["updates"] = updates_data
        return cls(name=name, **profile_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the profile to a flat dictionary."""

        data = self.model_dump(exclude={"name"})
        updates_data = data.pop("updates", {})
        data.update(updates_data)
        return data

    def get_development_summary(self, up_to_chapter: Optional[int] = None) -> List[str]:
        """Return development notes up to a chapter number."""

        notes: List[str] = []
        prefix = "development_in_chapter_"
        for key, val in sorted(self.updates.items()):
            if not key.startswith(prefix):
                continue
            try:
                chap = int(key.split("_")[-1])
            except (ValueError, IndexError):
                continue
            if up_to_chapter is None or chap <= up_to_chapter:
                if isinstance(val, str):
                    notes.append(val)
        return notes


class WorldItem(BaseModel):
    """Structured information about a world element."""

    id: str
    category: str
    name: str
    created_chapter: int = 0
    is_provisional: bool = False
    properties: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, category: str, name: str, data: Dict[str, Any]) -> "WorldItem":
        """Create a ``WorldItem`` from a raw dictionary."""

        # Get ID from data if present, otherwise generate one
        item_id = data.get("id", "")
        
        # Validate and normalize all core fields
        category, name, item_id = utils.validate_world_item_fields(category, name, item_id)

        created_chapter = int(data.get(KG_NODE_CREATED_CHAPTER, 0))
        is_provisional = bool(data.get(KG_IS_PROVISIONAL, False))

        props = {
            k: v
            for k, v in data.items()
            if k
            not in {
                "id",
                "category",
                "name",
                KG_NODE_CREATED_CHAPTER,
                KG_IS_PROVISIONAL,
            }
        }

        return cls(
            id=item_id,
            category=category,
            name=name,
            created_chapter=created_chapter,
            is_provisional=is_provisional,
            properties=props,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the item to a flat dictionary."""

        data = self.model_dump(exclude={"id", "category", "name"})
        properties_data = data.pop("properties", {})
        data.update(properties_data)
        return data

    def get_elaboration_summary(self, up_to_chapter: Optional[int] = None) -> List[str]:
        """Return elaboration notes up to a chapter number."""

        notes: List[str] = []
        prefix = "elaboration_in_chapter_"
        for key, val in sorted(self.properties.items()):
            if not key.startswith(prefix):
                continue
            try:
                chap = int(key.split("_")[-1])
            except (ValueError, IndexError):
                continue
            if up_to_chapter is None or chap <= up_to_chapter:
                if isinstance(val, str):
                    notes.append(val)
        return notes
