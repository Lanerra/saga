"""Neomodel node definitions for Saga."""

from __future__ import annotations

from neomodel import (
    AsyncStructuredNode,
    RelationshipTo,
    StringProperty,
    UniqueIdProperty,
)


class TraitNode(AsyncStructuredNode):
    """Represents a trait that a character may have."""

    name = StringProperty(unique_index=True, required=True)


class CharacterNode(AsyncStructuredNode):
    """Represents a character in the novel."""

    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    description = StringProperty()
    status = StringProperty()

    traits = RelationshipTo("TraitNode", "HAS_TRAIT")


class WorldElementNode(AsyncStructuredNode):
    """Represents a world element such as a location or item."""

    uid = UniqueIdProperty()
    identifier = StringProperty(unique_index=True, required=True)
    name = StringProperty()
    category = StringProperty()


__all__ = ["TraitNode", "CharacterNode", "WorldElementNode"]
