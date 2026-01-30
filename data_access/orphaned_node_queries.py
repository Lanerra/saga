"""Query patterns for detecting and cleaning up orphaned nodes in the knowledge graph.

This module provides Cypher queries to identify nodes that are not properly connected
to the rest of the graph according to SAGA's schema design.

Orphaned Node Definition:
    A node is considered orphaned if it:
    1. Has no relationships to other nodes (except self-references)
    2. Has no meaningful properties (empty or null values)
    3. Is not referenced by any other part of the graph
    4. Does not participate in the expected relationship patterns defined in schema-design.md

According to schema-design.md:
- Characters must have relationships to other characters
- Events must have relationships to characters, locations, and other events
- Scenes must have relationships to chapters, characters, locations, and items
- Chapters must have relationships to scenes
- Items must have relationships to characters and events/scenes
"""

from typing import Any

# -------------------------------------------------------------------------
# Character Orphan Detection
# -------------------------------------------------------------------------


async def detect_orphaned_characters() -> list[dict[str, Any]]:
    """Find Character nodes that have no relationships to other characters.

    According to schema-design.md, characters should have social/emotional relationships
    with other characters. An orphaned character has no such relationships.

    Returns:
        List of character nodes that are orphaned (no relationships to other characters)
    """
    query = """
    MATCH (c:Character)
    WHERE NOT EXISTS {
        MATCH (c)-[r]->(other:Character)
        WHERE other <> c
    }
    RETURN c
    """
    return query


async def cleanup_orphaned_characters() -> int:
    """Remove Character nodes that have no relationships to other characters.

    This is a destructive operation that should only be used in testing/validation.

    Returns:
        Number of orphaned characters removed
    """
    query = """
    MATCH (c:Character)
    WHERE NOT EXISTS {
        MATCH (c)-[r]->(other:Character)
        WHERE other <> c
    }
    DETACH DELETE c
    """
    return query


# -------------------------------------------------------------------------
# Event Orphan Detection
# -------------------------------------------------------------------------


async def detect_orphaned_events() -> list[dict[str, Any]]:
    """Find Event nodes that have no relationships to characters, locations, or other events.

    According to schema-design.md, events should have:
    - PART_OF relationships to other events
    - HAPPENS_BEFORE relationships to other events
    - INVOLVES relationships to characters
    - OCCURS_AT relationships to locations
    - OCCURS_IN_SCENE relationships to scenes

    Returns:
        List of event nodes that are orphaned (no relationships)
    """
    query = """
    MATCH (e:Event)
    WHERE NOT EXISTS {
        MATCH (e)-[r]->(other)
        WHERE other <> e
    }
    RETURN e
    """
    return query


async def cleanup_orphaned_events() -> int:
    """Remove Event nodes that have no relationships to other nodes.

    This is a destructive operation that should only be used in testing/validation.

    Returns:
        Number of orphaned events removed
    """
    query = """
    MATCH (e:Event)
    WHERE NOT EXISTS {
        MATCH (e)-[r]->(other)
        WHERE other <> e
    }
    DETACH DELETE e
    """
    return query


# -------------------------------------------------------------------------
# Location Orphan Detection
# -------------------------------------------------------------------------


async def detect_orphaned_locations() -> list[dict[str, Any]]:
    """Find Location nodes that have no relationships to events or scenes.

    According to schema-design.md, locations should have:
    - OCCURS_AT relationships from events
    - OCCURS_AT relationships from scenes

    Returns:
        List of location nodes that are orphaned (no relationships)
    """
    query = """
    MATCH (l:Location)
    WHERE NOT EXISTS {
        MATCH (l)<-[r]-(other)
        WHERE other <> l
    }
    RETURN l
    """
    return query


async def cleanup_orphaned_locations() -> int:
    """Remove Location nodes that have no relationships to other nodes.

    This is a destructive operation that should only be used in testing/validation.

    Returns:
        Number of orphaned locations removed
    """
    query = """
    MATCH (l:Location)
    WHERE NOT EXISTS {
        MATCH (l)<-[r]-(other)
        WHERE other <> l
    }
    DETACH DELETE l
    """
    return query


# -------------------------------------------------------------------------
# Item Orphan Detection
# -------------------------------------------------------------------------


async def detect_orphaned_items() -> list[dict[str, Any]]:
    """Find Item nodes that have no relationships to characters or events/scenes.

    According to schema-design.md, items should have:
    - POSSESSES relationships from characters
    - FEATURES_ITEM relationships from events
    - FEATURES_ITEM relationships from scenes

    Returns:
        List of item nodes that are orphaned (no relationships)
    """
    query = """
    MATCH (i:Item)
    WHERE NOT EXISTS {
        MATCH (i)<-[r]-(other)
        WHERE other <> i
    }
    RETURN i
    """
    return query


async def cleanup_orphaned_items() -> int:
    """Remove Item nodes that have no relationships to other nodes.

    This is a destructive operation that should only be used in testing/validation.

    Returns:
        Number of orphaned items removed
    """
    query = """
    MATCH (i:Item)
    WHERE NOT EXISTS {
        MATCH (i)<-[r]-(other)
        WHERE other <> i
    }
    DETACH DELETE i
    """
    return query


# -------------------------------------------------------------------------
# Scene Orphan Detection
# -------------------------------------------------------------------------


async def detect_orphaned_scenes() -> list[dict[str, Any]]:
    """Find Scene nodes that have no relationships to chapters or characters.

    According to schema-design.md, scenes should have:
    - PART_OF relationships to chapters
    - FEATURES_CHARACTER relationships to characters
    - OCCURS_AT relationships to locations
    - FEATURES_ITEM relationships to items
    - FOLLOWS relationships to other scenes

    Returns:
        List of scene nodes that are orphaned (no relationships)
    """
    query = """
    MATCH (s:Scene)
    WHERE NOT EXISTS {
        MATCH (s)-[r]->(other)
        WHERE other <> s
    }
    RETURN s
    """
    return query


async def cleanup_orphaned_scenes() -> int:
    """Remove Scene nodes that have no relationships to other nodes.

    This is a destructive operation that should only be used in testing/validation.

    Returns:
        Number of orphaned scenes removed
    """
    query = """
    MATCH (s:Scene)
    WHERE NOT EXISTS {
        MATCH (s)-[r]->(other)
        WHERE other <> s
    }
    DETACH DELETE s
    """
    return query


# -------------------------------------------------------------------------
# Chapter Orphan Detection
# -------------------------------------------------------------------------


async def detect_orphaned_chapters() -> list[dict[str, Any]]:
    """Find Chapter nodes that have no relationships to scenes.

    According to schema-design.md, chapters should have:
    - PART_OF relationships from scenes

    Returns:
        List of chapter nodes that are orphaned (no relationships)
    """
    query = """
    MATCH (c:Chapter)
    WHERE NOT EXISTS {
        MATCH (c)<-[r]-(other:Scene)
        WHERE other <> c
    }
    RETURN c
    """
    return query


async def cleanup_orphaned_chapters() -> int:
    """Remove Chapter nodes that have no relationships to other nodes.

    This is a destructive operation that should only be used in testing/validation.

    Returns:
        Number of orphaned chapters removed
    """
    query = """
    MATCH (c:Chapter)
    WHERE NOT EXISTS {
        MATCH (c)<-[r]-(other:Scene)
        WHERE other <> c
    }
    DETACH DELETE c
    """
    return query
