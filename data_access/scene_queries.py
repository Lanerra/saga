# data_access/scene_queries.py
"""Query functions for Scene and Event nodes in Neo4j.

This module provides functions to query scene-related data from the knowledge graph
for use in narrative generation context building.
"""

from typing import Any

import structlog

from core.db_manager import neo4j_manager
from core.exceptions import DatabaseError

logger = structlog.get_logger(__name__)


async def get_scene_events(
    chapter_number: int,
    scene_index: int,
) -> list[dict[str, Any]]:
    """Get all SceneEvent nodes for a specific scene.

    Args:
        chapter_number: Chapter number
        scene_index: Scene index within the chapter

    Returns:
        List of event dictionaries with keys:
        - name: Event name
        - description: Event description
        - conflict: Event conflict
        - outcome: Event outcome
        - pov_character: POV character name
        - characters_involved: List of character names involved

    Raises:
        DatabaseError: If there's an error querying Neo4j
    """
    query = """
    MATCH (s:Scene {chapter_number: $chapter_number, scene_index: $scene_index})
    OPTIONAL MATCH (s)<-[:OCCURS_IN_SCENE]-(e:Event {event_type: "SceneEvent"})
    OPTIONAL MATCH (e)-[:INVOLVES]->(c:Character)
    WITH e, s, collect(DISTINCT c.name) as characters
    WHERE e IS NOT NULL
    RETURN
        e.name as name,
        e.description as description,
        e.conflict as conflict,
        e.outcome as outcome,
        e.pov_character as pov_character,
        characters as characters_involved
    ORDER BY e.scene_index
    """

    try:
        params = {
            "chapter_number": chapter_number,
            "scene_index": scene_index,
        }
        records = await neo4j_manager.execute_read_query(query, params)

        logger.debug(
            "get_scene_events: fetched events",
            chapter=chapter_number,
            scene_index=scene_index,
            count=len(records),
        )

        return records

    except Exception as e:
        logger.error(
            "get_scene_events: error fetching scene events",
            chapter=chapter_number,
            scene_index=scene_index,
            error=str(e),
            exc_info=True,
        )
        raise DatabaseError(f"Failed to fetch scene events: {str(e)}") from e


async def get_act_events(
    act_number: int,
) -> dict[str, Any]:
    """Get event hierarchy for an act (MajorPlotPoint and ActKeyEvent nodes).

    Args:
        act_number: Act number (1, 2, or 3)

    Returns:
        Dictionary with keys:
        - major_plot_points: List of MajorPlotPoint events
        - act_key_events: List of ActKeyEvent events
        Each event dict includes:
        - name: Event name
        - description: Event description
        - sequence_order/sequence_in_act: Ordering within act
        - cause/effect: For ActKeyEvents
        - characters_involved: List of character names
        - location: Location name if applicable

    Raises:
        DatabaseError: If there's an error querying Neo4j
    """
    major_query = """
    MATCH (major:Event {event_type: "MajorPlotPoint"})
    OPTIONAL MATCH (major)-[:INVOLVES]->(c:Character)
    WITH major, collect(DISTINCT c.name) as characters_involved
    ORDER BY major.sequence_order
    RETURN collect({
        name: major.name,
        description: major.description,
        sequence_order: major.sequence_order
    }) as major_points
    """

    act_query = """
    MATCH (act:Event {event_type: "ActKeyEvent", act_number: $act_number})
    OPTIONAL MATCH (act)-[:INVOLVES]->(c:Character)
    OPTIONAL MATCH (act)-[:OCCURS_AT]->(loc:Location)
    OPTIONAL MATCH (act)-[:PART_OF]->(parent:Event)
    WITH act, collect(DISTINCT c.name) as act_characters, loc.name as location, parent.name as parent_event
    ORDER BY act.sequence_in_act
    RETURN collect({
        name: act.name,
        description: act.description,
        sequence_in_act: act.sequence_in_act,
        cause: act.cause,
        effect: act.effect,
        characters_involved: act_characters,
        location: location,
        part_of: parent_event
    }) as act_events
    """

    try:
        params = {"act_number": act_number}
        major_records = await neo4j_manager.execute_read_query(major_query)
        act_records = await neo4j_manager.execute_read_query(act_query, params)

        major_points = major_records[0].get("major_points", []) if major_records else []
        act_events = act_records[0].get("act_events", []) if act_records else []

        logger.debug(
            "get_act_events: fetched events",
            act_number=act_number,
            major_points=len(major_points),
            act_events=len(act_events),
        )

        return {
            "major_plot_points": major_points,
            "act_key_events": act_events,
        }

    except Exception as e:
        logger.error(
            "get_act_events: error fetching act events",
            act_number=act_number,
            error=str(e),
            exc_info=True,
        )
        raise DatabaseError(f"Failed to fetch act events: {str(e)}") from e


async def get_character_relationships_for_scene(
    character_names: list[str],
    chapter_limit: int,
) -> list[dict[str, Any]]:
    """Get relationships between characters that are relevant for a scene.

    Args:
        character_names: List of character names in the scene
        chapter_limit: Maximum chapter number to consider (for temporal filtering)

    Returns:
        List of relationship dictionaries with keys:
        - source: Source character name
        - relationship_type: Type of relationship
        - target: Target character name
        - description: Relationship description
        - chapter_added: When the relationship was added

    Raises:
        DatabaseError: If there's an error querying Neo4j
    """
    if not character_names:
        return []

    query = """
    MATCH (c1:Character)-[r]->(c2:Character)
    WHERE c1.name IN $character_names
      AND c2.name IN $character_names
      AND r.chapter_added <= $chapter_limit
    RETURN
        c1.name as source,
        type(r) as relationship_type,
        c2.name as target,
        r.description as description,
        r.chapter_added as chapter_added
    ORDER BY r.chapter_added
    """

    try:
        params = {
            "character_names": character_names,
            "chapter_limit": chapter_limit,
        }
        records = await neo4j_manager.execute_read_query(query, params)

        logger.debug(
            "get_character_relationships_for_scene: fetched relationships",
            characters=len(character_names),
            relationships=len(records),
        )

        return records

    except Exception as e:
        logger.error(
            "get_character_relationships_for_scene: error fetching relationships",
            characters=character_names,
            error=str(e),
            exc_info=True,
        )
        raise DatabaseError(f"Failed to fetch character relationships: {str(e)}") from e


async def get_character_items(
    character_names: list[str],
    chapter_limit: int,
) -> list[dict[str, Any]]:
    """Get items possessed by characters.

    Args:
        character_names: List of character names
        chapter_limit: Maximum chapter number to consider

    Returns:
        List of item dictionaries with keys:
        - character_name: Character who possesses the item
        - item_name: Item name
        - item_description: Item description
        - item_category: Item category
        - acquired_chapter: When the character acquired the item

    Raises:
        DatabaseError: If there's an error querying Neo4j
    """
    if not character_names:
        return []

    query = """
    MATCH (c:Character)-[r:POSSESSES]->(i:Item)
    WHERE c.name IN $character_names
      AND r.acquired_chapter <= $chapter_limit
    RETURN
        c.name as character_name,
        i.name as item_name,
        i.description as item_description,
        i.category as item_category,
        r.acquired_chapter as acquired_chapter
    ORDER BY r.acquired_chapter
    """

    try:
        params = {
            "character_names": character_names,
            "chapter_limit": chapter_limit,
        }
        records = await neo4j_manager.execute_read_query(query, params)

        logger.debug(
            "get_character_items: fetched items",
            characters=len(character_names),
            items=len(records),
        )

        return records

    except Exception as e:
        logger.error(
            "get_character_items: error fetching items",
            characters=character_names,
            error=str(e),
            exc_info=True,
        )
        raise DatabaseError(f"Failed to fetch character items: {str(e)}") from e


async def get_scene_items(
    chapter_number: int,
    scene_index: int,
) -> list[dict[str, Any]]:
    """Get items featured in a specific scene.

    Args:
        chapter_number: Chapter number
        scene_index: Scene index within the chapter

    Returns:
        List of item dictionaries with keys:
        - item_name: Item name
        - item_description: Item description
        - item_category: Item category

    Raises:
        DatabaseError: If there's an error querying Neo4j
    """
    query = """
    MATCH (s:Scene {chapter_number: $chapter_number, scene_index: $scene_index})
    MATCH (s)-[:FEATURES_ITEM]->(i:Item)
    RETURN
        i.name as item_name,
        i.description as item_description,
        i.category as item_category
    """

    try:
        params = {
            "chapter_number": chapter_number,
            "scene_index": scene_index,
        }
        records = await neo4j_manager.execute_read_query(query, params)

        logger.debug(
            "get_scene_items: fetched items",
            chapter=chapter_number,
            scene_index=scene_index,
            items=len(records),
        )

        return records

    except Exception as e:
        logger.error(
            "get_scene_items: error fetching scene items",
            chapter=chapter_number,
            scene_index=scene_index,
            error=str(e),
            exc_info=True,
        )
        raise DatabaseError(f"Failed to fetch scene items: {str(e)}") from e


__all__ = [
    "get_scene_events",
    "get_act_events",
    "get_character_relationships_for_scene",
    "get_character_items",
    "get_scene_items",
]
