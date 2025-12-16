# core/langgraph/graph_context.py
"""
Neo4j context construction for LangGraph workflow.

This module wraps existing data_access queries for use in LangGraph nodes.
NO major refactoring needed - queries are already well-structured.

Migration Reference: docs/langgraph_migration_plan.md - Step 1.3.1

Source Code Referenced:
- data_access/character_queries.py:
  - get_characters_for_chapter_context_native() (lines 710-750)
  - get_character_profile_by_name() (lines 376+)
- data_access/world_queries.py:
  - get_world_items_for_chapter_context_native() (lines 752-792)
- data_access/chapter_queries.py:
  - get_chapter_content_batch_native() (lines 304-353)
- data_access/kg_queries.py:
  - Various relationship queries
- data_access/plot_queries.py:
  - get_plot_outline_from_db() (lines 233+)
"""

from __future__ import annotations

from typing import Any

import structlog

from core.db_manager import neo4j_manager
from data_access import chapter_queries, character_queries, world_queries
from models.kg_models import CharacterProfile

logger = structlog.get_logger(__name__)


async def build_context_from_graph(
    current_chapter: int,
    active_character_names: list[str] | None = None,
    location_id: str | None = None,
    lookback_chapters: int = 5,
    max_characters: int = 10,
    max_world_items: int = 10,
) -> dict[str, Any]:
    """
    Query Neo4j for narrative context to inform chapter generation.

    This function wraps existing data_access queries to build a comprehensive
    context object for prompt construction. It reuses well-tested query logic
    without modification.

    USES EXISTING QUERIES FROM: data_access/

    Args:
        current_chapter: Chapter number currently being processed
        active_character_names: Optional list of specific character names to include
        location_id: Optional ID of the current location
        lookback_chapters: Number of previous chapters to include (default: 5)
        max_characters: Maximum number of characters to retrieve (default: 10)
        max_world_items: Maximum number of world items to retrieve (default: 10)

    Returns:
        Dictionary containing structured context:
        {
            "characters": List[CharacterProfile],
            "world_items": List[WorldItem],
            "relationships": List[Dict],
            "recent_summaries": List[Dict],
            "location": Optional[Dict],
        }
    """
    logger.info(
        "build_context_from_graph",
        chapter=current_chapter,
        lookback=lookback_chapters,
        char_count=len(active_character_names) if active_character_names else "auto",
    )

    context: dict[str, Any] = {
        "characters": [],
        "world_items": [],
        "relationships": [],
        "recent_summaries": [],
        "location": None,
    }

    try:
        # 1. Get character details
        # REUSES: character_queries.get_characters_for_chapter_context_native()
        if active_character_names:
            # Get specific characters by name
            characters = await _get_characters_by_names(active_character_names)
        else:
            # Get contextually relevant characters
            characters = await character_queries.get_characters_for_chapter_context_native(
                chapter_number=current_chapter,
                limit=max_characters,
            )
        context["characters"] = characters

        # 2. Get character relationships
        # REUSES: Custom query based on kg_queries patterns
        if characters:
            char_names = [c.name for c in characters]
            relationships = await _get_character_relationships(char_names)
            context["relationships"] = relationships

        # 3. Get world items for context
        # REUSES: world_queries.get_world_items_for_chapter_context_native()
        world_items = await world_queries.get_world_items_for_chapter_context_native(
            chapter_number=current_chapter,
            limit=max_world_items,
        )
        context["world_items"] = world_items

        # 4. Get recent chapter summaries
        # REUSES: chapter_queries.get_chapter_content_batch_native()
        summaries = await _get_recent_summaries(
            current_chapter=current_chapter,
            lookback_chapters=lookback_chapters,
        )
        context["recent_summaries"] = summaries

        # 5. Get location details if specified
        # REUSES: Neo4j query patterns from world_queries
        if location_id:
            location = await _get_location_details(location_id)
            context["location"] = location

        logger.info(
            "build_context_from_graph: context built",
            characters=len(context["characters"]),
            world_items=len(context["world_items"]),
            relationships=len(context["relationships"]),
            summaries=len(context["recent_summaries"]),
        )

        return context

    except Exception as e:
        logger.error(
            "build_context_from_graph: error building context",
            error=str(e),
            chapter=current_chapter,
            exc_info=True,
        )
        # Return empty context on error to avoid breaking the workflow
        return context


async def _get_characters_by_names(
    character_names: list[str],
) -> list[CharacterProfile]:
    """
    Get specific characters by their names.

    REUSES: character_queries.get_character_profile_by_name()

    Args:
        character_names: List of character names to retrieve

    Returns:
        List of CharacterProfile models
    """
    characters = []

    for name in character_names:
        try:
            char = await character_queries.get_character_profile_by_name(name)
            if char:
                characters.append(char)
        except Exception as e:
            logger.warning(
                "_get_characters_by_names: failed to get character",
                name=name,
                error=str(e),
            )

    return characters


async def _get_character_relationships(
    character_names: list[str],
) -> list[dict[str, Any]]:
    """
    Get relationships between characters.

    USES: Custom query based on kg_queries.py patterns

    Args:
        character_names: List of character names to find relationships for

    Returns:
        List of relationship dictionaries with keys:
        - source: Source character name
        - target: Target character name
        - rel_type: Relationship type
        - description: Relationship description
    """
    if not character_names:
        return []

    # Query for relationships involving any of the specified characters
    query = """
    MATCH (c1:Character)-[r]->(c2:Character)
    WHERE c1.name IN $char_names OR c2.name IN $char_names
    RETURN c1.name AS source,
           type(r) AS rel_type,
           c2.name AS target,
           coalesce(r.description, '') AS description,
           coalesce(r.confidence, 0.8) AS confidence
    ORDER BY confidence DESC
    LIMIT 50
    """

    try:
        results = await neo4j_manager.execute_read_query(query, {"char_names": character_names})

        relationships = []
        for record in results:
            relationships.append(
                {
                    "source": record.get("source", ""),
                    "target": record.get("target", ""),
                    "rel_type": record.get("rel_type", "RELATES_TO"),
                    "description": record.get("description", ""),
                    "confidence": record.get("confidence", 0.8),
                }
            )

        logger.debug(
            "_get_character_relationships: retrieved relationships",
            count=len(relationships),
        )

        return relationships

    except Exception as e:
        logger.error(
            "_get_character_relationships: error retrieving relationships",
            error=str(e),
            exc_info=True,
        )
        return []


async def _get_recent_summaries(
    current_chapter: int,
    lookback_chapters: int,
) -> list[dict[str, Any]]:
    """
    Get summaries of recent chapters for context.

    REUSES: chapter_queries.get_chapter_content_batch_native()

    Args:
        current_chapter: Current chapter being processed
        lookback_chapters: Number of previous chapters to retrieve

    Returns:
        List of summary dictionaries with keys:
        - chapter: Chapter number
        - summary: Chapter summary text
    """
    # Calculate chapter range
    start_chapter = max(1, current_chapter - lookback_chapters)
    chapter_numbers = list(range(start_chapter, current_chapter))

    if not chapter_numbers:
        return []

    try:
        # Use existing batch retrieval function
        chapter_data = await chapter_queries.get_chapter_content_batch_native(chapter_numbers)

        # Extract summaries
        summaries = []
        for chapter_num in sorted(chapter_numbers):
            if chapter_num in chapter_data:
                data = chapter_data[chapter_num]
                if data.get("summary"):
                    summaries.append(
                        {
                            "chapter": chapter_num,
                            "summary": data["summary"],
                        }
                    )

        logger.debug(
            "_get_recent_summaries: retrieved summaries",
            count=len(summaries),
            range=f"{start_chapter}-{current_chapter-1}",
        )

        return summaries

    except Exception as e:
        logger.error(
            "_get_recent_summaries: error retrieving summaries",
            error=str(e),
            exc_info=True,
        )
        return []


async def _get_location_details(location_id: str) -> dict[str, Any] | None:
    """
    Get details about a specific location.

    USES: Custom query based on world_queries patterns

    Args:
        location_id: ID of the location to retrieve

    Returns:
        Dictionary with location details or None if not found
    """
    query = """
    MATCH (l)
    WHERE l.id = $loc_id
      AND (l:Location OR l:Structure OR l:Settlement OR l:Region)
    RETURN l.id AS id,
           l.name AS name,
           l.description AS description,
           l.rules AS rules,
           l.category AS category
    LIMIT 1
    """

    try:
        results = await neo4j_manager.execute_read_query(query, {"loc_id": location_id})

        if results and len(results) > 0:
            record = results[0]
            location = {
                "id": record.get("id", ""),
                "name": record.get("name", ""),
                "description": record.get("description", ""),
                "rules": record.get("rules", []),
                "category": record.get("category", ""),
            }

            logger.debug(
                "_get_location_details: retrieved location",
                location_id=location_id,
                name=location.get("name"),
            )

            return location

        logger.debug(
            "_get_location_details: location not found",
            location_id=location_id,
        )
        return None

    except Exception as e:
        logger.error(
            "_get_location_details: error retrieving location",
            error=str(e),
            location_id=location_id,
            exc_info=True,
        )
        return None


async def get_key_events(
    current_chapter: int,
    lookback_chapters: int = 10,
    max_events: int = 20,
) -> list[dict[str, Any]]:
    """
    Get key events from recent chapters for context.

    USES: Custom query based on plot_queries patterns

    This function can be used by generation nodes to include important
    plot events in the prompt context.

    Args:
        current_chapter: Current chapter being processed
        lookback_chapters: Number of chapters to look back (default: 10)
        max_events: Maximum number of events to retrieve (default: 20)

    Returns:
        List of event dictionaries with keys:
        - description: Event description
        - chapter: Chapter where event occurred
        - importance: Importance score (if available)
    """
    # For the first chapter, there are no previous events
    if current_chapter <= 1:
        logger.debug(
            "get_key_events: skipping for first chapter",
            chapter=current_chapter,
        )
        return []

    start_chapter = max(1, current_chapter - lookback_chapters)

    query = """
    MATCH (e:Event)-[:OCCURRED_IN]->(ch:Chapter)
    WHERE ch.number >= $start_chapter AND ch.number < $current_chapter
    RETURN e.description AS description,
           coalesce(e.importance, 0.5) AS importance,
           ch.number AS chapter
    ORDER BY importance DESC, ch.number DESC
    LIMIT $max_events
    """

    try:
        results = await neo4j_manager.execute_read_query(
            query,
            {
                "start_chapter": start_chapter,
                "current_chapter": current_chapter,
                "max_events": max_events,
            },
        )

        events = []
        for record in results:
            events.append(
                {
                    "description": record.get("description", ""),
                    "chapter": record.get("chapter", 0),
                    "importance": record.get("importance", 0.5),
                }
            )

        logger.debug(
            "get_key_events: retrieved events",
            count=len(events),
            range=f"{start_chapter}-{current_chapter-1}",
        )

        return events

    except Exception as e:
        logger.warning(
            "get_key_events: error retrieving events (returning empty list)",
            error=str(e),
            chapter=current_chapter,
            start_chapter=start_chapter,
            exc_info=True,
        )
        # Gracefully degrade - return empty list so workflow continues
        return []


__all__ = ["build_context_from_graph", "get_key_events"]
