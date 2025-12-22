# core/langgraph/graph_context.py
"""
Build Neo4j-derived prompt context for LangGraph nodes.

Migration Reference: docs/langgraph_migration_plan.md - Step 1.3.1

This module is a thin adapter over existing `data_access` query functions. It
returns a compact context payload used to construct drafting prompts.

Notes:
- On query failures, context construction degrades gracefully by returning an
  empty/default context rather than raising. This keeps the workflow runnable
  when context is unavailable.
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
    """Build prompt context for a chapter by querying Neo4j.

    Args:
        current_chapter: Chapter number being generated.
        active_character_names: Optional allowlist of character names to include.
            When omitted, the query layer selects contextually relevant characters.
        location_id: Optional location node id to enrich context with location details.
        lookback_chapters: Number of prior chapters whose summaries may be included.
        max_characters: Maximum number of character profiles to include.
        max_world_items: Maximum number of world items to include.

    Returns:
        A context mapping with stable keys:
        - characters: Character profiles used for drafting context.
        - world_items: World items used for drafting context.
        - relationships: Relationship rows between characters (best-effort).
        - recent_summaries: Summary rows for recent chapters (best-effort).
        - location: Location details when `location_id` is provided (best-effort).

    Notes:
        This function degrades gracefully. On any query error it logs and returns
        an empty/default context rather than raising.
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
    """Load character profiles by name (best-effort).

    Args:
        character_names: Names to retrieve.

    Returns:
        Character profiles that were successfully loaded. Missing names are
        skipped (errors are logged).
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
    """Query relationships involving the provided character names.

    Args:
        character_names: Character names to match as relationship endpoints.

    Returns:
        Relationship rows with stable keys: `source`, `target`, `rel_type`,
        `description`, `confidence`.
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
    """Load recent chapter summaries for prompt context.

    Args:
        current_chapter: Chapter being generated.
        lookback_chapters: Number of previous chapters to include.

    Returns:
        Summary rows with keys `chapter` and `summary`.
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
    """Load location details by node id (best-effort)."""
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
    """Load key events from recent chapters for prompt context.

    Args:
        current_chapter: Chapter being generated.
        lookback_chapters: Number of prior chapters to search.
        max_events: Maximum number of events to return.

    Returns:
        Event rows with stable keys `description`, `chapter`, `importance`.

    Notes:
        This function degrades gracefully. On query failures it logs and returns
        an empty list rather than raising.
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
