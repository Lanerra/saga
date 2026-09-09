"""Plot and relationship retrieval for scene-based context.

Provides:
- Scene events
- Character relationships
- Character items/possessions
- Act-level plot structure (major plot points, key events)
"""

import structlog

from core.langgraph.content_manager import ContentManager
from core.langgraph.state import NarrativeState
from data_access import scene_queries

logger = structlog.get_logger(__name__)


async def get_scene_events_context(
    state: NarrativeState,
    chapter_number: int,
    scene_index: int,
    content_manager: ContentManager,
) -> str | None:
    """Get scene events context from Neo4j.

    Args:
        state: Workflow state.
        chapter_number: Current chapter number.
        scene_index: Current scene index.
        content_manager: Content manager instance.

    Returns:
        Formatted scene events context or None.
    """
    try:
        events = await scene_queries.get_scene_events(
            chapter_number=chapter_number,
            scene_index=scene_index,
        )

        if not events:
            return None

        events_text = "**Scene Events:**\n"
        for event in events:
            events_text += f"\n- **{event.get('name', 'Unnamed Event')}**: {event.get('description', '')}"
            if event.get("conflict"):
                events_text += f"\n  - Conflict: {event['conflict']}"
            if event.get("outcome"):
                events_text += f"\n  - Outcome: {event['outcome']}"
            if event.get("characters_involved"):
                chars = ", ".join(event["characters_involved"])
                events_text += f"\n  - Characters: {chars}"

        return events_text

    except Exception as e:
        logger.warning(
            "context_plot: non-fatal error getting scene events, continuing without them",
            chapter=chapter_number,
            scene_index=scene_index,
            error=str(e),
        )
        return None


async def get_character_relationships_context(
    state: NarrativeState,
    character_names: list[str],
    chapter_number: int,
    content_manager: ContentManager,
) -> str | None:
    """Get character relationships context from Neo4j.

    Args:
        state: Workflow state.
        character_names: List of character names in the scene.
        chapter_number: Current chapter number.
        content_manager: Content manager instance.

    Returns:
        Formatted character relationships context or None.
    """
    try:
        relationships = await scene_queries.get_character_relationships_for_scene(
            character_names=character_names,
            chapter_limit=chapter_number - 1 if chapter_number > 1 else 0,
        )

        if not relationships:
            return None

        relationships_text = "**Character Relationships:**\n"
        for rel in relationships:
            rel_type = rel.get("relationship_type", "").replace("_", " ").lower()
            source = rel.get("source", "")
            target = rel.get("target", "")
            description = rel.get("description", "")

            relationships_text += f"\n- {source} {rel_type} {target}"
            if description:
                relationships_text += f": {description}"

        return relationships_text

    except Exception as e:
        logger.warning(
            "context_plot: non-fatal error getting character relationships, continuing without them",
            characters=character_names,
            error=str(e),
        )
        return None


async def get_character_items_context(
    state: NarrativeState,
    character_names: list[str],
    chapter_number: int,
    content_manager: ContentManager,
) -> str | None:
    """Get character items/possessions context from Neo4j.

    Args:
        state: Workflow state.
        character_names: List of character names in the scene.
        chapter_number: Current chapter number.
        content_manager: Content manager instance.

    Returns:
        Formatted character items context or None.
    """
    try:
        items = await scene_queries.get_character_items(
            character_names=character_names,
            chapter_limit=chapter_number - 1 if chapter_number > 1 else 0,
        )

        if not items:
            return None

        items_text = "**Character Possessions:**\n"
        by_character: dict[str, list[dict]] = {}
        for item in items:
            char_name = item.get("character_name", "")
            if char_name not in by_character:
                by_character[char_name] = []
            by_character[char_name].append(item)

        for char_name, char_items in by_character.items():
            items_text += f"\n- {char_name}:"
            for item in char_items:
                item_name = item.get("item_name", "")
                item_desc = item.get("item_description", "")
                items_text += f"\n  - {item_name}"
                if item_desc:
                    items_text += f": {item_desc}"

        return items_text

    except Exception as e:
        logger.warning(
            "context_plot: non-fatal error getting character items, continuing without them",
            characters=character_names,
            error=str(e),
        )
        return None


async def get_act_events_context(
    state: NarrativeState,
    act_number: int,
    content_manager: ContentManager,
) -> str | None:
    """Get act events context from Neo4j.

    Args:
        state: Workflow state.
        act_number: Act number (1, 2, or 3).
        content_manager: Content manager instance.

    Returns:
        Formatted act events context or None.
    """
    try:
        events_data = await scene_queries.get_act_events(act_number=act_number)

        major_points = events_data.get("major_plot_points", [])
        act_events = events_data.get("act_key_events", [])

        if not major_points and not act_events:
            return None

        context_text = f"**Act {act_number} Plot Structure:**\n"

        if major_points:
            context_text += "\nMajor Plot Points:\n"
            for point in sorted(major_points, key=lambda x: x.get("sequence_order", 0)):
                context_text += f"- {point.get('name', '')}: {point.get('description', '')}\n"

        if act_events:
            context_text += f"\nKey Events in Act {act_number}:\n"
            for event in sorted(act_events, key=lambda x: x.get("sequence_in_act", 0)):
                context_text += f"- {event.get('name', '')}: {event.get('description', '')}\n"
                if event.get("cause"):
                    context_text += f"  - Cause: {event['cause']}\n"
                if event.get("effect"):
                    context_text += f"  - Effect: {event['effect']}\n"

        return context_text

    except Exception as e:
        logger.warning(
            "context_plot: non-fatal error getting act events, continuing without them",
            act_number=act_number,
            error=str(e),
        )
        return None
