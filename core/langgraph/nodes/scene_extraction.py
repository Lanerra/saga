"""Extract entities from individual scenes instead of full chapters.

This module provides scene-level extraction to reduce prompt sizes and improve
extraction quality by processing smaller text chunks (~5-10K chars each).
"""

from __future__ import annotations

from typing import Any

import structlog

import config
from core.exceptions import LLMServiceError
from core.llm_interface_refactored import llm_service
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)


async def extract_from_scene(
    scene_text: str,
    scene_index: int,
    chapter_number: int,
    novel_title: str,
    novel_genre: str,
    protagonist_name: str,
    model_name: str,
) -> dict[str, Any]:
    """Extract entities and relationships from a single scene.

    Args:
        scene_text: The text of the scene to extract from.
        scene_index: The index of the scene within the chapter.
        chapter_number: The chapter number this scene belongs to.
        novel_title: The title of the novel.
        novel_genre: The genre of the novel.
        protagonist_name: The protagonist's name.
        model_name: The LLM model name to use for extraction.

    Returns:
        Dictionary with keys:
        - characters: List of character entity dicts with scene_index.
        - world_items: List of location and event entity dicts with scene_index.
        - relationships: List of relationship dicts with scene_index.
    """
    logger.info(
        "extract_from_scene: starting",
        scene_index=scene_index,
        chapter=chapter_number,
        scene_text_length=len(scene_text),
    )

    characters = await _extract_characters_from_scene(
        scene_text=scene_text,
        scene_index=scene_index,
        chapter_number=chapter_number,
        novel_title=novel_title,
        novel_genre=novel_genre,
        protagonist_name=protagonist_name,
        model_name=model_name,
    )

    locations = await _extract_locations_from_scene(
        scene_text=scene_text,
        scene_index=scene_index,
        chapter_number=chapter_number,
        novel_title=novel_title,
        novel_genre=novel_genre,
        protagonist_name=protagonist_name,
        model_name=model_name,
    )

    events = await _extract_events_from_scene(
        scene_text=scene_text,
        scene_index=scene_index,
        chapter_number=chapter_number,
        novel_title=novel_title,
        novel_genre=novel_genre,
        protagonist_name=protagonist_name,
        model_name=model_name,
    )

    relationships = await _extract_relationships_from_scene(
        scene_text=scene_text,
        scene_index=scene_index,
        chapter_number=chapter_number,
        novel_title=novel_title,
        novel_genre=novel_genre,
        protagonist_name=protagonist_name,
        model_name=model_name,
    )

    world_items = locations + events

    logger.info(
        "extract_from_scene: complete",
        scene_index=scene_index,
        chapter=chapter_number,
        characters_count=len(characters),
        locations_count=len(locations),
        events_count=len(events),
        relationships_count=len(relationships),
    )

    return {
        "characters": characters,
        "world_items": world_items,
        "relationships": relationships,
    }


async def _extract_characters_from_scene(
    scene_text: str,
    scene_index: int,
    chapter_number: int,
    novel_title: str,
    novel_genre: str,
    protagonist_name: str,
    model_name: str,
) -> list[dict[str, Any]]:
    """Extract characters from scene text.

    Args:
        scene_text: The text of the scene to extract from.
        scene_index: The index of the scene within the chapter.
        chapter_number: The chapter number this scene belongs to.
        novel_title: The title of the novel.
        novel_genre: The genre of the novel.
        protagonist_name: The protagonist's name.
        model_name: The LLM model name to use for extraction.

    Returns:
        List of character entity dicts with scene_index field.
    """
    prompt = render_prompt(
        "knowledge_agent/extract_characters.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "protagonist": protagonist_name,
            "chapter_number": chapter_number,
            "novel_title": novel_title,
            "novel_genre": novel_genre,
            "chapter_text": scene_text,
        },
    )

    try:
        data, _ = await llm_service.async_call_llm_json_object(
            model_name=model_name,
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            system_prompt=get_system_prompt("knowledge_agent"),
            max_attempts=2,
        )

        character_updates = data.get("character_updates", {})
        if not isinstance(character_updates, dict):
            logger.warning(
                "_extract_characters_from_scene: character_updates is not a dict",
                scene_index=scene_index,
                chapter=chapter_number,
            )
            return []

        characters: list[dict[str, Any]] = []
        for name, info in character_updates.items():
            if not isinstance(info, dict):
                continue

            characters.append(
                {
                    "name": str(name),
                    "type": "Character",
                    "description": str(info.get("description", "")),
                    "first_appearance_chapter": chapter_number,
                    "scene_index": scene_index,
                    "attributes": {
                        "traits": list(info.get("traits", [])),
                        "status": str(info.get("status", "")),
                        "relationships": dict(info.get("relationships", {})),
                    },
                }
            )

        logger.debug(
            "_extract_characters_from_scene: extracted characters",
            scene_index=scene_index,
            chapter=chapter_number,
            count=len(characters),
        )

        return characters

    except LLMServiceError as e:
        logger.warning(
            "_extract_characters_from_scene: LLM failure",
            scene_index=scene_index,
            chapter=chapter_number,
            error=str(e),
        )
        return []
    except Exception as e:
        logger.warning(
            "_extract_characters_from_scene: failed",
            scene_index=scene_index,
            chapter=chapter_number,
            error=str(e),
        )
        return []


async def _extract_locations_from_scene(
    scene_text: str,
    scene_index: int,
    chapter_number: int,
    novel_title: str,
    novel_genre: str,
    protagonist_name: str,
    model_name: str,
) -> list[dict[str, Any]]:
    """Extract locations from scene text.

    Args:
        scene_text: The text of the scene to extract from.
        scene_index: The index of the scene within the chapter.
        chapter_number: The chapter number this scene belongs to.
        novel_title: The title of the novel.
        novel_genre: The genre of the novel.
        protagonist_name: The protagonist's name.
        model_name: The LLM model name to use for extraction.

    Returns:
        List of location entity dicts with scene_index field.
    """
    prompt = render_prompt(
        "knowledge_agent/extract_locations.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "protagonist": protagonist_name,
            "chapter_number": chapter_number,
            "novel_title": novel_title,
            "novel_genre": novel_genre,
            "chapter_text": scene_text,
        },
    )

    try:
        data, _ = await llm_service.async_call_llm_json_object(
            model_name=model_name,
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            system_prompt=get_system_prompt("knowledge_agent"),
            max_attempts=2,
        )

        world_updates = data.get("world_updates", {})
        if not isinstance(world_updates, dict):
            logger.warning(
                "_extract_locations_from_scene: world_updates is not a dict",
                scene_index=scene_index,
                chapter=chapter_number,
            )
            return []

        location_dict = world_updates.get("Location", {})
        if not isinstance(location_dict, dict):
            logger.warning(
                "_extract_locations_from_scene: Location is not a dict",
                scene_index=scene_index,
                chapter=chapter_number,
            )
            return []

        locations: list[dict[str, Any]] = []
        for name, info in location_dict.items():
            if not isinstance(info, dict):
                continue

            category = str(info.get("category", "Location")).strip()
            locations.append(
                {
                    "name": str(name),
                    "type": "Location",
                    "description": str(info.get("description", "")),
                    "first_appearance_chapter": chapter_number,
                    "scene_index": scene_index,
                    "attributes": {
                        "category": category or "location",
                        "goals": list(info.get("goals", [])),
                        "rules": list(info.get("rules", [])),
                        "key_elements": list(info.get("key_elements", [])),
                    },
                }
            )

        logger.debug(
            "_extract_locations_from_scene: extracted locations",
            scene_index=scene_index,
            chapter=chapter_number,
            count=len(locations),
        )

        return locations

    except LLMServiceError as e:
        logger.warning(
            "_extract_locations_from_scene: LLM failure",
            scene_index=scene_index,
            chapter=chapter_number,
            error=str(e),
        )
        return []
    except Exception as e:
        logger.warning(
            "_extract_locations_from_scene: failed",
            scene_index=scene_index,
            chapter=chapter_number,
            error=str(e),
        )
        return []


async def _extract_events_from_scene(
    scene_text: str,
    scene_index: int,
    chapter_number: int,
    novel_title: str,
    novel_genre: str,
    protagonist_name: str,
    model_name: str,
) -> list[dict[str, Any]]:
    """Extract events from scene text.

    Args:
        scene_text: The text of the scene to extract from.
        scene_index: The index of the scene within the chapter.
        chapter_number: The chapter number this scene belongs to.
        novel_title: The title of the novel.
        novel_genre: The genre of the novel.
        protagonist_name: The protagonist's name.
        model_name: The LLM model name to use for extraction.

    Returns:
        List of event entity dicts with scene_index field.
    """
    prompt = render_prompt(
        "knowledge_agent/extract_events.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "protagonist": protagonist_name,
            "chapter_number": chapter_number,
            "novel_title": novel_title,
            "novel_genre": novel_genre,
            "chapter_text": scene_text,
        },
    )

    try:
        data, _ = await llm_service.async_call_llm_json_object(
            model_name=model_name,
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            system_prompt=get_system_prompt("knowledge_agent"),
            max_attempts=2,
        )

        world_updates = data.get("world_updates", {})
        if not isinstance(world_updates, dict):
            logger.warning(
                "_extract_events_from_scene: world_updates is not a dict",
                scene_index=scene_index,
                chapter=chapter_number,
            )
            return []

        event_dict = world_updates.get("Event", {})
        if not isinstance(event_dict, dict):
            logger.warning(
                "_extract_events_from_scene: Event is not a dict",
                scene_index=scene_index,
                chapter=chapter_number,
            )
            return []

        events: list[dict[str, Any]] = []
        for name, info in event_dict.items():
            if not isinstance(info, dict):
                continue

            category = str(info.get("category", "Event")).strip()
            events.append(
                {
                    "name": str(name),
                    "type": "Event",
                    "description": str(info.get("description", "")),
                    "first_appearance_chapter": chapter_number,
                    "scene_index": scene_index,
                    "attributes": {
                        "category": category or "event",
                        "goals": list(info.get("goals", [])),
                        "rules": list(info.get("rules", [])),
                        "key_elements": list(info.get("key_elements", [])),
                    },
                }
            )

        logger.debug(
            "_extract_events_from_scene: extracted events",
            scene_index=scene_index,
            chapter=chapter_number,
            count=len(events),
        )

        return events

    except LLMServiceError as e:
        logger.warning(
            "_extract_events_from_scene: LLM failure",
            scene_index=scene_index,
            chapter=chapter_number,
            error=str(e),
        )
        return []
    except Exception as e:
        logger.warning(
            "_extract_events_from_scene: failed",
            scene_index=scene_index,
            chapter=chapter_number,
            error=str(e),
        )
        return []


async def _extract_relationships_from_scene(
    scene_text: str,
    scene_index: int,
    chapter_number: int,
    novel_title: str,
    novel_genre: str,
    protagonist_name: str,
    model_name: str,
) -> list[dict[str, Any]]:
    """Extract relationships from scene text.

    Args:
        scene_text: The text of the scene to extract from.
        scene_index: The index of the scene within the chapter.
        chapter_number: The chapter number this scene belongs to.
        novel_title: The title of the novel.
        novel_genre: The genre of the novel.
        protagonist_name: The protagonist's name.
        model_name: The LLM model name to use for extraction.

    Returns:
        List of relationship dicts with scene_index field.
    """
    prompt = render_prompt(
        "knowledge_agent/extract_relationships.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "protagonist": protagonist_name,
            "chapter_number": chapter_number,
            "novel_title": novel_title,
            "novel_genre": novel_genre,
            "chapter_text": scene_text,
        },
    )

    try:
        data, _ = await llm_service.async_call_llm_json_object(
            model_name=model_name,
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            system_prompt=get_system_prompt("knowledge_agent"),
            max_attempts=2,
        )

        kg_triples_list = data.get("kg_triples", [])
        if not isinstance(kg_triples_list, list):
            logger.warning(
                "_extract_relationships_from_scene: kg_triples is not a list",
                scene_index=scene_index,
                chapter=chapter_number,
            )
            return []

        relationships: list[dict[str, Any]] = []
        for triple in kg_triples_list:
            if not isinstance(triple, dict):
                continue

            subject = triple.get("subject", "")
            predicate = triple.get("predicate", "RELATES_TO")
            object_entity = triple.get("object_entity", "")
            description = triple.get("description", "")

            if isinstance(subject, dict):
                subject = subject.get("name", str(subject))
            if isinstance(object_entity, dict):
                object_entity = object_entity.get("name", str(object_entity))

            subject_text = str(subject) if subject else ""
            target_text = str(object_entity) if object_entity else ""
            predicate_text = str(predicate) if predicate else ""

            if subject_text and target_text and predicate_text:
                relationships.append(
                    {
                        "source_name": subject_text,
                        "target_name": target_text,
                        "relationship_type": predicate_text,
                        "description": str(description),
                        "chapter": chapter_number,
                        "scene_index": scene_index,
                        "confidence": 0.8,
                    }
                )

        logger.debug(
            "_extract_relationships_from_scene: extracted relationships",
            scene_index=scene_index,
            chapter=chapter_number,
            count=len(relationships),
        )

        return relationships

    except LLMServiceError as e:
        logger.warning(
            "_extract_relationships_from_scene: LLM failure",
            scene_index=scene_index,
            chapter=chapter_number,
            error=str(e),
        )
        return []
    except Exception as e:
        logger.warning(
            "_extract_relationships_from_scene: failed",
            scene_index=scene_index,
            chapter=chapter_number,
            error=str(e),
        )
        return []
