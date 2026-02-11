# core/langgraph/nodes/scene_extraction.py
"""Extract entities from individual scenes instead of full chapters.

This module provides scene-level extraction to reduce prompt sizes and improve
extraction quality by processing smaller text chunks (~5-10K chars each).
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic import BaseModel

import config
from core.exceptions import LLMServiceError
from core.langgraph.content_manager import ContentManager, get_scene_drafts, require_project_dir
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from core.text_processing_service import TextProcessingService
from models.kg_constants import RELATIONSHIP_TYPES
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)

# Global text processing service instance for entity validation
text_processing_service = TextProcessingService()


def _validate_entity_with_spacy(scene_text: str, entity_name: str) -> bool:
    """Validate that an extracted entity is actually present in the scene text.

    Args:
        scene_text: The source scene text.
        entity_name: The entity name to validate.

    Returns:
        True if entity is validated (present or validation disabled), False if not found.
    """
    if not config.settings.ENABLE_ENTITY_VALIDATION:
        logger.debug("_validate_entity_with_spacy: entity validation disabled by config")
        return True

    if not text_processing_service.spacy_service.is_loaded():
        logger.warning("_validate_entity_with_spacy: spaCy model not loaded, skipping validation")
        return True

    try:
        is_present = text_processing_service.spacy_service.verify_entity_presence(scene_text, entity_name, threshold=0.7)

        if not is_present:
            logger.warning(
                "_validate_entity_with_spacy: entity not found in text",
                entity_name=entity_name,
                entity_length=len(entity_name),
                scene_text_length=len(scene_text),
            )

        return is_present
    except Exception as e:
        logger.error("_validate_entity_with_spacy: validation failed, using fallback", error=str(e))
        # Fallback to simple substring matching
        return entity_name.lower() in scene_text.lower()


def _get_normalized_entity_key(name: str) -> str:
    """Get a normalized key for entity deduplication using spaCy.

    Args:
        name: The entity name to normalize.

    Returns:
        Normalized key for deduplication.
    """
    if config.settings.ENABLE_ENTITY_VALIDATION and text_processing_service.spacy_service.is_loaded():
        try:
            return text_processing_service.spacy_service.normalize_entity_name(name)
        except Exception as e:
            logger.warning("_get_normalized_entity_key: spaCy normalization failed, using fallback", error=str(e))

    # Fallback to simple case-insensitive normalization
    return name.lower()


def consolidate_scene_extractions(
    scene_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge and deduplicate extraction results from multiple scenes.

    Deduplication strategy:
    - Characters: Dedupe by name (spaCy-based normalization when available), keep longest description
    - World items: Dedupe by name (spaCy-based normalization when available), keep longest description
    - Relationships: Dedupe by (source, target, type) tuple with spaCy normalization

    Args:
        scene_results: List of extraction results from individual scenes.

    Returns:
        Consolidated dict with deduplicated characters, world_items, relationships.
    """
    characters_map: dict[str, dict[str, Any]] = {}
    world_items_map: dict[str, dict[str, Any]] = {}
    relationships_set: set[tuple[str, str, str]] = set()
    relationships: list[dict[str, Any]] = []

    for scene_result in scene_results:
        for character in scene_result.get("characters", []):
            name = character["name"]
            name_key = _get_normalized_entity_key(name)

            if name_key in characters_map:
                existing = characters_map[name_key]
                existing_desc_len = len(existing.get("description", ""))
                new_desc_len = len(character.get("description", ""))

                if new_desc_len > existing_desc_len:
                    characters_map[name_key] = character
            else:
                characters_map[name_key] = character

        for world_item in scene_result.get("world_items", []):
            name = world_item["name"]
            name_key = _get_normalized_entity_key(name)

            if name_key in world_items_map:
                existing = world_items_map[name_key]
                existing_desc_len = len(existing.get("description", ""))
                new_desc_len = len(world_item.get("description", ""))

                if new_desc_len > existing_desc_len:
                    world_items_map[name_key] = world_item
            else:
                world_items_map[name_key] = world_item

        for relationship in scene_result.get("relationships", []):
            source = relationship.get("source_name", "")
            target = relationship.get("target_name", "")
            rel_type = relationship.get("relationship_type", "")

            # Use spaCy normalization for relationship deduplication
            source_key = _get_normalized_entity_key(source)
            target_key = _get_normalized_entity_key(target)
            rel_type_key = rel_type.upper()

            relationship_key = (source_key, target_key, rel_type_key)

            if relationship_key not in relationships_set:
                relationships_set.add(relationship_key)
                relationships.append(relationship)

    return {
        "characters": list(characters_map.values()),
        "world_items": list(world_items_map.values()),
        "relationships": relationships,
    }


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
    # Check configuration settings to prevent extraction of structural entities
    # According to schema design, Stage 5 should only extract physical descriptions and embeddings
    # It should NOT create new structural entities (Characters, Events, Locations, Items)

    if (
        config.ENABLE_CHARACTER_EXTRACTION_FROM_NARRATIVE
        or config.ENABLE_LOCATION_EXTRACTION_FROM_NARRATIVE
        or config.ENABLE_EVENT_EXTRACTION_FROM_NARRATIVE
        or config.ENABLE_ITEM_EXTRACTION_FROM_NARRATIVE
        or config.ENABLE_RELATIONSHIP_EXTRACTION_FROM_NARRATIVE
    ):
        logger.warning(
            "Structural entity extraction is enabled. This violates Stage 5 schema design principles.",
            extra={
                "character_extraction": config.ENABLE_CHARACTER_EXTRACTION_FROM_NARRATIVE,
                "location_extraction": config.ENABLE_LOCATION_EXTRACTION_FROM_NARRATIVE,
                "event_extraction": config.ENABLE_EVENT_EXTRACTION_FROM_NARRATIVE,
                "item_extraction": config.ENABLE_ITEM_EXTRACTION_FROM_NARRATIVE,
                "relationship_extraction": config.ENABLE_RELATIONSHIP_EXTRACTION_FROM_NARRATIVE,
            },
        )

    logger.info(
        "extract_from_scene: starting",
        scene_index=scene_index,
        chapter=chapter_number,
        scene_text_length=len(scene_text),
    )

    # Clean input text with spaCy if entity validation is enabled
    if config.settings.ENABLE_ENTITY_VALIDATION:
        scene_text = text_processing_service.clean_text_with_spacy(scene_text, aggressive=False)

    # Extract characters only if enabled (should be False in Stage 5)
    if config.ENABLE_CHARACTER_EXTRACTION_FROM_NARRATIVE:
        characters = await _extract_characters_from_scene(
            scene_text=scene_text,
            scene_index=scene_index,
            chapter_number=chapter_number,
            novel_title=novel_title,
            novel_genre=novel_genre,
            protagonist_name=protagonist_name,
            model_name=model_name,
        )
    else:
        characters = []
        logger.info("Character extraction from narrative is disabled by configuration (Stage 5 compliance)")

    # Extract locations only if enabled (should be False in Stage 5)
    if config.ENABLE_LOCATION_EXTRACTION_FROM_NARRATIVE:
        locations = await _extract_locations_from_scene(
            scene_text=scene_text,
            scene_index=scene_index,
            chapter_number=chapter_number,
            novel_title=novel_title,
            novel_genre=novel_genre,
            protagonist_name=protagonist_name,
            model_name=model_name,
        )
    else:
        locations = []
        logger.info("Location extraction from narrative is disabled by configuration (Stage 5 compliance)")

    # Extract events only if enabled (should be False in Stage 5)
    if config.ENABLE_EVENT_EXTRACTION_FROM_NARRATIVE:
        events = await _extract_events_from_scene(
            scene_text=scene_text,
            scene_index=scene_index,
            chapter_number=chapter_number,
            novel_title=novel_title,
            novel_genre=novel_genre,
            protagonist_name=protagonist_name,
            model_name=model_name,
        )
    else:
        events = []
        logger.info("Event extraction from narrative is disabled by configuration (Stage 5 compliance)")

    # Extract relationships only if enabled (should be False in Stage 5)
    if config.ENABLE_RELATIONSHIP_EXTRACTION_FROM_NARRATIVE:
        relationships = await _extract_relationships_from_scene(
            scene_text=scene_text,
            scene_index=scene_index,
            chapter_number=chapter_number,
            novel_title=novel_title,
            novel_genre=novel_genre,
            protagonist_name=protagonist_name,
            model_name=model_name,
        )
    else:
        relationships = []
        logger.info("Relationship extraction from narrative is disabled by configuration (Stage 5 compliance)")

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
            "protagonist": protagonist_name,
            "chapter_number": chapter_number,
            "novel_title": novel_title,
            "novel_genre": novel_genre,
            "chapter_text": scene_text,
            "canonical_relationship_types": sorted(RELATIONSHIP_TYPES),
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

            character_name = str(name)

            should_classify, classification_reason = text_processing_service.spacy_service.should_classify_as_character(character_name)

            if not should_classify:
                logger.debug(
                    "_extract_characters_from_scene: rejecting entity based on classification",
                    entity_name=character_name,
                    reason=classification_reason,
                    scene_index=scene_index,
                    chapter=chapter_number,
                )
                continue

            is_validated = _validate_entity_with_spacy(scene_text, character_name)

            if not is_validated:
                logger.warning(
                    "_extract_characters_from_scene: skipping invalid character",
                    character_name=character_name,
                    scene_index=scene_index,
                    chapter=chapter_number,
                )
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

            # Validate entity presence using spaCy
            is_validated = _validate_entity_with_spacy(scene_text, str(name))

            if not is_validated:
                logger.warning(
                    "_extract_locations_from_scene: skipping invalid location",
                    location_name=str(name),
                    scene_index=scene_index,
                    chapter=chapter_number,
                )
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

            # Validate entity presence using spaCy
            is_validated = _validate_entity_with_spacy(scene_text, str(name))

            if not is_validated:
                logger.warning(
                    "_extract_events_from_scene: skipping invalid event",
                    event_name=str(name),
                    scene_index=scene_index,
                    chapter=chapter_number,
                )
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
            "protagonist": protagonist_name,
            "chapter_number": chapter_number,
            "novel_title": novel_title,
            "novel_genre": novel_genre,
            "chapter_text": scene_text,
            "canonical_relationship_types": sorted(RELATIONSHIP_TYPES),
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

            # Validate entity presence using spaCy for both subjects and targets
            subject_validated = _validate_entity_with_spacy(scene_text, subject_text)
            target_validated = _validate_entity_with_spacy(scene_text, target_text)

            if not subject_validated or not target_validated:
                logger.warning(
                    "_extract_relationships_from_scene: skipping invalid relationship",
                    subject=subject_text,
                    target=target_text,
                    predicate=predicate_text,
                    scene_index=scene_index,
                    chapter=chapter_number,
                )
                continue

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


async def extract_from_scenes(state: NarrativeState) -> dict[str, Any]:
    """Extract entities from individual scenes, then consolidate.

    This node replaces chapter-level extraction to avoid 135K+ char prompts.

    Args:
        state: Workflow state with scene_drafts_ref.

    Returns:
        Partial state update with extracted_entities and extracted_relationships.
    """
    logger.info(
        "extract_from_scenes: starting scene-level extraction",
        chapter=state.get("current_chapter", 1),
    )

    if state.get("has_fatal_error"):
        logger.warning("extract_from_scenes: skipping due to fatal error")
        return {"current_node": "extract_from_scenes"}

    content_manager = ContentManager(require_project_dir(state))

    try:
        scene_drafts = get_scene_drafts(state, content_manager)
    except Exception as e:
        error_msg = f"Failed to load scene drafts: {e}"
        logger.error("extract_from_scenes: fatal error", error=error_msg)
        return {
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "extract_from_scenes",
            "current_node": "extract_from_scenes",
        }

    if not scene_drafts:
        logger.warning("extract_from_scenes: no scene drafts found, creating empty externalized content")

        # Even with no scenes, create empty externalized content for consistency
        chapter_number = state.get("current_chapter", 1)
        current_version = content_manager.get_latest_version("extracted_entities", f"chapter_{chapter_number}") + 1

        extracted_entities_ref = content_manager.save_json(
            {"characters": [], "world_items": []},
            "extracted_entities",
            f"chapter_{chapter_number}",
            current_version,
        )

        extracted_relationships_ref = content_manager.save_json(
            [],
            "extracted_relationships",
            f"chapter_{chapter_number}",
            current_version,
        )

        logger.info(
            "extract_from_scenes: empty content externalized",
            chapter=chapter_number,
            version=current_version,
        )

        return {
            "extracted_entities_ref": extracted_entities_ref,
            "extracted_relationships_ref": extracted_relationships_ref,
            "current_node": "extract_from_scenes",
        }

    chapter_number = state.get("current_chapter", 1)
    novel_title = state.get("title", "")
    novel_genre = state.get("genre", "")
    protagonist_name = state.get("protagonist_name", "")
    model_name = state.get("extraction_model", config.MEDIUM_MODEL)

    # Load spaCy model for entity validation if enabled
    if config.settings.ENABLE_ENTITY_VALIDATION:
        logger.info("extract_from_scenes: loading spaCy model for entity validation")
        text_processing_service.load_spacy_model()

    logger.info(
        "extract_from_scenes: processing scenes",
        chapter=chapter_number,
        scene_count=len(scene_drafts),
    )

    scene_results: list[dict[str, Any]] = []
    for scene_index, scene_text in enumerate(scene_drafts):
        scene_result = await extract_from_scene(
            scene_text=scene_text,
            scene_index=scene_index,
            chapter_number=chapter_number,
            novel_title=novel_title,
            novel_genre=novel_genre,
            protagonist_name=protagonist_name,
            model_name=model_name,
        )
        scene_results.append(scene_result)

    logger.info(
        "extract_from_scenes: consolidating results",
        chapter=chapter_number,
        scene_results_count=len(scene_results),
    )

    consolidated = consolidate_scene_extractions(scene_results)

    def _normalize_dict_items(*, items: Any, item_kind: str) -> list[dict[str, Any]]:
        if items is None:
            return []
        if not isinstance(items, list):
            raise TypeError(f"extract_from_scenes: expected {item_kind} to be a list; got {type(items)}")

        normalized: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, BaseModel):
                normalized_value = item.model_dump(mode="json")
                if not isinstance(normalized_value, dict):
                    raise TypeError(f"extract_from_scenes: expected {item_kind} model_dump to produce dict; got {type(normalized_value)}")
                normalized.append(normalized_value)
                continue

            if isinstance(item, dict):
                normalized.append(item)
                continue

            raise TypeError(f"extract_from_scenes: expected {item_kind} item to be dict-like; got {type(item)}")

        return normalized

    characters = _normalize_dict_items(items=consolidated.get("characters", []), item_kind="characters")
    world_items = _normalize_dict_items(items=consolidated.get("world_items", []), item_kind="world_items")
    extracted_relationships = _normalize_dict_items(items=consolidated.get("relationships", []), item_kind="relationships")

    logger.info(
        "extract_from_scenes: extraction complete",
        chapter=chapter_number,
        characters_count=len(characters),
        world_items_count=len(world_items),
        relationships_count=len(extracted_relationships),
    )

    # Externalize extraction results immediately to avoid state bloat
    current_version = content_manager.get_latest_version("extracted_entities", f"chapter_{chapter_number}") + 1

    extracted_entities_ref = content_manager.save_json(
        {"characters": characters, "world_items": world_items},
        "extracted_entities",
        f"chapter_{chapter_number}",
        current_version,
    )

    extracted_relationships_ref = content_manager.save_json(
        extracted_relationships,
        "extracted_relationships",
        f"chapter_{chapter_number}",
        current_version,
    )

    logger.info(
        "extract_from_scenes: content externalized",
        chapter=chapter_number,
        version=current_version,
        entities_size=len(characters) + len(world_items),
        relationships_size=len(extracted_relationships),
    )

    return {
        "extracted_entities_ref": extracted_entities_ref,
        "extracted_relationships_ref": extracted_relationships_ref,
        "current_node": "extract_from_scenes",
    }
