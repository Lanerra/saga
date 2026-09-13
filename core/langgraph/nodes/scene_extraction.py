# core/langgraph/nodes/scene_extraction.py
"""Extract entities from individual scenes instead of full chapters.

This module provides scene-level extraction to reduce prompt sizes and improve
extraction quality by processing smaller text chunks (~5-10K chars each).

拆分说明 (Split overview):
    - scene_extraction_parsing.py       — LLM output parsing helpers
    - scene_extraction_validation.py   — spaCy entity validation + lazy TextProcessingService
    - scene_extraction_normalization.py — entity deduplication across scenes
    - This file                        — orchestration: extract_from_scenes, per-type extractors
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import structlog

import config
from core.exceptions import LLMServiceError
from core.langgraph.chapter_lifecycle import extraction_binding
from core.langgraph.content_manager import ContentManager, get_scene_drafts, require_project_dir
from core.langgraph.nodes.scene_extraction_normalization import consolidate_scene_extractions
from core.langgraph.nodes.scene_extraction_parsing import (
    SceneRelationships,
    normalize_dict_items,
    normalize_triple_entities,
    parse_character_updates,
    parse_kg_triples,
    parse_world_updates,
)
from core.langgraph.nodes.scene_extraction_validation import (
    _validate_entity_with_spacy,
    load_spacy_model_if_enabled,
)
from core.langgraph.state import NarrativeState, SceneExtractionOutcome, SceneExtractionType
from core.service_context import get_services
from models.kg_constants import RELATIONSHIP_TYPES
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
        Ordered extraction_outcomes and extraction_status. Only complete scenes
        include characters, world_items, and relationships payloads.
    """
    logger.info(
        "extract_from_scene: starting",
        scene_index=scene_index,
        chapter=chapter_number,
        scene_text_length=len(scene_text),
    )

    preprocessing_error = ""
    preprocessing_error_type = ""
    try:
        if not scene_text.strip():
            raise ValueError("Scene text must not be blank")
    except Exception as error:
        preprocessing_error = str(error)
        preprocessing_error_type = type(error).__name__

    outcomes: list[SceneExtractionOutcome] = []
    payloads: dict[str, list[dict[str, Any]]] = {}
    extractors: dict[SceneExtractionType, Callable[..., Awaitable[list[dict[str, Any]]]]] = {
        "characters": _extract_characters_from_scene,
        "locations": _extract_locations_from_scene,
        "events": _extract_events_from_scene,
        "relationships": _extract_relationships_from_scene,
    }
    for extraction_type, extractor in extractors.items():
        outcome: SceneExtractionOutcome = {
            "chapter_number": chapter_number,
            "scene_index": scene_index,
            "extraction_type": extraction_type,
            "status": "failed",
            "item_count": 0,
            "error_type": preprocessing_error_type,
            "error": preprocessing_error,
        }
        if not preprocessing_error_type:
            try:
                items = await extractor(
                    scene_text, scene_index, chapter_number, novel_title,
                    novel_genre, protagonist_name, model_name,
                )
                payloads[extraction_type] = items
                outcome["status"] = "succeeded"
                outcome["item_count"] = len(items)
            except Exception as error:
                outcome["error_type"] = type(error).__name__
                outcome["error"] = str(error)
        outcomes.append(outcome)

    if any(outcome["status"] == "failed" for outcome in outcomes):
        return {"extraction_outcomes": outcomes, "extraction_status": "failed"}

    return {
        "characters": payloads["characters"],
        "world_items": payloads["locations"] + payloads["events"],
        "relationships": payloads["relationships"],
        "extraction_outcomes": outcomes,
        "extraction_status": "complete",
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
        data, _ = await get_services().language_model.async_call_llm_json_object(
            model_name=model_name,
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            system_prompt=get_system_prompt("knowledge_agent"),
            max_attempts=2,
        )

        parsed = parse_character_updates(data, scene_index, chapter_number)

        characters: list[dict[str, Any]] = []
        for name, info in parsed:
            character_name = str(name)

            is_validated = _validate_entity_with_spacy(scene_text, character_name)

            if not is_validated:
                raise ValueError("Character extraction contains an ungrounded or unnamed entity")
            for target_name in info.get("relationships", {}):
                if not _validate_entity_with_spacy(scene_text, target_name):
                    raise ValueError("Character relationship contains an ungrounded or unnamed endpoint")

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
        raise
    except Exception as e:
        logger.error(
            "_extract_characters_from_scene: failed",
            scene_index=scene_index,
            chapter=chapter_number,
            error=str(e),
            exc_info=True,
        )
        raise


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
        data, _ = await get_services().language_model.async_call_llm_json_object(
            model_name=model_name,
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            system_prompt=get_system_prompt("knowledge_agent"),
            max_attempts=2,
        )

        parsed = parse_world_updates(data, "Location", scene_index, chapter_number)

        locations: list[dict[str, Any]] = []
        for name, info in parsed:
            # Validate entity presence using spaCy
            is_validated = _validate_entity_with_spacy(scene_text, str(name))

            if not is_validated:
                raise ValueError("Location extraction contains an ungrounded or unnamed entity")

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
        raise
    except Exception as e:
        logger.error(
            "_extract_locations_from_scene: failed",
            scene_index=scene_index,
            chapter=chapter_number,
            error=str(e),
            exc_info=True,
        )
        raise


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
        data, _ = await get_services().language_model.async_call_llm_json_object(
            model_name=model_name,
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            system_prompt=get_system_prompt("knowledge_agent"),
            max_attempts=2,
        )

        parsed = parse_world_updates(data, "Event", scene_index, chapter_number)

        events: list[dict[str, Any]] = []
        for name, info in parsed:
            # Validate entity presence using spaCy
            is_validated = _validate_entity_with_spacy(scene_text, str(name))

            if not is_validated:
                raise ValueError("Event extraction contains an ungrounded or unnamed entity")

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
        raise
    except Exception as e:
        logger.error(
            "_extract_events_from_scene: failed",
            scene_index=scene_index,
            chapter=chapter_number,
            error=str(e),
            exc_info=True,
        )
        raise


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
        data, _ = await get_services().language_model.async_call_llm_json_object(
            model_name=model_name,
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            system_prompt=get_system_prompt("knowledge_agent"),
            max_attempts=2,
            auto_clean_response=False,
            reject_duplicate_keys=True,
            response_format=SceneRelationships.response_format(),
        )

        parsed = parse_kg_triples(data, scene_index, chapter_number)

        relationships: list[dict[str, Any]] = []
        for triple in parsed:
            subject_text, target_text, predicate_text, description = normalize_triple_entities(triple)

            # Validate entity presence using spaCy for both subjects and targets
            subject_validated = _validate_entity_with_spacy(scene_text, subject_text)
            target_validated = _validate_entity_with_spacy(scene_text, target_text)

            if not subject_validated or not target_validated:
                raise ValueError("Relationship extraction contains an ungrounded or unnamed endpoint")

            if subject_text and target_text and predicate_text:
                relationships.append(
                    {
                        "source_name": subject_text,
                        "target_name": target_text,
                        "relationship_type": predicate_text,
                        "description": description,
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
        raise
    except Exception as e:
        logger.error(
            "_extract_relationships_from_scene: failed",
            scene_index=scene_index,
            chapter=chapter_number,
            error=str(e),
            exc_info=True,
        )
        raise


async def extract_from_scenes(state: NarrativeState) -> dict[str, Any]:
    """Extract entities from individual scenes, then consolidate.

    This node replaces chapter-level extraction to avoid 135K+ char prompts.

    Args:
        state: Workflow state with scene_drafts_ref.

    Returns:
        Completion telemetry and externalized refs, or fatal state with cleared
        refs when scene input or any extraction slot fails. Partial publication
        is not supported.
    """
    logger.info(
        "extract_from_scenes: starting scene-level extraction",
        chapter=state.get("current_chapter", 1),
    )

    if state.get("has_fatal_error"):
        logger.warning("extract_from_scenes: skipping due to fatal error")
        return {"current_node": "extract_from_scenes"}

    failure_update: NarrativeState = {
        "extraction_status": "failed",
        "extraction_source": None,
        "extraction_policy": "fail_closed",
        "extraction_outcomes": [],
        "extracted_entities_ref": None,
        "extracted_relationships_ref": None,
        "has_fatal_error": True,
        "error_node": "extract_from_scenes",
        "current_node": "extract_from_scenes",
    }
    try:
        content_manager = ContentManager(require_project_dir(state))
        scene_drafts = get_scene_drafts(state, content_manager)
        if not scene_drafts:
            raise ValueError("Scene extraction requires nonempty scene drafts")
        planned_count = state.get("chapter_plan_scene_count", 0)
        if planned_count > 0 and len(scene_drafts) != planned_count:
            raise ValueError(f"Expected {planned_count} scene drafts, received {len(scene_drafts)}")
    except Exception as e:
        error_msg = f"Failed to load scene drafts: {e}"
        logger.error("extract_from_scenes: fatal error", error=error_msg)
        return {**failure_update, "last_error": error_msg}

    chapter_number = state.get("current_chapter", 1)
    novel_title = state.get("title", "")
    novel_genre = state.get("genre", "")
    protagonist_name = state.get("protagonist_name", "")
    model_name = state.get("extraction_model", config.MEDIUM_MODEL)

    # Load spaCy model for entity validation if enabled
    load_spacy_model_if_enabled()

    logger.info(
        "extract_from_scenes: processing scenes",
        chapter=chapter_number,
        scene_count=len(scene_drafts),
    )

    scene_results: list[dict[str, Any]] = []
    outcomes: list[SceneExtractionOutcome] = []
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
        outcomes.extend(scene_result["extraction_outcomes"])

    failures = [outcome for outcome in outcomes if outcome["status"] == "failed"]
    if failures:
        error_message = "; ".join(
            f"Scene {outcome['scene_index']} {outcome['extraction_type']}: {outcome['error_type']}: {outcome['error']}"
            for outcome in failures
        )
        logger.error("extract_from_scenes: incomplete extraction", chapter=chapter_number, error=error_message)
        return {**failure_update, "last_error": error_message, "extraction_outcomes": outcomes}

    logger.info(
        "extract_from_scenes: consolidating results",
        chapter=chapter_number,
        scene_results_count=len(scene_results),
    )

    consolidated = consolidate_scene_extractions(scene_results)

    characters = normalize_dict_items(consolidated.get("characters", []), item_kind="characters")
    world_items = normalize_dict_items(consolidated.get("world_items", []), item_kind="world_items")
    extracted_relationships = normalize_dict_items(
        consolidated.get("relationships", []), item_kind="relationships"
    )

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
        content_manager.get_latest_version("extracted_relationships", f"chapter_{chapter_number}") + 1,
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
        "extraction_status": "complete",
        "extraction_policy": "fail_closed",
        "extraction_outcomes": outcomes,
        **({"extraction_source": extraction_binding(state, scene_drafts)} if "lifecycle_version" in state else {}),
    }
