# core/langgraph/nodes/extraction_nodes.py
"""
Granular entity extraction nodes for LangGraph workflow.

This module contains specialized extraction nodes that run in parallel
to extract characters, locations, events, and relationships.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

import structlog

import config
from core.exceptions import LLMServiceError
from core.langgraph.content_manager import ContentManager, get_draft_text
from core.langgraph.state import (
    ExtractedEntity,
    ExtractedRelationship,
    NarrativeState,
)
from core.llm_interface_refactored import llm_service
from core.schema_validator import schema_validator
from processing.entity_deduplication import generate_entity_id
from prompts.prompt_renderer import get_system_prompt, render_prompt
from utils.text_processing import validate_and_filter_traits

logger = structlog.get_logger(__name__)


async def extract_characters(state: NarrativeState) -> dict[str, Any]:
    """
    Extract character details from the chapter text.

    FIRST node in sequential extraction - CLEARS previous extraction state.
    """
    if state.get("has_fatal_error", False):
        # Fail-fast / no-op: upstream error should stop the workflow; don't do more LLM calls.
        return {"current_node": "extract_characters"}

    logger.info("extract_characters: starting (clearing previous extraction state)")

    # Initialize content manager and get draft text
    content_manager = ContentManager(state.get("project_dir", ""))
    draft_text = get_draft_text(state, content_manager)

    if not draft_text:
        # Clear state even if no draft
        return {
            "extracted_entities": {"characters": [], "world_items": []},
            "extracted_relationships": [],
        }

    prompt = render_prompt(
        "knowledge_agent/extract_characters.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "protagonist": state.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME),
            "chapter_number": state.get("current_chapter", 1),
            "novel_title": state.get("title", ""),
            "novel_genre": state.get("genre", ""),
            "chapter_text": draft_text,
        },
    )

    try:
        data, _ = await llm_service.async_call_llm_json_object(
            model_name=state.get("medium_model", ""),
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            system_prompt=get_system_prompt("knowledge_agent"),
            max_attempts=2,
        )

        if not data:
            return {
                "extracted_entities": {"characters": [], "world_items": []},
                "extracted_relationships": [],
            }

        # Process character updates into ExtractedEntity objects
        char_updates = []
        raw_updates = data.get("character_updates", {})

        for name, info in raw_updates.items():
            if isinstance(info, dict):
                # Validate and filter traits
                raw_traits = info.get("traits", [])
                validated_traits = validate_and_filter_traits(raw_traits)

                attributes = {
                    "traits": validated_traits,
                    "status": info.get("status", "Unknown"),
                    "relationships": info.get("relationships", {}),
                }

                # Validate entity type (Characters are always "Character")
                is_valid, normalized_type, err = schema_validator.validate_entity_type("Character")

                if is_valid:
                    char_updates.append(
                        ExtractedEntity(
                            name=name,
                            type=normalized_type,
                            description=info.get("description", ""),
                            first_appearance_chapter=state.get("current_chapter", 1),
                            attributes=attributes,
                        )
                    )
                else:
                    logger.warning(
                        "extract_characters: invalid entity type",
                        type="Character",
                        error=err,
                    )

        # Clear extraction state and set characters
        return {
            "extracted_entities": {"characters": char_updates, "world_items": []},
            "extracted_relationships": [],
        }

    except LLMServiceError as e:
        logger.error("extract_characters: LLM failure", error=str(e), exc_info=True)
        return {
            "current_node": "extract_characters",
            "last_error": f"Character extraction failed: {e}",
            "has_fatal_error": True,
            "error_node": "extract_characters",
            "extracted_entities": {"characters": [], "world_items": []},
            "extracted_relationships": [],
        }
    except Exception as e:
        logger.error("extract_characters: failed", error=str(e), exc_info=True)
        return {
            "current_node": "extract_characters",
            "last_error": f"Character extraction failed: {e}",
            "has_fatal_error": True,
            "error_node": "extract_characters",
            "extracted_entities": {"characters": [], "world_items": []},
            "extracted_relationships": [],
        }


async def extract_locations(state: NarrativeState) -> dict[str, Any]:
    """
    Extract location details from the chapter text.

    APPENDS to world_items (sequential extraction).
    """
    if state.get("has_fatal_error", False):
        return {"current_node": "extract_locations"}

    logger.info("extract_locations: starting")

    # Initialize content manager and get draft text
    content_manager = ContentManager(state.get("project_dir", ""))
    draft_text = get_draft_text(state, content_manager)

    # Get existing world_items to append to
    # Deep copy to avoid mutating original state
    existing_entities = copy.deepcopy(state.get("extracted_entities", {}))
    existing_world_items = existing_entities.get("world_items", [])
    existing_characters = existing_entities.get("characters", [])

    if not draft_text:
        return {}

    prompt = render_prompt(
        "knowledge_agent/extract_locations.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "protagonist": state.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME),
            "chapter_number": state.get("current_chapter", 1),
            "novel_title": state.get("title", ""),
            "novel_genre": state.get("genre", ""),
            "chapter_text": draft_text,
        },
    )

    # DEBUG: validate prompt/template wiring and detect schema mismatches early.
    # Do NOT log any prompt fragments (they can contain narrative/manuscript text).
    try:
        import hashlib

        prompt_sha = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
        logger.debug(
            "extract_locations: rendered prompt",
            template="knowledge_agent/extract_locations.j2",
            prompt_sha1=prompt_sha,
            prompt_len=len(prompt),
            chapter=state.get("current_chapter", 1),
        )
    except Exception as _e:  # pragma: no cover - debug logging must never break extraction
        logger.debug("extract_locations: failed to hash/log prompt for debugging")

    try:
        data: dict[str, Any] | None = None

        for attempt in range(1, 3):
            raw_text, _ = await llm_service.async_call_llm(
                model_name=state.get("medium_model", ""),
                prompt=prompt,
                temperature=config.Temperatures.KG_EXTRACTION,
                max_tokens=config.MAX_KG_TRIPLE_TOKENS,
                allow_fallback=True,
                system_prompt=get_system_prompt("knowledge_agent"),
            )

            try:
                decoded = json.loads(raw_text)
            except json.JSONDecodeError:
                import hashlib

                response_sha1 = hashlib.sha1(raw_text.encode("utf-8")).hexdigest()[:12]
                logger.error(
                    "extract_locations: failed to parse JSON",
                    response_sha1=response_sha1,
                    response_len=len(raw_text),
                )

                if attempt == 2:
                    return {
                        "current_node": "extract_locations",
                        "last_error": "Location extraction failed: LLM returned invalid JSON",
                        "has_fatal_error": True,
                        "error_node": "extract_locations",
                    }

                continue

            if not isinstance(decoded, dict):
                return {
                    "current_node": "extract_locations",
                    "last_error": "Location extraction failed: LLM returned invalid JSON",
                    "has_fatal_error": True,
                    "error_node": "extract_locations",
                }

            data = decoded
            break

        if data is None:
            return {
                "current_node": "extract_locations",
                "last_error": "Location extraction failed: LLM returned invalid JSON",
                "has_fatal_error": True,
                "error_node": "extract_locations",
            }

        # DEBUG: log parsed top-level keys and world_updates keys to confirm expected schema.
        try:
            top_keys = sorted(list(data.keys()))
            raw_updates = data.get("world_updates", {})
            wu_keys = sorted(list(raw_updates.keys()))[:30] if isinstance(raw_updates, dict) else []
            suspicious = sorted(
                set(wu_keys).intersection(
                    {
                        "entities",
                        "relationships",
                        "characters",
                        "world_items",
                        "chapters",
                        "items",
                        "traits",
                    }
                )
            )
            logger.info(
                "extract_locations: parsed response shape",
                top_keys=top_keys,
                world_updates_type=type(raw_updates).__name__,
                world_updates_keys_head=wu_keys,
                suspicious_world_updates_keys=suspicious,
            )
        except Exception:  # pragma: no cover - debug logging must never break extraction
            logger.debug("extract_locations: failed to log parsed response shape")

        if not data:
            return {}

        # Process world updates into ExtractedEntity objects
        world_updates: list[ExtractedEntity] = []
        raw_updates = data.get("world_updates", {})

        # We only want Locations here. Events are handled by extract_events, and other
        # canonical types (Item) must not be created by the
        # dedicated location extractor.
        allowed_type = "Location"

        # These keys have historically shown up when the model emits a "rolled-up"
        # structure. They are *structural buckets*, not semantic categories/subtypes.
        ignored_world_update_buckets = {
            "entities",
            "relationships",
            "character_updates",
            "characters",
            "world_items",
            "kg_triples",
            "chapters",
            "items",
            "traits",
        }

        # Backward compatible acceptance for earlier plural bucket names.
        bucket_aliases = {
            "location": "Location",
            "locations": "Location",
        }

        if not isinstance(raw_updates, dict):
            logger.warning(
                "extract_locations: world_updates is not a dict; skipping",
                world_updates_type=type(raw_updates).__name__,
            )
            raw_updates = {}

        for bucket_key, items in raw_updates.items():
            bucket_raw = str(bucket_key).strip() if bucket_key is not None else ""
            bucket_norm = bucket_aliases.get(bucket_raw.lower(), bucket_raw)

            if bucket_norm.lower() in ignored_world_update_buckets:
                logger.warning(
                    "extract_locations: ignoring unexpected world_updates bucket",
                    bucket=bucket_raw,
                )
                continue

            # The prompt contract is: world_updates must be keyed by canonical label "Location".
            # If we see something else, we treat it as drift and skip rather than misclassify.
            if bucket_norm != allowed_type:
                mapped = _map_category_to_type(bucket_norm)
                if mapped == allowed_type and isinstance(items, dict):
                    # Accept "subtype-as-bucket" drift (e.g., "Settlement": {...}) for backward compatibility.
                    logger.warning(
                        "extract_locations: non-canonical bucket accepted as subtype",
                        bucket=bucket_norm,
                        mapped_type=mapped,
                    )
                else:
                    logger.warning(
                        "extract_locations: skipping non-location bucket",
                        bucket=bucket_norm,
                        mapped_type=mapped,
                    )
                    continue

            if not isinstance(items, dict):
                continue

            for name, info in items.items():
                if not isinstance(info, dict):
                    continue

                entity_type = allowed_type

                # Prefer per-entity subtype category, fall back to bucket if we accepted
                # a subtype-as-bucket drift pattern.
                subtype_category = info.get("category")
                if isinstance(subtype_category, str):
                    subtype_category = subtype_category.strip()
                else:
                    subtype_category = ""

                if not subtype_category and bucket_norm != allowed_type:
                    subtype_category = bucket_norm

                # Soft category validation (warning only)
                if subtype_category:
                    _, cat_msg = schema_validator.validate_category(entity_type, subtype_category)
                    if cat_msg:
                        logger.warning(
                            "extract_locations: category validation warning",
                            type=entity_type,
                            category=subtype_category,
                            message=cat_msg,
                        )

                # Use category (when present) for deterministic IDs; otherwise fall back.
                id_category = subtype_category or allowed_type.lower()
                item_id = generate_entity_id(name, id_category, state.get("current_chapter", 1))

                attributes = {
                    "category": subtype_category or id_category,
                    "id": item_id,
                    "goals": info.get("goals", []),
                    "rules": info.get("rules", []),
                    "key_elements": info.get("key_elements", []),
                }

                world_updates.append(
                    ExtractedEntity(
                        name=name,
                        type=entity_type,
                        description=info.get("description", ""),
                        first_appearance_chapter=state.get("current_chapter", 1),
                        attributes=attributes,
                    )
                )

        # Append to existing world_items, preserving characters
        return {
            "extracted_entities": {
                "characters": existing_characters,
                "world_items": existing_world_items + world_updates,
            }
        }

    except LLMServiceError as e:
        logger.error("extract_locations: LLM failure", error=str(e), exc_info=True)
        return {
            "current_node": "extract_locations",
            "last_error": f"Location extraction failed: {e}",
            "has_fatal_error": True,
            "error_node": "extract_locations",
        }
    except Exception as e:
        logger.error("extract_locations: failed", error=str(e), exc_info=True)
        return {
            "current_node": "extract_locations",
            "last_error": f"Location extraction failed: {e}",
            "has_fatal_error": True,
            "error_node": "extract_locations",
        }


async def extract_events(state: NarrativeState) -> dict[str, Any]:
    """
    Extract significant events from the chapter text.

    APPENDS to world_items (sequential extraction).
    """
    if state.get("has_fatal_error", False):
        return {"current_node": "extract_events"}

    logger.info("extract_events: starting")

    content_manager = ContentManager(state.get("project_dir", ""))
    draft_text = get_draft_text(state, content_manager)

    existing_entities = copy.deepcopy(state.get("extracted_entities", {}))
    existing_world_items = existing_entities.get("world_items", [])
    existing_characters = existing_entities.get("characters", [])

    if not draft_text:
        return {}

    prompt = render_prompt(
        "knowledge_agent/extract_events.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "protagonist": state.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME),
            "chapter_number": state.get("current_chapter", 1),
            "novel_title": state.get("title", ""),
            "novel_genre": state.get("genre", ""),
            "chapter_text": draft_text,
        },
    )

    try:
        import hashlib

        prompt_sha = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
        logger.debug(
            "extract_events: rendered prompt",
            template="knowledge_agent/extract_events.j2",
            prompt_sha1=prompt_sha,
            prompt_len=len(prompt),
            chapter=state.get("current_chapter", 1),
        )
    except Exception:  # pragma: no cover
        logger.debug("extract_events: failed to hash/log prompt for debugging")

    try:
        try:
            data, _ = await llm_service.async_call_llm_json_object(
                model_name=state.get("medium_model", ""),
                prompt=prompt,
                temperature=config.Temperatures.KG_EXTRACTION,
                max_tokens=config.MAX_KG_TRIPLE_TOKENS,
                allow_fallback=True,
                system_prompt=get_system_prompt("knowledge_agent"),
                max_attempts=2,
            )
        except ValueError:
            return {
                "current_node": "extract_events",
                "last_error": "Event extraction failed: LLM returned invalid JSON",
                "has_fatal_error": True,
                "error_node": "extract_events",
            }

        try:
            top_keys = sorted(list(data.keys())) if isinstance(data, dict) else []
            raw_updates = data.get("world_updates", {}) if isinstance(data, dict) else {}
            wu_keys = sorted(list(raw_updates.keys()))[:30] if isinstance(raw_updates, dict) else []
            suspicious = sorted(
                set(wu_keys).intersection(
                    {
                        "entities",
                        "relationships",
                        "characters",
                        "world_items",
                        "chapters",
                        "items",
                        "traits",
                    }
                )
            )
            logger.info(
                "extract_events: parsed response shape",
                top_keys=top_keys,
                world_updates_type=type(raw_updates).__name__,
                world_updates_keys_head=wu_keys,
                suspicious_world_updates_keys=suspicious,
            )
        except Exception:  # pragma: no cover
            logger.debug("extract_events: failed to log parsed response shape")

        if not data:
            return {}

        raw_updates = data.get("world_updates", {})
        if not isinstance(raw_updates, dict):
            logger.warning(
                "extract_events: world_updates is not a dict; skipping",
                world_updates_type=type(raw_updates).__name__,
            )
            raw_updates = {}

        allowed_type = "Event"
        ignored_world_update_buckets = {
            "entities",
            "relationships",
            "character_updates",
            "characters",
            "world_items",
            "kg_triples",
            "chapters",
            "items",
            "traits",
        }
        bucket_aliases = {
            "event": "Event",
            "events": "Event",
        }

        event_updates: list[ExtractedEntity] = []

        for bucket_key, items in raw_updates.items():
            bucket_raw = str(bucket_key).strip() if bucket_key is not None else ""
            bucket_norm = bucket_aliases.get(bucket_raw.lower(), bucket_raw)

            if bucket_norm.lower() in ignored_world_update_buckets:
                logger.warning(
                    "extract_events: ignoring unexpected world_updates bucket",
                    bucket=bucket_raw,
                )
                continue

            if bucket_norm != allowed_type:
                mapped = _map_category_to_type(bucket_norm)
                if mapped == allowed_type and isinstance(items, dict):
                    logger.warning(
                        "extract_events: non-canonical bucket accepted as subtype",
                        bucket=bucket_norm,
                        mapped_type=mapped,
                    )
                else:
                    logger.warning(
                        "extract_events: skipping non-event bucket",
                        bucket=bucket_norm,
                        mapped_type=mapped,
                    )
                    continue

            if not isinstance(items, dict):
                continue

            for name, info in items.items():
                if not isinstance(info, dict):
                    continue

                subtype_category = info.get("category")
                subtype_category = subtype_category.strip() if isinstance(subtype_category, str) else ""
                if not subtype_category and bucket_norm != allowed_type:
                    subtype_category = bucket_norm

                if subtype_category:
                    _, cat_msg = schema_validator.validate_category(allowed_type, subtype_category)
                    if cat_msg:
                        logger.warning(
                            "extract_events: category validation warning",
                            type=allowed_type,
                            category=subtype_category,
                            message=cat_msg,
                        )

                id_category = subtype_category or allowed_type.lower()
                item_id = generate_entity_id(name, id_category, state.get("current_chapter", 1))

                attributes = {
                    "category": subtype_category or id_category,
                    "id": item_id,
                    "key_elements": info.get("key_elements", []),
                }

                event_updates.append(
                    ExtractedEntity(
                        name=name,
                        type=allowed_type,
                        description=info.get("description", ""),
                        first_appearance_chapter=state.get("current_chapter", 1),
                        attributes=attributes,
                    )
                )

        return {
            "extracted_entities": {
                "characters": existing_characters,
                "world_items": existing_world_items + event_updates,
            }
        }

    except LLMServiceError as e:
        logger.error("extract_events: LLM failure", error=str(e), exc_info=True)
        return {
            "current_node": "extract_events",
            "last_error": f"Event extraction failed: {e}",
            "has_fatal_error": True,
            "error_node": "extract_events",
        }
    except Exception as e:
        logger.error("extract_events: failed", error=str(e), exc_info=True)
        return {
            "current_node": "extract_events",
            "last_error": f"Event extraction failed: {e}",
            "has_fatal_error": True,
            "error_node": "extract_events",
        }


async def extract_relationships(state: NarrativeState) -> dict[str, Any]:
    """
    Extract relationships between entities.

    Sets extracted_relationships (sequential extraction).
    Also creates ExtractedEntity objects for entities found in relationships
    that weren't already extracted, preserving their type information.
    """
    if state.get("has_fatal_error", False):
        return {"current_node": "extract_relationships"}

    logger.info("extract_relationships: starting")

    content_manager = ContentManager(state.get("project_dir", ""))
    draft_text = get_draft_text(state, content_manager)

    existing_entities = copy.deepcopy(state.get("extracted_entities", {}))
    existing_characters = existing_entities.get("characters", [])
    existing_world_items = existing_entities.get("world_items", [])

    if not draft_text:
        return {"extracted_relationships": []}

    prompt = render_prompt(
        "knowledge_agent/extract_relationships.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "protagonist": state.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME),
            "chapter_number": state.get("current_chapter", 1),
            "novel_title": state.get("title", ""),
            "novel_genre": state.get("genre", ""),
            "chapter_text": draft_text,
        },
    )

    try:
        try:
            data, _ = await llm_service.async_call_llm_json_object(
                model_name=state.get("medium_model", ""),
                prompt=prompt,
                temperature=config.Temperatures.KG_EXTRACTION,
                max_tokens=config.MAX_KG_TRIPLE_TOKENS,
                allow_fallback=True,
                system_prompt=get_system_prompt("knowledge_agent"),
                max_attempts=2,
            )
        except ValueError:
            return {
                "current_node": "extract_relationships",
                "last_error": "Relationship extraction failed: LLM returned invalid JSON",
                "has_fatal_error": True,
                "error_node": "extract_relationships",
                "extracted_relationships": [],
            }

        if not data:
            return {"extracted_relationships": []}

        kg_triples_list = data.get("kg_triples", [])
        if not isinstance(kg_triples_list, list):
            logger.warning("extract_relationships: kg_triples is not a list", raw_data=data)
            return {"extracted_relationships": []}

        from processing.parsing_utils import _get_entity_type_and_name_from_text

        relationships: list[ExtractedRelationship] = []
        relationship_entities: dict[str, str] = {}

        existing_entity_names: set[str] = set()
        for entity in existing_characters:
            name = entity.name if hasattr(entity, "name") else entity.get("name", "")
            if name:
                existing_entity_names.add(name)
        for entity in existing_world_items:
            name = entity.name if hasattr(entity, "name") else entity.get("name", "")
            if name:
                existing_entity_names.add(name)

        for triple in kg_triples_list:
            if not isinstance(triple, dict):
                continue

            subject: Any = triple.get("subject", "")
            predicate: Any = triple.get("predicate", "RELATES_TO")
            object_entity: Any = triple.get("object_entity", "")

            if isinstance(subject, dict):
                subject = subject.get("name", str(subject))
            if isinstance(object_entity, dict):
                object_entity = object_entity.get("name", str(object_entity))

            subject_text = str(subject) if subject else ""
            target_text = str(object_entity) if object_entity else ""
            predicate_text = str(predicate) if predicate else ""

            subject_type: str | None = None
            target_type: str | None = None

            if ":" in subject_text:
                subject_parsed = _get_entity_type_and_name_from_text(subject_text)
                raw_type = subject_parsed.get("type")
                if raw_type:
                    is_valid, normalized, _ = schema_validator.validate_entity_type(raw_type)
                    subject_type = normalized if is_valid else raw_type
                subject_text = subject_parsed.get("name") or subject_text

            if ":" in target_text:
                target_parsed = _get_entity_type_and_name_from_text(target_text)
                raw_type = target_parsed.get("type")
                if raw_type:
                    is_valid, normalized, _ = schema_validator.validate_entity_type(raw_type)
                    target_type = normalized if is_valid else raw_type
                target_text = target_parsed.get("name") or target_text

            if subject_text and subject_type and subject_text not in existing_entity_names:
                relationship_entities[subject_text] = subject_type
            if target_text and target_type and target_text not in existing_entity_names:
                relationship_entities[target_text] = target_type

            if subject_text and target_text and predicate_text:
                relationships.append(
                    ExtractedRelationship(
                        source_name=subject_text,
                        target_name=target_text,
                        relationship_type=predicate_text,
                        description=triple.get("description", ""),
                        chapter=state.get("current_chapter", 1),
                        confidence=0.8,
                        source_type=subject_type,
                        target_type=target_type,
                    )
                )

        new_entities_from_relationships: list[tuple[ExtractedEntity, bool]] = []
        for entity_name, entity_type in relationship_entities.items():
            is_valid, normalized_type, _err = schema_validator.validate_entity_type(entity_type)
            final_type = normalized_type if is_valid else "Item"
            is_character = final_type == "Character"

            attributes: dict[str, Any] = {}
            if not is_character:
                raw_cat = str(entity_type).strip().lower() if entity_type else ""
                attributes["category"] = raw_cat or final_type.lower()

            new_entities_from_relationships.append(
                (
                    ExtractedEntity(
                        name=entity_name,
                        type=final_type,
                        description=f"Entity mentioned in relationships. Type: {final_type}",
                        first_appearance_chapter=state.get("current_chapter", 1),
                        attributes=attributes,
                    ),
                    is_character,
                )
            )

        new_characters = [entity for entity, is_char in new_entities_from_relationships if is_char]
        new_world_items = [entity for entity, is_char in new_entities_from_relationships if not is_char]

        logger.info(
            "extract_relationships: created entities from relationship parsing",
            new_characters=len(new_characters),
            new_characters_list=[(e.name, e.type) for e in new_characters],
            new_world_items=len(new_world_items),
            new_world_items_list=[(e.name, e.type) for e in new_world_items],
            total_relationships=len(relationships),
            existing_characters_count=len(existing_characters),
            existing_world_items_count=len(existing_world_items),
        )

        return {
            "extracted_relationships": relationships,
            "extracted_entities": {
                "characters": existing_characters + new_characters,
                "world_items": existing_world_items + new_world_items,
            },
        }

    except LLMServiceError as e:
        logger.error("extract_relationships: LLM failure", error=str(e), exc_info=True)
        return {
            "current_node": "extract_relationships",
            "last_error": f"Relationship extraction failed: {e}",
            "has_fatal_error": True,
            "error_node": "extract_relationships",
            "extracted_relationships": [],
        }
    except Exception as e:
        logger.error("extract_relationships: failed", error=str(e), exc_info=True)
        return {
            "current_node": "extract_relationships",
            "last_error": f"Relationship extraction failed: {e}",
            "has_fatal_error": True,
            "error_node": "extract_relationships",
            "extracted_relationships": [],
        }


def consolidate_extraction(state: NarrativeState) -> NarrativeState:
    """
    Finalize extraction after parallel nodes complete.

    With the reducer-based approach, parallel extraction results are automatically
    merged by merge_extracted_entities and merge_extracted_relationships reducers.
    This node marks the extraction phase as complete and externalizes the results.

    Note: The actual merging happens automatically via LangGraph reducers on
    the extracted_entities and extracted_relationships fields.
    """
    logger.info(
        "consolidate_extraction: extraction complete",
        characters=len(state.get("extracted_entities", {}).get("characters", [])),
        world_items=len(state.get("extracted_entities", {}).get("world_items", [])),
        relationships=len(state.get("extracted_relationships", [])),
    )

    # Externalize extracted entities and relationships to reduce state bloat
    content_manager = ContentManager(state.get("project_dir", ""))
    chapter_number = state.get("current_chapter", 1)

    # Get current version for this chapter's extractions
    current_version = content_manager.get_latest_version("extracted_entities", f"chapter_{chapter_number}") + 1

    # Serialize ExtractedEntity and ExtractedRelationship objects to dicts
    extracted_entities = state.get("extracted_entities", {})
    entities_dict = {
        "characters": [e.model_dump() if hasattr(e, "model_dump") else e for e in extracted_entities.get("characters", [])],
        "world_items": [e.model_dump() if hasattr(e, "model_dump") else e for e in extracted_entities.get("world_items", [])],
    }

    relationships = state.get("extracted_relationships", [])
    relationships_list = [r.model_dump() if hasattr(r, "model_dump") else r for r in relationships]

    # Save to external files
    from core.langgraph.content_manager import (
        save_extracted_entities,
        save_extracted_relationships,
    )

    entities_ref = save_extracted_entities(
        content_manager,
        entities_dict,
        chapter_number,
        current_version,
    )

    relationships_ref = save_extracted_relationships(
        content_manager,
        relationships_list,
        chapter_number,
        current_version,
    )

    logger.info(
        "consolidate_extraction: content externalized",
        chapter=chapter_number,
        version=current_version,
        entities_size=entities_ref["size_bytes"],
        relationships_size=relationships_ref["size_bytes"],
    )

    return {
        "extracted_entities_ref": entities_ref,
        "extracted_relationships_ref": relationships_ref,
        "current_node": "consolidate_extraction",
    }


def _map_category_to_type(category: str) -> str:
    """
    Map a category string to a canonical node label.

    Canonical labeling contract:
    - Node labels MUST be one of the 9 canonical labels in [`VALID_NODE_LABELS`](models/kg_constants.py:64).
    - Subtypes like "Faction", "Settlement", "PlotPoint", "Artifact", etc. are represented
      in properties (typically `category`), NOT as Neo4j labels.

    Strategy:
    1) If the category already matches a canonical label (case-insensitive), return that label.
    2) Otherwise, classify the category into a canonical label using [`classify_category_label()`](utils/text_processing.py:116),
       which already maps legacy-ish categories like "artifact"/"relic"/"document" to "Item".
    """
    from models.kg_constants import VALID_NODE_LABELS
    from utils.text_processing import classify_category_label

    raw = str(category) if category is not None else ""
    raw = raw.strip()
    if not raw:
        return "Item"

    for node_label in VALID_NODE_LABELS:
        if raw.lower() == node_label.lower():
            return node_label

    return classify_category_label(raw)
