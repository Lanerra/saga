# core/langgraph/nodes/extraction_nodes.py
"""Extract structured entities from chapter drafts.

These nodes run during Phase 2 knowledge extraction. Character, location, event, and
relationship extraction can be scheduled in parallel; each node returns a partial state
update that is merged by LangGraph reducers.

Notes:
    These nodes call the LLM and treat invalid JSON or schema validation failures as
    fatal, setting `has_fatal_error` to stop downstream nodes from doing more I/O.
"""

from __future__ import annotations

import copy
import re
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError, field_validator

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

logger = structlog.get_logger(__name__)


_TRAIT_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")


class _CharacterUpdatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    description: StrictStr
    traits: list[StrictStr]
    status: StrictStr
    relationships: dict[StrictStr, StrictStr]

    @field_validator("traits")
    @classmethod
    def _validate_traits(cls, traits: list[StrictStr]) -> list[StrictStr]:
        for trait in traits:
            if not _TRAIT_PATTERN.fullmatch(trait):
                raise ValueError("Trait must be a single word with letters/numbers/hyphen only")
        return traits


class _CharactersExtractionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    character_updates: dict[StrictStr, _CharacterUpdatePayload]


class _WorldUpdateEventPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    description: StrictStr
    category: StrictStr
    goals: list[StrictStr]
    rules: list[StrictStr]
    key_elements: list[StrictStr]


class _WorldUpdatesEventRootPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    Event: dict[StrictStr, _WorldUpdateEventPayload]


class _EventsExtractionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    world_updates: _WorldUpdatesEventRootPayload


class _WorldUpdateLocationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    description: StrictStr
    category: StrictStr
    goals: list[StrictStr]
    rules: list[StrictStr]
    key_elements: list[StrictStr]


class _WorldUpdatesLocationRootPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    Location: dict[StrictStr, _WorldUpdateLocationPayload]


class _LocationsExtractionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    world_updates: _WorldUpdatesLocationRootPayload


def _safe_validation_error_details(*, data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"root_type": type(data).__name__}

    details: dict[str, Any] = {"top_keys": sorted([str(key) for key in data.keys()])[:50]}

    world_updates = data.get("world_updates")
    if isinstance(world_updates, dict):
        details["world_updates_keys"] = sorted([str(key) for key in world_updates.keys()])[:50]

        event_bucket = world_updates.get("Event")
        if isinstance(event_bucket, dict):
            details["event_count"] = len(event_bucket)

        location_bucket = world_updates.get("Location")
        if isinstance(location_bucket, dict):
            details["location_count"] = len(location_bucket)

    character_updates = data.get("character_updates")
    if isinstance(character_updates, dict):
        details["character_count"] = len(character_updates)

    return details


def _safe_pydantic_error_summary(error: ValidationError) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for entry in error.errors()[:10]:
        summary.append(
            {
                "loc": [str(x) for x in entry.get("loc", [])],
                "type": str(entry.get("type", "")),
            }
        )
    return summary


async def extract_characters(state: NarrativeState) -> dict[str, Any]:
    """Extract character details from the current chapter draft.

    This is the first extraction node and clears any previous extraction state for the
    chapter before producing a new set of characters.

    Args:
        state: Workflow state.

    Returns:
        Partial state update containing `extracted_entities` and `extracted_relationships`.

        If the draft is missing/empty, returns cleared extraction results.
        If an upstream node has already set `has_fatal_error`, returns a no-op update.

    Notes:
        This node performs LLM I/O. Failures in JSON parsing or schema validation are
        treated as fatal and set `has_fatal_error`.
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

        try:
            payload = _CharactersExtractionPayload.model_validate(data)
        except ValidationError as validation_error:
            logger.error(
                "extract_characters: schema validation failed",
                details=_safe_validation_error_details(data=data),
                validation_errors=_safe_pydantic_error_summary(validation_error),
            )
            return {
                "current_node": "extract_characters",
                "last_error": "Character extraction failed: LLM response schema validation failed",
                "has_fatal_error": True,
                "error_node": "extract_characters",
                "extracted_entities": {"characters": [], "world_items": []},
                "extracted_relationships": [],
            }

        char_updates: list[ExtractedEntity] = []
        for name, info in payload.character_updates.items():
            attributes = {
                "traits": list(info.traits),
                "status": str(info.status),
                "relationships": dict(info.relationships),
            }

            is_valid, normalized_type, err = schema_validator.validate_entity_type("Character")
            if not is_valid:
                logger.warning(
                    "extract_characters: invalid entity type",
                    type="Character",
                    error=err,
                )
                continue

            char_updates.append(
                ExtractedEntity(
                    name=str(name),
                    type=normalized_type,
                    description=str(info.description),
                    first_appearance_chapter=state.get("current_chapter", 1),
                    attributes=attributes,
                )
            )

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
    """Extract location details from the current chapter draft.

    This node appends `Location` items to `extracted_entities["world_items"]` while
    preserving any previously extracted characters and world items.

    Args:
        state: Workflow state.

    Returns:
        Partial state update containing `extracted_entities` with appended locations.

        If an upstream node has already set `has_fatal_error`, returns a no-op update.
        If the draft is missing/empty, returns an empty update.

    Notes:
        This node performs LLM I/O. Invalid JSON or schema validation failures are
        treated as fatal and set `has_fatal_error`.
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
                "current_node": "extract_locations",
                "last_error": "Location extraction failed: LLM returned invalid JSON",
                "has_fatal_error": True,
                "error_node": "extract_locations",
            }

        try:
            payload = _LocationsExtractionPayload.model_validate(data)
        except ValidationError as validation_error:
            logger.error(
                "extract_locations: schema validation failed",
                details=_safe_validation_error_details(data=data),
                validation_errors=_safe_pydantic_error_summary(validation_error),
            )
            return {
                "current_node": "extract_locations",
                "last_error": "Location extraction failed: LLM response schema validation failed",
                "has_fatal_error": True,
                "error_node": "extract_locations",
            }

        world_updates: list[ExtractedEntity] = []
        allowed_type = "Location"

        for name, info in payload.world_updates.Location.items():
            subtype_category = str(info.category).strip()
            _, cat_msg = schema_validator.validate_category(allowed_type, subtype_category)
            if cat_msg:
                logger.warning(
                    "extract_locations: category validation warning",
                    type=allowed_type,
                    category=subtype_category,
                    message=cat_msg,
                )

            id_category = subtype_category or allowed_type.lower()
            item_id = generate_entity_id(str(name), id_category, state.get("current_chapter", 1))

            attributes = {
                "category": subtype_category or id_category,
                "id": item_id,
                "goals": list(info.goals),
                "rules": list(info.rules),
                "key_elements": list(info.key_elements),
            }

            world_updates.append(
                ExtractedEntity(
                    name=str(name),
                    type=allowed_type,
                    description=str(info.description),
                    first_appearance_chapter=state.get("current_chapter", 1),
                    attributes=attributes,
                )
            )

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
    """Extract significant events from the current chapter draft.

    This node appends `Event` items to `extracted_entities["world_items"]` while
    preserving any previously extracted characters and world items.

    Args:
        state: Workflow state.

    Returns:
        Partial state update containing `extracted_entities` with appended events.

        If an upstream node has already set `has_fatal_error`, returns a no-op update.
        If the draft is missing/empty, returns an empty update.

    Notes:
        This node performs LLM I/O. Invalid JSON or schema validation failures are
        treated as fatal and set `has_fatal_error`.
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
            payload = _EventsExtractionPayload.model_validate(data)
        except ValidationError as validation_error:
            logger.error(
                "extract_events: schema validation failed",
                details=_safe_validation_error_details(data=data),
                validation_errors=_safe_pydantic_error_summary(validation_error),
            )
            return {
                "current_node": "extract_events",
                "last_error": "Event extraction failed: LLM response schema validation failed",
                "has_fatal_error": True,
                "error_node": "extract_events",
            }

        allowed_type = "Event"
        event_updates: list[ExtractedEntity] = []

        for name, info in payload.world_updates.Event.items():
            subtype_category = str(info.category).strip()
            _, cat_msg = schema_validator.validate_category(allowed_type, subtype_category)
            if cat_msg:
                logger.warning(
                    "extract_events: category validation warning",
                    type=allowed_type,
                    category=subtype_category,
                    message=cat_msg,
                )

            id_category = subtype_category or allowed_type.lower()
            item_id = generate_entity_id(str(name), id_category, state.get("current_chapter", 1))

            attributes = {
                "category": subtype_category or id_category,
                "id": item_id,
                "goals": list(info.goals),
                "rules": list(info.rules),
                "key_elements": list(info.key_elements),
            }

            event_updates.append(
                ExtractedEntity(
                    name=str(name),
                    type=allowed_type,
                    description=str(info.description),
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
    """Extract relationships between entities mentioned in the current chapter draft.

    This node sets `extracted_relationships` and may also add new entities inferred
    from relationship mentions when the relationship payload includes typed names
    (for example, `"Location: The Citadel"`).

    Args:
        state: Workflow state.

    Returns:
        Partial state update containing `extracted_relationships` and (optionally)
        `extracted_entities` with any newly inferred entities appended.

        If the draft is missing/empty, returns an update with an empty
        `extracted_relationships` list.
        If an upstream node has already set `has_fatal_error`, returns a no-op update.

    Notes:
        This node performs LLM I/O. Invalid JSON is treated as fatal and sets
        `has_fatal_error`, but some malformed payload shapes (for example
        `kg_triples` not being a list) degrade gracefully by returning an empty result.
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
            logger.warning(
                "extract_relationships: kg_triples is not a list",
                details=_safe_validation_error_details(data=data),
                kg_triples_type=type(kg_triples_list).__name__,
            )
            return {"extracted_relationships": []}

        from processing.parsing_utils import _get_entity_type_and_name_from_text

        relationships: list[ExtractedRelationship] = []
        relationship_entities: dict[str, str] = {}

        existing_entity_names: set[str] = set()
        for entity in existing_characters:
            if isinstance(entity, dict):
                name = entity.get("name", "")
            else:
                name = getattr(entity, "name", "")
            if isinstance(name, str) and name:
                existing_entity_names.add(name)
        for entity in existing_world_items:
            if isinstance(entity, dict):
                name = entity.get("name", "")
            else:
                name = getattr(entity, "name", "")
            if isinstance(name, str) and name:
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
    """Externalize merged extraction results and mark extraction as complete.

    LangGraph reducers merge parallel extraction outputs into the in-memory state.
    This node persists the merged `extracted_entities` and `extracted_relationships`
    to external files to reduce state size for downstream nodes.

    Args:
        state: Workflow state.

    Returns:
        Partial state update containing:
        - extracted_entities_ref: Content reference for persisted entities.
        - extracted_relationships_ref: Content reference for persisted relationships.
        - current_node: `"consolidate_extraction"`.

    Notes:
        This node performs filesystem I/O via [`ContentManager`](core/langgraph/content_manager.py:70).
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

    characters_as_dicts: list[dict[str, Any]] = []
    for entity in extracted_entities.get("characters", []):
        if isinstance(entity, ExtractedEntity):
            characters_as_dicts.append(entity.model_dump())
        else:
            if not isinstance(entity, dict):
                raise TypeError("consolidate_extraction: expected extracted character entity to be dict-like")
            characters_as_dicts.append(entity)

    world_items_as_dicts: list[dict[str, Any]] = []
    for entity in extracted_entities.get("world_items", []):
        if isinstance(entity, ExtractedEntity):
            world_items_as_dicts.append(entity.model_dump())
        else:
            if not isinstance(entity, dict):
                raise TypeError("consolidate_extraction: expected extracted world item entity to be dict-like")
            world_items_as_dicts.append(entity)

    entities_dict: dict[str, list[dict[str, Any]]] = {
        "characters": characters_as_dicts,
        "world_items": world_items_as_dicts,
    }

    relationships = state.get("extracted_relationships", [])
    relationships_list: list[dict[str, Any]] = []
    for rel in relationships:
        if isinstance(rel, ExtractedRelationship):
            relationships_list.append(rel.model_dump())
        else:
            if not isinstance(rel, dict):
                raise TypeError("consolidate_extraction: expected extracted relationship to be dict-like")
            relationships_list.append(rel)

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
    """Map a free-form category string to a canonical node label.

    Canonical labeling contract:
    - Node labels MUST be one of the canonical labels in [`VALID_NODE_LABELS`](models/kg_constants.py:64).
    - Subtypes like "Faction", "Settlement", "PlotPoint", and "Artifact" are represented in
      properties (typically `category`), not as Neo4j labels.

    Strategy:
    1) If the category already matches a canonical label (case-insensitive), return that label.
    2) Otherwise, classify the category into a canonical label using
       [`classify_category_label()`](utils/text_processing.py:116).

    Args:
        category: Category string produced by extraction or normalization.

    Returns:
        Canonical label suitable for persistence.
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
