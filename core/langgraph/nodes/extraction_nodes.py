# core/langgraph/nodes/extraction_nodes.py
"""
Granular entity extraction nodes for LangGraph workflow.

This module contains specialized extraction nodes that run in parallel
to extract characters, locations, events, and relationships.
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

import config
from core.langgraph.content_manager import ContentManager, get_draft_text
from core.langgraph.nodes.extraction_node import (
    _map_category_to_type,
)
from core.langgraph.state import (
    ExtractedEntity,
    ExtractedRelationship,
    NarrativeState,
)
from core.llm_interface_refactored import llm_service
from processing.entity_deduplication import generate_entity_id
from prompts.grammar_loader import load_grammar
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)


async def extract_characters(state: NarrativeState) -> dict[str, Any]:
    """
    Extract character details from the chapter text.

    FIRST node in sequential extraction - CLEARS previous extraction state.
    """
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
            "protagonist": state.get(
                "protagonist_name", config.DEFAULT_PROTAGONIST_NAME
            ),
            "chapter_number": state.get("current_chapter", 1),
            "novel_title": state.get("title", ""),
            "novel_genre": state.get("genre", ""),
            "chapter_text": draft_text,
        },
    )

    try:
        # Load grammar for character extraction
        grammar_content = load_grammar("extraction")
        # Prepend root rule for character extraction
        grammar = re.sub(r"^root ::= .*$", "", grammar_content, flags=re.MULTILINE)
        grammar = f"root ::= character-extraction\n{grammar}"

        logger.debug("extract_characters: using grammar", grammar_head=grammar[:100])

        raw_text, _ = await llm_service.async_call_llm(
            model_name=state.get("medium_model", ""),
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            system_prompt=get_system_prompt("knowledge_agent"),
            grammar=grammar,
        )

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.error("extract_characters: failed to parse JSON", raw_text=raw_text)
            return {
                "extracted_entities": {"characters": [], "world_items": []},
                "extracted_relationships": [],
            }

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
                attributes = {
                    "traits": info.get("traits", []),
                    "status": info.get("status", "Unknown"),
                    "relationships": info.get("relationships", {}),
                }

                char_updates.append(
                    ExtractedEntity(
                        name=name,
                        type="Character",  # Capitalized to match ontology node types
                        description=info.get("description", ""),
                        first_appearance_chapter=state.get("current_chapter", 1),
                        attributes=attributes,
                    )
                )

        # Clear extraction state and set characters
        return {
            "extracted_entities": {"characters": char_updates, "world_items": []},
            "extracted_relationships": [],
        }

    except Exception as e:
        logger.error("extract_characters: failed", error=str(e))
        return {
            "extracted_entities": {"characters": [], "world_items": []},
            "extracted_relationships": [],
        }


async def extract_locations(state: NarrativeState) -> dict[str, Any]:
    """
    Extract location details from the chapter text.

    APPENDS to world_items (sequential extraction).
    """
    logger.info("extract_locations: starting")

    # Initialize content manager and get draft text
    content_manager = ContentManager(state.get("project_dir", ""))
    draft_text = get_draft_text(state, content_manager)

    # Get existing world_items to append to
    existing_entities = state.get("extracted_entities", {})
    existing_world_items = existing_entities.get("world_items", [])
    existing_characters = existing_entities.get("characters", [])

    if not draft_text:
        return {}

    prompt = render_prompt(
        "knowledge_agent/extract_locations.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "protagonist": state.get(
                "protagonist_name", config.DEFAULT_PROTAGONIST_NAME
            ),
            "chapter_number": state.get("current_chapter", 1),
            "novel_title": state.get("title", ""),
            "novel_genre": state.get("genre", ""),
            "chapter_text": draft_text,
        },
    )

    try:
        # Load grammar for world extraction
        grammar_content = load_grammar("extraction")
        # Prepend root rule for world extraction
        grammar = re.sub(r"^root ::= .*$", "", grammar_content, flags=re.MULTILINE)
        grammar = f"root ::= world-extraction\n{grammar}"

        logger.debug("extract_events: using grammar", grammar_head=grammar[:100])

        raw_text, _ = await llm_service.async_call_llm(
            model_name=state.get("medium_model", ""),
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            system_prompt=get_system_prompt("knowledge_agent"),
            grammar=grammar,
        )

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.error("extract_locations: failed to parse JSON", raw_text=raw_text)
            return {}

        if not data:
            return {}

        # Process world updates into ExtractedEntity objects
        world_updates = []
        raw_updates = data.get("world_updates", {})

        # Event-related types that should be skipped (handled by extract_events)
        event_related_types = {
            "Event",
            "DevelopmentEvent",
            "WorldElaborationEvent",
            "PlotPoint",
            "Era",
            "Moment",
            "Timeline",
        }

        for category, items in raw_updates.items():
            if isinstance(items, dict):
                for name, info in items.items():
                    if isinstance(info, dict):
                        entity_type = _map_category_to_type(category)
                        # Only process non-events here (events handled by extract_events)
                        if entity_type not in event_related_types:
                            item_id = generate_entity_id(
                                name, category, state.get("current_chapter", 1)
                            )
                            attributes = {
                                "category": category,
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
                                    first_appearance_chapter=state.get(
                                        "current_chapter", 1
                                    ),
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

    except Exception as e:
        logger.error("extract_locations: failed", error=str(e))
        return {}


async def extract_events(state: NarrativeState) -> dict[str, Any]:
    """
    Extract significant events from the chapter text.

    APPENDS to world_items (sequential extraction).
    """
    logger.info("extract_events: starting")

    # Initialize content manager and get draft text
    content_manager = ContentManager(state.get("project_dir", ""))
    draft_text = get_draft_text(state, content_manager)

    # Get existing world_items to append to
    existing_entities = state.get("extracted_entities", {})
    existing_world_items = existing_entities.get("world_items", [])
    existing_characters = existing_entities.get("characters", [])

    if not draft_text:
        return {}

    prompt = render_prompt(
        "knowledge_agent/extract_events.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "protagonist": state.get(
                "protagonist_name", config.DEFAULT_PROTAGONIST_NAME
            ),
            "chapter_number": state.get("current_chapter", 1),
            "novel_title": state.get("title", ""),
            "novel_genre": state.get("genre", ""),
            "chapter_text": draft_text,
        },
    )

    try:
        # Load grammar for world extraction (includes events)
        grammar_content = load_grammar("extraction")
        # Prepend root rule for world extraction
        grammar = re.sub(r"^root ::= .*$", "", grammar_content, flags=re.MULTILINE)
        grammar = f"root ::= world-extraction\n{grammar}"

        logger.debug("extract_locations: using grammar", grammar_head=grammar[:100])

        raw_text, _ = await llm_service.async_call_llm(
            model_name=state.get("medium_model", ""),
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            system_prompt=get_system_prompt("knowledge_agent"),
            grammar=grammar,
        )

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.error("extract_events: failed to parse JSON", raw_text=raw_text)
            return {}

        if not data:
            return {}

        event_updates = []
        raw_updates = data.get("world_updates", {})

        # Look for event-related categories (Event, DevelopmentEvent, WorldElaborationEvent, PlotPoint, etc.)
        event_related_types = {
            "Event",
            "DevelopmentEvent",
            "WorldElaborationEvent",
            "PlotPoint",
            "Era",
            "Moment",
            "Timeline",
        }

        for category, items in raw_updates.items():
            mapped_type = _map_category_to_type(category)
            if mapped_type in event_related_types and isinstance(items, dict):
                for name, info in items.items():
                    if isinstance(info, dict):
                        item_id = generate_entity_id(
                            name, category, state.get("current_chapter", 1)
                        )
                        attributes = {
                            "category": category,
                            "id": item_id,
                            "key_elements": info.get("key_elements", []),
                        }

                        event_updates.append(
                            ExtractedEntity(
                                name=name,
                                type=mapped_type,  # Use the specific type from mapping
                                description=info.get("description", ""),
                                first_appearance_chapter=state.get(
                                    "current_chapter", 1
                                ),
                                attributes=attributes,
                            )
                        )

        # Append to existing world_items, preserving characters
        return {
            "extracted_entities": {
                "characters": existing_characters,
                "world_items": existing_world_items + event_updates,
            }
        }

    except Exception as e:
        logger.error("extract_events: failed", error=str(e))
        return {}


async def extract_relationships(state: NarrativeState) -> dict[str, Any]:
    """
    Extract relationships between entities.

    Sets extracted_relationships (sequential extraction).
    Also creates ExtractedEntity objects for entities found in relationships
    that weren't already extracted, preserving their type information.
    """
    logger.info("extract_relationships: starting")

    # Initialize content manager and get draft text
    content_manager = ContentManager(state.get("project_dir", ""))
    draft_text = get_draft_text(state, content_manager)

    # Get existing entities to check if we need to create new ones
    existing_entities = state.get("extracted_entities", {})
    existing_characters = existing_entities.get("characters", [])
    existing_world_items = existing_entities.get("world_items", [])

    if not draft_text:
        return {"extracted_relationships": []}

    prompt = render_prompt(
        "knowledge_agent/extract_relationships.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "protagonist": state.get(
                "protagonist_name", config.DEFAULT_PROTAGONIST_NAME
            ),
            "chapter_number": state.get("current_chapter", 1),
            "novel_title": state.get("title", ""),
            "novel_genre": state.get("genre", ""),
            "chapter_text": draft_text,
        },
    )

    try:
        # Load grammar for relationship extraction
        grammar_content = load_grammar("extraction")
        # Prepend root rule for relationship extraction
        grammar = re.sub(r"^root ::= .*$", "", grammar_content, flags=re.MULTILINE)
        grammar = f"root ::= relationship-extraction\n{grammar}"

        logger.debug("extract_relationships: using grammar", grammar_head=grammar[:100])

        raw_text, _ = await llm_service.async_call_llm(
            model_name=state.get("medium_model", ""),
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            system_prompt=get_system_prompt("knowledge_agent"),
            grammar=grammar,
        )

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.error(
                "extract_relationships: failed to parse JSON", raw_text=raw_text
            )
            return {"extracted_relationships": []}

        if not data:
            return {"extracted_relationships": []}

        relationships = []
        kg_triples_list = data.get("kg_triples", [])

        if not isinstance(kg_triples_list, list):
            logger.warning(
                "extract_relationships: kg_triples is not a list", raw_data=data
            )
            return {"extracted_relationships": []}

        # Import parsing utility
        from processing.parsing_utils import _get_entity_type_and_name_from_text

        # Track entities found in relationships to create ExtractedEntity objects
        # Map: entity_name -> entity_type
        relationship_entities = {}

        # Build set of already extracted entity names for quick lookup
        existing_entity_names = set()
        for e in existing_characters:
            name = e.name if hasattr(e, "name") else e.get("name", "")
            if name:
                existing_entity_names.add(name)
        for e in existing_world_items:
            name = e.name if hasattr(e, "name") else e.get("name", "")
            if name:
                existing_entity_names.add(name)

        for triple in kg_triples_list:
            if not isinstance(triple, dict):
                continue

            subject = triple.get("subject", "")
            predicate = triple.get("predicate", "RELATES_TO")
            object_entity = triple.get("object_entity", "")

            # Note: Grammar doesn't support object_literal in triple_object, only object_entity
            # triple_object ::= "{" ws "\"subject\"" ws ":" ws json_string "," ws "\"predicate\"" ws ":" ws json_string "," ws "\"object_entity\"" ws ":" ws json_string "," ws "\"description\"" ws ":" ws json_string ws "}"

            # Handle edge cases where values might be dicts despite grammar (though grammar enforces strings)
            # Keeping safe casting just in case
            if isinstance(subject, dict):
                subject = subject.get("name", str(subject))
            if isinstance(object_entity, dict):
                object_entity = object_entity.get("name", str(object_entity))

            subject = str(subject) if subject else ""
            target = str(object_entity) if object_entity else ""

            # Parse "Type:Name" format from subject and object
            # The prompt asks for format like "Character:Elara" or "Location:Sunken Library"
            # We need to extract the name and type separately
            subject_type = None
            target_type = None

            if ":" in subject:
                subject_parsed = _get_entity_type_and_name_from_text(subject)
                subject_type = subject_parsed.get("type")
                subject = subject_parsed.get("name") or subject

            if ":" in target:
                target_parsed = _get_entity_type_and_name_from_text(target)
                target_type = target_parsed.get("type")
                target = target_parsed.get("name") or target

            # Track entity types for entities not already extracted
            if subject and subject_type and subject not in existing_entity_names:
                relationship_entities[subject] = subject_type
            if target and target_type and target not in existing_entity_names:
                relationship_entities[target] = target_type

            if subject and target and predicate:
                relationships.append(
                    ExtractedRelationship(
                        source_name=subject,
                        target_name=target,
                        relationship_type=predicate,
                        description=triple.get("description", ""),
                        chapter=state.get("current_chapter", 1),
                        confidence=0.8,
                    )
                )

        # Create ExtractedEntity objects for entities found in relationships
        # but not already in the extraction results
        new_entities_from_relationships = []
        for entity_name, entity_type in relationship_entities.items():
            # Determine if this is a character or world item
            is_character = entity_type == "Character"

            entity = ExtractedEntity(
                name=entity_name,
                type=entity_type,
                description=f"Entity mentioned in relationships. Type: {entity_type}",
                first_appearance_chapter=state.get("current_chapter", 1),
                attributes={"category": entity_type.lower()} if not is_character else {},
            )
            new_entities_from_relationships.append((entity, is_character))

        # Add new entities to the appropriate lists
        new_characters = [e for e, is_char in new_entities_from_relationships if is_char]
        new_world_items = [
            e for e, is_char in new_entities_from_relationships if not is_char
        ]

        logger.info(
            "extract_relationships: created entities from relationship parsing",
            new_characters=len(new_characters),
            new_world_items=len(new_world_items),
            total_relationships=len(relationships),
        )

        return {
            "extracted_relationships": relationships,
            "extracted_entities": {
                "characters": existing_characters + new_characters,
                "world_items": existing_world_items + new_world_items,
            },
        }

    except Exception as e:
        logger.error("extract_relationships: failed", error=str(e))
        return {"extracted_relationships": []}


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
    current_version = (
        content_manager.get_latest_version(
            "extracted_entities", f"chapter_{chapter_number}"
        )
        + 1
    )

    # Serialize ExtractedEntity and ExtractedRelationship objects to dicts
    extracted_entities = state.get("extracted_entities", {})
    entities_dict = {
        "characters": [
            e.model_dump() if hasattr(e, "model_dump") else e
            for e in extracted_entities.get("characters", [])
        ],
        "world_items": [
            e.model_dump() if hasattr(e, "model_dump") else e
            for e in extracted_entities.get("world_items", [])
        ],
    }

    relationships = state.get("extracted_relationships", [])
    relationships_list = [
        r.model_dump() if hasattr(r, "model_dump") else r for r in relationships
    ]

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
