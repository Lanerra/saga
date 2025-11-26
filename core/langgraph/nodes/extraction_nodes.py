# core/langgraph/nodes/extraction_nodes.py
"""
Granular entity extraction nodes for LangGraph workflow.

This module contains specialized extraction nodes that run in parallel
to extract characters, locations, events, and relationships.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import structlog

import config
from core.langgraph.state import (
    ExtractedEntity,
    ExtractedRelationship,
    NarrativeState,
)
from core.llm_interface_refactored import llm_service
from processing.entity_deduplication import generate_entity_id
from processing.parsing_utils import parse_llm_triples
from prompts.prompt_renderer import get_system_prompt, render_prompt
from core.langgraph.nodes.extraction_node import _parse_extraction_json, _map_category_to_type

logger = structlog.get_logger(__name__)


async def extract_characters(state: NarrativeState) -> Dict[str, Any]:
    """
    Extract character details from the chapter text.
    """
    logger.info("extract_characters: starting")
    
    if not state.get("draft_text"):
        return {"character_updates": []}

    prompt = render_prompt(
        "knowledge_agent/extract_characters.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "protagonist": state.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME),
            "chapter_number": state["current_chapter"],
            "novel_title": state["title"],
            "novel_genre": state["genre"],
            "chapter_text": state["draft_text"],
        },
    )

    try:
        raw_text, _ = await llm_service.async_call_llm(
            model_name=state["medium_model"],
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            system_prompt=get_system_prompt("knowledge_agent"),
        )
        
        data = await _parse_extraction_json(raw_text, state["current_chapter"])
        if not data:
            return {"character_updates": []}

        # Process character updates into ExtractedEntity objects
        char_updates = []
        raw_updates = data.get("character_updates", {})
        
        for name, info in raw_updates.items():
            if isinstance(info, dict):
                attributes = {
                    "traits": info.get("traits", []),
                    "status": info.get("status", "Unknown"),
                    "relationships": info.get("relationships", {})
                }
                
                char_updates.append(
                    ExtractedEntity(
                        name=name,
                        type="Character",  # Capitalized to match ontology node types
                        description=info.get("description", ""),
                        first_appearance_chapter=state["current_chapter"],
                        attributes=attributes,
                    )
                )
                
        return {"character_updates": char_updates}

    except Exception as e:
        logger.error("extract_characters: failed", error=str(e))
        return {"character_updates": []}


async def extract_locations(state: NarrativeState) -> Dict[str, Any]:
    """
    Extract location details from the chapter text.
    """
    logger.info("extract_locations: starting")
    
    if not state.get("draft_text"):
        return {"location_updates": []}

    prompt = render_prompt(
        "knowledge_agent/extract_locations.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "protagonist": state.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME),
            "chapter_number": state["current_chapter"],
            "novel_title": state["title"],
            "novel_genre": state["genre"],
            "chapter_text": state["draft_text"],
        },
    )

    try:
        raw_text, _ = await llm_service.async_call_llm(
            model_name=state["medium_model"],
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            system_prompt=get_system_prompt("knowledge_agent"),
        )
        
        data = await _parse_extraction_json(raw_text, state["current_chapter"])
        if not data:
            return {"location_updates": []}

        # Process world updates into ExtractedEntity objects
        world_updates = []
        raw_updates = data.get("world_updates", {})

        # Event-related types that should be skipped (handled by extract_events)
        event_related_types = {
            "Event", "DevelopmentEvent", "WorldElaborationEvent", "PlotPoint",
            "Era", "Moment", "Timeline"
        }

        for category, items in raw_updates.items():
            if isinstance(items, dict):
                for name, info in items.items():
                    if isinstance(info, dict):
                        entity_type = _map_category_to_type(category)
                        # Only process non-events here (events handled by extract_events)
                        if entity_type not in event_related_types:
                            item_id = generate_entity_id(name, category, state["current_chapter"])
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
                                    first_appearance_chapter=state["current_chapter"],
                                    attributes=attributes,
                                )
                            )
                            
        return {"location_updates": world_updates}

    except Exception as e:
        logger.error("extract_locations: failed", error=str(e))
        return {"location_updates": []}


async def extract_events(state: NarrativeState) -> Dict[str, Any]:
    """
    Extract significant events from the chapter text.
    """
    logger.info("extract_events: starting")
    
    if not state.get("draft_text"):
        return {"event_updates": []}

    prompt = render_prompt(
        "knowledge_agent/extract_events.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "protagonist": state.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME),
            "chapter_number": state["current_chapter"],
            "novel_title": state["title"],
            "novel_genre": state["genre"],
            "chapter_text": state["draft_text"],
        },
    )

    try:
        raw_text, _ = await llm_service.async_call_llm(
            model_name=state["medium_model"],
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            system_prompt=get_system_prompt("knowledge_agent"),
        )
        
        data = await _parse_extraction_json(raw_text, state["current_chapter"])
        if not data:
            return {"event_updates": []}

        event_updates = []
        raw_updates = data.get("world_updates", {})

        # Look for event-related categories (Event, DevelopmentEvent, WorldElaborationEvent, PlotPoint, etc.)
        event_related_types = {
            "Event", "DevelopmentEvent", "WorldElaborationEvent", "PlotPoint",
            "Era", "Moment", "Timeline"
        }

        for category, items in raw_updates.items():
            mapped_type = _map_category_to_type(category)
            if mapped_type in event_related_types and isinstance(items, dict):
                for name, info in items.items():
                    if isinstance(info, dict):
                        item_id = generate_entity_id(name, category, state["current_chapter"])
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
                                first_appearance_chapter=state["current_chapter"],
                                attributes=attributes,
                            )
                        )
                        
        return {"event_updates": event_updates}

    except Exception as e:
        logger.error("extract_events: failed", error=str(e))
        return {"event_updates": []}


async def extract_relationships(state: NarrativeState) -> Dict[str, Any]:
    """
    Extract relationships between entities.
    """
    logger.info("extract_relationships: starting")
    
    if not state.get("draft_text"):
        return {"relationship_updates": []}

    prompt = render_prompt(
        "knowledge_agent/extract_relationships.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "protagonist": state.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME),
            "chapter_number": state["current_chapter"],
            "novel_title": state["title"],
            "novel_genre": state["genre"],
            "chapter_text": state["draft_text"],
        },
    )

    try:
        raw_text, _ = await llm_service.async_call_llm(
            model_name=state["medium_model"],
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            system_prompt=get_system_prompt("knowledge_agent"),
        )
        
        data = await _parse_extraction_json(raw_text, state["current_chapter"])
        if not data:
            return {"relationship_updates": []}

        relationships = []
        kg_triples_list = data.get("kg_triples", [])
        
        if isinstance(kg_triples_list, list):
            kg_triples_text = "\n".join([str(t) for t in kg_triples_list])
        else:
            kg_triples_text = str(kg_triples_list)

        parsed_triples = parse_llm_triples(kg_triples_text)

        for triple in parsed_triples:
            subject = triple.get("subject", "")
            predicate = triple.get("predicate", "RELATES_TO")
            object_entity = triple.get("object_entity", "")
            object_literal = triple.get("object_literal", "")
            target = object_entity if object_entity else object_literal

            if isinstance(subject, dict):
                subject = subject.get("name", str(subject))
            if isinstance(target, dict):
                target = target.get("name", str(target))

            subject = str(subject) if subject else ""
            target = str(target) if target else ""

            if subject and target and predicate:
                relationships.append(
                    ExtractedRelationship(
                        source_name=subject,
                        target_name=target,
                        relationship_type=predicate,
                        description=triple.get("description", ""),
                        chapter=state["current_chapter"],
                        confidence=0.8,
                    )
                )
                
        return {"relationship_updates": relationships}

    except Exception as e:
        logger.error("extract_relationships: failed", error=str(e))
        return {"relationship_updates": []}


def consolidate_extraction(state: NarrativeState) -> NarrativeState:
    """
    Merge results from parallel extractions.
    
    This node expects the state to have been updated with partial results
    from the parallel branches. However, since LangGraph parallel branches
    return separate state updates that are merged, we need to handle
    how LangGraph merges them.
    
    In LangGraph, if multiple nodes return updates to the same key, the behavior
    depends on the reducer. If we return different keys from each node,
    they will all be present in the state passed to this node.
    """
    logger.info("consolidate_extraction: merging results")
    
    # Collect all updates
    # Note: In a real parallel execution, the state passed here would contain
    # the merged results of the parallel branches if they wrote to different keys.
    # We'll assume the parallel nodes return dicts that get merged into the state.
    
    char_updates = state.get("character_updates", [])
    location_updates = state.get("location_updates", [])
    event_updates = state.get("event_updates", [])
    relationship_updates = state.get("relationship_updates", [])
    
    # Combine world items (locations + events)
    world_items = location_updates + event_updates
    
    extracted_entities = {
        "characters": char_updates,
        "world_items": world_items,
    }
    
    # Clean up temporary keys
    # (We can't easily remove keys in TypedDict state, but we can ignore them)
    
    return {
        **state,
        "extracted_entities": extracted_entities,
        "extracted_relationships": relationship_updates,
        "current_node": "consolidate_extraction",
    }
