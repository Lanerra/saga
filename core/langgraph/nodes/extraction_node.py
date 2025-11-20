# core/langgraph/nodes/extraction_node.py
"""
Entity extraction node for LangGraph workflow.

This module contains the entity extraction logic ported from agents/knowledge_agent.py
for use in the LangGraph-based narrative generation workflow.

Migration Reference: docs/langgraph_migration_plan.md - Step 1.1.2

Source Code Ported From:
- agents/knowledge_agent.py:
  - _llm_extract_updates() (lines 660-701)
  - extract_and_merge_knowledge() (lines 703-820)
  - _extract_updates_as_models() (lines 822-996)
  - _parse_extraction_json() (lines 998-1030)
  - _clean_llm_json() (lines 1032-1051)
"""

from __future__ import annotations

import json
import re
from typing import Any

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

logger = structlog.get_logger(__name__)


async def extract_entities(state: NarrativeState) -> NarrativeState:
    """
    Extract entities and relationships from generated chapter text.

    This is the main LangGraph node function that orchestrates entity extraction.
    It calls the LLM with a specialized prompt to identify characters, locations,
    objects, events, and their relationships within the generated text.

    PORTED FROM: KnowledgeAgent.extract_and_merge_knowledge()

    Process Flow:
    1. Validate draft text exists
    2. Call LLM with extraction prompt
    3. Parse JSON response
    4. Convert to ExtractedEntity and ExtractedRelationship models
    5. Update state with extraction results

    Args:
        state: Current narrative state containing draft_text to analyze

    Returns:
        Updated state with extracted_entities and extracted_relationships populated
    """
    logger.info(
        "extract_entities",
        chapter=state["current_chapter"],
        word_count=state.get("draft_word_count", 0),
    )

    # Validate we have text to extract from
    if not state.get("draft_text"):
        error_msg = "No draft text available for entity extraction"
        logger.error("extract_entities: fatal error", error=error_msg)
        return {
            **state,
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "extract",
            "extracted_entities": {},
            "extracted_relationships": [],
            "current_node": "extract_entities",
        }

    # Call LLM to extract structured updates
    raw_text, usage = await _llm_extract_updates(
        plot_outline=state.get("plot_outline", {}),
        chapter_text=state["draft_text"],
        chapter_number=state["current_chapter"],
        title=state["title"],
        genre=state["genre"],
        protagonist_name=state.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME),
        extraction_model=state["extraction_model"],
    )

    # Handle extraction failure
    if not raw_text or not raw_text.strip():
        logger.warning(
            "extract_entities: LLM extraction returned no text",
            chapter=state["current_chapter"],
        )
        return {
            **state,
            "extracted_entities": {},
            "extracted_relationships": [],
            "current_node": "extract_entities",
            "last_error": "LLM extraction returned empty response",
        }

    # Parse extraction results
    try:
        char_updates, world_updates, relationships = await _extract_updates_as_models(
            raw_text=raw_text,
            chapter_number=state["current_chapter"],
        )

        logger.info(
            "extract_entities: extraction complete",
            characters=len(char_updates),
            world_items=len(world_updates),
            relationships=len(relationships),
        )

        # Convert to state format
        extracted_entities = {
            "characters": char_updates,
            "world_items": world_updates,
        }

        return {
            **state,
            "extracted_entities": extracted_entities,
            "extracted_relationships": relationships,
            "current_node": "extract_entities",
            "last_error": None,
        }

    except Exception as e:
        error_msg = f"Entity extraction failed: {str(e)}"
        logger.error(
            "extract_entities: fatal error",
            error=error_msg,
            chapter=state["current_chapter"],
            exc_info=True,
        )
        return {
            **state,
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "extract",
            "extracted_entities": {},
            "extracted_relationships": [],
            "current_node": "extract_entities",
        }


async def _llm_extract_updates(
    *,
    plot_outline: dict[str, Any],
    chapter_text: str,
    chapter_number: int,
    title: str,
    genre: str,
    protagonist_name: str,
    extraction_model: str,
) -> tuple[str, dict[str, int] | None]:
    """
    Call the LLM to extract structured updates from chapter text.

    PORTED FROM: KnowledgeAgent._llm_extract_updates()

    This function constructs a specialized prompt that instructs the LLM to:
    - Identify named entities (characters, locations, objects, etc.)
    - Extract relationship triples between entities
    - Return results in structured JSON format

    Args:
        plot_outline: Plot information for context
        chapter_text: The chapter text to analyze
        chapter_number: Current chapter number
        title: Novel title
        genre: Novel genre
        protagonist_name: Protagonist name
        extraction_model: Model to use for extraction

    Returns:
        Tuple of (extracted_text, usage_stats)
    """
    # Load available schema types for the prompt
    from models.kg_constants import (
        NODE_LABELS,
        RELATIONSHIP_TYPES,
    )

    prompt = render_prompt(
        "knowledge_agent/extract_updates.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "protagonist": protagonist_name,
            "chapter_number": chapter_number,
            "novel_title": title,
            "novel_genre": genre,
            "chapter_text": chapter_text,
            "available_node_labels": sorted(NODE_LABELS),
            "available_relationship_types": sorted(RELATIONSHIP_TYPES),
        },
    )

    try:
        text, usage = await llm_service.async_call_llm(
            model_name=extraction_model,
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            stream_to_disk=False,
            frequency_penalty=config.FREQUENCY_PENALTY_KG_EXTRACTION,
            presence_penalty=config.PRESENCE_PENALTY_KG_EXTRACTION,
            auto_clean_response=True,
            system_prompt=get_system_prompt("knowledge_agent"),
        )
        return text, usage
    except Exception as e:
        logger.error(
            "_llm_extract_updates: LLM call failed", error=str(e), exc_info=True
        )
        return "", None


async def _extract_updates_as_models(
    raw_text: str, chapter_number: int
) -> tuple[list[ExtractedEntity], list[ExtractedEntity], list[ExtractedRelationship]]:
    """
    Parse LLM extraction response and convert to model instances.

    PORTED FROM: KnowledgeAgent._extract_updates_as_models()

    This function:
    1. Parses the JSON response from the LLM
    2. Converts character updates to ExtractedEntity instances
    3. Converts world updates to ExtractedEntity instances
    4. Extracts and parses relationship triples
    5. Returns structured results

    Args:
        raw_text: Raw JSON text from LLM
        chapter_number: Current chapter number for tracking

    Returns:
        Tuple of (character_entities, world_entities, relationships)
    """
    char_updates: list[ExtractedEntity] = []
    world_updates: list[ExtractedEntity] = []
    relationships: list[ExtractedRelationship] = []

    # Parse JSON
    extraction_data = await _parse_extraction_json(raw_text, chapter_number)
    if not extraction_data:
        return [], [], []

    try:
        # Extract characters
        char_data = extraction_data.get("character_updates", {})
        for name, char_info in char_data.items():
            if isinstance(char_info, dict):
                description = char_info.get("description", "")

                # Process traits
                traits = char_info.get("traits", [])
                if isinstance(traits, list):
                    traits_dict = {
                        trait: "" for trait in traits if isinstance(trait, str)
                    }
                else:
                    traits_dict = {}

                # Process relationships from char_info
                raw_relationships = char_info.get("relationships", {})
                relationships_dict = {}
                if isinstance(raw_relationships, dict):
                    relationships_dict = {
                        str(k): str(v) for k, v in raw_relationships.items()
                    }
                elif isinstance(raw_relationships, list):
                    # Convert list format to dict
                    for rel_entry in raw_relationships:
                        if isinstance(rel_entry, str):
                            if ":" in rel_entry:
                                parts = rel_entry.split(":", 1)
                                if len(parts) == 2 and parts[0].strip():
                                    relationships_dict[parts[0].strip()] = parts[
                                        1
                                    ].strip()
                            elif rel_entry.strip():
                                relationships_dict[rel_entry.strip()] = "related"
                        elif isinstance(rel_entry, dict):
                            target_name = rel_entry.get("name")
                            detail = rel_entry.get("detail", "related")
                            if target_name and isinstance(target_name, str):
                                relationships_dict[target_name] = detail

                # Create entity with all extracted attributes
                attributes = {
                    **traits_dict,
                    **relationships_dict,
                    "status": char_info.get("status", "Unknown"),
                }

                char_updates.append(
                    ExtractedEntity(
                        name=name,
                        type="character",
                        description=description,
                        first_appearance_chapter=chapter_number,
                        attributes=attributes,
                    )
                )

        # Extract world items
        world_data = extraction_data.get("world_updates", {})
        for category, items in world_data.items():
            if isinstance(items, dict):
                for item_name, item_info in items.items():
                    if isinstance(item_info, dict):
                        description = item_info.get("description", "")

                        # Generate deterministic ID
                        item_id = generate_entity_id(
                            item_name, category, chapter_number
                        )

                        # Extract attributes
                        attributes = {
                            "category": category,
                            "id": item_id,
                            "goals": item_info.get("goals", []),
                            "rules": item_info.get("rules", []),
                            "key_elements": item_info.get("key_elements", []),
                        }

                        # Determine entity type based on category
                        entity_type = _map_category_to_type(category)

                        world_updates.append(
                            ExtractedEntity(
                                name=item_name,
                                type=entity_type,
                                description=description,
                                first_appearance_chapter=chapter_number,
                                attributes=attributes,
                            )
                        )

        # Extract KG triples for relationships
        kg_triples_list = extraction_data.get("kg_triples", [])
        if isinstance(kg_triples_list, list):
            kg_triples_text = "\n".join([str(t) for t in kg_triples_list])
        else:
            kg_triples_text = str(kg_triples_list)

        # Parse triples into structured format
        parsed_triples = parse_llm_triples(kg_triples_text)

        for triple in parsed_triples:
            # Extract components from parsed triple
            subject = triple.get("subject", "")
            predicate = triple.get("predicate", "RELATES_TO")
            object_entity = triple.get("object_entity", "")
            object_literal = triple.get("object_literal", "")

            # Use object_entity if available, otherwise object_literal
            target = object_entity if object_entity else object_literal

            # Normalize subject and target - extract 'name' field if they're dicts
            # (LLMs sometimes return {"type": "Character", "name": "Alice"} instead of just "Alice")
            if isinstance(subject, dict):
                subject = subject.get("name", str(subject))
            if isinstance(target, dict):
                target = target.get("name", str(target))

            # Ensure they're strings
            subject = str(subject) if subject else ""
            target = str(target) if target else ""

            if subject and target and predicate:
                relationships.append(
                    ExtractedRelationship(
                        source_name=subject,
                        target_name=target,
                        relationship_type=predicate,
                        description=triple.get("description", ""),
                        chapter=chapter_number,
                        confidence=0.8,
                    )
                )

        logger.debug(
            "_extract_updates_as_models: extraction complete",
            characters=len(char_updates),
            world_items=len(world_updates),
            relationships=len(relationships),
        )

        return char_updates, world_updates, relationships

    except Exception as e:
        logger.error(
            "_extract_updates_as_models: error processing extraction data",
            error=str(e),
            chapter=chapter_number,
            exc_info=True,
        )
        return [], [], []


async def _parse_extraction_json(
    raw_text: str, chapter_number: int
) -> dict[str, Any] | None:
    """
    Parse and clean LLM JSON response.

    PORTED FROM: KnowledgeAgent._parse_extraction_json()

    This function handles common LLM JSON formatting issues:
    - Markdown code blocks
    - Trailing commas
    - Incorrect quote characters
    - Malformed JSON structure

    Args:
        raw_text: Raw text from LLM
        chapter_number: Chapter number for logging

    Returns:
        Parsed dictionary or None if parsing fails
    """
    if not raw_text or not raw_text.strip():
        logger.warning(
            "_parse_extraction_json: empty extraction text", chapter=chapter_number
        )
        return None

    # Clean up common LLM JSON formatting issues
    cleaned_text = _clean_llm_json(raw_text)

    try:
        extraction_data = json.loads(cleaned_text)

        if not isinstance(extraction_data, dict):
            logger.error(
                "_parse_extraction_json: extraction JSON was not a dictionary",
                chapter=chapter_number,
            )
            return None

        return extraction_data

    except json.JSONDecodeError as e:
        logger.warning(
            "_parse_extraction_json: JSON parsing failed",
            error=str(e),
            chapter=chapter_number,
        )
        logger.error("_parse_extraction_json: failed raw_text", raw_text=raw_text)
        # Return minimal structure to avoid breaking the pipeline
        return {
            "character_updates": {},
            "world_updates": {},
            "kg_triples": [],
        }


def _clean_llm_json(raw_text: str) -> str:
    """
    Clean up common LLM JSON formatting issues.

    PORTED FROM: KnowledgeAgent._clean_llm_json()

    Handles:
    - Markdown code blocks (```json ... ```)
    - Trailing commas before closing brackets
    - Incorrect quote characters (curly quotes)

    Args:
        raw_text: Raw text from LLM

    Returns:
        Cleaned JSON text
    """
    # Remove markdown code blocks if present
    if raw_text.strip().startswith("```"):
        lines = raw_text.strip().split("\n")
        if len(lines) > 2:
            # Remove first and last lines (markdown markers)
            cleaned = "\n".join(lines[1:-1])
        else:
            cleaned = raw_text
    else:
        cleaned = raw_text

    # Remove common trailing commas before closing brackets/braces
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    # Fix common quote issues (curly quotes to straight quotes)
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')

    return cleaned.strip()


def _map_category_to_type(category: str) -> str:
    """
    Map world item category to entity type for ExtractedEntity.

    This provides a consistent mapping from the flexible category naming
    used in world_updates to the strict Literal types in ExtractedEntity.

    Args:
        category: Category string from world_updates

    Returns:
        One of: "location", "object", "event", "character"
    """
    category_lower = category.lower()

    # Location-related categories
    location_keywords = [
        "location",
        "place",
        "settlement",
        "region",
        "structure",
        "landmark",
        "territory",
        "room",
        "path",
    ]
    if any(keyword in category_lower for keyword in location_keywords):
        return "location"

    # Event-related categories
    event_keywords = ["event", "ceremony", "battle", "incident", "moment", "era"]
    if any(keyword in category_lower for keyword in event_keywords):
        return "event"

    # Character-related (rare in world_updates, but possible)
    character_keywords = ["character", "person", "creature", "spirit", "deity"]
    if any(keyword in category_lower for keyword in character_keywords):
        return "character"

    # Default to object for everything else
    # (artifacts, items, resources, documents, etc.)
    return "object"


__all__ = ["extract_entities"]
