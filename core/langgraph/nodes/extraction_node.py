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
        extraction_model=state["medium_model"],
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
                        type="Character",  # Capitalized to match ontology node types
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
    - Truncated JSON responses

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

        # Attempt to repair truncated JSON
        repaired_data = _attempt_json_repair(cleaned_text, chapter_number)
        if repaired_data:
            logger.info(
                "_parse_extraction_json: successfully repaired truncated JSON",
                chapter=chapter_number,
            )
            return repaired_data

        logger.error(
            "_parse_extraction_json: failed raw_text",
            raw_text=raw_text[:500] + "..." if len(raw_text) > 500 else raw_text,
        )
        # Return minimal structure to avoid breaking the pipeline
        return {
            "character_updates": {},
            "world_updates": {},
            "kg_triples": [],
        }


def _attempt_json_repair(text: str, chapter_number: int) -> dict[str, Any] | None:
    """
    Attempt to repair truncated or malformed JSON.

    This function tries multiple strategies to recover data from
    partially valid JSON:
    1. Close open brackets/braces
    2. Extract valid portions
    3. Parse individual sections

    Args:
        text: Cleaned but malformed JSON text
        chapter_number: Chapter number for logging

    Returns:
        Repaired dictionary or None if repair fails
    """
    # Strategy 1: Try to close unclosed brackets/braces
    repaired = _close_json_brackets(text)
    try:
        data = json.loads(repaired)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Strategy 2: Find and extract the main JSON object
    # Look for the opening brace and try to find as much valid JSON as possible
    brace_start = text.find("{")
    if brace_start == -1:
        return None

    json_text = text[brace_start:]

    # Try progressively shorter versions until we get valid JSON
    for end_pos in range(len(json_text), 0, -100):
        attempt = json_text[:end_pos]
        closed = _close_json_brackets(attempt)
        try:
            data = json.loads(closed)
            if isinstance(data, dict):
                logger.debug(
                    "_attempt_json_repair: recovered partial JSON",
                    chapter=chapter_number,
                    original_length=len(text),
                    recovered_length=end_pos,
                )
                return data
        except json.JSONDecodeError:
            continue

    # Strategy 3: Try to extract individual sections
    result = {
        "character_updates": {},
        "world_updates": {},
        "kg_triples": [],
    }

    # Try to extract character_updates
    char_match = re.search(
        r'"character_updates"\s*:\s*(\{[^}]*(?:\{[^}]*\}[^}]*)*\})',
        text,
        re.DOTALL
    )
    if char_match:
        try:
            result["character_updates"] = json.loads(char_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to extract kg_triples
    triples_match = re.search(
        r'"kg_triples"\s*:\s*\[(.*?)\]',
        text,
        re.DOTALL
    )
    if triples_match:
        # Parse the array content
        triples_content = triples_match.group(1)
        # Extract individual quoted strings
        triple_strings = re.findall(r'"([^"]*)"', triples_content)
        result["kg_triples"] = triple_strings

    # If we recovered anything, return it
    if result["character_updates"] or result["kg_triples"]:
        logger.debug(
            "_attempt_json_repair: extracted partial data",
            chapter=chapter_number,
            characters=len(result["character_updates"]),
            triples=len(result["kg_triples"]),
        )
        return result

    return None


def _close_json_brackets(text: str) -> str:
    """
    Close unclosed JSON brackets and braces.

    Counts open brackets/braces and adds closing ones as needed.

    Args:
        text: JSON text that may be truncated

    Returns:
        Text with brackets closed
    """
    # Count brackets
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            open_braces += 1
        elif char == "}":
            open_braces -= 1
        elif char == "[":
            open_brackets += 1
        elif char == "]":
            open_brackets -= 1

    # If we're in a string, close it
    if in_string:
        text += '"'

    # Remove trailing comma before closing
    text = re.sub(r',\s*$', '', text)

    # Add closing brackets/braces
    result = text
    for _ in range(max(0, open_brackets)):
        result += "]"
    for _ in range(max(0, open_braces)):
        result += "}"

    return result


def _clean_llm_json(raw_text: str) -> str:
    """
    Clean up common LLM JSON formatting issues.

    PORTED FROM: KnowledgeAgent._clean_llm_json()

    Handles:
    - Markdown code blocks (```json ... ```)
    - Trailing commas before closing brackets
    - Incorrect quote characters (curly quotes)
    - Truncated responses without closing markdown

    Args:
        raw_text: Raw text from LLM

    Returns:
        Cleaned JSON text
    """
    cleaned = raw_text.strip()

    # Remove markdown code blocks if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")

        # Find the first line (skip the opening ```)
        start_idx = 1

        # Find the closing ``` - it might not exist if truncated
        end_idx = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break

        # Extract content between markers
        if end_idx > start_idx:
            cleaned = "\n".join(lines[start_idx:end_idx])
        elif len(lines) > 1:
            # No closing ```, just skip the first line
            cleaned = "\n".join(lines[1:])

    # Also handle case where ``` appears at the end of the content
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    # Remove any remaining ``` markers that might be embedded
    cleaned = re.sub(r'^```\w*\n?', '', cleaned)
    cleaned = re.sub(r'\n?```$', '', cleaned)

    # Remove common trailing commas before closing brackets/braces
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    # Fix common quote issues (curly quotes to straight quotes)
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")

    # Remove any BOM or other unicode artifacts
    cleaned = cleaned.lstrip('\ufeff')

    return cleaned.strip()


def _map_category_to_type(category: str) -> str:
    """
    Map world item category to entity type for ExtractedEntity.

    This function now accepts specific node types directly from the ontology.
    If the category matches a known node label, it returns it as-is.
    Otherwise, it falls back to heuristic mapping for backward compatibility.

    Args:
        category: Category string from world_updates (should be a specific node type)

    Returns:
        A valid node type (preferably specific like "DevelopmentEvent", "PlotPoint", etc.)
    """
    from models.kg_constants import NODE_LABELS

    # First, check if category is already a valid node label (case-insensitive match)
    for node_label in NODE_LABELS:
        if category.lower() == node_label.lower():
            return node_label  # Return the canonical form from NODE_LABELS

    # Fallback: Try to map using heuristics for backward compatibility
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
        # Return more specific types if keywords match
        if "settlement" in category_lower:
            return "Settlement"
        elif "structure" in category_lower:
            return "Structure"
        elif "region" in category_lower:
            return "Region"
        elif "landmark" in category_lower:
            return "Landmark"
        elif "room" in category_lower:
            return "Room"
        elif "path" in category_lower:
            return "Path"
        elif "territory" in category_lower:
            return "Territory"
        return "Location"  # Generic fallback

    # Event-related categories
    event_keywords = [
        "event",
        "developmentevent",
        "worldelaborationevent",
        "plotpoint",
        "ceremony",
        "battle",
        "incident",
        "moment",
        "era",
    ]
    if any(keyword in category_lower for keyword in event_keywords):
        # Return specific event types if available
        if "development" in category_lower:
            return "DevelopmentEvent"
        elif "elaboration" in category_lower or "worldelaboration" in category_lower:
            return "WorldElaborationEvent"
        elif "plot" in category_lower:
            return "PlotPoint"
        elif "era" in category_lower:
            return "Era"
        elif "moment" in category_lower:
            return "Moment"
        return "Event"  # Generic fallback

    # Organization-related categories
    org_keywords = ["faction", "organization", "guild", "house", "order", "council"]
    if any(keyword in category_lower for keyword in org_keywords):
        if "faction" in category_lower:
            return "Faction"
        elif "guild" in category_lower:
            return "Guild"
        elif "house" in category_lower:
            return "House"
        elif "order" in category_lower:
            return "Order"
        elif "council" in category_lower:
            return "Council"
        return "Organization"

    # Character-related (rare in world_updates, but possible)
    character_keywords = ["character", "person", "creature", "spirit", "deity"]
    if any(keyword in category_lower for keyword in character_keywords):
        if "person" in category_lower:
            return "Person"
        elif "creature" in category_lower:
            return "Creature"
        elif "spirit" in category_lower:
            return "Spirit"
        elif "deity" in category_lower:
            return "Deity"
        return "Character"

    # Object-related categories
    object_keywords = ["artifact", "document", "relic", "item", "object", "resource", "currency"]
    if any(keyword in category_lower for keyword in object_keywords):
        if "artifact" in category_lower:
            return "Artifact"
        elif "document" in category_lower:
            return "Document"
        elif "relic" in category_lower:
            return "Relic"
        elif "resource" in category_lower:
            return "Resource"
        elif "currency" in category_lower:
            return "Currency"
        return "Object"

    # System-related categories
    system_keywords = ["magic", "technology", "religion", "culture", "system"]
    if any(keyword in category_lower for keyword in system_keywords):
        if "magic" in category_lower:
            return "Magic"
        elif "technology" in category_lower or "tech" in category_lower:
            return "Technology"
        elif "religion" in category_lower:
            return "Religion"
        elif "culture" in category_lower:
            return "Culture"
        return "System"

    # Information-related categories
    info_keywords = ["lore", "knowledge", "secret", "rumor", "trait"]
    if any(keyword in category_lower for keyword in info_keywords):
        if "lore" in category_lower:
            return "Lore"
        elif "secret" in category_lower:
            return "Secret"
        elif "rumor" in category_lower:
            return "Rumor"
        elif "trait" in category_lower:
            return "Trait"
        return "Knowledge"

    # Default to Object for unrecognized categories
    logger.warning(
        "_map_category_to_type: unrecognized category, defaulting to Object",
        category=category,
    )
    return "Object"


__all__ = ["extract_entities"]
