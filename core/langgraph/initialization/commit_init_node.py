"""
Commit Initialization Data to Knowledge Graph Node.

This node bridges the initialization phase to the generation loop by converting
initialization data (character sheets, outlines) into Neo4j-compatible models
and committing them to the knowledge graph.
"""

from __future__ import annotations

import structlog

import config
from core.knowledge_graph_service import knowledge_graph_service
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from models.kg_models import CharacterProfile, WorldItem
from prompts.prompt_renderer import get_system_prompt

logger = structlog.get_logger(__name__)


async def commit_initialization_to_graph(state: NarrativeState) -> NarrativeState:
    """
    Convert initialization data to Neo4j models and commit to knowledge graph.

    This node acts as a bridge between the initialization phase (which generates
    rich text descriptions) and the generation loop (which needs structured
    CharacterProfile and WorldItem models in Neo4j).

    Process Flow:
    1. Parse character sheets to extract structured traits
    2. Convert to CharacterProfile models
    3. Extract world elements from outlines
    4. Convert to WorldItem models
    5. Commit all to Neo4j using existing persistence layer

    Args:
        state: Current narrative state with character_sheets and outlines

    Returns:
        Updated state with initialization data committed to graph
    """
    logger.info(
        "commit_initialization_to_graph: starting initialization data commit",
        characters=len(state.get("character_sheets", {})),
        has_global_outline=bool(state.get("global_outline")),
    )

    character_sheets = state.get("character_sheets", {})
    global_outline = state.get("global_outline")

    if not character_sheets:
        logger.warning("commit_initialization_to_graph: no character sheets to commit")

    try:
        # Step 1: Parse character sheets into CharacterProfile models
        character_profiles = []
        if character_sheets:
            character_profiles = await _parse_character_sheets_to_profiles(
                character_sheets
            )

        # Step 2: Extract world items from outlines
        world_items = []
        if global_outline:
            world_items = await _extract_world_items_from_outline(
                global_outline, state.get("setting", "")
            )

        # Step 3: Commit to Neo4j using existing persistence layer
        if character_profiles or world_items:
            success = await knowledge_graph_service.persist_entities(
                character_profiles,
                world_items,
                chapter_number=0,  # Initialization entities exist before any chapters
            )

            if not success:
                logger.warning(
                    "commit_initialization_to_graph: persistence returned failure flag"
                )

        logger.info(
            "commit_initialization_to_graph: successfully committed initialization data",
            characters=len(character_profiles),
            world_items=len(world_items),
        )

        # Step 4: Update active_characters with committed profiles
        # This makes characters immediately available to the generation loop
        updated_state = {
            **state,
            "active_characters": character_profiles[:5],  # Top 5 for initial context
            "world_items": world_items,
            "current_node": "commit_initialization",
            "last_error": None,
            "initialization_step": "committed_to_graph",
        }

        return updated_state

    except Exception as e:
        error_msg = f"Failed to commit initialization data: {e}"
        logger.error(
            "commit_initialization_to_graph: error during commit",
            error=str(e),
            exc_info=True,
        )
        return {
            **state,
            "current_node": "commit_initialization",
            "last_error": error_msg,
            "initialization_step": "commit_failed",
        }


async def _parse_character_sheets_to_profiles(
    character_sheets: dict[str, dict],
) -> list[CharacterProfile]:
    """
    Parse character sheet text descriptions into structured CharacterProfile models.

    Uses LLM to extract structured information (traits, status, etc.) from
    the free-form character sheet descriptions.

    Args:
        character_sheets: Dict of character_name -> character_sheet

    Returns:
        List of CharacterProfile models ready for Neo4j persistence
    """
    profiles = []

    for name, sheet in character_sheets.items():
        description = sheet.get("description", "")
        is_protagonist = sheet.get("is_protagonist", False)

        # Use LLM to extract structured traits from description
        structured_data = await _extract_structured_character_data(name, description)

        # Create CharacterProfile model
        profile = CharacterProfile(
            name=name,
            description=description,
            traits=structured_data.get("traits", []),
            status=structured_data.get("status", "Active"),
            relationships={},  # Will be populated during generation
            created_chapter=0,  # Initialization entities created before chapters
            is_provisional=False,  # Initialization characters are canonical
            updates={
                "is_protagonist": is_protagonist,
                "motivations": structured_data.get("motivations", ""),
                "background": structured_data.get("background", ""),
            },
        )

        profiles.append(profile)
        logger.debug(
            "_parse_character_sheets_to_profiles: created profile",
            name=name,
            traits=len(profile.traits),
        )

    return profiles


async def _extract_structured_character_data(
    name: str, description: str
) -> dict[str, any]:
    """
    Use LLM to extract structured data from character sheet description.

    Args:
        name: Character name
        description: Free-form character description

    Returns:
        Dictionary with extracted structured data (traits, status, motivations, etc.)
    """
    # Simple prompt to extract structured info
    prompt = f"""Extract structured character information from this character sheet.

Character Name: {name}

Character Sheet:
{description}

Extract the following in a structured format:
1. **Traits** (list 3-7 key personality traits)
2. **Status** (their current state: Active, Deceased, Missing, etc.)
3. **Motivations** (what drives them, in 1-2 sentences)
4. **Background** (brief summary of history, 1-2 sentences)

Format your response as:
TRAITS: trait1, trait2, trait3
STATUS: Active
MOTIVATIONS: What the character wants and why
BACKGROUND: Brief history"""

    try:
        response, _ = await llm_service.async_call_llm(
            model_name=config.NARRATIVE_MODEL,
            prompt=prompt,
            temperature=0.3,  # Low temp for extraction
            max_tokens=500,
            allow_fallback=True,
            auto_clean_response=True,
            system_prompt=get_system_prompt("knowledge_agent"),
        )

        # Parse the response
        structured = _parse_character_extraction_response(response)
        return structured

    except Exception as e:
        logger.warning(
            "_extract_structured_character_data: extraction failed, using defaults",
            character=name,
            error=str(e),
        )
        # Return defaults on failure
        return {
            "traits": [],
            "status": "Active",
            "motivations": "",
            "background": "",
        }


def _parse_character_extraction_response(response: str) -> dict[str, any]:
    """
    Parse the LLM's structured character extraction response.

    Args:
        response: LLM response with structured data

    Returns:
        Dictionary with parsed structured data
    """
    structured = {
        "traits": [],
        "status": "Active",
        "motivations": "",
        "background": "",
    }

    lines = response.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("TRAITS:"):
            traits_str = line.replace("TRAITS:", "").strip()
            traits = [t.strip() for t in traits_str.split(",") if t.strip()]
            structured["traits"] = traits[:7]  # Limit to 7 traits
        elif line.startswith("STATUS:"):
            structured["status"] = line.replace("STATUS:", "").strip()
        elif line.startswith("MOTIVATIONS:"):
            structured["motivations"] = line.replace("MOTIVATIONS:", "").strip()
        elif line.startswith("BACKGROUND:"):
            structured["background"] = line.replace("BACKGROUND:", "").strip()

    return structured


async def _extract_world_items_from_outline(
    global_outline: dict, setting: str
) -> list[WorldItem]:
    """
    Extract world-building elements from the global outline.

    Uses LLM to identify key locations, objects, and world elements mentioned
    in the outline that should be added to the knowledge graph.

    Args:
        global_outline: Global outline with story structure
        setting: Story setting description

    Returns:
        List of WorldItem models ready for Neo4j persistence
    """
    outline_text = global_outline.get("raw_text", "")
    if not outline_text:
        return []

    # Use LLM to extract world items
    prompt = f"""Extract key world-building elements from this story outline.

Setting: {setting}

Story Outline:
{outline_text[:1500]}

Identify and list:
1. Important LOCATIONS (cities, buildings, regions)
2. Important OBJECTS (artifacts, items, technology)
3. Important CONCEPTS (magic systems, technologies, organizations)

For each item, provide:
- Name
- Category (Location, Object, Concept)
- Brief description (1 sentence)

Format each as: [CATEGORY] Name: Description

Example:
[Location] Seattle Public Library Ruins: Abandoned library now home to rogue AI
[Object] Memory Implant: Device that allows AI to implant false memories
[Concept] The Nexus: Central AI controlling all implanted humans"""

    try:
        response, _ = await llm_service.async_call_llm(
            model_name=config.NARRATIVE_MODEL,
            prompt=prompt,
            temperature=0.5,
            max_tokens=1000,
            allow_fallback=True,
            auto_clean_response=True,
            system_prompt=get_system_prompt("knowledge_agent"),
        )

        # Parse the response into WorldItem models
        world_items = _parse_world_items_extraction(response)
        return world_items

    except Exception as e:
        logger.warning(
            "_extract_world_items_from_outline: extraction failed",
            error=str(e),
        )
        return []


def _parse_world_items_extraction(response: str) -> list[WorldItem]:
    """
    Parse LLM response into WorldItem models.

    Args:
        response: LLM response with world items

    Returns:
        List of WorldItem models
    """
    items = []
    lines = response.split("\n")

    for line in lines:
        line = line.strip()
        if not line or not line.startswith("["):
            continue

        # Parse format: [CATEGORY] Name: Description
        try:
            # Extract category
            cat_end = line.index("]")
            category = line[1:cat_end].strip().lower()

            # Extract name and description
            rest = line[cat_end + 1 :].strip()
            if ":" not in rest:
                continue

            name, description = rest.split(":", 1)
            name = name.strip()
            description = description.strip()

            # Create WorldItem model
            from processing.entity_deduplication import generate_entity_id

            item = WorldItem(
                id=generate_entity_id(
                    name, category, chapter=0
                ),  # chapter=0 for initialization
                name=name,
                description=description,
                category=category,
                created_chapter=0,  # Initialization items created before chapters
                is_provisional=False,
            )

            items.append(item)

        except (ValueError, IndexError) as e:
            logger.debug(
                "_parse_world_items_extraction: failed to parse line",
                line=line,
                error=str(e),
            )
            continue

    logger.info(
        "_parse_world_items_extraction: extracted world items",
        count=len(items),
    )

    return items


__all__ = ["commit_initialization_to_graph"]
