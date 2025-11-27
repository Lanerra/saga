# core/langgraph/nodes/scene_planning_node.py
import json

import structlog

from core.langgraph.content_manager import ContentManager, get_chapter_outlines
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from data_access.character_queries import get_all_character_names, sync_characters
from models.kg_models import CharacterProfile
from prompts.prompt_renderer import get_system_prompt, render_prompt
from utils.text_processing import normalize_entity_name

logger = structlog.get_logger(__name__)


async def _ensure_scene_characters_exist(
    chapter_plan: list[dict],
    chapter_number: int,
) -> None:
    """
    Ensure all characters referenced in scene plans exist in Neo4j.

    Creates stub profiles (is_provisional=True) for any new characters
    that the LLM introduced in scene plans but don't exist in the knowledge graph.

    Args:
        chapter_plan: List of scene dictionaries with characters_involved
        chapter_number: Current chapter number for tracking
    """
    # Extract all unique character names from scene plans
    scene_characters = set()
    for scene in chapter_plan:
        # Check various possible field names for character lists
        for field in ["characters", "characters_involved", "character_list", "cast"]:
            chars = scene.get(field)
            if chars:
                if isinstance(chars, list):
                    for char in chars:
                        if isinstance(char, str) and char.strip():
                            clean_name = normalize_entity_name(char)
                            if clean_name:
                                scene_characters.add(clean_name)
                        elif isinstance(char, dict) and char.get("name"):
                            clean_name = normalize_entity_name(char["name"])
                            if clean_name:
                                scene_characters.add(clean_name)
                elif isinstance(chars, str):
                    # Comma-separated list
                    for c in chars.split(","):
                        if c.strip():
                            clean_name = normalize_entity_name(c)
                            if clean_name:
                                scene_characters.add(clean_name)
                break

    if not scene_characters:
        logger.debug(
            "_ensure_scene_characters_exist: no characters found in scene plans"
        )
        return

    # Get existing characters from Neo4j
    try:
        existing_names = await get_all_character_names()
        existing_names_set = set(existing_names)
    except Exception as e:
        logger.error(
            "_ensure_scene_characters_exist: failed to fetch existing characters",
            error=str(e),
        )
        return

    # Find new characters that need stub profiles
    new_characters = scene_characters - existing_names_set

    if not new_characters:
        logger.debug(
            "_ensure_scene_characters_exist: all scene characters exist in Neo4j",
            character_count=len(scene_characters),
        )
        return

    logger.info(
        "_ensure_scene_characters_exist: creating stub profiles for new characters",
        new_characters=list(new_characters),
        count=len(new_characters),
    )

    # Create stub profiles for new characters
    stub_profiles = []
    for char_name in new_characters:
        stub = CharacterProfile(
            name=char_name,
            description=f"Character appearing in chapter {chapter_number}. Role and background to be developed through narrative.",
            traits=["to_be_developed"],  # Marker trait for provisional characters
            relationships={},
            status="Active",  # Default to Active so they can participate in scenes
            created_chapter=chapter_number,
            is_provisional=True,
        )
        stub_profiles.append(stub)

    # Persist stub profiles to Neo4j
    try:
        success = await sync_characters(stub_profiles, chapter_number)
        if success:
            logger.info(
                "_ensure_scene_characters_exist: successfully created stub profiles",
                count=len(stub_profiles),
            )

            # Link stub profiles to their chapter for context retrieval during enrichment
            from core.db_manager import neo4j_manager

            for char_name in new_characters:
                link_query = """
                    MATCH (c:Character {name: $char_name}), (chap:Chapter {number: $chapter})
                    WHERE c.is_provisional = true
                    MERGE (c)-[:MENTIONED_IN]->(chap)
                """
                try:
                    await neo4j_manager.execute_write_query(
                        link_query, {"char_name": char_name, "chapter": chapter_number}
                    )
                except Exception as link_error:
                    logger.debug(
                        "_ensure_scene_characters_exist: could not link to chapter (may not exist yet)",
                        char_name=char_name,
                        error=str(link_error),
                    )
        else:
            logger.warning(
                "_ensure_scene_characters_exist: failed to persist stub profiles"
            )
    except Exception as e:
        logger.error(
            "_ensure_scene_characters_exist: error persisting stub profiles",
            error=str(e),
            exc_info=True,
        )


async def plan_scenes(state: NarrativeState) -> NarrativeState:
    """
    Break the chapter into scenes based on the outline.
    """
    logger.info(
        "plan_scenes: planning scenes for chapter", chapter=state["current_chapter"]
    )

    chapter_number = state["current_chapter"]

    # Initialize content manager and get outlines
    content_manager = ContentManager(state["project_dir"])
    chapter_outlines = get_chapter_outlines(state, content_manager)
    outline = chapter_outlines.get(chapter_number)

    if not outline:
        logger.error(
            "plan_scenes: no outline found for chapter", chapter=chapter_number
        )
        return {
            **state,
            "last_error": f"No outline found for chapter {chapter_number}",
            "current_node": "plan_scenes",
        }

    # Determine number of scenes (heuristic or config)
    # For now, we'll ask for 3-5 scenes depending on complexity, or just default to 3
    num_scenes = 3

    prompt = render_prompt(
        "narrative_agent/plan_scenes.j2",
        {
            "novel_title": state["title"],
            "novel_genre": state["genre"],
            "novel_theme": state["theme"],
            "chapter_number": chapter_number,
            "outline": outline,
            "num_scenes": num_scenes,
        },
    )

    try:
        response, _ = await llm_service.async_call_llm(
            model_name=state["large_model"],
            prompt=prompt,
            temperature=0.7,
            max_tokens=2000,
            system_prompt=get_system_prompt("narrative_agent"),
        )

        # Parse JSON response
        # The LLM might wrap it in markdown code blocks, so we need to clean it
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]

        scenes = json.loads(cleaned_response)

        if not isinstance(scenes, list):
            raise ValueError("LLM response is not a list of scenes")

        logger.info("plan_scenes: successfully planned scenes", count=len(scenes))

        # Ensure all characters in the scene plans exist in Neo4j
        # This creates stub profiles for any new characters the LLM introduced
        await _ensure_scene_characters_exist(scenes, chapter_number)

        return {
            "chapter_plan": scenes,
            "current_scene_index": 0,
            "scene_drafts_ref": None,
            "current_node": "plan_scenes",
        }

    except Exception as e:
        logger.error("plan_scenes: error planning scenes", error=str(e))
        return {
            **state,
            "last_error": f"Error planning scenes: {str(e)}",
            "current_node": "plan_scenes",
        }
