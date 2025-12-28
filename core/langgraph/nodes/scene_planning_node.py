# core/langgraph/nodes/scene_planning_node.py
"""Plan scenes for chapter drafting.

This module defines the scene planning node used by the scene-based generation
workflow. It requests a structured scene plan from the LLM, validates the plan’s
shape, externalizes it, and ensures any newly introduced characters exist in
Neo4j (as provisional stubs) so downstream context retrieval can resolve them.
"""

import json
from json import JSONDecodeError
from typing import Any, cast

import structlog

import config
from core.langgraph.content_manager import (
    ContentManager,
    get_chapter_outlines,
    require_project_dir,
    save_chapter_plan,
)
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from data_access.character_queries import get_all_character_names, sync_characters
from models.agent_models import SceneDetail
from models.kg_models import CharacterProfile
from prompts.prompt_renderer import get_system_prompt, render_prompt
from utils.common import extract_json_candidates_from_response
from utils.text_processing import normalize_entity_name

logger = structlog.get_logger(__name__)


_SCENE_REQUIRED_KEYS: tuple[str, ...] = (
    "title",
    "pov_character",
    "setting",
    "characters",
    "plot_point",
    "conflict",
    "outcome",
)


def _validate_scene_plan_structure(scenes: Any) -> list[str]:
    """Validate the structure of a scene plan candidate.

    Args:
        scenes: Parsed JSON candidate.

    Returns:
        A list of validation error messages. An empty list means the structure is valid.
    """
    errors: list[str] = []

    if not isinstance(scenes, list):
        errors.append(f"Expected a JSON list of scenes, got {type(scenes).__name__}")
        return errors

    for i, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            errors.append(f"Scene[{i}] must be an object, got {type(scene).__name__}")
            continue

        missing = [k for k in _SCENE_REQUIRED_KEYS if k not in scene]
        if missing:
            errors.append(f"Scene[{i}] is missing required keys: {missing}")

        chars = scene.get("characters")
        if "characters" in scene:
            if not isinstance(chars, list) or not all(isinstance(c, str) and c.strip() for c in chars):
                errors.append(f"Scene[{i}].characters must be a non-empty list of character name strings")

    return errors


def _parse_scene_plan_json_from_llm_response(response: str) -> list[dict[str, Any]]:
    """Parse a validated scene plan list from an LLM response.

    Args:
        response: Raw LLM response text.

    Returns:
        A validated list of scene dictionaries.

    Raises:
        ValueError: When no JSON candidate contains a valid list of scenes with the
            required keys.
    """
    candidates = extract_json_candidates_from_response(response)
    if not candidates:
        raise ValueError("LLM returned an empty response; expected a JSON list of scene objects. " "Required keys per scene: " + ", ".join(_SCENE_REQUIRED_KEYS))

    parse_errors: list[str] = []

    for source, cand in candidates:
        try:
            parsed = json.loads(cand)
        except JSONDecodeError as e:
            parse_errors.append(f"{source}: JSONDecodeError at pos {e.pos}: {e.msg}")
            continue

        # Allow a wrapper object (some models return {"scenes": [...]})
        if isinstance(parsed, dict) and "scenes" in parsed:
            parsed = parsed["scenes"]

        structure_errors = _validate_scene_plan_structure(parsed)
        if structure_errors:
            parse_errors.append(f"{source}: invalid structure: {structure_errors}")
            continue

        return parsed  # type: ignore[return-value]

    # If we got here, no candidate worked.
    debug_sources = [s for s, _ in candidates[:5]]
    raise ValueError(
        "Failed to parse a valid scene plan JSON list from LLM response. "
        f"Tried candidates: {debug_sources}. "
        "Expected: a JSON list of scene objects with keys " + ", ".join(_SCENE_REQUIRED_KEYS) + ". Errors: " + "; ".join(parse_errors[:5])
    )


async def _ensure_scene_characters_exist(
    chapter_plan: list[dict],
    chapter_number: int,
) -> None:
    """Ensure all characters referenced by the plan exist in Neo4j.

    When the LLM introduces a new character in the scene plan, downstream context
    retrieval expects that character to exist in the knowledge graph. This helper
    creates provisional stub profiles for any missing names.

    Args:
        chapter_plan: Scene plan list produced by the planner.
        chapter_number: Chapter number used for provenance and linkage.

    Notes:
        This function performs Neo4j I/O and is best-effort. Failures are logged and
        do not raise, because the workflow can still proceed without stubs.
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
        logger.debug("_ensure_scene_characters_exist: no characters found in scene plans")
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

    try:
        success = await sync_characters(stub_profiles, chapter_number)
        if success:
            logger.info(
                "_ensure_scene_characters_exist: successfully created stub profiles",
                count=len(stub_profiles),
            )

            from core.db_manager import neo4j_manager

            for char_name in new_characters:
                link_query = """
                    MATCH (c:Character {name: $char_name}), (chap:Chapter {number: $chapter})
                    WHERE c.is_provisional = true
                    MERGE (c)-[:MENTIONED_IN]->(chap)
                """
                try:
                    await neo4j_manager.execute_write_query(link_query, {"char_name": char_name, "chapter": chapter_number})
                except Exception as link_error:
                    logger.debug(
                        "_ensure_scene_characters_exist: could not link to chapter (may not exist yet)",
                        char_name=char_name,
                        error=str(link_error),
                    )
        else:
            logger.warning("_ensure_scene_characters_exist: failed to persist stub profiles")
    except Exception as e:
        logger.error(
            "_ensure_scene_characters_exist: error persisting stub profiles",
            error=str(e),
            exc_info=True,
        )


async def plan_scenes(state: NarrativeState) -> NarrativeState:
    """Break the chapter outline into a structured list of scenes.

    Args:
        state: Workflow state. Reads chapter outline data (preferring externalized
            refs) and persists the resulting plan via `save_chapter_plan()`.

    Returns:
        Updated state containing:
        - chapter_plan_ref: Content reference for the externalized plan.
        - current_scene_index: Reset for drafting loop.
        - current_node: `"plan_scenes"`.

        If the outline is missing, returns an error update and does not set
        `has_fatal_error`.

    Notes:
        This node performs LLM I/O and may create provisional character stubs in
        Neo4j for any newly introduced names in the plan.
    """
    logger.info(
        "plan_scenes: planning scenes for chapter",
        chapter=state.get("current_chapter", 1),
    )

    chapter_number = state.get("current_chapter", 1)

    # Initialize content manager and get outlines
    content_manager = ContentManager(require_project_dir(state))
    chapter_outlines = get_chapter_outlines(state, content_manager)
    outline = chapter_outlines.get(chapter_number)

    if not outline:
        logger.error("plan_scenes: no outline found for chapter", chapter=chapter_number)
        return {
            "last_error": f"No outline found for chapter {chapter_number}",
            "current_node": "plan_scenes",
        }

    # Determine number of scenes (heuristic or config)
    # For now, we'll ask for 3-5 scenes depending on complexity, or just default to 3
    num_scenes = 3

    prompt = render_prompt(
        "narrative_agent/plan_scenes.j2",
        {
            "novel_title": state.get("title", ""),
            "novel_genre": state.get("genre", ""),
            "novel_theme": state.get("theme", ""),
            "chapter_number": chapter_number,
            "outline": outline,
            "num_scenes": num_scenes,
        },
    )

    try:
        response, _ = await llm_service.async_call_llm(
            model_name=state.get("large_model", config.LARGE_MODEL),
            prompt=prompt,
            temperature=0.7,
            max_tokens=16384,
            system_prompt=get_system_prompt("narrative_agent"),
        )

        scenes_untyped = _parse_scene_plan_json_from_llm_response(response)
        scenes = cast(list[SceneDetail], scenes_untyped)

        logger.info("plan_scenes: successfully planned scenes", count=len(scenes))

        # Ensure all characters in the scene plans exist in Neo4j
        # This creates stub profiles for any new characters the LLM introduced
        await _ensure_scene_characters_exist(scenes_untyped, chapter_number)

        # Externalize chapter_plan to reduce state bloat
        content_manager = ContentManager(require_project_dir(state))

        # Get current version for this chapter's plan
        current_version = content_manager.get_latest_version("chapter_plan", f"chapter_{chapter_number}") + 1

        # Save chapter plan to external file
        chapter_plan_ref = save_chapter_plan(
            content_manager,
            scenes_untyped,
            chapter_number,
            current_version,
        )

        logger.info(
            "plan_scenes: chapter plan externalized",
            chapter=chapter_number,
            version=current_version,
            plan_size=chapter_plan_ref["size_bytes"],
        )

        return {
            "chapter_plan_ref": chapter_plan_ref,
            "chapter_plan_scene_count": len(scenes),
            "current_scene_index": 0,
            "scene_drafts_ref": None,
            "current_node": "plan_scenes",
        }

    except Exception as e:
        logger.error("plan_scenes: error planning scenes", error=str(e))
        return {
            "last_error": ("Error planning scenes: " + str(e) + " | Expected: JSON list of scene objects with keys: " + ", ".join(_SCENE_REQUIRED_KEYS)),
            "current_node": "plan_scenes",
        }
