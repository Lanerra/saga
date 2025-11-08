"""
File Persistence Node for Initialization Data.

This node writes initialization data (character sheets, outlines) to human-readable
YAML/Markdown files following the SAGA 2.0 file system structure.

File System Structure (from docs/langgraph-architecture.md):
    /my-novel/
    ├── outline/
    │   ├── structure.yaml         # Act/scene structure
    │   └── beats.yaml             # Key plot beats
    ├── characters/
    │   ├── protagonist.yaml
    │   └── {character}.yaml
    └── world/
        ├── setting.yaml
        └── items.yaml
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import structlog
import yaml

from core.langgraph.state import NarrativeState

logger = structlog.get_logger(__name__)


async def persist_initialization_files(state: NarrativeState) -> NarrativeState:
    """
    Write initialization data to human-readable files on disk.

    Following the file system structure from langgraph-architecture.md section 6,
    this node creates:
    - characters/{name}.yaml for each character
    - outline/structure.yaml with global and act outlines
    - outline/beats.yaml with key plot beats
    - world/items.yaml with world items

    Args:
        state: Current narrative state with initialization data

    Returns:
        Updated state with file persistence complete
    """
    logger.info(
        "persist_initialization_files: writing initialization data to disk",
        project_dir=state["project_dir"],
    )

    project_dir = Path(state["project_dir"])

    try:
        # Create directory structure
        _create_directory_structure(project_dir)

        # Write character files
        character_sheets = state.get("character_sheets", {})
        if character_sheets:
            _write_character_files(project_dir, character_sheets)

        # Write outline files
        global_outline = state.get("global_outline")
        act_outlines = state.get("act_outlines", {})
        if global_outline or act_outlines:
            _write_outline_files(project_dir, global_outline, act_outlines, state)

        # Write world items file
        world_items = state.get("world_items", [])
        if world_items:
            _write_world_items_file(project_dir, world_items, state.get("setting", ""))

        logger.info(
            "persist_initialization_files: successfully wrote all files",
            characters=len(character_sheets),
            acts=len(act_outlines),
            world_items=len(world_items),
        )

        return {
            **state,
            "current_node": "persist_files",
            "last_error": None,
            "initialization_step": "files_persisted",
        }

    except Exception as e:
        error_msg = f"Failed to persist initialization files: {e}"
        logger.error(
            "persist_initialization_files: error writing files",
            error=str(e),
            exc_info=True,
        )
        return {
            **state,
            "current_node": "persist_files",
            "last_error": error_msg,
            "initialization_step": "file_persistence_failed",
        }


def _create_directory_structure(project_dir: Path) -> None:
    """Create the SAGA 2.0 directory structure."""
    directories = [
        project_dir / ".saga",
        project_dir / "outline",
        project_dir / "characters",
        project_dir / "world",
        project_dir / "chapters",
        project_dir / "summaries",
        project_dir / "exports",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    logger.debug(
        "_create_directory_structure: created directories",
        project_dir=str(project_dir),
    )


def _write_character_files(project_dir: Path, character_sheets: dict) -> None:
    """
    Write individual YAML files for each character.

    Format:
        characters/{character_name}.yaml
    """
    characters_dir = project_dir / "characters"

    for name, sheet in character_sheets.items():
        # Sanitize filename
        safe_name = name.lower().replace(" ", "_").replace("'", "")
        file_path = characters_dir / f"{safe_name}.yaml"

        character_data = {
            "name": name,
            "role": "protagonist" if sheet.get("is_protagonist") else "character",
            "description": sheet.get("description", ""),
            "generated_at": datetime.now().isoformat(),
            "source": "initialization",
        }

        with open(file_path, "w") as f:
            yaml.dump(character_data, f, default_flow_style=False, sort_keys=False)

        logger.debug(
            "_write_character_files: wrote character file",
            character=name,
            path=str(file_path),
        )


def _write_outline_files(
    project_dir: Path,
    global_outline: dict | None,
    act_outlines: dict,
    state: NarrativeState,
) -> None:
    """
    Write outline structure and beats to YAML files.

    Creates:
    - outline/structure.yaml: Act/scene structure
    - outline/beats.yaml: Key plot beats
    """
    outline_dir = project_dir / "outline"

    # Write structure.yaml
    structure_data = {
        "title": state["title"],
        "genre": state["genre"],
        "theme": state.get("theme", ""),
        "setting": state.get("setting", ""),
        "total_chapters": state.get("total_chapters", 20),
        "target_word_count": state.get("target_word_count", 80000),
        "generated_at": datetime.now().isoformat(),
    }

    if global_outline:
        structure_data["global_outline"] = {
            "act_count": global_outline.get("act_count", 3),
            "structure_type": global_outline.get("structure_type", "3-act"),
            "summary": global_outline.get("raw_text", "")[:500] + "...",  # Truncate
        }

    if act_outlines:
        structure_data["acts"] = {}
        for act_num, act_data in sorted(act_outlines.items()):
            structure_data["acts"][f"act_{act_num}"] = {
                "act_number": act_num,
                "role": act_data.get("act_role", ""),
                "chapters": act_data.get("chapters_in_act", 0),
                "summary": act_data.get("raw_text", "")[:300] + "...",  # Truncate
            }

    structure_path = outline_dir / "structure.yaml"
    with open(structure_path, "w") as f:
        yaml.dump(structure_data, f, default_flow_style=False, sort_keys=False)

    logger.debug(
        "_write_outline_files: wrote structure file",
        path=str(structure_path),
    )

    # Write beats.yaml with full outline text
    beats_data = {
        "generated_at": datetime.now().isoformat(),
        "source": "initialization",
    }

    if global_outline:
        beats_data["global_outline"] = global_outline.get("raw_text", "")

    if act_outlines:
        beats_data["act_outlines"] = {}
        for act_num, act_data in sorted(act_outlines.items()):
            beats_data["act_outlines"][f"act_{act_num}"] = {
                "act_number": act_num,
                "role": act_data.get("act_role", ""),
                "outline": act_data.get("raw_text", ""),
            }

    beats_path = outline_dir / "beats.yaml"
    with open(beats_path, "w") as f:
        yaml.dump(beats_data, f, default_flow_style=False, sort_keys=False)

    logger.debug(
        "_write_outline_files: wrote beats file",
        path=str(beats_path),
    )


def _write_world_items_file(
    project_dir: Path, world_items: list, setting: str
) -> None:
    """
    Write world items to YAML file.

    Creates:
    - world/items.yaml: Locations, objects, concepts
    """
    world_dir = project_dir / "world"

    world_data = {
        "setting": setting,
        "generated_at": datetime.now().isoformat(),
        "source": "initialization",
        "items": [],
    }

    for item in world_items:
        world_data["items"].append(
            {
                "id": item.id,
                "name": item.name,
                "category": item.category,
                "description": item.description,
            }
        )

    items_path = world_dir / "items.yaml"
    with open(items_path, "w") as f:
        yaml.dump(world_data, f, default_flow_style=False, sort_keys=False)

    logger.debug(
        "_write_world_items_file: wrote world items file",
        path=str(items_path),
        count=len(world_items),
    )


__all__ = ["persist_initialization_files"]
