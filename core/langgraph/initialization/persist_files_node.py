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


# Custom YAML string class for literal block scalar (|) formatting
class _LiteralString(str):
    """String subclass that renders as literal block scalar in YAML."""
    pass


def _literal_string_representer(dumper, data):
    """YAML representer for _LiteralString to use literal block scalar."""
    if len(data.splitlines()) > 1:  # Only use block scalar for multiline
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


# Register the custom representer
yaml.add_representer(_LiteralString, _literal_string_representer)


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

        # Write world items file (always, even if empty)
        world_items = state.get("world_items", [])
        _write_world_items_file(project_dir, world_items, state.get("setting", ""))

        # Write placeholder summaries README
        _write_summaries_readme(project_dir)

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


def _parse_character_sheet_text(text: str) -> dict:
    """
    Parse character sheet text into structured fields.

    Looks for common section headers like:
    - Physical Description
    - Personality
    - Background
    - Motivations
    - Skills/Abilities
    - Relationships
    - Character Arc
    """
    import re

    # Common section patterns
    sections = {
        "physical_description": r"\*\*Physical Description:\*\*\s*(.*?)(?=\*\*|\Z)",
        "personality": r"\*\*Personality:\*\*\s*(.*?)(?=\*\*|\Z)",
        "background": r"\*\*Background:\*\*\s*(.*?)(?=\*\*|\Z)",
        "motivations": r"\*\*Motivations:\*\*\s*(.*?)(?=\*\*|\Z)",
        "skills": r"\*\*Skills/Abilities:\*\*\s*(.*?)(?=\*\*|\Z)",
        "relationships": r"\*\*Relationships:\*\*\s*(.*?)(?=\*\*|\Z)",
        "character_arc": r"\*\*Character Arc:\*\*\s*(.*?)(?=\*\*|\Z)",
    }

    parsed = {}
    for field, pattern in sections.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            # Clean up the text
            content = match.group(1).strip()
            # Remove excessive whitespace
            content = re.sub(r'\s+', ' ', content)
            parsed[field] = content

    # If we couldn't parse anything, fall back to raw description
    if not parsed:
        parsed["description"] = text

    return parsed


def _write_character_files(project_dir: Path, character_sheets: dict) -> None:
    """
    Write individual YAML files for each character.

    Format:
        characters/{character_name}.yaml

    Parses the character sheet text into structured fields for readability.
    """
    characters_dir = project_dir / "characters"

    for name, sheet in character_sheets.items():
        # Sanitize filename
        safe_name = name.lower().replace(" ", "_").replace("'", "")
        file_path = characters_dir / f"{safe_name}.yaml"

        # Parse character sheet into structured fields
        description_text = sheet.get("description", "")
        parsed_data = _parse_character_sheet_text(description_text)

        character_data = {
            "name": name,
            "role": "protagonist" if sheet.get("is_protagonist") else "character",
            **parsed_data,  # Merge parsed fields
            "generated_at": datetime.now().isoformat(),
            "source": "initialization",
        }

        with open(file_path, "w") as f:
            yaml.dump(character_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=100)

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
    # Use literal block scalar for better markdown readability
    beats_data = {
        "generated_at": datetime.now().isoformat(),
        "source": "initialization",
    }

    if global_outline:
        raw_text = global_outline.get("raw_text", "")
        # Use literal block scalar class for multiline text
        beats_data["global_outline"] = _LiteralString(raw_text)

    if act_outlines:
        beats_data["act_outlines"] = {}
        for act_num, act_data in sorted(act_outlines.items()):
            raw_text = act_data.get("raw_text", "")
            beats_data["act_outlines"][f"act_{act_num}"] = {
                "act_number": act_num,
                "role": act_data.get("act_role", ""),
                "outline": _LiteralString(raw_text),
            }

    beats_path = outline_dir / "beats.yaml"
    with open(beats_path, "w") as f:
        yaml.dump(
            beats_data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=100,
        )

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


def _write_summaries_readme(project_dir: Path) -> None:
    """
    Write README file in summaries/ directory.

    The summaries directory is populated during chapter generation,
    not initialization. This creates a placeholder README to explain that.
    """
    summaries_dir = project_dir / "summaries"
    readme_path = summaries_dir / "README.md"

    readme_content = """# Chapter Summaries

This directory will contain chapter summaries as they are generated.

Summaries are created automatically during chapter generation and saved as:
- `chapter_001_summary.txt`
- `chapter_002_summary.txt`
- etc.

These summaries are used to maintain narrative coherence across chapters
by providing context to the LLM during generation of subsequent chapters.
"""

    with open(readme_path, "w") as f:
        f.write(readme_content)

    logger.debug(
        "_write_summaries_readme: wrote summaries README",
        path=str(readme_path),
    )


__all__ = ["persist_initialization_files"]
