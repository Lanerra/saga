# core/langgraph/initialization/persist_files_node.py
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

from datetime import datetime
from pathlib import Path

import structlog
import yaml

from core.langgraph.content_manager import (
    ContentManager,
    get_act_outlines,
    get_character_sheets,
    get_global_outline,
)
from core.langgraph.state import NarrativeState
from utils.file_io import write_yaml_file

logger = structlog.get_logger(__name__)


# Custom YAML string class for literal block scalar (|) formatting
class _LiteralString(str):
    """String subclass that renders as literal block scalar in YAML."""


def _literal_string_representer(dumper, data: _LiteralString):
    """YAML representer for _LiteralString to use literal block scalar.

    Uses literal block style for multiline values to keep prose readable.
    Falls back to plain style for single-line values.

    Args:
        dumper: PyYAML dumper instance.
        data: String data to represent.

    Returns:
        YAML scalar node with appropriate style (literal block or plain).
    """
    text = str(data)
    if "\n" in text:
        return dumper.represent_scalar("tag:yaml.org,2002:str", text, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", text)


# Register the custom representer once at import time
yaml.add_representer(_LiteralString, _literal_string_representer)


def _normalize_prose(value: str | None) -> _LiteralString | str:
    """Normalize long-form/prose text for YAML emission.

    - Treats ``None`` as empty string (no structural change downstream).
    - Ensures any embedded ``\\n`` sequences become real newlines.
    - Returns ``_LiteralString`` for multiline prose so PyYAML emits ``|`` blocks.
    - Safe to call repeatedly (idempotent on already-normalized input).
    - Only intended for clearly prose-oriented fields at call sites.

    Args:
        value: Input text to normalize, may be None.

    Returns:
        _LiteralString if multiline, plain str if single-line, empty str if None.
    """
    if value is None:
        return ""

    # Coerce to str (in case of unexpected types) without changing semantics.
    text = str(value)

    # Convert literal backslash-n sequences to actual newlines.
    # This is deliberately narrow: only "\\n" becomes "\n".
    if "\\n" in text:
        text = text.replace("\\n", "\n")

    # Idempotence: if there are already real newlines, don't mangle them.
    # Decide whether to wrap as literal block.
    if "\n" in text:
        return _LiteralString(text)

    # Single-line prose: keep as plain string (no forced literal block).
    return text


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
        project_dir=state.get("project_dir", ""),
    )

    project_dir = Path(state.get("project_dir", ""))

    # Initialize content manager for reading externalized content
    content_manager = ContentManager(state.get("project_dir", ""))

    # Get character sheets, global outline, and act outlines (from external files)
    character_sheets = get_character_sheets(state, content_manager)
    global_outline = get_global_outline(state, content_manager)
    act_outlines = get_act_outlines(state, content_manager)

    try:
        # Create directory structure
        _create_directory_structure(project_dir)

        # Write character files
        if character_sheets:
            _write_character_files(project_dir, character_sheets)

        # Write outline files
        if global_outline or act_outlines:
            _write_outline_files(project_dir, global_outline, act_outlines, state)

        # Write world items file (always, even if empty)
        world_items = state.get("world_items", [])
        _write_world_items_file(project_dir, world_items, state.get("setting", ""))

        # Write saga.yaml with top-level metadata and path references
        _write_saga_yaml(project_dir, state)

        # Write world rules and history stubs/files
        _write_world_rules_file(project_dir, state)
        _write_world_history_file(project_dir, state)

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
    """Create the SAGA 2.0 directory structure.

    Args:
        project_dir: Root project directory path.
    """
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
    """Parse character sheet text into structured fields.

    Looks for common section headers like:
    - Physical Description
    - Personality
    - Background
    - Motivations
    - Skills/Abilities
    - Relationships
    - Character Arc

    Args:
        text: Raw character sheet text to parse.

    Returns:
        Dictionary with parsed fields, or {"description": text} if parsing fails.
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
            content = re.sub(r"\s+", " ", content)
            parsed[field] = content

    # If we couldn't parse anything, fall back to raw description
    if not parsed:
        parsed["description"] = text

    return parsed


def _write_character_files(project_dir: Path, character_sheets: dict) -> None:
    """Write individual YAML files for each character.

    Format:
        characters/{character_name}.yaml

    Parses the character sheet text into structured fields for readability.

    Args:
        project_dir: Root project directory path.
        character_sheets: Dictionary mapping character names to sheet data.
    """
    characters_dir = project_dir / "characters"

    for name, sheet in character_sheets.items():
        # Sanitize filename
        safe_name = name.lower().replace(" ", "_").replace("'", "")
        file_path = characters_dir / f"{safe_name}.yaml"

        # Parse character sheet into structured fields
        description_text = sheet.get("description", "")
        parsed_data = _parse_character_sheet_text(description_text)

        # Normalize prose fields parsed from the description while keeping structure.
        normalized_parsed = {}
        for key, val in parsed_data.items():
            # Only apply normalization to textual fields.
            if isinstance(val, str):
                normalized_parsed[key] = _normalize_prose(val)
            else:
                normalized_parsed[key] = val

        character_data = {
            "name": name,
            "role": "protagonist" if sheet.get("is_protagonist") else "character",
            **normalized_parsed,  # Merge parsed fields
            "generated_at": datetime.now().isoformat(),
            "source": "initialization",
        }

        write_yaml_file(file_path, character_data)

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
    """Write outline structure and beats to YAML files.

    Creates:
    - outline/structure.yaml: Act/scene structure
    - outline/beats.yaml: Key plot beats

    Args:
        project_dir: Root project directory path.
        global_outline: Global story outline data (may be None).
        act_outlines: Dictionary mapping act numbers to act outline data.
        state: Current narrative state with metadata.
    """
    outline_dir = project_dir / "outline"

    # Write structure.yaml
    structure_data = {
        "title": state.get("title", ""),
        "genre": state.get("genre", ""),
        "theme": state.get("theme", ""),
        "setting": state.get("setting", ""),
        "total_chapters": state.get("total_chapters", 20),
        "target_word_count": state.get("target_word_count", 80000),
        "generated_at": datetime.now().isoformat(),
    }

    if global_outline:
        raw = global_outline.get("raw_text", "") + "..."
        structure_data["global_outline"] = {
            "act_count": global_outline.get("act_count", 3),
            "structure_type": global_outline.get("structure_type", "3-act"),
            # Summary is prose; normalize so any embedded "\n" become readable.
            "summary": _normalize_prose(raw),
        }

    if act_outlines:
        structure_data["acts"] = {}
        for act_num, act_data in sorted(act_outlines.items()):
            raw = act_data.get("raw_text", "") + "..."
            structure_data["acts"][f"act_{act_num}"] = {
                "act_number": act_num,
                "role": act_data.get("act_role", ""),
                "chapters": act_data.get("chapters_in_act", 0),
                # Summary is prose; normalize for readable YAML.
                "summary": _normalize_prose(raw),
            }

    structure_path = outline_dir / "structure.yaml"
    write_yaml_file(structure_path, structure_data)

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
        # Global outline is long-form prose; normalize into literal block.
        beats_data["global_outline"] = _normalize_prose(raw_text)

    if act_outlines:
        beats_data["act_outlines"] = {}
        for act_num, act_data in sorted(act_outlines.items()):
            raw_text = act_data.get("raw_text", "")
            beats_data["act_outlines"][f"act_{act_num}"] = {
                "act_number": act_num,
                "role": act_data.get("act_role", ""),
                # Outline is long-form prose; normalize into literal block.
                "outline": _normalize_prose(raw_text),
            }

    beats_path = outline_dir / "beats.yaml"
    write_yaml_file(beats_path, beats_data)

    logger.debug(
        "_write_outline_files: wrote beats file",
        path=str(beats_path),
    )


def _write_world_items_file(project_dir: Path, world_items: list, setting: str) -> None:
    """Write world items to YAML file.

    Creates:
    - world/items.yaml: Locations, objects, concepts

    Args:
        project_dir: Root project directory path.
        world_items: List of WorldItem objects to write.
        setting: Story setting description.
    """
    world_dir = project_dir / "world"

    world_data = {
        "setting": setting,
        "generated_at": datetime.now().isoformat(),
        "source": "initialization",
        "items": [],
    }

    for item in world_items:
        description = getattr(item, "description", "")
        world_data["items"].append(
            {
                "id": item.id,
                "name": item.name,
                "category": item.category,
                # Description is prose; normalize into multiline-safe form.
                "description": _normalize_prose(description),
            }
        )

    items_path = world_dir / "items.yaml"
    write_yaml_file(items_path, world_data)

    logger.debug(
        "_write_world_items_file: wrote world items file",
        path=str(items_path),
        count=len(world_items),
    )


def _write_saga_yaml(project_dir: Path, state: NarrativeState) -> None:
    """Write saga.yaml at the project root with basic metadata and directory paths.

    This file acts as a lightweight entrypoint/manifest for the SAGA workspace.
    It must be robust to partial state: only emit fields that are present.

    Args:
        project_dir: Root project directory path.
        state: Current narrative state with metadata.
    """
    saga_data: dict = {}

    # Conservative metadata extraction - tolerate missing keys.
    # Using `in state` guards against callers providing partial dict-like state.
    for key in (
        "title",
        "genre",
        "theme",
        "setting",
        "total_chapters",
        "target_word_count",
        "project_id",
    ):
        if key in state:
            saga_data[key] = state.get(key)

    # Paths section: simple relative directory references.
    saga_data["paths"] = {
        "outline": "outline/",
        "characters": "characters/",
        "world": "world/",
        "summaries": "summaries/",
        "exports": "exports/",
    }

    saga_path = project_dir / "saga.yaml"
    write_yaml_file(saga_path, saga_data)

    logger.debug(
        "_write_saga_yaml: wrote saga manifest",
        path=str(saga_path),
    )


def _write_world_rules_file(project_dir: Path, state: NarrativeState) -> None:
    """Write world/rules.yaml capturing world rules/constraints if available.

    Structure:
        rules:
          - name: ...
            description: ...
        note: "..."

    Falls back to an empty stub that guides the user when no data is present.

    Args:
        project_dir: Root project directory path.
        state: Current narrative state with world rules data.
    """
    world_dir = project_dir / "world"
    rules_path = world_dir / "rules.yaml"

    rules_data: dict = {}

    # Prefer explicit structured rules if such a field is introduced later.
    # Support both `world_rules` and existing `current_world_rules` list[str].
    raw_rules = state.get("world_rules") or state.get("current_world_rules") or []

    normalized_rules: list = []
    if isinstance(raw_rules, list):
        for idx, rule in enumerate(raw_rules, start=1):
            if isinstance(rule, str):
                normalized_rules.append(
                    {
                        "name": f"Rule {idx}",
                        "description": _normalize_prose(rule),
                    }
                )
            elif isinstance(rule, dict):
                # Keep conservative: only map obvious fields.
                name = rule.get("name") or rule.get("title") or f"Rule {idx}"
                desc = rule.get("description") or rule.get("text") or ""
                normalized_rules.append(
                    {
                        "name": str(name),
                        "description": _normalize_prose(desc),
                    }
                )

    if normalized_rules:
        rules_data["rules"] = normalized_rules
    else:
        rules_data["rules"] = []
        rules_data["note"] = "Add world rules, constraints, and systems here."

    write_yaml_file(rules_path, rules_data)

    logger.debug(
        "_write_world_rules_file: wrote world rules file",
        path=str(rules_path),
        count=len(rules_data.get("rules", [])),
    )


def _write_world_history_file(project_dir: Path, state: NarrativeState) -> None:
    """Write world/history.yaml capturing historical events/timeline if available.

    Structure:
        events:
          - id: ...
            description: ...
            era: ...
        note: "..."

    Falls back to an empty stub when no history data is present.

    Args:
        project_dir: Root project directory path.
        state: Current narrative state with historical events data.
    """
    world_dir = project_dir / "world"
    history_path = world_dir / "history.yaml"

    history_data: dict = {}

    # Accept several potential history/timeline fields without requiring any.
    raw_events = state.get("world_history") or state.get("history_events") or state.get("timeline") or []

    events: list = []
    if isinstance(raw_events, list):
        for idx, ev in enumerate(raw_events, start=1):
            if isinstance(ev, str):
                events.append(
                    {
                        "id": f"event_{idx}",
                        "description": _normalize_prose(ev),
                    }
                )
            elif isinstance(ev, dict):
                # Only surface obvious keys, keep schema minimal and extensible.
                event_entry: dict = {}
                event_entry["id"] = ev.get("id", f"event_{idx}")

                if "description" in ev:
                    event_entry["description"] = _normalize_prose(ev.get("description", ""))
                elif "text" in ev:
                    event_entry["description"] = _normalize_prose(ev.get("text", ""))
                elif "summary" in ev:
                    event_entry["description"] = _normalize_prose(ev.get("summary", ""))

                # Optional known-lore fields if present.
                for key in ("era", "age", "year", "location", "faction"):
                    if key in ev:
                        event_entry[key] = ev[key]

                events.append(event_entry)

    if events:
        history_data["events"] = events
    else:
        history_data["events"] = []
        history_data["note"] = "Add key historical events, eras, and background lore here."

    write_yaml_file(history_path, history_data)

    logger.debug(
        "_write_world_history_file: wrote world history file",
        path=str(history_path),
        count=len(history_data.get("events", [])),
    )


def _write_summaries_readme(project_dir: Path) -> None:
    """Write README file in summaries/ directory.

    The summaries directory is populated during chapter generation,
    not initialization. This creates a placeholder README to explain that.

    Args:
        project_dir: Root project directory path.
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

    from utils.file_io import write_text_file

    write_text_file(readme_path, readme_content)

    logger.debug(
        "_write_summaries_readme: wrote summaries README",
        path=str(readme_path),
    )


__all__ = ["persist_initialization_files"]
