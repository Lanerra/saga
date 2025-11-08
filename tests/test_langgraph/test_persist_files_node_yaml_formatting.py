from pathlib import Path

import yaml

from core.langgraph.initialization.persist_files_node import (
    persist_initialization_files,
)
from core.langgraph.state import NarrativeState


def _make_state(tmp_path: Path) -> NarrativeState:
    """Construct a minimal NarrativeState suitable for persist_initialization_files.

    Focused on exercising YAML formatting / normalization behavior only.
    """
    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    # Prose with embedded literal "\n" sequences and real newlines
    character_description = (
        "Hero of the story with a complex past.\\n"
        "Second line of bio that should appear on its own line.\n"
        "Third line already separated."
    )

    global_outline_text = (
        "This is the high-level arc of the story.\\n"
        "It spans multiple acts and should be readable.\n"
        "Final line."
    )

    act1_outline_text = (
        "Act I sets up the world and characters.\\n"
        "Multiple beats are described here.\n"
        "Closes on an inciting incident."
    )

    world_item_description = (
        "An ancient city hidden beneath the desert sands.\\n"
        "Legends speak of its cursed guardians.\n"
        "Explorers rarely return."
    )

    # Minimal state; only fields needed by persist_initialization_files
    state: NarrativeState = {
        "project_dir": str(project_dir),
        "title": "Test Novel",
        "genre": "Test Genre",
        "theme": "Courage vs. Fear",
        "setting": "Far future desert world",
        "total_chapters": 3,
        "target_word_count": 90000,
        "character_sheets": {
            "Test Protagonist": {
                "description": character_description,
                "is_protagonist": True,
            }
        },
        "global_outline": {
            "act_count": 3,
            "structure_type": "3-act",
            "raw_text": global_outline_text,
        },
        "act_outlines": {
            1: {
                "act_role": "setup",
                "chapters_in_act": 5,
                "raw_text": act1_outline_text,
            }
        },
        "world_items": [
            type(
                "WorldItemStub",
                (),
                {
                    "id": "world-1",
                    "name": "Ancient City",
                    "category": "Location",
                    "description": world_item_description,
                },
            )()
        ],
    }
    return state


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_persist_initialization_files_yaml_prose_formatting(tmp_path):
    state = _make_state(tmp_path)

    # Execute node
    new_state = (
        yaml.safe_load(
            yaml.safe_dump(
                {
                    **(awaitable := persist_initialization_files(state)),  # type: ignore[func-returns-value]
                }
            )
        )
        if False
        else None
    )  # pragma: no cover - placeholder to keep static analysers calm

    # Actually run the async function properly using a minimal event loop
    # without introducing extra runtime dependencies.
    import asyncio

    result_state: NarrativeState = asyncio.get_event_loop().run_until_complete(
        persist_initialization_files(state)
    )

    # Sanity: state updated and no error
    assert result_state["last_error"] is None
    assert result_state["initialization_step"] == "files_persisted"

    project_dir = Path(state["project_dir"])

    # 1) characters/*.yaml
    character_files = list((project_dir / "characters").glob("*.yaml"))
    assert character_files, "Character YAML file was not created"

    character_yaml_text = _read_file(character_files[0])
    character_data = yaml.safe_load(character_yaml_text)

    # Non-prose structural fields remain intact
    assert character_data["name"] == "Test Protagonist"
    assert character_data["role"] == "protagonist"
    assert character_data["source"] == "initialization"

    # Prose fields should contain real newlines (not literal "\n")
    bio_fields = [
        v
        for k, v in character_data.items()
        if k
        in {
            "description",
            "background",
            "physical_description",
            "personality",
            "character_arc",
        }
    ]
    if bio_fields:
        for text in bio_fields:
            if isinstance(text, str):
                # At least one newline should be present after normalization
                assert "\n" in text
                # No stray literal "\n" sequences
                assert "\\n" not in text

    # Also ensure serialized YAML for characters does not contain "\n" escapes for prose
    assert "\\n" not in character_yaml_text

    # 2) outline/structure.yaml
    structure_path = project_dir / "outline" / "structure.yaml"
    assert structure_path.is_file(), "outline/structure.yaml not created"

    structure_yaml_text = _read_file(structure_path)
    structure_data = yaml.safe_load(structure_yaml_text)

    # Structural fields
    assert structure_data["title"] == "Test Novel"
    assert structure_data["genre"] == "Test Genre"
    assert structure_data["total_chapters"] == 3
    assert structure_data["target_word_count"] == 90000

    # Prose summaries should preserve newlines and avoid "\n" escapes
    global_summary = structure_data["global_outline"]["summary"]
    assert isinstance(global_summary, str)
    if "\n" in global_summary:
        assert "\\n" not in global_summary

    act_summary = structure_data["acts"]["act_1"]["summary"]
    assert isinstance(act_summary, str)
    if "\n" in act_summary:
        assert "\\n" not in act_summary

    # YAML text itself for structure should not contain "\n" escapes for these summaries
    assert "\\n" not in structure_yaml_text

    # 3) outline/beats.yaml
    beats_path = project_dir / "outline" / "beats.yaml"
    assert beats_path.is_file(), "outline/beats.yaml not created"

    beats_yaml_text = _read_file(beats_path)
    beats_data = yaml.safe_load(beats_yaml_text)

    global_outline_prose = beats_data["global_outline"]
    assert isinstance(global_outline_prose, str)
    # Expect multiline content; no literal "\n"
    assert "\n" in global_outline_prose
    assert "\\n" not in global_outline_prose

    act_outline_prose = beats_data["act_outlines"]["act_1"]["outline"]
    assert isinstance(act_outline_prose, str)
    assert "\n" in act_outline_prose
    assert "\\n" not in act_outline_prose

    # Serialized beats YAML should use real newlines (no "\n" escapes) for prose sections
    assert "\\n" not in beats_yaml_text

    # 4) world/items.yaml
    items_path = project_dir / "world" / "items.yaml"
    assert items_path.is_file(), "world/items.yaml not created"

    items_yaml_text = _read_file(items_path)
    items_data = yaml.safe_load(items_yaml_text)

    assert items_data["setting"] == "Far future desert world"
    assert items_data["source"] == "initialization"

    item = items_data["items"][0]
    assert item["id"] == "world-1"
    assert item["name"] == "Ancient City"
    assert item["category"] == "Location"

    description_text = item["description"]
    assert isinstance(description_text, str)
    # Description should contain real newlines and no literal "\n"
    assert "\n" in description_text
    assert "\\n" not in description_text

    # Serialized YAML for world items should not contain "\n" escapes in description
    assert "\\n" not in items_yaml_text
