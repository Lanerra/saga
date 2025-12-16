# tests/test_langgraph/test_persist_files_node_initialization_artifacts.py
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization.persist_files_node import (
    persist_initialization_files,
)
from core.langgraph.state import NarrativeState


def _minimal_state(tmp_path: Path) -> NarrativeState:
    """Construct a minimal NarrativeState for testing initialization artifacts.

    This intentionally does NOT include externalized `*_ref` fields, so the
    persist node will be unable to write outline/character artifacts and the
    post-write validation will fail.
    """
    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    state: NarrativeState = {
        "project_dir": str(project_dir),
        "title": "Saga Init Test",
        "genre": "Science Fantasy",
        "theme": "Hope vs. Despair",
        "setting": "Ringworld frontier",
        "total_chapters": 10,
        "target_word_count": 120000,
        # Intentionally omit: character_sheets_ref, global_outline_ref, act_outlines_ref
        "world_items": [],
        # Fields referenced by NarrativeState but not needed here are intentionally omitted
    }
    return state


def _state_with_required_init_refs(tmp_path: Path) -> NarrativeState:
    """Construct a state that allows persist node to write all required artifacts.

    [`persist_initialization_files()`](core/langgraph/initialization/persist_files_node.py:104)
    reads character sheets / outlines from externalized refs, so tests that
    expect validation success must populate those refs.
    """
    state = _minimal_state(tmp_path)
    project_dir = Path(state["project_dir"])

    cm = ContentManager(str(project_dir))

    character_sheets = {
        "Hero": {
            "description": "**Background:** Raised on the ringworld frontier.\n\n**Motivations:** Protect home.",
            "is_protagonist": True,
        }
    }
    global_outline = {
        "raw_text": "A sweeping saga across the ringworld frontier.",
        "act_count": 3,
        "structure_type": "3-act",
    }
    act_outlines = {
        1: {
            "raw_text": "Act I: The call to adventure.",
            "act_role": "setup",
            "chapters_in_act": 3,
        }
    }

    state["character_sheets_ref"] = cm.save_json(character_sheets, "character_sheets", "all", version=1)
    state["global_outline_ref"] = cm.save_json(global_outline, "global_outline", "global", version=1)
    state["act_outlines_ref"] = cm.save_json(act_outlines, "act_outlines", "all", version=1)

    return state


@pytest.mark.asyncio
async def test_saga_yaml_created_with_paths_and_metadata(tmp_path: Path) -> None:
    state = _state_with_required_init_refs(tmp_path)

    # P0-2: file-write cache invalidation hook should be invoked after writing artifacts.
    # Patch the method on the real class (persist node constructs its own instance).
    with patch.object(ContentManager, "clear_cache", autospec=True) as mock_clear_cache:
        result_state = await persist_initialization_files(state)
        assert mock_clear_cache.called

    # Node should report success
    assert result_state["last_error"] is None
    assert result_state["initialization_step"] == "files_persisted"
    assert result_state.get("has_fatal_error", False) is False

    project_dir = Path(state["project_dir"])

    # Validation success implies required artifacts exist
    assert (project_dir / "outline" / "structure.yaml").is_file()
    assert (project_dir / "outline" / "beats.yaml").is_file()
    assert any((project_dir / "characters").glob("*.yaml"))
    assert (project_dir / "world" / "items.yaml").is_file()
    assert (project_dir / "world" / "rules.yaml").is_file()
    assert (project_dir / "world" / "history.yaml").is_file()
    assert (project_dir / "saga.yaml").is_file()

    saga_path = project_dir / "saga.yaml"
    data = yaml.safe_load(saga_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)

    # Metadata: use conservative checks; fields we provided should be present.
    assert data.get("title") == "Saga Init Test"
    assert data.get("genre") == "Science Fantasy"
    assert data.get("theme") == "Hope vs. Despair"
    assert data.get("setting") == "Ringworld frontier"
    assert data.get("total_chapters") == 10
    assert data.get("target_word_count") == 120000

    # Paths mapping must exist and contain expected relative directories.
    paths = data.get("paths")
    assert isinstance(paths, dict), "saga.yaml should contain a 'paths' mapping"

    assert paths.get("outline") == "outline/"
    assert paths.get("characters") == "characters/"
    assert paths.get("world") == "world/"
    assert paths.get("summaries") == "summaries/"
    assert paths.get("exports") == "exports/"


@pytest.mark.asyncio
async def test_world_rules_and_history_stubs_when_missing(tmp_path: Path) -> None:
    state = _state_with_required_init_refs(tmp_path)

    result = await persist_initialization_files(state)
    assert result["last_error"] is None

    project_dir = Path(state["project_dir"])

    # world/rules.yaml
    rules_path = project_dir / "world" / "rules.yaml"
    assert rules_path.is_file(), "world/rules.yaml was not created"

    rules_data = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    assert isinstance(rules_data, dict)

    # When no rules provided, should be an empty list plus a helpful note.
    assert "rules" in rules_data
    assert isinstance(rules_data["rules"], list)
    assert rules_data["rules"] == []
    assert isinstance(rules_data.get("note"), str)
    assert rules_data["note"], "Expected non-empty note in world/rules.yaml stub"

    # world/history.yaml
    history_path = project_dir / "world" / "history.yaml"
    assert history_path.is_file(), "world/history.yaml was not created"

    history_data = yaml.safe_load(history_path.read_text(encoding="utf-8"))
    assert isinstance(history_data, dict)

    assert "events" in history_data
    assert isinstance(history_data["events"], list)
    assert history_data["events"] == []
    assert isinstance(history_data.get("note"), str)
    assert history_data["note"], "Expected non-empty note in world/history.yaml stub"


@pytest.mark.asyncio
async def test_world_rules_and_history_populated_when_present(tmp_path: Path) -> None:
    """Optional coverage: ensure provided rules/history surface into YAML."""
    state = _state_with_required_init_refs(tmp_path)

    # Inject sample rules and history-like structures
    state["current_world_rules"] = [
        "Magic cannot resurrect the dead.",
        {"name": "No FTL", "description": "Faster-than-light travel does not exist."},
    ]
    state["world_history"] = [
        "The Shattering of the Ring, two centuries ago.",
        {
            "id": "founding",
            "description": "Founding of the frontier habitats.",
            "era": "Post-Shattering",
        },
    ]

    result = await persist_initialization_files(state)
    assert result["last_error"] is None

    project_dir = Path(state["project_dir"])

    # Validate rules mapping
    rules_path = project_dir / "world" / "rules.yaml"
    rules_data = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    rules = rules_data.get("rules")
    assert isinstance(rules, list)
    assert len(rules) == 2

    # First rule from string
    assert rules[0]["name"].startswith("Rule ")
    assert "Magic cannot resurrect the dead." in rules[0]["description"]

    # Second rule from mapping
    assert rules[1]["name"] == "No FTL"
    assert "Faster-than-light travel does not exist." in rules[1]["description"]

    # Validate history mapping
    history_path = project_dir / "world" / "history.yaml"
    history_data = yaml.safe_load(history_path.read_text(encoding="utf-8"))
    events = history_data.get("events")
    assert isinstance(events, list)
    assert len(events) == 2

    # First event from string
    assert events[0]["id"].startswith("event_")
    assert "Shattering of the Ring" in events[0]["description"]

    # Second event from mapping
    assert events[1]["id"] == "founding"
    assert "Founding of the frontier habitats." in events[1]["description"]
    assert events[1].get("era") == "Post-Shattering"


@pytest.mark.asyncio
async def test_persist_initialization_files_validation_failure_sets_fatal_error(tmp_path: Path) -> None:
    """Validation failure should halt the workflow via fatal-error fields."""
    state = _minimal_state(tmp_path)

    result = await persist_initialization_files(state)

    assert result["current_node"] == "persist_files"
    assert result["initialization_step"] == "validation_failed"

    assert result["last_error"] is not None
    assert "Initialization artifacts validation failed" in result["last_error"]

    # Fatal-error pattern (consistent with other nodes like finalize)
    assert result["has_fatal_error"] is True
    assert result["error_node"] == "persist_files"

    # Ensure missing artifacts are surfaced in the message (stable helper phrasing)
    assert "Missing outline/structure.yaml" in result["last_error"]
    assert "Missing outline/beats.yaml" in result["last_error"]
    assert "Missing any characters/*.yaml files" in result["last_error"]
