import asyncio
from pathlib import Path

import yaml

from core.langgraph.initialization.persist_files_node import (
    persist_initialization_files,
)
from core.langgraph.state import NarrativeState


def _minimal_state(tmp_path: Path) -> NarrativeState:
    """Construct a minimal NarrativeState for testing initialization artifacts.

    Only include fields required by persist_initialization_files and saga.yaml.
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
        # Required initialization collections (can be empty for this test)
        "character_sheets": {},
        "global_outline": None,
        "act_outlines": {},
        "world_items": [],
        # Fields referenced by NarrativeState but not needed here are intentionally omitted
    }
    return state


def _run_persist(state: NarrativeState) -> NarrativeState:
    # Run async node without introducing new dependencies.
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(persist_initialization_files(state))


def test_saga_yaml_created_with_paths_and_metadata(tmp_path: Path) -> None:
    state = _minimal_state(tmp_path)

    result_state = _run_persist(state)

    # Node should report success
    assert result_state["last_error"] is None
    assert result_state["initialization_step"] == "files_persisted"

    project_dir = Path(state["project_dir"])

    saga_path = project_dir / "saga.yaml"
    assert saga_path.is_file(), "saga.yaml was not created at project root"

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


def test_world_rules_and_history_stubs_when_missing(tmp_path: Path) -> None:
    state = _minimal_state(tmp_path)

    _run_persist(state)

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


def test_world_rules_and_history_populated_when_present(tmp_path: Path) -> None:
    """Optional coverage: ensure provided rules/history surface into YAML."""
    state = _minimal_state(tmp_path)

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

    _run_persist(state)

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
