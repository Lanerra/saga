"""Project metadata and fatal routing stay fail-closed."""

import json
from pathlib import Path

import pytest

from core.langgraph.state import NarrativeState
from core.langgraph.workflow import should_continue_init, should_continue_to_next_chapter
from core.project_config import NarrativeProjectConfig
from core.project_manager import ProjectManager
from orchestration.langgraph_orchestrator import LangGraphOrchestrator


@pytest.mark.parametrize("candidate", [False, True])
def test_project_config_load_rejects_symbolic_links(tmp_path: Path, candidate: bool) -> None:
    directory = tmp_path / "project"
    directory.mkdir()
    outside = tmp_path / "outside.json"
    configuration = NarrativeProjectConfig(title="Synthetic", genre="Mystery", theme="Trust", setting="Archive", protagonist_name="Traveler", narrative_style="Third person", total_chapters=1, target_word_count=2000)
    outside.write_text(configuration.model_dump_json())
    name = "config.candidate.json" if candidate else "config.json"
    (directory / name).symlink_to(outside)
    loader = ProjectManager.load_candidate_config if candidate else ProjectManager.load_config
    with pytest.raises((ValueError, OSError), match="regular file|symbolic link"):
        loader(directory)


@pytest.mark.parametrize("payload", ["project_id: [invalid", "[]", "project_id: 123", "project_id: ' padded '"])
def test_existing_invalid_project_identity_cannot_fall_back(tmp_path: Path, payload: str) -> None:
    (tmp_path / "saga.yaml").write_text(payload)
    with pytest.raises(ValueError):
        LangGraphOrchestrator(project_dir=tmp_path)._get_requested_project_id()


def test_workflow_project_identity_rejects_linked_metadata(tmp_path: Path) -> None:
    directory = tmp_path / "project"
    directory.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text(json.dumps({"project_id": "foreign-id"}))
    (directory / "saga.yaml").symlink_to(outside)
    with pytest.raises((ValueError, OSError), match="regular file|symbolic link"):
        LangGraphOrchestrator(project_dir=directory)._get_requested_project_id()


def test_initialization_stops_on_bare_fatal_flag() -> None:
    assert should_continue_init({"has_fatal_error": True}) == "error"


def test_chapter_advance_stops_on_compensation_barrier() -> None:
    state: NarrativeState = {
        "current_chapter": 1, "total_chapters": 2,
        "revision_rollback_failure": {"chapter_number": 1, "iteration_count": 0, "error": "Synthetic unresolved rollback", "previous_error": None, "previous_error_node": None},
    }
    assert should_continue_to_next_chapter(state) == "error"
