"""Offline workflow admission and retained-policy regressions."""

import json
from pathlib import Path
from typing import Any

import pytest

import config
from core.exceptions import CheckpointResumeConflictError
from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.validation_node import _is_plot_stagnant, validate_consistency
from core.langgraph.quality_policy import SCORE_FIELDS, configured_policy
from core.langgraph.state import NarrativeState
from core.langgraph.subgraphs.validation import evaluate_quality
from core.langgraph.workflow import create_checkpointer
from core.project_config import NarrativeProjectConfig, allocate_word_target
from core.service_context import get_services
from orchestration.langgraph_orchestrator import LangGraphOrchestrator
from prompts.prompt_renderer import render_prompt


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
async def test_checkpoint_rejects_linked_database_and_sidecars(tmp_path: Path, suffix: str) -> None:
    directory = tmp_path / "project/checkpoints"
    directory.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"Preserve this unrelated file exactly")
    path = directory / "saga.db"
    Path(str(path) + suffix).symlink_to(outside)
    with pytest.raises((ValueError, OSError), match="regular file|link"):
        async with create_checkpointer(str(path)):
            pytest.fail("Untrusted SQLite path admitted")
    assert outside.read_bytes() == b"Preserve this unrelated file exactly"


@pytest.mark.parametrize("foreign", [True, False])
def test_resume_requires_the_requested_project_directory(tmp_path: Path, foreign: bool) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state: NarrativeState = {"project_id": "same-id", "current_chapter": 1}
    if foreign:
        state["project_dir"] = str(tmp_path / "other-project")
    orchestrator = LangGraphOrchestrator(project_dir=project)
    with pytest.raises(CheckpointResumeConflictError, match="project_dir"):
        orchestrator._validate_resume_state_or_raise(checkpoint_state=state, requested_project_id="same-id")
    assert not (tmp_path / "other-project").exists()


async def test_evaluation_uses_checkpoint_threshold_not_current_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ContentManager(str(tmp_path))
    policy = {**configured_policy(), "minimum_score": 0.95}
    state: NarrativeState = {
        "project_id": "synthetic", "project_dir": str(tmp_path), "current_chapter": 1,
        "quality_policy": policy, "draft_ref": manager.save_text("Synthetic prose.", "draft", "chapter_1"),
    }

    async def response(**arguments: Any) -> tuple[str, dict[str, Any]]:
        return json.dumps({**dict.fromkeys(SCORE_FIELDS, 0.8), "feedback": "Synthetic evaluation."}), {}

    monkeypatch.setattr(get_services().language_model, "async_call_llm", response)
    monkeypatch.setitem(vars(config), "settings", config.settings.model_copy(update={"MIN_QUALITY_THRESHOLD": 0.5}))
    result = await evaluate_quality(state)
    assert result["quality_checks"]["evaluation"]["status"] == "completed"
    assert [issue.type for issue in result["contradictions"]] == ["quality_issue"]
    assert "0.95" in result["contradictions"][0].description


async def test_consistency_uses_checkpoint_disable_policy_without_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "must-not-create"
    state: NarrativeState = {
        "project_dir": str(project), "current_chapter": 1,
        "quality_policy": {**configured_policy(), "consistency_enabled": False},
    }
    monkeypatch.setitem(vars(config), "settings", config.settings.model_copy(update={"validation": config.settings.validation.model_copy(update={"ENABLE_VALIDATION": True})}))
    result = await validate_consistency(state)
    assert result["quality_checks"]["consistency"]["status"] == "skipped"
    assert result["quality_policy"] == state["quality_policy"]
    assert not project.exists()


def test_retained_major_issue_is_word_floor_not_filtered_relationship_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(vars(config), "settings", config.settings.model_copy(update={
        "PLOT_STAGNATION_MIN_WORD_COUNT": 1500, "PLOT_STAGNATION_MIN_ENTITIES": 1, "PLOT_STAGNATION_MIN_RELATIONSHIPS": 1,
    }))
    entities = {"characters": [None] * 6, "world_items": [None] * 15, "events": []}
    assert _is_plot_stagnant({"draft_word_count": 1431}, entities, [None] * 15) is True
    assert _is_plot_stagnant({"draft_word_count": 1500}, entities, [None] * 15) is False
    assert _is_plot_stagnant({"draft_word_count": 1500}, entities, [None] * 2) is False


@pytest.mark.parametrize("total_words,chapters,chapter", [(1000, 1, 1), (2001, 3, 3), (2000, 3, 1), (2000, 3, 3)])
def test_supported_short_chapter_target_is_not_impossible(total_words: int, chapters: int, chapter: int, monkeypatch: pytest.MonkeyPatch) -> None:
    configuration = NarrativeProjectConfig(title="Synthetic", genre="Mystery", theme="Trust", setting="Archive", protagonist_name="Traveler", narrative_style="Third person", total_chapters=chapters, target_word_count=total_words)
    target = allocate_word_target(configuration.target_word_count, configuration.total_chapters, chapter)
    scene_targets = [allocate_word_target(target, 2, scene) for scene in (1, 2)]
    assert sum(scene_targets) == target
    for number, scene_target in enumerate(scene_targets, start=1):
        prompt = render_prompt("narrative_agent/draft_scene.j2", {
            "chapter_number": chapter, "novel_title": "Synthetic", "narrative_style": "Third person", "total_scenes": 2,
            "scene": {"title": "Discovery", "scene_number": number, "pov_character": "Traveler", "setting": "Archive", "characters": ["Traveler"], "plot_point": "Discovery", "conflict": "Locked record", "outcome": "A revelation", "beats": ["Find record"]},
            "previous_scenes": [], "novel_genre": "Mystery", "novel_theme": "Trust", "hybrid_context": "Synthetic context", "revision_guidance": "", "target_word_count": scene_target,
        })
        assert f"Length: ~{scene_target} words." in prompt
    monkeypatch.setitem(vars(config), "settings", config.settings.model_copy(update={"PLOT_STAGNATION_MIN_WORD_COUNT": 1500}))
    state: NarrativeState = {"target_word_count": total_words, "total_chapters": chapters, "current_chapter": chapter, "draft_word_count": target}
    assert _is_plot_stagnant(state, {"events": ["synthetic revelation"]}, []) is False
    state["draft_word_count"] = target - 1
    assert _is_plot_stagnant(state, {"events": ["synthetic revelation"]}, []) is True


def test_normal_chapter_keeps_configured_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(vars(config), "settings", config.settings.model_copy(update={"PLOT_STAGNATION_MIN_WORD_COUNT": 1500}))
    state: NarrativeState = {"target_word_count": 4000, "total_chapters": 2, "current_chapter": 1, "draft_word_count": 1431}
    assert _is_plot_stagnant(state, {"events": ["synthetic revelation"]}, []) is True
