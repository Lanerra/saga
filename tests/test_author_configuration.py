"""Author policy at real bootstrap, state, planning, and drafting boundaries."""
import inspect
import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import config
from core.langgraph.content_manager import ContentManager, get_scene_drafts
from core.langgraph.nodes.scene_generation_node import draft_scene
from core.langgraph.nodes.scene_planning_node import plan_scenes
from core.langgraph.state import NarrativeState, create_initial_state
from core.project_bootstrapper import ProjectBootstrapper
from core.project_config import NarrativeProjectConfig
from core.project_manager import ProjectManager
from core.service_context import get_services
from orchestration.langgraph_orchestrator import LangGraphOrchestrator
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager

STYLE = "First-person present, spare concrete sentences; no access to other minds."
PROJECT = {
    "title": "The Spare Key", "genre": "Mystery", "theme": "Trust",
    "setting": "An island archive", "protagonist_name": "Mara",
    "narrative_style": STYLE, "total_chapters": 3,
}
SCENE = {
    "title": "The archive", "pov_character": "Mara", "setting": "Archive",
    "characters": ["Mara"], "plot_point": "Read ledger", "conflict": "Locked door",
    "outcome": "Mara reads the ledger", "beats": ["Mara opens the archive"],
}


class RecordingProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    async def completion(self, model_name: str, prompt: str, **options: Any) -> tuple[str, dict[str, int]]:
        self.prompts.append(prompt)
        return self.response, {}


def author_state(directory: Path, target: int = 101) -> NarrativeState:
    state = create_initial_state(
        project_id=directory.name, project_dir=str(directory), title="The Spare Key",
        genre="Mystery", theme="Trust", setting="Archive", protagonist_name="Mara",
        narrative_style=STYLE, total_chapters=3, target_word_count=target,
    )
    manager = ContentManager(str(directory))
    state["chapter_outlines_ref"] = manager.save_json(
        {"1": {"scene_description": "Open archive", "key_beats": ["Mara opens the archive"]}},
        "chapter_outlines", "all", 1,
    )
    state["chapter_plan_ref"] = manager.save_json([SCENE] * 3, "chapter_plan", "chapter_1", 1)
    return state


def record_provider(monkeypatch: pytest.MonkeyPatch, response: str) -> RecordingProvider:
    provider = RecordingProvider(response)
    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider.completion)
    return provider


async def test_bootstrap_preserves_requested_style(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = record_provider(monkeypatch, json.dumps({**PROJECT, "target_word_count": 101}))
    result = await ProjectBootstrapper(get_services().language_model).generate_metadata(
        "A first-person present mystery in spare concrete sentences, 101 words in 3 chapters."
    )
    assert result.model_dump() == {
        **PROJECT, "target_word_count": 101, "created_from": "bootstrap",
        "original_prompt": "A first-person present mystery in spare concrete sentences, 101 words in 3 chapters.",
    }
    assert "Always use" not in provider.prompts[0]
    assert "Honor the author's requested narrative style" in provider.prompts[0]


async def test_project_target_roundtrips_to_real_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ProjectManager, "projects_root", tmp_path)
    project = NarrativeProjectConfig.model_validate({**PROJECT, "target_word_count": 101})
    directory = ProjectManager.save_config(project, review=False)
    original = (directory / "config.json").read_bytes()
    loaded = ProjectManager.load_config(directory)
    database = FakeNeo4jManager()
    monkeypatch.setattr(get_services().database, "execute_read_query", database.execute_read_query)
    state = await LangGraphOrchestrator(project_dir=directory)._load_or_create_state(project_id=directory.name, narrative_config=loaded)
    assert (state["narrative_style"], state["target_word_count"], state["total_chapters"]) == (STYLE, 101, 3)
    assert (directory / "config.json").read_bytes() == original


def test_legacy_project_uses_active_word_default() -> None:
    project = NarrativeProjectConfig.model_validate(PROJECT)
    assert project.target_word_count == config.TARGET_WORD_COUNT


@pytest.mark.parametrize("target", [0, -1, True, 1.5, "101", 2])
def test_invalid_project_word_targets_reject(target: Any) -> None:
    with pytest.raises(ValidationError):
        NarrativeProjectConfig.model_validate({**PROJECT, "target_word_count": target})


async def test_drafting_keeps_project_style(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = record_provider(monkeypatch, "I open the archive.")
    state = author_state(tmp_path)
    original = dict(state)
    result = await draft_scene(state)
    assert result.get("has_fatal_error", False) is False
    style_match = re.search(r"Narrative Style & Voice:\n(.*?)\n\n", provider.prompts[0], re.S)
    assert style_match is not None
    assert style_match.group(1) == STYLE
    assert get_scene_drafts(result, ContentManager(str(tmp_path))) == ["I open the archive."]
    assert state == original
    assert Path(inspect.getfile(draft_scene)).resolve() == Path(__file__).resolve().parents[1] / "core/langgraph/nodes/scene_generation_node.py"


async def test_draft_targets_preserve_both_remainders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = record_provider(monkeypatch, "I open the archive.")
    state = author_state(tmp_path)
    targets: list[list[int]] = []
    for chapter in range(1, 4):
        state["current_chapter"] = chapter
        chapter_targets = []
        for scene_index in range(3):
            state["current_scene_index"] = scene_index
            result = await draft_scene(state)
            assert result.get("has_fatal_error", False) is False
            length_match = re.search(r"Length: ~(\d+) words", provider.prompts[-1])
            assert length_match is not None
            chapter_targets.append(int(length_match.group(1)))
        targets.append(chapter_targets)
    assert targets == [[12, 11, 11], [12, 11, 11], [11, 11, 11]]
    assert sum(map(sum, targets)) == 101


async def test_planning_receives_style_and_same_chapter_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = record_provider(monkeypatch, json.dumps([SCENE] * 3))
    database = FakeNeo4jManager()
    database.configure_response(r"RETURN c.name AS name", [{"name": "Mara"}])
    monkeypatch.setattr(get_services().database, "execute_read_query", database.execute_read_query)
    result = await plan_scenes(author_state(tmp_path))
    assert result.get("has_fatal_error", False) is False
    assert "Narrative Style & Voice:\n" + STYLE + "\n" in provider.prompts[0]
    assert "Chapter length target: ~34 words in total" in provider.prompts[0]
    assert result["chapter_plan_scene_count"] == 3


async def test_impossible_scene_targets_fail_before_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = record_provider(monkeypatch, "I open the archive.")
    state = author_state(tmp_path, target=3)
    result = await draft_scene(state)
    assert result.get("has_fatal_error", False) is True
    assert provider.prompts == []
    assert result["error_node"] == "draft_scene"


async def test_author_omniscient_perspective_is_not_overridden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = record_provider(monkeypatch, "Mara and Iven both feared the tide.")
    state = author_state(tmp_path)
    state["narrative_style"] = "Third-person omniscient; access to all characters' thoughts."
    result = await draft_scene(state)
    assert result.get("has_fatal_error", False) is False
    assert "- Stay in the POV character's interior and sensory experience." not in provider.prompts[0]
    assert state["narrative_style"] in provider.prompts[0]


@pytest.mark.run_settings(TARGET_SCENES_MIN=0)
async def test_invalid_requested_scene_count_stops_planning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = record_provider(monkeypatch, json.dumps([SCENE]))
    result = await plan_scenes(author_state(tmp_path))
    assert result["has_fatal_error"] is True
    assert provider.prompts == []


@pytest.mark.parametrize("node", [plan_scenes, draft_scene])
@pytest.mark.parametrize("target", [0, -1, True, 1.5])
async def test_invalid_state_targets_fail_closed(node: Any, target: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = record_provider(monkeypatch, json.dumps([SCENE]))
    state = author_state(tmp_path)
    state["target_word_count"] = target
    result = await node(state)
    assert result["has_fatal_error"] is True
    assert provider.prompts == []


async def test_planner_rejects_more_scenes_than_words(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = record_provider(monkeypatch, json.dumps([SCENE] * 2))
    state = author_state(tmp_path, target=3)
    result = await plan_scenes(state)
    assert result["has_fatal_error"] is True
    assert result["chapter_plan_ref"] is None
    assert len(provider.prompts) == 1
    assert provider.prompts[0].startswith("You are breaking a chapter outline into 1 scenes.")

