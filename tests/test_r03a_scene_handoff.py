"""Synthetic selected-outline handoff tests; no narrative-quality claim."""

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

import config
from core.http_client_service import CompletionHTTPClient
from core.langgraph.content_manager import ContentManager, get_chapter_plan
from core.langgraph.nodes.scene_generation_node import draft_scene
from core.langgraph.nodes.scene_planning_node import _ScenePlanEntry, plan_scenes
from core.langgraph.state import NarrativeState
from core.service_context import get_services
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager
from tests.test_staged_initialization import example_state


@pytest.mark.parametrize("case", ["missing_beats", "wrong_type", "wrong_count"])
async def test_scene_plan_must_preserve_selected_outline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    # Preserve R05's three reproductions, including the independently invalid POV.
    state = example_state(tmp_path)
    state["current_chapter"] = 1
    state["target_word_count"] = 101
    scene: dict[str, Any] = dict(title="Harbor", pov_character="", setting="Harbor", characters=[], plot_point="Choice", conflict="Duty", outcome="Stay", beats=["Ada chooses"])
    if case == "missing_beats":
        scene["beats"] = ["Unselected new beat"]
    elif case == "wrong_type":
        scene["setting"] = 123
    monkeypatch.setattr("config.TARGET_SCENES_MIN", 2)
    scenes = [scene] if case == "wrong_count" else [scene, {**scene, "beats": []}]

    async def transport(**arguments: Any) -> tuple[str, dict[str, Any]]:
        return json.dumps(scenes), {}

    monkeypatch.setattr(get_services().language_model, "async_call_llm", transport)
    result = await plan_scenes(state)
    assert result.get("has_fatal_error") is True
    assert result.get("chapter_plan_ref") is None


def scene(beats: list[str], characters: list[str] | None = None) -> dict[str, Any]:
    return dict(title="Harbor", pov_character="Ada", setting="Harbor", characters=characters or [], plot_point="Choice", conflict="Duty", outcome="Stay", beats=beats)


def selected_state(directory: Path, beats: Any, target: int = 101, chapters: int = 1, chapter: int = 1) -> NarrativeState:
    state = example_state(directory)
    state["current_chapter"] = chapter
    state["target_word_count"] = target
    state["total_chapters"] = chapters
    manager = ContentManager(str(directory))
    # Another chapter and a stale version must not replace the selected content.
    manager.save_json({str(chapter): {"scene_description": "Stale", "key_beats": ["STALE"]}}, "chapter_outlines", "all", 1)
    state["chapter_outlines_ref"] = manager.save_json(
        {str(chapter): {"scene_description": "Selected", "key_beats": beats}, str(chapters + 1): {"scene_description": "Other", "key_beats": ["OTHER"]}},
        "chapter_outlines", "all", 2,
    )
    return state


class Producer:
    def __init__(self, responses: list[str], directory: Path, database: FakeNeo4jManager) -> None:
        self.responses = responses
        self.prompts: list[str] = []
        self.directory = directory
        self.database = database

    async def completion(self, client: CompletionHTTPClient, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int, **keywords: Any) -> dict[str, Any]:
        assert self.database.executed_queries == []
        assert list((self.directory / ".saga/content/chapter_plan").glob("*.json")) == []
        self.prompts.append(messages[-1]["content"])
        response = self.responses[min(len(self.prompts) - 1, len(self.responses) - 1)]
        return {"choices": [{"message": {"content": response}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def install_producer(monkeypatch: pytest.MonkeyPatch, directory: Path, responses: list[str]) -> Producer:
    database = FakeNeo4jManager()
    database.configure_response(r"RETURN c.name AS name", [{"name": "Newcomer"}])
    monkeypatch.setattr(get_services().database, "execute_read_query", database.execute_read_query)
    monkeypatch.setattr(get_services().database, "execute_cypher_batch", database.execute_cypher_batch)
    monkeypatch.setattr(get_services().database, "execute_write_query", database.execute_write_query)
    producer = Producer(responses, directory, database)
    monkeypatch.setattr(CompletionHTTPClient, "get_completion", lambda client, *args, **kwargs: producer.completion(client, *args, **kwargs))
    monkeypatch.setattr(config, "TARGET_SCENES_MIN", 2)
    assert config.settings.SCENE_PLAN_MAX_ATTEMPTS == 3
    return producer


@pytest.mark.parametrize("groups", [
    [["A", "B", "A"]],
    [["A"], ["B"], ["A"]],
    [["A"], ["B"]],
    [["A", "A"], ["B"]],
    [["A", "B"], ["INVENTED"]],
    [["A", "B"], ["A", "A"]],
    [[" A", "B"], ["A"]],
    [["a", "B"], ["A"]],
    [["OTHER"], []],
    [["STALE"], []],
])
async def test_handoff_rejects_count_or_nonexact_partition_before_side_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, groups: list[list[str]]) -> None:
    state = selected_state(tmp_path, ["A", "B", "A"])
    original = dict(state)
    producer = install_producer(monkeypatch, tmp_path, [json.dumps([scene(beats, ["Newcomer"]) for beats in groups])])
    result = await plan_scenes(state)
    assert result.get("has_fatal_error") is True
    assert result["chapter_plan_ref"] is None
    assert result["chapter_plan_scene_count"] == 0
    assert result["error_node"] == "plan_scenes"
    assert "Scene plan contract violation" in (result["last_error"] or "")
    assert len(producer.prompts) == 3
    assert producer.prompts[1] == producer.prompts[2]
    assert len(producer.prompts[1]) - len(producer.prompts[0]) < 1200
    assert producer.database.executed_queries == []
    assert list((tmp_path / ".saga/content/chapter_plan").glob("*.json")) == []
    assert state == original


async def test_count_then_beats_correction_saves_only_exact_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    beats = ["A", "B", "A"]
    state = selected_state(tmp_path, beats)
    valid = [scene(["A"]), scene(["B", "A"])]
    responses = [[scene(beats, ["Newcomer"])], [scene(["A"], ["Newcomer"]), scene(["B"])], valid]
    producer = install_producer(monkeypatch, tmp_path, [json.dumps(value) for value in responses])
    result = await plan_scenes(state)
    assert not result.get("has_fatal_error", False)
    assert len(producer.prompts) == 3
    assert result["chapter_plan_scene_count"] == 2
    assert get_chapter_plan(result, ContentManager(str(tmp_path))) == valid
    assert len(list((tmp_path / ".saga/content/chapter_plan").glob("*.json"))) == 1
    assert producer.database.executed_queries == []
    assert "exactly 2 scenes" in producer.prompts[0]
    assert "including duplicates" in producer.prompts[0]
    assert "non-whitespace" in producer.prompts[0]
    assert "beats may be []" in producer.prompts[0]
    assert "Selected ordered key_beats (JSON): [\"A\",\"B\",\"A\"]" in producer.prompts[0]
    assert "Scene word targets in order (not output fields): [51,50]" in producer.prompts[0]
    assert "Your last response was invalid." in producer.prompts[1]
    assert producer.prompts[1] == producer.prompts[2]
    assert Path(inspect.getfile(plan_scenes)).resolve() == Path(__file__).resolve().parents[1] / "core/langgraph/nodes/scene_planning_node.py"
    schema = _ScenePlanEntry.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(valid[0])
    assert schema["properties"]["beats"]["type"] == "array"
    assert schema["properties"]["pov_character"]["pattern"] == r"\S"


@pytest.mark.parametrize(("beats", "groups"), [([], [[], []]), (["A"], [[], ["A"]]), (["A", "A"], [["A"], ["A"]]), ([" É\nchooses ", "É"], [[" É\nchooses "], ["É"]])])
async def test_exact_partitions_allow_empty_groups_and_preserve_duplicates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, beats: list[str], groups: list[list[str]]) -> None:
    state = selected_state(tmp_path, beats)
    valid = [scene(group) for group in groups]
    producer = install_producer(monkeypatch, tmp_path, [json.dumps(valid)])
    result = await plan_scenes(state)
    assert not result.get("has_fatal_error", False)
    assert len(producer.prompts) == 1
    assert get_chapter_plan(result, ContentManager(str(tmp_path))) == valid


@pytest.mark.parametrize("beats", [None, "A", [123], [" "], [True]])
async def test_unrepresentable_selected_beats_fail_before_producer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, beats: Any) -> None:
    state = selected_state(tmp_path, beats)
    producer = install_producer(monkeypatch, tmp_path, [json.dumps([scene([]), scene([])])])
    result = await plan_scenes(state)
    assert result.get("has_fatal_error") is True
    assert result["chapter_plan_ref"] is None
    assert producer.prompts == []
    assert producer.database.executed_queries == []


@pytest.mark.parametrize(("target", "chapter", "expected"), [(101, 1, [12, 11, 11]), (101, 3, [11, 11, 11]), (3, 2, [1])])
async def test_computed_count_and_word_allocations_reach_drafting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: int, chapter: int, expected: list[int]) -> None:
    state = selected_state(tmp_path, ["A"], target=target, chapters=3, chapter=chapter)
    valid = [scene(["A"])] + [scene([]) for _ in expected[1:]]
    producer = install_producer(monkeypatch, tmp_path, [json.dumps(valid)])
    monkeypatch.setattr(config, "TARGET_SCENES_MIN", 3)
    result = await plan_scenes(state)
    assert not result.get("has_fatal_error", False)
    assert result["chapter_plan_scene_count"] == len(expected)
    assert f"Scene word targets in order (not output fields): {json.dumps(expected, separators=(',', ':'))}" in producer.prompts[0]
    prompts: list[str] = []

    async def draft_transport(**arguments: Any) -> tuple[str, dict[str, Any]]:
        prompts.append(arguments["prompt"])
        return "Synthetic drafting boundary response.", {}

    monkeypatch.setattr(get_services().language_model, "async_call_llm", draft_transport)
    for index, word_target in enumerate(expected):
        update = await draft_scene({**state, **result, "current_scene_index": index})
        assert not update.get("has_fatal_error", False)
        assert f"- Length: ~{word_target} words." in prompts[-1]
