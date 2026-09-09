# tests/test_langgraph/test_generation_node.py
import pytest

from core.langgraph.state import NarrativeState
from core.langgraph.subgraphs.generation import create_generation_subgraph, should_continue_scenes


class LegacySceneRoutingState(NarrativeState, total=False):
    chapter_plan: list[str]


def test_should_continue_scenes_routes_when_scene_count_positive() -> None:
    state: NarrativeState = {"chapter_plan_scene_count": 1, "current_scene_index": 0}
    assert should_continue_scenes(state) == "continue"


def test_should_continue_scenes_stops_after_last_scene() -> None:
    state: NarrativeState = {"chapter_plan_scene_count": 1, "current_scene_index": 1}
    assert should_continue_scenes(state) == "end"


def test_should_continue_scenes_stops_when_scene_count_zero() -> None:
    state: NarrativeState = {"chapter_plan_scene_count": 0, "current_scene_index": 0}
    assert should_continue_scenes(state) == "end"


def test_should_continue_scenes_raises_type_error_when_scene_count_not_int(monkeypatch: pytest.MonkeyPatch) -> None:
    state: NarrativeState = {"current_scene_index": 0}
    monkeypatch.setitem(state, "chapter_plan_scene_count", "1")
    assert state == {"chapter_plan_scene_count": "1", "current_scene_index": 0}
    with pytest.raises(TypeError, match="^chapter_plan_scene_count must be an int$"):
        should_continue_scenes(state)


def test_should_continue_scenes_does_not_infer_scene_count_from_chapter_plan() -> None:
    state: LegacySceneRoutingState = {"chapter_plan": ["scene"], "current_scene_index": 0}
    assert should_continue_scenes(state) == "end"


def test_create_generation_subgraph_returns_compiled_graph() -> None:
    graph = create_generation_subgraph()
    assert hasattr(graph, "ainvoke")
    assert hasattr(graph, "astream")
