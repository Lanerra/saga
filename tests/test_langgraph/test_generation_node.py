# tests/test_langgraph/test_generation_node.py
from core.langgraph.subgraphs.generation import generate_chapter, should_continue_scenes


def test_should_continue_scenes_routes_when_plan_present() -> None:
    state = {"chapter_plan": ["scene"], "current_scene_index": 0}
    assert should_continue_scenes(state) == "continue"


def test_should_continue_scenes_stops_after_last_scene() -> None:
    state = {"chapter_plan": ["scene"], "current_scene_index": 1}
    assert should_continue_scenes(state) == "end"


def test_should_continue_scenes_stops_without_plan() -> None:
    state = {"chapter_plan": [], "current_scene_index": 0}
    assert should_continue_scenes(state) == "end"


def test_generate_chapter_returns_compiled_graph() -> None:
    graph = generate_chapter()
    assert hasattr(graph, "ainvoke")
    assert hasattr(graph, "astream")
