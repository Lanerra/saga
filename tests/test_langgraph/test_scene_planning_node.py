import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.langgraph.nodes.scene_planning_node import plan_scenes


@pytest.fixture
def base_state():
    return {
        "project_dir": "/tmp/test-project",
        "title": "Test Novel",
        "genre": "Fantasy",
        "theme": "Adventure",
        "large_model": "test-model",
        "current_chapter": 1,
    }


@pytest.fixture
def mock_content_manager():
    with patch("core.langgraph.nodes.scene_planning_node.ContentManager") as mock:
        instance = MagicMock()
        instance.get_latest_version.return_value = 0
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_get_chapter_outlines():
    with patch("core.langgraph.nodes.scene_planning_node.get_chapter_outlines") as mock:
        mock.return_value = {
            1: {
                "scene_description": "A test chapter",
                "key_beats": ["Beat 1", "Beat 2"],
            }
        }
        yield mock


@pytest.fixture
def mock_save_chapter_plan():
    with patch("core.langgraph.nodes.scene_planning_node.save_chapter_plan") as mock:
        mock.return_value = {"path": "mock/path/chapter_plan.json", "size_bytes": 123, "version": 1}
        yield mock


@pytest.fixture
def mock_llm_service():
    with patch("core.langgraph.nodes.scene_planning_node.llm_service") as mock:
        mock.async_call_llm = AsyncMock(return_value=("[]", {}))
        yield mock


@pytest.fixture
def mock_character_sync():
    with patch("core.langgraph.nodes.scene_planning_node.get_all_character_names") as mock_get_names, patch("core.langgraph.nodes.scene_planning_node.sync_characters") as mock_sync:
        mock_get_names.return_value = []
        mock_sync.return_value = True
        yield {"get_names": mock_get_names, "sync": mock_sync}


def _valid_scene_list():
    return [
        {
            "title": "Scene 1",
            "pov_character": "Hero",
            "setting": "Castle",
            "characters": ["Hero", "Mentor"],
            "plot_point": "Quest begins",
            "conflict": "Doubt vs duty",
            "outcome": "Hero departs",
        }
    ]


@pytest.mark.asyncio
async def test_plan_scenes_parses_valid_json_list(
    base_state,
    mock_content_manager,
    mock_get_chapter_outlines,
    mock_save_chapter_plan,
    mock_llm_service,
    mock_character_sync,
):
    scenes = _valid_scene_list()
    mock_llm_service.async_call_llm = AsyncMock(return_value=(json.dumps(scenes), {}))

    result = await plan_scenes(base_state)

    assert result["current_node"] == "plan_scenes"
    assert result["last_error"] is None if "last_error" in result else True
    assert result["chapter_plan"] == scenes
    assert result["chapter_plan_ref"]["path"] == "mock/path/chapter_plan.json"


@pytest.mark.asyncio
async def test_plan_scenes_parses_json_in_markdown_fence(
    base_state,
    mock_content_manager,
    mock_get_chapter_outlines,
    mock_save_chapter_plan,
    mock_llm_service,
    mock_character_sync,
):
    scenes = _valid_scene_list()
    response = "```json\n" + json.dumps(scenes) + "\n```"
    mock_llm_service.async_call_llm = AsyncMock(return_value=(response, {}))

    result = await plan_scenes(base_state)

    assert result["chapter_plan"] == scenes


@pytest.mark.asyncio
async def test_plan_scenes_parses_json_with_commentary_before_and_after(
    base_state,
    mock_content_manager,
    mock_get_chapter_outlines,
    mock_save_chapter_plan,
    mock_llm_service,
    mock_character_sync,
):
    scenes = _valid_scene_list()
    response = "Here is the plan:\n\n" + json.dumps(scenes) + "\n\nHope this helps."
    mock_llm_service.async_call_llm = AsyncMock(return_value=(response, {}))

    result = await plan_scenes(base_state)

    assert result["chapter_plan"] == scenes


@pytest.mark.asyncio
async def test_plan_scenes_invalid_json_returns_clear_error(
    base_state,
    mock_content_manager,
    mock_get_chapter_outlines,
    mock_save_chapter_plan,
    mock_llm_service,
    mock_character_sync,
):
    # Not JSON
    mock_llm_service.async_call_llm = AsyncMock(return_value=("not json at all", {}))

    result = await plan_scenes(base_state)

    assert result["current_node"] == "plan_scenes"
    assert "last_error" in result
    assert "Expected: JSON list of scene objects" in result["last_error"]
