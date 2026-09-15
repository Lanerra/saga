# tests/test_langgraph/test_scene_planning_node.py
import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import config
from core.langgraph.nodes.scene_planning_node import plan_scenes
from core.langgraph.state import NarrativeState
from tests.fakes.service_context import patch_service
from tests.test_r08g_catalog_fixtures import catalog_state


@pytest.fixture(autouse=True)
def one_scene_settings() -> Iterator[None]:
    effective = config.EffectiveSettings.model_validate({**config.snapshot_settings().model_dump(), "TARGET_SCENES_MIN": 1})
    with config.bind_settings(effective):
        yield


@pytest.fixture
def base_state(tmp_path: Path) -> NarrativeState:
    state: NarrativeState = {
        "project_dir": str(tmp_path),
        "title": "Test Novel",
        "genre": "Fantasy",
        "theme": "Adventure",
        "large_model": "test-model",
        "current_chapter": 1,
    }
    return catalog_state(tmp_path, characters=("Hero", "Mentor"), locations=("Castle",), events=("Beat 1", "Beat 2"), existing=state)


@pytest.fixture
def mock_content_manager() -> Iterator[MagicMock]:
    with patch("core.langgraph.nodes.scene_planning_node.ContentManager") as mock:
        instance = MagicMock()
        instance.get_latest_version.return_value = 0
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_get_chapter_outlines() -> Iterator[MagicMock]:
    with patch("core.langgraph.nodes.scene_planning_node.get_chapter_outlines") as mock:
        mock.return_value = {
            1: {
                "scene_description": "A test chapter",
                "key_beats": ["Beat 1", "Beat 2"],
            }
        }
        yield mock


@pytest.fixture
def mock_save_chapter_plan() -> Iterator[MagicMock]:
    with patch("core.langgraph.nodes.scene_planning_node.save_chapter_plan") as mock:
        mock.return_value = {"path": "mock/path/chapter_plan.json", "size_bytes": 123, "version": 1}
        yield mock


@pytest.fixture
def mock_llm_service() -> Iterator[MagicMock]:
    with patch_service('language_model') as mock:
        mock.async_call_llm = AsyncMock(return_value=("[]", {}))
        yield mock


@pytest.fixture
def mock_character_sync() -> Iterator[dict[str, AsyncMock]]:
    with patch("core.langgraph.nodes.scene_planning_node.get_all_character_names") as mock_get_names, patch("core.langgraph.nodes.scene_planning_node.sync_characters") as mock_sync:
        mock_get_names.return_value = []
        mock_sync.return_value = True
        yield {"get_names": mock_get_names, "sync": mock_sync}


def _valid_scene_list() -> list[dict[str, object]]:
    return [
        {
            "title": "Scene 1",
            "pov_character": "Hero",
            "setting": "Castle",
            "characters": ["Hero", "Mentor"],
            "plot_point": "Quest begins",
            "conflict": "Doubt vs duty",
            "outcome": "Hero departs",
            "beats": ["Beat 1", "Beat 2"],
        }
    ]


@pytest.mark.asyncio
async def test_plan_scenes_parses_valid_json_list(
    base_state: NarrativeState,
    mock_content_manager: MagicMock,
    mock_get_chapter_outlines: MagicMock,
    mock_save_chapter_plan: MagicMock,
    mock_llm_service: MagicMock,
    mock_character_sync: dict[str, AsyncMock],
) -> None:
    scenes = _valid_scene_list()
    mock_llm_service.async_call_llm = AsyncMock(return_value=(json.dumps(scenes), {}))

    result = await plan_scenes(base_state)

    assert result["current_node"] == "plan_scenes"
    assert result["last_error"] is None if "last_error" in result else True
    assert result["chapter_plan_scene_count"] == len(scenes)
    assert result["chapter_plan_ref"] is not None
    assert result["chapter_plan_ref"]["path"] == "mock/path/chapter_plan.json"


@pytest.mark.asyncio
async def test_plan_scenes_parses_strict_json_array(
    base_state: NarrativeState,
    mock_content_manager: MagicMock,
    mock_get_chapter_outlines: MagicMock,
    mock_save_chapter_plan: MagicMock,
    mock_llm_service: MagicMock,
    mock_character_sync: dict[str, AsyncMock],
) -> None:
    scenes = _valid_scene_list()
    response = json.dumps(scenes)
    mock_llm_service.async_call_llm = AsyncMock(return_value=(response, {}))

    result = await plan_scenes(base_state)

    assert result["chapter_plan_scene_count"] == len(scenes)


@pytest.mark.asyncio
async def test_plan_scenes_rejects_json_with_surrounding_text(
    base_state: NarrativeState,
    mock_content_manager: MagicMock,
    mock_get_chapter_outlines: MagicMock,
    mock_save_chapter_plan: MagicMock,
    mock_llm_service: MagicMock,
    mock_character_sync: dict[str, AsyncMock],
) -> None:
    scenes = _valid_scene_list()
    response = "Here is the plan:\n\n" + json.dumps(scenes) + "\n\nHope this helps."
    mock_llm_service.async_call_llm = AsyncMock(return_value=(response, {}))

    result = await plan_scenes(base_state)

    assert "last_error" in result
    assert result["last_error"] is not None
    assert "Scene plan contract violation" in result["last_error"]


@pytest.mark.asyncio
async def test_plan_scenes_invalid_json_returns_clear_error(
    base_state: NarrativeState,
    mock_content_manager: MagicMock,
    mock_get_chapter_outlines: MagicMock,
    mock_save_chapter_plan: MagicMock,
    mock_llm_service: MagicMock,
    mock_character_sync: dict[str, AsyncMock],
) -> None:
    # Not JSON
    mock_llm_service.async_call_llm = AsyncMock(return_value=("not json at all", {}))

    result = await plan_scenes(base_state)

    assert result["current_node"] == "plan_scenes"
    assert "last_error" in result
    assert result["last_error"] is not None
    assert "Expected: JSON array of scene objects" in result["last_error"]
