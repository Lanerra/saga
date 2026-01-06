import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.scene_planning_node import _parse_scene_plan_json_from_llm_response, plan_scenes
from core.langgraph.state import create_initial_state


def _valid_scene_plan_json() -> str:
    return json.dumps(
        [
            {
                "title": "Scene 1",
                "pov_character": "Hero",
                "setting": "Room",
                "characters": ["Hero"],
                "plot_point": "Start",
                "conflict": "None",
                "outcome": "Next",
                "beats": "Hero discovers something unexpected",
            }
        ]
    )


def test_parse_scene_plan_rejects_wrapper_object() -> None:
    response = json.dumps({"scenes": json.loads(_valid_scene_plan_json())})

    with pytest.raises(ValueError, match=r"^Scene plan contract violation: top-level JSON must be an array, not an object\.$"):
        _parse_scene_plan_json_from_llm_response(response)


def test_parse_scene_plan_missing_required_keys_fails() -> None:
    response = json.dumps([{"title": "Only title"}])

    with pytest.raises(ValueError, match=r"^Scene plan contract violation: invalid structure;"):
        _parse_scene_plan_json_from_llm_response(response)


def test_parse_scene_plan_missing_beats_field_fails() -> None:
    scene_without_beats = {
        "title": "Scene 1",
        "pov_character": "Hero",
        "setting": "Room",
        "characters": ["Hero"],
        "plot_point": "Start",
        "conflict": "None",
        "outcome": "Next",
    }
    response = json.dumps([scene_without_beats])

    with pytest.raises(ValueError, match=r"Scene plan contract violation: invalid structure.*missing required keys.*beats"):
        _parse_scene_plan_json_from_llm_response(response)


def test_parse_scene_plan_with_beats_field_passes() -> None:
    scene_with_beats = {
        "title": "Scene 1",
        "pov_character": "Hero",
        "setting": "Room",
        "characters": ["Hero"],
        "plot_point": "Start",
        "conflict": "None",
        "outcome": "Next",
        "beats": "Hero discovers something unexpected",
    }
    response = json.dumps([scene_with_beats])

    parsed = _parse_scene_plan_json_from_llm_response(response)

    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["beats"] == "Hero discovers something unexpected"


def test_parse_scene_plan_valid_top_level_array_passes() -> None:
    response = _valid_scene_plan_json()

    parsed = _parse_scene_plan_json_from_llm_response(response)

    assert isinstance(parsed, list)
    assert parsed == json.loads(response)


@pytest.mark.asyncio
async def test_plan_scenes_retries_on_invalid_then_succeeds(tmp_path: Path) -> None:
    project_dir = str(tmp_path)

    state = create_initial_state(
        project_id="test",
        title="Test Novel",
        genre="Sci-Fi",
        theme="Testing",
        setting="Lab",
        target_word_count=1000,
        total_chapters=1,
        project_dir=project_dir,
        protagonist_name="Hero",
    )
    state["current_chapter"] = 1

    content_manager = ContentManager(project_dir)
    chapter_outlines = {1: {"scene_description": "Test Chapter", "key_beats": ["Beat 1", "Beat 2"]}}
    state["chapter_outlines_ref"] = content_manager.save_json(chapter_outlines, "chapter_outlines", "all", 1)

    invalid_first_response = json.dumps({"scenes": json.loads(_valid_scene_plan_json())})
    valid_second_response = _valid_scene_plan_json()

    with (
        patch(
            "core.langgraph.nodes.scene_planning_node.llm_service.async_call_llm",
            new_callable=AsyncMock,
        ) as mock_llm,
        patch(
            "core.langgraph.nodes.scene_planning_node._ensure_scene_characters_exist",
            new_callable=AsyncMock,
        ) as mock_ensure_characters,
    ):
        mock_llm.side_effect = [
            (invalid_first_response, {}),
            (valid_second_response, {}),
        ]

        result = await plan_scenes(state)

        assert result["current_node"] == "plan_scenes"
        assert result["chapter_plan_scene_count"] == 1
        assert result["current_scene_index"] == 0
        assert result["chapter_plan_ref"] is not None

        assert mock_llm.call_count == 2
        second_call_prompt = mock_llm.call_args_list[1].kwargs["prompt"]
        assert "Your last response was invalid." in second_call_prompt

        mock_ensure_characters.assert_awaited_once()
