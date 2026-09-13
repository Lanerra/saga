"""Cross-owner scene-plan admission reproductions for coordinator integration."""
from pathlib import Path
from typing import Any
import json

import pytest

from core.langgraph.nodes.scene_planning_node import plan_scenes
from core.service_context import get_services
from tests.test_staged_initialization import example_state


@pytest.mark.parametrize("case", ["missing_beats", "wrong_type", "wrong_count"])
async def test_scene_plan_must_preserve_selected_outline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
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
