"""Cross-owner scene-plan admission reproductions for coordinator integration."""
import json
from pathlib import Path
from typing import Any

import pytest

from core.langgraph.nodes.scene_planning_node import plan_scenes
from core.service_context import get_services
from tests.test_r08s_producer_composition import prepare_enriched_state


@pytest.mark.run_settings(TARGET_SCENES_MIN=2)
@pytest.mark.parametrize("case", ["missing_beats", "wrong_type", "wrong_count"])
async def test_scene_plan_must_preserve_selected_outline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    state, _ = await prepare_enriched_state(tmp_path, monkeypatch)
    state["current_chapter"] = 1
    scene: dict[str, Any] = dict(title="Harbor", pov_character="Ada", setting="Harbor", characters=["Ada"], plot_point="Choice", conflict="Duty", outcome="Stay", beats=["Ada chooses", "Ada visits", "Ada remains"])
    if case == "missing_beats":
        scene["beats"] = ["Unselected new beat"]
    elif case == "wrong_type":
        scene["setting"] = 123
    scenes = [scene] if case == "wrong_count" else [scene, {**scene, "beats": []}]
    requests = []

    async def transport(**arguments: Any) -> tuple[str, dict[str, Any]]:
        requests.append(arguments)
        return json.dumps(scenes), {}

    monkeypatch.setattr(get_services().language_model, "async_call_llm", transport)
    result = await plan_scenes(state)
    assert requests
    assert result.get("has_fatal_error") is True
    assert result.get("chapter_plan_ref") is None
    expected = {"missing_beats": "ordered exhaustive partition", "wrong_type": "setting", "wrong_count": "exactly 2 scenes"}
    assert expected[case] in result["last_error"]
