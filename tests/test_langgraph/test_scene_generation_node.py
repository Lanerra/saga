# tests/test_langgraph/test_scene_generation_node.py
from __future__ import annotations

from pathlib import Path

import pytest

from core.langgraph.nodes.scene_generation_node import draft_scene
from core.langgraph.state import NarrativeState


@pytest.mark.asyncio
async def test_draft_scene_returns_partial_update_on_invalid_index(tmp_path: Path) -> None:
    """Invalid scene index should return only current_node."""
    state: NarrativeState = {
        "project_dir": str(tmp_path),
        "current_chapter": 1,
        "current_scene_index": 5,
        "chapter_plan": [{"title": "Intro"}],
    }

    result = await draft_scene(state)

    assert result == {"current_node": "draft_scene"}
