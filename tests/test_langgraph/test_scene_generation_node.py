# tests/test_langgraph/test_scene_generation_node.py
from __future__ import annotations

from pathlib import Path

import pytest

from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.scene_generation_node import draft_scene
from core.langgraph.state import NarrativeState


@pytest.mark.asyncio
async def test_draft_scene_returns_partial_update_on_invalid_index(tmp_path: Path) -> None:
    """Invalid scene index should return only current_node."""
    content_manager = ContentManager(str(tmp_path))
    chapter_plan_ref = content_manager.save_json(
        [{"title": "Intro"}],
        content_type="chapter_plan",
        identifier="chapter_1",
        version=1,
    )

    state: NarrativeState = {
        "project_dir": str(tmp_path),
        "current_chapter": 1,
        "current_scene_index": 5,
        "chapter_plan_ref": chapter_plan_ref,
    }

    result = await draft_scene(state)

    assert result["current_node"] == "draft_scene"
    assert result["has_fatal_error"] is True
    assert result["error_node"] == "draft_scene"
    assert "Invalid scene index" in result["last_error"]


@pytest.mark.asyncio
async def test_draft_scene_raises_when_project_dir_missing(tmp_path: Path) -> None:
    content_manager = ContentManager(str(tmp_path))
    chapter_plan_ref = content_manager.save_json(
        [{"title": "Intro"}],
        content_type="chapter_plan",
        identifier="chapter_1",
        version=1,
    )

    state: NarrativeState = {
        "current_chapter": 1,
        "current_scene_index": 0,
        "chapter_plan_ref": chapter_plan_ref,
    }

    with pytest.raises(ValueError) as exc:
        await draft_scene(state)

    assert str(exc.value) == "project_dir is required"
