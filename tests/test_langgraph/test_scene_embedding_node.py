# tests/test_langgraph/test_scene_embedding_node.py
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from core.langgraph.content_manager import ContentManager, load_scene_embeddings
from core.langgraph.nodes.embedding_node import generate_scene_embeddings


@pytest.mark.asyncio
async def test_generate_scene_embeddings_persists_and_returns_ref(tmp_path: Path) -> None:
    project_dir = str(tmp_path / "proj")
    manager = ContentManager(project_dir)

    scene_drafts_ref = manager.save_list_of_texts(
        ["Scene one text.", "Scene two text."],
        "scenes",
        "chapter_1",
        version=1,
    )

    state = {
        "project_dir": project_dir,
        "current_chapter": 1,
        "scene_drafts_ref": scene_drafts_ref,
    }

    fake_embeddings = [
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        np.array([0.4, 0.5, 0.6], dtype=np.float32),
    ]

    with patch(
        "core.langgraph.nodes.embedding_node.llm_service.async_get_embeddings_batch",
        new=AsyncMock(return_value=fake_embeddings),
    ):
        update = await generate_scene_embeddings(state)  # type: ignore[arg-type]

    assert update["current_node"] == "generate_scene_embeddings"
    assert update.get("scene_embeddings_ref"), "generate_scene_embeddings must set scene_embeddings_ref on success"

    scene_embeddings_ref = update["scene_embeddings_ref"]
    assert isinstance(scene_embeddings_ref, dict)
    assert scene_embeddings_ref["path"].endswith(".json")

    loaded = load_scene_embeddings(manager, scene_embeddings_ref)
    assert loaded == [
        pytest.approx([0.1, 0.2, 0.3], abs=1e-6),
        pytest.approx([0.4, 0.5, 0.6], abs=1e-6),
    ]


@pytest.mark.asyncio
async def test_generate_scene_embeddings_missing_scene_drafts_ref_sets_last_error(tmp_path: Path) -> None:
    project_dir = str(tmp_path / "proj")
    state = {
        "project_dir": project_dir,
        "current_chapter": 1,
        "scene_drafts_ref": None,
    }

    update = await generate_scene_embeddings(state)  # type: ignore[arg-type]

    assert update == {
        "current_node": "generate_scene_embeddings",
        "last_error": "Scene embedding generation skipped: scene_drafts_ref is missing",
    }


@pytest.mark.asyncio
async def test_generate_scene_embeddings_empty_scenes_sets_last_error(tmp_path: Path) -> None:
    project_dir = str(tmp_path / "proj")
    manager = ContentManager(project_dir)

    scene_drafts_ref = manager.save_list_of_texts(
        [],
        "scenes",
        "chapter_1",
        version=1,
    )

    state = {
        "project_dir": project_dir,
        "current_chapter": 1,
        "scene_drafts_ref": scene_drafts_ref,
    }

    update = await generate_scene_embeddings(state)  # type: ignore[arg-type]

    assert update == {
        "current_node": "generate_scene_embeddings",
        "last_error": "Scene embedding generation skipped: no scene drafts found",
    }


@pytest.mark.asyncio
async def test_generate_scene_embeddings_single_scene_chapter(tmp_path: Path) -> None:
    project_dir = str(tmp_path / "proj")
    manager = ContentManager(project_dir)

    scene_drafts_ref = manager.save_list_of_texts(
        ["Only scene text."],
        "scenes",
        "chapter_1",
        version=1,
    )

    state = {
        "project_dir": project_dir,
        "current_chapter": 1,
        "scene_drafts_ref": scene_drafts_ref,
    }

    fake_embeddings = [np.array([0.25, 0.5], dtype=np.float32)]

    with patch(
        "core.langgraph.nodes.embedding_node.llm_service.async_get_embeddings_batch",
        new=AsyncMock(return_value=fake_embeddings),
    ):
        update = await generate_scene_embeddings(state)  # type: ignore[arg-type]

    assert update["current_node"] == "generate_scene_embeddings"
    assert update.get("scene_embeddings_ref")

    loaded = load_scene_embeddings(manager, update["scene_embeddings_ref"])
    assert loaded == [pytest.approx([0.25, 0.5], abs=1e-6)]
