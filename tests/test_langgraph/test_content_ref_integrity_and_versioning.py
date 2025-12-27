from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from core.langgraph.content_manager import ContentManager


def test_content_ref_is_immutable(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))

    ref = manager.save_text("hello", content_type="unit_test_text", identifier="sample", version=1)

    with pytest.raises(TypeError, match=r"ContentRef is immutable"):
        ref["path"] = "mutated/path.txt"  # type: ignore[misc]


def test_load_text_raises_on_checksum_mismatch(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))

    ref = manager.save_text("hello", content_type="unit_test_text", identifier="sample", version=1)

    full_path = tmp_path / ref["path"]
    assert full_path.is_file()

    # Corrupt the file content after persisting the reference.
    full_path.write_text("corrupted", encoding="utf-8")

    with pytest.raises(ValueError, match=r"checksum mismatch"):
        manager.load_text(ref)


def test_load_text_allows_missing_checksum_for_backward_compatibility(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))

    ref = manager.save_text("hello", content_type="unit_test_text", identifier="sample", version=1)

    ref_without_checksum = {"path": ref["path"], "content_type": ref["content_type"], "version": ref["version"]}

    assert manager.load_text(ref_without_checksum) == "hello"


@pytest.mark.asyncio
async def test_retrieve_context_writes_new_versions_without_overwriting(tmp_path: Path) -> None:
    from core.langgraph.nodes.context_retrieval_node import retrieve_context

    project_dir = str(tmp_path)
    manager = ContentManager(project_dir)

    chapter_plan_ref = manager.save_json(
        [
            {
                "title": "Scene One",
                "scene_description": "A first scene",
                "location": None,
            }
        ],
        content_type="chapter_plan",
        identifier="chapter_1",
        version=1,
    )

    state = {
        "project_dir": project_dir,
        "current_chapter": 1,
        "current_scene_index": 0,
        "chapter_plan_ref": chapter_plan_ref,
        "narrative_model": "unit-test-model",
        "small_model": "unit-test-model",
        "summaries_ref": None,
        "scene_drafts_ref": None,
    }

    with (
        patch(
            "core.langgraph.nodes.context_retrieval_node._get_scene_character_context",
            new=AsyncMock(side_effect=["CHARACTER_CONTEXT_ONE", "CHARACTER_CONTEXT_TWO"]),
        ),
        patch(
            "core.langgraph.nodes.context_retrieval_node._get_scene_specific_kg_facts",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "core.langgraph.nodes.context_retrieval_node._get_previous_scenes_context",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "core.langgraph.nodes.context_retrieval_node._get_scene_location_context",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "core.langgraph.nodes.context_retrieval_node._get_semantic_context",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "core.langgraph.nodes.context_retrieval_node.count_tokens",
            return_value=0,
        ),
    ):
        first_update = await retrieve_context(state)  # type: ignore[arg-type]
        first_ref = first_update["hybrid_context_ref"]
        assert isinstance(first_ref, dict)
        assert first_ref["version"] == 1

        second_state = {**state, **first_update}
        second_update = await retrieve_context(second_state)  # type: ignore[arg-type]
        second_ref = second_update["hybrid_context_ref"]
        assert isinstance(second_ref, dict)
        assert second_ref["version"] == 2

    assert first_ref["path"] != second_ref["path"]

    # Both artifacts must exist and remain readable (older checkpoint still valid).
    assert (tmp_path / first_ref["path"]).is_file()
    assert (tmp_path / second_ref["path"]).is_file()

    first_text = manager.load_text(first_ref)
    second_text = manager.load_text(second_ref)

    assert "CHARACTER_CONTEXT_ONE" in first_text
    assert "CHARACTER_CONTEXT_TWO" in second_text