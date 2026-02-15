"""Tests for assemble_chapter node with real ContentManager."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.assemble_chapter_node import assemble_chapter

SCENE_SEPARATOR = "\n\n# ***\n\n"


class TestAssembleChapterEmptyScenes:
    """Empty scene drafts produce a null draft."""

    @pytest.mark.asyncio
    async def test_no_scene_drafts_ref(self, tmp_path: Path) -> None:
        state = {
            "project_dir": str(tmp_path),
            "current_chapter": 1,
            "scene_drafts_ref": None,
        }

        result = await assemble_chapter(state)

        assert result["draft_ref"] is None
        assert result["draft_word_count"] == 0
        assert result["current_node"] == "assemble_chapter"

    @pytest.mark.asyncio
    async def test_empty_scene_list(self, tmp_path: Path) -> None:
        content_manager = ContentManager(str(tmp_path))
        scene_drafts_ref = content_manager.save_list_of_texts([], "scenes", "chapter_1", 1)

        state = {
            "project_dir": str(tmp_path),
            "current_chapter": 1,
            "scene_drafts_ref": scene_drafts_ref,
        }

        result = await assemble_chapter(state)

        assert result["draft_ref"] is None
        assert result["draft_word_count"] == 0
        assert result["current_node"] == "assemble_chapter"


class TestAssembleChapterSingleScene:
    """Single scene produces a chapter with no separator."""

    @pytest.mark.asyncio
    async def test_single_scene_content(self, tmp_path: Path) -> None:
        content_manager = ContentManager(str(tmp_path))
        scene_text = "The sun rose over the distant mountains."
        scene_drafts_ref = content_manager.save_list_of_texts([scene_text], "scenes", "chapter_1", 1)

        state = {
            "project_dir": str(tmp_path),
            "current_chapter": 1,
            "scene_drafts_ref": scene_drafts_ref,
        }

        result = await assemble_chapter(state)

        assert result["draft_ref"] is not None
        assert result["current_node"] == "assemble_chapter"

        draft_text = content_manager.load_text(result["draft_ref"])
        assert draft_text == scene_text

    @pytest.mark.asyncio
    async def test_single_scene_word_count(self, tmp_path: Path) -> None:
        content_manager = ContentManager(str(tmp_path))
        scene_text = "one two three four five"
        scene_drafts_ref = content_manager.save_list_of_texts([scene_text], "scenes", "chapter_1", 1)

        state = {
            "project_dir": str(tmp_path),
            "current_chapter": 1,
            "scene_drafts_ref": scene_drafts_ref,
        }

        result = await assemble_chapter(state)

        assert result["draft_word_count"] == 5


class TestAssembleChapterMultipleScenes:
    """Multiple scenes are joined with the scene separator."""

    @pytest.mark.asyncio
    async def test_two_scenes_joined_with_separator(self, tmp_path: Path) -> None:
        content_manager = ContentManager(str(tmp_path))
        scenes = [
            "First scene text here.",
            "Second scene text here.",
        ]
        scene_drafts_ref = content_manager.save_list_of_texts(scenes, "scenes", "chapter_1", 1)

        state = {
            "project_dir": str(tmp_path),
            "current_chapter": 1,
            "scene_drafts_ref": scene_drafts_ref,
        }

        result = await assemble_chapter(state)

        draft_text = content_manager.load_text(result["draft_ref"])
        expected = SCENE_SEPARATOR.join(scenes)
        assert draft_text == expected

    @pytest.mark.asyncio
    async def test_three_scenes_joined_with_separator(self, tmp_path: Path) -> None:
        content_manager = ContentManager(str(tmp_path))
        scenes = [
            "The dawn broke quietly.",
            "By midday the storm arrived.",
            "Night brought an uneasy peace.",
        ]
        scene_drafts_ref = content_manager.save_list_of_texts(scenes, "scenes", "chapter_1", 1)

        state = {
            "project_dir": str(tmp_path),
            "current_chapter": 1,
            "scene_drafts_ref": scene_drafts_ref,
        }

        result = await assemble_chapter(state)

        draft_text = content_manager.load_text(result["draft_ref"])
        assert draft_text.count("# ***") == 2

    @pytest.mark.asyncio
    async def test_multi_scene_word_count(self, tmp_path: Path) -> None:
        content_manager = ContentManager(str(tmp_path))
        scenes = [
            "alpha beta gamma",
            "delta epsilon",
        ]
        scene_drafts_ref = content_manager.save_list_of_texts(scenes, "scenes", "chapter_1", 1)

        state = {
            "project_dir": str(tmp_path),
            "current_chapter": 1,
            "scene_drafts_ref": scene_drafts_ref,
        }

        result = await assemble_chapter(state)

        full_text = SCENE_SEPARATOR.join(scenes)
        expected_word_count = len(full_text.split())
        assert result["draft_word_count"] == expected_word_count


class TestAssembleChapterExternalization:
    """Assembled drafts and scene lists are externalized to disk."""

    @pytest.mark.asyncio
    async def test_draft_and_scenes_externalized(self, tmp_path: Path) -> None:
        content_manager = ContentManager(str(tmp_path))
        scenes = ["Scene one.", "Scene two."]
        scene_drafts_ref = content_manager.save_list_of_texts(scenes, "scenes", "chapter_1", 1)

        state = {
            "project_dir": str(tmp_path),
            "current_chapter": 1,
            "scene_drafts_ref": scene_drafts_ref,
        }

        result = await assemble_chapter(state)

        assert content_manager.exists(result["draft_ref"]) is True
        assert content_manager.exists(result["scene_drafts_ref"]) is True

    @pytest.mark.asyncio
    async def test_draft_ref_metadata(self, tmp_path: Path) -> None:
        content_manager = ContentManager(str(tmp_path))
        scenes = ["Words here."]
        scene_drafts_ref = content_manager.save_list_of_texts(scenes, "scenes", "chapter_3", 1)

        state = {
            "project_dir": str(tmp_path),
            "current_chapter": 3,
            "scene_drafts_ref": scene_drafts_ref,
        }

        result = await assemble_chapter(state)

        draft_ref = result["draft_ref"]
        assert draft_ref["content_type"] == "draft"
        assert draft_ref["version"] == 1
        assert draft_ref["size_bytes"] > 0
