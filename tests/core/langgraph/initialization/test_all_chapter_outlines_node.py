"""Tests for core/langgraph/initialization/all_chapter_outlines_node.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from core.langgraph.initialization.all_chapter_outlines_node import (
    generate_all_chapter_outlines,
)


def _make_state(tmp_path: str, total_chapters: int = 3) -> dict:
    return {
        "project_dir": tmp_path,
        "title": "The Lost Kingdom",
        "genre": "Fantasy",
        "theme": "Adventure",
        "setting": "Medieval realm",
        "total_chapters": total_chapters,
        "protagonist_name": "Aria",
        "target_word_count": 60000,
    }


def _fake_outline(chapter_number: int) -> dict:
    return {
        "chapter_number": chapter_number,
        "title": f"Chapter {chapter_number}",
        "summary": f"Summary for chapter {chapter_number}",
    }


class TestGenerateAllChapterOutlines:
    """Tests for the generate_all_chapter_outlines node."""

    async def test_config_disabled_skips(self, tmp_path) -> None:
        state = _make_state(str(tmp_path))

        with patch("config.settings.settings") as fake_settings:
            fake_settings.GENERATE_ALL_CHAPTER_OUTLINES_AT_INIT = False

            result = await generate_all_chapter_outlines(state)

        assert result["current_node"] == "all_chapter_outlines"
        assert result["initialization_step"] == "all_chapter_outlines_skipped"
        assert "chapter_outlines_ref" not in result

    async def test_successful_generation(self, tmp_path) -> None:
        state = _make_state(str(tmp_path), total_chapters=3)

        with (
            patch("config.settings.settings") as fake_settings,
            patch(
                "core.langgraph.initialization.all_chapter_outlines_node._determine_act_for_chapter",
                return_value=1,
            ),
            patch(
                "core.langgraph.initialization.all_chapter_outlines_node._generate_single_chapter_outline",
                new_callable=AsyncMock,
            ) as fake_generate,
        ):
            fake_settings.GENERATE_ALL_CHAPTER_OUTLINES_AT_INIT = True
            fake_generate.side_effect = lambda state, chapter_number, act_number: _fake_outline(chapter_number)

            result = await generate_all_chapter_outlines(state)

        assert result["current_node"] == "all_chapter_outlines"
        assert result["initialization_step"] == "all_chapter_outlines_complete"
        assert result["last_error"] is None
        ref = result["chapter_outlines_ref"]
        assert ref["content_type"] == "chapter_outlines"
        assert ref["version"] == 0

    async def test_partial_failure_continues(self, tmp_path) -> None:
        state = _make_state(str(tmp_path), total_chapters=3)

        call_count = 0

        async def partial_generator(state, chapter_number, act_number):
            nonlocal call_count
            call_count += 1
            if chapter_number == 2:
                return None
            return _fake_outline(chapter_number)

        with (
            patch("config.settings.settings") as fake_settings,
            patch(
                "core.langgraph.initialization.all_chapter_outlines_node._determine_act_for_chapter",
                return_value=1,
            ),
            patch(
                "core.langgraph.initialization.all_chapter_outlines_node._generate_single_chapter_outline",
                new_callable=AsyncMock,
                side_effect=partial_generator,
            ),
        ):
            fake_settings.GENERATE_ALL_CHAPTER_OUTLINES_AT_INIT = True

            result = await generate_all_chapter_outlines(state)

        assert call_count == 3
        assert result["current_node"] == "all_chapter_outlines"
        assert result["initialization_step"] == "all_chapter_outlines_complete"
        assert result["last_error"] is None
        assert result["chapter_outlines_ref"] is not None

    async def test_total_failure_returns_error(self, tmp_path) -> None:
        state = _make_state(str(tmp_path), total_chapters=2)

        with (
            patch("config.settings.settings") as fake_settings,
            patch(
                "core.langgraph.initialization.all_chapter_outlines_node._determine_act_for_chapter",
                return_value=1,
            ),
            patch(
                "core.langgraph.initialization.all_chapter_outlines_node._generate_single_chapter_outline",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            fake_settings.GENERATE_ALL_CHAPTER_OUTLINES_AT_INIT = True

            result = await generate_all_chapter_outlines(state)

        assert result["current_node"] == "all_chapter_outlines"
        assert result["initialization_step"] == "all_chapter_outlines_failed"
        assert "Failed to generate any chapter outlines" in result["last_error"]
        assert "chapter_outlines_ref" not in result
