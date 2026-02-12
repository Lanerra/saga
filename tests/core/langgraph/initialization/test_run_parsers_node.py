"""Tests for core/langgraph/initialization/run_parsers_node.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from core.langgraph.initialization.run_parsers_node import run_initialization_parsers


def _make_state(project_dir: str = "/tmp/fake-project") -> dict:
    return {"project_dir": project_dir}


class TestRunInitializationParsers:
    """Tests for the run_initialization_parsers node."""

    async def test_all_parsers_succeed(self, tmp_path: object) -> None:
        project_dir = str(tmp_path)
        state = _make_state(project_dir)

        fake_results = {
            "character_sheets": (True, "ok"),
            "global_outline": (True, "ok"),
            "act_outlines": (True, "ok"),
            "chapter_outlines": (True, "ok"),
        }

        with patch("core.langgraph.initialization.run_parsers_node.ParserRunner") as FakeRunner:
            instance = FakeRunner.return_value
            instance.run_all_parsers = AsyncMock(return_value=fake_results)

            result = await run_initialization_parsers(state)

        assert result["current_node"] == "run_parsers"
        assert result["last_error"] is None
        assert result["initialization_step"] == "parsers_complete"
        assert "has_fatal_error" not in result

    async def test_some_parsers_fail(self, tmp_path: object) -> None:
        project_dir = str(tmp_path)
        state = _make_state(project_dir)

        fake_results = {
            "character_sheets": (True, "ok"),
            "global_outline": (False, "parse error in outline"),
            "act_outlines": (True, "ok"),
            "chapter_outlines": (False, "missing data"),
        }

        with patch("core.langgraph.initialization.run_parsers_node.ParserRunner") as FakeRunner:
            instance = FakeRunner.return_value
            instance.run_all_parsers = AsyncMock(return_value=fake_results)

            result = await run_initialization_parsers(state)

        assert result["current_node"] == "run_parsers"
        assert result["has_fatal_error"] is True
        assert result["error_node"] == "run_parsers"
        assert result["initialization_step"] == "parsers_failed"
        assert "global_outline" in result["last_error"]
        assert "chapter_outlines" in result["last_error"]

    async def test_exception_in_runner(self, tmp_path: object) -> None:
        project_dir = str(tmp_path)
        state = _make_state(project_dir)

        with patch("core.langgraph.initialization.run_parsers_node.ParserRunner") as FakeRunner:
            instance = FakeRunner.return_value
            instance.run_all_parsers = AsyncMock(side_effect=RuntimeError("disk full"))

            result = await run_initialization_parsers(state)

        assert result["current_node"] == "run_parsers"
        assert result["has_fatal_error"] is True
        assert result["error_node"] == "run_parsers"
        assert result["initialization_step"] == "parsers_failed"
        assert "disk full" in result["last_error"]

    async def test_missing_project_dir_raises(self) -> None:
        state: dict = {}
        with pytest.raises(ValueError, match="project_dir is required"):
            await run_initialization_parsers(state)
