from __future__ import annotations

import time
from collections.abc import Generator
from unittest.mock import patch

import pytest
from rich.console import Console

from ui.rich_display import RichDisplayManager


@pytest.fixture(autouse=True)
def _reset_shared_console() -> Generator[None, None, None]:
    original = RichDisplayManager._shared_console
    RichDisplayManager._shared_console = None
    yield
    RichDisplayManager._shared_console = original


class TestDisabledMode:
    @patch("config.ENABLE_RICH_PROGRESS", False)
    def test_live_is_none_when_disabled(self) -> None:
        manager = RichDisplayManager()
        assert manager.live is None

    @patch("config.ENABLE_RICH_PROGRESS", False)
    def test_update_is_noop_when_live_is_none(self) -> None:
        manager = RichDisplayManager()
        manager.update(novel_title="Title", chapter_num=1, step="Step")
        assert manager.status_text_novel_title.plain == "Novel: N/A"

    @patch("config.ENABLE_RICH_PROGRESS", False)
    def test_start_is_noop_when_live_is_none(self) -> None:
        manager = RichDisplayManager()
        manager.start()
        assert manager._task is None
        assert manager.run_start_time == 0.0

    @patch("config.ENABLE_RICH_PROGRESS", False)
    async def test_stop_is_safe_when_no_task_running(self) -> None:
        manager = RichDisplayManager()
        await manager.stop()
        assert manager._task is None


class TestGetSharedConsole:
    def test_returns_console_instance(self) -> None:
        console = RichDisplayManager.get_shared_console()
        assert isinstance(console, Console)

    def test_returns_same_instance_on_second_call(self) -> None:
        first = RichDisplayManager.get_shared_console()
        second = RichDisplayManager.get_shared_console()
        assert first is second


class TestUpdateStatusText:
    @patch("ui.rich_display.llm_service")
    @patch("config.ENABLE_RICH_PROGRESS", True)
    @patch("ui.rich_display.RICH_AVAILABLE", True)
    def test_update_novel_title(self, fake_llm_service: object) -> None:
        fake_llm_service.get_combined_statistics.return_value = {"completion_service": {"completions_requested": 0}}
        manager = RichDisplayManager()
        manager.run_start_time = time.time()
        manager.update(novel_title="The Great Story")
        assert manager.status_text_novel_title.plain == "Novel: The Great Story"

    @patch("ui.rich_display.llm_service")
    @patch("config.ENABLE_RICH_PROGRESS", True)
    @patch("ui.rich_display.RICH_AVAILABLE", True)
    def test_update_chapter_number(self, fake_llm_service: object) -> None:
        fake_llm_service.get_combined_statistics.return_value = {"completion_service": {"completions_requested": 0}}
        manager = RichDisplayManager()
        manager.run_start_time = time.time()
        manager.update(chapter_num=5)
        assert manager.status_text_current_chapter.plain == "Current Chapter: 5"

    @patch("ui.rich_display.llm_service")
    @patch("config.ENABLE_RICH_PROGRESS", True)
    @patch("ui.rich_display.RICH_AVAILABLE", True)
    def test_update_step(self, fake_llm_service: object) -> None:
        fake_llm_service.get_combined_statistics.return_value = {"completion_service": {"completions_requested": 0}}
        manager = RichDisplayManager()
        manager.run_start_time = time.time()
        manager.update(step="Generating")
        assert manager.status_text_current_step.plain == "Current Step: Generating"

    @patch("ui.rich_display.llm_service")
    @patch("config.ENABLE_RICH_PROGRESS", True)
    @patch("ui.rich_display.RICH_AVAILABLE", True)
    def test_elapsed_time_formatted_as_hhmmss(self, fake_llm_service: object) -> None:
        fake_llm_service.get_combined_statistics.return_value = {"completion_service": {"completions_requested": 0}}
        manager = RichDisplayManager()
        # 3723 seconds = 1 hour, 2 minutes, 3 seconds
        manager.update(run_start_time=time.time() - 3723)
        assert manager.status_text_elapsed_time.plain == "Elapsed Time: 01:02:03"


class TestInitialization:
    @patch("config.ENABLE_RICH_PROGRESS", False)
    def test_default_status_texts(self) -> None:
        manager = RichDisplayManager()
        assert manager.status_text_novel_title.plain == "Novel: N/A"
        assert manager.status_text_current_chapter.plain == "Current Chapter: N/A"
        assert manager.status_text_current_step.plain == "Current Step: Initializing..."
        assert manager.status_text_elapsed_time.plain == "Elapsed Time: 0s"
        assert manager.status_text_requests_per_minute.plain == "Requests/Min: 0.0"
