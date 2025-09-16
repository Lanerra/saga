# ui/rich_display.py
from __future__ import annotations

import asyncio
import time
from typing import Any

import config
from core.llm_interface_refactored import llm_service

try:
    from rich.console import Group, Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    RICH_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when Rich isn't installed
    RICH_AVAILABLE = False

    class Live:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def start(self) -> None:  # pragma: no cover - noop fallback
            pass

        def stop(self) -> None:  # pragma: no cover - noop fallback
            pass

    class Text:  # type: ignore
        def __init__(self, initial_text: str = "") -> None:
            self.plain = initial_text

    class Group:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class Panel:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


class RichDisplayManager:
    """Handles Rich-based display updates."""

    # Shared Console singleton to coordinate Live + logging output
    _shared_console: Console | None = None

    @classmethod
    def get_shared_console(cls) -> Console:
        if cls._shared_console is None:
            # Create a single Console instance for the whole app
            cls._shared_console = Console()
        return cls._shared_console

    def __init__(self) -> None:
        self.live: Live | None = None
        self.group: Group | None = None
        self.status_text_novel_title: Text = Text("Novel: N/A")
        self.status_text_current_chapter: Text = Text("Current Chapter: N/A")
        self.status_text_current_step: Text = Text("Current Step: Initializing...")
        self.status_text_elapsed_time: Text = Text("Elapsed Time: 0s")
        self.status_text_requests_per_minute: Text = Text("Requests/Min: 0.0")
        self.run_start_time: float = 0.0
        self._stop_event: asyncio.Event = asyncio.Event()
        self._task: asyncio.Task | None = None

        if RICH_AVAILABLE and config.ENABLE_RICH_PROGRESS:
            self.group = Group(
                self.status_text_novel_title,
                self.status_text_current_chapter,
                self.status_text_current_step,
                self.status_text_requests_per_minute,
                self.status_text_elapsed_time,
            )
            # Use a single shared Console so Live and Rich logging share output
            console = self.get_shared_console()
            self.live = Live(
                Panel(
                    self.group,
                    title="SAGA Progress",
                    border_style="blue",
                    expand=True,
                ),
                refresh_per_second=4,
                transient=False,
                # Capture any stray prints/stdout so the banner stays anchored
                redirect_stdout=True,
                redirect_stderr=True,
                console=console,
            )

    def start(self) -> None:
        if self.live:
            self.run_start_time = time.time()
            self.live.start()
            self._stop_event.clear()
            self._task = asyncio.create_task(self._auto_refresh())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task:
            await self._task
            self._task = None
        if self.live and self.live.is_started:
            self.live.stop()

    async def _auto_refresh(self) -> None:
        while not self._stop_event.is_set():
            self.update()
            await asyncio.sleep(1)

    def update(
        self,
        plot_outline: dict[str, Any] | None = None,
        chapter_num: int | None = None,
        step: str | None = None,
        run_start_time: float | None = None,
    ) -> None:
        if not (self.live and self.group):
            return
        if plot_outline is not None:
            self.status_text_novel_title.plain = (
                f"Novel: {plot_outline.get('title', 'N/A')}"
            )
        if chapter_num is not None:
            self.status_text_current_chapter.plain = f"Current Chapter: {chapter_num}"
        if step is not None:
            self.status_text_current_step.plain = f"Current Step: {step}"
        start_time = run_start_time or self.run_start_time
        elapsed_seconds = time.time() - start_time
        # Get request count from the refactored service statistics
        try:
            stats = llm_service.get_combined_statistics()
            request_count = stats.get("completion_service", {}).get(
                "completions_requested", 0
            )
            requests_per_minute = (
                request_count / (elapsed_seconds / 60) if elapsed_seconds > 0 else 0.0
            )
        except (AttributeError, KeyError):
            requests_per_minute = 0.0
        self.status_text_requests_per_minute.plain = (
            f"Requests/Min: {requests_per_minute:.2f}"
        )
        self.status_text_elapsed_time.plain = (
            f"Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_seconds))}"
        )
        # Force a refresh to keep the live panel anchored and updated
        try:
            self.live.refresh()
        except Exception:
            pass
