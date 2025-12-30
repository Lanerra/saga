# ui/rich_display.py
"""Render best-effort terminal progress for SAGA generation runs.

This module provides a small wrapper around Rich Live rendering. It is designed
to be safe to call from orchestration code regardless of whether Rich is
installed or Rich progress is enabled in configuration.

Non-goals:
    - Provide a stable API for building arbitrary dashboards.
    - Guarantee that display refresh succeeds under all terminal conditions.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

import config
from core.llm_interface_refactored import llm_service

logger = structlog.get_logger(__name__)

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    RICH_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when Rich isn't installed
    RICH_AVAILABLE = False

    class Live:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def start(self) -> None:  # pragma: no cover - noop fallback
            pass

        def stop(self) -> None:  # pragma: no cover - noop fallback
            pass

    class Text:  # type: ignore[no-redef]
        def __init__(self, initial_text: str = "") -> None:
            self.plain = initial_text

    class Group:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class Panel:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


class RichDisplayManager:
    """Render best-effort Rich Live status for a single generation run.

    The display manager is intentionally lightweight: it tracks a small set of
    status lines (novel title, chapter, step, request rate, elapsed time) and
    refreshes a Rich `Live` panel when available.

    Lifecycle:
        - Call [`start()`](ui/rich_display.py:98) once before emitting updates.
        - Call [`stop()`](ui/rich_display.py:105) to end background refresh and
          stop the Rich `Live` session.

    Error policy:
        - Rendering/refresh is best-effort. Failures during `Live.refresh()` are
          swallowed and logged at debug level.
        - When Rich is unavailable or disabled, all methods are no-ops.

    Notes:
        This class reads request statistics from the global
        [`llm_service`](core/llm_interface_refactored.py:1). Statistics collection
        is best-effort and must not interfere with generation.
    """

    # Shared Console singleton to coordinate Live + logging output
    _shared_console: Console | None = None

    @classmethod
    def get_shared_console(cls) -> Console:
        """Return the process-wide Rich `Console` used by Live rendering.

        The orchestrator and logging output share a single console to reduce
        flicker and avoid interleaving issues.

        Returns:
            A singleton `Console` instance.
        """
        if cls._shared_console is None:
            # Create a single Console instance for the whole app
            cls._shared_console = Console()
        return cls._shared_console

    def __init__(self) -> None:
        """Initialize a display manager.

        Initialization is side-effect free: the Rich `Live` session does not
        start until [`start()`](ui/rich_display.py:98) is called.
        """
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
                redirect_stdout=False,
                redirect_stderr=False,
                console=console,
            )

    def start(self) -> None:
        """Start the Rich `Live` session and background refresh task.

        If Rich is not available or progress is disabled, this method is a no-op.

        Side Effects:
            - Starts a Rich Live render loop (terminal output).
            - Schedules an asyncio task that periodically calls
              [`update()`](ui/rich_display.py:118).
        """
        if self.live:
            self.run_start_time = time.time()
            self.live.start()
            self._stop_event.clear()
            self._task = asyncio.create_task(self._auto_refresh())

    async def stop(self) -> None:
        """Stop background refresh and end the Rich `Live` session.

        This method is intended to be awaited during orchestrator shutdown.

        If Rich is not available or progress is disabled, this method is a no-op.

        Notes:
            This method does not close or reset the shared Rich `Console`.
        """
        self._stop_event.set()
        if self._task:
            await self._task
            self._task = None
        if self.live and self.live.is_started:
            self.live.stop()

    async def _auto_refresh(self) -> None:
        """Refresh the panel periodically until stopped.

        The refresh loop calls [`update()`](ui/rich_display.py:118) with no
        arguments, which recomputes elapsed time and request-rate telemetry.
        """
        while not self._stop_event.is_set():
            self.update()
            await asyncio.sleep(1)

    def update(
        self,
        novel_title: str | None = None,
        chapter_num: int | None = None,
        step: str | None = None,
        run_start_time: float | None = None,
    ) -> None:
        """Update status lines and refresh the Rich Live panel.

        Args:
            novel_title: Optional novel title to display. When omitted, the prior
                value is preserved.
            chapter_num: Optional chapter number to display. When omitted, the
                prior value is preserved.
            step: Optional human-readable step label (for example, the current
                workflow node description).
            run_start_time: Optional override for elapsed-time computation.
                When omitted, uses the timestamp recorded by
                [`start()`](ui/rich_display.py:98).

        Notes:
            - When Rich is disabled/unavailable, this method returns without
              doing anything.
            - Request-rate telemetry is best-effort. If statistics are
              unavailable, `"Requests/Min"` falls back to `0.00`.
            - `Live.refresh()` is best-effort and failures are swallowed to avoid
              interfering with generation.
        """
        if not (self.live and self.group):
            return
        if novel_title is not None:
            self.status_text_novel_title.plain = f"Novel: {novel_title}"
        if chapter_num is not None:
            self.status_text_current_chapter.plain = f"Current Chapter: {chapter_num}"
        if step is not None:
            self.status_text_current_step.plain = f"Current Step: {step}"
        start_time = run_start_time or self.run_start_time
        elapsed_seconds = time.time() - start_time
        # Get request count from the refactored service statistics
        try:
            stats = llm_service.get_combined_statistics()
            request_count = stats.get("completion_service", {}).get("completions_requested", 0)
            requests_per_minute = request_count / (elapsed_seconds / 60) if elapsed_seconds > 0 else 0.0
        except (AttributeError, KeyError):
            requests_per_minute = 0.0
        self.status_text_requests_per_minute.plain = f"Requests/Min: {requests_per_minute:.2f}"
        self.status_text_elapsed_time.plain = f"Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_seconds))}"
        # Force a refresh to keep the live panel anchored and updated
        try:
            self.live.refresh()
        except Exception as e:
            logger.debug(
                "Failed to refresh Rich live display",
                error=str(e),
            )
