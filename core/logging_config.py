# core/logging_config.py
"""Configure SAGA logging sinks and formatting.

This module configures:
- Standard library logging handlers (console and optional rotating file).
- Optional Rich console integration when available.
- Baseline log level overrides for noisy third-party libraries.

Notes:
    This module intentionally performs side-effectful logger configuration and should be
    called once at process startup via [`core.logging_config.setup_saga_logging()`](core/logging_config.py:36).
"""

import logging as stdlib_logging
import logging.handlers
import os

import structlog

import config
from config import rich_formatter, simple_formatter

try:
    from rich.logging import RichHandler

    from ui.rich_display import RichDisplayManager

    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False

    class RichHandler(stdlib_logging.Handler):  # type: ignore[no-redef]
        def emit(self, record: stdlib_logging.LogRecord) -> None:
            stdlib_logging.getLogger(__name__).handle(record)


def setup_saga_logging() -> None:
    """Set up SAGA logging handlers and formatting.

    This configures:
    - Console logging in simple mode.
    - Rotating file logging when a log file is configured.
    - Rich console output when Rich is available and enabled.

    Notes:
        This function mutates the root logger handler list and is intended to be called
        once during application startup.
    """
    stdlib_logging.basicConfig(
        level=config.LOG_LEVEL_STR,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
        handlers=[],
    )
    root_logger = stdlib_logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if getattr(config, "SIMPLE_LOGGING_MODE", False):
        stream_handler = stdlib_logging.StreamHandler()
        stream_handler.setLevel(config.LOG_LEVEL_STR)
        stream_handler.setFormatter(rich_formatter)  # Use rich formatter for better console output
        root_logger.addHandler(stream_handler)
        root_logger.info("Simple logging mode enabled: console only.")
    elif config.LOG_FILE:
        try:
            log_file = config.settings.LOG_FILE
            if log_file is None:
                raise ValueError("LOG_FILE is None but config.LOG_FILE was truthy")
            log_path = os.path.join(config.settings.BASE_OUTPUT_DIR, log_file)
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            file_handler = stdlib_logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                mode="a",
                encoding="utf-8",
            )
            file_handler.setLevel(config.LOG_LEVEL_STR)
            file_handler.setFormatter(simple_formatter)
            root_logger.addHandler(file_handler)
            root_logger.info(f"File logging enabled. Log file: {log_path}")
        except Exception as e:
            console_handler_fallback = stdlib_logging.StreamHandler()
            console_handler_fallback.setFormatter(simple_formatter)
            root_logger.addHandler(console_handler_fallback)
            root_logger.error(
                f"Failed to configure file logging: {e}. Logging to console instead.",
                exc_info=True,
            )

    if not getattr(config, "SIMPLE_LOGGING_MODE", False) and RICH_AVAILABLE and config.ENABLE_RICH_PROGRESS:
        existing_console = None
        if root_logger.handlers:
            for _h_idx, h in enumerate(root_logger.handlers):
                if hasattr(h, "console") and not isinstance(h, stdlib_logging.FileHandler):
                    existing_console = h.console
                    break

        if existing_console is None:
            try:
                existing_console = RichDisplayManager.get_shared_console()
            except Exception:
                existing_console = None

        rich_handler = RichHandler(
            level=config.LOG_LEVEL_STR,
            rich_tracebacks=True,
            show_path=False,
            markup=True,
            show_time=False,  # Timestamp already in our formatter
            show_level=False,  # Level already in our formatter
            console=existing_console,
        )
        rich_handler.setFormatter(rich_formatter)
        root_logger.addHandler(rich_handler)
        root_logger.info("Rich logging handler enabled for console.")
    elif not any(isinstance(h, stdlib_logging.StreamHandler) for h in root_logger.handlers):
        stream_handler = stdlib_logging.StreamHandler()
        stream_handler.setLevel(config.LOG_LEVEL_STR)
        stream_handler.setFormatter(rich_formatter)  # Use rich formatter for better console output
        root_logger.addHandler(stream_handler)
        root_logger.info("Standard stream logging handler enabled for console.")

    stdlib_logging.getLogger("neo4j.notifications").setLevel(stdlib_logging.WARNING)
    stdlib_logging.getLogger("neo4j").setLevel(stdlib_logging.WARNING)
    stdlib_logging.getLogger("httpx").setLevel(stdlib_logging.WARNING)
    stdlib_logging.getLogger("httpcore").setLevel(stdlib_logging.WARNING)

    structlog.get_logger().info(f"SAGA Logging setup complete. Application Log Level: {stdlib_logging.getLevelName(config.LOG_LEVEL_STR)}.")
