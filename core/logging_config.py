# core/logging_config.py
import logging as stdlib_logging
import logging.handlers
import os

import structlog

import config
from config import simple_formatter

try:
    from rich.logging import RichHandler
    from ui.rich_display import RichDisplayManager

    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False

    class RichHandler(stdlib_logging.Handler):
        def emit(self, record: stdlib_logging.LogRecord) -> None:
            stdlib_logging.getLogger(__name__).handle(record)


def setup_saga_logging():
    """
    Setup SAGA logging infrastructure with Rich console output and file rotation.

    Extracted from NANA orchestrator, now available for all orchestrators.
    Supports simple logging mode, file rotation, and Rich console output.
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
        stream_handler.setFormatter(simple_formatter)
        root_logger.addHandler(stream_handler)
        root_logger.info("Simple logging mode enabled: console only.")
    elif config.LOG_FILE:
        try:
            log_path = os.path.join(
                config.settings.BASE_OUTPUT_DIR, config.settings.LOG_FILE
            )
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

    if (
        not getattr(config, "SIMPLE_LOGGING_MODE", False)
        and RICH_AVAILABLE
        and config.ENABLE_RICH_PROGRESS
    ):
        existing_console = None
        if root_logger.handlers:
            for h_idx, h in enumerate(root_logger.handlers):
                if hasattr(h, "console") and not isinstance(
                    h, stdlib_logging.FileHandler
                ):
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
            show_time=True,
            show_level=True,
            console=existing_console,
        )
        root_logger.addHandler(rich_handler)
        root_logger.info("Rich logging handler enabled for console.")
    elif not any(
        isinstance(h, stdlib_logging.StreamHandler) for h in root_logger.handlers
    ):
        stream_handler = stdlib_logging.StreamHandler()
        stream_handler.setLevel(config.LOG_LEVEL_STR)
        stream_handler.setFormatter(simple_formatter)
        root_logger.addHandler(stream_handler)
        root_logger.info("Standard stream logging handler enabled for console.")

    stdlib_logging.getLogger("neo4j.notifications").setLevel(stdlib_logging.WARNING)
    stdlib_logging.getLogger("neo4j").setLevel(stdlib_logging.WARNING)
    stdlib_logging.getLogger("httpx").setLevel(stdlib_logging.WARNING)
    stdlib_logging.getLogger("httpcore").setLevel(stdlib_logging.WARNING)

    structlog.get_logger().info(
        f"SAGA Logging setup complete. Application Log Level: {stdlib_logging.getLevelName(config.LOG_LEVEL_STR)}."
    )
