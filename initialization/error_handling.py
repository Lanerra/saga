# initialization/error_handling.py
"""Standardized error handling for bootstrap operations."""

from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for bootstrap errors."""

    WARNING = "warning"  # Non-blocking issue that allows continuation
    ERROR = "error"  # Blocking issue that prevents specific operation but allows overall continuation
    CRITICAL = "critical"  # Fatal error that should stop the entire bootstrap process


class BootstrapError(Exception):
    """Base exception for bootstrap operations."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.context = context or {}


"""
Note: Additional specialized exception types and helper wrappers previously lived
here but were unused in the current single-user pipeline and have been removed
to reduce indirection. Keep this module lean: use handle_bootstrap_error for
uniform logging and flow control.
"""


def handle_bootstrap_error(
    error: Exception,
    operation: str,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    context: dict[str, Any] | None = None,
    reraise: bool = False,
) -> bool:
    """Standardized error handling for bootstrap operations.

    Args:
        error: The exception that occurred
        operation: Description of the operation that failed
        severity: How severe the error is
        context: Additional context for debugging
        reraise: Whether to re-raise the error after logging

    Returns:
        True if operation should continue, False if it should be aborted
    """
    context = context or {}
    error_msg = f"{operation} failed: {error}"

    # Log based on severity
    if severity == ErrorSeverity.WARNING:
        logger.warning(error_msg, error_type=type(error).__name__, context=context)
        return True  # Continue operation
    elif severity == ErrorSeverity.ERROR:
        logger.error(
            error_msg, error_type=type(error).__name__, context=context, exc_info=True
        )
        if reraise:
            raise
        return False  # Abort this specific operation but continue overall
    elif severity == ErrorSeverity.CRITICAL:
        logger.critical(
            error_msg, error_type=type(error).__name__, context=context, exc_info=True
        )
        if reraise:
            raise
        return False  # Abort entire process
