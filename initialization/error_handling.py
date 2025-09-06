# initialization/error_handling.py
"""Standardized error handling for bootstrap operations."""

from typing import Any, Optional
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for bootstrap errors."""
    WARNING = "warning"      # Non-blocking issue that allows continuation
    ERROR = "error"          # Blocking issue that prevents specific operation but allows overall continuation
    CRITICAL = "critical"    # Fatal error that should stop the entire bootstrap process


class BootstrapError(Exception):
    """Base exception for bootstrap operations."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.context = context or {}


class ValidationError(BootstrapError):
    """Raised when validation fails."""
    pass


class GenerationError(BootstrapError):
    """Raised when content generation fails."""
    pass


class DataLoadError(BootstrapError):
    """Raised when data loading fails."""
    pass


def handle_bootstrap_error(
    error: Exception,
    operation: str,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    context: dict[str, Any] | None = None,
    reraise: bool = False
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
        logger.error(error_msg, error_type=type(error).__name__, context=context, exc_info=True)
        if reraise:
            raise
        return False  # Abort this specific operation but continue overall
    elif severity == ErrorSeverity.CRITICAL:
        logger.critical(error_msg, error_type=type(error).__name__, context=context, exc_info=True)
        if reraise:
            raise
        return False  # Abort entire process


def safe_bootstrap_operation(
    operation_name: str,
    operation_func,
    *args,
    default_return=None,
    error_severity: ErrorSeverity = ErrorSeverity.ERROR,
    context: dict[str, Any] | None = None,
    **kwargs
) -> tuple[Any, bool]:
    """Safely execute a bootstrap operation with standardized error handling.
    
    Args:
        operation_name: Human-readable name for the operation
        operation_func: The function to execute
        *args: Arguments to pass to operation_func
        default_return: Value to return if operation fails
        error_severity: How to handle errors
        context: Additional context for error reporting
        **kwargs: Keyword arguments to pass to operation_func
        
    Returns:
        Tuple of (result, success_flag)
    """
    try:
        result = operation_func(*args, **kwargs)
        return result, True
    except Exception as e:
        should_continue = handle_bootstrap_error(
            e, operation_name, error_severity, context
        )
        return default_return, should_continue


async def safe_async_bootstrap_operation(
    operation_name: str,
    operation_func,
    *args,
    default_return=None,
    error_severity: ErrorSeverity = ErrorSeverity.ERROR,
    context: dict[str, Any] | None = None,
    **kwargs
) -> tuple[Any, bool]:
    """Async version of safe_bootstrap_operation."""
    try:
        result = await operation_func(*args, **kwargs)
        return result, True
    except Exception as e:
        should_continue = handle_bootstrap_error(
            e, operation_name, error_severity, context
        )
        return default_return, should_continue


def validate_bootstrap_preconditions(checks: dict[str, Any]) -> None:
    """Validate preconditions before starting bootstrap operations.
    
    Args:
        checks: Dictionary of check_name -> check_result pairs
        
    Raises:
        ValidationError: If any critical precondition fails
    """
    failed_checks = []
    for check_name, check_result in checks.items():
        if not check_result:
            failed_checks.append(check_name)
    
    if failed_checks:
        raise ValidationError(
            f"Bootstrap preconditions failed: {', '.join(failed_checks)}",
            ErrorSeverity.CRITICAL,
            {"failed_checks": failed_checks}
        )


def create_error_recovery_context(
    operation: str,
    attempt_number: int,
    max_attempts: int,
    error: Exception
) -> dict[str, Any]:
    """Create context for error recovery operations."""
    return {
        "operation": operation,
        "attempt": attempt_number,
        "max_attempts": max_attempts,
        "error_type": type(error).__name__,
        "error_message": str(error)
    }