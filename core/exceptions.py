# core/exceptions.py
"""Define standardized exception types for SAGA core.

This module provides a small exception hierarchy and helpers used across `core/`
to propagate actionable error details without losing the original exception.
"""

from typing import Any


class SAGACoreError(Exception):
    """Base exception for all SAGA core system errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class DatabaseError(SAGACoreError):
    """Errors related to database operations."""


class DatabaseConnectionError(DatabaseError):
    """Errors related to database connection issues."""


class DatabaseTransactionError(DatabaseError):
    """Errors related to database transaction handling."""


class KnowledgeGraphPersistenceError(DatabaseError):
    """Signal a failure to persist knowledge graph changes.

    Notes:
        CORE-007: Persistence boundaries should not swallow exceptions or rely on
        ambiguous sentinel values. This exception exists so workflows can catch and
        handle persistence failures explicitly.
    """


class ValidationError(SAGACoreError):
    """Errors related to data validation."""


class LLMServiceError(SAGACoreError):
    """Errors related to LLM service operations."""


class CheckpointResumeConflictError(SAGACoreError):
    """Raised when checkpointed state conflicts with persisted artifacts during resume.

    Policy:
        - Checkpoint state is the single source of truth.
        - Neo4j/filesystem artifacts are treated as persisted artifacts.
        - Any detected mismatch must fail fast with a clear, stable message.
    """


def create_error_context(**kwargs: Any) -> dict[str, Any]:
    """Build a context dictionary for structured errors.

    Args:
        **kwargs: Key-value pairs to include.

    Returns:
        A dictionary containing only keys whose values are not `None`.
    """
    return {k: v for k, v in kwargs.items() if v is not None}


def handle_database_error(operation: str, original_error: Exception, **context: Any) -> DatabaseError:
    """Convert an exception into a standardized database error.

    Args:
        operation: Name/description of the database operation that failed.
        original_error: The caught exception.
        **context: Additional structured context to attach.

    Returns:
        A `DatabaseError` subclass chosen by heuristics over the original error text.
    """
    error_details = create_error_context(
        operation=operation,
        original_error=str(original_error),
        error_type=type(original_error).__name__,
        **context,
    )

    if "connection" in str(original_error).lower():
        return DatabaseConnectionError(f"Database connection failed during {operation}", details=error_details)
    elif "transaction" in str(original_error).lower():
        return DatabaseTransactionError(f"Database transaction failed during {operation}", details=error_details)
    else:
        return DatabaseError(f"Database error during {operation}", details=error_details)
