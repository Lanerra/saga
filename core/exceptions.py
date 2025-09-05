# core/exceptions.py
"""
Standardized exception types for the SAGA core system.

This module defines consistent exception hierarchies and error handling patterns
for improved debugging and error recovery throughout the core modules.
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


class ValidationError(SAGACoreError):
    """Errors related to data validation."""


class SchemaValidationError(ValidationError):
    """Errors related to schema validation."""


class RelationshipValidationError(ValidationError):
    """Errors related to relationship validation."""


class LLMServiceError(SAGACoreError):
    """Errors related to LLM service operations."""


class LLMConnectionError(LLMServiceError):
    """Errors related to LLM service connectivity."""


class LLMResponseError(LLMServiceError):
    """Errors related to LLM response processing."""


class ConfigurationError(SAGACoreError):
    """Errors related to system configuration."""


class SecurityError(SAGACoreError):
    """Errors related to security violations."""


def create_error_context(**kwargs) -> dict[str, Any]:
    """
    Helper function to create standardized error context dictionaries.

    Args:
        **kwargs: Key-value pairs to include in the error context

    Returns:
        Dictionary with error context information
    """
    return {k: v for k, v in kwargs.items() if v is not None}


def handle_database_error(
    operation: str, original_error: Exception, **context
) -> DatabaseError:
    """
    Convert generic exceptions to standardized database errors.

    Args:
        operation: Description of the database operation that failed
        original_error: The original exception that was caught
        **context: Additional context information

    Returns:
        Appropriate DatabaseError subclass
    """
    error_details = create_error_context(
        operation=operation,
        original_error=str(original_error),
        error_type=type(original_error).__name__,
        **context,
    )

    if "connection" in str(original_error).lower():
        return DatabaseConnectionError(
            f"Database connection failed during {operation}", details=error_details
        )
    elif "transaction" in str(original_error).lower():
        return DatabaseTransactionError(
            f"Database transaction failed during {operation}", details=error_details
        )
    else:
        return DatabaseError(
            f"Database error during {operation}", details=error_details
        )


def handle_llm_error(
    operation: str, original_error: Exception, **context
) -> LLMServiceError:
    """
    Convert generic exceptions to standardized LLM service errors.

    Args:
        operation: Description of the LLM operation that failed
        original_error: The original exception that was caught
        **context: Additional context information

    Returns:
        Appropriate LLMServiceError subclass
    """
    error_details = create_error_context(
        operation=operation,
        original_error=str(original_error),
        error_type=type(original_error).__name__,
        **context,
    )

    if any(
        keyword in str(original_error).lower()
        for keyword in ["connection", "timeout", "network"]
    ):
        return LLMConnectionError(
            f"LLM service connection failed during {operation}", details=error_details
        )
    elif "response" in str(original_error).lower():
        return LLMResponseError(
            f"LLM response processing failed during {operation}", details=error_details
        )
    else:
        return LLMServiceError(
            f"LLM service error during {operation}", details=error_details
        )
