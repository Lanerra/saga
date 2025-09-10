# core/validation_service_provider.py
"""
Service provider for relationship validation with dependency injection.

This module implements dependency injection for the validation system,
replacing the problematic global validator instance with a cleaner,
testable architecture.
"""

import threading
from contextlib import contextmanager
from typing import Any, Protocol

import structlog

logger = structlog.get_logger(__name__)


class ValidationServiceInterface(Protocol):
    """Protocol defining the interface for validation services."""

    def validate_relationship(
        self,
        subject_type: str,
        predicate: str,
        object_type: str,
        context: dict[str, Any] | None = None,
    ) -> Any:  # ValidationResult
        """Validate a relationship between two entity types."""
        ...

    def validate_triple(self, triple_dict: dict[str, Any]) -> Any:  # ValidationResult
        """Validate a complete triple from the extraction pipeline."""
        ...

    def validate_batch(
        self, triples: list[dict[str, Any]]
    ) -> list[Any]:  # list[ValidationResult]
        """Validate a batch of triples efficiently."""
        ...

    def get_validation_statistics(self) -> dict[str, Any]:
        """Get validation statistics for monitoring."""
        ...


class TypeInferenceServiceInterface(Protocol):
    """Protocol defining the interface for type inference services."""

    def infer_subject_type(self, subject_info: dict[str, Any]) -> str:
        """Infer the type of a subject entity."""
        ...

    def infer_object_type(
        self, object_info: dict[str, Any], is_literal: bool = False
    ) -> str:
        """Infer the type of an object entity."""
        ...


class ValidationServiceProvider:
    """
    Service provider for validation system with dependency injection.

    This provider manages the lifecycle and dependencies of validation services,
    replacing the problematic global singleton pattern with proper dependency
    injection.
    """

    def __init__(self):
        self._validation_service: ValidationServiceInterface | None = None
        self._type_inference_service: TypeInferenceServiceInterface | None = None
        self._lock = threading.RLock()
        self._context_stack: list[dict[str, Any]] = []

    def register_validation_service(self, service: ValidationServiceInterface) -> None:
        """Register a validation service implementation."""
        with self._lock:
            self._validation_service = service
            logger.info(f"Registered validation service: {type(service).__name__}")

    def register_type_inference_service(
        self, service: TypeInferenceServiceInterface
    ) -> None:
        """Register a type inference service implementation."""
        with self._lock:
            self._type_inference_service = service
            logger.info(f"Registered type inference service: {type(service).__name__}")

    def get_validation_service(self) -> ValidationServiceInterface:
        """
        Get the current validation service instance.

        Raises:
            RuntimeError: If no validation service is registered
        """
        with self._lock:
            if self._validation_service is None:
                # Lazy initialization with default implementation
                self._initialize_default_services()

            if self._validation_service is None:
                raise RuntimeError(
                    "No validation service registered. Call register_validation_service() first."
                )

            return self._validation_service

    def get_type_inference_service(self) -> TypeInferenceServiceInterface:
        """
        Get the current type inference service instance.

        Raises:
            RuntimeError: If no type inference service is registered
        """
        with self._lock:
            if self._type_inference_service is None:
                # Lazy initialization with default implementation
                self._initialize_default_services()

            if self._type_inference_service is None:
                raise RuntimeError(
                    "No type inference service registered. Call register_type_inference_service() first."
                )

            return self._type_inference_service

    def _initialize_default_services(self) -> None:
        """Initialize default service implementations."""
        try:
            # Import here to avoid circular dependencies
            if self._type_inference_service is None:
                # Use IntelligentTypeInference as the new default (ML-inspired system)
                from core.intelligent_type_inference import IntelligentTypeInference
                from core.schema_introspector import SchemaIntrospector

                # Create schema introspector dependency
                schema_introspector = SchemaIntrospector()
                
                # Initialize the ML-inspired type inference system
                self._type_inference_service = IntelligentTypeInference(schema_introspector)
                logger.info("Initialized default ML-inspired type inference service")

            if self._validation_service is None:
                # Import and initialize the refactored validator
                from core.relationship_validator import RelationshipConstraintValidator

                self._validation_service = RelationshipConstraintValidator()
                logger.info("Initialized default validation service")

        except ImportError as e:
            logger.error(f"Failed to initialize default services: {e}")
            # Re-raise the error since we no longer have fallback options
            raise RuntimeError(
                f"Unable to initialize required services: {e}. "
                "Ensure IntelligentTypeInference and RelationshipConstraintValidator are properly configured."
            ) from e

    @contextmanager
    def validation_context(self, **context_data):
        """
        Context manager for providing validation context.

        This allows passing contextual information (like chapter number,
        confidence thresholds, etc.) to validation operations within a scope.
        """
        with self._lock:
            self._context_stack.append(context_data)

        try:
            yield
        finally:
            with self._lock:
                if self._context_stack:
                    self._context_stack.pop()

    def get_current_context(self) -> dict[str, Any]:
        """Get the current validation context."""
        with self._lock:
            if self._context_stack:
                # Merge all contexts in the stack, with later ones taking precedence
                merged_context = {}
                for context in self._context_stack:
                    merged_context.update(context)
                return merged_context
            return {}

    def reset_services(self) -> None:
        """Reset all registered services (useful for testing)."""
        with self._lock:
            self._validation_service = None
            self._type_inference_service = None
            self._context_stack.clear()
            logger.info("Reset all validation services")

    def get_service_status(self) -> dict[str, Any]:
        """Get the status of registered services."""
        with self._lock:
            return {
                "validation_service_registered": self._validation_service is not None,
                "validation_service_type": type(self._validation_service).__name__
                if self._validation_service
                else None,
                "type_inference_service_registered": self._type_inference_service
                is not None,
                "type_inference_service_type": type(
                    self._type_inference_service
                ).__name__
                if self._type_inference_service
                else None,
                "context_stack_depth": len(self._context_stack),
                "current_context": self.get_current_context(),
            }


# Thread-local storage for service providers to avoid global state issues
_local = threading.local()


def get_service_provider() -> ValidationServiceProvider:
    """
    Get the current thread's validation service provider.

    This function ensures thread safety by using thread-local storage
    instead of a global singleton.
    """
    if not hasattr(_local, "service_provider"):
        _local.service_provider = ValidationServiceProvider()

    return _local.service_provider


def set_service_provider(provider: ValidationServiceProvider) -> None:
    """Set the service provider for the current thread."""
    _local.service_provider = provider


# Convenience functions that delegate to the current service provider
def get_validation_service() -> ValidationServiceInterface:
    """Get the current validation service instance."""
    return get_service_provider().get_validation_service()


def get_type_inference_service() -> TypeInferenceServiceInterface:
    """Get the current type inference service instance."""
    return get_service_provider().get_type_inference_service()


@contextmanager
def validation_context(**context_data):
    """Context manager for validation operations."""
    with get_service_provider().validation_context(**context_data):
        yield


def register_validation_service(service: ValidationServiceInterface) -> None:
    """Register a validation service for the current thread."""
    get_service_provider().register_validation_service(service)


def register_type_inference_service(service: TypeInferenceServiceInterface) -> None:
    """Register a type inference service for the current thread."""
    get_service_provider().register_type_inference_service(service)
