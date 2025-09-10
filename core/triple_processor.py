# core/triple_processor.py
"""
Triple processing service for knowledge graph validation.

This module handles the complex logic of processing and validating triples,
separating concerns from the main validation logic to improve maintainability.

UPDATED: Now uses unified dependency injection system for better testability
and consistent service management across SAGA.
"""

from typing import Any, Optional, Dict

import structlog

# Import the new DI system
from core.service_registry import resolve, register_singleton
from core.validation_service_provider import TypeInferenceServiceInterface

logger = structlog.get_logger(__name__)


class TripleProcessor:
    """
    Service for processing and preparing triples for validation.

    This class handles the complex logic of extracting and preparing entity
    information from triple dictionaries, separating this concern from the
    core validation logic.
    
    UPDATED: Now uses unified service registry for dependency injection
    instead of the previous mixed approach.
    """

    def __init__(self, type_inference_service: Optional[TypeInferenceServiceInterface] = None):
        """
        Initialize the triple processor.

        Args:
            type_inference_service: Service for inferring entity types.
                                  If None, will use service registry for dependency injection.
        """
        self._type_inference_service = type_inference_service
        self._service_registry_used = False
        self._processing_stats = {
            "total_triples_processed": 0,
            "successful_extractions": 0,
            "extraction_errors": 0,
            "type_inferences": 0,
            "service_registry_resolutions": 0,
            "fallback_resolutions": 0,
        }

    def process_triple(self, triple_dict: dict[str, Any]) -> dict[str, Any] | None:
        """
        Process a triple dictionary and extract validated entity information.

        Args:
            triple_dict: Raw triple dictionary from extraction pipeline

        Returns:
            Processed triple information ready for validation, or None if invalid

        Expected triple_dict format:
        {
            "subject": {"name": "...", "type": "...", "category": "..."},
            "predicate": "...",
            "object_entity": {"name": "...", "type": "...", "category": "..."} | None,
            "object_literal": "..." | None,
            "is_literal_object": bool
        }
        """
        self._processing_stats["total_triples_processed"] += 1

        try:
            # Extract and validate subject information
            subject_info = self._extract_subject_info(triple_dict)
            if not subject_info:
                return None

            # Extract predicate
            predicate = self._extract_predicate(triple_dict)
            if not predicate:
                return None

            # Extract and validate object information
            object_info = self._extract_object_info(triple_dict)
            if not object_info:
                return None

            processed_triple = {
                "subject_type": subject_info["type"],
                "subject_name": subject_info["name"],
                "predicate": predicate,
                "object_type": object_info["type"],
                "object_name": object_info.get("name"),
                "is_literal_object": object_info.get("is_literal", False),
                "original_triple": triple_dict,
            }

            self._processing_stats["successful_extractions"] += 1
            return processed_triple

        except Exception as e:
            logger.error(f"Error processing triple {triple_dict}: {e}", exc_info=True)
            self._processing_stats["extraction_errors"] += 1
            return None

    def _extract_subject_info(
        self, triple_dict: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract and validate subject information from triple."""
        subject_info = triple_dict.get("subject")
        if not isinstance(subject_info, dict):
            logger.warning(f"Invalid subject info in triple: {triple_dict}")
            return None

        subject_name = subject_info.get("name")
        if not subject_name or not str(subject_name).strip():
            logger.warning(f"Missing or empty subject name in triple: {triple_dict}")
            return None

        # Use type inference service to get proper subject type
        subject_type = self._get_type_inference_service().infer_subject_type(
            subject_info
        )
        self._processing_stats["type_inferences"] += 1

        return {
            "name": str(subject_name).strip(),
            "type": subject_type,
            "original_info": subject_info,
        }

    def _extract_predicate(self, triple_dict: dict[str, Any]) -> str | None:
        """Extract and validate predicate from triple."""
        predicate = triple_dict.get("predicate")
        if not predicate or not str(predicate).strip():
            logger.warning(f"Missing or empty predicate in triple: {triple_dict}")
            return None

        return str(predicate).strip()

    def _extract_object_info(
        self, triple_dict: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract and validate object information from triple."""
        is_literal_object = triple_dict.get("is_literal_object", False)

        if is_literal_object:
            # Handle literal objects
            literal_value = triple_dict.get("object_literal")
            if literal_value is None:
                logger.warning(
                    f"Missing literal value in literal triple: {triple_dict}"
                )
                return None

            return {
                "type": "ValueNode",
                "name": str(literal_value),
                "is_literal": True,
                "original_value": literal_value,
            }
        else:
            # Handle entity objects
            object_entity_info = triple_dict.get("object_entity")
            if not isinstance(object_entity_info, dict):
                logger.warning(f"Invalid object entity info in triple: {triple_dict}")
                return None

            object_name = object_entity_info.get("name")
            if not object_name or not str(object_name).strip():
                logger.warning(f"Missing or empty object name in triple: {triple_dict}")
                return None

            # Use type inference service to get proper object type
            object_type = self._get_type_inference_service().infer_object_type(
                object_entity_info, is_literal=False
            )
            self._processing_stats["type_inferences"] += 1

            return {
                "type": object_type,
                "name": str(object_name).strip(),
                "is_literal": False,
                "original_info": object_entity_info,
            }

    def _get_type_inference_service(self) -> TypeInferenceServiceInterface:
        """
        Get type inference service via unified dependency injection.
        
        Uses the new service registry first, with fallback to the existing
        validation service provider for backward compatibility.
        """
        if self._type_inference_service is None:
            # Try unified service registry first
            try:
                self._type_inference_service = resolve("type_inference_service")
                self._service_registry_used = True
                self._processing_stats["service_registry_resolutions"] += 1
                logger.debug("Type inference service resolved via service registry")
                
            except (ValueError, Exception) as registry_error:
                # Fallback to existing validation service provider for backward compatibility
                logger.debug(f"Service registry resolution failed ({registry_error}), using fallback")
                try:
                    from core.validation_service_provider import get_type_inference_service

                    self._type_inference_service = get_type_inference_service()
                    self._processing_stats["fallback_resolutions"] += 1
                    logger.debug("Type inference service resolved via validation service provider")
                    
                except Exception as fallback_error:
                    logger.error(f"Both service registry and fallback failed: registry={registry_error}, fallback={fallback_error}")
                    raise RuntimeError(
                        f"Unable to initialize type inference service. "
                        f"Service registry error: {registry_error}. "
                        f"Fallback error: {fallback_error}. "
                        "Ensure type inference service is properly registered or configured."
                    ) from fallback_error

        return self._type_inference_service

    def process_batch(self, triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Process a batch of triples efficiently.

        Args:
            triples: List of triple dictionaries

        Returns:
            List of processed triple information (excluding invalid triples)
        """
        processed_triples = []

        for triple_dict in triples:
            processed = self.process_triple(triple_dict)
            if processed:
                processed_triples.append(processed)

        return processed_triples

    def get_processing_statistics(self) -> dict[str, Any]:
        """Get statistics about triple processing operations."""
        total = self._processing_stats["total_triples_processed"]

        if total == 0:
            return self._processing_stats

        return {
            **self._processing_stats,
            "success_rate": self._processing_stats["successful_extractions"]
            / total
            * 100,
            "error_rate": self._processing_stats["extraction_errors"] / total * 100,
            "avg_inferences_per_triple": self._processing_stats["type_inferences"]
            / total
            if total > 0
            else 0,
        }

    def reset_statistics(self):
        """Reset processing statistics (useful for testing)."""
        self._processing_stats = {
            "total_triples_processed": 0,
            "successful_extractions": 0,
            "extraction_errors": 0,
            "type_inferences": 0,
            "service_registry_resolutions": 0,
            "fallback_resolutions": 0,
        }
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get service information for monitoring (ServiceInterface compliance).
        """
        return {
            "service_name": "TripleProcessor",
            "service_type": "triple_processing",
            "dependency_injection_method": "service_registry" if self._service_registry_used else "validation_provider",
            "type_inference_service_available": self._type_inference_service is not None,
            "type_inference_service_type": type(self._type_inference_service).__name__
                if self._type_inference_service else None,
            "processing_statistics": self._processing_stats,
            "supports_batch_processing": True,
            "supports_statistics_reset": True
        }
    
    def set_type_inference_service(self, service: TypeInferenceServiceInterface) -> None:
        """
        Set the type inference service explicitly (useful for testing).
        
        Args:
            service: Type inference service instance to use
        """
        self._type_inference_service = service
        self._service_registry_used = False
        logger.debug(f"Type inference service set explicitly: {type(service).__name__}")
    
    def dispose(self) -> None:
        """
        Dispose of the triple processor and clean up resources.
        
        Called by service lifecycle manager during shutdown.
        """
        self._type_inference_service = None
        self._service_registry_used = False
        logger.debug("Triple processor disposed")


# Service registration for dependency injection
def register_triple_processor_service():
    """Register the triple processor with the service registry."""
    register_singleton(
        name="triple_processor",
        factory=lambda: TripleProcessor(),
        dependencies=["type_inference_service"],
        interface=TripleProcessor
    )
    logger.info("Triple processor service registered")


def get_triple_processor() -> TripleProcessor:
    """
    Get a triple processor instance from the service registry.
    
    Returns:
        TripleProcessor instance
    """
    try:
        return resolve("triple_processor")
    except ValueError:
        # Fallback: create instance directly for backward compatibility
        logger.debug("Triple processor not in service registry, creating new instance")
        return TripleProcessor()


# Factory function for enhanced triple processor creation
def create_triple_processor_with_service(
    type_inference_service: TypeInferenceServiceInterface
) -> TripleProcessor:
    """
    Create a triple processor with a specific type inference service.
    
    Useful for testing and specialized configurations.
    
    Args:
        type_inference_service: Type inference service to use
        
    Returns:
        Configured TripleProcessor instance
    """
    processor = TripleProcessor(type_inference_service)
    logger.debug(f"Created triple processor with service: {type(type_inference_service).__name__}")
    return processor
