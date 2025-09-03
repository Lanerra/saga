# core/type_inference_service.py
"""
Type inference service for knowledge graph entities.

This module provides centralized type inference capabilities, extracting the logic
from the relationship validator to reduce coupling and improve maintainability.
"""

from typing import Any

import structlog
import models.kg_constants

logger = structlog.get_logger(__name__)


class TypeInferenceService:
    """
    Service for inferring and validating entity types in the knowledge graph.
    
    This service centralizes type inference logic that was previously scattered
    across validation components, improving maintainability and testability.
    """

    def __init__(self):
        self._inference_stats = {
            "total_inferences": 0,
            "successful_inferences": 0,
            "fallbacks_to_entity": 0,
            "inference_errors": 0
        }

    def infer_subject_type(self, subject_info: dict[str, Any]) -> str:
        """
        Infer the type of a subject entity from available information.

        Args:
            subject_info: Dictionary containing subject information with keys:
                - name: Entity name
                - type: Current/proposed type (may be None or invalid)
                - category: Optional category information

        Returns:
            Inferred valid node type
        """
        self._inference_stats["total_inferences"] += 1
        
        original_type = subject_info.get("type")
        subject_name = subject_info.get("name", "UNKNOWN")
        
        # If type is missing or None, try to infer it
        if original_type is None or original_type == "":
            logger.warning(f"Subject type missing or None for subject '{subject_name}'. Attempting inference.")
            return self._infer_type_from_context(subject_info)
        
        # If type exists but is not in valid labels, try to improve it
        if original_type not in models.kg_constants.NODE_LABELS:
            logger.warning(f"Subject type '{original_type}' not in valid node labels. Attempting inference.")
            return self._improve_invalid_type(original_type, subject_info)
        
        # Type is valid, return as-is
        self._inference_stats["successful_inferences"] += 1
        return original_type

    def infer_object_type(self, object_info: dict[str, Any], is_literal: bool = False) -> str:
        """
        Infer the type of an object entity from available information.

        Args:
            object_info: Dictionary containing object information
            is_literal: Whether this is a literal value (becomes ValueNode)

        Returns:
            Inferred valid node type
        """
        self._inference_stats["total_inferences"] += 1
        
        # Handle literal objects
        if is_literal:
            return "ValueNode"
        
        if not object_info:
            self._inference_stats["fallbacks_to_entity"] += 1
            return "Entity"
            
        original_type = object_info.get("type")
        object_name = object_info.get("name", "UNKNOWN")
        
        # If type is missing or None, try to infer it
        if original_type is None or original_type == "":
            logger.warning(f"Object type missing or None for object '{object_name}'. Attempting inference.")
            return self._infer_type_from_context(object_info)
        
        # If type exists but is not in valid labels, try to improve it
        if original_type not in models.kg_constants.NODE_LABELS:
            logger.warning(f"Object type '{original_type}' not in valid node labels. Attempting inference.")
            return self._improve_invalid_type(original_type, object_info)
        
        # Type is valid, return as-is
        self._inference_stats["successful_inferences"] += 1
        return original_type

    def _infer_type_from_context(self, entity_info: dict[str, Any]) -> str:
        """
        Infer entity type using enhanced node taxonomy when available.

        Args:
            entity_info: Entity information dictionary

        Returns:
            Inferred type or 'Entity' fallback
        """
        try:
            from core.enhanced_node_taxonomy import (
                validate_node_type,
                suggest_better_node_type,
                infer_node_type_from_name,
                infer_node_type_from_category
            )
            
            entity_name = entity_info.get("name", "")
            category = entity_info.get("category", "")
            
            # Try name-based inference first
            if entity_name:
                inferred_type = infer_node_type_from_name(entity_name)
                if validate_node_type(inferred_type):
                    logger.info(f"Inferred type from name '{entity_name}': {inferred_type}")
                    self._inference_stats["successful_inferences"] += 1
                    return inferred_type
            
            # Try category-based inference
            if category:
                inferred_type = infer_node_type_from_category(category)
                if validate_node_type(inferred_type):
                    logger.info(f"Inferred type from category '{category}': {inferred_type}")
                    self._inference_stats["successful_inferences"] += 1
                    return inferred_type
                    
        except ImportError:
            logger.debug("Enhanced node taxonomy not available, using fallback")
        except Exception as e:
            logger.warning(f"Type inference failed for entity '{entity_info.get('name', 'UNKNOWN')}': {e}")
            self._inference_stats["inference_errors"] += 1
        
        # Fallback to Entity
        self._inference_stats["fallbacks_to_entity"] += 1
        return "Entity"

    def _improve_invalid_type(self, invalid_type: str, entity_info: dict[str, Any]) -> str:
        """
        Attempt to improve an invalid type using various strategies.

        Args:
            invalid_type: The invalid/unrecognized type
            entity_info: Entity information for context

        Returns:
            Improved type or fallback
        """
        try:
            from core.enhanced_node_taxonomy import (
                validate_node_type,
                suggest_better_node_type,
            )
            
            entity_name = entity_info.get("name", "")
            category = entity_info.get("category", "")
            
            # Try to get a better type suggestion
            improved_type = suggest_better_node_type(invalid_type, entity_name, category)
            if validate_node_type(improved_type):
                logger.info(f"Improved invalid type '{invalid_type}' -> '{improved_type}' for '{entity_name}'")
                self._inference_stats["successful_inferences"] += 1
                return improved_type
                
        except ImportError:
            logger.debug("Enhanced node taxonomy not available for type improvement")
        except Exception as e:
            logger.warning(f"Type improvement failed for '{invalid_type}': {e}")
            self._inference_stats["inference_errors"] += 1
        
        # Try common type mappings for known problematic types
        improved_type = self._apply_common_type_mappings(invalid_type)
        if improved_type != invalid_type:
            logger.info(f"Applied mapping: '{invalid_type}' -> '{improved_type}'")
            self._inference_stats["successful_inferences"] += 1
            return improved_type
        
        # Final fallback to Entity
        entity_name = entity_info.get("name", "UNKNOWN")
        logger.warning(f"Fallback to 'Entity' for entity '{entity_name}' with invalid type '{invalid_type}'")
        self._inference_stats["fallbacks_to_entity"] += 1
        return "Entity"

    def _apply_common_type_mappings(self, invalid_type: str) -> str:
        """
        Apply common type mappings for known problematic types.

        Args:
            invalid_type: The problematic type string

        Returns:
            Mapped type or original if no mapping found
        """
        type_lower = invalid_type.lower()
        
        # Common mappings for frequently seen invalid types
        if "literal" in type_lower or "value" in type_lower:
            return "ValueNode"
        elif "response" in type_lower:
            return "ValueNode"
        elif "person" in type_lower or "human" in type_lower:
            return "Character"
        elif "place" in type_lower or "location" in type_lower:
            return "Location"
        elif "item" in type_lower or "object" in type_lower:
            return "Object"
        elif "group" in type_lower or "organization" in type_lower:
            return "Faction"
        
        return invalid_type  # No mapping found

    def get_inference_statistics(self) -> dict[str, Any]:
        """Get statistics about type inference operations."""
        total = self._inference_stats["total_inferences"]
        
        if total == 0:
            return self._inference_stats
            
        return {
            **self._inference_stats,
            "success_rate": self._inference_stats["successful_inferences"] / total * 100,
            "fallback_rate": self._inference_stats["fallbacks_to_entity"] / total * 100,
            "error_rate": self._inference_stats["inference_errors"] / total * 100
        }

    def reset_statistics(self):
        """Reset inference statistics (useful for testing)."""
        self._inference_stats = {
            "total_inferences": 0,
            "successful_inferences": 0,
            "fallbacks_to_entity": 0,
            "inference_errors": 0
        }


# Default service instance for backward compatibility
# This will be replaced by dependency injection in the next step
_default_service = TypeInferenceService()


def infer_subject_type(subject_info: dict[str, Any]) -> str:
    """Convenience function using default service instance."""
    return _default_service.infer_subject_type(subject_info)


def infer_object_type(object_info: dict[str, Any], is_literal: bool = False) -> str:
    """Convenience function using default service instance."""
    return _default_service.infer_object_type(object_info, is_literal)


def get_type_inference_stats() -> dict[str, Any]:
    """Get current type inference statistics."""
    return _default_service.get_inference_statistics()