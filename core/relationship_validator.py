# core/relationship_validator.py
"""
Core relationship validation framework for the SAGA knowledge graph.

This module provides the main validation engine that integrates with the existing
pipeline to validate relationship semantics before they are stored in the database.

REFACTORED: This module has been refactored as part of Phase 3 architectural improvements.
- Type inference logic extracted to core.type_inference_service
- Complex triple processing extracted to core.triple_processor  
- Global validator instance replaced with dependency injection via core.validation_service_provider
"""

from typing import Any

import structlog
import models.kg_constants

import config
from core.relationship_constraints import (
    get_all_valid_relationships_for_node_pair,
    get_relationship_suggestions,
    validate_relationship_semantics,
)

try:  # pragma: no cover - fallback when data access layer unavailable
    from data_access.kg_queries import validate_relationship_type
except Exception:  # pragma: no cover
    def validate_relationship_type(proposed_type: str) -> str:
        """Fallback normalizer when data layer is unavailable."""
        return proposed_type.upper()

logger = structlog.get_logger(__name__)


def validate_relationship_types(rel_types: list[str]) -> list[str]:
    """
    DEPRECATED: This function is deprecated and should not be used in new code.
    
    Validate relationship types against the predefined narrative taxonomy.
    
    MIGRATION: Use the new validation service provider with proper dependency injection:
        from core.validation_service_provider import get_validation_service
        validation_service = get_validation_service()
        # Use validation_service methods instead

    Returns a list of validation errors. Empty list means valid.
    """
    errors = []

    if not isinstance(rel_types, list):
        errors.append("Relationship types must be a list")
        return errors

    for rel_type in rel_types:
        if not isinstance(rel_type, str) or not rel_type.strip():
            errors.append("Relationship types must be non-empty strings")
            continue

        # Basic format validation
        if not rel_type.isupper():
            errors.append(f"Relationship type '{rel_type}' should be uppercase")

        # Check against predefined taxonomy
        if rel_type not in models.kg_constants.RELATIONSHIP_TYPES:
            # Check if it can be normalized to a valid type
            from data_access.kg_queries import normalize_relationship_type

            normalized = normalize_relationship_type(rel_type)

            if normalized in models.kg_constants.RELATIONSHIP_TYPES:
                # It's normalizable - suggest normalization rather than error
                logger.info(
                    f"Relationship type '{rel_type}' can be normalized to '{normalized}'"
                )
            else:
                errors.append(
                    f"Relationship type '{rel_type}' is not in the predefined narrative taxonomy"
                )

    return errors


def suggest_relationship_normalization(rel_types: list[str]) -> dict[str, str]:
    """
    DEPRECATED: This function is deprecated and should not be used in new code.
    
    Suggest normalizations for relationship types that don't match the predefined taxonomy.
    
    MIGRATION: Use the new validation service provider with proper dependency injection:
        from core.validation_service_provider import get_validation_service
        validation_service = get_validation_service()
        # Use validation_service.suggest_alternatives() instead

    Returns a dict mapping original -> suggested canonical form.
    """
    suggestions = {}

    for rel_type in rel_types:
        if isinstance(rel_type, str) and rel_type.strip():
            if rel_type not in models.kg_constants.RELATIONSHIP_TYPES:
                from data_access.kg_queries import normalize_relationship_type
                normalized = normalize_relationship_type(rel_type)
                if (
                    normalized in models.kg_constants.RELATIONSHIP_TYPES
                    and normalized != rel_type
                ):
                    suggestions[rel_type] = normalized

    return suggestions


class ValidationResult:
    """Result of relationship validation with detailed feedback."""

    def __init__(
        self,
        is_valid: bool,
        original_relationship: str,
        validated_relationship: str | None = None,
        errors: list[str] | None = None,
        suggestions: list[tuple[str, str]] | None = None,
        confidence: float = 1.0,
    ):
        self.is_valid = is_valid
        self.original_relationship = original_relationship
        self.validated_relationship = validated_relationship or original_relationship
        self.errors = errors or []
        self.suggestions = suggestions or []
        self.confidence = confidence

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return f"ValidationResult({status}: {self.original_relationship} -> {self.validated_relationship})"


class RelationshipConstraintValidator:
    """
    Main validation engine for relationship constraints.
    
    REFACTORED: Simplified to focus only on core relationship validation logic.
    Complex type inference and triple processing have been extracted to separate services.
    """

    def __init__(self, triple_processor=None) -> None:
        self.validation_stats = {
            "total_validations": 0,
            "valid_relationships": 0,
            "invalid_relationships": 0,
            "corrected_relationships": 0,
            "fallback_relationships": 0,
        }
        # Dependency injection for triple processor
        self._triple_processor = triple_processor

    def validate_relationship(
        self,
        subject_type: str,
        predicate: str,
        object_type: str,
        context: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """
        Comprehensive relationship validation with correction attempts.

        Args:
            subject_type: Node type of the relationship subject
            predicate: Relationship type/predicate
            object_type: Node type of the relationship object
            context: Optional context for validation (e.g., chapter info, confidence scores)

        Returns:
            ValidationResult with validation outcome and suggestions
        """
        self.validation_stats["total_validations"] += 1

        # Early exit if semantic flattening is disabled - preserve original relationship
        if config.settings.DISABLE_RELATIONSHIP_SEMANTIC_FLATTENING:
            # Simply return the original predicate without any validation or correction
            return ValidationResult(
                is_valid=True,
                original_relationship=predicate,
                validated_relationship=predicate,
                errors=[],
                confidence=1.0,
            )

        # Step 1: Normalize the relationship type using existing logic
        normalized_predicate = validate_relationship_type(predicate)

        # Step 2: Validate semantic compatibility
        is_semantically_valid, errors = validate_relationship_semantics(
            subject_type, normalized_predicate, object_type
        )

        if is_semantically_valid:
            self.validation_stats["valid_relationships"] += 1
            if normalized_predicate != predicate:
                self.validation_stats["corrected_relationships"] += 1

            return ValidationResult(
                is_valid=True,
                original_relationship=predicate,
                validated_relationship=normalized_predicate,
                errors=[],
                confidence=1.0 if normalized_predicate == predicate else 0.9,
            )

        # Step 3: Try to find alternative valid relationships
        suggestions = get_relationship_suggestions(subject_type, object_type)

        # Step 4: Attempt automatic correction based on semantic similarity (if enabled)
        if config.settings.RELATIONSHIP_CONSTRAINT_AUTO_CORRECT:
            best_alternative = self._find_best_semantic_match(
                predicate, subject_type, object_type
            )

            if best_alternative:
                self.validation_stats["corrected_relationships"] += 1
                if config.settings.RELATIONSHIP_CONSTRAINT_LOG_VIOLATIONS:
                    logger.info(
                        f"Auto-corrected invalid relationship: "
                        f"{subject_type} {predicate} {object_type} -> "
                        f"{subject_type} {best_alternative} {object_type}"
                    )

                return ValidationResult(
                    is_valid=True,
                    original_relationship=predicate,
                    validated_relationship=best_alternative,
                    errors=[
                        f"Original relationship was invalid, corrected to {best_alternative}"
                    ],
                    suggestions=suggestions,
                    confidence=0.7,  # Lower confidence for auto-corrections
                )

        # Step 5: Fall back to RELATES_TO if enabled and no better option found
        if not config.settings.RELATIONSHIP_CONSTRAINT_STRICT_MODE and "RELATES_TO" in [
            rel for rel, _ in suggestions
        ]:
            self.validation_stats["fallback_relationships"] += 1
            if config.settings.RELATIONSHIP_CONSTRAINT_LOG_VIOLATIONS:
                logger.warning(
                    f"Using RELATES_TO fallback for invalid relationship: "
                    f"{subject_type} {predicate} {object_type}"
                )

            return ValidationResult(
                is_valid=True,  # We're allowing it with fallback
                original_relationship=predicate,
                validated_relationship="RELATES_TO",
                errors=errors
                + ["Used RELATES_TO fallback due to invalid relationship"],
                suggestions=suggestions,
                confidence=0.3,  # Very low confidence for fallbacks
            )

        # Step 6: Complete rejection (or fallback to ASSOCIATED_WITH in permissive mode)
        if (
            not config.settings.RELATIONSHIP_CONSTRAINT_STRICT_MODE
            and "ASSOCIATED_WITH" in [rel for rel, _ in suggestions]
        ):
            self.validation_stats["fallback_relationships"] += 1
            if config.settings.RELATIONSHIP_CONSTRAINT_LOG_VIOLATIONS:
                logger.warning(
                    f"Using ASSOCIATED_WITH fallback for invalid relationship: "
                    f"{subject_type} {predicate} {object_type}"
                )

            return ValidationResult(
                is_valid=True,  # Ultimate fallback
                original_relationship=predicate,
                validated_relationship="ASSOCIATED_WITH",
                errors=errors
                + [
                    "Used ASSOCIATED_WITH ultimate fallback due to invalid relationship"
                ],
                suggestions=suggestions,
                confidence=0.1,  # Very low confidence for ultimate fallbacks
            )

        # Complete rejection
        self.validation_stats["invalid_relationships"] += 1
        if config.settings.RELATIONSHIP_CONSTRAINT_LOG_VIOLATIONS:
            logger.error(
                f"Relationship completely invalid and no fallback possible: "
                f"{subject_type} {predicate} {object_type}. Errors: {errors}"
            )

        return ValidationResult(
            is_valid=False,
            original_relationship=predicate,
            validated_relationship=None,
            errors=errors,
            suggestions=suggestions,
            confidence=0.0,
        )

    def _find_best_semantic_match(
        self, predicate: str, subject_type: str, object_type: str
    ) -> str | None:
        """
        Find the best semantically similar valid relationship.

        Uses keyword matching and relationship category analysis.
        """
        valid_relationships = get_all_valid_relationships_for_node_pair(
            subject_type, object_type
        )

        if not valid_relationships:
            return None

        # Try to find semantic matches using keyword analysis
        predicate_lower = predicate.lower()
        predicate_words = set(predicate_lower.replace("_", " ").split())

        best_match = None
        best_score = 0.0

        for candidate in valid_relationships:
            candidate_words = set(candidate.lower().replace("_", " ").split())

            # Calculate word overlap score
            overlap = len(predicate_words.intersection(candidate_words))
            if overlap > 0:
                score = overlap / max(len(predicate_words), len(candidate_words))
                if score > best_score:
                    best_score = score
                    best_match = candidate

        # If we found a good match (>50% word overlap), return it
        if best_score > 0.5:
            return best_match

        # Try category-based matching
        predicate_category = self._get_relationship_category(predicate)
        if predicate_category:
            for candidate in valid_relationships:
                if (
                    candidate
                    in models.kg_constants.RELATIONSHIP_CATEGORIES[predicate_category]
                ):
                    return candidate

        return None

    def _get_relationship_category(self, relationship: str) -> str | None:
        """Determine which category a relationship belongs to."""
        for category, relationships in models.kg_constants.RELATIONSHIP_CATEGORIES.items():
            if relationship in relationships:
                return category
        return None

    def validate_triple(self, triple_dict: dict[str, Any]) -> ValidationResult:
        """
        Validate a complete triple from the extraction pipeline.
        
        REFACTORED: Simplified to use TripleProcessor for complex extraction logic.
        This method now focuses only on orchestrating validation rather than handling
        complex type inference and entity extraction.

        Args:
            triple_dict: Dictionary containing subject, predicate, object information
        """
        try:
            # Use triple processor to extract and prepare entity information
            triple_processor = self._get_triple_processor()
            processed_triple = triple_processor.process_triple(triple_dict)
            
            if not processed_triple:
                # Triple processing failed
                return ValidationResult(
                    is_valid=False,
                    original_relationship=triple_dict.get("predicate", "UNKNOWN"),
                    errors=["Failed to process triple - invalid or incomplete information"],
                )

            # Delegate to relationship validation with processed information
            return self.validate_relationship(
                processed_triple["subject_type"],
                processed_triple["predicate"], 
                processed_triple["object_type"],
                triple_dict  # Pass original context
            )

        except Exception as e:
            logger.error(f"Error validating triple {triple_dict}: {e}", exc_info=True)
            return ValidationResult(
                is_valid=False,
                original_relationship=triple_dict.get("predicate", "UNKNOWN"),
                errors=[f"Validation error: {str(e)}"],
            )
    
    def _get_triple_processor(self):
        """Get triple processor via dependency injection."""
        if self._triple_processor is None:
            # Use lazy initialization with dependency injection
            try:
                from core.triple_processor import TripleProcessor
                self._triple_processor = TripleProcessor()
            except Exception as e:
                logger.error(f"Failed to initialize triple processor: {e}")
                raise
        
        return self._triple_processor

    def validate_batch(self, triples: list[dict[str, Any]]) -> list[ValidationResult]:
        """Validate a batch of triples efficiently."""
        results = []
        for triple in triples:
            result = self.validate_triple(triple)
            results.append(result)
        return results

    def get_validation_statistics(self) -> dict[str, Any]:
        """Get validation statistics for monitoring and debugging."""
        total = self.validation_stats["total_validations"]

        if total == 0:
            return self.validation_stats

        return {
            **self.validation_stats,
            "success_rate": self.validation_stats["valid_relationships"] / total * 100,
            "correction_rate": self.validation_stats["corrected_relationships"]
            / total
            * 100,
            "fallback_rate": self.validation_stats["fallback_relationships"]
            / total
            * 100,
            "rejection_rate": self.validation_stats["invalid_relationships"]
            / total
            * 100,
        }

    def suggest_alternatives(
        self, subject_type: str, object_type: str
    ) -> list[tuple[str, str]]:
        """Get suggested valid relationships for a node type pair."""
        return get_relationship_suggestions(subject_type, object_type)

    def get_valid_relationships(self, subject_type: str, object_type: str) -> list[str]:
        """Return all valid relationships for a type pair."""
        return get_all_valid_relationships_for_node_pair(subject_type, object_type)


# REFACTORED: Replaced global singleton with dependency injection
# The global validator instance has been replaced with service provider pattern
# for better testability, thread safety, and dependency management.

def validate_relationship_constraint(
    subject_type: str,
    predicate: str,
    object_type: str,
    context: dict[str, Any] | None = None,
) -> ValidationResult:
    """
    Convenience function for relationship validation using dependency injection.

    REFACTORED: Now uses validation service provider instead of global singleton.
    This is the main entry point that should be used throughout the codebase.
    """
    try:
        from core.validation_service_provider import get_validation_service
        validation_service = get_validation_service()
        return validation_service.validate_relationship(
            subject_type, predicate, object_type, context
        )
    except Exception as e:
        logger.error(f"Failed to get validation service, using fallback: {e}")
        # Fallback to direct instantiation for backward compatibility
        fallback_validator = RelationshipConstraintValidator()
        return fallback_validator.validate_relationship(
            subject_type, predicate, object_type, context
        )


def validate_triple_constraint(triple_dict: dict[str, Any]) -> ValidationResult:
    """
    Convenience function for triple validation using dependency injection.

    REFACTORED: Now uses validation service provider instead of global singleton.
    Validates a complete triple extracted from text.
    """
    try:
        from core.validation_service_provider import get_validation_service
        validation_service = get_validation_service()
        return validation_service.validate_triple(triple_dict)
    except Exception as e:
        logger.error(f"Failed to get validation service, using fallback: {e}")
        # Fallback to direct instantiation for backward compatibility
        fallback_validator = RelationshipConstraintValidator()
        return fallback_validator.validate_triple(triple_dict)


def validate_batch_constraints(triples: list[dict[str, Any]]) -> list[ValidationResult]:
    """
    Convenience function for batch validation using dependency injection.

    REFACTORED: Now uses validation service provider instead of global singleton.
    Efficiently validates multiple triples at once.
    """
    try:
        from core.validation_service_provider import get_validation_service
        validation_service = get_validation_service()
        return validation_service.validate_batch(triples)
    except Exception as e:
        logger.error(f"Failed to get validation service, using fallback: {e}")
        # Fallback to direct instantiation for backward compatibility
        fallback_validator = RelationshipConstraintValidator()
        return fallback_validator.validate_batch(triples)


def get_relationship_alternatives(
    subject_type: str, object_type: str
) -> list[tuple[str, str]]:
    """Get suggested alternative relationships for a node type pair."""
    try:
        from core.validation_service_provider import get_validation_service
        validation_service = get_validation_service()
        return validation_service.suggest_alternatives(subject_type, object_type)
    except Exception as e:
        logger.error(f"Failed to get validation service, using fallback: {e}")
        # Fallback to direct instantiation for backward compatibility
        fallback_validator = RelationshipConstraintValidator()
        return fallback_validator.suggest_alternatives(subject_type, object_type)


def get_validation_stats() -> dict[str, Any]:
    """Get current validation statistics."""
    try:
        from core.validation_service_provider import get_validation_service
        validation_service = get_validation_service()
        return validation_service.get_validation_statistics()
    except Exception as e:
        logger.error(f"Failed to get validation service, using fallback: {e}")
        # Fallback to direct instantiation for backward compatibility
        fallback_validator = RelationshipConstraintValidator()
        return fallback_validator.get_validation_statistics()


# Integration helper functions for existing codebase
def enhance_triple_with_validation(triple_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Enhance a triple dictionary with validation results.

    Adds validation metadata while preserving original structure.
    """
    validation_result = validate_triple_constraint(triple_dict)

    # Add validation metadata
    enhanced_triple = triple_dict.copy()
    enhanced_triple["validation"] = {
        "is_valid": validation_result.is_valid,
        "original_predicate": validation_result.original_relationship,
        "validated_predicate": validation_result.validated_relationship,
        "confidence": validation_result.confidence,
        "errors": validation_result.errors,
        "has_suggestions": len(validation_result.suggestions) > 0,
    }

    # Update the predicate if validation corrected it
    if validation_result.is_valid and validation_result.validated_relationship:
        enhanced_triple["predicate"] = validation_result.validated_relationship

    return enhanced_triple


def should_accept_relationship(
    validation_result: ValidationResult, min_confidence: float | None = None
) -> bool:
    """
    CREATIVE WRITING MODE: Always accept relationships for narrative flexibility.
    
    Creative writing systems need maximum flexibility to explore narrative possibilities.
    Rigid constraints inhibit storytelling creativity and prevent interesting relationships.

    Args:
        validation_result: Result from relationship validation
        min_confidence: Minimum confidence threshold (ignored for creative writing)

    Returns:
        Always True - creative writing needs relationship freedom!
    """
    # ALWAYS ACCEPT for creative writing - no more rejections!
    # Creative writers should be free to explore any relationship they can imagine
    
    if not validation_result.is_valid:
        # Log the creative usage for learning but don't reject
        logger.debug(f"Creative relationship usage: {validation_result.errors} - allowing for narrative exploration")
    
    return True  # Always accept - creativity over constraints!
