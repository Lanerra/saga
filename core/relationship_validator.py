# core/relationship_validator.py
"""
Core relationship validation framework for the SAGA knowledge graph.

This module provides the main validation engine that integrates with the existing
pipeline to validate relationship semantics before they are stored in the database.
"""

import logging
from typing import Any

import models.kg_constants

import config
from core.relationship_constraints import (
    get_all_valid_relationships_for_node_pair,
    get_relationship_suggestions,
    validate_relationship_semantics,
)
from data_access.kg_queries import (
    validate_relationship_type,
)

logger = logging.getLogger(__name__)


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
    """Main validation engine for relationship constraints."""

    def __init__(self):
        self.validation_stats = {
            "total_validations": 0,
            "valid_relationships": 0,
            "invalid_relationships": 0,
            "corrected_relationships": 0,
            "fallback_relationships": 0,
        }

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
        best_score = 0

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

        Args:
            triple_dict: Dictionary containing subject, predicate, object information
                        Expected format: {
                            "subject": {"name": "...", "type": "..."},
                            "predicate": "...",
                            "object_entity": {"name": "...", "type": "..."} | None,
                            "object_literal": "..." | None,
                            "is_literal_object": bool
                        }
        """
        try:
            subject_info = triple_dict.get("subject", {})
            subject_type = subject_info.get("type", "Entity")
            predicate = triple_dict.get("predicate", "")

            # Handle literal vs entity objects
            is_literal_object = triple_dict.get("is_literal_object", False)
            if is_literal_object:
                object_type = "ValueNode"  # Literals become ValueNode entities
            else:
                object_info = triple_dict.get("object_entity", {})
                object_type = object_info.get("type", "Entity")

            # Handle invalid node types gracefully
            if subject_type not in models.kg_constants.NODE_LABELS:
                # Try to infer a better type or default to Entity
                if "literal" in subject_type.lower() or "value" in subject_type.lower():
                    subject_type = "ValueNode"
                elif "response" in subject_type.lower():
                    subject_type = "ValueNode"  # Treat responses as value nodes
                else:
                    subject_type = "Entity"

            if object_type not in models.kg_constants.NODE_LABELS:
                if "literal" in object_type.lower() or "value" in object_type.lower():
                    object_type = "ValueNode"
                elif "response" in object_type.lower():
                    object_type = "ValueNode"
                else:
                    object_type = "Entity"

            return self.validate_relationship(
                subject_type, predicate, object_type, triple_dict
            )

        except Exception as e:
            logger.error(f"Error validating triple {triple_dict}: {e}", exc_info=True)
            return ValidationResult(
                is_valid=False,
                original_relationship=triple_dict.get("predicate", "UNKNOWN"),
                errors=[f"Validation error: {str(e)}"],
            )

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


# Global validator instance
validator = RelationshipConstraintValidator()


def validate_relationship_constraint(
    subject_type: str,
    predicate: str,
    object_type: str,
    context: dict[str, Any] | None = None,
) -> ValidationResult:
    """
    Convenience function for relationship validation.

    This is the main entry point that should be used throughout the codebase.
    """
    return validator.validate_relationship(
        subject_type, predicate, object_type, context
    )


def validate_triple_constraint(triple_dict: dict[str, Any]) -> ValidationResult:
    """
    Convenience function for triple validation.

    Validates a complete triple extracted from text.
    """
    return validator.validate_triple(triple_dict)


def validate_batch_constraints(triples: list[dict[str, Any]]) -> list[ValidationResult]:
    """
    Convenience function for batch validation.

    Efficiently validates multiple triples at once.
    """
    return validator.validate_batch(triples)


def get_relationship_alternatives(
    subject_type: str, object_type: str
) -> list[tuple[str, str]]:
    """Get suggested alternative relationships for a node type pair."""
    return validator.suggest_alternatives(subject_type, object_type)


def get_validation_stats() -> dict[str, Any]:
    """Get current validation statistics."""
    return validator.get_validation_statistics()


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
