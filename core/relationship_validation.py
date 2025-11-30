"""
Relationship validation system for SAGA LangGraph workflow.

This module provides semantic validation for relationships extracted from narrative text.
It ensures that relationships make semantic sense given the entity types involved,
while maintaining flexibility for creative writing.

The validation system is designed to be:
1. Semantically sound - prevents nonsensical relationships (e.g., "Character FRIENDS_WITH Location")
2. Flexible - allows creative combinations that make narrative sense
3. Maintainable - easy to extend with new rules
4. Well-documented - clear rationale for each constraint

Design Philosophy:
- Relationships should respect basic semantic categories (social, spatial, etc.)
- Creative flexibility is prioritized over rigid constraints
- Validation should guide toward better modeling, not block creativity
- When in doubt, allow the relationship but log a warning
"""

from __future__ import annotations

from typing import Literal

import structlog

from models.kg_constants import (
    ABILITY_RELATIONSHIPS,
    CHARACTER_EMOTIONAL_RELATIONSHIPS,
    CHARACTER_SOCIAL_RELATIONSHIPS,
    NODE_LABELS,
    ORGANIZATIONAL_RELATIONSHIPS,
    PHYSICAL_RELATIONSHIPS,
    PLOT_CAUSAL_RELATIONSHIPS,
    POSSESSION_RELATIONSHIPS,
    RELATIONSHIP_TYPES,
    SPATIAL_TEMPORAL_RELATIONSHIPS,
    STATUS_RELATIONSHIPS,
    STRUCTURAL_RELATIONSHIPS,
    THEMATIC_RELATIONSHIPS,
)

logger = structlog.get_logger(__name__)


# Entity type categories for validation
# These group entity types by their semantic nature
SENTIENT_TYPES = {
    "Character",
    "Person",
    "Creature",  # Some creatures are sentient
    "Spirit",
    "Deity",
}

LIVING_TYPES = SENTIENT_TYPES | {
    "Creature",  # All creatures (even non-sentient)
}

PHYSICAL_OBJECT_TYPES = {
    "Object",
    "Artifact",
    "Document",
    "Item",
    "Relic",
}

LOCATION_TYPES = {
    "Location",
    "Structure",
    "Region",
    "Landmark",
    "Territory",
    "Path",
    "Room",
    "Settlement",
}

EVENT_TYPES = {
    "Event",
    "DevelopmentEvent",
    "WorldElaborationEvent",
    "Era",
    "Moment",
}

ORGANIZATION_TYPES = {
    "Faction",
    "Organization",
    "Guild",
    "House",
    "Order",
    "Council",
}

ABSTRACT_TYPES = {
    "Concept",
    "Law",
    "Tradition",
    "Symbol",
    "Lore",
    "Knowledge",
    "Secret",
    "Rumor",
}

SYSTEM_TYPES = {
    "System",
    "Magic",
    "Technology",
    "Religion",
    "Culture",
}

QUALITY_TYPES = {
    "Trait",
    "Attribute",
    "Quality",
    "Reputation",
    "Status",
}


class RelationshipValidationRule:
    """
    Defines a validation rule for relationship types.

    Each rule specifies:
    - Which relationship types it applies to
    - Valid source entity type categories
    - Valid target entity type categories
    - Optional rationale for the constraint
    """

    def __init__(
        self,
        relationship_types: set[str],
        valid_source_types: set[str] | Literal["ANY"],
        valid_target_types: set[str] | Literal["ANY"],
        rule_name: str,
        rationale: str = "",
    ):
        """
        Initialize a relationship validation rule.

        Args:
            relationship_types: Set of relationship types this rule applies to
            valid_source_types: Set of valid source entity types, or "ANY"
            valid_target_types: Set of valid target entity types, or "ANY"
            rule_name: Name of the rule for logging
            rationale: Explanation of why this constraint exists
        """
        self.relationship_types = relationship_types
        self.valid_source_types = valid_source_types
        self.valid_target_types = valid_target_types
        self.rule_name = rule_name
        self.rationale = rationale

    def validate(
        self, relationship_type: str, source_type: str, target_type: str
    ) -> tuple[bool, str | None]:
        """
        Validate a relationship against this rule.

        Args:
            relationship_type: The type of relationship
            source_type: The type of the source entity
            target_type: The type of the target entity

        Returns:
            Tuple of (is_valid, error_message)
            If valid, error_message is None
        """
        # Check if this rule applies to this relationship type
        if relationship_type not in self.relationship_types:
            return True, None

        # Validate source type
        if self.valid_source_types != "ANY":
            if source_type not in self.valid_source_types:
                error = (
                    f"Invalid source type '{source_type}' for relationship '{relationship_type}'. "
                    f"Expected one of: {', '.join(sorted(self.valid_source_types))}. "
                    f"Rationale: {self.rationale}"
                )
                return False, error

        # Validate target type
        if self.valid_target_types != "ANY":
            if target_type not in self.valid_target_types:
                error = (
                    f"Invalid target type '{target_type}' for relationship '{relationship_type}'. "
                    f"Expected one of: {', '.join(sorted(self.valid_target_types))}. "
                    f"Rationale: {self.rationale}"
                )
                return False, error

        return True, None


# Define validation rules for each relationship category
VALIDATION_RULES = [
    # Character Social Relationships - require sentient participants
    RelationshipValidationRule(
        relationship_types=CHARACTER_SOCIAL_RELATIONSHIPS,
        valid_source_types=SENTIENT_TYPES,
        valid_target_types=SENTIENT_TYPES,
        rule_name="character_social",
        rationale="Social relationships (friendship, rivalry, etc.) require sentient participants",
    ),
    # Character Emotional Relationships - require sentient source
    RelationshipValidationRule(
        relationship_types=CHARACTER_EMOTIONAL_RELATIONSHIPS,
        valid_source_types=SENTIENT_TYPES,
        valid_target_types="ANY",  # Can love/hate anything
        rule_name="character_emotional",
        rationale="Emotional relationships require a sentient being as the source",
    ),
    # Spatial relationships - target must be a location or physical entity
    RelationshipValidationRule(
        relationship_types={"LOCATED_IN", "LOCATED_AT"},
        valid_source_types="ANY",  # Anything can be located somewhere
        valid_target_types=LOCATION_TYPES | PHYSICAL_OBJECT_TYPES | ORGANIZATION_TYPES,
        rule_name="spatial_containment",
        rationale="Location relationships require a place or physical container as target",
    ),
    # Temporal relationships between events
    RelationshipValidationRule(
        relationship_types={"HAPPENS_BEFORE", "HAPPENS_AFTER", "OCCURS_DURING"},
        valid_source_types=EVENT_TYPES,
        valid_target_types=EVENT_TYPES | {"Era", "Chapter"},
        rule_name="temporal_sequence",
        rationale="Temporal sequence relationships should connect events or time periods",
    ),
    # Organizational membership
    RelationshipValidationRule(
        relationship_types={"MEMBER_OF", "LEADER_OF", "FOUNDED"},
        valid_source_types=SENTIENT_TYPES,
        valid_target_types=ORGANIZATION_TYPES,
        rule_name="organizational_membership",
        rationale="Organizational relationships require sentient beings and organizations",
    ),
    # Possession relationships - source should be sentient or organization
    RelationshipValidationRule(
        relationship_types={"OWNS", "POSSESSES"},
        valid_source_types=SENTIENT_TYPES | ORGANIZATION_TYPES,
        valid_target_types=PHYSICAL_OBJECT_TYPES | LOCATION_TYPES,
        rule_name="ownership",
        rationale="Ownership requires a sentient being or organization owning a physical entity",
    ),
    # Physical containment
    RelationshipValidationRule(
        relationship_types={"CONTAINS", "PART_OF"},
        valid_source_types=PHYSICAL_OBJECT_TYPES | LOCATION_TYPES,
        valid_target_types=PHYSICAL_OBJECT_TYPES | LOCATION_TYPES,
        rule_name="physical_containment",
        rationale="Physical containment relationships require physical entities or locations",
    ),
    # Ability/Trait relationships
    RelationshipValidationRule(
        relationship_types={"HAS_ABILITY", "HAS_TRAIT", "SKILLED_IN", "WEAK_IN"},
        valid_source_types=SENTIENT_TYPES,
        valid_target_types=QUALITY_TYPES | ABSTRACT_TYPES | {"Skill"},
        rule_name="ability_trait",
        rationale="Abilities and traits should be possessed by sentient beings",
    ),
    # Status relationships
    RelationshipValidationRule(
        relationship_types=STATUS_RELATIONSHIPS,
        valid_source_types=SENTIENT_TYPES | ORGANIZATION_TYPES,
        valid_target_types="ANY",
        rule_name="status",
        rationale="Status relationships describe the state of sentient beings or organizations",
    ),
]


class RelationshipValidator:
    """
    Validates relationships for semantic correctness.

    This validator checks that relationships make semantic sense given the
    entity types involved. It uses a rule-based system that can be extended
    and customized.
    """

    def __init__(self):
        """Initialize the relationship validator."""
        self.rules = VALIDATION_RULES
        self.known_relationship_types = RELATIONSHIP_TYPES
        self.known_node_labels = NODE_LABELS

    def validate_relationship_type(self, relationship_type: str) -> tuple[bool, str | None]:
        """
        Check if a relationship type is recognized.

        Args:
            relationship_type: The relationship type to validate

        Returns:
            Tuple of (is_valid, warning_message)
        """
        if relationship_type not in self.known_relationship_types:
            warning = (
                f"Unknown relationship type '{relationship_type}'. "
                f"This relationship type is not in the canonical schema. "
                f"Consider using one of the standard types or adding this to kg_constants.py"
            )
            return False, warning
        return True, None

    def validate_entity_types(
        self, source_type: str, target_type: str
    ) -> tuple[bool, list[str]]:
        """
        Check if entity types are recognized.

        Args:
            source_type: The source entity type
            target_type: The target entity type

        Returns:
            Tuple of (all_valid, list of warnings)
        """
        warnings = []

        if source_type not in self.known_node_labels:
            warnings.append(
                f"Unknown source entity type '{source_type}'. "
                f"Consider using a type from the ontology (docs/ontology.md)"
            )

        if target_type not in self.known_node_labels:
            warnings.append(
                f"Unknown target entity type '{target_type}'. "
                f"Consider using a type from the ontology (docs/ontology.md)"
            )

        return len(warnings) == 0, warnings

    def validate_semantic_compatibility(
        self, relationship_type: str, source_type: str, target_type: str
    ) -> tuple[bool, list[str]]:
        """
        Validate that the relationship makes semantic sense for the entity types.

        Args:
            relationship_type: The type of relationship
            source_type: The type of the source entity
            target_type: The type of the target entity

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Apply all validation rules
        for rule in self.rules:
            is_valid, error = rule.validate(relationship_type, source_type, target_type)
            if not is_valid and error:
                errors.append(error)

        return len(errors) == 0, errors

    def validate(
        self,
        relationship_type: str,
        source_name: str,
        source_type: str,
        target_name: str,
        target_type: str,
        severity_mode: Literal["strict", "flexible"] = "flexible",
    ) -> tuple[bool, list[str], list[str]]:
        """
        Perform complete validation of a relationship.

        Args:
            relationship_type: The type of relationship
            source_name: Name of the source entity
            source_type: Type of the source entity
            target_name: Name of the target entity
            target_type: Type of the target entity
            severity_mode: "strict" fails on warnings, "flexible" allows them

        Returns:
            Tuple of (is_valid, list of errors, list of warnings)
        """
        errors = []
        warnings = []

        # 1. Validate relationship type
        type_valid, type_warning = self.validate_relationship_type(relationship_type)
        if not type_valid and type_warning:
            if severity_mode == "strict":
                errors.append(type_warning)
            else:
                warnings.append(type_warning)

        # 2. Validate entity types
        entities_valid, entity_warnings = self.validate_entity_types(source_type, target_type)
        if not entities_valid:
            if severity_mode == "strict":
                errors.extend(entity_warnings)
            else:
                warnings.extend(entity_warnings)

        # 3. Validate semantic compatibility
        semantic_valid, semantic_errors = self.validate_semantic_compatibility(
            relationship_type, source_type, target_type
        )
        if not semantic_valid:
            errors.extend(semantic_errors)

        is_valid = len(errors) == 0

        # Log validation results
        if not is_valid:
            logger.warning(
                "relationship_validation_failed",
                relationship=f"{source_name}({source_type}) -{relationship_type}-> {target_name}({target_type})",
                errors=errors,
                warnings=warnings,
            )
        elif warnings:
            logger.info(
                "relationship_validation_warnings",
                relationship=f"{source_name}({source_type}) -{relationship_type}-> {target_name}({target_type})",
                warnings=warnings,
            )

        return is_valid, errors, warnings


# Global validator instance
_validator = None


def get_relationship_validator() -> RelationshipValidator:
    """Get or create the global relationship validator instance."""
    global _validator
    if _validator is None:
        _validator = RelationshipValidator()
    return _validator


__all__ = [
    "RelationshipValidator",
    "RelationshipValidationRule",
    "get_relationship_validator",
    "SENTIENT_TYPES",
    "LIVING_TYPES",
    "PHYSICAL_OBJECT_TYPES",
    "LOCATION_TYPES",
    "EVENT_TYPES",
    "ORGANIZATION_TYPES",
    "ABSTRACT_TYPES",
    "SYSTEM_TYPES",
    "QUALITY_TYPES",
]
