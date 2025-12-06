"""
Relationship validation system for SAGA LangGraph workflow.

This module provides semantic validation for relationships extracted from narrative text.
It allows LLMs maximum creative freedom in assigning relationships while optionally
flagging potentially unusual combinations for review.

**PERMISSIVE MODE** (Default):
The validation system now operates in a fully permissive mode by default, allowing
the LLM to create any relationships it deems narratively appropriate. This supports:
- Creative and unconventional narrative structures
- Metaphorical and symbolic relationships
- Genre-specific relationship types (sci-fi, fantasy, experimental)
- Emergent relationship patterns not anticipated in the schema

Design Philosophy:
- Trust the LLM to create semantically appropriate relationships
- Creative flexibility is paramount - no hard constraints
- Validation logs informational warnings only, never blocks
- Unknown relationship types are allowed and logged
- Entity type mismatches are allowed with warnings
- The system learns from what the LLM creates rather than constraining it
"""

from __future__ import annotations

from typing import Literal

import structlog

from models.kg_constants import (
    CHARACTER_EMOTIONAL_RELATIONSHIPS,
    CHARACTER_SOCIAL_RELATIONSHIPS,
    RELATIONSHIP_TYPES,
    STATUS_RELATIONSHIPS,
    VALID_NODE_LABELS,
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
# NOTE: These rules are now DISABLED by default for permissive mode.
# They are kept for documentation and optional analytical purposes.
# To enable strict validation, set ENABLE_STRICT_VALIDATION = True in config.
VALIDATION_RULES = [
    # Character Social Relationships - typically involve sentient participants
    # (informational only - not enforced)
    RelationshipValidationRule(
        relationship_types=CHARACTER_SOCIAL_RELATIONSHIPS,
        valid_source_types=SENTIENT_TYPES,
        valid_target_types=SENTIENT_TYPES,
        rule_name="character_social",
        rationale="[INFO] Social relationships typically involve sentient participants",
    ),
    # Character Emotional Relationships - typically have sentient source
    # (informational only - not enforced)
    RelationshipValidationRule(
        relationship_types=CHARACTER_EMOTIONAL_RELATIONSHIPS,
        valid_source_types=SENTIENT_TYPES,
        valid_target_types="ANY",
        rule_name="character_emotional",
        rationale="[INFO] Emotional relationships typically have a sentient being as source",
    ),
    # Spatial relationships - typically target locations or physical entities
    # (informational only - not enforced)
    RelationshipValidationRule(
        relationship_types={"LOCATED_IN", "LOCATED_AT"},
        valid_source_types="ANY",
        valid_target_types=LOCATION_TYPES
        | PHYSICAL_OBJECT_TYPES
        | ORGANIZATION_TYPES
        | SENTIENT_TYPES,
        rule_name="spatial_containment",
        rationale="[INFO] Location relationships typically reference places or containers",
    ),
    # Temporal relationships - typically between events
    # (informational only - not enforced)
    RelationshipValidationRule(
        relationship_types={"HAPPENS_BEFORE", "HAPPENS_AFTER", "OCCURS_DURING"},
        valid_source_types=EVENT_TYPES,
        valid_target_types=EVENT_TYPES | {"Era", "Chapter"},
        rule_name="temporal_sequence",
        rationale="[INFO] Temporal relationships typically connect events or time periods",
    ),
    # Organizational membership - typically sentient beings and organizations
    # (informational only - not enforced)
    RelationshipValidationRule(
        relationship_types={"MEMBER_OF", "LEADER_OF", "FOUNDED"},
        valid_source_types=SENTIENT_TYPES,
        valid_target_types=ORGANIZATION_TYPES,
        rule_name="organizational_membership",
        rationale="[INFO] Organizational relationships typically involve sentient beings and organizations",
    ),
    # Possession relationships - typically sentient owners
    # (informational only - not enforced)
    RelationshipValidationRule(
        relationship_types={"OWNS", "POSSESSES"},
        valid_source_types=SENTIENT_TYPES | ORGANIZATION_TYPES,
        valid_target_types=PHYSICAL_OBJECT_TYPES | LOCATION_TYPES,
        rule_name="ownership",
        rationale="[INFO] Ownership typically involves sentient beings or organizations",
    ),
    # Physical and informational containment
    # (informational only - not enforced)
    RelationshipValidationRule(
        relationship_types={"CONTAINS", "PART_OF"},
        valid_source_types=PHYSICAL_OBJECT_TYPES
        | LOCATION_TYPES
        | EVENT_TYPES
        | ABSTRACT_TYPES,
        valid_target_types="ANY",
        rule_name="containment",
        rationale="[INFO] Containment relationships typically involve physical or conceptual containers",
    ),
    # Ability/Trait relationships
    # (informational only - not enforced)
    RelationshipValidationRule(
        relationship_types={"HAS_ABILITY", "HAS_TRAIT", "SKILLED_IN", "WEAK_IN"},
        valid_source_types=SENTIENT_TYPES,
        valid_target_types=QUALITY_TYPES | ABSTRACT_TYPES | {"Skill"},
        rule_name="ability_trait",
        rationale="[INFO] Abilities and traits are typically possessed by sentient beings",
    ),
    # Status relationships
    # (informational only - not enforced)
    RelationshipValidationRule(
        relationship_types=STATUS_RELATIONSHIPS,
        valid_source_types=SENTIENT_TYPES | ORGANIZATION_TYPES,
        valid_target_types="ANY",
        rule_name="status",
        rationale="[INFO] Status relationships typically describe states of sentient beings or organizations",
    ),
]


class RelationshipValidator:
    """
    Validates relationships for semantic correctness.

    **PERMISSIVE MODE** (Default):
    This validator operates in permissive mode by default, allowing all relationships
    and only logging informational messages. It trusts the LLM to create semantically
    appropriate relationships for the narrative context.

    The validator can optionally flag unusual combinations for review, but never
    blocks or rejects relationships. Unknown relationship types and entity types
    are allowed and encouraged, supporting creative and emergent narrative patterns.
    """

    def __init__(self, enable_strict_mode: bool = False) -> None:
        """
        Initialize the relationship validator.

        Args:
            enable_strict_mode: If True, enforces validation rules strictly.
                               If False (default), operates in permissive mode.
        """
        self.rules = VALIDATION_RULES
        self.known_relationship_types = RELATIONSHIP_TYPES
        self.known_node_labels = VALID_NODE_LABELS
        self.enable_strict_mode = enable_strict_mode

    def validate_relationship_type(
        self, relationship_type: str
    ) -> tuple[bool, str | None]:
        """
        Check if a relationship type is recognized.

        In permissive mode (default), this always returns True and only logs
        informational messages for unknown types.

        Args:
            relationship_type: The relationship type to validate

        Returns:
            Tuple of (is_valid, warning_message)
            In permissive mode, is_valid is always True.
        """
        if relationship_type not in self.known_relationship_types:
            info = (
                f"[INFO] Novel relationship type '{relationship_type}' detected. "
                f"This will be added to the knowledge graph. "
                f"The LLM is free to create new relationship types as needed for the narrative."
            )
            # In permissive mode, we allow unknown types
            if not self.enable_strict_mode:
                return True, info  # Valid, with info message
            else:
                return False, info  # Invalid in strict mode
        return True, None

    def validate_entity_types(
        self, source_type: str, target_type: str
    ) -> tuple[bool, list[str]]:
        """
        Check if entity types are recognized.

        In permissive mode (default), this always returns True and only logs
        informational messages for unknown types.

        Args:
            source_type: The source entity type
            target_type: The target entity type

        Returns:
            Tuple of (all_valid, list of warnings)
            In permissive mode, all_valid is always True.
        """
        info_messages = []

        if source_type not in self.known_node_labels:
            info_messages.append(
                f"[INFO] Novel entity type '{source_type}' detected. "
                f"This will be added to the knowledge graph as a new node type."
            )

        if target_type not in self.known_node_labels:
            info_messages.append(
                f"[INFO] Novel entity type '{target_type}' detected. "
                f"This will be added to the knowledge graph as a new node type."
            )

        # In permissive mode, unknown types are valid
        if not self.enable_strict_mode:
            return True, info_messages
        else:
            return len(info_messages) == 0, info_messages

    def validate_semantic_compatibility(
        self, relationship_type: str, source_type: str, target_type: str
    ) -> tuple[bool, list[str]]:
        """
        Validate that the relationship makes semantic sense for the entity types.

        In permissive mode (default), this always returns True and may log
        informational messages about unusual combinations.

        Args:
            relationship_type: The type of relationship
            source_type: The type of the source entity
            target_type: The type of the target entity

        Returns:
            Tuple of (is_valid, list of error/info messages)
            In permissive mode, is_valid is always True.
        """
        info_messages = []

        # In permissive mode, we skip rule validation entirely
        # All relationships are valid - trust the LLM
        if not self.enable_strict_mode:
            # Optionally log if this is an unusual combination
            # but don't consider it an error
            for rule in self.rules:
                is_valid, message = rule.validate(
                    relationship_type, source_type, target_type
                )
                if not is_valid and message:
                    # Convert error to info message
                    info_msg = message.replace("Invalid", "[INFO] Unusual").replace(
                        "Expected", "Typically"
                    )
                    info_messages.append(info_msg)
            return True, info_messages  # Always valid in permissive mode
        else:
            # Strict mode: apply all validation rules
            errors = []
            for rule in self.rules:
                is_valid, error = rule.validate(
                    relationship_type, source_type, target_type
                )
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

        In permissive mode (default), this always returns valid=True and only
        logs informational messages.

        Args:
            relationship_type: The type of relationship
            source_name: Name of the source entity
            source_type: Type of the source entity
            target_name: Name of the target entity
            target_type: Type of the target entity
            severity_mode: "strict" fails on warnings, "flexible" allows them
                          (Note: permissive mode overrides this)

        Returns:
            Tuple of (is_valid, list of errors, list of warnings/info)
            In permissive mode, is_valid is always True and errors is always empty.
        """
        errors = []
        info_warnings = []

        # 1. Validate relationship type
        type_valid, type_message = self.validate_relationship_type(relationship_type)
        if not type_valid and type_message:
            if self.enable_strict_mode and severity_mode == "strict":
                errors.append(type_message)
            else:
                info_warnings.append(type_message)

        # 2. Validate entity types
        entities_valid, entity_messages = self.validate_entity_types(
            source_type, target_type
        )
        if not entities_valid or entity_messages:
            if self.enable_strict_mode and severity_mode == "strict":
                errors.extend(entity_messages)
            else:
                info_warnings.extend(entity_messages)

        # 3. Validate semantic compatibility
        semantic_valid, semantic_messages = self.validate_semantic_compatibility(
            relationship_type, source_type, target_type
        )
        if not semantic_valid:
            if self.enable_strict_mode:
                errors.extend(semantic_messages)
            else:
                info_warnings.extend(semantic_messages)

        # In permissive mode, always valid
        is_valid = len(errors) == 0 if self.enable_strict_mode else True

        # Log validation results
        if not is_valid and self.enable_strict_mode:
            logger.warning(
                "relationship_validation_failed",
                relationship=f"{source_name}({source_type}) -{relationship_type}-> {target_name}({target_type})",
                errors=errors,
                warnings=info_warnings,
            )
        elif info_warnings:
            logger.debug(
                "relationship_validation_info",
                relationship=f"{source_name}({source_type}) -{relationship_type}-> {target_name}({target_type})",
                info=info_warnings,
            )

        return is_valid, errors, info_warnings


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
