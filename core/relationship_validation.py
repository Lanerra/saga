# core/relationship_validation.py
"""Validate relationship semantics for the SAGA workflow.

This module provides semantic validation helpers for relationships extracted from narrative
text. It is intentionally not the schema contract authority.

Notes:
    CORE-011 contract alignment:
    - Canonical node labels are enforced at persistence boundaries (commit/write paths).
    - Relationship types must be safe for Cypher interpolation at persistence boundaries.
    - This validator must not imply that unknown labels/types will be added to the graph.
"""

from __future__ import annotations

from typing import Literal

import structlog

from models.kg_constants import (
    CHARACTER_EMOTIONAL_RELATIONSHIPS,
    CHARACTER_SOCIAL_RELATIONSHIPS,
    RELATIONSHIP_TYPES,
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
    """Define a semantic constraint for one or more relationship types.

    A rule describes which relationship types it applies to and which source/target entity
    labels are considered semantically typical for those types.
    """

    def __init__(
        self,
        relationship_types: set[str],
        valid_source_types: set[str] | Literal["ANY"],
        valid_target_types: set[str] | Literal["ANY"],
        rule_name: str,
        rationale: str = "",
    ):
        """Initialize the rule.

        Args:
            relationship_types: Relationship types this rule applies to.
            valid_source_types: Allowed source node labels, or `"ANY"`.
            valid_target_types: Allowed target node labels, or `"ANY"`.
            rule_name: Short identifier used for logging.
            rationale: Human-readable explanation for informational diagnostics.
        """
        self.relationship_types = relationship_types
        self.valid_source_types = valid_source_types
        self.valid_target_types = valid_target_types
        self.rule_name = rule_name
        self.rationale = rationale

    def validate(self, relationship_type: str, source_type: str, target_type: str) -> tuple[bool, str | None]:
        """Validate a relationship triple against this rule.

        Args:
            relationship_type: Relationship type.
            source_type: Source node label.
            target_type: Target node label.

        Returns:
            Tuple of `(is_valid, message)`. When valid, message is None.
        """
        # Check if this rule applies to this relationship type
        if relationship_type not in self.relationship_types:
            return True, None

        # Validate source type
        if self.valid_source_types != "ANY":
            if source_type not in self.valid_source_types:
                error = (
                    f"Invalid source type '{source_type}' for relationship '{relationship_type}'. " f"Expected one of: {', '.join(sorted(self.valid_source_types))}. " f"Rationale: {self.rationale}"
                )
                return False, error

        # Validate target type
        if self.valid_target_types != "ANY":
            if target_type not in self.valid_target_types:
                error = (
                    f"Invalid target type '{target_type}' for relationship '{relationship_type}'. " f"Expected one of: {', '.join(sorted(self.valid_target_types))}. " f"Rationale: {self.rationale}"
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
        valid_target_types=LOCATION_TYPES | PHYSICAL_OBJECT_TYPES | SENTIENT_TYPES,
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
    # Possession relationships - typically sentient owners
    # (informational only - not enforced)
    RelationshipValidationRule(
        relationship_types={"OWNS", "POSSESSES"},
        valid_source_types=SENTIENT_TYPES,
        valid_target_types=PHYSICAL_OBJECT_TYPES | LOCATION_TYPES,
        rule_name="ownership",
        rationale="[INFO] Ownership typically involves sentient beings",
    ),
    # Physical and informational containment
    # (informational only - not enforced)
    RelationshipValidationRule(
        relationship_types={"CONTAINS", "PART_OF"},
        valid_source_types=PHYSICAL_OBJECT_TYPES | LOCATION_TYPES | EVENT_TYPES | ABSTRACT_TYPES,
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
]


class RelationshipValidator:
    """Validate relationships for semantic plausibility.

    By default, the validator runs in permissive mode: it never blocks relationships
    and only emits informational diagnostics.

    Notes:
        Unknown relationship types and non-canonical entity labels may appear in extracted
        data. Canonicalization and safety checks are enforced at persistence boundaries.
    """

    def __init__(self, enable_strict_mode: bool = False) -> None:
        """Initialize the validator.

        Args:
            enable_strict_mode: If True, enforce rules as hard failures. If False, treat
                rule violations as informational diagnostics.
        """
        self.rules = VALIDATION_RULES
        self.known_relationship_types = RELATIONSHIP_TYPES
        self.known_node_labels = VALID_NODE_LABELS
        self.enable_strict_mode = enable_strict_mode

    def validate_relationship_type(self, relationship_type: str) -> tuple[bool, str | None]:
        """Validate a relationship type against the reference vocabulary.

        Args:
            relationship_type: Relationship type to check.

        Returns:
            Tuple of `(is_valid, message)`.

            In permissive mode, unknown types return `(True, info_message)`.

            In strict mode, unknown types return `(False, error_message)`.

        Notes:
            CORE-011: This validator does not expand the schema. Persistence may still
            accept novel relationship types if they are safe for Cypher interpolation;
            that safety enforcement happens at commit/persist boundaries, not here.
        """
        if relationship_type not in self.known_relationship_types:
            msg = (
                f"Unknown relationship type '{relationship_type}'. "
                "Semantic validator does not expand the schema. "
                "Use an existing canonical relationship type when possible; "
                "otherwise ensure the type is normalized (UPPERCASE_WITH_UNDERSCORES) "
                "and will be accepted by persistence."
            )
            if not self.enable_strict_mode:
                return True, msg
            return False, msg

        return True, None

    def validate_entity_types(self, source_type: str, target_type: str) -> tuple[bool, list[str]]:
        """Validate entity labels against the canonical label set.

        Args:
            source_type: Source entity label.
            target_type: Target entity label.

        Returns:
            Tuple of `(all_valid, messages)`.

            In permissive mode, this returns `(True, messages)` (informational only).

            In strict mode, this returns `(False, messages)` when any label is not in the
            canonical set.

        Notes:
            CORE-011: Unknown entity labels are not added as new labels. Persistence
            enforces canonical node labels; extracted subtypes must be mapped to a
            canonical label before writing.
        """
        messages: list[str] = []

        if source_type not in self.known_node_labels:
            messages.append(
                f"Unknown entity type '{source_type}'. "
                "Node labels are canonical at persistence boundaries; "
                "use a canonical label (e.g., Character/Location/Event/Item/Trait/Chapter) "
                "or ensure it maps to one."
            )

        if target_type not in self.known_node_labels:
            messages.append(f"Unknown entity type '{target_type}'. " "Node labels are canonical at persistence boundaries; " "use a canonical label or ensure it maps to one.")

        if not self.enable_strict_mode:
            return True, messages

        return len(messages) == 0, messages

    def validate_semantic_compatibility(self, relationship_type: str, source_type: str, target_type: str) -> tuple[bool, list[str]]:
        """Validate semantic compatibility of a relationship triple.

        Args:
            relationship_type: Relationship type.
            source_type: Source node label.
            target_type: Target node label.

        Returns:
            Tuple of `(is_valid, messages)`.

            In permissive mode, `is_valid` is always True and messages are informational.

            In strict mode, `is_valid` is False when any rule rejects the combination.
        """
        info_messages = []

        # In permissive mode, we skip rule validation entirely
        # All relationships are valid - trust the LLM
        if not self.enable_strict_mode:
            # Optionally log if this is an unusual combination
            # but don't consider it an error
            for rule in self.rules:
                is_valid, message = rule.validate(relationship_type, source_type, target_type)
                if not is_valid and message:
                    # Convert error to info message
                    info_msg = message.replace("Invalid", "[INFO] Unusual").replace("Expected", "Typically")
                    info_messages.append(info_msg)
            return True, info_messages  # Always valid in permissive mode
        else:
            # Strict mode: apply all validation rules
            errors = []
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
        """Validate a relationship and return diagnostics.

        Args:
            relationship_type: Relationship type.
            source_name: Source entity name (for diagnostic messages only).
            source_type: Source node label.
            target_name: Target entity name (for diagnostic messages only).
            target_type: Target node label.
            severity_mode: Controls whether strict-mode warnings are treated as errors.

        Returns:
            Tuple of `(is_valid, errors, warnings)`.

            In permissive mode, `is_valid` is always True and `errors` is always empty.
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
        entities_valid, entity_messages = self.validate_entity_types(source_type, target_type)
        if not entities_valid or entity_messages:
            if self.enable_strict_mode and severity_mode == "strict":
                errors.extend(entity_messages)
            else:
                info_warnings.extend(entity_messages)

        # 3. Validate semantic compatibility
        semantic_valid, semantic_messages = self.validate_semantic_compatibility(relationship_type, source_type, target_type)
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
    """Return the process-global relationship validator instance."""
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
    "ABSTRACT_TYPES",
    "SYSTEM_TYPES",
    "QUALITY_TYPES",
]
