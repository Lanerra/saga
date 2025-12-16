# core/schema_validator.py
"""
Schema validation utilities for the knowledge graph.

Implements enforcement of the Node Label Schema, including:
- Type validation against allowed labels
- Label normalization (mapping variants to canonical types)
- Category suggestions and validation
"""

from typing import Any

import structlog

from config import (
    ENFORCE_SCHEMA_VALIDATION,
    LOG_SCHEMA_VIOLATIONS,
    NORMALIZE_COMMON_VARIANTS,
)
from models.kg_constants import (
    LABEL_NORMALIZATION_MAP,
    SUGGESTED_CATEGORIES,
    VALID_NODE_LABELS,
)
from models.kg_models import CharacterProfile, WorldItem

logger = structlog.get_logger(__name__)


class SchemaValidationService:
    """
    Service for validating and normalizing entity types and categories
    against the defined Node Label Schema.
    """

    def __init__(self):
        self.enabled = ENFORCE_SCHEMA_VALIDATION
        self.normalize_variants = NORMALIZE_COMMON_VARIANTS
        self.log_violations = LOG_SCHEMA_VIOLATIONS

    def validate_entity_type(self, type_name: str) -> tuple[bool, str, str | None]:
        """
        Validate and optionally normalize an entity type.

        Args:
            type_name: The entity type label to validate.

        Returns:
            Tuple containing:
            - is_valid (bool): Whether the type is valid (after normalization if enabled).
            - normalized_name (str): The normalized type name (or original if no change).
            - error_msg (Optional[str]): Error message if invalid, else None.
        """
        if not self.enabled:
            return True, type_name, None

        raw = str(type_name).strip() if type_name is not None else ""
        if not raw:
            return False, raw, "Entity type cannot be empty"

        # Check exact match first (canonical casing)
        if raw in VALID_NODE_LABELS:
            return True, raw, None

        # Case-insensitive canonical match (e.g., "character" -> "Character")
        canonical_case_map: dict[str, str] = {lbl.lower(): lbl for lbl in VALID_NODE_LABELS}
        canonical_ci = canonical_case_map.get(raw.lower())
        if canonical_ci:
            if self.log_violations and raw != canonical_ci:
                logger.info(
                    "Normalizing entity type (case)",
                    original=raw,
                    normalized=canonical_ci,
                )
            return True, canonical_ci, None

        # Check normalization map (legacy aliases + subtype labels -> canonical labels)
        if self.normalize_variants:
            mapped = LABEL_NORMALIZATION_MAP.get(raw) or LABEL_NORMALIZATION_MAP.get(raw.lower())
            if mapped:
                if self.log_violations:
                    logger.info(
                        "Normalizing entity type",
                        original=raw,
                        normalized=mapped,
                    )
                return True, mapped, None

        # If we get here, it's not a valid label
        error_msg = f"Invalid entity type '{raw}'. " f"Must be one of: {', '.join(sorted(VALID_NODE_LABELS))}"

        if self.log_violations:
            logger.warning("Schema violation", error=error_msg, type_name=raw)

        return False, raw, error_msg

    def validate_category(self, type_name: str, category: str) -> tuple[bool, str | None]:
        """
        Validate if a category is appropriate for the given entity type.
        This is a soft validation (warning only) as categories are open-ended.

        Args:
            type_name: The entity type (should be normalized first).
            category: The category to check.

        Returns:
            Tuple containing:
            - is_known (bool): True if category is in the suggested list.
            - suggestion_msg (Optional[str]): Message suggesting valid categories if unknown.
        """
        if not category or type_name not in SUGGESTED_CATEGORIES:
            return True, None

        suggested = SUGGESTED_CATEGORIES[type_name]
        # Case-insensitive check
        if any(c.lower() == category.lower() for c in suggested):
            return True, None

        msg = f"Category '{category}' is not in suggested list for {type_name}. " f"Suggested: {', '.join(suggested)}"
        return False, msg


# Global instance
schema_validator = SchemaValidationService()


def canonicalize_entity_type_for_persistence(type_name: str) -> str:
    """
    Canonicalize an entity type into a strict canonical node label for persistence.

    CORE-011 contract (canonical-label-first at persistence boundaries):
    - By the time we build Cypher that interpolates labels, the label MUST be one of
      [`VALID_NODE_LABELS`](models/kg_constants.py:66).
    - Subtypes / legacy labels (e.g., "Structure", "Guild", "DevelopmentEvent") are
      permitted as *inputs* but are canonicalized via [`LABEL_NORMALIZATION_MAP`](models/kg_constants.py:83).
    - Unknown/unmappable labels are rejected with a clear, actionable error.

    Notes:
    - This function intentionally does NOT consult config flags like ENFORCE_SCHEMA_VALIDATION.
      Persistence boundaries must be deterministic and safe regardless of "early validation" toggles.

    Raises:
        ValueError: If the type cannot be normalized to a canonical label.

    Returns:
        Canonical label string (one of VALID_NODE_LABELS).
    """
    raw = str(type_name).strip() if type_name is not None else ""
    if not raw:
        raise ValueError("Entity type cannot be empty at persistence boundary")

    # Fast path: exact canonical match
    if raw in VALID_NODE_LABELS:
        return raw

    # Case-insensitive canonical match (e.g., "character" -> "Character")
    canonical_case_map: dict[str, str] = {lbl.lower(): lbl for lbl in VALID_NODE_LABELS}
    canonical_ci = canonical_case_map.get(raw.lower())
    if canonical_ci:
        return canonical_ci

    # Alias/subtype normalization map (e.g., "Person" -> "Character", "Structure" -> "Location")
    mapped = LABEL_NORMALIZATION_MAP.get(raw) or LABEL_NORMALIZATION_MAP.get(raw.lower())
    if mapped and mapped in VALID_NODE_LABELS:
        return mapped

    raise ValueError(f"Invalid entity type '{raw}' at persistence boundary. " f"Must be one of {sorted(VALID_NODE_LABELS)} " f"(or a known alias/subtype that maps via LABEL_NORMALIZATION_MAP).")


def validate_kg_object(obj: Any) -> list[str]:
    """
    Validate a knowledge graph object (CharacterProfile or WorldItem).

    Returns a list of validation errors. Empty list means valid.
    """
    errors = []

    if isinstance(obj, CharacterProfile):
        # Validate CharacterProfile
        if not obj.name or not obj.name.strip():
            errors.append("CharacterProfile name cannot be empty")

        # Validate traits
        if not isinstance(obj.traits, list):
            errors.append("CharacterProfile traits must be a list")
        else:
            for trait in obj.traits:
                if not isinstance(trait, str) or not trait.strip():
                    errors.append("CharacterProfile traits must be non-empty strings")

        # Validate relationships
        if not isinstance(obj.relationships, dict):
            errors.append("CharacterProfile relationships must be a dict")

        # Validate status
        if not isinstance(obj.status, str):
            errors.append("CharacterProfile status must be a string")

    elif isinstance(obj, WorldItem):
        # Validate WorldItem
        if not obj.name or not obj.name.strip():
            errors.append("WorldItem name cannot be empty")

        if not obj.category or not obj.category.strip():
            errors.append("WorldItem category cannot be empty")

        # Validate structured fields
        if not isinstance(obj.description, str):
            errors.append("WorldItem description must be a string")

        if not isinstance(obj.goals, list):
            errors.append("WorldItem goals must be a list")
        else:
            for goal in obj.goals:
                if not isinstance(goal, str) or not goal.strip():
                    errors.append("WorldItem goals must be non-empty strings")

        if not isinstance(obj.rules, list):
            errors.append("WorldItem rules must be a list")
        else:
            for rule in obj.rules:
                if not isinstance(rule, str) or not rule.strip():
                    errors.append("WorldItem rules must be non-empty strings")

        if not isinstance(obj.key_elements, list):
            errors.append("WorldItem key_elements must be a list")
        else:
            for element in obj.key_elements:
                if not isinstance(element, str) or not element.strip():
                    errors.append("WorldItem key_elements must be non-empty strings")

        if not isinstance(obj.traits, list):
            errors.append("WorldItem traits must be a list")
        else:
            for trait in obj.traits:
                if not isinstance(trait, str) or not trait.strip():
                    errors.append("WorldItem traits must be non-empty strings")

        # Validate additional properties
        if not isinstance(obj.additional_properties, dict):
            errors.append("WorldItem additional_properties must be a dict")

        # Schema validation for WorldItem type
        # World items usually default to "Item" but might be other things
        # This check is soft unless we enforce strict typing on the model itself
        if hasattr(obj, "type") and obj.type:
            is_valid, normalized, err = schema_validator.validate_entity_type(obj.type)
            if not is_valid:
                errors.append(f"Invalid WorldItem type: {err}")

    else:
        errors.append(f"Unknown object type for validation: {type(obj)}")

    return errors


def validate_node_labels(labels: list[str]) -> list[str]:
    """
    Validate node labels for the knowledge graph.

    Returns a list of validation errors. Empty list means valid.
    """
    errors = []

    if not isinstance(labels, list):
        errors.append("Node labels must be a list")
        return errors

    for label in labels:
        if not isinstance(label, str) or not label.strip():
            errors.append("Node labels must be non-empty strings")

        # Apply schema validation
        is_valid, _, err = schema_validator.validate_entity_type(label)
        if not is_valid and ENFORCE_SCHEMA_VALIDATION:
            errors.append(f"Invalid label '{label}': {err}")

        elif not label[0].isupper():
            errors.append(f"Node label '{label}' should start with an uppercase letter")
        elif not label.isalnum():
            errors.append(f"Node label '{label}' should only contain alphanumeric characters")

    return errors


def validate_world_item(item: WorldItem) -> list[str]:
    """
    Validate a WorldItem object.

    Returns a list of validation errors. Empty list means valid.
    """
    # Use centralized validation logic to eliminate code duplication
    return validate_kg_object(item)
