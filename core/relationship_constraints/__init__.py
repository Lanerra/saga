"""Relationship constraint system with plugin-based categories."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

import structlog
from models.kg_constants import NODE_LABELS

from .classifications import NodeClassifications

logger = structlog.get_logger(__name__)

# Global registry populated from plugins
RELATIONSHIP_CONSTRAINTS: dict[str, dict[str, Any]] = {}


def _load_plugins() -> None:
    """Load all relationship constraint plugins."""
    package = f"{__name__}.plugins"
    try:
        pkg = importlib.import_module(package)
    except ModuleNotFoundError:
        logger.error(f"Plugin package {package} not found")
        return

    for _, module_name, _ in pkgutil.iter_modules(pkg.__path__):
        module = importlib.import_module(f"{package}.{module_name}")
        constraints = getattr(module, "RELATIONSHIP_CONSTRAINTS", {})
        RELATIONSHIP_CONSTRAINTS.update(constraints)
        logger.debug(f"Loaded {len(constraints)} constraints from {module_name}")


_load_plugins()


SEMANTIC_COMPATIBILITY_RULES = {
    # Emotional relationships require conscious subjects
    "emotional_subjects_must_be_conscious": {
        "relationships": ["LOVES", "HATES", "FEARS", "RESPECTS", "ENVIES", "PITIES"],
        "rule": "Subject must be from CONSCIOUS classification",
        "violation_message": "Only conscious entities can have emotional relationships",
    },
    # Social relationships require sentient participants
    "social_requires_sentient": {
        "relationships": ["FAMILY_OF", "FRIEND_OF", "ROMANTIC_WITH", "ALLY_OF"],
        "rule": "Both subject and object must be SENTIENT",
        "violation_message": "Social relationships require sentient participants",
    },
    # Physical relationships require physical presence
    "physical_requires_presence": {
        "relationships": [
            "LOCATED_IN",
            "LOCATED_AT",
            "NEAR",
            "ADJACENT_TO",
            "PART_OF",
        ],  # Removed CONTAINS as it now handles informational containment too
        "rule": "Both nodes must have PHYSICAL_PRESENCE trait",
        "violation_message": "Physical relationships require entities with physical presence",
    },
    # Ownership cannot create slavery
    "no_sentient_ownership": {
        "relationships": ["OWNS", "POSSESSES"],
        "rule": "Cannot own/possess sentient beings",
        "violation_message": "Ownership of sentient beings is not permitted",
    },
    # Cognitive relationships require conscious subjects
    "cognitive_requires_consciousness": {
        "relationships": [
            "BELIEVES",
            "REALIZES",
            "REMEMBERS",
            "UNDERSTANDS",
            "THINKS_ABOUT",
        ],
        "rule": "Subject must be CONSCIOUS",
        "violation_message": "Cognitive relationships require conscious subjects",
    },
}


def get_node_classifications(node_type: str) -> set[str]:
    """Get all classifications that apply to a given node type."""
    classifications = set()

    if node_type in NodeClassifications.SENTIENT:
        classifications.add("SENTIENT")
    if node_type in NodeClassifications.INANIMATE:
        classifications.add("INANIMATE")
    if node_type in NodeClassifications.SPATIAL:
        classifications.add("SPATIAL")
    if node_type in NodeClassifications.ABSTRACT:
        classifications.add("ABSTRACT")
    if node_type in NodeClassifications.TEMPORAL:
        classifications.add("TEMPORAL")
    if node_type in NodeClassifications.ORGANIZATIONAL:
        classifications.add("ORGANIZATIONAL")
    if node_type in NodeClassifications.PHYSICAL_PRESENCE:
        classifications.add("PHYSICAL_PRESENCE")
    if node_type in NodeClassifications.CONSCIOUS:
        classifications.add("CONSCIOUS")
    if node_type in NodeClassifications.LOCATABLE:
        classifications.add("LOCATABLE")
    if node_type in NodeClassifications.OWNABLE:
        classifications.add("OWNABLE")
    if node_type in NodeClassifications.SOCIAL:
        classifications.add("SOCIAL")

    return classifications


def get_constraint_for_relationship(relationship_type: str) -> dict[str, Any] | None:
    """Get constraint definition for a relationship type."""
    return RELATIONSHIP_CONSTRAINTS.get(relationship_type)


def get_all_valid_relationships_for_node_pair(
    subject_type: str, object_type: str
) -> list[str]:
    """Get all valid relationship types between two node types."""
    valid_relationships = []

    for rel_type, _constraints in RELATIONSHIP_CONSTRAINTS.items():
        if is_relationship_valid(subject_type, rel_type, object_type):
            valid_relationships.append(rel_type)

    return valid_relationships


def is_relationship_valid(
    subject_type: str, relationship_type: str, object_type: str
) -> bool:
    """Check if a specific relationship between two node types is valid."""
    constraints = RELATIONSHIP_CONSTRAINTS.get(relationship_type)
    if not constraints:
        logger.warning(
            f"No constraints defined for relationship type: {relationship_type}"
        )
        return False

    # Check subject type validity
    valid_subjects = constraints.get("valid_subject_types", set())
    if subject_type not in valid_subjects:
        return False

    # Check object type validity
    valid_objects = constraints.get("valid_object_types", set())
    if object_type not in valid_objects:
        return False

    # Check invalid combinations
    invalid_combos = constraints.get("invalid_combinations", [])
    if (subject_type, object_type) in invalid_combos:
        return False

    # Check semantic compatibility rules
    for _rule_name, rule_config in SEMANTIC_COMPATIBILITY_RULES.items():
        if relationship_type in rule_config["relationships"]:
            if not _check_semantic_rule(
                rule_config, subject_type, object_type, relationship_type
            ):
                return False

    return True


def _check_semantic_rule(
    rule_config: dict[str, Any],
    subject_type: str,
    object_type: str,
    relationship_type: str,
) -> bool:
    """Check a specific semantic compatibility rule."""
    rule = rule_config["rule"]

    if rule == "Subject must be from CONSCIOUS classification":
        return subject_type in NodeClassifications.CONSCIOUS

    elif rule == "Both subject and object must be SENTIENT":
        return (
            subject_type in NodeClassifications.SENTIENT
            and object_type in NodeClassifications.SENTIENT
        )

    elif rule == "Both nodes must have PHYSICAL_PRESENCE trait":
        return (
            subject_type in NodeClassifications.PHYSICAL_PRESENCE
            and object_type in NodeClassifications.PHYSICAL_PRESENCE
        )

    elif rule == "Cannot own/possess sentient beings":
        return not (
            relationship_type in ["OWNS", "POSSESSES"]
            and object_type in NodeClassifications.SENTIENT
        )

    elif rule == "Subject must be CONSCIOUS":
        return subject_type in NodeClassifications.CONSCIOUS

    return True


def get_relationship_suggestions(
    subject_type: str, object_type: str
) -> list[tuple[str, str]]:
    """Get suggested valid relationships for a node type pair with explanations."""
    suggestions = []

    valid_relationships = get_all_valid_relationships_for_node_pair(
        subject_type, object_type
    )

    for rel_type in valid_relationships:
        constraint = RELATIONSHIP_CONSTRAINTS[rel_type]
        description = constraint.get("description", "No description available")
        suggestions.append((rel_type, description))

    return suggestions


def validate_relationship_semantics(
    subject_type: str, relationship_type: str, object_type: str
) -> tuple[bool, list[str]]:
    """Comprehensive relationship validation with detailed error messages."""
    errors = []

    # Check if relationship type exists
    if relationship_type not in RELATIONSHIP_CONSTRAINTS:
        errors.append(f"Unknown relationship type: {relationship_type}")
        return False, errors

    # Check if node types are valid
    if subject_type not in NODE_LABELS:
        errors.append(f"Invalid subject node type: {subject_type}")
    if object_type not in NODE_LABELS:
        errors.append(f"Invalid object node type: {object_type}")

    if errors:  # Don't proceed if node types are invalid
        return False, errors

    constraints = RELATIONSHIP_CONSTRAINTS[relationship_type]

    # Check subject type constraints
    valid_subjects = constraints.get("valid_subject_types", set())
    if subject_type not in valid_subjects:
        errors.append(
            f"Invalid subject type '{subject_type}' for relationship '{relationship_type}'. "
            f"Valid subjects: {sorted(valid_subjects)}"
        )

    # Check object type constraints
    valid_objects = constraints.get("valid_object_types", set())
    if object_type not in valid_objects:
        errors.append(
            f"Invalid object type '{object_type}' for relationship '{relationship_type}'. "
            f"Valid objects: {sorted(valid_objects)}"
        )

    # Check invalid combinations
    invalid_combos = constraints.get("invalid_combinations", [])
    if (subject_type, object_type) in invalid_combos:
        errors.append(
            f"Invalid combination: {subject_type} {relationship_type} {object_type} "
            f"is explicitly forbidden"
        )

    # Check semantic rules
    for _rule_name, rule_config in SEMANTIC_COMPATIBILITY_RULES.items():
        if relationship_type in rule_config["relationships"]:
            if not _check_semantic_rule(
                rule_config, subject_type, object_type, relationship_type
            ):
                errors.append(rule_config["violation_message"])

    return len(errors) == 0, errors
