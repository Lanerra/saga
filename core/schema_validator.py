"""Schema validation utilities for the knowledge graph."""

import logging
from typing import Any

from models.kg_models import CharacterProfile, WorldItem

logger = logging.getLogger(__name__)


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
        elif not label[0].isupper():
            errors.append(f"Node label '{label}' should start with an uppercase letter")
        elif not label.isalnum():
            errors.append(
                f"Node label '{label}' should only contain alphanumeric characters"
            )

    return errors


def validate_relationship_types(rel_types: list[str]) -> list[str]:
    """
    Validate relationship types against the predefined narrative taxonomy.

    Returns a list of validation errors. Empty list means valid.
    """
    import kg_constants
    
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
        if rel_type not in kg_constants.RELATIONSHIP_TYPES:
            # Check if it can be normalized to a valid type
            from data_access.kg_queries import normalize_relationship_type
            normalized = normalize_relationship_type(rel_type)
            
            if normalized in kg_constants.RELATIONSHIP_TYPES:
                # It's normalizable - suggest normalization rather than error
                logger.info(f"Relationship type '{rel_type}' can be normalized to '{normalized}'")
            else:
                errors.append(f"Relationship type '{rel_type}' is not in the predefined narrative taxonomy")

    return errors


def suggest_relationship_normalization(rel_types: list[str]) -> dict[str, str]:
    """
    Suggest normalizations for relationship types that don't match the predefined taxonomy.
    
    Returns a dict mapping original -> suggested canonical form.
    """
    import kg_constants
    from data_access.kg_queries import normalize_relationship_type
    
    suggestions = {}
    
    for rel_type in rel_types:
        if isinstance(rel_type, str) and rel_type.strip():
            if rel_type not in kg_constants.RELATIONSHIP_TYPES:
                normalized = normalize_relationship_type(rel_type)
                if normalized in kg_constants.RELATIONSHIP_TYPES and normalized != rel_type:
                    suggestions[rel_type] = normalized
                    
    return suggestions


def validate_character_profile(profile: CharacterProfile) -> list[str]:
    """
    Validate a CharacterProfile object.

    Returns a list of validation errors. Empty list means valid.
    """
    errors = []
    # Check name
    if not profile.name or not profile.name.strip():
        errors.append("CharacterProfile name cannot be empty")
    # Check traits
    if not isinstance(profile.traits, list):
        errors.append("CharacterProfile traits must be a list")
    else:
        for trait in profile.traits:
            if not isinstance(trait, str) or not trait.strip():
                errors.append("CharacterProfile traits must be non-empty strings")
    # Check relationships (just dict)
    if not isinstance(profile.relationships, dict):
        errors.append("CharacterProfile relationships must be a dict")
    # Check status
    if not isinstance(profile.status, str):
        errors.append("CharacterProfile status must be a string")
    return errors


def validate_world_item(item: WorldItem) -> list[str]:
    """
    Validate a WorldItem object.

    Returns a list of validation errors. Empty list means valid.
    """
    errors = []
    if not item.name or not item.name.strip():
        errors.append("WorldItem name cannot be empty")
    if not item.category or not item.category.strip():
        errors.append("WorldItem category cannot be empty")
    if not isinstance(item.description, str):
        errors.append("WorldItem description must be a string")
    if not isinstance(item.goals, list):
        errors.append("WorldItem goals must be a list")
    else:
        for goal in item.goals:
            if not isinstance(goal, str) or not goal.strip():
                errors.append("WorldItem goals must be non-empty strings")
    # Check rules
    if not isinstance(item.rules, list):
        errors.append("WorldItem rules must be a list")
    else:
        for rule in item.rules:
            if not isinstance(rule, str) or not rule.strip():
                errors.append("WorldItem rules must be non-empty strings")
    # Check key_elements
    if not isinstance(item.key_elements, list):
        errors.append("WorldItem key_elements must be a list")
    else:
        for element in item.key_elements:
            if not isinstance(element, str) or not element.strip():
                errors.append("WorldItem key_elements must be non-empty strings")
    # Check traits
    if not isinstance(item.traits, list):
        errors.append("WorldItem traits must be a list")
    else:
        for trait in item.traits:
            if not isinstance(trait, str) or not trait.strip():
                errors.append("WorldItem traits must be non-empty strings")
    # Check additional_properties
    if not isinstance(item.additional_properties, dict):
        errors.append("WorldItem additional_properties must be a dict")
    return errors
