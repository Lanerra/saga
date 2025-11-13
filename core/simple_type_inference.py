# core/simple_type_inference.py
"""
Simple keyword-based type inference for SAGA knowledge graph entities.

This module provides straightforward type inference using keyword matching
without the complexity of ML-inspired pattern learning systems.
"""

import structlog

logger = structlog.get_logger(__name__)


def infer_entity_type(
    name: str = "",
    category: str = "",
    description: str = "",
    existing_type: str = "",
) -> str:
    """
    Infer node type using simple keyword-based pattern matching.

    Args:
        name: Entity name
        category: Entity category (if provided)
        description: Entity description (if provided)
        existing_type: Already proposed type (if valid, will be used)

    Returns:
        Inferred node type string
    """
    if not name or not name.strip():
        return "Entity"

    if existing_type and _is_valid_type(existing_type):
        return existing_type

    entity_name = name.lower().strip()
    category_lower = category.lower().strip() if category else ""

    # Category-based inference (highest priority)
    if category_lower:
        category_mapping = {
            "locations": "Location",
            "location": "Location",
            "factions": "Faction",
            "faction": "Faction",
            "characters": "Character",
            "character": "Character",
            "people": "Person",
            "person": "Person",
            "creatures": "Creature",
            "creature": "Creature",
            "objects": "Object",
            "object": "Object",
            "items": "Item",
            "item": "Item",
            "artifacts": "Artifact",
            "artifact": "Artifact",
            "documents": "Document",
            "document": "Document",
            "events": "Event",
            "event": "Event",
            "systems": "System",
            "system": "System",
            "lore": "Lore",
            "magic": "Magic",
            "technology": "Technology",
            "religion": "Religion",
            "organizations": "Organization",
            "organization": "Organization",
            "structures": "Structure",
            "structure": "Structure",
            "regions": "Region",
            "region": "Region",
            "territories": "Territory",
            "territory": "Territory",
            "landmarks": "Landmark",
            "landmark": "Landmark",
            "settlements": "Settlement",
            "settlement": "Settlement",
            "guilds": "Guild",
            "guild": "Guild",
            "houses": "House",
            "house": "House",
            "orders": "Order",
            "order": "Order",
            "councils": "Council",
            "council": "Council",
            "roles": "Role",
            "role": "Role",
            "ranks": "Rank",
            "rank": "Rank",
            "traditions": "Tradition",
            "tradition": "Tradition",
            "languages": "Language",
            "language": "Language",
            "deities": "Deity",
            "deity": "Deity",
            "spirits": "Spirit",
            "spirit": "Spirit",
            "concepts": "Concept",
            "concept": "Concept",
            "laws": "Law",
            "law": "Law",
            "skills": "Skill",
            "skill": "Skill",
            "traits": "Trait",
            "trait": "Trait",
            "resources": "Resource",
            "resource": "Resource",
            "materials": "Material",
            "material": "Material",
            "currency": "Currency",
            "trade": "Trade",
            "food": "Food",
        }

        if category_lower in category_mapping:
            return category_mapping[category_lower]

        # Partial matching
        for cat_key, node_type in category_mapping.items():
            if cat_key in category_lower or category_lower in cat_key:
                return node_type

    # Name-based pattern inference
    if entity_name:
        # Character patterns
        if any(
            title in entity_name
            for title in [
                "dr.",
                "doctor",
                "captain",
                "commander",
                "sir",
                "lady",
                "lord",
                "professor",
                "general",
            ]
        ):
            return "Character"

        # Location patterns
        if any(
            suffix in entity_name
            for suffix in [
                "city",
                "town",
                "village",
                "castle",
                "tower",
                "forest",
                "mountain",
                "valley",
                "river",
                "lake",
                "sea",
                "ocean",
                "desert",
                "plains",
            ]
        ):
            return "Location"

        # Creature patterns
        if any(
            creature in entity_name
            for creature in [
                "dragon",
                "orc",
                "goblin",
                "wolf",
                "bear",
                "beast",
                "troll",
                "demon",
                "angel",
            ]
        ):
            return "Creature"

        # Object patterns
        if any(
            obj in entity_name
            for obj in [
                "sword",
                "shield",
                "bow",
                "staff",
                "ring",
                "crown",
                "artifact",
                "weapon",
                "armor",
                "blade",
            ]
        ):
            return "Object"

    # Category substring patterns
    if category_lower:
        if any(
            word in category_lower
            for word in ["creature", "beast", "monster", "animal"]
        ):
            return "Creature"
        elif any(
            word in category_lower
            for word in ["character", "person", "human", "humanoid"]
        ):
            return "Character"
        elif any(
            word in category_lower for word in ["location", "place", "area"]
        ):
            return "Location"
        elif any(
            word in category_lower for word in ["structure", "building", "construct"]
        ):
            return "Structure"
        elif any(
            word in category_lower for word in ["weapon", "tool", "item", "equipment"]
        ):
            return "Object"

    return "Entity"


def _is_valid_type(type_name: str) -> bool:
    """Check if a type name is valid according to kg_constants."""
    try:
        import models.kg_constants

        return type_name in models.kg_constants.NODE_LABELS
    except ImportError:
        common_types = {
            "Character",
            "Location",
            "Object",
            "Event",
            "Concept",
            "Entity",
            "ValueNode",
            "Creature",
            "Structure",
            "Region",
            "Person",
            "Faction",
            "Item",
            "Artifact",
        }
        return type_name in common_types


def infer_subject_type(subject_info: dict) -> str:
    """
    Infer subject type from entity information dictionary.

    Args:
        subject_info: Dictionary containing:
            - name: Entity name
            - type: Proposed type (optional)
            - category: Category (optional)
            - description: Description (optional)

    Returns:
        Inferred node type string
    """
    if not subject_info:
        logger.warning("Empty subject_info provided")
        return "Entity"

    return infer_entity_type(
        name=subject_info.get("name", ""),
        category=subject_info.get("category", ""),
        description=subject_info.get("description", ""),
        existing_type=subject_info.get("type", ""),
    )


def infer_object_type(object_info: dict, is_literal: bool = False) -> str:
    """
    Infer object type from entity information dictionary.

    Args:
        object_info: Dictionary containing entity information
        is_literal: If True, returns ValueNode for literal values

    Returns:
        Inferred node type string
    """
    if is_literal:
        return "ValueNode"

    if not object_info:
        logger.warning("Empty object_info provided")
        return "Entity"

    return infer_entity_type(
        name=object_info.get("name", ""),
        category=object_info.get("category", ""),
        description=object_info.get("description", ""),
        existing_type=object_info.get("type", ""),
    )
