# core/langgraph/nodes/commit_entity_conversion.py
"""
Entity conversion helpers: ExtractedEntity -> CharacterProfile/WorldItem.

Extracted from commit_node.py as part of the module-split refactor (I4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from core.schema_validator import canonicalize_entity_type_for_persistence
from models.kg_models import CharacterProfile, WorldItem
from utils import classify_category_label
from utils.text_processing import validate_and_filter_traits

if TYPE_CHECKING:
    from core.langgraph.state import ExtractedEntity


logger = structlog.get_logger(__name__)


def _explicit_entity_id(entity: ExtractedEntity) -> str:
    if "id" not in entity.attributes:
        return ""
    identifier = entity.attributes["id"]
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("Explicit entity ID must be a nonblank string")
    return identifier


def _convert_to_character_profiles(
    entities: list[ExtractedEntity],
    name_mappings: dict[str, str],
    chapter: int,
) -> list[CharacterProfile]:
    """
    Convert ExtractedEntity instances to CharacterProfile models.

    This function bridges the LangGraph state model (ExtractedEntity) with
    the existing SAGA model (CharacterProfile) for persistence.

    Args:
        entities: List of character ExtractedEntity instances
        name_mappings: Dict mapping extracted names to deduplicated names
        chapter: Current chapter number

    Returns:
        List of CharacterProfile models ready for persistence
    """
    profiles = []

    for entity in entities:
        # Use deduplicated name
        final_name = name_mappings.get(entity.name, entity.name)

        # Extract and validate traits from attributes
        raw_traits = entity.attributes.get("traits", [])
        traits = validate_and_filter_traits(raw_traits)

        if len(traits) != len(raw_traits):
            logger.warning(
                "_extract_character_profiles_from_entities: filtered invalid traits",
                character=final_name,
                original_count=len(raw_traits),
                filtered_count=len(traits),
            )

        # Extract status
        status = entity.attributes.get("status", "Unknown")

        # Extract relationships
        relationships = entity.attributes.get("relationships", {})

        profiles.append(
            CharacterProfile(
                name=final_name,
                id=_explicit_entity_id(entity),
                personality_description=entity.description,
                traits=traits,
                status=status if isinstance(status, str) else "Unknown",
                relationships=relationships,
                created_chapter=entity.first_appearance_chapter,
                is_provisional=False,  # Entities from finalized draft are not provisional
                updates={},  # Empty updates for new extraction
            )
        )

    return profiles


def _convert_to_world_items(
    entities: list[ExtractedEntity],
    id_mappings: dict[str, str],
    chapter: int,
) -> list[WorldItem]:
    """
    Convert ExtractedEntity instances to WorldItem models.

    This function bridges the LangGraph state model (ExtractedEntity) with
    the existing SAGA model (WorldItem) for persistence.

    Args:
        entities: List of world item ExtractedEntity instances
        id_mappings: Dict mapping extracted names to deduplicated IDs
        chapter: Current chapter number

    Returns:
        List of WorldItem models ready for persistence
    """
    items = []

    for entity in entities:
        # Use deduplicated ID
        final_id = _explicit_entity_id(entity) if "id" in entity.attributes else id_mappings.get(entity.name, "")

        # Use the category from attributes (preserves specific type like "artifact", "document")
        # The ExtractedEntity validator automatically stores the original type here before normalization
        category = entity.attributes.get("category", entity.type.lower() if entity.type else "")
        if not isinstance(category, str) or classify_category_label(category) != canonicalize_entity_type_for_persistence(entity.type):
            raise ValueError("World entity category conflicts with canonical label")

        # Extract structured fields
        goals = entity.attributes.get("goals", [])
        rules = entity.attributes.get("rules", [])
        key_elements = entity.attributes.get("key_elements", [])

        # Ensure these are lists
        if not isinstance(goals, list):
            goals = [str(goals)] if goals else []
        if not isinstance(rules, list):
            rules = [str(rules)] if rules else []
        if not isinstance(key_elements, list):
            key_elements = [str(key_elements)] if key_elements else []

        # Collect additional properties
        additional_properties = {k: v for k, v in entity.attributes.items() if k not in {"category", "id", "goals", "rules", "key_elements"}}

        items.append(
            WorldItem(
                id=final_id,
                category=category,
                name=entity.name,
                description=entity.description,
                goals=goals,
                rules=rules,
                key_elements=key_elements,
                traits=[],  # Traits typically not used for world items
                created_chapter=entity.first_appearance_chapter,
                is_provisional=False,
                additional_properties=additional_properties,
            )
        )

    return items
