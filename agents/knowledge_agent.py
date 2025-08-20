# knowledge_agent.py
import asyncio
import json
import logging
import re
from typing import Any

from async_lru import alru_cache  # type: ignore
from jinja2 import Template

import config
import utils  # Ensure utils is imported for normalize_trait_name
from core.db_manager import neo4j_manager
from core.llm_interface import llm_service
from core.schema_validator import validate_node_labels, validate_relationship_types
from data_access import (
    character_queries,
    kg_queries,
    plot_queries,
    world_queries,
)
from models.kg_models import CharacterProfile, WorldItem
from core.schema_validator import validate_kg_object
from core.knowledge_graph_service import knowledge_graph_service
from parsing_utils import (
    parse_rdf_triples_with_rdflib,
)
from prompt_renderer import render_prompt

logger = logging.getLogger(__name__)


@alru_cache(maxsize=config.SUMMARY_CACHE_SIZE)
async def _llm_summarize_full_chapter_text(
    chapter_text: str, chapter_number: int
) -> tuple[str, dict[str, int] | None]:
    """Summarize full chapter text via the configured LLM."""
    prompt = render_prompt(
        "knowledge_agent/chapter_summary.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "chapter_number": chapter_number,
            "chapter_text": chapter_text,
        },
    )
    summary, usage_data = await llm_service.async_call_llm(
        model_name=config.SMALL_MODEL,  # Using SMALL_MODEL for summarization
        prompt=prompt,
        temperature=config.Temperatures.SUMMARY,
        max_tokens=config.MAX_SUMMARY_TOKENS,  # Should be small for 1-3 sentences
        stream_to_disk=False,
        frequency_penalty=config.FREQUENCY_PENALTY_SUMMARY,
        presence_penalty=config.PRESENCE_PENALTY_SUMMARY,
        auto_clean_response=True,
    )
    summary_text = summary.strip()
    if summary_text:
        try:
            parsed = json.loads(summary_text)
            if isinstance(parsed, dict):
                summary_text = parsed.get("summary", "")
        except json.JSONDecodeError:
            logger.debug(f"Summary for chapter {chapter_number} was not a JSON object.")
    return summary_text, usage_data


# Prompt template for entity resolution, embedded to avoid new file dependency
ENTITY_RESOLUTION_PROMPT_TEMPLATE = """/no_think
You are an expert knowledge graph analyst for a creative writing project. Your task is to determine if two entities from the narrative's knowledge graph are referring to the same canonical thing based on their names, properties, and relationships.

**Entity 1 Details:**
- Name: {{ entity1.name }}
- Labels: {{ entity1.labels }}
- Properties:
{{ entity1.properties | tojson(indent=2) }}
- Key Relationships (up to 10):
{% if entity1.relationships %}
{% for rel in entity1.relationships %}
  - Related to '{{ rel.other_node_name }}' (Labels: {{ rel.other_node_labels }}) via relationship of type '{{ rel.rel_type }}'
{% endfor %}
{% else %}
  - No relationships found.
{% endif %}

**Entity 2 Details:**
- Name: {{ entity2.name }}
- Labels: {{ entity2.labels }}
- Properties:
{{ entity2.properties | tojson(indent=2) }}
- Key Relationships (up to 10):
{% if entity2.relationships %}
{% for rel in entity2.relationships %}
  - Related to '{{ rel.other_node_name }}' (Labels: {{ rel.other_node_labels }}) via relationship of type '{{ rel.rel_type }}'
{% endfor %}
{% else %}
  - No relationships found.
{% endif %}

**Analysis Task:**
Based on all the provided context, including name similarity, properties (like descriptions), and shared relationships, are "Entity 1" and "Entity 2" the same entity within the story's canon? For example, "The Locket" and "The Pendant" might be the same item, or "The Shattered Veil" and "Shattered Veil" are likely the same faction.

**Response Format:**
Respond in JSON format only, with no other text, commentary, or markdown. Your entire response must be a single, valid JSON object with the following structure:
{
  "is_same_entity": boolean,
  "confidence_score": float (from 0.0 to 1.0, representing your certainty),
  "reason": "A brief explanation for your decision."
}
"""

# Prompt template for dynamic relationship resolution
DYNAMIC_REL_RESOLUTION_PROMPT_TEMPLATE = """/no_think
You analyze a relationship from the novel's knowledge graph and provide a
single, canonical predicate name in ALL_CAPS_WITH_UNDERSCORES describing the
relationship between the subject and object.

Subject: {{ subject }} ({{ subject_labels }})
Object: {{ object }} ({{ object_labels }})
Existing Type: {{ type }}
Subject Description: {{ subject_desc }}
Object Description: {{ object_desc }}

Respond with only the predicate string, no extra words.
"""

# Moved from kg_maintainer/parsing.py
CHAR_UPDATE_KEY_MAP = {
    "desc": "description",
    "description": "description",
    "traits": "traits",
    "relationships": "relationships",
    "status": "status",
    "modification proposal": "modification_proposal",
    # Add other keys LLM might produce, mapping to CharacterProfile fields
    # e.g. "aliases": "aliases" if LLM provides aliases as a list
}

CHAR_UPDATE_LIST_INTERNAL_KEYS = [
    "traits",
    "relationships",
    "aliases",
]  # Added aliases as example

WORLD_UPDATE_DETAIL_KEY_MAP = {
    # Ensure these keys match what LLM will produce in JSON
    "desc": "description",
    "description": "description",
    "atmosphere": "atmosphere",  # Added from original example
    "goals": "goals",
    "rules": "rules",
    "key elements": "key_elements",
    "traits": "traits",  # Ensure this is a list if LLM provides a string
    "modification proposal": "modification_proposal",
    # NOTE: elaborations/elaboration will be handled specially in processing
}
WORLD_UPDATE_DETAIL_LIST_INTERNAL_KEYS = [
    "goals",
    "rules",
    "key_elements",
    "traits",
]  # Ensure these are lists


def _normalize_attributes(
    attributes_dict: dict[str, Any],
    key_map: dict[str, str],
    list_keys: list[str],
) -> dict[str, Any]:
    normalized_attrs: dict[str, Any] = {}
    if not isinstance(attributes_dict, dict):
        logger.warning(
            "Input to _normalize_attributes was not a dict: %s",
            type(attributes_dict),
        )
        return {}

    for key, value in attributes_dict.items():
        # Normalize the key from LLM JSON for matching against key_map
        normalized_llm_key = key.lower().replace(" ", "_")
        mapped_key = key_map.get(
            normalized_llm_key, normalized_llm_key
        )  # Use normalized if not in map

        if mapped_key in list_keys:
            if isinstance(value, list):
                normalized_attrs[mapped_key] = value
            elif isinstance(value, dict):
                normalized_attrs[mapped_key] = value
            elif isinstance(value, str):
                normalized_attrs[mapped_key] = [
                    v.strip() for v in value.split(",") if v.strip()
                ]
            elif value is None:
                normalized_attrs[mapped_key] = []
            else:
                normalized_attrs[mapped_key] = [value]
        else:
            normalized_attrs[mapped_key] = value

    # Ensure all list_keys are present and are lists in the final output
    for l_key in list_keys:
        if l_key not in normalized_attrs:
            normalized_attrs[l_key] = []
        elif not isinstance(normalized_attrs[l_key], list):
            if isinstance(normalized_attrs[l_key], dict):
                continue
            if (
                normalized_attrs[l_key] is not None
                and str(normalized_attrs[l_key]).strip()
            ):
                normalized_attrs[l_key] = [str(normalized_attrs[l_key])]
            else:
                normalized_attrs[l_key] = []

    return normalized_attrs


def parse_unified_character_updates(
    json_text_block: str, chapter_number: int
) -> dict[str, CharacterProfile]:
    """Parse character update JSON provided by LLM."""
    char_updates: dict[str, CharacterProfile] = {}
    if not json_text_block.strip():
        return char_updates

    try:
        # LLM is expected to output a dict where keys are character names
        # and values are dicts of their attributes.
        parsed_data = json.loads(json_text_block)
        if not isinstance(parsed_data, dict):
            logger.error(
                "Character updates JSON was not a dictionary. Received: %s",
                type(parsed_data),
            )
            return char_updates
    except json.JSONDecodeError as e:
        logger.error(
            "Failed to parse character updates JSON: %s. Input: %s...",
            e,
            json_text_block[:500],
        )
        return char_updates

    for char_name, char_attributes_llm in parsed_data.items():
        if not char_name or not isinstance(char_attributes_llm, dict):
            logger.warning(
                "Skipping character with invalid name or attributes: Name='%s',"
                " Attrs_Type='%s'",
                char_name,
                type(char_attributes_llm),
            )
            continue

        # Normalize keys from LLM (e.g. "desc" to "description")
        # and ensure list types
        processed_char_attributes = _normalize_attributes(
            char_attributes_llm, CHAR_UPDATE_KEY_MAP, CHAR_UPDATE_LIST_INTERNAL_KEYS
        )

        # Canonicalize trait names for consistency
        traits_val = processed_char_attributes.get("traits", [])
        if isinstance(traits_val, list):
            processed_char_attributes["traits"] = [
                utils.normalize_trait_name(t)
                for t in traits_val
                if isinstance(t, str) and utils.normalize_trait_name(t)
            ]

        # Handle relationships if they need structuring from list to dict
        # Assuming LLM provides relationships as a list of strings
        # like "Target: Detail" or just "Target"
        # Or ideally, as a dict: {"Target": "Detail"}
        rels_val = processed_char_attributes.get("relationships")
        if isinstance(rels_val, list):
            rels_list = rels_val
            rels_dict: dict[str, str] = {}
            for rel_entry in rels_list:
                if isinstance(rel_entry, str):
                    if ":" in rel_entry:
                        parts = rel_entry.split(":", 1)
                        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                            rels_dict[parts[0].strip()] = parts[1].strip()
                        elif parts[0].strip():  # If only name is there before colon
                            rels_dict[parts[0].strip()] = "related"
                    elif rel_entry.strip():  # No colon, just a name
                        rels_dict[rel_entry.strip()] = "related"
                elif isinstance(
                    rel_entry, dict
                ):  # If LLM sends [{"name": "X", "detail": "Y"}]
                    target_name = rel_entry.get("name")
                    detail = rel_entry.get("detail", "related")
                    if (
                        target_name
                        and isinstance(target_name, str)
                        and target_name.strip()
                    ):
                        rels_dict[target_name] = detail

            processed_char_attributes["relationships"] = rels_dict
        elif isinstance(rels_val, dict):
            processed_char_attributes["relationships"] = {
                str(k): str(v) for k, v in rels_val.items()
            }
        else:  # Ensure it's always a dict
            processed_char_attributes["relationships"] = {}

        dev_key_standard = f"development_in_chapter_{chapter_number}"
        # If LLM includes this key (even with different casing/spacing), it will be normalized by _normalize_attributes
        # if dev_key_standard is in CHAR_UPDATE_KEY_MAP. For now, handle it explicitly.
        specific_dev_key_from_llm = next(
            (
                k
                for k in processed_char_attributes
                if k.lower().replace(" ", "_") == dev_key_standard
            ),
            None,
        )

        if specific_dev_key_from_llm and specific_dev_key_from_llm != dev_key_standard:
            processed_char_attributes[dev_key_standard] = processed_char_attributes.pop(
                specific_dev_key_from_llm
            )

        # Add default development note if no specific one and other attributes exist
        has_other_meaningful_attrs = any(
            k not in ["modification_proposal", dev_key_standard] and v
            for k, v in processed_char_attributes.items()
        )
        if (
            not processed_char_attributes.get(dev_key_standard)
            and has_other_meaningful_attrs
        ):
            processed_char_attributes[dev_key_standard] = (
                f"Character '{char_name}' details updated in Chapter {chapter_number}."
            )

        try:
            char_updates[char_name] = CharacterProfile.from_dict(
                char_name, processed_char_attributes
            )
        except Exception as e:
            logger.error(
                f"Error creating CharacterProfile for '{char_name}': {e}. Attributes: {processed_char_attributes}",
                exc_info=True,
            )

    return char_updates


def parse_unified_world_updates(
    json_text_block: str, chapter_number: int
) -> dict[str, dict[str, WorldItem]]:
    """Parse world update JSON provided by LLM."""
    world_updates: dict[str, dict[str, WorldItem]] = {}
    if not json_text_block.strip():
        return world_updates

    try:
        # LLM is expected to output a dict where keys are category display names (e.g., "Locations")
        # and values are dicts of item names to their attribute dicts.
        parsed_data = json.loads(json_text_block)
        if not isinstance(parsed_data, dict):
            logger.error(
                f"World updates JSON was not a dictionary. Received: {type(parsed_data)}"
            )
            return world_updates
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse world updates JSON: {e}. Input: {json_text_block[:500]}..."
        )
        return world_updates

    results: dict[str, dict[str, WorldItem]] = {}
    for category_name_llm, items_llm in parsed_data.items():
        if not isinstance(items_llm, dict):
            logger.warning(
                f"Skipping category '{category_name_llm}' as its content is not a dictionary of items."
            )
            continue

        # category_name_llm is e.g. "Locations", "Faction Alpha". This is used as the .category for WorldItem
        # The WorldItem model itself might normalize this for ID generation.

        category_dict_by_item_name: dict[str, WorldItem] = {}
        elaboration_key_standard = f"elaboration_in_chapter_{chapter_number}"

        if (
            category_name_llm.lower() == "overview"
            or category_name_llm.lower() == "_overview_"
        ):
            # Overview is a single item, its details are directly in items_llm
            processed_overview_details = _normalize_attributes(
                items_llm,
                WORLD_UPDATE_DETAIL_KEY_MAP,
                WORLD_UPDATE_DETAIL_LIST_INTERNAL_KEYS,
            )

            # Handle common LLM output variations for elaborations
            if "elaborations" in processed_overview_details:
                processed_overview_details[elaboration_key_standard] = (
                    processed_overview_details.pop("elaborations")
                )
            elif "elaboration" in processed_overview_details:
                processed_overview_details[elaboration_key_standard] = (
                    processed_overview_details.pop("elaboration")
                )
            if any(k != "modification_proposal" for k in processed_overview_details):
                # check if any meaningful data
                # Add default elaboration if not present
                if not processed_overview_details.get(elaboration_key_standard):
                    processed_overview_details[elaboration_key_standard] = (
                        f"Overall world overview updated in Chapter {chapter_number}."
                    )
                try:
                    # For overview, item_name is fixed (e.g., "_overview_")
                    overview_item = WorldItem.from_dict(
                        category_name_llm,
                        "_overview_",
                        processed_overview_details,
                    )
                    results.setdefault(category_name_llm, {})["_overview_"] = (
                        overview_item
                    )
                except Exception as e:
                    logger.error(
                        "Error creating WorldItem for overview in category '%s': %s",
                        category_name_llm,
                        e,
                        exc_info=True,
                    )
        else:  # Regular category with multiple items
            for item_name_llm, item_attributes_llm in items_llm.items():
                if not item_name_llm or not isinstance(item_attributes_llm, dict):
                    # Check if this might be a property that was incorrectly placed at the item level
                    # This can happen when LLM outputs properties at the category level instead of item level
                    if item_name_llm and isinstance(item_name_llm, str):
                        # Check if the "item name" is actually a known property name
                        normalized_key = item_name_llm.lower().replace(" ", "_")
                        # Include common elaboration variations that should be treated as properties, not items
                        known_property_names = (
                            set(WORLD_UPDATE_DETAIL_KEY_MAP.keys())
                            | set(WORLD_UPDATE_DETAIL_LIST_INTERNAL_KEYS)
                            | {"elaborations", "elaboration"}
                        )
                        if normalized_key in known_property_names:
                            logger.debug(
                                "Ignoring property '%s' at item level in category '%s' (likely LLM formatting issue)",
                                item_name_llm,
                                category_name_llm,
                            )
                            continue

                    logger.warning(
                        "Skipping item with invalid name or attributes in "
                        "category '%s': Name='%s'",
                        category_name_llm,
                        item_name_llm,
                    )
                    continue

                processed_item_details = _normalize_attributes(
                    item_attributes_llm,
                    WORLD_UPDATE_DETAIL_KEY_MAP,
                    WORLD_UPDATE_DETAIL_LIST_INTERNAL_KEYS,
                )

                # Handle common LLM output variations for elaborations
                if "elaborations" in processed_item_details:
                    processed_item_details[elaboration_key_standard] = (
                        processed_item_details.pop("elaborations")
                    )
                elif "elaboration" in processed_item_details:
                    processed_item_details[elaboration_key_standard] = (
                        processed_item_details.pop("elaboration")
                    )

                # Add default elaboration if not present and other
                # attributes exist
                has_other_meaningful_item_attrs = any(
                    k
                    not in [
                        "modification_proposal",
                        elaboration_key_standard,
                    ]
                    and v
                    for k, v in processed_item_details.items()
                )
                if (
                    not processed_item_details.get(elaboration_key_standard)
                    and has_other_meaningful_item_attrs
                ):
                    processed_item_details[elaboration_key_standard] = (
                        f"Item '{item_name_llm}' in category '{category_name_llm}' "
                        f"updated in Chapter {chapter_number}."
                    )

                if not category_name_llm or not category_name_llm.strip():
                    logger.warning(
                        "Skipping WorldItem with missing category: %s", item_name_llm
                    )
                    continue

                try:
                    # item_name_llm is the display name from JSON key.
                    # WorldItem stores this as .name and normalizes it for .id
                    # along with category_name_llm.
                    world_item_instance = WorldItem.from_dict(
                        category_name_llm,
                        item_name_llm,
                        processed_item_details,
                    )

                    # Store in this category's dictionary using the item's display name as key.
                    # If LLM provides duplicate item names within the same category, last one wins.
                    category_dict_by_item_name[world_item_instance.name] = (
                        world_item_instance
                    )
                except Exception as e:
                    logger.error(
                        "Error creating WorldItem for '%s' in category '%s': %s",
                        item_name_llm,
                        category_name_llm,
                        e,
                        exc_info=True,
                    )

            if category_dict_by_item_name:
                results[category_name_llm] = category_dict_by_item_name

    return results


# Phase 3 optimization: Native parsing methods that eliminate dict conversion overhead
def parse_unified_character_updates_native(
    json_text_block: str, chapter_number: int
) -> dict[str, CharacterProfile]:
    """
    Native version of character update parsing that creates models directly.
    Eliminates .from_dict() conversion overhead.
    """
    char_updates: dict[str, CharacterProfile] = {}
    if not json_text_block or json_text_block.strip() in ["null", "None", ""]:
        return char_updates

    try:
        parsed_json = json.loads(json_text_block)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse character updates JSON: {e}")
        return char_updates

    for char_name, raw_attributes in parsed_json.items():
        if not isinstance(raw_attributes, dict):
            logger.warning(
                f"Skipping character '{char_name}': attributes not in dict format"
            )
            continue

        processed_char_attributes = {}
        for key, value in raw_attributes.items():
            if key in ["traits", "relationships", "skills"]:
                if isinstance(value, str):
                    processed_char_attributes[key] = [
                        item.strip() for item in value.split(",") if item.strip()
                    ]
                elif isinstance(value, list):
                    processed_char_attributes[key] = [
                        str(item).strip() for item in value if str(item).strip()
                    ]
                else:
                    processed_char_attributes[key] = []
            else:
                processed_char_attributes[key] = value

        try:
            # Create model directly without dict intermediate
            char_updates[char_name] = CharacterProfile(
                name=char_name,
                description=processed_char_attributes.get("description", ""),
                traits=processed_char_attributes.get("traits", []),
                relationships=processed_char_attributes.get("relationships", []),
                skills=processed_char_attributes.get("skills", []),
                status=processed_char_attributes.get("status", "active"),
                # Copy any additional attributes
                **{k: v for k, v in processed_char_attributes.items() 
                   if k not in ["name", "description", "traits", "relationships", "skills", "status"]}
            )
        except Exception as e:
            logger.error(
                f"Error creating CharacterProfile for '{char_name}': {e}. "
                f"Attributes: {processed_char_attributes}",
                exc_info=True,
            )

    return char_updates


def parse_unified_world_updates_native(
    json_text_block: str, chapter_number: int
) -> dict[str, dict[str, WorldItem]]:
    """
    Native version of world update parsing that creates models directly.
    Eliminates .from_dict() conversion overhead.
    """
    world_updates: dict[str, dict[str, WorldItem]] = {}
    if not json_text_block or json_text_block.strip() in ["null", "None", ""]:
        return world_updates

    try:
        parsed_json = json.loads(json_text_block)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse world updates JSON: {e}")
        return world_updates

    for category_name_llm, world_items_in_category in parsed_json.items():
        if not isinstance(world_items_in_category, dict):
            logger.warning(
                f"Skipping category '{category_name_llm}': items not in dict format"
            )
            continue

        category_dict_by_item_name: dict[str, WorldItem] = {}

        for item_name_llm, raw_item_attributes in world_items_in_category.items():
            if not isinstance(raw_item_attributes, dict):
                logger.warning(
                    f"Skipping item '{item_name_llm}' in '{category_name_llm}': "
                    f"attributes not in dict format"
                )
                continue

            try:
                # Create model directly without dict intermediate
                world_item_instance = WorldItem(
                    name=item_name_llm,
                    category=category_name_llm,
                    description=raw_item_attributes.get("description", ""),
                    # Copy any additional attributes
                    **{k: v for k, v in raw_item_attributes.items() 
                       if k not in ["name", "category", "description"]}
                )
                category_dict_by_item_name[item_name_llm] = world_item_instance
            except Exception as e:
                logger.error(
                    f"Error creating WorldItem for '{item_name_llm}' in category '{category_name_llm}': {e}",
                    exc_info=True,
                )

        if category_dict_by_item_name:
            world_updates[category_name_llm] = category_dict_by_item_name

    return world_updates


# Moved from kg_maintainer/merge.py
def initialize_new_character_profile(
    char_name: str, char_update: CharacterProfile, chapter_number: int
) -> CharacterProfile:
    """Create a new character profile from parsed updates."""
    provisional_key = f"source_quality_chapter_{chapter_number}"
    dev_key = f"development_in_chapter_{chapter_number}"
    data = char_update.to_dict()
    new_profile = CharacterProfile(
        name=char_name,
        description=data.get(
            "description",
            f"A character newly introduced in Chapter {chapter_number}.",
        ),
        traits=sorted(
            {t for t in data.get("traits", []) if isinstance(t, str) and t.strip()}
        ),
        relationships=data.get("relationships", {}),
        status=data.get("status", "Newly introduced"),
        updates={
            dev_key: data.get(
                dev_key,
                (f"Character '{char_name}' introduced in Chapter {chapter_number}."),
            )
        },
    )
    if provisional_key in data:
        new_profile.updates[provisional_key] = data[provisional_key]
    return new_profile


def merge_character_profile_updates(
    profiles: dict[str, CharacterProfile],
    updates: dict[str, CharacterProfile],
    chapter_number: int,
    from_flawed_draft: bool,
) -> None:
    """Merge character updates into existing profile dictionary."""
    # Validate all updates before merging
    for name, update in updates.items():
        errors = validate_kg_object(update)
        if errors:
            logger.warning("Invalid CharacterProfile for '%s': %s", name, errors)
    
    provisional_key = f"source_quality_chapter_{chapter_number}"
    for name, update in updates.items():
        data = update.to_dict()
        if from_flawed_draft:
            data[provisional_key] = "provisional_from_unrevised_draft"
        dev_key = f"development_in_chapter_{chapter_number}"
        if name not in profiles:
            profiles[name] = initialize_new_character_profile(
                name, update, chapter_number
            )
            continue
        profile = profiles[name]
        prof_dict = profile.to_dict()
        modified = False
        for key, val in data.items():
            if key in {"modification_proposal", provisional_key} or (
                key.startswith("development_in_chapter_")
            ):
                continue
            if key == "traits" and isinstance(val, list):
                new_traits = sorted(
                    set(profile.traits).union(
                        {t for t in val if isinstance(t, str) and t.strip()}
                    )
                )
                if new_traits != profile.traits:
                    profile.traits = new_traits
                    modified = True
            elif key == "relationships" and isinstance(val, dict):
                for target, rel in val.items():
                    if profile.relationships.get(target) != rel:
                        profile.relationships[target] = rel
                        modified = True
            elif isinstance(val, str) and val.strip() and prof_dict.get(key) != val:
                profile.updates[key] = val
                modified = True
        if dev_key in data and isinstance(data[dev_key], str):
            profile.updates[dev_key] = data[dev_key]
            modified = True
        if from_flawed_draft:
            profile.updates[provisional_key] = "provisional_from_unrevised_draft"
        if modified:
            logger.debug("Profile for %s modified", name)


def merge_world_item_updates(
    world: dict[str, dict[str, WorldItem]],
    updates: dict[str, dict[str, WorldItem]],
    chapter_number: int,
    from_flawed_draft: bool,
) -> None:
    """Merge world item updates into the current world dictionary."""
    # Validate all updates before merging
    for category, cat_updates in updates.items():
        for name, update in cat_updates.items():
            errors = validate_kg_object(update)
            if errors:
                logger.warning("Invalid WorldItem for '%s' in category '%s': %s", name, category, errors)
    
    provisional_key = f"source_quality_chapter_{chapter_number}"
    for category, cat_updates in updates.items():
        if category not in world:
            world[category] = {}
        for name, update in cat_updates.items():
            data = update.to_dict()
            if from_flawed_draft:
                data[provisional_key] = "provisional_from_unrevised_draft"
            if name not in world[category]:
                world[category][name] = update
                world[category][name].additional_properties.setdefault(
                    f"added_in_chapter_{chapter_number}", True
                )
                continue
            item = world[category][name]
            item_props = item.to_dict()
            for key, val in data.items():
                if key in {provisional_key, "modification_proposal"} or (
                    key.startswith(
                        (
                            "updated_in_chapter_",
                            "added_in_chapter_",
                            "source_quality_chapter_",
                        )
                    )
                ):
                    if (
                        key.startswith("elaboration_in_chapter_")
                        and isinstance(val, str)
                        and val.strip()
                    ):
                        # Handle structured fields
                        if key == "description":
                            item.description = val
                        elif key == "goals":
                            item.goals = val if isinstance(val, list) else [val]
                        elif key == "rules":
                            item.rules = val if isinstance(val, list) else [val]
                        elif key == "key_elements":
                            item.key_elements = val if isinstance(val, list) else [val]
                        elif key == "traits":
                            item.traits = val if isinstance(val, list) else [val]
                        else:
                            # Handle additional properties
                            item.additional_properties[key] = val
                    continue
                cur_val = item_props.get(key)
                if isinstance(val, list):
                    # Handle structured fields that are lists
                    if key == "goals":
                        item.goals = list(set(item.goals + val))
                    elif key == "rules":
                        item.rules = list(set(item.rules + val))
                    elif key == "key_elements":
                        item.key_elements = list(set(item.key_elements + val))
                    elif key == "traits":
                        item.traits = list(set(item.traits + val))
                    else:
                        # Handle additional properties that are lists
                        cur_list = item.additional_properties.get(key, [])
                        for elem in val:
                            if elem not in cur_list:
                                cur_list.append(elem)
                        item.additional_properties[key] = cur_list
                elif isinstance(val, dict):
                    # Handle additional properties that are dictionaries
                    sub = item.additional_properties.get(key, {})
                    if not isinstance(sub, dict):
                        sub = {}
                    sub.update(val)
                    item.additional_properties[key] = sub
                elif cur_val != val:
                    # Handle structured fields that are not lists or dicts
                    if key == "description":
                        item.description = val
                    else:
                        # Handle additional properties that are not lists or dicts
                        item.additional_properties[key] = val
            item.additional_properties.setdefault(
                f"updated_in_chapter_{chapter_number}",
                True,
            )


class KnowledgeAgent:
    """High level interface for KG parsing and persistence."""

    def __init__(self, model_name: str = config.KNOWLEDGE_UPDATE_MODEL):
        self.model_name = model_name
        self.node_labels: list[str] = []
        self.relationship_types: list[str] = []
        logger.info(
            "KnowledgeAgent initialized with model for extraction: %s",
            self.model_name,
        )

    async def load_schema_from_db(self):
        """Loads and caches the defined KG schema from the database."""
        self.node_labels = await kg_queries.get_defined_node_labels()
        self.relationship_types = await kg_queries.get_defined_relationship_types()
        logger.info(
            f"Loaded {len(self.node_labels)} node labels and {len(self.relationship_types)} relationship types from DB."
        )

    def parse_character_updates(
        self, text: str, chapter_number: int
    ) -> dict[str, CharacterProfile]:
        """Parse character update text into structured profiles."""
        return parse_unified_character_updates(text, chapter_number)

    def parse_world_updates(
        self, text: str, chapter_number: int
    ) -> dict[str, dict[str, WorldItem]]:
        """Parse world update text into structured items."""
        return parse_unified_world_updates(text, chapter_number)

    # Phase 3 optimization: Native parsing methods
    def parse_character_updates_native(
        self, text: str, chapter_number: int
    ) -> dict[str, CharacterProfile]:
        """Native character update parsing that eliminates dict conversion overhead."""
        return parse_unified_character_updates_native(text, chapter_number)

    def parse_world_updates_native(
        self, text: str, chapter_number: int
    ) -> dict[str, dict[str, WorldItem]]:
        """Native world update parsing that eliminates dict conversion overhead."""
        return parse_unified_world_updates_native(text, chapter_number)

    def merge_updates(
        self,
        current_profiles: dict[str, CharacterProfile],
        current_world: dict[str, dict[str, WorldItem]],
        char_updates_parsed: dict[str, CharacterProfile],
        world_updates_parsed: dict[str, dict[str, WorldItem]],
        chapter_number: int,
        from_flawed_draft: bool = False,
    ) -> None:
        """Merge parsed updates into existing state (Python objects)."""
        merge_character_profile_updates(
            current_profiles, char_updates_parsed, chapter_number, from_flawed_draft
        )
        merge_world_item_updates(
            current_world, world_updates_parsed, chapter_number, from_flawed_draft
        )

    async def persist_profiles(
        self,
        profiles_to_persist: dict[str, CharacterProfile],
        chapter_number_for_delta: int,
        full_sync: bool = False,
    ) -> None:
        """Persist character profiles to Neo4j."""
        await character_queries.sync_characters(
            profiles_to_persist, chapter_number_for_delta, full_sync=full_sync
        )

    async def persist_world(
        self,
        world_items_to_persist: dict[str, dict[str, WorldItem]],
        chapter_number_for_delta: int,
        full_sync: bool = False,
    ) -> None:
        """Persist world elements to Neo4j."""
        await world_queries.sync_world_items(
            world_items_to_persist, chapter_number_for_delta, full_sync=full_sync
        )

    async def add_plot_point(self, description: str, prev_plot_point_id: str) -> str:
        """Persist a new plot point and link it in sequence."""
        return await plot_queries.append_plot_point(description, prev_plot_point_id)

    async def summarize_chapter(
        self, chapter_text: str | None, chapter_number: int
    ) -> tuple[str | None, dict[str, int] | None]:
        if (
            not chapter_text
            or len(chapter_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH // 2
        ):
            logger.warning(
                "Chapter %s text too short for summarization (%d chars, min_req for meaningful summary: %d).",
                chapter_number,
                len(chapter_text or ""),
                config.MIN_ACCEPTABLE_DRAFT_LENGTH // 2,
            )
            return None, None

        try:
            cleaned_summary, usage = await _llm_summarize_full_chapter_text(
                chapter_text, chapter_number
            )
            if cleaned_summary:
                logger.info(
                    "Generated summary for ch %d: '%s...'",
                    chapter_number,
                    cleaned_summary[:100].strip(),
                )
                return cleaned_summary, usage
            logger.warning("LLM returned empty summary for ch %d.", chapter_number)
            return None, usage
        except Exception as e:
            logger.error(
                f"Error during chapter summarization for ch {chapter_number}: {e}",
                exc_info=True,
            )
            return None, None

    async def _llm_extract_updates(
        self,
        plot_outline: dict[str, Any],
        chapter_text: str,
        chapter_number: int,
    ) -> tuple[str, dict[str, int] | None]:
        """Call the LLM to extract structured updates from chapter text, including typed entities in triples."""
        protagonist = plot_outline.get(
            "protagonist_name", config.DEFAULT_PROTAGONIST_NAME
        )

        prompt = render_prompt(
            "knowledge_agent/extract_updates.j2",
            {
                "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
                "protagonist": protagonist,
                "chapter_number": chapter_number,
                "novel_title": plot_outline.get("title", "Untitled Novel"),
                "novel_genre": plot_outline.get("genre", "Unknown"),
                "chapter_text": chapter_text,
                "available_node_labels": self.node_labels,
                "available_relationship_types": self.relationship_types,
            },
        )

        try:
            text, usage = await llm_service.async_call_llm(
                model_name=self.model_name,
                prompt=prompt,
                temperature=config.Temperatures.KG_EXTRACTION,
                max_tokens=config.MAX_KG_TRIPLE_TOKENS,
                allow_fallback=True,
                stream_to_disk=False,
                frequency_penalty=config.FREQUENCY_PENALTY_KG_EXTRACTION,
                presence_penalty=config.PRESENCE_PENALTY_KG_EXTRACTION,
                auto_clean_response=True,
            )
            return text, usage
        except Exception as e:
            logger.error(f"LLM call for KG extraction failed: {e}", exc_info=True)
            return "", None

    async def extract_and_merge_knowledge(
        self,
        plot_outline: dict[str, Any],
        character_profiles: dict[str, CharacterProfile],
        world_building: dict[str, dict[str, WorldItem]],
        chapter_number: int,
        chapter_text: str,
        is_from_flawed_draft: bool = False,
    ) -> dict[str, int] | None:
        if not chapter_text:
            logger.warning(
                "Skipping knowledge extraction for chapter %s: no text provided.",
                chapter_number,
            )
            return None

        logger.info(
            "KnowledgeAgent: Starting knowledge extraction for chapter %d. Flawed draft: %s",
            chapter_number,
            is_from_flawed_draft,
        )

        raw_extracted_text, usage_data = await self._llm_extract_updates(
            plot_outline, chapter_text, chapter_number
        )

        if not raw_extracted_text.strip():
            logger.warning(
                "LLM extraction returned no text for chapter %d.", chapter_number
            )
            return usage_data

        char_updates_raw = "{}"
        world_updates_raw = "{}"
        kg_triples_text = ""

        try:
            parsed_json = json.loads(raw_extracted_text)
            char_updates_raw = json.dumps(parsed_json.get("character_updates", {}))
            world_updates_raw = json.dumps(parsed_json.get("world_updates", {}))
            kg_triples_list = parsed_json.get("kg_triples", [])
            if isinstance(kg_triples_list, list):
                kg_triples_text = "\n".join([str(t) for t in kg_triples_list])
            else:
                kg_triples_text = str(kg_triples_list)

        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse full extraction JSON for chapter {chapter_number}: {e}. "
                f"Attempting to extract individual sections with regex."
            )
            # Fallback to regex extraction
            char_match = re.search(
                r'"character_updates"\s*:\s*({.*?})', raw_extracted_text, re.DOTALL
            )
            if char_match:
                char_updates_raw = char_match.group(1)
                logger.info(
                    f"Regex successfully extracted character_updates block for Ch {chapter_number}."
                )
            else:
                logger.warning(
                    f"Could not find character_updates JSON block via regex for Ch {chapter_number}."
                )

            world_match = re.search(
                r'"world_updates"\s*:\s*({.*?})', raw_extracted_text, re.DOTALL
            )
            if world_match:
                world_updates_raw = world_match.group(1)
                logger.info(
                    f"Regex successfully extracted world_updates block for Ch {chapter_number}."
                )
            else:
                logger.warning(
                    f"Could not find world_updates JSON block via regex for Ch {chapter_number}."
                )

            triples_match = re.search(
                r'"kg_triples"\s*:\s*(\[.*?\])', raw_extracted_text, re.DOTALL
            )
            if triples_match:
                try:
                    triples_list_from_regex = json.loads(triples_match.group(1))
                    if isinstance(triples_list_from_regex, list):
                        kg_triples_text = "\n".join(
                            [str(t) for t in triples_list_from_regex]
                        )
                        logger.info(
                            f"Regex successfully extracted and parsed kg_triples block for Ch {chapter_number}."
                        )
                except json.JSONDecodeError:
                    logger.warning(
                        f"Found kg_triples block via regex for Ch {chapter_number}, but it was invalid JSON."
                    )
            else:
                logger.warning(
                    f"Could not find kg_triples JSON array via regex for Ch {chapter_number}."
                )

        # Use native parsing for optimal performance (Phase 3 optimization)
        char_updates_from_llm = self.parse_character_updates_native(
            char_updates_raw, chapter_number
        )
        world_updates_from_llm = self.parse_world_updates_native(
            world_updates_raw, chapter_number
        )

        parsed_triples_structured = parse_rdf_triples_with_rdflib(kg_triples_text)

        logger.info(
            f"Chapter {chapter_number} LLM Extraction: "
            f"{len(char_updates_from_llm)} char updates, "
            f"{sum(len(items) for items in world_updates_from_llm.values())} world item updates, "
            f"{len(parsed_triples_structured)} KG triples."
        )

        current_char_profiles_models = character_profiles
        current_world_models = world_building

        self.merge_updates(
            current_char_profiles_models,  # Pass model instances
            current_world_models,  # Pass model instances
            char_updates_from_llm,  # Already model instances
            world_updates_from_llm,  # Already model instances
            chapter_number,
            is_from_flawed_draft,
        )
        logger.info(
            f"Merged LLM updates into in-memory state for chapter {chapter_number}."
        )

        # Persist the DELTA of updates (char_updates_from_llm, world_updates_from_llm)
        # These functions expect model instances and the chapter number for delta context.
        if char_updates_from_llm:
            await self.persist_profiles(char_updates_from_llm, chapter_number)
        if world_updates_from_llm:
            await self.persist_world(world_updates_from_llm, chapter_number)

        if parsed_triples_structured:
            try:
                await kg_queries.add_kg_triples_batch_to_db(
                    parsed_triples_structured, chapter_number, is_from_flawed_draft
                )
                logger.info(
                    f"Persisted {len(parsed_triples_structured)} KG triples for chapter {chapter_number} to Neo4j."
                )
            except Exception as e:
                logger.error(
                    f"Failed to persist KG triples for chapter {chapter_number}: {e}",
                    exc_info=True,
                )

        logger.info(
            "Knowledge extraction, in-memory merge, and delta persistence complete for chapter %d.",
            chapter_number,
        )
        return usage_data
    
    async def extract_and_merge_knowledge_native(
        self,
        plot_outline: dict[str, Any],
        characters: list[CharacterProfile],
        world_items: list[WorldItem],
        chapter_number: int,
        chapter_text: str,
        is_from_flawed_draft: bool = False,
    ) -> dict[str, int] | None:
        """
        Native model version of extract_and_merge_knowledge.
        Eliminates dict conversion overhead by working directly with model instances.
        
        Args:
            plot_outline: Plot information dict
            characters: List of CharacterProfile models (will be modified in-place)
            world_items: List of WorldItem models (will be modified in-place)
            chapter_number: Current chapter number
            chapter_text: Chapter text to extract from
            is_from_flawed_draft: Whether text is from a flawed/unrevised draft
            
        Returns:
            LLM usage data dict or None if extraction failed
        """
        if not chapter_text:
            logger.warning(
                "Skipping knowledge extraction for chapter %s: no text provided.",
                chapter_number,
            )
            return None

        logger.info(
            "KnowledgeAgent (Native): Starting knowledge extraction for chapter %d. Flawed draft: %s",
            chapter_number,
            is_from_flawed_draft,
        )

        # Extract updates using LLM
        raw_extracted_text, usage_data = await self._llm_extract_updates(
            plot_outline, chapter_text, chapter_number
        )

        if not raw_extracted_text.strip():
            logger.warning(
                "LLM extraction returned no text for chapter %d.", chapter_number
            )
            return usage_data

        # Parse extraction results directly to models
        try:
            char_updates, world_updates, kg_triples_text = await self._extract_updates_as_models(
                raw_extracted_text, chapter_number
            )
            
            # Process KG triples for relationships (CRITICAL: This was missing!)
            parsed_triples_structured = parse_rdf_triples_with_rdflib(kg_triples_text)
            
            # Merge updates directly into existing model lists
            self._merge_character_updates_native(characters, char_updates, chapter_number)
            self._merge_world_updates_native(world_items, world_updates, chapter_number)
            
            # Persist models directly using native service
            await knowledge_graph_service.persist_entities(
                characters, world_items, chapter_number
            )
            
            # CRITICAL FIX: Persist KG triples/relationships
            if parsed_triples_structured:
                try:
                    await kg_queries.add_kg_triples_batch_to_db(
                        parsed_triples_structured, chapter_number, is_from_flawed_draft
                    )
                    logger.info(
                        f"Persisted {len(parsed_triples_structured)} KG triples for chapter {chapter_number} to Neo4j."
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to persist KG triples for chapter {chapter_number}: {e}",
                        exc_info=True,
                    )
            
            # DIAGNOSTIC: Log information for healing process debugging
            logger.info(
                f"Native extraction created entities for healing process to consider: "
                f"Characters: {[c.name for c in char_updates]}, "
                f"World items: {[w.name for w in world_updates]}"
            )
            
            logger.info(
                f"Native knowledge extraction complete for chapter {chapter_number}: "
                f"{len(char_updates)} character updates, {len(world_updates)} world updates, "
                f"{len(parsed_triples_structured)} KG triples"
            )
            
            return usage_data
            
        except Exception as e:
            logger.error(
                f"Error during native knowledge extraction for chapter {chapter_number}: {e}",
                exc_info=True,
            )
            return usage_data
    
    async def _extract_updates_as_models(
        self, 
        raw_text: str, 
        chapter_number: int
    ) -> tuple[list[CharacterProfile], list[WorldItem], str]:
        """
        Extract updates and return as models directly - no intermediate dict phase.
        
        Args:
            raw_text: Raw LLM extraction text (JSON format)
            chapter_number: Current chapter for tracking
            
        Returns:
            Tuple of (character_updates, world_updates, kg_triples_text)
        """
        char_updates = []
        world_updates = []
        kg_triples_text = ""
        
        try:
            extraction_data = json.loads(raw_text)
            
            # Convert character updates directly to models
            char_data = extraction_data.get("character_updates", {})
            for name, char_info in char_data.items():
                if isinstance(char_info, dict):
                    char_updates.append(CharacterProfile(
                        name=name,  # Use original name - let healing process handle deduplication
                        description=char_info.get("description", ""),
                        traits=char_info.get("traits", []),
                        status=char_info.get("status", "Unknown"),
                        relationships=char_info.get("relationships", {}),
                        created_chapter=char_info.get("created_chapter", chapter_number),
                        is_provisional=char_info.get("is_provisional", False),
                        updates=char_info  # Store original for reference
                    ))
            
            # Convert world updates directly to models  
            world_data = extraction_data.get("world_updates", {})
            for category, items in world_data.items():
                if isinstance(items, dict):
                    for item_name, item_info in items.items():
                        if isinstance(item_info, dict):
                            world_updates.append(WorldItem.from_dict(
                                category, item_name, item_info
                            ))
            
            # Extract KG triples for relationships (CRITICAL FIX!)
            kg_triples_list = extraction_data.get("kg_triples", [])
            if isinstance(kg_triples_list, list):
                kg_triples_text = "\n".join([str(t) for t in kg_triples_list])
            else:
                kg_triples_text = str(kg_triples_list)
            
            logger.debug(
                f"Extracted {len(char_updates)} character, {len(world_updates)} world item updates, "
                f"and {len(kg_triples_list)} KG triples as native models"
            )
            
            return char_updates, world_updates, kg_triples_text
            
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse extraction JSON for chapter {chapter_number}: {e}. "
                f"Attempting fallback regex parsing for KG triples."
            )
            
            # Fallback regex extraction for KG triples (critical for relationships!)
            triples_match = re.search(
                r'"kg_triples"\s*:\s*(\[.*?\])', raw_text, re.DOTALL
            )
            if triples_match:
                try:
                    triples_list_from_regex = json.loads(triples_match.group(1))
                    if isinstance(triples_list_from_regex, list):
                        kg_triples_text = "\n".join(
                            [str(t) for t in triples_list_from_regex]
                        )
                        logger.info(
                            f"Regex successfully extracted and parsed kg_triples block for Ch {chapter_number}."
                        )
                except json.JSONDecodeError:
                    logger.warning(
                        f"Found kg_triples block via regex for Ch {chapter_number}, but it was invalid JSON."
                    )
            else:
                logger.warning(
                    f"Could not find kg_triples JSON array via regex for Ch {chapter_number}."
                )
            
            # Could implement character and world fallback parsing here if needed
            return [], [], kg_triples_text
    
    def _merge_character_updates_native(
        self,
        existing_characters: list[CharacterProfile],
        new_updates: list[CharacterProfile], 
        chapter_number: int
    ) -> None:
        """
        Merge character updates directly into existing model list.
        
        Args:
            existing_characters: List of existing CharacterProfile models (modified in-place)
            new_updates: List of CharacterProfile updates to merge
            chapter_number: Current chapter for tracking
        """
        # Create lookup for existing characters
        char_lookup = {char.name: char for char in existing_characters}
        
        for update in new_updates:
            if update.name in char_lookup:
                # Update existing character
                existing_char = char_lookup[update.name]
                if update.description:
                    existing_char.description = update.description
                if update.traits:
                    # Merge traits, avoiding duplicates
                    existing_char.traits = list(set(existing_char.traits + update.traits))
                if update.status != "Unknown":
                    existing_char.status = update.status
                if update.relationships:
                    existing_char.relationships.update(update.relationships)
                existing_char.is_provisional = existing_char.is_provisional or update.is_provisional
            else:
                # Add new character
                update.created_chapter = chapter_number
                existing_characters.append(update)
                char_lookup[update.name] = update
    
    def _merge_world_updates_native(
        self,
        existing_world_items: list[WorldItem],
        new_updates: list[WorldItem],
        chapter_number: int
    ) -> None:
        """
        Merge world item updates directly into existing model list.
        
        Args:
            existing_world_items: List of existing WorldItem models (modified in-place)
            new_updates: List of WorldItem updates to merge
            chapter_number: Current chapter for tracking
        """
        # Create lookup for existing items
        item_lookup = {item.id: item for item in existing_world_items}
        
        for update in new_updates:
            if update.id in item_lookup:
                # Update existing item
                existing_item = item_lookup[update.id]
                if update.description:
                    existing_item.description = update.description
                if update.goals:
                    existing_item.goals = list(set(existing_item.goals + update.goals))
                if update.rules:
                    existing_item.rules = list(set(existing_item.rules + update.rules))
                if update.key_elements:
                    existing_item.key_elements = list(set(existing_item.key_elements + update.key_elements))
                if update.traits:
                    existing_item.traits = list(set(existing_item.traits + update.traits))
                if update.additional_properties:
                    existing_item.additional_properties.update(update.additional_properties)
                existing_item.is_provisional = existing_item.is_provisional or update.is_provisional
            else:
                # Add new item
                update.created_chapter = chapter_number
                existing_world_items.append(update)
                item_lookup[update.id] = update

    async def heal_and_enrich_kg(
        self, new_entities: list[dict[str, Any]] | None = None
    ):
        """
        Performs maintenance on the Knowledge Graph by enriching thin nodes,
        checking for inconsistencies, and resolving duplicate entities.

        Args:
            new_entities: Optional list of newly added entities to process incrementally.
                          If provided, only these entities will be processed for duplicates and enrichment.
                          If None, the entire graph will be processed (less efficient).
        """
        logger.info("KG Healer/Enricher: Starting maintenance cycle.")

        # If new entities provided, only process those (incremental update)
        if new_entities:
            logger.info(f"Processing {len(new_entities)} new entities incrementally.")
            for entity in new_entities:
                await self._resolve_duplicates_for_entity(entity)
                await self._enrich_entity_if_needed(entity)
        else:
            # Original full graph processing (less efficient)
            logger.info("Processing entire graph (full cycle).")

            # 1. Enrichment (which includes healing orphans/stubs)
            enrichment_cypher = await self._find_and_enrich_thin_nodes()

            if enrichment_cypher:
                logger.info(
                    f"Applying {len(enrichment_cypher)} enrichment updates to the KG."
                )
                try:
                    await neo4j_manager.execute_cypher_batch(enrichment_cypher)
                except Exception as e:
                    logger.error(
                        f"KG Healer/Enricher: Error applying enrichment batch: {e}",
                        exc_info=True,
                    )
            else:
                logger.info(
                    "KG Healer/Enricher: No thin nodes found for enrichment in this cycle."
                )

            # 2. Consistency Checks
            await self._run_consistency_checks()

            # 3. Entity Resolution
            await self._run_entity_resolution()

        # 4. Resolve dynamic relationship types using LLM guidance (always run)
        await self._resolve_dynamic_relationships()

        # 5. Relationship Healing (always run)
        promoted = await kg_queries.promote_dynamic_relationships()
        if promoted:
            logger.info("KG Healer: Promoted %d dynamic relationships.", promoted)
        removed = await kg_queries.deduplicate_relationships()
        if removed:
            logger.info("KG Healer: Deduplicated %d relationships.", removed)

        logger.info("KG Healer/Enricher: Maintenance cycle complete.")

    async def _resolve_duplicates_for_entity(self, entity: dict[str, Any]) -> None:
        """Resolve duplicates for a single entity using Neo4j's MERGE with uniqueness constraints."""
        # Extract entity information
        entity_name = entity.get("name")
        entity_type = entity.get("type", "Entity")

        if not entity_name:
            logger.warning("Cannot resolve duplicates for entity without name")
            return

        logger.debug(
            f"Resolving duplicates for entity: {entity_name} (type: {entity_type})"
        )

        # Create labels for the entity based on its type
        labels = ":Entity"
        if entity_type:
            # Normalize the entity type to create valid Neo4j labels
            normalized_type = "".join(c for c in entity_type.title() if c.isalnum())
            labels = f":{normalized_type}{labels}"

        # Validate node labels
        errors = validate_node_labels([entity_type])
        if errors:
            logger.warning("Invalid node labels for entity '%s': %s", entity_name, errors)

        # Use MERGE to ensure we have a single entity with this name
        # This will either match an existing entity or create a new one
        merge_query = f"""
        MERGE (e{labels} {{name: $entity_name}})
        ON CREATE SET e.created_ts = timestamp()
        ON MATCH SET e.last_seen_ts = timestamp()
        RETURN e
        """

        try:
            await neo4j_manager.execute_write_query(
                merge_query, {"entity_name": entity_name}
            )
            logger.debug(
                f"Successfully processed entity {entity_name} for duplicate resolution"
            )
        except Exception as e:
            logger.error(
                f"Error resolving duplicates for entity {entity_name}: {e}",
                exc_info=True,
            )

    async def _enrich_entity_if_needed(self, entity: dict[str, Any]) -> None:
        """Enrich a single entity if it's sparse."""
        # Extract entity information
        entity_name = entity.get("name")
        entity_type = entity.get("type", "Entity")
        entity_id = entity.get("id")

        if not entity_name:
            logger.warning("Cannot enrich entity without name")
            return

        logger.debug(
            f"Checking if entity needs enrichment: {entity_name} (type: {entity_type})"
        )

        # Check if the entity is sparse (missing description or other key information)
        is_sparse = await self._is_entity_sparse(entity_name, entity_type, entity_id)

        if is_sparse:
            logger.info(f"Entity {entity_name} is sparse, enriching...")
            await self._enrich_entity(entity_name, entity_type, entity_id)
        else:
            logger.debug(f"Entity {entity_name} is not sparse, skipping enrichment")

    async def _is_entity_sparse(
        self, entity_name: str, entity_type: str, entity_id: str | None = None
    ) -> bool:
        """Check if an entity is sparse (missing key information)."""
        # Query to check if entity has a description or other key properties
        if entity_type.lower() == "character":
            # For characters, check if they have a description
            query = """
            MATCH (c:Character {name: $entity_name})
            RETURN c.description AS description
            """
        elif entity_type.lower() == "worldelement":
            # For world elements, check if they have a description
            if entity_id:
                query = """
                MATCH (we:WorldElement {id: $entity_id})
                RETURN we.description AS description
                """
            else:
                query = """
                MATCH (we:WorldElement {name: $entity_name})
                RETURN we.description AS description
                """
        else:
            # For other entity types, check if they have a description
            query = """
            MATCH (e:Entity {name: $entity_name})
            RETURN e.description AS description
            """

        try:
            params = {"entity_name": entity_name}
            if entity_id:
                params["entity_id"] = entity_id

            results = await neo4j_manager.execute_read_query(query, params)
            if results:
                description = results[0].get("description")
                # Entity is considered sparse if it has no description or a very short one
                return not description or len(str(description).strip()) < 10
            else:
                # If no entity found, consider it sparse
                return True
        except Exception as e:
            logger.error(
                f"Error checking if entity {entity_name} is sparse: {e}", exc_info=True
            )
            # If we can't determine, assume it's not sparse to avoid unnecessary enrichment
            return False

    async def _enrich_entity(
        self, entity_name: str, entity_type: str, entity_id: str | None = None
    ) -> None:
        """Enrich an entity using LLM."""
        try:
            # Get chapter context for the entity
            context_chapters = await kg_queries.get_chapter_context_for_entity(
                entity_name=entity_name if not entity_id else None, entity_id=entity_id
            )

            # Choose the appropriate prompt based on entity type
            if entity_type.lower() == "character":
                prompt = render_prompt(
                    "knowledge_agent/enrich_character.j2",
                    {
                        "character_name": entity_name,
                        "chapter_context": context_chapters,
                    },
                )
            elif entity_type.lower() == "worldelement":
                # Get additional information about the world element
                element_info = {
                    "name": entity_name,
                    "category": "Unknown",
                    "id": entity_id or entity_name,
                }
                if entity_id:
                    # Try to get more detailed information about the world element
                    query = """
                    MATCH (we:WorldElement {id: $entity_id})
                    RETURN we.category AS category
                    """
                    try:
                        results = await neo4j_manager.execute_read_query(
                            query, {"entity_id": entity_id}
                        )
                        if results:
                            element_info["category"] = results[0].get(
                                "category", "Unknown"
                            )
                    except Exception:
                        pass

                prompt = render_prompt(
                    "knowledge_agent/enrich_world_element.j2",
                    {"element": element_info, "chapter_context": context_chapters},
                )
            else:
                # For other entity types, use a generic approach
                prompt = f"""
                /no_think
                You are a knowledge graph enrichment expert. Please provide a concise description for the following entity:
                
                Entity Name: {entity_name}
                Entity Type: {entity_type}
                
                Chapter Context:
                {context_chapters}
                
                Please respond with a JSON object containing a "description" field with the entity description.
                """

            # Call LLM to generate enrichment
            enrichment_text, _ = await llm_service.async_call_llm(
                model_name=config.KNOWLEDGE_UPDATE_MODEL,
                prompt=prompt,
                temperature=config.Temperatures.KG_EXTRACTION,
                auto_clean_response=True,
            )

            if enrichment_text:
                try:
                    data = json.loads(enrichment_text)
                    new_description = data.get("description")
                    if new_description and isinstance(new_description, str):
                        logger.info(f"Generated new description for '{entity_name}'.")

                        # Update the entity in the database
                        if entity_type.lower() == "character":
                            update_query = """
                            MATCH (c:Character {name: $name})
                            SET c.description = $desc, c.enriched_ts = timestamp()
                            """
                            await neo4j_manager.execute_write_query(
                                update_query,
                                {"name": entity_name, "desc": new_description},
                            )
                        elif entity_type.lower() == "worldelement" and entity_id:
                            update_query = """
                            MATCH (we:WorldElement {id: $id})
                            SET we.description = $desc, we.enriched_ts = timestamp()
                            """
                            await neo4j_manager.execute_write_query(
                                update_query, {"id": entity_id, "desc": new_description}
                            )
                        elif entity_type.lower() == "worldelement":
                            update_query = """
                            MATCH (we:WorldElement {name: $name})
                            SET we.description = $desc, we.enriched_ts = timestamp()
                            """
                            await neo4j_manager.execute_write_query(
                                update_query,
                                {"name": entity_name, "desc": new_description},
                            )
                        else:
                            update_query = """
                            MATCH (e:Entity {name: $name})
                            SET e.description = $desc, e.enriched_ts = timestamp()
                            """
                            await neo4j_manager.execute_write_query(
                                update_query,
                                {"name": entity_name, "desc": new_description},
                            )

                        logger.info(
                            f"Successfully enriched entity '{entity_name}' with new description."
                        )
                    else:
                        logger.warning(
                            f"Failed to parse description from LLM response for entity '{entity_name}': {enrichment_text}"
                        )
                except json.JSONDecodeError:
                    logger.error(
                        f"Failed to parse enrichment JSON for entity '{entity_name}': {enrichment_text}"
                    )
            else:
                logger.warning(
                    f"LLM returned empty response for entity enrichment: {entity_name}"
                )
        except Exception as e:
            logger.error(f"Error enriching entity {entity_name}: {e}", exc_info=True)

    async def _find_and_enrich_thin_nodes(self) -> list[tuple[str, dict[str, Any]]]:
        """Finds thin characters and world elements and generates enrichment updates in parallel."""
        statements: list[tuple[str, dict[str, Any]]] = []
        enrichment_tasks = []

        # Find all thin nodes first
        thin_chars = await character_queries.find_thin_characters_for_enrichment()
        thin_elements = await world_queries.find_thin_world_elements_for_enrichment()

        # Create tasks for enriching characters
        for char_info in thin_chars:
            enrichment_tasks.append(self._create_character_enrichment_task(char_info))

        # Create tasks for enriching world elements
        for element_info in thin_elements:
            enrichment_tasks.append(
                self._create_world_element_enrichment_task(element_info)
            )

        if not enrichment_tasks:
            return []

        logger.info(
            f"KG Healer: Found {len(enrichment_tasks)} thin nodes to enrich. Running LLM calls in parallel."
        )
        results = await asyncio.gather(*enrichment_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"KG Healer: An enrichment task failed: {result}")
            elif result:
                statements.append(result)

        return statements

    async def _create_character_enrichment_task(
        self, char_info: dict[str, Any]
    ) -> tuple[str, dict[str, Any]] | None:
        char_name = char_info.get("name")
        if not char_name:
            return None

        logger.info(f"KG Healer: Found thin character '{char_name}' for enrichment.")
        context_chapters = await kg_queries.get_chapter_context_for_entity(
            entity_name=char_name
        )
        prompt = render_prompt(
            "knowledge_agent/enrich_character.j2",
            {"character_name": char_name, "chapter_context": context_chapters},
        )
        enrichment_text, _ = await llm_service.async_call_llm(
            model_name=config.KNOWLEDGE_UPDATE_MODEL,
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            auto_clean_response=True,
        )
        if enrichment_text:
            try:
                data = json.loads(enrichment_text)
                new_description = data.get("description")
                if new_description and isinstance(new_description, str):
                    logger.info(
                        f"KG Healer: Generated new description for '{char_name}'."
                    )
                    return (
                        "MATCH (c:Character {name: $name}) SET c.description = $desc",
                        {"name": char_name, "desc": new_description},
                    )
            except json.JSONDecodeError:
                logger.error(
                    f"KG Healer: Failed to parse enrichment JSON for character '{char_name}': {enrichment_text}"
                )
        return None

    async def _create_world_element_enrichment_task(
        self, element_info: dict[str, Any]
    ) -> tuple[str, dict[str, Any]] | None:
        element_id = element_info.get("id")
        if not element_id:
            return None

        logger.info(
            f"KG Healer: Found thin world element '{element_info.get('name')}' (id: {element_id}) for enrichment."
        )
        context_chapters = await kg_queries.get_chapter_context_for_entity(
            entity_id=element_id
        )
        prompt = render_prompt(
            "knowledge_agent/enrich_world_element.j2",
            {"element": element_info, "chapter_context": context_chapters},
        )
        enrichment_text, _ = await llm_service.async_call_llm(
            model_name=config.KNOWLEDGE_UPDATE_MODEL,
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            auto_clean_response=True,
        )
        if enrichment_text:
            try:
                data = json.loads(enrichment_text)
                new_description = data.get("description")
                if new_description and isinstance(new_description, str):
                    logger.info(
                        f"KG Healer: Generated new description for world element id '{element_id}'."
                    )
                    return (
                        "MATCH (we:WorldElement {id: $id}) SET we.description = $desc",
                        {"id": element_id, "desc": new_description},
                    )
            except json.JSONDecodeError:
                logger.error(
                    f"KG Healer: Failed to parse enrichment JSON for world element id '{element_id}': {enrichment_text}"
                )
        return None

    async def _run_consistency_checks(self) -> None:
        """Runs various consistency checks on the KG and logs findings."""
        logger.info("KG Healer: Running consistency checks...")

        # 1. Check for contradictory traits
        contradictory_pairs = [
            ("Brave", "Cowardly"),
            ("Honest", "Deceitful"),
            ("Kind", "Cruel"),
            ("Generous", "Selfish"),
            ("Loyal", "Treacherous"),
        ]
        trait_findings = await kg_queries.find_contradictory_trait_characters(
            contradictory_pairs
        )
        if trait_findings:
            for finding in trait_findings:
                logger.warning(
                    f"KG Consistency Alert: Character '{finding.get('character_name')}' has contradictory traits: "
                    f"'{finding.get('trait1')}' and '{finding.get('trait2')}'."
                )
        else:
            logger.info("KG Consistency Check: No contradictory traits found.")

        # 2. Check for post-mortem activity
        activity_findings = await kg_queries.find_post_mortem_activity()
        if activity_findings:
            for finding in activity_findings:
                logger.warning(
                    f"KG Consistency Alert: Character '{finding.get('character_name')}' was marked dead in chapter "
                    f"{finding.get('death_chapter')} but has later activities: {finding.get('post_mortem_activities')}."
                )
        else:
            logger.info("KG Consistency Check: No post-mortem activity found.")

    async def _run_entity_resolution(
        self, new_entities: list[dict[str, Any]] | None = None
    ) -> None:
        """Finds and resolves potential duplicate entities in the KG.

        Args:
            new_entities: Optional list of newly added entities to check for duplicates.
                          If provided, only these entities will be checked.
                          If None, the entire graph will be processed for duplicates.
        """
        logger.info("KG Healer: Running entity resolution...")

        # If new entities provided, only check those for duplicates (incremental)
        if new_entities:
            # Process only new entities for duplicates
            logger.info(
                f"Checking {len(new_entities)} new entities for duplicates incrementally."
            )
            for entity in new_entities:
                await self._resolve_duplicates_for_new_entity(entity)
        else:
            # Original full graph processing (less efficient)
            # Use lower similarity threshold to catch character name variations like "Nuyara" vs "Nuyara Vex"
            candidate_pairs = await kg_queries.find_candidate_duplicate_entities(similarity_threshold=0.6)

            if not candidate_pairs:
                logger.info("KG Healer: No candidate duplicate entities found.")
                return

            logger.info(
                f"KG Healer: Found {len(candidate_pairs)} candidate pairs for entity resolution."
            )

            jinja_template = Template(ENTITY_RESOLUTION_PROMPT_TEMPLATE)

            for pair in candidate_pairs:
                id1, id2 = pair.get("id1"), pair.get("id2")
                if not id1 or not id2:
                    continue

                # Fetch context for both entities in parallel
                context1_task = kg_queries.get_entity_context_for_resolution(id1)
                context2_task = kg_queries.get_entity_context_for_resolution(id2)
                context1, context2 = await asyncio.gather(context1_task, context2_task)

                if not context1 or not context2:
                    logger.warning(
                        f"Could not fetch full context for pair ({id1}, {id2}). Skipping."
                    )
                    continue

                prompt = jinja_template.render(entity1=context1, entity2=context2)
                llm_response, _ = await llm_service.async_call_llm(
                    model_name=config.KNOWLEDGE_UPDATE_MODEL,
                    prompt=prompt,
                    temperature=0.1,
                    auto_clean_response=True,
                )

                try:
                    decision_data = json.loads(llm_response)
                    if (
                        decision_data.get("is_same_entity") is True
                        and decision_data.get("confidence_score", 0.0) > 0.8
                    ):
                        logger.info(
                            f"LLM confirmed merge for '{context1.get('name')}' (id: {id1}) and "
                            f"'{context2.get('name')}' (id: {id2}). Reason: {decision_data.get('reason')}"
                        )

                        # Heuristic to decide which node to keep
                        degree1 = context1.get("degree", 0)
                        degree2 = context2.get("degree", 0)

                        # Prefer node with more relationships
                        if degree1 > degree2:
                            target_id, source_id = id1, id2
                        elif degree2 > degree1:
                            target_id, source_id = id2, id1
                        else:
                            # Tie-breaker: prefer the one with a more detailed description
                            desc1_len = len(
                                context1.get("properties", {}).get("description", "")
                            )
                            desc2_len = len(
                                context2.get("properties", {}).get("description", "")
                            )
                            if desc1_len >= desc2_len:
                                target_id, source_id = id1, id2
                            else:
                                target_id, source_id = id2, id1

                        await kg_queries.merge_entities(target_id, source_id)
                    else:
                        logger.info(
                            f"LLM decided NOT to merge '{context1.get('name')}' and '{context2.get('name')}'. "
                            f"Reason: {decision_data.get('reason')}"
                        )

                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(
                        f"Failed to parse entity resolution response from LLM for pair ({id1}, {id2}): {e}. Response: {llm_response}"
                    )

    async def _resolve_duplicates_for_new_entity(self, entity: dict[str, Any]) -> None:
        """Resolve duplicates for a single new entity using Neo4j's MERGE with uniqueness constraints."""
        # Extract entity information
        entity_name = entity.get("name")
        entity_type = entity.get("type", "Entity")

        if not entity_name:
            logger.warning("Cannot resolve duplicates for new entity without name")
            return

        logger.debug(
            f"Resolving duplicates for new entity: {entity_name} (type: {entity_type})"
        )

        # Create labels for the entity based on its type
        labels = ":Entity"
        if entity_type:
            # Normalize the entity type to create valid Neo4j labels
            normalized_type = "".join(c for c in entity_type.title() if c.isalnum())
            labels = f":{normalized_type}{labels}"

        # Validate node labels
        errors = validate_node_labels([entity_type])
        if errors:
            logger.warning("Invalid node labels for new entity '%s': %s", entity_name, errors)

        # Use MERGE to ensure we have a single entity with this name
        # This will either match an existing entity or create a new one
        merge_query = f"""
        MERGE (e{labels} {{name: $entity_name}})
        ON CREATE SET
            e.created_ts = timestamp(),
            e.type = $entity_type
        ON MATCH SET
            e.last_seen_ts = timestamp(),
            e.type = coalesce(e.type, $entity_type)
        RETURN e
        """

        try:
            await neo4j_manager.execute_write_query(
                merge_query, {"entity_name": entity_name, "entity_type": entity_type}
            )
            logger.debug(
                f"Successfully processed new entity {entity_name} for duplicate resolution"
            )
        except Exception as e:
            logger.error(
                f"Error resolving duplicates for new entity {entity_name}: {e}",
                exc_info=True,
            )

    async def _resolve_dynamic_relationships(self) -> None:
        """Resolve generic DYNAMIC_REL types using a lightweight LLM."""
        logger.info("KG Healer: Resolving dynamic relationship types via LLM...")
        dyn_rels = await kg_queries.fetch_unresolved_dynamic_relationships()
        if not dyn_rels:
            logger.info("KG Healer: No unresolved dynamic relationships found.")
            return
        jinja_template = Template(DYNAMIC_REL_RESOLUTION_PROMPT_TEMPLATE)
        for rel in dyn_rels:
            prompt = jinja_template.render(rel)
            new_type_raw, _ = await llm_service.async_call_llm(
                model_name=config.MEDIUM_MODEL,
                prompt=prompt,
                temperature=config.Temperatures.KG_EXTRACTION,
                max_tokens=10,
                auto_clean_response=True,
            )
            new_type = kg_queries.normalize_relationship_type(new_type_raw)
            # Validate the new relationship type
            errors = validate_relationship_types([new_type])
            if errors:
                logger.warning("Invalid relationship type from LLM: %s", errors)
            if new_type and new_type != "UNKNOWN":
                await kg_queries.update_dynamic_relationship_type(
                    rel["rel_id"], new_type
                )
                logger.info(
                    "KG Healer: Updated relationship %s -> %s",
                    rel["rel_id"],
                    new_type,
                )
            else:
                logger.info(
                    "KG Healer: LLM could not refine relationship %s (response: %s)",
                    rel["rel_id"],
                    new_type_raw,
                )

    async def heal_schema(self) -> None:
        """Ensure all nodes and relationships follow the expected schema."""
        logger.info("KG Healer: Checking base schema conformity...")
        statements = [
            ("MATCH (n) WHERE NOT n:Entity SET n:Entity", {}),
            (
                "MATCH ()-[r:DYNAMIC_REL]-() WHERE r.type IS NULL SET r.type = 'UNKNOWN'",
                {},
            ),
        ]
        try:
            await neo4j_manager.execute_cypher_batch(statements)
            await kg_queries.normalize_existing_relationship_types()
        except Exception as exc:  # pragma: no cover - narrow DB errors
            logger.error("KG Healer: Schema healing failed: %s", exc, exc_info=True)


__all__ = ["KnowledgeAgent"]
