# agents/knowledge_agent.py
# knowledge_agent.py
import asyncio
import json
import logging
import re
from typing import Any

from async_lru import alru_cache  # type: ignore

import config
from core.db_manager import neo4j_manager
from core.knowledge_graph_service import knowledge_graph_service
from core.llm_interface_refactored import llm_service
from core.schema_validator import (
    validate_kg_object,
    validate_node_labels,
)
from data_access import (
    character_queries,
    kg_queries,
    plot_queries,
    world_queries,
)

# Import native versions for performance optimization
from data_access.character_queries import (
    sync_characters,
)
from data_access.world_queries import (
    sync_world_items,
)
from models.kg_models import CharacterProfile, WorldItem
from processing.parsing_utils import (
    parse_rdf_triples_with_rdflib,
)
from prompts.prompt_renderer import render_prompt

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
    """Parse character update JSON provided by LLM and create models directly."""
    char_updates: dict[str, CharacterProfile] = {}
    if not json_text_block or json_text_block.strip() in ["null", "None", ""]:
        return char_updates

    try:
        parsed_json = json.loads(json_text_block)
        if not isinstance(parsed_json, dict):
            logger.error(
                f"Character updates JSON was not a dictionary. Received: {type(parsed_json)}"
            )
            return char_updates
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse character updates JSON: {e}")
        return char_updates

    for char_name, raw_attributes in parsed_json.items():
        if not isinstance(raw_attributes, dict):
            logger.warning(
                f"Skipping character '{char_name}': attributes not in dict format"
            )
            continue

        # Simple, efficient attribute processing
        traits = []
        skills = []
        relationships = {}

        # Process key attributes efficiently
        if "traits" in raw_attributes:
            traits_val = raw_attributes["traits"]
            if isinstance(traits_val, list):
                traits = [str(item).strip() for item in traits_val if str(item).strip()]
            elif isinstance(traits_val, str):
                traits = [
                    item.strip() for item in traits_val.split(",") if item.strip()
                ]

        if "skills" in raw_attributes:
            skills_val = raw_attributes["skills"]
            if isinstance(skills_val, list):
                skills = [str(item).strip() for item in skills_val if str(item).strip()]
            elif isinstance(skills_val, str):
                skills = [
                    item.strip() for item in skills_val.split(",") if item.strip()
                ]

        if "relationships" in raw_attributes:
            rels_val = raw_attributes["relationships"]
            if isinstance(rels_val, dict):
                relationships = {str(k): str(v) for k, v in rels_val.items()}
            elif isinstance(rels_val, list):
                # Convert list to dict with default relationship
                for rel_entry in rels_val:
                    if isinstance(rel_entry, str) and rel_entry.strip():
                        relationships[rel_entry.strip()] = "related"

        try:
            # Create model directly without dict intermediate
            char_updates[char_name] = CharacterProfile(
                name=char_name,
                description=raw_attributes.get("description", ""),
                traits=traits,
                relationships=relationships,
                skills=skills,
                status=raw_attributes.get("status", "active"),
            )
        except Exception as e:
            logger.error(
                f"Error creating CharacterProfile for '{char_name}': {e}. "
                f"Attributes: {raw_attributes}",
                exc_info=True,
            )

    return char_updates


def parse_unified_world_updates(
    json_text_block: str, chapter_number: int
) -> dict[str, dict[str, WorldItem]]:
    """Parse world update JSON provided by LLM and create models directly."""
    world_updates: dict[str, dict[str, WorldItem]] = {}
    if not json_text_block or json_text_block.strip() in ["null", "None", ""]:
        return world_updates

    try:
        parsed_json = json.loads(json_text_block)
        if not isinstance(parsed_json, dict):
            logger.error(
                f"World updates JSON was not a dictionary. Received: {type(parsed_json)}"
            )
            return world_updates
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
                # Generate ID from name and category
                item_id = f"{category_name_llm.lower().replace(' ', '_')}_{item_name_llm.lower().replace(' ', '_')}"

                world_item_instance = WorldItem(
                    id=item_id,
                    name=item_name_llm,
                    category=category_name_llm,
                    description=raw_item_attributes.get("description", ""),
                    # Copy any additional attributes
                    **{
                        k: v
                        for k, v in raw_item_attributes.items()
                        if k not in ["id", "name", "category", "description"]
                    },
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
            logger.warning(f"Invalid CharacterProfile for '{name}': {errors}")

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
            logger.debug(f"Profile for {name} modified")


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
                logger.warning(
                    "Invalid WorldItem for '%s' in category '%s': %s",
                    name,
                    category,
                    errors,
                )

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
        profiles_to_persist: dict[str, CharacterProfile] | list[CharacterProfile],
        chapter_number_for_delta: int,
        full_sync: bool = False,
    ) -> None:
        """Persist character profiles to Neo4j with enhanced validation."""

        # Handle both dict and list input formats
        profiles_list = []
        if isinstance(profiles_to_persist, dict):
            # Validate all profiles before persisting
            for name, profile in profiles_to_persist.items():
                validation_errors = validate_kg_object(profile)
                if validation_errors:
                    logger.warning(
                        f"Validation issues for character profile {name}: {validation_errors}"
                    )
            profiles_list = list(profiles_to_persist.values())
        else:
            # Validate all profiles before persisting
            for profile in profiles_to_persist:
                validation_errors = validate_kg_object(profile)
                if validation_errors:
                    logger.warning(
                        f"Validation issues for character profile {profile.name}: {validation_errors}"
                    )
            profiles_list = profiles_to_persist

        # Use native model version for better performance
        await sync_characters(profiles_list, chapter_number_for_delta)

    async def persist_world(
        self,
        world_items_to_persist: dict[str, dict[str, WorldItem]] | list[WorldItem],
        chapter_number_for_delta: int,
        full_sync: bool = False,
    ) -> None:
        """Persist world elements to Neo4j with enhanced node typing and validation."""
        if config.BOOTSTRAP_USE_ENHANCED_NODE_TYPES:
            from core.schema_validator import validate_kg_object

            # Handle both dict and list input formats
            if isinstance(world_items_to_persist, dict):
                # Enhance world items with proper node typing and validation
                for category, items_dict in world_items_to_persist.items():
                    if not isinstance(items_dict, dict):
                        continue

                    for item_name, item in items_dict.items():
                        if not isinstance(item, WorldItem):
                            continue

                        # Validate and enhance with better node typing
                        validation_errors = validate_kg_object(item)
                        if validation_errors:
                            logger.warning(
                                f"Validation issues for world item {category}/{item_name}: {validation_errors}"
                            )

        # Use native model version for better performance
        # Convert to list format if needed
        if isinstance(world_items_to_persist, dict):
            # Flatten the nested dict structure into a list
            world_items_list = []
            for category_items in world_items_to_persist.values():
                if isinstance(category_items, dict):
                    world_items_list.extend(category_items.values())
        else:
            world_items_list = world_items_to_persist

        await sync_world_items(world_items_list, chapter_number_for_delta)

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

        # Include bootstrap elements in extraction prompt for early chapters
        if (
            config.BOOTSTRAP_INTEGRATION_ENABLED
            and chapter_number <= config.BOOTSTRAP_INTEGRATION_CHAPTERS
        ):
            bootstrap_context = await self._get_bootstrap_context_for_extraction(
                chapter_number
            )
            if bootstrap_context:
                # Add bootstrap context to the prompt
                prompt += (
                    f"\n\nBootstrap World Context to Consider:\n{bootstrap_context}"
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

    async def _get_bootstrap_context_for_extraction(self, chapter_number: int) -> str:
        """Get bootstrap elements that should be considered for relationship extraction."""
        # Only include bootstrap context for early chapters (less than 10 as per clarification)
        if chapter_number >= 10:
            return ""

        # Get bootstrap elements from world queries
        bootstrap_elements = await world_queries.get_bootstrap_world_elements()

        # Limit to a reasonable number to prevent prompt bloat (6 as suggested in CLAUDE.md)
        context_lines = []
        for element in bootstrap_elements[: config.MAX_BOOTSTRAP_ELEMENTS_PER_CONTEXT]:
            if element.description:
                context_lines.append(
                    f"- {element.name} ({element.category}): {element.description[:150]}..."
                )

        return "\n".join(context_lines) if context_lines else ""

    async def _heal_bootstrap_connectivity(self, chapter_number: int) -> None:
        """Connect orphaned bootstrap elements to the active narrative graph."""
        # Only perform bootstrap connectivity healing for early chapters
        if chapter_number > config.BOOTSTRAP_INTEGRATION_CHAPTERS:
            return

        # Find orphaned bootstrap elements
        orphaned_bootstrap = await kg_queries.find_orphaned_bootstrap_elements()

        # Limit the number of orphaned elements to heal per cycle
        for element in orphaned_bootstrap[: config.BOOTSTRAP_HEALING_LIMIT]:
            # Find potential bridge characters/locations
            bridge_candidates = await kg_queries.find_potential_bridges(element)

            if bridge_candidates:
                # Create contextual relationship with the most connected bridge candidate
                await kg_queries.create_contextual_relationship(
                    element, bridge_candidates[0], "CONTEXTUALLY_RELATED"
                )
                logger.info(
                    f"Connected orphaned bootstrap element '{element.get('name')}' "
                    f"to '{bridge_candidates[0].get('name')}' to establish narrative presence."
                )

    async def extract_and_merge_knowledge(
        self,
        plot_outline: dict[str, Any],
        characters: list[CharacterProfile],
        world_items: list[WorldItem],
        chapter_number: int,
        chapter_text: str,
        is_from_flawed_draft: bool = False,
    ) -> dict[str, int] | None:
        """
        Extract knowledge from chapter text and merge into existing model lists.
        Works directly with model instances for optimal performance.

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
            (
                char_updates,
                world_updates,
                kg_triples_text,
            ) = await self._extract_updates_as_models(
                raw_extracted_text, chapter_number
            )

            # Process KG triples for relationships (CRITICAL: This was missing!)
            parsed_triples_structured = parse_rdf_triples_with_rdflib(kg_triples_text)

            # Log each parsed triple
            for triple in parsed_triples_structured:
                object_value = triple.get("object_entity", triple.get("object_literal"))
                logger.info(
                    f"Parsed: {triple['subject']} | {triple['predicate']} | {object_value}"
                )

            # Merge updates directly into existing model lists
            self._merge_character_updates_native(
                characters, char_updates, chapter_number
            )
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
        self, raw_text: str, chapter_number: int
    ) -> tuple[list[CharacterProfile], list[WorldItem], str]:
        """
        Extract updates and return as models directly - no intermediate dict phase.
        Now includes proactive duplicate prevention.

        Args:
            raw_text: Raw LLM extraction text (JSON format)
            chapter_number: Current chapter for tracking

        Returns:
            Tuple of (character_updates, world_updates, kg_triples_text)
        """
        char_updates = []
        world_updates = []
        kg_triples_text = ""

        # Optimized JSON parsing with single attempt and better error handling
        extraction_data = await self._parse_extraction_json(raw_text, chapter_number)
        if not extraction_data:
            return [], [], ""

        try:
            # Import duplicate prevention functions
            from processing.entity_deduplication import (
                generate_entity_id,
                prevent_character_duplication,
                prevent_world_item_duplication,
            )

            # Convert character updates directly to models with duplicate prevention
            char_data = extraction_data.get("character_updates", {})
            for name, char_info in char_data.items():
                if isinstance(char_info, dict):
                    description = char_info.get("description", "")
                    final_name = name  # Default to original name

                    # Check for existing similar character (only if enabled in config)
                    if (
                        config.ENABLE_DUPLICATE_PREVENTION
                        and config.DUPLICATE_PREVENTION_CHARACTER_ENABLED
                    ):
                        existing_name = await prevent_character_duplication(
                            name, description
                        )

                        if existing_name:
                            # Use existing character name to prevent duplicate
                            final_name = existing_name
                            logger.info(
                                f"Using existing character '{final_name}' instead of creating duplicate '{name}'"
                            )

                    # Handle relationships if they need structuring from list to dict
                    # Assuming LLM provides relationships as a list of strings
                    # like "Target: Detail" or just "Target"
                    # Or ideally, as a dict: {"Target": "Detail"}
                    raw_relationships = char_info.get("relationships", {})
                    processed_relationships = {}
                    if isinstance(raw_relationships, list):
                        rels_list = raw_relationships
                        for rel_entry in rels_list:
                            if isinstance(rel_entry, str):
                                if ":" in rel_entry:
                                    parts = rel_entry.split(":", 1)
                                    if (
                                        len(parts) == 2
                                        and parts[0].strip()
                                        and parts[1].strip()
                                    ):
                                        processed_relationships[parts[0].strip()] = (
                                            parts[1].strip()
                                        )
                                    elif parts[
                                        0
                                    ].strip():  # If only name is there before colon
                                        processed_relationships[parts[0].strip()] = (
                                            "related"
                                        )
                                elif rel_entry.strip():  # No colon, just a name
                                    processed_relationships[rel_entry.strip()] = (
                                        "related"
                                    )
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
                                    processed_relationships[target_name] = detail
                    elif isinstance(raw_relationships, dict):
                        processed_relationships = {
                            str(k): str(v) for k, v in raw_relationships.items()
                        }
                    # If it's neither a list nor a dict, we'll use an empty dict

                    char_updates.append(
                        CharacterProfile(
                            name=final_name,
                            description=description,
                            traits=char_info.get("traits", []),
                            status=char_info.get("status", "Unknown"),
                            relationships=processed_relationships,
                            created_chapter=char_info.get(
                                "created_chapter", chapter_number
                            ),
                            is_provisional=char_info.get("is_provisional", False),
                            updates=char_info,  # Store original for reference
                        )
                    )

            # Convert world updates directly to models with duplicate prevention
            world_data = extraction_data.get("world_updates", {})
            for category, items in world_data.items():
                if isinstance(items, dict):
                    for item_name, item_info in items.items():
                        if isinstance(item_info, dict):
                            description = item_info.get("description", "")

                            # Check for existing similar world item (only if enabled in config)
                            if (
                                config.ENABLE_DUPLICATE_PREVENTION
                                and config.DUPLICATE_PREVENTION_WORLD_ITEM_ENABLED
                            ):
                                existing_id = await prevent_world_item_duplication(
                                    item_name, category, description
                                )

                                if existing_id:
                                    # Use existing world item ID to prevent duplicate
                                    logger.info(
                                        f"Using existing world item with ID '{existing_id}' instead of creating duplicate '{item_name}' in category '{category}'"
                                    )
                                    # Update the item_info to use existing ID
                                    item_info["id"] = existing_id
                                else:
                                    # Generate deterministic ID for new item
                                    item_info["id"] = generate_entity_id(
                                        item_name, category, chapter_number
                                    )
                            else:
                                # Use existing ID generation logic when duplicate prevention is disabled
                                if not item_info.get("id"):
                                    item_info["id"] = generate_entity_id(
                                        item_name, category, chapter_number
                                    )

                            world_updates.append(
                                WorldItem.from_dict(category, item_name, item_info)
                            )

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

        except Exception as e:
            logger.error(
                f"Error processing extraction data for chapter {chapter_number}: {e}"
            )
            return [], [], ""

    async def _parse_extraction_json(
        self, raw_text: str, chapter_number: int
    ) -> dict[str, Any] | None:
        """
        Optimized JSON parsing with single attempt and robust fallback handling.

        Eliminates multiple JSON decode attempts for better performance.
        """
        if not raw_text or not raw_text.strip():
            logger.warning(f"Empty extraction text for chapter {chapter_number}")
            return None

        # Clean up common LLM JSON formatting issues before parsing
        cleaned_text = self._clean_llm_json(raw_text)

        try:
            # Single JSON parse attempt with cleaned text
            extraction_data = json.loads(cleaned_text)

            if not isinstance(extraction_data, dict):
                logger.error(
                    f"Extraction JSON was not a dictionary for chapter {chapter_number}"
                )
                return None

            return extraction_data

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed for chapter {chapter_number}: {e}")

            # Single fallback attempt using regex for critical KG triples only
            fallback_data = self._extract_kg_triples_fallback(raw_text, chapter_number)
            return fallback_data

    def _clean_llm_json(self, raw_text: str) -> str:
        """Clean up common LLM JSON formatting issues."""
        # Remove markdown code blocks if present
        if raw_text.strip().startswith("```"):
            lines = raw_text.strip().split("\n")
            if len(lines) > 2:
                # Remove first and last lines (markdown markers)
                cleaned = "\n".join(lines[1:-1])
            else:
                cleaned = raw_text
        else:
            cleaned = raw_text

        # Remove common trailing commas before closing brackets/braces
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

        # Fix common quote issues
        cleaned = cleaned.replace('"', '"').replace('"', '"')

        return cleaned.strip()

    def _extract_kg_triples_fallback(
        self, raw_text: str, chapter_number: int
    ) -> dict[str, Any]:
        """
        Fallback extraction that only attempts to recover KG triples using regex.
        This eliminates multiple JSON parsing attempts for better performance.
        """
        fallback_data = {"character_updates": {}, "world_updates": {}, "kg_triples": []}

        # Single regex attempt for KG triples
        triples_match = re.search(r'"kg_triples"\s*:\s*(\[.*?\])', raw_text, re.DOTALL)
        if triples_match:
            try:
                triples_list = json.loads(triples_match.group(1))
                if isinstance(triples_list, list):
                    fallback_data["kg_triples"] = triples_list
                    logger.info(
                        f"Fallback successfully recovered kg_triples for chapter {chapter_number}"
                    )
            except json.JSONDecodeError:
                logger.warning(
                    f"Fallback kg_triples parsing failed for chapter {chapter_number}"
                )

        return fallback_data

    def _merge_character_updates_native(
        self,
        existing_characters: list[CharacterProfile],
        new_updates: list[CharacterProfile],
        chapter_number: int,
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
                    existing_char.traits = list(
                        set(existing_char.traits + update.traits)
                    )
                if update.status != "Unknown":
                    existing_char.status = update.status
                if update.relationships:
                    existing_char.relationships.update(update.relationships)
                existing_char.is_provisional = (
                    existing_char.is_provisional or update.is_provisional
                )
            else:
                # Add new character
                update.created_chapter = chapter_number
                existing_characters.append(update)
                char_lookup[update.name] = update

    def _merge_world_updates_native(
        self,
        existing_world_items: list[WorldItem],
        new_updates: list[WorldItem],
        chapter_number: int,
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
                    existing_item.key_elements = list(
                        set(existing_item.key_elements + update.key_elements)
                    )
                if update.traits:
                    existing_item.traits = list(
                        set(existing_item.traits + update.traits)
                    )
                if update.additional_properties:
                    existing_item.additional_properties.update(
                        update.additional_properties
                    )
                existing_item.is_provisional = (
                    existing_item.is_provisional or update.is_provisional
                )
            else:
                # Add new item
                update.created_chapter = chapter_number
                existing_world_items.append(update)
                item_lookup[update.id] = update

    async def heal_and_enrich_kg(
        self,
        new_entities: list[dict[str, Any]] | None = None,
        chapter_number: int | None = None,
    ):
        """
        Performs optimized maintenance on the Knowledge Graph by enriching thin nodes,
        checking for inconsistencies, and resolving duplicate entities.

        Performance optimized: Uses incremental processing when possible,
        skips expensive operations when not needed, and uses caching.

        Args:
            new_entities: Optional list of newly added entities to process incrementally.
                          If provided, only these entities will be processed (O(n) complexity).
                          If None, the entire graph will be processed (O(n²) complexity).
            chapter_number: Current chapter for context-aware processing
        """
        logger.info("KG Healer/Enricher: Starting maintenance cycle.")

        # Always prefer incremental processing for performance
        if new_entities:
            logger.info(
                f"Running incremental KG maintenance for {len(new_entities)} new entities."
            )
            await self._process_entities_incrementally(new_entities, chapter_number)
            return

        # Full graph processing - only when absolutely necessary
        logger.warning("Running full KG maintenance cycle (resource intensive).")

        # Determine if full processing is actually needed based on recent activity
        recent_activity = await self._check_recent_kg_activity()
        if not recent_activity:
            logger.info(
                "No recent KG activity detected. Skipping full maintenance cycle."
            )
            return

        # Run optimized full graph processing
        await self._run_full_maintenance_cycle(chapter_number)

    async def _process_entities_incrementally(
        self, new_entities: list[dict[str, Any]], chapter_number: int | None
    ) -> None:
        """Process new entities incrementally with batching for performance."""
        batch_size = 10  # Process in small batches to avoid memory issues

        for i in range(0, len(new_entities), batch_size):
            batch = new_entities[i : i + batch_size]
            batch_tasks = []

            for entity in batch:
                # Run duplicate resolution and enrichment in parallel for each entity
                batch_tasks.extend(
                    [
                        self._resolve_duplicates_for_new_entity(entity),
                        self._enrich_entity_if_needed(entity, chapter_number),
                    ]
                )

            # Execute batch in parallel
            await asyncio.gather(*batch_tasks, return_exceptions=True)
            logger.debug(f"Processed incremental batch {i//batch_size + 1}")

    async def _check_recent_kg_activity(self) -> bool:
        """Check if there has been recent KG activity that would warrant full maintenance."""
        try:
            # Check if there are any entities modified in the last 24 hours
            # Use actual timestamp properties that exist: created_ts, updated_ts
            results = await neo4j_manager.execute_read_query(
                "MATCH (n) WHERE n.created_ts > timestamp() - 86400000 OR n.updated_ts > timestamp() - 86400000 RETURN count(n) as recent_count",
                {},
            )
            recent_count = results[0].get("recent_count", 0) if results else 0
            return (
                recent_count > 5
            )  # Only run full cycle if significant recent activity
        except Exception as e:
            logger.warning(
                f"Could not check recent KG activity: {e}. Assuming activity exists."
            )
            return True

    async def _run_full_maintenance_cycle(self, chapter_number: int | None) -> None:
        """Run the full maintenance cycle with optimizations."""
        try:
            # 1. Enrichment - find and fix thin nodes (limit to most critical)
            enrichment_cypher = await self._find_and_enrich_thin_nodes(limit=20)

            if enrichment_cypher:
                logger.info(
                    f"Applying {len(enrichment_cypher)} critical enrichment updates."
                )
                await neo4j_manager.execute_cypher_batch(enrichment_cypher)
            else:
                logger.info("No critical thin nodes found for enrichment.")

            # 2. Consistency Checks (lightweight version)
            await self._run_consistency_checks()

            # 3. Entity Resolution (only if we have new entities to check)
            await self._run_entity_resolution()

        except Exception as e:
            logger.error(f"Error during full maintenance cycle: {e}", exc_info=True)

    async def _enrich_entity_if_needed(
        self, entity: dict[str, Any], chapter_number: int | None = None
    ) -> None:
        """Enrich a single entity if it appears to be thin or incomplete."""
        entity_id = entity.get("id")
        entity_name = entity.get("name")

        if not entity_id or not entity_name:
            return

        # Check if entity needs enrichment (has minimal description)
        description = entity.get("description", "")
        if len(description.strip()) > 50:  # Skip if already well-described
            return

        logger.debug(f"Enriching entity: {entity_name} (id: {entity_id})")

        # Use the existing enrichment logic for different entity types
        if entity.get("type") == "Character":
            await self._enrich_character_by_name(entity_name)
        elif entity.get("type") in ["WorldElement", "Location", "Organization"]:
            await self._enrich_world_element_by_id(entity_id)

        # 4. Resolve dynamic relationship types using LLM guidance (always run)
        await self._resolve_dynamic_relationships()

        # 5. Relationship Healing (always run)
        promoted = await kg_queries.promote_dynamic_relationships()
        if promoted:
            logger.info("KG Healer: Promoted %d dynamic relationships.", promoted)

        # 5a. Consolidate similar relationships using predefined taxonomy
        consolidated = await kg_queries.consolidate_similar_relationships()
        if consolidated:
            logger.info(
                "KG Healer: Consolidated %d relationships to canonical forms.",
                consolidated,
            )

        removed = await kg_queries.deduplicate_relationships()
        if removed:
            logger.info("KG Healer: Deduplicated %d relationships.", removed)

        # 6. Bootstrap Connectivity Healing (only for early chapters)
        if chapter_number is not None:
            await self._heal_bootstrap_connectivity(chapter_number)

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
            logger.warning(
                "Invalid node labels for entity '%s': %s", entity_name, errors
            )

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

    async def _find_and_enrich_thin_nodes(
        self, limit: int | None = None
    ) -> list[tuple[str, dict[str, Any]]]:
        """Finds thin characters and world elements and generates enrichment updates in parallel."""
        statements: list[tuple[str, dict[str, Any]]] = []
        enrichment_tasks = []

        # Find thin nodes with optional limit for performance
        thin_chars = await character_queries.find_thin_characters_for_enrichment()
        thin_elements = await world_queries.find_thin_world_elements_for_enrichment()

        # Apply limit if specified (prioritize most critical nodes)
        if limit:
            total_available = len(thin_chars) + len(thin_elements)
            if total_available > limit:
                # Prioritize characters over world elements, but take a mix
                char_limit = min(len(thin_chars), limit // 2)
                element_limit = min(len(thin_elements), limit - char_limit)
                thin_chars = thin_chars[:char_limit]
                thin_elements = thin_elements[:element_limit]
                logger.info(
                    f"Limited thin node enrichment to {limit} most critical nodes "
                    f"({len(thin_chars)} chars, {len(thin_elements)} elements)"
                )

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

        Performance optimized: Uses incremental processing when possible,
        batched context fetching, and early termination conditions.

        Args:
            new_entities: Optional list of newly added entities to check for duplicates.
                          If provided, only these entities will be checked (O(n) complexity).
                          If None, the entire graph will be processed for duplicates (O(n²)).
        """
        logger.info("KG Healer: Running entity resolution...")

        # Prefer incremental processing for better performance
        if new_entities:
            logger.info(
                f"Running incremental entity resolution for {len(new_entities)} new entities."
            )
            # Process in batches to avoid overwhelming the system
            batch_size = min(10, len(new_entities))
            for i in range(0, len(new_entities), batch_size):
                batch = new_entities[i : i + batch_size]
                await asyncio.gather(
                    *[
                        self._resolve_duplicates_for_new_entity(entity)
                        for entity in batch
                    ]
                )
                logger.debug(f"Processed entity resolution batch {i//batch_size + 1}")
            return

        # Full graph processing - use performance optimizations
        logger.info("Running full graph entity resolution (less efficient).")

        # Use cached similarity threshold to catch character name variations
        candidate_pairs = await kg_queries.find_candidate_duplicate_entities(
            similarity_threshold=0.4
        )

        if not candidate_pairs:
            logger.info("KG Healer: No candidate duplicate entities found.")
            return

        logger.info(
            f"Processing {len(candidate_pairs)} candidate pairs for resolution."
        )

        # Process pairs in batches for better memory management
        batch_size = 5  # Reduced batch size to limit concurrent LLM calls
        processed_pairs = 0

        for i in range(0, len(candidate_pairs), batch_size):
            batch = candidate_pairs[i : i + batch_size]

            # Pre-fetch all contexts for this batch in parallel
            context_tasks = []
            for pair in batch:
                id1, id2 = pair.get("id1"), pair.get("id2")
                if id1 and id2:
                    context_tasks.append(
                        (
                            id1,
                            id2,
                            kg_queries.get_entity_context_for_resolution(id1),
                            kg_queries.get_entity_context_for_resolution(id2),
                        )
                    )

            if not context_tasks:
                continue

            # Execute all context fetches in parallel
            contexts = []
            for id1, id2, task1, task2 in context_tasks:
                try:
                    context1, context2 = await asyncio.gather(task1, task2)
                    if context1 and context2:
                        contexts.append((id1, id2, context1, context2))
                    else:
                        logger.warning(
                            f"Missing context for pair ({id1}, {id2}). Skipping."
                        )
                except Exception as e:
                    logger.error(f"Error fetching context for pair ({id1}, {id2}): {e}")
                    continue

            # Process resolution decisions for this batch
            for id1, id2, context1, context2 in contexts:
                try:
                    await self._process_entity_pair_resolution(
                        id1, id2, context1, context2
                    )
                    processed_pairs += 1
                except Exception as e:
                    logger.error(f"Error processing entity pair ({id1}, {id2}): {e}")
                    continue

            logger.debug(
                f"Completed resolution batch {i//batch_size + 1}, processed {processed_pairs} pairs total"
            )

    async def _process_entity_pair_resolution(
        self, id1: str, id2: str, context1: dict, context2: dict
    ) -> None:
        """Process a single entity pair for potential resolution."""
        prompt = render_prompt(
            "knowledge_agent/entity_resolution.j2",
            {"entity1": context1, "entity2": context2},
        )

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

                # Optimized heuristic to decide which node to keep
                target_id, source_id = self._select_merge_target(
                    id1, id2, context1, context2
                )

                # Use LLM's reasoning for the merge
                reason = decision_data.get("reason", "LLM entity resolution merge")
                await kg_queries.merge_entities(source_id, target_id, reason)
            else:
                logger.info(
                    f"LLM decided NOT to merge '{context1.get('name')}' and '{context2.get('name')}'. "
                    f"Reason: {decision_data.get('reason')}"
                )

        except (json.JSONDecodeError, TypeError) as e:
            logger.error(
                f"Failed to parse entity resolution response from LLM for pair ({id1}, {id2}): {e}. Response: {llm_response}"
            )

    def _select_merge_target(
        self, id1: str, id2: str, context1: dict, context2: dict
    ) -> tuple[str, str]:
        """Optimized heuristic to decide which node to keep during merge.

        Returns:
            tuple: (target_id, source_id) where target_id is kept, source_id is merged into it
        """
        # Prefer node with more relationships
        degree1 = context1.get("degree", 0)
        degree2 = context2.get("degree", 0)

        if degree1 > degree2:
            return id1, id2
        elif degree2 > degree1:
            return id2, id1

        # Tie-breaker: prefer the one with a more detailed description
        desc1_len = len(context1.get("properties", {}).get("description", ""))
        desc2_len = len(context2.get("properties", {}).get("description", ""))

        if desc1_len >= desc2_len:
            return id1, id2
        else:
            return id2, id1

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
            logger.warning(
                "Invalid node labels for new entity '%s': %s", entity_name, errors
            )

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
        # Add early return if normalization is disabled
        if config.DISABLE_RELATIONSHIP_NORMALIZATION:
            logger.info(
                "Relationship normalization disabled - skipping dynamic relationship resolution"
            )
            return

        logger.info("KG Healer: Resolving dynamic relationship types via LLM...")
        dyn_rels = await kg_queries.fetch_unresolved_dynamic_relationships()
        if not dyn_rels:
            logger.info("KG Healer: No unresolved dynamic relationships found.")
            return
        for rel in dyn_rels:
            prompt = render_prompt(
                "knowledge_agent/dynamic_relationship_resolution.j2", rel
            )
            new_type_raw, _ = await llm_service.async_call_llm(
                model_name=config.MEDIUM_MODEL,
                prompt=prompt,
                temperature=config.Temperatures.KG_EXTRACTION,
                max_tokens=10,
                auto_clean_response=True,
            )
            new_type = kg_queries.normalize_relationship_type(new_type_raw)
            # The normalize_relationship_type already validates and returns a valid type
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
            logger.error(f"KG Healer: Schema healing failed: {exc}", exc_info=True)


__all__ = ["KnowledgeAgent"]
