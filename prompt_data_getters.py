# prompt_data_getters.py
"""
Helper functions to prepare specific data snippets for LLM prompts in the SAGA system.
These functions typically filter or format parts of the agent's state,
increasingly by querying Neo4j directly for richer, graph-aware context.
"""

import asyncio
import copy
import logging
import re
from typing import Any, Dict, List, Optional, Set

import config
import utils  # For _is_fill_in

from data_access import character_queries, kg_queries, world_queries
from models import CharacterProfile, SceneDetail, WorldItem
from processing.neo4j_query import (
    execute_factual_query,
    get_most_recent_entity_status,
    get_novel_info_property,
    get_entity_context_for_resolution,
    get_shortest_path_length_between_entities,
)

logger = logging.getLogger(__name__)


async def get_character_state_snippet_for_prompt(
    character_profiles: Dict[str, CharacterProfile],
    plot_outline: Dict[str, Any],
    current_chapter_num_for_filtering: Optional[int] = None,
) -> str:
    """
    Creates a concise plain text string of key character states for prompts,
    fetching data directly from Neo4j.
    """
    text_output_lines_list: List[str] = []
    char_names_to_process: List[str] = []

    protagonist_name = plot_outline.get("protagonist_name")

    character_profiles_data = character_profiles
    all_known_char_names_from_state: List[str] = list(character_profiles_data.keys())

    if protagonist_name and protagonist_name in all_known_char_names_from_state:
        char_names_to_process.append(protagonist_name)

    for name in sorted(all_known_char_names_from_state):
        if (
            name != protagonist_name
            and len(char_names_to_process)
            < config.PLANNING_CONTEXT_MAX_CHARACTERS_IN_SNIPPET
        ):
            char_names_to_process.append(name)
        elif (
            len(char_names_to_process)
            >= config.PLANNING_CONTEXT_MAX_CHARACTERS_IN_SNIPPET
        ):
            break

    effective_filter_chapter = (
        (current_chapter_num_for_filtering - 1)
        if current_chapter_num_for_filtering is not None
        and current_chapter_num_for_filtering > 0
        else config.KG_PREPOPULATION_CHAPTER_NUM
    )

    fetch_tasks = [
        character_queries.get_character_info_for_snippet_from_db(
            name, effective_filter_chapter
        )  # MODIFIED
        for name in char_names_to_process
    ]

    if not fetch_tasks:
        return "No character profiles available or applicable to fetch for snippet."

    character_info_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    for i, res_or_exc in enumerate(character_info_results):
        char_name_for_snippet = char_names_to_process[i]
        text_output_lines_list.append(f"Character: {char_name_for_snippet}")

        if isinstance(res_or_exc, Exception):
            logger.error(
                f"Error fetching snippet info for character '{char_name_for_snippet}' from Neo4j: {res_or_exc}"
            )
            profile_fallback_obj = character_profiles_data.get(char_name_for_snippet)
            if isinstance(profile_fallback_obj, CharacterProfile):
                profile_fallback = profile_fallback_obj.to_dict()
            elif isinstance(profile_fallback_obj, dict):
                profile_fallback = profile_fallback_obj
            else:
                profile_fallback = {}

            desc_fb = (
                (
                    str(
                        profile_fallback.get(
                            "description", "Error fetching description."
                        )
                    )[: config.PLANNING_CONTEXT_MAX_CHARS_PER_PROFILE_DESC]
                ).strip()
                + "..."
            )
            status_fb = str(
                profile_fallback.get("status", "Unknown (error fetching status)")
            )
            dev_note_fb = "Error fetching development note."
            dev_keys_fb = sorted(
                [
                    k
                    for k in profile_fallback
                    if k.startswith("development_in_chapter_")
                    and int(k.split("_")[-1]) <= effective_filter_chapter
                ],
                key=lambda x: int(x.split("_")[-1]),
                reverse=True,
            )
            if dev_keys_fb:
                dev_note_fb = (
                    (
                        str(profile_fallback.get(dev_keys_fb[0], "N/A"))[
                            : config.PLANNING_CONTEXT_MAX_RECENT_DEV_PER_PROFILE
                        ]
                    ).strip()
                    + "..."
                )
            text_output_lines_list.append(f"  Description Snippet: {desc_fb}")
            text_output_lines_list.append(
                f"  Current Status: {status_fb} (Error in Neo4j fetch, using local state)"
            )
            text_output_lines_list.append(
                f"  Most Recent Development Note: {dev_note_fb}"
            )
        elif res_or_exc is None:
            logger.warning(
                f"Neo4j character snippet fetch for '{char_name_for_snippet}' was None. Using local state as fallback."
            )
            profile_fallback_obj = character_profiles_data.get(char_name_for_snippet)
            if isinstance(profile_fallback_obj, CharacterProfile):
                profile_fallback = profile_fallback_obj.to_dict()
            elif isinstance(profile_fallback_obj, dict):
                profile_fallback = profile_fallback_obj
            else:
                profile_fallback = {}

            desc_fb = (
                (
                    str(profile_fallback.get("description", "N/A"))[
                        : config.PLANNING_CONTEXT_MAX_CHARS_PER_PROFILE_DESC
                    ]
                ).strip()
                + "..."
            )
            status_fb = str(profile_fallback.get("status", "Unknown"))
            dev_note_fb = "N/A"
            dev_keys_fb = sorted(
                [
                    k
                    for k in profile_fallback
                    if k.startswith("development_in_chapter_")
                    and int(k.split("_")[-1]) <= effective_filter_chapter
                ],
                key=lambda x: int(x.split("_")[-1]),
                reverse=True,
            )
            if dev_keys_fb:
                dev_note_fb = (
                    (
                        str(profile_fallback.get(dev_keys_fb[0], "N/A"))[
                            : config.PLANNING_CONTEXT_MAX_RECENT_DEV_PER_PROFILE
                        ]
                    ).strip()
                    + "..."
                )
            text_output_lines_list.append(f"  Description Snippet: {desc_fb}")
            text_output_lines_list.append(
                f"  Current Status: {status_fb} (Neo4j fetch was None, using local state)"
            )
            text_output_lines_list.append(
                f"  Most Recent Development Note: {dev_note_fb}"
            )
        else:
            profile_summary = res_or_exc
            provisional_note = (
                " (Note: Some info may be provisional)"
                if profile_summary.get("is_provisional_overall")
                else ""
            )
            desc_text = str(profile_summary.get("description", "N/A"))
            status_text = str(profile_summary.get("current_status", "Unknown"))
            dev_note_text = str(
                profile_summary.get("most_recent_development_note", "N/A")
            )
            text_output_lines_list.append(
                f"  Description Snippet: {(desc_text[: config.PLANNING_CONTEXT_MAX_CHARS_PER_PROFILE_DESC]).strip()}..."
            )
            text_output_lines_list.append(
                f"  Current Status: {status_text}{provisional_note}"
            )
            text_output_lines_list.append(
                f"  Most Recent Development Note: {(dev_note_text[: config.PLANNING_CONTEXT_MAX_RECENT_DEV_PER_PROFILE]).strip()}..."
            )
        text_output_lines_list.append("")

    return (
        "\n".join(text_output_lines_list).strip()
        if text_output_lines_list
        else "No character profiles available or applicable."
    )


async def get_world_state_snippet_for_prompt(
    world_building: Dict[str, Dict[str, WorldItem]],
    current_chapter_num_for_filtering: Optional[int] = None,
) -> str:
    """
    Creates a concise plain text string of key world states for prompts,
    fetching data directly from Neo4j.
    """
    text_output_lines_list: List[str] = []
    any_provisional_in_world_snippet = False

    effective_filter_chapter = (
        (current_chapter_num_for_filtering - 1)
        if current_chapter_num_for_filtering is not None
        and current_chapter_num_for_filtering > 0
        else config.KG_PREPOPULATION_CHAPTER_NUM
    )

    world_categories_to_fetch = {
        "locations": config.PLANNING_CONTEXT_MAX_LOCATIONS_IN_SNIPPET,
        "systems": config.PLANNING_CONTEXT_MAX_SYSTEMS_IN_SNIPPET,
        "factions": config.PLANNING_CONTEXT_MAX_FACTIONS_IN_SNIPPET,
    }

    fetch_tasks_world = []
    category_names_for_tasks = []

    for category_name, max_items in world_categories_to_fetch.items():
        if max_items > 0:
            fetch_tasks_world.append(
                world_queries.get_world_elements_for_snippet_from_db(
                    category_name, effective_filter_chapter, max_items
                )  # MODIFIED
            )
            category_names_for_tasks.append(category_name)

    if not fetch_tasks_world:
        return "No significant world-building data available or applicable to fetch for snippet."

    world_building_data = world_building
    world_info_results = await asyncio.gather(
        *fetch_tasks_world, return_exceptions=True
    )

    for i, res_or_exc in enumerate(world_info_results):
        category_name_result = category_names_for_tasks[i]
        category_title = f"Key {category_name_result.capitalize()}:"
        category_items_lines = []

        if isinstance(res_or_exc, Exception):
            logger.error(
                f"Error fetching snippet info for world category '{category_name_result}' from Neo4j: {res_or_exc}"
            )
            category_items_lines.append("  - Error fetching data.")
        elif not res_or_exc:  # Empty list from DB query
            logger.debug(
                f"Neo4j world snippet fetch for '{category_name_result}' empty. Using local state as fallback."
            )  # Changed to debug
            category_data_fallback = world_building_data.get(category_name_result, {})
            item_count_fb = 0
            for item_name_fb, item_details_fb in sorted(category_data_fallback.items()):
                if item_count_fb >= world_categories_to_fetch[category_name_result]:
                    break
                if item_name_fb.startswith(
                    ("_", "source_quality_chapter_", "category_updated_in_chapter_")
                ):
                    continue
                desc_snip_fb = ""
                if isinstance(item_details_fb, dict) and item_details_fb.get(
                    "description"
                ):
                    desc_snip_fb = (
                        f": {str(item_details_fb['description'])[:50].strip()}..."
                    )
                category_items_lines.append(f"  - {item_name_fb}{desc_snip_fb}")
                item_count_fb += 1
            if item_count_fb == 0:
                category_items_lines.append(
                    "  - None notable or available (from local state)."
                )
        else:  # Results from DB
            items_list = res_or_exc
            category_has_provisional = False
            for item_data in items_list:
                name = str(item_data.get("name", "Unknown Item"))
                desc_snip = str(item_data.get("description_snippet", ""))
                prov_item_note = (
                    " (provisional)" if item_data.get("is_provisional") else ""
                )
                display_desc_snip = (
                    desc_snip.replace(name, "", 1).strip()
                    if desc_snip.startswith(name)
                    else desc_snip
                )
                display_desc_snip = (
                    (": " + display_desc_snip) if display_desc_snip else ""
                )
                category_items_lines.append(
                    f"  - {name}{prov_item_note}{display_desc_snip}"
                )
                if item_data.get("is_provisional"):
                    category_has_provisional = True
                    any_provisional_in_world_snippet = True
            if not items_list:
                category_items_lines.append(
                    "  - None notable or available (from Neo4j)."
                )
            if category_has_provisional:
                category_title += " (Note: Some items may be provisional)"

        text_output_lines_list.append(category_title)
        text_output_lines_list.extend(category_items_lines)
        text_output_lines_list.append("")

    if any_provisional_in_world_snippet and text_output_lines_list:
        text_output_lines_list.insert(
            0,
            "Overall World Note: Some world information might be provisional from unrevised prior chapters.\n",
        )

    return (
        "\n".join(text_output_lines_list).strip()
        if text_output_lines_list
        else "No significant world-building data available or applicable."
    )


def _format_dict_for_plain_text_prompt(
    data: Dict[str, Any], indent_level: int = 0, name_override: Optional[str] = None
) -> List[str]:
    lines = []
    indent = "  " * indent_level
    if name_override:
        lines.append(f"{indent}{name_override}:")
        indent_level += 1
        indent = "  " * indent_level

    priority_keys = [
        "description",
        "status",
        "traits",
        "relationships",
        "character_arc_summary",
        "role",
        "role_in_story",
        "goals",
        "rules",
        "key_elements",
        "atmosphere",
        "motivations",
        "prompt_notes",
    ]

    data_keys = list(data.keys())
    sorted_keys = [pk for pk in priority_keys if pk in data_keys]
    remaining_keys = sorted([k for k in data_keys if k not in priority_keys])
    sorted_keys.extend(remaining_keys)

    for key in sorted_keys:
        value = data[key]

        if key.startswith(
            ("source_quality_chapter_", "updated_in_chapter_", "added_in_chapter_")
        ) and key not in ["prompt_notes"]:
            continue
        if key == "is_provisional_hint":
            continue

        key_str_display = str(key).replace("_", " ").capitalize()

        if isinstance(value, dict):
            if value:
                lines.append(f"{indent}{key_str_display}:")
                lines.extend(
                    _format_dict_for_plain_text_prompt(value, indent_level + 1)
                )
        elif isinstance(value, list):
            if not value:
                lines.append(f"{indent}{key_str_display}: (empty list or N/A)")
            else:
                lines.append(f"{indent}{key_str_display}:")
                try:
                    display_items = sorted(
                        [str(x) for x in value if not isinstance(x, dict)]
                    )
                    dict_items = [x for x in value if isinstance(x, dict)]
                except TypeError:
                    display_items = [str(x) for x in value if not isinstance(x, dict)]
                    dict_items = [x for x in value if isinstance(x, dict)]

                for item_str in display_items:
                    lines.append(f"{indent}  - {item_str}")
                for item_dict in dict_items:
                    lines.extend(
                        _format_dict_for_plain_text_prompt(
                            item_dict, indent_level + 1, name_override="- Item"
                        )
                    )
        elif value is not None and (isinstance(value, bool) or str(value).strip()):
            lines.append(f"{indent}{key_str_display}: {str(value)}")

    return lines


def _add_provisional_notes_and_filter_developments(
    item_data_original: Dict[str, Any],
    up_to_chapter_inclusive: Optional[int],
    is_character: bool = True,
) -> Dict[str, Any]:
    item_data = copy.deepcopy(item_data_original)
    prompt_notes_list = []
    effective_filter_chapter = (
        config.KG_PREPOPULATION_CHAPTER_NUM
        if up_to_chapter_inclusive == config.KG_PREPOPULATION_CHAPTER_NUM
        else up_to_chapter_inclusive
    )

    dev_elaboration_prefix = (
        "development_in_chapter_" if is_character else "elaboration_in_chapter_"
    )
    added_prefix = "added_in_chapter_"

    keys_to_remove = []
    has_provisional_data_relevant_to_filter = False

    for key in list(item_data.keys()):
        if key.startswith((dev_elaboration_prefix, added_prefix)):
            try:
                chap_num_of_key_str = key.split("_")[-1]
                chap_num_of_key = (
                    int(re.match(r"\d+", chap_num_of_key_str).group(0))
                    if re.match(r"\d+", chap_num_of_key_str)
                    else -1
                )
                if (
                    effective_filter_chapter is not None
                    and chap_num_of_key > effective_filter_chapter
                ):
                    keys_to_remove.append(key)
            except (ValueError, IndexError, AttributeError) as e:
                logger.warning(
                    f"Could not parse chapter from key '{key}' during filtering: {e}"
                )

        if key.startswith("source_quality_chapter_"):
            try:
                chap_num_of_source_str = key.split("_")[-1]
                chap_num_of_source = (
                    int(re.match(r"\d+", chap_num_of_source_str).group(0))
                    if re.match(r"\d+", chap_num_of_source_str)
                    else -1
                )

                if (
                    effective_filter_chapter is None
                    or chap_num_of_source <= effective_filter_chapter
                ) and item_data[key] == "provisional_from_unrevised_draft":
                    has_provisional_data_relevant_to_filter = True
                    note = f"Data from Chapter {chap_num_of_source} may be provisional (from unrevised draft)."
                    if note not in prompt_notes_list:
                        prompt_notes_list.append(note)
            except (ValueError, IndexError, AttributeError) as e:
                logger.warning(
                    f"Could not parse chapter from source_quality key '{key}': {e}"
                )

    for k_rem in keys_to_remove:
        item_data.pop(k_rem, None)

    if has_provisional_data_relevant_to_filter:
        item_data["is_provisional_hint"] = True

    if prompt_notes_list:
        item_data["prompt_notes"] = sorted(list(set(prompt_notes_list)))

    return item_data


async def _get_character_profiles_dict_with_notes(
    character_names: List[str],
    up_to_chapter_inclusive: Optional[int],
) -> Dict[str, Any]:
    logger.debug(
        "Internal: Getting character profiles dict with notes up to chapter %s.",
        up_to_chapter_inclusive,
    )
    processed_profiles: Dict[str, Any] = {}
    filter_chapter = up_to_chapter_inclusive

    if not character_names:
        logger.warning("Character name list is empty.")
        return {}

    for char_name in character_names:
        try:
            profile_obj = await character_queries.get_character_profile_by_name(
                char_name
            )
        except Exception as exc:
            logger.error(
                "Error fetching character '%s' from Neo4j: %s",
                char_name,
                exc,
            )
            continue

        if not profile_obj:
            logger.warning("Character '%s' not found in Neo4j.", char_name)
            continue

        profile_dict = profile_obj.to_dict()
        processed_profiles[char_name] = _add_provisional_notes_and_filter_developments(
            profile_dict,
            filter_chapter,
            is_character=True,
        )

    return processed_profiles


async def get_filtered_character_profiles_for_prompt_plain_text(
    character_names: List[str],
    up_to_chapter_inclusive: Optional[int] = None,
) -> str:
    logger.info(
        f"Fetching and formatting filtered character profiles as PLAIN TEXT up to chapter {up_to_chapter_inclusive}."
    )
    filter_chapter_for_profiles = (
        config.KG_PREPOPULATION_CHAPTER_NUM
        if up_to_chapter_inclusive == 0
        else up_to_chapter_inclusive
    )

    profiles_dict_with_notes = await _get_character_profiles_dict_with_notes(
        character_names,
        filter_chapter_for_profiles,
    )
    if not profiles_dict_with_notes:
        return "No character profiles available."

    output_lines_list = ["Key Character Profiles:"]
    sorted_char_names = sorted(profiles_dict_with_notes.keys())

    for char_name in sorted_char_names:
        profile_data = profiles_dict_with_notes[char_name]
        if not isinstance(profile_data, dict) or not profile_data:
            continue

        output_lines_list.append("")
        formatted_profile_lines = _format_dict_for_plain_text_prompt(
            profile_data, indent_level=0, name_override=char_name
        )
        output_lines_list.extend(formatted_profile_lines)

    return "\n".join(output_lines_list).strip()


async def _get_world_data_dict_with_notes(
    world_item_ids_by_category: Dict[str, List[str]],
    up_to_chapter_inclusive: Optional[int],
) -> Dict[str, Any]:
    logger.debug(
        "Internal: Getting world data dict with notes up to chapter %s.",
        up_to_chapter_inclusive,
    )
    processed_world_data: Dict[str, Any] = {}
    filter_chapter = up_to_chapter_inclusive

    if not world_item_ids_by_category:
        logger.warning("World item ID mapping is empty.")
        return {}

    for category_name, item_id_list in world_item_ids_by_category.items():
        processed_category: Dict[str, Any] = {}

        for item_id in item_id_list:
            try:
                item_obj = await world_queries.get_world_item_by_id(item_id)
            except Exception as exc:
                logger.error(
                    "Error fetching world item '%s' from Neo4j: %s",
                    item_id,
                    exc,
                )
                continue

            if not item_obj:
                logger.warning(
                    "World item id '%s' not found in Neo4j.",
                    item_id,
                )
                continue

            item_dict = item_obj.to_dict()
            processed_category[item_obj.name] = (
                _add_provisional_notes_and_filter_developments(
                    item_dict,
                    filter_chapter,
                    is_character=False,
                )
            )

        processed_world_data[category_name] = processed_category

    return processed_world_data


async def get_filtered_world_data_for_prompt_plain_text(
    world_item_ids_by_category: Dict[str, List[str]],
    up_to_chapter_inclusive: Optional[int] = None,
) -> str:
    logger.info(
        f"Fetching and formatting filtered world data as PLAIN TEXT up to chapter {up_to_chapter_inclusive}."
    )
    filter_chapter_for_world = (
        config.KG_PREPOPULATION_CHAPTER_NUM
        if up_to_chapter_inclusive == 0
        else up_to_chapter_inclusive
    )

    world_data_dict_with_notes = await _get_world_data_dict_with_notes(
        world_item_ids_by_category,
        filter_chapter_for_world,
    )
    if not world_data_dict_with_notes:
        return "No world-building data available."

    output_lines_list = []
    overview_data = world_data_dict_with_notes.get("_overview_")
    if (
        overview_data
        and isinstance(overview_data, dict)
        and overview_data.get("description")
    ):
        output_lines_list.append("World-Building Overview:")
        output_lines_list.extend(
            _format_dict_for_plain_text_prompt(overview_data, indent_level=0)
        )
        output_lines_list.append("")

    sorted_categories = sorted(
        [
            cat
            for cat in world_data_dict_with_notes.keys()
            if cat not in ["_overview_", "is_default", "source", "user_supplied_data"]
        ]
    )

    for category_name in sorted_categories:
        category_items = world_data_dict_with_notes[category_name]
        if not isinstance(category_items, dict) or not category_items:
            continue

        category_header_added = False
        sorted_item_names = sorted(category_items.keys())

        for item_name in sorted_item_names:
            item_data = category_items[item_name]
            if not isinstance(item_data, dict) or not item_data:
                continue

            if not category_header_added:
                output_lines_list.append(
                    f"{category_name.replace('_', ' ').capitalize()}:"
                )
                category_header_added = True

            item_lines = _format_dict_for_plain_text_prompt(
                item_data, indent_level=0, name_override=item_name
            )
            output_lines_list.extend(item_lines)
            if item_lines:
                output_lines_list.append("")

        if output_lines_list and output_lines_list[-1] == "" and category_header_added:
            pass
        elif category_header_added:
            output_lines_list.append("")

    if not output_lines_list:
        return "No significant world-building data available after filtering."
    while output_lines_list and not output_lines_list[-1].strip():
        output_lines_list.pop()
    return "\n".join(output_lines_list).strip()




async def get_reliable_kg_facts_for_drafting_prompt(
    plot_outline: Dict[str, Any],
    chapter_number: int,
    chapter_plan: Optional[List[SceneDetail]] = None,
    max_facts_per_char: int = 2,
    max_total_facts: int = 7,
) -> str:
    if chapter_number <= 0:
        return "No KG facts applicable for pre-first chapter."

    kg_chapter_limit = (
        config.KG_PREPOPULATION_CHAPTER_NUM
        if chapter_number == 1
        else chapter_number - 1
    )

    facts_for_prompt_list: List[str] = []

    plot_outline_data = plot_outline
    protagonist_name = plot_outline_data.get(
        "protagonist_name", config.DEFAULT_PROTAGONIST_NAME
    )

    characters_of_interest: Set[str] = (
        {protagonist_name}
        if protagonist_name and not utils._is_fill_in(protagonist_name)
        else set()
    )

    if chapter_plan and isinstance(chapter_plan, list):
        for scene_detail in chapter_plan:
            if (
                isinstance(scene_detail, dict)
                and "characters_involved" in scene_detail
                and isinstance(scene_detail["characters_involved"], list)
            ):
                for char_name_in_plan in scene_detail["characters_involved"]:
                    if (
                        isinstance(char_name_in_plan, str)
                        and char_name_in_plan.strip()
                        and not utils._is_fill_in(char_name_in_plan)
                    ):
                        characters_of_interest.add(char_name_in_plan.strip())

    logger.debug(
        f"KG fact gathering for Ch {chapter_number} draft: Chars of interest: {characters_of_interest}, KG chapter limit: {kg_chapter_limit}"
    )

    if protagonist_name and characters_of_interest:
        pruned: Set[str] = set()
        for c in characters_of_interest:
            if c == protagonist_name:
                pruned.add(c)
                continue
            path_len = await get_shortest_path_length_between_entities(
                protagonist_name, c
            )
            if path_len is not None and path_len <= 3:
                pruned.add(c)
        characters_of_interest = pruned

    # Parallel execution of novel info property queries
    novel_info_tasks = [
        get_novel_info_property("theme"),
        get_novel_info_property("central_conflict")
    ]
    novel_info_results = await asyncio.gather(*novel_info_tasks, return_exceptions=True)
    
    # Process theme
    if len(facts_for_prompt_list) < max_total_facts and not isinstance(novel_info_results[0], Exception):
        value = novel_info_results[0]
        if value:
            fact_text = f"- The novel's central theme is: {value}."
            if fact_text not in facts_for_prompt_list:
                facts_for_prompt_list.append(fact_text)
    elif isinstance(novel_info_results[0], Exception):
        logger.warning(f"KG Query for novel context 'theme' failed: {novel_info_results[0]}")
    
    # Process central conflict
    if len(facts_for_prompt_list) < max_total_facts and not isinstance(novel_info_results[1], Exception):
        value = novel_info_results[1]
        if value:
            fact_text = f"- The main conflict summary: {value}."
            if fact_text not in facts_for_prompt_list:
                facts_for_prompt_list.append(fact_text)
    elif isinstance(novel_info_results[1], Exception):
        logger.warning(f"KG Query for novel context 'central_conflict' failed: {novel_info_results[1]}")

    # Prepare character-related queries for parallel execution
    character_tasks = []
    character_names_list = list(characters_of_interest)[:3]
    
    # Create tasks for all character-related queries
    for char_name in character_names_list:
        character_tasks.extend([
            get_most_recent_entity_status(char_name, "status_is", kg_chapter_limit),
            get_most_recent_entity_status(char_name, "located_in", kg_chapter_limit),
            execute_factual_query(subject=char_name, chapter_limit=kg_chapter_limit)
        ])
    
    # Execute all character-related queries in parallel
    character_results = await asyncio.gather(*character_tasks, return_exceptions=True)
    
    # Process results
    for i, char_name in enumerate(character_names_list):
        if len(facts_for_prompt_list) >= max_total_facts:
            break
        facts_for_this_char = 0
        
        # Get results for this character (3 results per character)
        status_result = character_results[i * 3]
        location_result = character_results[i * 3 + 1]
        relationships_result = character_results[i * 3 + 2]
        
        # Process status
        if (not isinstance(status_result, Exception) and
            status_result and facts_for_this_char < max_facts_per_char and
            len(facts_for_prompt_list) < max_total_facts):
            fact_text = f"- {char_name}'s status is: {status_result}."
            if fact_text not in facts_for_prompt_list:
                facts_for_prompt_list.append(fact_text)
                facts_for_this_char += 1
        elif isinstance(status_result, Exception):
            logger.warning(f"KG Query for {char_name}'s status failed: {status_result}")
        
        # Process location
        if (not isinstance(location_result, Exception) and
            location_result and facts_for_this_char < max_facts_per_char and
            len(facts_for_prompt_list) < max_total_facts):
            fact_text = f"- {char_name} is located in: {location_result}."
            if fact_text not in facts_for_prompt_list:
                facts_for_prompt_list.append(fact_text)
                facts_for_this_char += 1
        elif isinstance(location_result, Exception):
            logger.warning(f"KG Query for {char_name}'s location failed: {location_result}")
        
        # Process relationships
        if (not isinstance(relationships_result, Exception) and
            facts_for_this_char < max_facts_per_char and
            len(facts_for_prompt_list) < max_total_facts):
            interesting_rel_types = [
                "ally_of",
                "enemy_of",
                "mentor_of",
                "protege_of",
                "works_for",
                "related_to",
            ]
            for rel_res in relationships_result:
                if (facts_for_this_char >= max_facts_per_char or
                    len(facts_for_prompt_list) >= max_total_facts):
                    break
                if rel_res.get("predicate") in interesting_rel_types:
                    rel_type_display = rel_res["predicate"].replace("_", " ")
                    fact_text = f"- {char_name} has a key relationship ({rel_type_display}) with: {rel_res.get('object')}."
                    if fact_text not in facts_for_prompt_list:
                        facts_for_prompt_list.append(fact_text)
                        facts_for_this_char += 1
        elif isinstance(relationships_result, Exception):
            logger.warning(f"KG Query for {char_name}'s relationships failed: {relationships_result}")

    if not facts_for_prompt_list:
        return "No specific reliable KG facts identified as highly relevant for this chapter's current focus from Neo4j."

    unique_facts = sorted(list(set(facts_for_prompt_list)))

    final_prompt_parts = [
        "**Key Reliable KG Facts (from Neo4j - up to previous chapter/initial state):**"
    ]
    final_prompt_parts.extend(unique_facts[:max_total_facts])
    return "\n".join(final_prompt_parts)
