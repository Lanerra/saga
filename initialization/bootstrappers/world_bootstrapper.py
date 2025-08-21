import asyncio
from collections.abc import Coroutine
from typing import Any

import structlog

import config
import utils
from models import WorldItem

from .common import bootstrap_field

logger = structlog.get_logger(__name__)

WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL: dict[str, str] = {
    "overview": "_overview_",
    "locations": "locations",
    "society": "society",
    "systems": "systems",
    "lore": "lore",
    "history": "history",
    "factions": "factions",
}

WORLD_DETAIL_LIST_INTERNAL_KEYS: list[str] = []


async def generate_world_building_logic(
    world_building: dict[str, Any], plot_outline: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, int] | None]:
    """Stub world-building generation function."""
    logger.warning("generate_world_building_logic stub called")
    if not world_building:
        world_building = create_default_world()
    return world_building, None


def create_default_world() -> dict[str, dict[str, WorldItem]]:
    """Create a default world-building structure."""
    world_data: dict[str, dict[str, WorldItem]] = {
        "_overview_": {
            "_overview_": WorldItem.from_dict(
                "_overview_",
                "_overview_",
                {
                    "description": config.CONFIGURED_SETTING_DESCRIPTION,
                    "source": "default_overview",
                    "id": f"{utils._normalize_for_id('_overview_')}_{utils._normalize_for_id('_overview_')}",
                },
            )
        },
        "is_default": True,  # type: ignore
        "source": "default_fallback",  # type: ignore
    }

    standard_categories = [
        "locations",
        "society",
        "systems",
        "lore",
        "history",
        "factions",
    ]

    # Create empty world elements for each category instead of using placeholders
    for cat_key in standard_categories:
        # Create an empty WorldItem with empty name and description
        world_data[cat_key] = {
            "": WorldItem.from_dict(
                cat_key,
                "",  # Empty name instead of placeholder
                {
                    "description": "",
                    "source": "default_placeholder",
                    "id": f"{utils._normalize_for_id(cat_key)}_",
                },  # Empty description instead of placeholder
                allow_empty_name=True,  # Allow empty name during bootstrapping
            )
        }

    return world_data


async def bootstrap_world(
    world_building: dict[str, Any],
    plot_outline: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int] | None]:
    """Fill missing world-building information via LLM."""
    overall_usage_data: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def _accumulate_usage(item_usage: dict[str, int] | None) -> None:
        if item_usage:
            for key, val in item_usage.items():
                overall_usage_data[key] = overall_usage_data.get(key, 0) + val

    # Stage 0: Bootstrap _overview_ description
    if "_overview_" in world_building and "_overview_" in world_building["_overview_"]:
        overview_item_obj = world_building["_overview_"]["_overview_"]
        if isinstance(overview_item_obj, WorldItem) and utils._is_fill_in(
            overview_item_obj.description
        ):
            logger.info("Bootstrapping _overview_ description.")
            desc_value, desc_usage = await bootstrap_field(
                "description",
                {
                    "world_item": overview_item_obj.to_dict(),
                    "plot_outline": plot_outline,
                    "target_category": "_overview_",
                    "category_description": "Bootstrap a description for the world overview.",
                },
                "bootstrapper/fill_world_item_field.j2",
            )
            _accumulate_usage(desc_usage)
            if (
                desc_value
                and isinstance(desc_value, str)
                and not utils._is_fill_in(desc_value)
            ):
                overview_item_obj.description = desc_value
                current_source = overview_item_obj.additional_properties.get(
                    "source", ""
                )
                if isinstance(current_source, str):
                    overview_item_obj.additional_properties["source"] = (
                        f"{current_source}_descr_bootstrapped"
                        if current_source
                        else "descr_bootstrapped"
                    )
                else:
                    overview_item_obj.additional_properties["source"] = (
                        "descr_bootstrapped"
                    )

    # Stage 1: Bootstrap names for items with missing/empty names (sequential to prevent duplicates)
    items_needing_names: list[tuple[str, str, WorldItem]] = []
    for category, items_dict in world_building.items():
        if not isinstance(items_dict, dict) or category == "_overview_":
            continue
        for item_name, item_obj in items_dict.items():
            # Check if the item name is missing or empty (instead of just checking for placeholder)
            if isinstance(item_obj, WorldItem) and (
                not item_obj.name or not item_obj.name.strip()
            ):
                logger.info(
                    "Identified item for name bootstrapping in category '%s': Current name '%s'",
                    category,
                    item_name,
                )
                items_needing_names.append((category, item_name, item_obj))

    if items_needing_names:
        logger.info(
            "Found %d items requiring name bootstrapping.", len(items_needing_names)
        )

        # Process sequentially to inject context about existing names
        name_results_list = []
        name_task_keys = []
        generated_names: dict[str, str] = {}  # name -> category mapping

        for category, item_name, item_obj in items_needing_names:
            # Inject existing generated names to prevent duplicates
            existing_names_list = list(generated_names.keys())
            context_data = {
                "world_item": item_obj.to_dict(),
                "plot_outline": plot_outline,
                "target_category": category,
                "category_description": f"Bootstrap a name for a {category} element in the world.",
                "existing_world_names": existing_names_list,
            }

            # Attempt generation with retry for duplicates
            max_retries = 3
            generated_name = None
            name_usage = None

            for attempt in range(max_retries):
                temp_name, temp_usage = await bootstrap_field(
                    "name", context_data, "bootstrapper/fill_world_item_field.j2"
                )
                name_usage = temp_usage

                if (
                    temp_name
                    and isinstance(temp_name, str)
                    and temp_name.strip()
                    and not utils._is_fill_in(temp_name)
                    and temp_name != config.FILL_IN
                    and temp_name not in generated_names
                    and temp_name not in world_building.get(category, {})
                ):
                    generated_name = temp_name
                    generated_names[temp_name] = category
                    break
                else:
                    if attempt < max_retries - 1:
                        logger.warning(
                            "Name generation attempt %d for '%s/%s' resulted in duplicate or invalid name '%s'. Retrying with updated context.",
                            attempt + 1,
                            category,
                            item_name,
                            temp_name,
                        )
                        # Update context with more explicit diversity instruction
                        context_data["existing_world_names"] = list(
                            generated_names.keys()
                        )
                        context_data["retry_attempt"] = attempt + 1

            name_results_list.append((generated_name, name_usage))
            name_task_keys.append((category, item_name))

        new_items_to_add_stage1: dict[str, dict[str, WorldItem]] = {}
        items_to_remove_stage1: dict[str, list[str]] = {}

        for i, (new_name_value, name_usage) in enumerate(name_results_list):
            _accumulate_usage(name_usage)
            original_category, original_fill_in_name = name_task_keys[i]

            if (
                new_name_value
                and isinstance(new_name_value, str)
                and new_name_value.strip()
                and not utils._is_fill_in(new_name_value)
                and new_name_value != config.FILL_IN
            ):
                original_item_obj = world_building[original_category][
                    original_fill_in_name
                ]

                logger.info(
                    "Successfully bootstrapped name for '%s/%s': New name is '%s'",
                    original_category,
                    original_fill_in_name,
                    new_name_value,
                )
                properties_with_id = original_item_obj.to_dict()
                properties_with_id["id"] = (
                    f"{utils._normalize_for_id(original_category)}_{utils._normalize_for_id(new_name_value)}"
                )
                # Ensure we have a valid ID
                if not properties_with_id["id"] or properties_with_id["id"] == "_":
                    properties_with_id["id"] = (
                        f"element_{hash(original_category + new_name_value)}"
                    )
                new_item_renamed = WorldItem.from_dict(
                    original_category, new_name_value, properties_with_id
                )
                new_item_renamed.additional_properties["source"] = "bootstrapped_name"

                new_items_to_add_stage1.setdefault(original_category, {})[
                    new_name_value
                ] = new_item_renamed
                items_to_remove_stage1.setdefault(original_category, []).append(
                    original_fill_in_name
                )
            else:
                logger.warning(
                    "Name bootstrapping failed for item in category '%s' (original key: '%s'). Received: '%s'",
                    original_category,
                    original_fill_in_name,
                    new_name_value,
                )

        for cat, names_to_remove in items_to_remove_stage1.items():
            for name_key in names_to_remove:
                if name_key in world_building[cat]:
                    del world_building[cat][name_key]
        for cat, new_items_map in new_items_to_add_stage1.items():
            if cat not in world_building:
                world_building[cat] = {}
            world_building[cat].update(new_items_map)
        logger.info(
            "Finished applying name changes from Stage 1 to world_building structure."
        )

        # Cross-category duplicate name validation (should be minimal now due to sequential generation)
        all_bootstrapped_names = {}
        duplicate_count = 0
        for cat, items_dict in new_items_to_add_stage1.items():
            for item_name, item_obj in items_dict.items():
                normalized_name = utils._normalize_for_id(item_name)
                if normalized_name in all_bootstrapped_names:
                    logger.warning(
                        "Cross-category duplicate name detected: '%s' in category '%s' (previously used in '%s'). This should be rare with sequential generation.",
                        item_name,
                        cat,
                        all_bootstrapped_names[normalized_name],
                    )
                    duplicate_count += 1
                else:
                    all_bootstrapped_names[normalized_name] = cat

        logger.info(
            "Finished cross-category duplicate name validation. Found %d unique names across categories (%d duplicates).",
            len(all_bootstrapped_names),
            duplicate_count,
        )

    # Stage 2: Bootstrap properties for all items (excluding _overview_ top-level)
    property_bootstrap_tasks: dict[tuple[str, str, str], Coroutine] = {}
    for category, items_dict in world_building.items():
        if not isinstance(items_dict, dict) or category == "_overview_":
            continue
        for item_name, item_obj in items_dict.items():
            if not isinstance(item_obj, WorldItem) or (
                not item_name or not item_name.strip()
            ):
                continue
            # Check for properties that need bootstrapping
            # For structured fields like description, goals, rules, key_elements, traits
            if not item_obj.description:
                logger.info(
                    "Identified property 'description' for bootstrapping in item '%s/%s'.",
                    category,
                    item_name,
                )
                task_key = (category, item_name, "description")
                context_data = {
                    "world_item": item_obj.to_dict(),
                    "plot_outline": plot_outline,
                    "target_category": category,
                    "category_description": f"Bootstrap a description for a {category} element in the world.",
                }
                property_bootstrap_tasks[task_key] = bootstrap_field(
                    "description", context_data, "bootstrapper/fill_world_item_field.j2"
                )

            # Check additional_properties for other properties
            for prop_name, prop_value in item_obj.additional_properties.items():
                # Check if the property value is missing or empty (instead of just checking for placeholder)
                if not prop_value or (
                    isinstance(prop_value, str) and not prop_value.strip()
                ):
                    logger.info(
                        "Identified property '%s' for bootstrapping in item '%s/%s'.",
                        prop_name,
                        category,
                        item_name,
                    )
                    task_key = (category, item_name, prop_name)
                    context_data = {
                        "world_item": item_obj.to_dict(),
                        "plot_outline": plot_outline,
                        "target_category": category,
                        "category_description": f"Bootstrap a {prop_name} for a {category} element in the world.",
                    }
                    property_bootstrap_tasks[task_key] = bootstrap_field(
                        prop_name, context_data, "bootstrapper/fill_world_item_field.j2"
                    )

    if property_bootstrap_tasks:
        logger.info(
            "Found %d properties requiring bootstrapping.",
            len(property_bootstrap_tasks),
        )
        property_results_list = await asyncio.gather(*property_bootstrap_tasks.values())
        property_task_keys = list(property_bootstrap_tasks.keys())

        for i, (prop_fill_value, prop_usage) in enumerate(property_results_list):
            _accumulate_usage(prop_usage)
            category, item_name, prop_name_filled = property_task_keys[i]

            target_item = world_building.get(category, {}).get(item_name)
            if not target_item:
                logger.warning(
                    "Item %s/%s not found while trying to update property %s. Skipping.",
                    category,
                    item_name,
                    prop_name_filled,
                )
                continue

            if prop_fill_value is not None and (
                not isinstance(prop_fill_value, str)
                or (
                    isinstance(prop_fill_value, str)
                    and prop_fill_value.strip()
                    and not utils._is_fill_in(prop_fill_value)
                )
            ):
                logger.info(
                    "Successfully bootstrapped property '%s' for item '%s/%s'.",
                    prop_name_filled,
                    category,
                    item_name,
                )
                # Handle structured fields
                if prop_name_filled == "description":
                    target_item.description = prop_fill_value
                elif prop_name_filled == "goals":
                    target_item.goals = (
                        prop_fill_value
                        if isinstance(prop_fill_value, list)
                        else [prop_fill_value]
                    )
                elif prop_name_filled == "rules":
                    target_item.rules = (
                        prop_fill_value
                        if isinstance(prop_fill_value, list)
                        else [prop_fill_value]
                    )
                elif prop_name_filled == "key_elements":
                    target_item.key_elements = (
                        prop_fill_value
                        if isinstance(prop_fill_value, list)
                        else [prop_fill_value]
                    )
                elif prop_name_filled == "traits":
                    target_item.traits = (
                        prop_fill_value
                        if isinstance(prop_fill_value, list)
                        else [prop_fill_value]
                    )
                else:
                    # Handle additional properties
                    target_item.additional_properties[prop_name_filled] = (
                        prop_fill_value
                    )

                current_source = target_item.additional_properties.get("source", "")
                if isinstance(current_source, str):
                    append_source = f"_prop_{prop_name_filled}_bootstrapped"
                    if append_source not in current_source:
                        target_item.additional_properties["source"] = (
                            f"{current_source}{append_source}"
                            if current_source
                            else append_source.lstrip("_")
                        )
                else:
                    target_item.additional_properties["source"] = (
                        f"prop_{prop_name_filled}_bootstrapped"
                    )
            else:
                logger.warning(
                    "Property bootstrapping for '%s' in '%s/%s' resulted in empty or FILL_IN value.",
                    prop_name_filled,
                    category,
                    item_name,
                )

    if overall_usage_data["total_tokens"] > 0:
        world_building["is_default"] = False  # type: ignore
        current_top_source = world_building.get("source", "")
        if isinstance(current_top_source, str):
            world_building["source"] = (
                f"{current_top_source}_bootstrapped_items"
                if current_top_source and "default" not in current_top_source
                else "bootstrapped_items"
            )  # type: ignore
        else:
            world_building["source"] = "bootstrapped_items"  # type: ignore
        logger.info(
            "World building bootstrapping complete. Marking as not default and source as bootstrapped."
        )

    return world_building, overall_usage_data if overall_usage_data[
        "total_tokens"
    ] > 0 else None
