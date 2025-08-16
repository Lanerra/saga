import asyncio
from typing import Any, Coroutine, Dict, List, Optional, Tuple

import structlog

import config
import utils
from models import WorldItem

from .common import bootstrap_field

logger = structlog.get_logger(__name__)

WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL: Dict[str, str] = {
    "overview": "_overview_",
    "locations": "locations",
    "society": "society",
    "systems": "systems",
    "lore": "lore",
    "history": "history",
    "factions": "factions",
}

WORLD_DETAIL_LIST_INTERNAL_KEYS: List[str] = []


async def generate_world_building_logic(
    world_building: Dict[str, Any], plot_outline: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
    """Stub world-building generation function."""
    logger.warning("generate_world_building_logic stub called")
    if not world_building:
        world_building = create_default_world()
    return world_building, None


def create_default_world() -> Dict[str, Dict[str, WorldItem]]:
    """Create a default world-building structure."""
    world_data: Dict[str, Dict[str, WorldItem]] = {
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

    for cat_key in standard_categories:
        world_data[cat_key] = {
            config.FILL_IN: WorldItem.from_dict(
                cat_key,
                config.FILL_IN,
                {"description": config.FILL_IN, "source": "default_placeholder", "id": f"{utils._normalize_for_id(cat_key)}_{utils._normalize_for_id(config.FILL_IN)}"},
            )
        }

    return world_data


async def bootstrap_world(
    world_building: Dict[str, Any],
    plot_outline: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
    """Fill missing world-building information via LLM."""
    overall_usage_data: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def _accumulate_usage(item_usage: Optional[Dict[str, int]]) -> None:
        if item_usage:
            for key, val in item_usage.items():
                overall_usage_data[key] = overall_usage_data.get(key, 0) + val

    # Stage 0: Bootstrap _overview_ description
    if "_overview_" in world_building and "_overview_" in world_building["_overview_"]:
        overview_item_obj = world_building["_overview_"]["_overview_"]
        if isinstance(overview_item_obj, WorldItem) and utils._is_fill_in(
            overview_item_obj.properties.get("description")
        ):
            logger.info("Bootstrapping _overview_ description.")
            desc_value, desc_usage = await bootstrap_field(
                "description",
                {
                    "world_item": overview_item_obj.to_dict(),
                    "plot_outline": plot_outline,
                    "target_category": "_overview_",
                },
                "bootstrapper/fill_world_item_field.j2",
            )
            _accumulate_usage(desc_usage)
            if (
                desc_value
                and isinstance(desc_value, str)
                and not utils._is_fill_in(desc_value)
            ):
                overview_item_obj.properties["description"] = desc_value
                current_source = overview_item_obj.properties.get("source", "")
                if isinstance(current_source, str):
                    overview_item_obj.properties["source"] = (
                        f"{current_source}_descr_bootstrapped"
                        if current_source
                        else "descr_bootstrapped"
                    )
                else:
                    overview_item_obj.properties["source"] = "descr_bootstrapped"

    # Stage 1: Bootstrap names for [Fill-in] items
    name_bootstrap_tasks: Dict[Tuple[str, str], Coroutine] = {}
    for category, items_dict in world_building.items():
        if not isinstance(items_dict, dict) or category == "_overview_":
            continue
        for item_name, item_obj in items_dict.items():
            if isinstance(item_obj, WorldItem) and utils._is_fill_in(item_obj.name):
                logger.info(
                    "Identified item for name bootstrapping in category '%s': Current name '%s'",
                    category,
                    item_name,
                )
                task_key = (category, item_name)
                context_data = {
                    "world_item": item_obj.to_dict(),
                    "plot_outline": plot_outline,
                    "target_category": category,
                }
                name_bootstrap_tasks[task_key] = bootstrap_field(
                    "name", context_data, "bootstrapper/fill_world_item_field.j2"
                )

    if name_bootstrap_tasks:
        logger.info(
            "Found %d items requiring name bootstrapping.", len(name_bootstrap_tasks)
        )
        name_results_list = await asyncio.gather(*name_bootstrap_tasks.values())
        name_task_keys = list(name_bootstrap_tasks.keys())

        new_items_to_add_stage1: Dict[str, Dict[str, WorldItem]] = {}
        items_to_remove_stage1: Dict[str, List[str]] = {}

        for i, (new_name_value, name_usage) in enumerate(name_results_list):
            _accumulate_usage(name_usage)
            original_category, original_fill_in_name = name_task_keys[i]

            if (
                new_name_value
                and isinstance(new_name_value, str)
                and not utils._is_fill_in(new_name_value)
                and new_name_value != config.FILL_IN
            ):
                original_item_obj = world_building[original_category][
                    original_fill_in_name
                ]

                if new_name_value in world_building[
                    original_category
                ] or new_name_value in new_items_to_add_stage1.get(
                    original_category, {}
                ):
                    logger.warning(
                        "Name bootstrap for '%s/%s' resulted in a duplicate name '%s'. Skipping this item.",
                        original_category,
                        original_fill_in_name,
                        new_name_value,
                    )
                    continue

                logger.info(
                    "Successfully bootstrapped name for '%s/%s': New name is '%s'",
                    original_category,
                    original_fill_in_name,
                    new_name_value,
                )
                properties_with_id = {
                    **original_item_obj.properties,
                    "id": f"{utils._normalize_for_id(original_category)}_{utils._normalize_for_id(new_name_value)}"
                }
                # Ensure we have a valid ID
                if not properties_with_id["id"] or properties_with_id["id"] == "_":
                    properties_with_id["id"] = f"element_{hash(original_category + new_name_value)}"
                new_item_renamed = WorldItem.from_dict(
                    original_category, new_name_value, properties_with_id
                )
                new_item_renamed.properties["source"] = "bootstrapped_name"

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

    # Stage 2: Bootstrap properties for all items (excluding _overview_ top-level)
    property_bootstrap_tasks: Dict[Tuple[str, str, str], Coroutine] = {}
    for category, items_dict in world_building.items():
        if not isinstance(items_dict, dict) or category == "_overview_":
            continue
        for item_name, item_obj in items_dict.items():
            if not isinstance(item_obj, WorldItem) or utils._is_fill_in(item_name):
                continue
            for prop_name, prop_value in item_obj.properties.items():
                if utils._is_fill_in(prop_value):
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

            if prop_fill_value is not None and not (
                isinstance(prop_fill_value, str) and utils._is_fill_in(prop_fill_value)
            ):
                logger.info(
                    "Successfully bootstrapped property '%s' for item '%s/%s'.",
                    prop_name_filled,
                    category,
                    item_name,
                )
                target_item.properties[prop_name_filled] = prop_fill_value

                current_source = target_item.properties.get("source", "")
                if isinstance(current_source, str):
                    append_source = f"_prop_{prop_name_filled}_bootstrapped"
                    if append_source not in current_source:
                        target_item.properties["source"] = (
                            f"{current_source}{append_source}"
                            if current_source
                            else append_source.lstrip("_")
                        )
                else:
                    target_item.properties["source"] = (
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
