# initialization/bootstrappers/world_bootstrapper.py
import asyncio
from collections.abc import Coroutine
from typing import Any

import structlog

import config
import utils
from models import WorldItem
from processing.state_tracker import StateTracker

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


# Enhanced world building targets
ENHANCED_WORLD_TARGETS = {
    "locations": 4,  # vs current ~1
    "society": 3,  # vs current ~1
    "factions": 3,  # vs current ~1
    "history": 2,  # vs current ~1
    "lore": 2,  # vs current ~1
    "systems": 2,  # vs current ~1
}



def create_default_world() -> dict[str, dict[str, WorldItem]]:
    """Create enhanced world-building structure with multiple elements per category."""
    world_data: dict[str, dict[str, WorldItem]] = {
        "_overview_": {
            "_overview_": WorldItem.from_dict(
                "_overview_",
                "_overview_",
                {
                    "description": config.CONFIGURED_SETTING_DESCRIPTION,
                    "source": "bootstrap_overview",
                    "id": f"{utils._normalize_for_id('_overview_')}_{utils._normalize_for_id('_overview_')}",
                },
            )
        },
        "is_default": True,  # type: ignore
        "source": "bootstrap_fallback",  # type: ignore
    }

    # Create multiple elements per category using enhanced targets
    for cat_key, target_count in ENHANCED_WORLD_TARGETS.items():
        world_data[cat_key] = {}

        # Create multiple placeholder elements per category
        for i in range(target_count):
            element_name = f"{cat_key}_element_{i+1}"  # Will be filled by LLM

            # Prepare for enhanced node typing (will be used during persistence)
            world_data[cat_key][element_name] = WorldItem.from_dict(
                cat_key,
                element_name,
                {
                    "description": "",  # To be filled by LLM
                    "source": "bootstrap_placeholder",
                    "id": f"{utils._normalize_for_id(cat_key)}_{i+1}",
                    "element_index": i + 1,  # For tracking during bootstrap
                },
                allow_empty_name=True,
            )

    return world_data


async def _bootstrap_world_overview(
    world_building: dict[str, Any],
    plot_outline: dict[str, Any],
) -> dict[str, int] | None:
    """Bootstrap the world overview description."""
    usage_data: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def _accumulate_usage(item_usage: dict[str, int] | None) -> None:
        if item_usage:
            for key, val in item_usage.items():
                usage_data[key] = usage_data.get(key, 0) + val

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

    return usage_data if usage_data["total_tokens"] > 0 else None


async def _bootstrap_world_names(
    world_building: dict[str, Any],
    plot_outline: dict[str, Any],
    state_tracker: StateTracker,
) -> dict[str, int] | None:
    """Bootstrap names for world items that need them."""
    usage_data: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def _accumulate_usage(item_usage: dict[str, int] | None) -> None:
        if item_usage:
            for key, val in item_usage.items():
                usage_data[key] = usage_data.get(key, 0) + val

    # Find items needing names
    items_needing_names: list[tuple[str, str, WorldItem]] = []
    for category, items_dict in world_building.items():
        if not isinstance(items_dict, dict) or category == "_overview_":
            continue
        for item_name, item_obj in items_dict.items():
            # Check if the item name is missing, empty, or a placeholder pattern
            if isinstance(item_obj, WorldItem) and (
                not item_obj.name
                or not item_obj.name.strip()
                or item_name.endswith(
                    ("_element_1", "_element_2", "_element_3", "_element_4")
                )
            ):
                logger.info(
                    "Identified item for name bootstrapping in category '%s': Current name '%s'",
                    category,
                    item_name,
                )
                items_needing_names.append((category, item_name, item_obj))

    if not items_needing_names:
        return None

    logger.info(
        "Found %d items requiring name bootstrapping.", len(items_needing_names)
    )

    # Process names in parallel with semaphore to limit concurrent LLM calls
    generated_names: dict[str, str] = {}  # name -> category mapping
    new_items_to_add_stage1: dict[str, dict[str, WorldItem]] = {}
    items_to_remove_stage1: dict[str, list[str]] = {}
    
    # Use parallel processing for better performance
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_LLM_CALLS)
    name_generation_tasks = []
    
    async def generate_name_for_item(
        category: str, item_name: str, item_obj: WorldItem
    ) -> tuple[str, str, WorldItem, str | None, dict[str, int] | None]:
        """Generate a name for a single item with retry logic."""
        async with semaphore:
            # Get snapshot of existing names at time of generation
            existing_names = set(generated_names.keys())
            existing_category_names = set(world_building.get(category, {}).keys())
            
            context_data = {
                "world_item": item_obj.to_dict(),
                "plot_outline": plot_outline,
                "target_category": category,
                "category_description": f"Bootstrap a name for a {category} element in the world.",
                "existing_world_names": list(existing_names),
            }
            
            max_retries = 3
            generated_name = None
            cumulative_usage = None
            
            for attempt in range(max_retries):
                temp_name, temp_usage = await bootstrap_field(
                    "name", context_data, "bootstrapper/fill_world_item_field.j2"
                )
                
                # Accumulate usage across retries
                if cumulative_usage is None:
                    cumulative_usage = temp_usage or {}
                elif temp_usage:
                    for key, val in temp_usage.items():
                        cumulative_usage[key] = cumulative_usage.get(key, 0) + val
                
                if (
                    temp_name
                    and isinstance(temp_name, str)
                    and temp_name.strip()
                    and not utils._is_fill_in(temp_name)
                    and temp_name != config.FILL_IN
                    and temp_name not in existing_names
                    and temp_name not in existing_category_names
                ):
                    # Check StateTracker for conflicts
                    existing_metadata = await state_tracker.check(temp_name)
                    if existing_metadata:
                        logger.debug(
                            f"StateTracker conflict for name '{temp_name}' - already reserved as {existing_metadata['type']}"
                        )
                        continue
                        
                    generated_name = temp_name
                    break
                else:
                    if attempt < max_retries - 1:
                        logger.debug(
                            f"Name generation attempt {attempt + 1} for '{category}/{item_name}' "
                            f"resulted in duplicate or invalid name '{temp_name}'. Retrying."
                        )
                        # Update context for retry
                        context_data["retry_attempt"] = attempt + 1
                        context_data["diversity_instruction"] = (
                            "Generate a unique name that does not match any of the existing names listed."
                        )
            
            return category, item_name, item_obj, generated_name, cumulative_usage
    
    # Create all name generation tasks
    for category, item_name, item_obj in items_needing_names:
        task = generate_name_for_item(category, item_name, item_obj)
        name_generation_tasks.append(task)
    
    logger.info(f"Starting parallel name generation for {len(name_generation_tasks)} items...")
    
    # Execute all name generation tasks in parallel
    name_results = await asyncio.gather(*name_generation_tasks, return_exceptions=True)
    
    # Process results and check for duplicates
    successful_generations = []
    failed_generations = []
    
    for result in name_results:
        if isinstance(result, Exception):
            from ..error_handling import handle_bootstrap_error, ErrorSeverity
            handle_bootstrap_error(
                result,
                "Parallel name generation task",
                ErrorSeverity.ERROR,
                {"task_type": "name_generation"}
            )
            continue
        
        category, item_name, item_obj, generated_name, name_usage = result
        _accumulate_usage(name_usage)
        
        if generated_name:
            successful_generations.append((category, item_name, item_obj, generated_name))
        else:
            failed_generations.append((category, item_name))
    
    # Handle duplicate resolution for parallel generation
    name_conflicts = {}
    final_assignments = {}
    
    for category, item_name, item_obj, generated_name in successful_generations:
        if generated_name not in name_conflicts:
            name_conflicts[generated_name] = []
        name_conflicts[generated_name].append((category, item_name, item_obj))
    
    # Resolve conflicts - first item wins, others get sequential retry
    for generated_name, conflicting_items in name_conflicts.items():
        if len(conflicting_items) == 1:
            # No conflict
            category, item_name, item_obj = conflicting_items[0]
            final_assignments[f"{category}:{item_name}"] = (item_obj, generated_name)
            generated_names[generated_name] = category
        else:
            # Conflict resolution: first item keeps name, others get retried
            logger.warning(
                f"Name conflict detected for '{generated_name}' among {len(conflicting_items)} items. "
                "Resolving with sequential fallback."
            )
            
            # First item gets the name
            category, item_name, item_obj = conflicting_items[0]
            final_assignments[f"{category}:{item_name}"] = (item_obj, generated_name)
            generated_names[generated_name] = category
            
            # Other items need new names (fallback to sequential)
            for category, item_name, item_obj in conflicting_items[1:]:
                failed_generations.append((category, item_name))
    
    # Sequential fallback for failed/conflicted items
    for category, item_name in failed_generations:
        # Find the original item_obj from items_needing_names
        item_obj = None
        for orig_cat, orig_name, orig_obj in items_needing_names:
            if orig_cat == category and orig_name == item_name:
                item_obj = orig_obj
                break
        
        if item_obj is None:
            logger.error(f"Could not find original item object for {category}:{item_name}")
            continue
        
        # Sequential retry with current generated_names context
        existing_names_list = list(generated_names.keys())
        context_data = {
            "world_item": item_obj.to_dict(),
            "plot_outline": plot_outline,
            "target_category": category,
            "category_description": f"Bootstrap a name for a {category} element in the world.",
            "existing_world_names": existing_names_list,
            "fallback_generation": True,
        }
        
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
                # Check StateTracker for conflicts  
                existing_metadata = await state_tracker.check(temp_name)
                if existing_metadata:
                    logger.debug(
                        f"StateTracker conflict for fallback name '{temp_name}' - already reserved as {existing_metadata['type']}"
                    )
                    if attempt < max_retries - 1:
                        context_data["existing_world_names"] = list(generated_names.keys())
                        context_data["retry_attempt"] = attempt + 1
                    continue
                    
                generated_name = temp_name
                generated_names[temp_name] = category
                break
            else:
                if attempt < max_retries - 1:
                    context_data["existing_world_names"] = list(generated_names.keys())
                    context_data["retry_attempt"] = attempt + 1
        
        _accumulate_usage(name_usage)
        
        if generated_name:
            # Reserve the name in StateTracker
            description = item_obj.description or f"{category} element"
            reservation_success = await state_tracker.reserve(generated_name, "world_item", description)
            if not reservation_success:
                logger.warning(
                    f"Failed to reserve name '{generated_name}' for world item {category}:{item_name}"
                )
            
            final_assignments[f"{category}:{item_name}"] = (item_obj, generated_name)
        else:
            logger.warning(
                f"Failed to generate name for '{category}/{item_name}' after parallel and sequential attempts."
            )
    
    # Update world items with final name assignments
    for item_key, (item_obj, generated_name) in final_assignments.items():
        category, item_name = item_key.split(":", 1)
        
        logger.info(
            f"Assigned name '{generated_name}' to item '{item_name}' in category '{category}'."
        )
        
        # Update the WorldItem object
        item_obj.name = generated_name
        
        # CRITICAL FIX: Update the ID to match the new name
        # This ensures the database uses proper IDs instead of placeholder numbers
        normalized_category = utils._normalize_for_id(category)
        normalized_name = utils._normalize_for_id(generated_name)
        new_id = f"{normalized_category}_{normalized_name}" if normalized_category and normalized_name else f"element_{hash(category + generated_name)}"
        item_obj.id = new_id
        
        current_source = item_obj.additional_properties.get("source", "")
        if isinstance(current_source, str):
            item_obj.additional_properties["source"] = (
                f"{current_source}_name_bootstrapped"
                if current_source
                else "name_bootstrapped"
            )
        else:
            item_obj.additional_properties["source"] = "name_bootstrapped"
        
        # Add to the new items list for world_building dictionary update
        new_items_to_add_stage1.setdefault(category, {})[generated_name] = item_obj
        items_to_remove_stage1.setdefault(category, []).append(item_name)
    
    logger.info(
        f"Parallel name generation complete. Successfully generated {len(final_assignments)} names "
        f"out of {len(items_needing_names)} requested."
    )

    # Update world_building dictionary
    for category_to_update, items_to_remove in items_to_remove_stage1.items():
        for old_name in items_to_remove:
            if (
                category_to_update in world_building
                and old_name in world_building[category_to_update]
            ):
                del world_building[category_to_update][old_name]

    for category_to_update, items_to_add in new_items_to_add_stage1.items():
        if category_to_update not in world_building:
            world_building[category_to_update] = {}
        for new_name, new_item in items_to_add.items():
            world_building[category_to_update][new_name] = new_item

    # Cross-category duplicate name validation
    if new_items_to_add_stage1:
        logger.info("Running cross-category duplicate name validation.")
        all_bootstrapped_names = {}
        duplicate_count = 0
        for cat, items_dict in new_items_to_add_stage1.items():
            for item_name, _ in items_dict.items():
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

    return usage_data if usage_data["total_tokens"] > 0 else None


async def _bootstrap_world_properties(
    world_building: dict[str, Any],
    plot_outline: dict[str, Any],
) -> dict[str, int] | None:
    """Bootstrap properties for world items."""
    usage_data: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def _accumulate_usage(item_usage: dict[str, int] | None) -> None:
        if item_usage:
            for key, val in item_usage.items():
                usage_data[key] = usage_data.get(key, 0) + val

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
                    f"Identified property 'description' for bootstrapping in item '{category}/{item_name}'."
                )
                task_key = (category, item_name, "description")
                property_bootstrap_tasks[task_key] = bootstrap_field(
                    "description",
                    {
                        "world_item": item_obj.to_dict(),
                        "plot_outline": plot_outline,
                        "target_category": category,
                        "category_description": f"Bootstrap a description for a {category} element named '{item_name}' in the world.",
                    },
                    "bootstrapper/fill_world_item_field.j2",
                )

    if property_bootstrap_tasks:
        logger.info(
            "Bootstrapping properties for %d world items.", len(property_bootstrap_tasks)
        )
        property_results = await asyncio.gather(
            *property_bootstrap_tasks.values(), return_exceptions=True
        )
        logger.info("Property bootstrapping phase complete.")

        # Process property results
        for (category, item_name, prop_name), result in zip(
            property_bootstrap_tasks.keys(), property_results, strict=False
        ):
            if isinstance(result, Exception):
                from ..error_handling import handle_bootstrap_error, ErrorSeverity
                handle_bootstrap_error(
                    result,
                    f"Property bootstrapping: {category}/{item_name}.{prop_name}",
                    ErrorSeverity.ERROR,
                    {"category": category, "item_name": item_name, "property": prop_name}
                )
                continue

            prop_value, prop_usage = result
            _accumulate_usage(prop_usage)

            target_item = world_building[category][item_name]
            prop_name_filled = prop_name

            if (
                prop_value
                and isinstance(prop_value, str)
                and not utils._is_fill_in(prop_value)
            ):
                # Set the property
                setattr(target_item, prop_name_filled, prop_value)
                current_source = target_item.additional_properties.get("source", "")
                if isinstance(current_source, str):
                    target_item.additional_properties["source"] = (
                        f"{current_source}_prop_{prop_name_filled}_bootstrapped"
                        if current_source
                        else f"prop_{prop_name_filled}_bootstrapped"
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

    return usage_data if usage_data["total_tokens"] > 0 else None


async def bootstrap_world(
    world_building: dict[str, Any],
    plot_outline: dict[str, Any],
    state_tracker: StateTracker | None = None,
) -> tuple[dict[str, Any], dict[str, int] | None]:
    """Fill missing world-building information via LLM."""
    overall_usage_data: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    
    # Initialize StateTracker if not provided
    if state_tracker is None:
        state_tracker = StateTracker()

    def _accumulate_usage(item_usage: dict[str, int] | None) -> None:
        if item_usage:
            for key, val in item_usage.items():
                overall_usage_data[key] = overall_usage_data.get(key, 0) + val

    # Stage 0: Bootstrap world overview
    overview_usage = await _bootstrap_world_overview(world_building, plot_outline)
    _accumulate_usage(overview_usage)

    # Stage 1: Bootstrap names for items
    names_usage = await _bootstrap_world_names(world_building, plot_outline, state_tracker)
    _accumulate_usage(names_usage)

    # Stage 2: Bootstrap properties for items
    properties_usage = await _bootstrap_world_properties(world_building, plot_outline)
    _accumulate_usage(properties_usage)

    # Final cleanup and metadata update
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
