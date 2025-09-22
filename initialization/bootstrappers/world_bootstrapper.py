# initialization/bootstrappers/world_bootstrapper.py
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

# Backward compatibility for old tests - these are referenced in xfail tests
WORLD_DETAIL_LIST_INTERNAL_KEYS = [
    "description",
    "goals",
    "rules",
    "key_elements",
    "traits",
    "features",
    "atmosphere",
    "history",
    "structure",
    "population",
    "notable_figures",
]


async def generate_world_building_logic(world_building, plot_outline):
    """Legacy function stub for backward compatibility with xfail tests."""
    # This function is no longer used - replaced by bootstrap_world
    # Only kept for test import compatibility
    raise NotImplementedError("This function has been replaced by bootstrap_world")


# Default: one seed element per category (empty name placeholder)
ENHANCED_WORLD_TARGETS = {
    "locations": 1,
    "society": 1,
    "factions": 1,
    "history": 1,
    "lore": 1,
    "systems": 1,
}


def _world_item_context(item: WorldItem) -> dict[str, Any]:
    """Build a context dict for prompts that always includes name and category.

    WorldItem.to_dict() intentionally omits core identifiers like name/category
    for storage concerns. Prompt templates, however, expect them. This helper
    merges those identifiers back in for safe Jinja rendering.
    """
    data = item.to_dict()
    # Ensure required identifiers are present for templates
    data["name"] = item.name
    data["category"] = item.category
    data["id"] = item.id
    return data


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

    # Create seed elements per category using targets (default 1 each)
    for cat_key, target_count in ENHANCED_WORLD_TARGETS.items():
        world_data[cat_key] = {}

        # Create placeholder elements per category with empty name as key
        for i in range(target_count):
            element_key = ""  # empty-name placeholder; name will be filled by LLM

            # Build seed item; allow empty name
            world_data[cat_key][element_key] = WorldItem.from_dict(
                cat_key,
                "",
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
    state_tracker: StateTracker | None = None,
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
                    "world_item": _world_item_context(overview_item_obj),
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

    # Ensure '_overview_' is tracked in StateTracker to avoid reconciliation warnings
    if state_tracker is not None:
        try:
            existing = await state_tracker.check("_overview_")
            if not existing:
                await state_tracker.reserve(
                    "_overview_",
                    "world_item",
                    str(getattr(overview_item_obj, "description", ""))
                    if "overview_item_obj" in locals()
                    else "",
                )
        except Exception:
            # Non-fatal; validator reconciliation will attempt to repair
            pass

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
                not item_obj.name or not item_obj.name.strip() or not item_name.strip()
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

    # Strictly sequential name generation to avoid similar-sounding duplicates
    generated_names: dict[str, str] = {}  # name -> category mapping
    new_items_to_add_stage1: dict[str, dict[str, WorldItem]] = {}
    items_to_remove_stage1: dict[str, list[str]] = {}

    async def generate_name_for_item(
        category: str, item_name: str, item_obj: WorldItem
    ) -> tuple[str, str, WorldItem, str | None, dict[str, int] | None]:
        """Generate a name for a single item with retry logic using live StateTracker context."""
        # Get live context from StateTracker instead of stale snapshots
        tracked_entities = await state_tracker.get_all()
        existing_names = set(tracked_entities.keys()).union(set(generated_names.keys()))
        existing_category_names = set(world_building.get(category, {}).keys())

        context_data = {
            "world_item": _world_item_context(item_obj),
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

    # Execute name generation strictly sequentially
    logger.info(
        f"Starting sequential name generation for {len(items_needing_names)} items..."
    )
    name_results = []
    for category, item_name, item_obj in items_needing_names:
        try:
            result = await generate_name_for_item(category, item_name, item_obj)
            name_results.append(result)
            # Proactively record generated names to inform subsequent sequential generations
            if (
                isinstance(result, tuple)
                and len(result) == 5
                and isinstance(result[3], str)
                and result[3]
            ):
                generated_names[result[3]] = category
        except Exception as e:
            name_results.append(e)

    # Process results and check for duplicates
    successful_generations = []
    failed_generations = []

    for result in name_results:
        if isinstance(result, Exception):
            from ..error_handling import ErrorSeverity, handle_bootstrap_error

            handle_bootstrap_error(
                result,
                "Sequential name generation task",
                ErrorSeverity.ERROR,
                {"task_type": "name_generation"},
            )
            continue

        category, item_name, item_obj, generated_name, name_usage = result
        _accumulate_usage(name_usage)

        if generated_name:
            successful_generations.append(
                (category, item_name, item_obj, generated_name)
            )
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
            logger.error(
                f"Could not find original item object for {category}:{item_name}"
            )
            continue

        # Sequential retry with live StateTracker context
        tracked_entities = await state_tracker.get_all()
        existing_names_list = set(tracked_entities.keys()).union(
            set(generated_names.keys())
        )
        context_data = {
            "world_item": _world_item_context(item_obj),
            "plot_outline": plot_outline,
            "target_category": category,
            "category_description": f"Bootstrap a name for a {category} element in the world.",
            "existing_world_names": list(existing_names_list),
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
                        # Update with live StateTracker context for retry
                        tracked_entities = await state_tracker.get_all()
                        context_data["existing_world_names"] = set(
                            tracked_entities.keys()
                        ).union(set(generated_names.keys()))
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
            reservation_success = await state_tracker.reserve(
                generated_name, "world_item", description
            )
            if not reservation_success:
                logger.warning(
                    f"Failed to reserve name '{generated_name}' for world item {category}:{item_name}"
                )

            final_assignments[f"{category}:{item_name}"] = (item_obj, generated_name)
        else:
            logger.warning(
                f"Failed to generate name for '{category}/{item_name}' after sequential attempts."
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
        new_id = (
            f"{normalized_category}_{normalized_name}"
            if normalized_category and normalized_name
            else f"element_{hash(category + generated_name)}"
        )
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

        # Reserve the new name in StateTracker for ALL successful generations
        try:
            description = item_obj.description or f"{category} element"
            await state_tracker.reserve(generated_name, "world_item", description)
        except Exception:
            # Non-fatal: validation will emit a warning if untracked
            pass

        # Add to the new items list for world_building dictionary update
        new_items_to_add_stage1.setdefault(category, {})[generated_name] = item_obj
        items_to_remove_stage1.setdefault(category, []).append(item_name)

    logger.info(
        f"Sequential name generation complete. Successfully generated {len(final_assignments)} names "
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
                        "world_item": _world_item_context(item_obj),
                        "plot_outline": plot_outline,
                        "target_category": category,
                        "category_description": f"Bootstrap a description for a {category} element named '{item_name}' in the world.",
                    },
                    "bootstrapper/fill_world_item_field.j2",
                )

            # Add minimal list-valued properties (goals, rules, key_elements) with 1-2 items each
            # Only bootstrap if the lists are empty to avoid overwriting user-provided data
            list_properties_to_bootstrap = [
                ("goals", "goals", "1-2 key goals or objectives"),
                ("rules", "rules", "1-2 important rules or principles"),
                (
                    "key_elements",
                    "key_elements",
                    "1-2 distinctive features or characteristics",
                ),
            ]

            for prop_name, prop_field, prop_description in list_properties_to_bootstrap:
                current_value = getattr(item_obj, prop_name, None)
                if not current_value or len(current_value) == 0:
                    logger.info(
                        f"Identified property '{prop_name}' for bootstrapping in item '{category}/{item_name}'."
                    )
                    task_key = (category, item_name, prop_name)
                    property_bootstrap_tasks[task_key] = bootstrap_field(
                        prop_name,
                        {
                            "world_item": _world_item_context(item_obj),
                            "plot_outline": plot_outline,
                            "target_category": category,
                            "category_description": f"Bootstrap {prop_description} for a {category} element named '{item_name}' in the world.",
                            "list_count": "1-2",  # Minimal count to respect token and hardware constraints
                        },
                        "bootstrapper/fill_world_item_field.j2",
                        is_list=True,  # This is a list property
                        list_count=2,  # Request 2 items for list properties
                    )

    if property_bootstrap_tasks:
        logger.info(
            "Bootstrapping properties sequentially for %d world items.",
            len(property_bootstrap_tasks),
        )
        # Process property results sequentially to avoid parallel LLM calls
        for (category, item_name, prop_name), coro in property_bootstrap_tasks.items():
            try:
                prop_value, prop_usage = await coro
            except Exception as result:
                from ..error_handling import ErrorSeverity, handle_bootstrap_error

                handle_bootstrap_error(
                    result,
                    f"Property bootstrapping: {category}/{item_name}.{prop_name}",
                    ErrorSeverity.ERROR,
                    {
                        "category": category,
                        "item_name": item_name,
                        "property": prop_name,
                    },
                )
                continue
            _accumulate_usage(prop_usage)

            target_item = world_building[category][item_name]
            prop_name_filled = prop_name

            # Handle list properties (goals, rules, key_elements) differently from string properties
            if prop_name in ["goals", "rules", "key_elements"]:
                # For list properties, expect either a list or a string that can be converted to a list
                if prop_value and not utils._is_fill_in(str(prop_value)):
                    if isinstance(prop_value, list):
                        # Already a list, use as-is
                        list_value = prop_value
                    elif isinstance(prop_value, str):
                        # Convert string to list by splitting on common delimiters
                        if "," in prop_value:
                            list_value = [
                                item.strip()
                                for item in prop_value.split(",")
                                if item.strip()
                            ]
                        elif ";" in prop_value:
                            list_value = [
                                item.strip()
                                for item in prop_value.split(";")
                                if item.strip()
                            ]
                        elif "|" in prop_value:
                            list_value = [
                                item.strip()
                                for item in prop_value.split("|")
                                if item.strip()
                            ]
                        else:
                            # Treat as single item
                            list_value = [prop_value.strip()]
                    else:
                        # Convert other types to string and treat as single item
                        list_value = [str(prop_value).strip()]

                    # Filter out empty items but be more lenient with FILL_IN markers
                    # Only remove items that are entirely FILL_IN markers or empty
                    filtered_list_value = []
                    for item in list_value:
                        item_str = str(item).strip()
                        if item_str and item_str != config.FILL_IN:
                            # Remove FILL_IN markers from within the text but keep the rest
                            cleaned_item = item_str.replace(config.FILL_IN, "").strip()
                            if (
                                cleaned_item
                            ):  # Only add if there's content left after cleaning
                                filtered_list_value.append(cleaned_item)

                    if (
                        filtered_list_value
                    ):  # Only set if we have meaningful items after cleaning
                        setattr(target_item, prop_name_filled, filtered_list_value)
                        current_source = target_item.additional_properties.get(
                            "source", ""
                        )
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
                            "Property bootstrapping for '%s' in '%s/%s' resulted in empty list after filtering.",
                            prop_name_filled,
                            category,
                            item_name,
                        )
                else:
                    logger.warning(
                        "Property bootstrapping for '%s' in '%s/%s' resulted in empty or FILL_IN value.",
                        prop_name_filled,
                        category,
                        item_name,
                    )
            else:
                # Handle string properties (description)
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
    overview_usage = await _bootstrap_world_overview(
        world_building, plot_outline, state_tracker
    )
    _accumulate_usage(overview_usage)

    # Stage 1: Bootstrap names for items
    names_usage = await _bootstrap_world_names(
        world_building, plot_outline, state_tracker
    )
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
