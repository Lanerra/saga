# initialization/bootstrappers/character_bootstrapper.py
import asyncio
from collections.abc import Coroutine
from typing import Any

import structlog

import config
import utils
from models import CharacterProfile
from processing.state_tracker import StateTracker

from .common import bootstrap_field

logger = structlog.get_logger(__name__)


def create_default_characters(protagonist_name: str) -> dict[str, CharacterProfile]:
    """Create enhanced character roster with protagonist, antagonist, and supporting characters."""
    profiles = {}

    # Protagonist (enhanced with more fields)
    protagonist = CharacterProfile(name=protagonist_name)
    protagonist.description = config.FILL_IN
    protagonist.updates["role"] = "protagonist"
    profiles[protagonist_name] = protagonist

    # Antagonist
    antagonist_name = "Antagonist"  # Will be filled by LLM
    antagonist = CharacterProfile(name=antagonist_name)
    antagonist.description = config.FILL_IN
    antagonist.updates["role"] = "antagonist"
    profiles[antagonist_name] = antagonist

    # Supporting characters
    for i in range(3):
        support_name = f"SupportingChar{i+1}"  # Will be filled by LLM
        support = CharacterProfile(name=support_name)
        support.description = config.FILL_IN
        support.updates["role"] = "supporting"
        profiles[support_name] = support

    return profiles


async def bootstrap_characters(
    character_profiles: dict[str, CharacterProfile],
    plot_outline: dict[str, Any],
    state_tracker: StateTracker | None = None,
) -> tuple[dict[str, CharacterProfile], dict[str, int] | None]:
    """Fill missing character profile data via LLM with proactive shared state management."""
    tasks: dict[tuple[str, str], Coroutine] = {}
    usage_data: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    # Initialize StateTracker if not provided
    if state_tracker is None:
        state_tracker = StateTracker()

    # Pre-reserve all placeholder names to prevent conflicts during parallel generation
    for name, profile in character_profiles.items():
        if name in [
            "Antagonist",
            "SupportingChar1",
            "SupportingChar2",
            "SupportingChar3",
        ]:
            # Reserve placeholder names upfront with temporary descriptions
            temp_desc = f"Character placeholder for {name} role"
            await state_tracker.reserve(name, "character", temp_desc)
            logger.debug(f"Reserved placeholder name: {name}")

    # Also reserve the actual protagonist name if it exists and is not a placeholder
    protagonist_name = plot_outline.get("protagonist_name", "")
    if protagonist_name and protagonist_name not in [
        "Antagonist",
        "SupportingChar1",
        "SupportingChar2",
        "SupportingChar3",
    ]:
        temp_desc = "Main protagonist character"
        await state_tracker.reserve(protagonist_name, "character", temp_desc)
        logger.debug(f"Reserved protagonist name: {protagonist_name}")

    for name, profile in character_profiles.items():
        context = {"profile": profile.to_dict(), "plot_outline": plot_outline}
        role = profile.updates.get("role", "supporting")

        # Bootstrap character name if it's a placeholder
        if name in [
            "Antagonist",
            "SupportingChar1",
            "SupportingChar2",
            "SupportingChar3",
        ]:
            tasks[(name, "name")] = bootstrap_field(
                "name", context, "bootstrapper/fill_character_field.j2"
            )

        if not profile.description or utils._is_fill_in(profile.description):
            tasks[(name, "description")] = bootstrap_field(
                "description", context, "bootstrapper/fill_character_field.j2"
            )

        if not profile.status or utils._is_fill_in(profile.status):
            tasks[(name, "status")] = bootstrap_field(
                "status", context, "bootstrapper/fill_character_field.j2"
            )

        # Enhanced trait generation with role-based minimums
        trait_fill_count = sum(1 for t in profile.traits if utils._is_fill_in(t))
        role_min_traits = {
            "protagonist": config.BOOTSTRAP_MIN_TRAITS_PROTAGONIST,
            "antagonist": config.BOOTSTRAP_MIN_TRAITS_ANTAGONIST,
            "supporting": config.BOOTSTRAP_MIN_TRAITS_SUPPORTING,
        }.get(role, 3)

        if trait_fill_count or not profile.traits:
            tasks[(name, "traits")] = bootstrap_field(
                "traits",
                context,
                "bootstrapper/fill_character_field.j2",
                is_list=True,
                list_count=max(trait_fill_count, role_min_traits),
            )

        if "motivation" in profile.updates and utils._is_fill_in(
            profile.updates["motivation"]
        ):
            tasks[(name, "motivation")] = bootstrap_field(
                "motivation", context, "bootstrapper/fill_character_field.j2"
            )

    if not tasks:
        return character_profiles, None

    results = await asyncio.gather(*tasks.values())
    task_keys = list(tasks.keys())

    # Track name changes for profile key updates
    name_changes = {}

    for i, (value, usage) in enumerate(results):
        name, field = task_keys[i]
        if usage:
            for k, v in usage.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        usage_data[sub_k] = usage_data.get(sub_k, 0) + sub_v
                else:
                    usage_data[k] = usage_data.get(k, 0) + v
        if value:
            logger.debug(f"Processing character field update: {name}.{field} = {value}")
            if field == "name" and value != name:
                # Character name was changed from placeholder
                # Check for conflicts before renaming
                existing_metadata = await state_tracker.check(value)
                if existing_metadata:
                    logger.warning(
                        "Character name conflict detected",
                        old_name=name,
                        new_name=value,
                        existing_type=existing_metadata["type"],
                    )
                    # Generate a unique narrative-appropriate name to avoid conflicts
                    conflict_context = {
                        "profile": character_profiles[name].to_dict(),
                        "plot_outline": plot_outline,
                        "conflicting_name": value,
                        "existing_names": [p.name for p in character_profiles.values()],
                    }
                    # Make additional LLM call to generate unique name using conflict resolution template
                    try:
                        unique_name_result, _ = await bootstrap_field(
                            "name",
                            conflict_context,
                            "bootstrapper/fill_character_name_conflict.j2",
                        )
                        if unique_name_result and unique_name_result != value:
                            # Check if this new name also conflicts
                            while await state_tracker.check(
                                unique_name_result
                            ) or unique_name_result in [
                                p.name for p in character_profiles.values()
                            ]:
                                # If it still conflicts, try once more with explicit uniqueness request
                                conflict_context["previous_conflict"] = (
                                    unique_name_result
                                )
                                unique_name_result, _ = await bootstrap_field(
                                    "name",
                                    conflict_context,
                                    "bootstrapper/fill_character_field.j2",
                                )
                                break  # Avoid infinite loop

                            value = unique_name_result
                            logger.info(
                                "Generated unique narrative name to resolve conflict",
                                old_name=name,
                                conflicting_name=value,
                                unique_name=value,
                            )
                        else:
                            # Fallback to numbered approach if LLM fails
                            base_name = value
                            counter = 1
                            unique_name = f"{base_name} {counter}"
                            while await state_tracker.check(
                                unique_name
                            ) or unique_name in [
                                p.name for p in character_profiles.values()
                            ]:
                                counter += 1
                                unique_name = f"{base_name} {counter}"
                            value = unique_name
                            logger.info(
                                "Falling back to numbered name to resolve conflict",
                                old_name=name,
                                conflicting_name=value,
                                unique_name=unique_name,
                            )
                    except Exception as e:
                        # Fallback to numbered approach if LLM call fails
                        logger.warning(f"Failed to generate unique name via LLM: {e}")
                        base_name = value
                        counter = 1
                        unique_name = f"{base_name} {counter}"
                        while await state_tracker.check(unique_name) or unique_name in [
                            p.name for p in character_profiles.values()
                        ]:
                            counter += 1
                            unique_name = f"{base_name} {counter}"
                        value = unique_name
                        logger.info(
                            "Falling back to numbered name due to LLM error",
                            old_name=name,
                            conflicting_name=value,
                            unique_name=unique_name,
                        )

                # Even if no conflict, ensure we have a valid name change
                # This handles cases where LLM might return the same name or invalid response
                if value == name:
                    # LLM returned the same name, generate a unique one using LLM
                    logger.info(
                        "LLM returned same name, generating unique narrative name",
                        old_name=name,
                        same_name=value,
                    )
                    conflict_context = {
                        "profile": character_profiles[name].to_dict(),
                        "plot_outline": plot_outline,
                        "conflicting_name": value,
                        "existing_names": [p.name for p in character_profiles.values()],
                        "issue": "LLM returned identical name",
                    }
                    try:
                        unique_name_result, _ = await bootstrap_field(
                            "name",
                            conflict_context,
                            "bootstrapper/fill_character_name_conflict.j2",
                        )
                        if unique_name_result and unique_name_result != name:
                            value = unique_name_result
                            logger.info(
                                "Generated unique narrative name for unchanged placeholder",
                                old_name=name,
                                new_name=value,
                            )
                        else:
                            # Fallback to numbered approach
                            base_name = name
                            counter = 1
                            unique_name = f"{base_name} {counter}"
                            while await state_tracker.check(
                                unique_name
                            ) or unique_name in [
                                p.name for p in character_profiles.values()
                            ]:
                                counter += 1
                                unique_name = f"{base_name} {counter}"
                            value = unique_name
                            logger.info(
                                "Falling back to numbered name for unchanged placeholder",
                                old_name=name,
                                new_name=unique_name,
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate unique name for unchanged placeholder: {e}"
                        )
                        base_name = name
                        counter = 1
                        unique_name = f"{base_name} {counter}"
                        while await state_tracker.check(unique_name) or unique_name in [
                            p.name for p in character_profiles.values()
                        ]:
                            counter += 1
                            unique_name = f"{base_name} {counter}"
                        value = unique_name
                        logger.info(
                            "Falling back to numbered name due to error for unchanged placeholder",
                            old_name=name,
                            new_name=unique_name,
                        )

                name_changes[name] = value
                character_profiles[name].name = value
                logger.info(f"Character name updated: {name} -> {value}")

                # Reserve the new name
                description = character_profiles[name].description or "Character"
                await state_tracker.reserve(value, "character", description)

            elif field == "name" and value == name:
                # LLM returned the same name, but we still need to ensure it's unique
                logger.debug(f"Character name unchanged by LLM: {name}")
                # Check if this name conflicts with any reserved names
                existing_metadata = await state_tracker.check(value)
                if existing_metadata and name in [
                    "Antagonist",
                    "SupportingChar1",
                    "SupportingChar2",
                    "SupportingChar3",
                ]:
                    # This is a placeholder that conflicts with an existing name
                    logger.info(
                        "Placeholder conflicts with existing name, generating unique narrative name",
                        placeholder_name=name,
                        conflicting_with=value,
                    )
                    conflict_context = {
                        "profile": character_profiles[name].to_dict(),
                        "plot_outline": plot_outline,
                        "conflicting_name": value,
                        "existing_names": [p.name for p in character_profiles.values()],
                        "issue": "Placeholder conflicts with existing name",
                    }
                    try:
                        unique_name_result, _ = await bootstrap_field(
                            "name",
                            conflict_context,
                            "bootstrapper/fill_character_name_conflict.j2",
                        )
                        if unique_name_result and unique_name_result != name:
                            value = unique_name_result
                            logger.info(
                                "Generated unique narrative name for conflicting placeholder",
                                old_name=name,
                                new_name=value,
                            )
                        else:
                            # Fallback to numbered approach
                            base_name = name
                            counter = 1
                            unique_name = f"{base_name} {counter}"
                            while await state_tracker.check(
                                unique_name
                            ) or unique_name in [
                                p.name for p in character_profiles.values()
                            ]:
                                counter += 1
                                unique_name = f"{base_name} {counter}"
                            value = unique_name
                            logger.info(
                                "Falling back to numbered name for conflicting placeholder",
                                old_name=name,
                                new_name=unique_name,
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate unique name for conflicting placeholder: {e}"
                        )
                        base_name = name
                        counter = 1
                        unique_name = f"{base_name} {counter}"
                        while await state_tracker.check(unique_name) or unique_name in [
                            p.name for p in character_profiles.values()
                        ]:
                            counter += 1
                            unique_name = f"{base_name} {counter}"
                        value = unique_name
                        logger.info(
                            "Falling back to numbered name due to error for conflicting placeholder",
                            old_name=name,
                            new_name=unique_name,
                        )
                    name_changes[name] = value
                    character_profiles[name].name = value

            elif field == "description":
                character_profiles[name].description = value

                # Check for similar descriptions
                similar_name = await state_tracker.has_similar_description(
                    value, "character"
                )
                if similar_name:
                    logger.warning(
                        "Similar character description found",
                        current_name=name,
                        similar_to=similar_name,
                    )

            elif field == "traits":
                character_profiles[name].traits = value  # type: ignore
            elif field == "status":
                character_profiles[name].status = value
            else:  # motivation
                character_profiles[name].updates[field] = value
            character_profiles[name].updates["source"] = "bootstrapped"

    # Update dictionary keys for renamed characters
    for old_name, new_name in name_changes.items():
        if old_name in character_profiles:
            character_profiles[new_name] = character_profiles.pop(old_name)
            logger.info(f"Renamed character: {old_name} -> {new_name}")

    return character_profiles, usage_data if usage_data["total_tokens"] > 0 else None
