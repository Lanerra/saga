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
    """Fill missing character profile data via LLM."""
    tasks: dict[tuple[str, str], Coroutine] = {}
    usage_data: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    
    # Initialize StateTracker if not provided
    if state_tracker is None:
        state_tracker = StateTracker()

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
            if field == "name" and value != name:
                # Character name was changed from placeholder
                # Check for conflicts before renaming
                existing_metadata = await state_tracker.check(value)
                if existing_metadata:
                    logger.warning(
                        "Character name conflict detected",
                        old_name=name,
                        new_name=value,
                        existing_type=existing_metadata["type"]
                    )
                    # Skip rename if name already reserved
                    continue
                
                name_changes[name] = value
                character_profiles[name].name = value
                
                # Reserve the new name
                description = character_profiles[name].description or "Character"
                await state_tracker.reserve(value, "character", description)
                
            elif field == "description":
                character_profiles[name].description = value
                
                # Check for similar descriptions
                similar_name = await state_tracker.has_similar_description(value, "character")
                if similar_name:
                    logger.warning(
                        "Similar character description found",
                        current_name=name,
                        similar_to=similar_name
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
