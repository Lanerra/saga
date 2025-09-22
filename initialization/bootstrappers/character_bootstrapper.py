# initialization/bootstrappers/character_bootstrapper.py
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


async def _try_generate_unique_name(
    base_context: dict[str, Any],
    state_tracker: StateTracker,
    existing_profile_names: list[str],
    max_attempts: int = 3,
) -> str | None:
    """Attempt to generate a unique, narrative-appropriate name via LLM with retries.

    Tries the conflict-specific template first, then falls back to the general
    field template with added diversity instructions. Verifies uniqueness against
    StateTracker and existing profiles after each attempt.
    """
    context = dict(base_context)
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            if attempt == 1:
                candidate, _ = await bootstrap_field(
                    "name",
                    context,
                    "bootstrapper/fill_character_name_conflict.j2",
                )
            else:
                # Enrich context to push for diversity on subsequent attempts
                context["diversity_instruction"] = (
                    "Provide a distinct, non-derivative name unlike any listed."
                )
                candidate, _ = await bootstrap_field(
                    "name",
                    context,
                    "bootstrapper/fill_character_field.j2",
                )
        except Exception:
            candidate = None

        if not candidate or not isinstance(candidate, str) or not candidate.strip():
            continue

        # Uniqueness checks
        if await state_tracker.check(candidate):
            continue
        if candidate in existing_profile_names:
            continue
        return candidate

    return None


def _morph_name_variants(base: str) -> list[str]:
    """Generate deterministic, human-like variants of a name without numbering.

    Applies simple phonetic tweaks and token swaps to produce distinct but
    related alternatives. This is intentionally lightweight and local-first.
    """
    variants: list[str] = []
    name = base.strip()
    if not name:
        return variants

    # Basic vowel swaps
    vowel_swaps = {"e": "a", "a": "e", "i": "y", "o": "u", "u": "o"}
    swapped = "".join(vowel_swaps.get(ch, ch) for ch in name)
    if swapped != name:
        variants.append(swapped)

    # Soft consonant tweaks
    cons_swaps = {"v": "w", "ph": "f", "c": "k", "s": "z"}
    cons_variant = name
    for src, dst in cons_swaps.items():
        cons_variant = cons_variant.replace(src, dst).replace(src.upper(), dst.upper())
    if cons_variant != name:
        variants.append(cons_variant)

    # Token rearrangement for two-token names
    tokens = name.split()
    if len(tokens) == 2:
        variants.append(f"{tokens[1]} {tokens[0]}")

    # Minimal suffix alteration (non-numeric)
    if not name.endswith("on"):
        variants.append(name + "on")
    elif not name.endswith("ar"):
        variants.append(name[:-2] + "ar")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_variants = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique_variants.append(v)
    return unique_variants


async def bootstrap_characters(
    character_profiles: dict[str, CharacterProfile],
    plot_outline: dict[str, Any],
    state_tracker: StateTracker | None = None,
    world_building: dict[str, Any] | None = None,
) -> tuple[dict[str, CharacterProfile], dict[str, int] | None]:
    """Fill missing character profile data via LLM with proactive shared state management."""
    # Sequential processing replaces parallel task batching
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
    placeholder_reservations = {}
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
            placeholder_reservations[name] = True
            logger.debug(f"Reserved placeholder name: {name}")

    # Also reserve the actual protagonist name if it exists and is not a placeholder
    protagonist_name = (
        plot_outline.get("protagonist_name", "") or config.DEFAULT_PROTAGONIST_NAME
    )
    if protagonist_name and protagonist_name not in [
        "Antagonist",
        "SupportingChar1",
        "SupportingChar2",
        "SupportingChar3",
    ]:
        temp_desc = "Main protagonist character"
        await state_tracker.reserve(protagonist_name, "character", temp_desc)
        logger.debug(f"Reserved protagonist name: {protagonist_name}")

    # Prepare an optional serializable world context for prompts
    serial_world: dict[str, Any] | None = None
    if isinstance(world_building, dict) and world_building:
        try:
            serial_world = {}
            for cat, items in world_building.items():
                if not isinstance(items, dict):
                    continue
                serial_world[cat] = {}
                for item_name, item in items.items():
                    if hasattr(item, "to_dict"):
                        serial_world[cat][item_name] = item.to_dict()
                    elif isinstance(item, dict):
                        serial_world[cat][item_name] = item
                    else:
                        serial_world[cat][item_name] = str(item)
        except Exception:
            serial_world = None

    for name, profile in character_profiles.items():
        profile_dict = profile.to_dict()
        profile_dict["name"] = (
            profile.name
        )  # Include name in profile dict for template access
        context = {"profile": profile_dict, "plot_outline": plot_outline}
        if serial_world is not None:
            context["world"] = serial_world
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

    # Track name changes for profile key updates
    name_changes = {}

    # Process each character field sequentially to reduce similar-sounding outputs
    for (name, field), coro in tasks.items():
        value, usage = await coro
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
                    profile_dict = character_profiles[name].to_dict()
                    profile_dict["name"] = character_profiles[name].name
                    conflict_context = {
                        "profile": profile_dict,
                        "plot_outline": plot_outline,
                        "conflicting_name": value,
                        "existing_names": [p.name for p in character_profiles.values()],
                    }
                    if serial_world is not None:
                        conflict_context["world"] = serial_world
                    # Make additional LLM call to generate unique name using conflict resolution template
                    unique_name_result = await _try_generate_unique_name(
                        conflict_context,
                        state_tracker,
                        [p.name for p in character_profiles.values()],
                    )
                    if unique_name_result:
                        value = unique_name_result
                        logger.info(
                            "Generated unique narrative name to resolve conflict",
                            old_name=name,
                            unique_name=value,
                        )
                    else:
                        # Try deterministic, human-like variants before giving up
                        for variant in _morph_name_variants(value):
                            if not await state_tracker.check(
                                variant
                            ) and variant not in [
                                p.name for p in character_profiles.values()
                            ]:
                                value = variant
                                logger.info(
                                    "Resolved conflict using local variant",
                                    old_name=name,
                                    unique_name=value,
                                )
                                break

                # Even if no conflict, ensure we have a valid name change
                # This handles cases where LLM might return the same name or invalid response
                if value == name:
                    # LLM returned the same name, generate a unique one using LLM
                    logger.info(
                        "LLM returned same name, generating unique narrative name",
                        old_name=name,
                        same_name=value,
                    )
                    profile_dict = character_profiles[name].to_dict()
                    profile_dict["name"] = character_profiles[name].name
                    conflict_context = {
                        "profile": profile_dict,
                        "plot_outline": plot_outline,
                        "conflicting_name": value,
                        "existing_names": [p.name for p in character_profiles.values()],
                        "issue": "LLM returned identical name",
                    }
                    if serial_world is not None:
                        conflict_context["world"] = serial_world
                    unique_name_result = await _try_generate_unique_name(
                        conflict_context,
                        state_tracker,
                        [p.name for p in character_profiles.values()],
                    )
                    if unique_name_result and unique_name_result != name:
                        value = unique_name_result
                        logger.info(
                            "Generated unique narrative name for unchanged placeholder",
                            old_name=name,
                            new_name=value,
                        )
                    else:
                        for variant in _morph_name_variants(name):
                            if not await state_tracker.check(
                                variant
                            ) and variant not in [
                                p.name for p in character_profiles.values()
                            ]:
                                value = variant
                                logger.info(
                                    "Resolved unchanged placeholder using local variant",
                                    old_name=name,
                                    new_name=value,
                                )
                                break

                name_changes[name] = value
                character_profiles[name].name = value
                logger.info(f"Character name updated: {name} -> {value}")

                # Update StateTracker atomically: rename placeholder -> new value
                try:
                    renamed = await state_tracker.rename(name, value)
                    if not renamed:
                        # If rename wasn't possible (e.g., placeholder wasn't reserved), reserve the new name
                        description = (
                            character_profiles[name].description or "Character"
                        )
                        await state_tracker.reserve(value, "character", description)
                except Exception:
                    # Non-fatal: validator reconciliation will attempt to repair
                    pass

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
                    profile_dict = character_profiles[name].to_dict()
                    profile_dict["name"] = character_profiles[name].name
                    conflict_context = {
                        "profile": profile_dict,
                        "plot_outline": plot_outline,
                        "conflicting_name": value,
                        "existing_names": [p.name for p in character_profiles.values()],
                        "issue": "Placeholder conflicts with existing name",
                    }
                    if serial_world is not None:
                        conflict_context["world"] = serial_world
                    unique_name_result = await _try_generate_unique_name(
                        conflict_context,
                        state_tracker,
                        [p.name for p in character_profiles.values()],
                    )
                    if unique_name_result and unique_name_result != name:
                        value = unique_name_result
                        logger.info(
                            "Generated unique narrative name for conflicting placeholder",
                            old_name=name,
                            new_name=value,
                        )
                    else:
                        for variant in _morph_name_variants(name):
                            if not await state_tracker.check(
                                variant
                            ) and variant not in [
                                p.name for p in character_profiles.values()
                            ]:
                                value = variant
                                logger.info(
                                    "Resolved conflicting placeholder using local variant",
                                    old_name=name,
                                    new_name=value,
                                )
                                break
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
