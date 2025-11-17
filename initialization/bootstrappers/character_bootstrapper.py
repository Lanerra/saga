# initialization/bootstrappers/character_bootstrapper.py
from collections.abc import Coroutine
from typing import Any

import structlog

import config
import utils
from models import CharacterProfile

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
    state_tracker: Any,
    existing_profile_names: list[str],
    max_attempts: int = 5,
) -> str | None:
    """Attempt to generate a unique, diverse name via LLM with enhanced diversity checks."""
    context = dict(base_context)
    attempt = 0
    used_names: set[str] = set()

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
                # Enrich context with diversity instructions on subsequent attempts
                context["diversity_instruction"] = (
                    "Provide a completely different name with different cultural origin, "
                    "phonetic structure, and visual appearance from any listed."
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

        candidate = candidate.strip()

        # Comprehensive uniqueness and diversity checks
        if await state_tracker.check(candidate):
            continue
        if candidate in existing_profile_names:
            continue
        if candidate in used_names:
            continue
        if _is_too_similar_to_existing(candidate, existing_profile_names):
            continue
        if _is_too_similar_to_used(candidate, used_names):
            continue

        used_names.add(candidate)
        return candidate

    # Fallback: use deterministic diverse generation
    for variant in _generate_diverse_name_variants(
        base_context.get("conflicting_name", "Generic"),
        existing_profile_names + list(used_names),
    ):
        if (
            not await state_tracker.check(variant)
            and variant not in existing_profile_names
            and variant not in used_names
            and not _is_too_similar_to_existing(variant, existing_profile_names)
        ):
            return variant

    return None


def _generate_diverse_name_variants(base: str, existing_names: list[str]) -> list[str]:
    """Generate truly diverse name variants using multiple strategies."""
    variants: list[str] = []
    name = base.strip()
    if not name:
        return variants

    # Strategy 1: Cultural/linguistic diversification
    cultural_variants = _get_cultural_variants(name, existing_names)
    variants.extend(cultural_variants)

    # Strategy 2: Structural diversification (first/last name separation)
    structural_variants = _get_structural_variants(name, existing_names)
    variants.extend(structural_variants)

    # Strategy 3: Phonetic diversification
    phonetic_variants = _get_phonetic_variants(name, existing_names)
    variants.extend(phonetic_variants)

    # Strategy 4: Genre-appropriate diversification
    genre_variants = _get_genre_variants(name, existing_names)
    variants.extend(genre_variants)

    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique_variants: list[str] = []
    for v in variants:
        if v not in seen and v != base:
            seen.add(v)
            unique_variants.append(v)
    return unique_variants


def _get_cultural_variants(base_name: str, existing_names: list[str]) -> list[str]:
    """Generate names from different cultural naming traditions."""
    variants: list[str] = []

    cultural_patterns = {
        "nordic": [
            "Astrid",
            "Bjorn",
            "Erik",
            "Freya",
            "Gunnar",
            "Ingrid",
            "Lars",
            "Sigrid",
        ],
        "japanese": [
            "Akira",
            "Hiroshi",
            "Kenji",
            "Mariko",
            "Nobu",
            "Sato",
            "Tanaka",
            "Yuki",
        ],
        "arabic": [
            "Amara",
            "Farid",
            "Hakim",
            "Leila",
            "Malik",
            "Nadia",
            "Omar",
            "Zara",
        ],
        "slavic": [
            "Boris",
            "Dmitri",
            "Ivana",
            "Katarina",
            "Mikhail",
            "Nikolai",
            "Svetlana",
        ],
        "celtic": [
            "Aiden",
            "Bran",
            "Cian",
            "Deirdre",
            "Finn",
            "Maeve",
            "Oisín",
            "Siobhan",
        ],
        "latin": [
            "Antonius",
            "Caesar",
            "Flavius",
            "Julius",
            "Lucius",
            "Marcus",
            "Octavius",
            "Valerius",
        ],
    }

    parts = base_name.split()
    if parts:
        for culture, names in cultural_patterns.items():
            if not _appears_to_be_cultural(base_name, culture):
                for nm in names[:3]:
                    variants.append(
                        f"{nm} {parts[-1] if len(parts) > 1 else _generate_generic_surname()}"
                    )

    return variants


def _get_structural_variants(base_name: str, existing_names: list[str]) -> list[str]:
    """Generate variants by changing name structure and components."""
    variants: list[str] = []
    parts = base_name.split()

    if len(parts) == 2:
        first, last = parts
        variants.append(f"{last} {first}")

        new_firsts = _generate_alternative_first_names(first)
        for new_first in new_firsts:
            variants.append(f"{new_first} {last}")

        new_lasts = _generate_alternative_last_names(last)
        for new_last in new_lasts:
            variants.append(f"{first} {new_last}")

    elif len(parts) == 1:
        new_lasts = _generate_alternative_last_names("Generic")
        for new_last in new_lasts:
            variants.append(f"{parts[0]} {new_last}")

    return variants


def _get_phonetic_variants(base_name: str, existing_names: list[str]) -> list[str]:
    """Generate phonetically diverse variants to avoid similar-sounding names."""
    variants: list[str] = []
    name = base_name.lower()

    similar_sounds = {
        "v": ["w", "f"],
        "w": ["v", "f"],
        "f": ["v", "w"],
        "b": ["p"],
        "p": ["b"],
        "d": ["t"],
        "t": ["d"],
        "g": ["k"],
        "k": ["g"],
        "s": ["z", "c"],
        "z": ["s"],
        "x": ["ks", "z"],
        "c": ["s", "k"],
    }

    for char, replacements in similar_sounds.items():
        if char in name:
            for replacement in replacements:
                if replacement not in name:
                    new_name = name.replace(char, replacement)
                    if _calculate_name_distance(base_name, new_name) > 0.3:
                        variants.append(new_name.title())

    return variants


def _get_genre_variants(base_name: str, existing_names: list[str]) -> list[str]:
    """Generate genre-appropriate names that fit the story context."""
    return [
        f"{_generate_random_first_name()} {_generate_random_surname()}"
        for _ in range(3)
    ]


def _calculate_name_distance(name1: str, name2: str) -> float:
    """Calculate a simple distance metric between names."""
    from difflib import SequenceMatcher

    return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()


def _appears_to_be_cultural(name: str, culture: str) -> bool:
    """Simple heuristic to determine if name appears to be from a specific culture."""
    cultural_indicators = {
        "nordic": ["bj", "erik", "ing", "gunn", "sig", "astr", "fre"],
        "japanese": ["ak", "hiro", "ken", "mar", "nobu", "tan", "yuk"],
        "arabic": ["far", "hak", "lei", "mal", "nad", "om", "zar"],
    }

    name_lower = name.lower()
    if culture in cultural_indicators:
        return any(
            indicator in name_lower for indicator in cultural_indicators[culture]
        )
    return False


def _generate_generic_surname() -> str:
    """Generate a generic surname."""
    import random

    generic_surnames = [
        "Smith",
        "Johnson",
        "Williams",
        "Brown",
        "Jones",
        "Miller",
        "Davis",
        "Garcia",
    ]
    return random.choice(generic_surnames)


def _generate_alternative_first_names(original_first: str) -> list[str]:
    """Generate alternative first names that are structurally different."""
    alternatives = {
        "short": ["Alex", "Sam", "Max", "Jay", "Roe", "Kai", "Zoe", "Ace"],
        "long": [
            "Alexander",
            "Samantha",
            "Maximilian",
            "Sebastian",
            "Valentina",
            "Constantine",
        ],
        "unique": ["Aether", "Nyx", "Zephyr", "Aurora", "Caspian", "Seraphina"],
    }

    if len(original_first) <= 4:
        return alternatives["long"][:3]
    else:
        return alternatives["short"][:3]


def _generate_alternative_last_names(original_last: str) -> list[str]:
    """Generate alternative last names that are structurally different."""
    import random

    return [
        f"{random.choice(['Dark', 'Light', 'Storm', 'Raven', 'Iron', 'Stone'])}{random.choice(['hold', 'brook', 'ward', 'crest', 'blade', 'heart'])}"
        for _ in range(5)
    ]


def _generate_random_first_name() -> str:
    """Generate a random first name."""
    import random

    first_names = [
        "Aria",
        "Caelum",
        "Dante",
        "Elara",
        "Fenris",
        "Gideon",
        "Hazel",
        "Iris",
        "Jasper",
        "Kira",
        "Luna",
        "Milo",
        "Nova",
        "Orion",
        "Piper",
        "Quinn",
        "Riley",
        "Sage",
        "Tobias",
        "Una",
        "Vera",
        "Wyatt",
        "Xander",
        "Yara",
        "Zion",
    ]
    return random.choice(first_names)


def _generate_random_surname() -> str:
    """Generate a random surname."""
    import random

    surnames = [
        "Blackwood",
        "Crestfall",
        "Duskbane",
        "Emberheart",
        "Frostwind",
        "Goldleaf",
        "Holloway",
        "Ironforge",
        "Nightshade",
        "Oakenheart",
        "Palewater",
        "Quickshot",
        "Ravencrest",
        "Stormwright",
        "Thornfield",
        "Underhill",
        "Valewood",
        "Whitestone",
        "Youngblade",
        "Zephyrwind",
    ]
    return random.choice(surnames)


def _is_too_similar_to_existing(candidate: str, existing_names: list[str]) -> bool:
    """Check if candidate name is too similar to existing names."""
    for existing in existing_names:
        if _calculate_name_distance(candidate, existing) > 0.6:
            return True
    return False


def _is_too_similar_to_used(candidate: str, used_names: set[str]) -> bool:
    """Check if candidate name is too similar to recently used names."""
    for used in used_names:
        if _calculate_name_distance(candidate, used) > 0.6:
            return True
    return False


async def bootstrap_characters(
    character_profiles: dict[str, CharacterProfile],
    plot_outline: dict[str, Any],
    state_tracker: Any = None,
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

    # StateTracker support removed - using Neo4j MERGE for deduplication instead
    if state_tracker is None:
        state_tracker = type('MockStateTracker', (), {
            'reserve': lambda self, *args: None,
            'check': lambda self, *args: False
        })()

    # Pre-reserve all placeholder names to prevent conflicts during parallel generation
    placeholder_reservations = {}
    for name, _profile in character_profiles.items():
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
                        # Try deterministic, diverse variants before giving up
                        for variant in _generate_diverse_name_variants(
                            value, [p.name for p in character_profiles.values()]
                        ):
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

                # Additional diversity check when name changed successfully
                if hasattr(state_tracker, "check_name_diversity"):
                    try:
                        diversity_ok = await state_tracker.check_name_diversity(value)
                    except Exception:
                        diversity_ok = True
                    if not diversity_ok:
                        logger.warning(
                            "Generated name lacks diversity, generating alternative",
                            old_name=name,
                            new_name=value,
                        )
                        profile_dict = character_profiles[name].to_dict()
                        profile_dict["name"] = character_profiles[name].name
                        conflict_context = {
                            "profile": profile_dict,
                            "plot_outline": plot_outline,
                            "conflicting_name": value,
                            "existing_names": [
                                p.name for p in character_profiles.values()
                            ],
                            "diversity_issue": True,
                        }
                        if serial_world is not None:
                            conflict_context["world"] = serial_world

                        diverse_name_result = await _try_generate_unique_name(
                            conflict_context,
                            state_tracker,
                            [p.name for p in character_profiles.values()],
                        )
                        if diverse_name_result:
                            value = diverse_name_result
                            logger.info(
                                "Generated diverse name to resolve similarity issue",
                                old_name=name,
                                diverse_name=value,
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
                        for variant in _generate_diverse_name_variants(
                            name, [p.name for p in character_profiles.values()]
                        ):
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
                        for variant in _generate_diverse_name_variants(
                            name, [p.name for p in character_profiles.values()]
                        ):
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
