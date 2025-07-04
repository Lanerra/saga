import asyncio
from typing import Any, Coroutine, Dict, Optional, Tuple

import structlog

import config
import utils
from kg_maintainer.models import CharacterProfile

from .common import bootstrap_field

logger = structlog.get_logger(__name__)


def create_default_characters(protagonist_name: str) -> Dict[str, CharacterProfile]:
    """Create a default protagonist profile."""
    profile = CharacterProfile(name=protagonist_name)
    profile.description = config.FILL_IN
    profile.updates["role"] = "protagonist"
    return {protagonist_name: profile}


async def bootstrap_characters(
    character_profiles: Dict[str, CharacterProfile],
    plot_outline: Dict[str, Any],
) -> Tuple[Dict[str, CharacterProfile], Optional[Dict[str, int]]]:
    """Fill missing character profile data via LLM."""
    tasks: Dict[Tuple[str, str], Coroutine] = {}
    usage_data: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    for name, profile in character_profiles.items():
        context = {"profile": profile.to_dict(), "plot_outline": plot_outline}

        if not profile.description or utils._is_fill_in(profile.description):
            tasks[(name, "description")] = bootstrap_field(
                "description", context, "bootstrapper/fill_character_field.j2"
            )

        if not profile.status or utils._is_fill_in(profile.status):
            tasks[(name, "status")] = bootstrap_field(
                "status", context, "bootstrapper/fill_character_field.j2"
            )

        trait_fill_count = sum(1 for t in profile.traits if utils._is_fill_in(t))
        if trait_fill_count or not profile.traits:
            tasks[(name, "traits")] = bootstrap_field(
                "traits",
                context,
                "bootstrapper/fill_character_field.j2",
                is_list=True,
                list_count=max(trait_fill_count, 3),
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

    for i, (value, usage) in enumerate(results):
        name, field = task_keys[i]
        if usage:
            for k, v in usage.items():
                usage_data[k] = usage_data.get(k, 0) + v
        if value:
            if field == "description":
                character_profiles[name].description = value
            elif field == "traits":
                character_profiles[name].traits = value  # type: ignore
            elif field == "status":
                character_profiles[name].status = value
            else:  # motivation
                character_profiles[name].updates[field] = value
            character_profiles[name].updates["source"] = "bootstrapped"

    return character_profiles, usage_data if usage_data["total_tokens"] > 0 else None
