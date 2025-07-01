# chapter_generation/context_kg_utils.py
"""Utility functions for context-related KG operations."""

from __future__ import annotations

import json
from typing import Any

import structlog
from config import settings
from core.llm_interface import llm_service
from data_access import character_queries, world_queries

logger = structlog.get_logger(__name__)


async def get_canonical_truths_from_kg() -> list[str]:
    """Return canonical truths stored in the knowledge graph."""

    lines: list[str] = []
    query = (
        "MATCH (c:Character:Entity)-[:HAS_TRAIT]->"
        "(t:Trait:Entity {is_canonical_truth: true}) "
        "RETURN c.name AS name, t.name AS trait"
    )
    try:
        records = await character_queries.neo4j_manager.execute_read_query(query)
    except Exception as exc:  # pragma: no cover - DB failures logged
        logger.error("Failed to load canonical truths: %s", exc, exc_info=True)
        records = []

    for rec in records:
        name = rec.get("name")
        trait = rec.get("trait")
        if name and trait:
            line = f"- {name} is {trait}"
            if line not in lines:
                lines.append(line)

    return lines


async def get_reliable_kg_facts_for_drafting_prompt(
    plot_outline: dict[str, Any],
    chapter_number: int,
    chapter_plan: list[dict[str, Any]] | None = None,
) -> str:
    """Return KG facts relevant to the chapter focus.

    The function first attempts to gather facts from the knowledge graph. If no
    facts are available, it falls back to a short LLM generated summary based on
    the provided plot outline.
    """

    try:
        characters = await character_queries.get_character_profiles_from_db()
        world = await world_queries.get_world_building_from_db()
    except Exception as exc:  # pragma: no cover - DB failures logged
        logger.error("Failed to retrieve KG facts: %s", exc, exc_info=True)
        characters = {}
        world = {}

    lines: list[str] = []
    for name, profile in characters.items():
        traits = profile.get("traits")
        if traits:
            lines.append(f"- {name}: {', '.join(traits)}")

    factions = world.get("factions")
    if isinstance(factions, list) and factions:
        lines.append("Factions: " + ", ".join(map(str, factions)))

    if lines:
        return "\n".join(lines)

    prompt = (
        "Provide three short bullet points summarizing important character or "
        f"world facts for chapter {chapter_number} using this plot outline:\n"
        f"{json.dumps(plot_outline)}"
    )
    text, _ = await llm_service.async_call_llm(
        model_name=settings.SMALL_MODEL,
        prompt=prompt,
        temperature=settings.TEMPERATURE_SUMMARY,
        max_tokens=settings.MAX_SUMMARY_TOKENS,
        allow_fallback=True,
    )
    return text.strip()


async def get_kg_reasoning_guidance_for_prompt(
    plot_outline: dict[str, Any],
    chapter_number: int,
    chapter_plan: list[dict[str, Any]] | None = None,
    max_guidelines: int = 5,
) -> str:
    """Return KG reasoning hints for the chapter.

    If the KG does not provide enough information, guidance is generated via the
    LLM based on the plot outline.
    """

    prompt = (
        f"Provide {max_guidelines} short guidelines to ensure story continuity "
        f"for chapter {chapter_number}. Plot outline:\n{json.dumps(plot_outline)}"
    )
    text, _ = await llm_service.async_call_llm(
        model_name=settings.SMALL_MODEL,
        prompt=prompt,
        temperature=settings.TEMPERATURE_SUMMARY,
        max_tokens=settings.MAX_SUMMARY_TOKENS,
        allow_fallback=True,
    )
    return text.strip()
