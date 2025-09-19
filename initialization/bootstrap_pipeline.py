"""Standalone and integrated bootstrap pipeline orchestrator.

Runs the multi-phase bootstrap (world -> characters -> plot) with per-phase
validation and optional KG persistence/healing, reusing existing bootstrappers
and validators.
"""

from __future__ import annotations

from typing import Any, Literal, Tuple

import structlog

import config
from agents.knowledge_agent import KnowledgeAgent
from data_access import plot_queries
from initialization.bootstrap_validator import (
    BootstrapValidationResult,
    validate_bootstrap_results,
    quick_validate_world,
    quick_validate_characters,
)
from initialization.bootstrappers.character_bootstrapper import (
    bootstrap_characters,
    create_default_characters,
)
from initialization.bootstrappers.plot_bootstrapper import (
    bootstrap_plot_outline,
    create_default_plot,
)
from initialization.bootstrappers.world_bootstrapper import (
    bootstrap_world,
    create_default_world,
)
from initialization.data_loader import convert_model_to_objects, load_user_supplied_model
from models import CharacterProfile, WorldItem
from processing.state_tracker import StateTracker

logger = structlog.get_logger(__name__)


BootstrapPhase = Literal["world", "characters", "plot", "all"]
BootstrapLevel = Literal["basic", "enhanced", "max"]


def _apply_level_overrides(level: BootstrapLevel) -> None:
    """Map a level to config knobs. Keep minimal and reversible."""
    if level == "basic":
        config.set("TARGET_PLOT_POINTS_INITIAL_GENERATION", 10)
        config.set("BOOTSTRAP_MIN_CHARACTERS", 2)
        config.set("BOOTSTRAP_MIN_WORLD_ELEMENTS", 3)
    elif level == "enhanced":
        # Defaults already tuned for enhanced
        pass
    elif level == "max":
        config.set("TARGET_PLOT_POINTS_INITIAL_GENERATION", 30)
        config.set("BOOTSTRAP_MIN_CHARACTERS", 5)
        config.set("BOOTSTRAP_MIN_WORLD_ELEMENTS", 6)


async def run_world_phase(
    plot_outline: dict[str, Any],
    world_building: dict[str, dict[str, WorldItem]] | None,
    state_tracker: StateTracker,
    *,
    dry_run: bool = False,
    kg_heal: bool = False,
    kg_agent: KnowledgeAgent | None = None,
) -> Tuple[dict[str, dict[str, WorldItem]], BootstrapValidationResult]:
    """Generate world, validate uniqueness, optionally persist and heal."""
    world_building = world_building or create_default_world()

    world_building, _ = await bootstrap_world(world_building, plot_outline, state_tracker)

    # Phase-local validation: world uniqueness and basic checks only
    # Avoid full cross-component validation to prevent spurious errors
    # (e.g., missing characters during the world phase).
    validation = await quick_validate_world(
        plot_outline,
        world_building,
        state_tracker,
    )

    # Enforce duplicate-check requirements for locations specifically
    if validation.is_valid:
        locations = world_building.get("locations", {})
        if isinstance(locations, dict):
            seen: set[str] = set()
            dups: list[str] = []
            for name in locations.keys():
                if name in seen:
                    dups.append(name)
                seen.add(name)
            for name in dups:
                validation.add_error(f"Duplicate location name detected: {name}")

    if not validation.is_valid and getattr(config, "BOOTSTRAP_FAIL_FAST", True):
        return world_building, validation

    if not dry_run and getattr(config, "BOOTSTRAP_PUSH_TO_KG_EACH_PHASE", True):
        kg = kg_agent or KnowledgeAgent()
        await kg.persist_world(world_building, config.KG_PREPOPULATION_CHAPTER_NUM, full_sync=True)
        if kg_heal and getattr(config, "BOOTSTRAP_RUN_KG_HEAL", True):
            await kg.heal_and_enrich_kg()

    return world_building, validation


async def run_characters_phase(
    plot_outline: dict[str, Any],
    character_profiles: dict[str, CharacterProfile] | None,
    state_tracker: StateTracker,
    *,
    dry_run: bool = False,
    kg_heal: bool = False,
    world_building: dict[str, dict[str, WorldItem]] | None = None,
    kg_agent: KnowledgeAgent | None = None,
) -> Tuple[dict[str, CharacterProfile], BootstrapValidationResult]:
    """Generate characters, validate duplicates, optionally persist and heal."""
    protagonist_name = plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
    character_profiles = character_profiles or create_default_characters(protagonist_name)

    character_profiles, _ = await bootstrap_characters(
        character_profiles,
        plot_outline,
        state_tracker,
        world_building=world_building,
    )

    # Phase-local validation: characters only to avoid plot/title empties
    # causing fail-fast before the plot phase fills them in.
    validation = await quick_validate_characters(
        character_profiles,
        state_tracker,
    )

    # Enforce hard fail on duplicate character names
    if not validation.is_valid and getattr(config, "BOOTSTRAP_FAIL_FAST", True):
        return character_profiles, validation

    if not dry_run and getattr(config, "BOOTSTRAP_PUSH_TO_KG_EACH_PHASE", True):
        kg = kg_agent or KnowledgeAgent()
        await kg.persist_profiles(
            character_profiles, config.KG_PREPOPULATION_CHAPTER_NUM, full_sync=True
        )
        if kg_heal and getattr(config, "BOOTSTRAP_RUN_KG_HEAL", True):
            await kg.heal_and_enrich_kg()

    return character_profiles, validation


async def run_plot_phase(
    plot_outline: dict[str, Any] | None,
    character_profiles: dict[str, CharacterProfile],
    world_building: dict[str, dict[str, WorldItem]],
    *,
    dry_run: bool = False,
    kg_heal: bool = False,
) -> Tuple[dict[str, Any], BootstrapValidationResult]:
    """Generate plot outline leveraging world + characters; validate and persist."""
    plot_outline = plot_outline or create_default_plot(
        character_profiles.get("protagonist", CharacterProfile(name=config.DEFAULT_PROTAGONIST_NAME)).name
        if character_profiles else config.DEFAULT_PROTAGONIST_NAME
    )

    # Use enriched context for interplay-aware points
    plot_outline, _ = await bootstrap_plot_outline(
        plot_outline,
        character_profiles=character_profiles,
        world_building=world_building,
    )

    validation = await validate_bootstrap_results(
        plot_outline,
        character_profiles,
        world_building,
        None,
    )

    if not validation.is_valid and getattr(config, "BOOTSTRAP_FAIL_FAST", True):
        return plot_outline, validation

    if not dry_run and getattr(config, "BOOTSTRAP_PUSH_TO_KG_EACH_PHASE", True):
        await plot_queries.save_plot_outline_to_db(plot_outline)

    return plot_outline, validation


async def run_bootstrap_pipeline(
    phase: BootstrapPhase = "all",
    level: BootstrapLevel = "enhanced",
    *,
    dry_run: bool = False,
    kg_heal: bool = False,
) -> tuple[dict[str, Any], dict[str, CharacterProfile], dict[str, dict[str, WorldItem]], list[str]]:
    """Top-level pipeline that executes selected phases sequentially.

    Returns: (plot_outline, character_profiles, world_building, warnings)
    """
    _apply_level_overrides(level)

    warnings: list[str] = []

    # Load optional user model
    model = load_user_supplied_model()
    if model:
        plot_outline, character_profiles, world_building = convert_model_to_objects(model)
    else:
        plot_outline = create_default_plot(config.DEFAULT_PROTAGONIST_NAME)
        character_profiles = create_default_characters(plot_outline["protagonist_name"])
        world_building = create_default_world()

    # Shared state tracker
    state_tracker = StateTracker()

    # Phase execution in order: world -> characters -> plot (unless limited by phase)
    # Reuse a single KnowledgeAgent instance for the run to avoid repeated init
    kg_agent: KnowledgeAgent | None = KnowledgeAgent()

    if phase in ("world", "all"):
        world_building, world_val = await run_world_phase(
            plot_outline, world_building, state_tracker, dry_run=dry_run, kg_heal=kg_heal, kg_agent=kg_agent
        )
        # Deduplicate warnings across phases
        warnings.extend([w for w in world_val.warnings if w not in warnings])
        if not world_val.is_valid and getattr(config, "BOOTSTRAP_FAIL_FAST", True):
            return plot_outline, {}, world_building, warnings

    if phase in ("characters", "all"):
        character_profiles, char_val = await run_characters_phase(
            plot_outline,
            character_profiles,
            state_tracker,
            dry_run=dry_run,
            kg_heal=kg_heal,
            world_building=world_building,
            kg_agent=kg_agent,
        )
        warnings.extend([w for w in char_val.warnings if w not in warnings])
        if not char_val.is_valid and getattr(config, "BOOTSTRAP_FAIL_FAST", True):
            return plot_outline, character_profiles, {}, warnings

    if phase in ("plot", "all"):
        plot_outline, plot_val = await run_plot_phase(
            plot_outline, character_profiles, world_building, dry_run=dry_run, kg_heal=kg_heal
        )
        warnings.extend([w for w in plot_val.warnings if w not in warnings])

    return plot_outline, character_profiles, world_building, warnings
