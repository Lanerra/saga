# initialization/bootstrap_pipeline.py
"""Standalone and integrated bootstrap pipeline orchestrator.

Runs the multi-phase bootstrap (world -> characters -> plot) with per-phase
validation and optional KG persistence/healing, reusing existing bootstrappers
and validators.
"""

from __future__ import annotations

from typing import Any, Literal

import structlog

import config
import utils
from agents.knowledge_agent import KnowledgeAgent
from data_access import plot_queries
from initialization.bootstrap_validator import (
    BootstrapValidationResult,
    create_bootstrap_validation_report,
    quick_validate_characters,
    quick_validate_world,
    validate_bootstrap_result,
    validate_bootstrap_results,
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
from initialization.data_loader import (
    convert_model_to_objects,
    load_user_supplied_model,
)
from models import CharacterProfile, WorldItem
from processing.state_tracker import StateTracker

logger = structlog.get_logger(__name__)


BootstrapPhase = Literal["world", "characters", "plot", "all"]
BootstrapLevel = Literal["basic", "enhanced", "max"]


def _has_required_user_story_data(
    plot_outline: dict[str, Any],
    character_profiles: dict[str, CharacterProfile],
    world_building: dict[str, dict[str, WorldItem]],
) -> bool:
    """Basic gate to decide if user-supplied story can bypass bootstrap."""

    title = plot_outline.get("title")
    if not title or utils._is_fill_in(title):
        return False

    plot_points = plot_outline.get("plot_points", [])
    has_concrete_plot_point = any(
        isinstance(point, str) and not utils._is_fill_in(point) and point.strip()
        for point in plot_points
    )
    if not has_concrete_plot_point:
        return False

    protagonist_name = plot_outline.get("protagonist_name")
    has_protagonist = False
    for profile in character_profiles.values():
        if not isinstance(profile, CharacterProfile):
            continue
        if profile.updates.get("role") == "protagonist" or (
            protagonist_name and profile.name == protagonist_name
        ):
            has_protagonist = True
            break
    if not has_protagonist:
        return False

    has_world_category = any(
        isinstance(items, dict) and items
        for category, items in world_building.items()
        if category not in {"source", "is_default"}
    )
    return has_world_category


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
) -> tuple[dict[str, dict[str, WorldItem]], BootstrapValidationResult]:
    """Generate world, validate uniqueness, optionally persist and heal."""
    world_building = world_building or create_default_world()

    world_building, _ = await bootstrap_world(
        world_building, plot_outline, state_tracker
    )

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
        await kg.persist_world(
            world_building, config.KG_PREPOPULATION_CHAPTER_NUM, full_sync=True
        )
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
) -> tuple[dict[str, CharacterProfile], BootstrapValidationResult]:
    """Generate characters, validate duplicates, optionally persist and heal."""
    protagonist_name = plot_outline.get(
        "protagonist_name", config.DEFAULT_PROTAGONIST_NAME
    )
    character_profiles = character_profiles or create_default_characters(
        protagonist_name
    )

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
) -> tuple[dict[str, Any], BootstrapValidationResult]:
    """Generate plot outline leveraging world + characters; validate and persist."""
    plot_outline = plot_outline or create_default_plot(
        character_profiles.get(
            "protagonist", CharacterProfile(name=config.DEFAULT_PROTAGONIST_NAME)
        ).name
        if character_profiles
        else config.DEFAULT_PROTAGONIST_NAME
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

        # Precompute and persist a Chapter 1 KG hint bundle for fast reuse
        try:
            protagonist_name = plot_outline.get("protagonist_name") or (
                next(
                    (
                        cp.name
                        for cp in character_profiles.values()
                        if cp.updates.get("role") == "protagonist"
                    ),
                    None,
                )
                if character_profiles
                else None
            )

            # Fetch a compact set of bootstrap world elements
            world_snippet = ""
            try:
                from data_access.world_queries import get_bootstrap_world_elements
                from prompts.prompt_data_getters import (
                    get_world_state_snippet_for_prompt,
                )

                bootstrap_world_items = await get_bootstrap_world_elements()
                if bootstrap_world_items:
                    world_snippet = (
                        await get_world_state_snippet_for_prompt(
                            bootstrap_world_items,
                            current_chapter_num_for_filtering=config.KG_PREPOPULATION_CHAPTER_NUM,
                        )
                        or ""
                    )
            except Exception:
                world_snippet = ""

            # Assemble bundle text (protagonist-centric + a few world facts)
            lines: list[str] = []
            title = plot_outline.get("title") or config.DEFAULT_PLOT_OUTLINE_TITLE
            lines.append(f"Title: {title}")
            if protagonist_name:
                lines.append(f"Protagonist: {protagonist_name}")
            if character_profiles and isinstance(character_profiles, dict):
                # Try to find antagonist and a couple allies by simple heuristics
                antagonist = next(
                    (
                        cp.name
                        for cp in character_profiles.values()
                        if cp.updates.get("role") == "antagonist"
                    ),
                    None,
                )
                if antagonist:
                    lines.append(f"Antagonist: {antagonist}")
            if world_snippet and world_snippet.strip():
                lines.append("")
                lines.append("World At A Glance:")
                lines.append(world_snippet.strip())

            hint_bundle = "\n".join(lines).strip()
            if hint_bundle:
                await plot_queries.set_first_chapter_kg_hint(hint_bundle)
        except Exception as e:
            logger.warning("Failed to precompute Chapter 1 KG bundle", error=str(e))

    return plot_outline, validation


async def run_bootstrap_pipeline(
    phase: BootstrapPhase = "all",
    level: BootstrapLevel = "enhanced",
    *,
    dry_run: bool = False,
    kg_heal: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, CharacterProfile],
    dict[str, dict[str, WorldItem]],
    list[str],
]:
    """Top-level pipeline that executes selected phases sequentially.

    Returns: (plot_outline, character_profiles, world_building, warnings)
    """
    _apply_level_overrides(level)

    warnings: list[str] = []

    # Load optional user model
    model = load_user_supplied_model()
    if model:
        plot_outline, character_profiles, world_building = convert_model_to_objects(
            model
        )
        plot_outline.setdefault("source", "user_story_elements")
        world_building.setdefault("source", "user_story_elements")

        initial_validation = await validate_bootstrap_results(
            plot_outline,
            character_profiles,
            world_building,
        )
        warnings.extend([w for w in initial_validation.warnings if w not in warnings])

        if (
            phase == "all"
            and initial_validation.is_valid
            and _has_required_user_story_data(
                plot_outline, character_profiles, world_building
            )
        ):
            logger.info(
                "User story elements file supplies complete bootstrap data; skipping generation phases.",
                title=plot_outline.get("title"),
                protagonist=plot_outline.get("protagonist_name"),
                plot_point_count=len(plot_outline.get("plot_points", [])),
            )
            if not dry_run and getattr(config, "BOOTSTRAP_PUSH_TO_KG_EACH_PHASE", True):
                kg = KnowledgeAgent()
                await plot_queries.save_plot_outline_to_db(plot_outline)
                await kg.persist_world(
                    world_building,
                    config.KG_PREPOPULATION_CHAPTER_NUM,
                    full_sync=True,
                )
                await kg.persist_profiles(
                    character_profiles,
                    config.KG_PREPOPULATION_CHAPTER_NUM,
                    full_sync=True,
                )
                if kg_heal and getattr(config, "BOOTSTRAP_RUN_KG_HEAL", True):
                    await kg.heal_and_enrich_kg()
            return plot_outline, character_profiles, world_building, warnings

        logger.info(
            "User story elements detected but additional bootstrapping required.",
            validation_passed=initial_validation.is_valid,
            warning_count=len(initial_validation.warnings),
        )
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
            plot_outline,
            world_building,
            state_tracker,
            dry_run=dry_run,
            kg_heal=kg_heal,
            kg_agent=kg_agent,
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
            plot_outline,
            character_profiles,
            world_building,
            dry_run=dry_run,
            kg_heal=kg_heal,
        )
        warnings.extend([w for w in plot_val.warnings if w not in warnings])

    # Validate final bootstrap result using the new validation pipeline
    if phase == "all" and getattr(config, "BOOTSTRAP_USE_VALIDATION", True):
        try:
            # Create bootstrap result dictionary for validation
            bootstrap_result = {
                "plot_outline": plot_outline,
                "character_profiles": character_profiles,
                "world_building": world_building,
                "bootstrap_source": f"bootstrap_pipeline_{level}",
            }

            # Validate bootstrap result
            (
                corrected_result,
                was_corrected,
                validation_errors,
            ) = await validate_bootstrap_result(
                bootstrap_result,
                auto_correct=False,  # Don't auto-correct in pipeline
            )

            # Update results if corrections were applied
            if was_corrected:
                plot_outline = corrected_result.get("plot_outline", plot_outline)
                character_profiles = corrected_result.get(
                    "character_profiles", character_profiles
                )
                world_building = corrected_result.get("world_building", world_building)
                warnings.append(
                    f"Bootstrap validation applied corrections for {len(validation_errors)} issues"
                )

            # Add validation warnings
            if validation_errors:
                validation_warnings = [
                    f"Validation: {error.message}" for error in validation_errors
                ]
                warnings.extend(validation_warnings)

                # Create validation report for debugging
                validation_report = create_bootstrap_validation_report(
                    validation_errors
                )
                logger.info(
                    "Bootstrap validation completed",
                    validation_passed=len(validation_errors) == 0,
                    error_count=len(validation_errors),
                    report_summary=validation_report.get("summary"),
                )

        except Exception as e:
            logger.error("Bootstrap validation failed", error=str(e))
            warnings.append(f"Validation pipeline error: {str(e)}")

    return plot_outline, character_profiles, world_building, warnings
