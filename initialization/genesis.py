# initialization/genesis.py
from typing import Any

import structlog

import config
from agents.knowledge_agent import KnowledgeAgent
from data_access import plot_queries
from models import CharacterProfile, WorldItem
from processing.state_tracker import StateTracker

from .bootstrappers.character_bootstrapper import (
    bootstrap_characters,
    create_default_characters,
)
from .bootstrappers.plot_bootstrapper import bootstrap_plot_outline, create_default_plot
from .bootstrappers.world_bootstrapper import bootstrap_world, create_default_world
from .data_loader import convert_model_to_objects, load_user_supplied_model

logger = structlog.get_logger(__name__)


async def run_genesis_phase() -> (
    tuple[
        dict[str, Any],
        dict[str, CharacterProfile],
        dict[str, dict[str, WorldItem]],
    ]
):
    """Execute the initial bootstrapping phase."""

    model = load_user_supplied_model()
    if model:
        plot_outline, character_profiles, world_building = convert_model_to_objects(
            model
        )
        plot_outline["source"] = "user_supplied_yaml"
        world_building["source"] = "user_supplied_yaml"
        logger.info("Loaded user story elements from YAML file.")
    else:
        logger.info("No valid user YAML found. Using default placeholders.")
        plot_outline = create_default_plot(config.DEFAULT_PROTAGONIST_NAME)
        character_profiles = create_default_characters(plot_outline["protagonist_name"])
        world_building = create_default_world()

    # Create shared StateTracker instance for the bootstrap process
    state_tracker = StateTracker()
    logger.info("Created StateTracker instance for bootstrap coordination")

    plot_outline, _ = await bootstrap_plot_outline(plot_outline)
    character_profiles, _ = await bootstrap_characters(
        character_profiles, plot_outline, state_tracker
    )
    # Log the final character names for debugging
    final_char_names = [profile.name for profile in character_profiles.values()]
    logger.info(f"Final character names after bootstrapping: {final_char_names}")
    world_building, _ = await bootstrap_world(
        world_building, plot_outline, state_tracker
    )

    await plot_queries.save_plot_outline_to_db(plot_outline)
    logger.info("Persisted bootstrapped plot outline to Neo4j.")

    kg_agent = KnowledgeAgent()
    world_items_for_kg: dict[str, dict[str, WorldItem]] = {
        k: v
        for k, v in world_building.items()
        if k not in ["is_default", "source"] and isinstance(v, dict)
    }
    await kg_agent.persist_profiles(
        character_profiles, config.KG_PREPOPULATION_CHAPTER_NUM, full_sync=True
    )
    await kg_agent.persist_world(
        world_items_for_kg, config.KG_PREPOPULATION_CHAPTER_NUM, full_sync=True
    )
    logger.info("Knowledge graph pre-population complete (full sync).")

    # Create basic relationship network using existing validation infrastructure
    if config.BOOTSTRAP_CREATE_RELATIONSHIPS:
        logger.info("Creating bootstrap relationship network...")
        await _create_bootstrap_relationships(
            character_profiles, world_items_for_kg, plot_outline
        )
        logger.info("Bootstrap relationship network complete.")

    # Validate bootstrap results before proceeding
    from .bootstrap_validator import validate_bootstrap_results

    validation_result = await validate_bootstrap_results(
        plot_outline, character_profiles, world_items_for_kg, state_tracker
    )

    if not validation_result.is_valid:
        error_summary = "; ".join(validation_result.errors[:3])  # Show first 3 errors
        if len(validation_result.errors) > 3:
            error_summary += f" (and {len(validation_result.errors) - 3} more)"
        raise RuntimeError(f"Bootstrap validation failed: {error_summary}")

    if validation_result.warnings:
        warning_count = len(validation_result.warnings)
        logger.warning(f"Bootstrap completed with {warning_count} warnings")
    else:
        logger.info("Bootstrap validation passed with no warnings")

    # Log StateTracker statistics for debugging
    all_tracked_entities = await state_tracker.get_all()
    if all_tracked_entities:
        char_count = len(await state_tracker.get_entities_by_type("character"))
        world_count = len(await state_tracker.get_entities_by_type("world_item"))
        logger.info(
            f"StateTracker final statistics: {len(all_tracked_entities)} total entities "
            f"({char_count} characters, {world_count} world items) tracked during bootstrap"
        )
    else:
        logger.warning(
            "StateTracker is empty - no entities were tracked during bootstrap"
        )

    return plot_outline, character_profiles, world_items_for_kg


async def _create_bootstrap_relationships(
    character_profiles: dict[str, CharacterProfile],
    world_items: dict[str, dict[str, WorldItem]],
    plot_outline: dict[str, Any],
) -> None:
    """Create basic relationship network using existing validation infrastructure."""
    from core.relationship_validator import RelationshipConstraintValidator
    from data_access import kg_queries

    validator = RelationshipConstraintValidator()
    created_relationships = []

    # Get character names and roles
    # Use actual profile names rather than dictionary keys to ensure we have the correct names
    char_names = [profile.name for profile in character_profiles.values()]
    protagonist = plot_outline.get("protagonist_name")
    antagonist = None
    supporting_chars = []

    for name, profile in character_profiles.items():
        role = profile.updates.get("role", "supporting")
        char_actual_name = profile.name
        if role == "antagonist":
            antagonist = char_actual_name
        elif role == "supporting":
            supporting_chars.append(char_actual_name)

    # 1. Core conflict relationship: protagonist vs antagonist
    if protagonist and antagonist and protagonist != antagonist:
        # Use direct relationship type instead of enhanced suggestions
        rel_type = "ENEMY_OF"  # Direct conflict relationship
        validation_result = validator.validate_relationship(
            "Character", rel_type, "Character"
        )
        if validation_result.is_valid:
            created_relationships.append(
                (protagonist, validation_result.validated_relationship, antagonist)
            )
            logger.info(
                f"Bootstrap relationship: {protagonist} {validation_result.validated_relationship} {antagonist}"
            )

    # 2. Alliance relationships: protagonist with supporting characters
    if protagonist and supporting_chars:
        for support_char in supporting_chars[:2]:  # Limit to first 2 supporting chars
            # Use direct relationship type instead of enhanced suggestions
            rel_type = "ALLY_OF"  # Direct alliance relationship
            validation_result = validator.validate_relationship(
                "Character", rel_type, "Character"
            )
            if validation_result.is_valid:
                created_relationships.append(
                    (
                        protagonist,
                        validation_result.validated_relationship,
                        support_char,
                    )
                )
                logger.info(
                    f"Bootstrap relationship: {protagonist} {validation_result.validated_relationship} {support_char}"
                )

    # 3. Character-to-world relationships: characters reside in/belong to world elements
    for char_name in char_names[:4]:  # Limit to prevent too many relationships
        # Each character belongs to a faction (if available)
        factions = world_items.get("factions", {})
        if factions:
            faction_names = list(factions.keys())
            if faction_names:
                # Use deterministic character-based distribution instead of hash()
                faction_index = sum(ord(c) for c in char_name) % len(faction_names)
                faction = faction_names[faction_index]
                validation_result = validator.validate_relationship(
                    "Character", "MEMBER_OF", "Faction"
                )
                if validation_result.is_valid:
                    created_relationships.append(
                        (char_name, validation_result.validated_relationship, faction)
                    )
                    logger.info(
                        f"Bootstrap relationship: {char_name} {validation_result.validated_relationship} {faction}"
                    )

        # Each character resides in a location (if available)
        locations = world_items.get("locations", {})
        if locations:
            location_names = list(locations.keys())
            if location_names:
                # Use deterministic character-based distribution instead of hash()
                location_key = char_name + "loc"
                location_index = sum(ord(c) for c in location_key) % len(location_names)
                location = location_names[location_index]
                validation_result = validator.validate_relationship(
                    "Character", "LOCATED_IN", "Location"
                )
                if validation_result.is_valid:
                    created_relationships.append(
                        (char_name, validation_result.validated_relationship, location)
                    )
                    logger.info(
                        f"Bootstrap relationship: {char_name} {validation_result.validated_relationship} {location}"
                    )

    # 4. World element relationships: factions control locations, etc.
    factions = world_items.get("factions", {})
    locations = world_items.get("locations", {})
    if factions and locations:
        faction_list = list(factions.keys())
        location_list = list(locations.keys())
        # Create some faction-location control relationships
        for i, faction in enumerate(faction_list[:2]):  # Limit to first 2 factions
            if i < len(location_list):
                location = location_list[i]
                validation_result = validator.validate_relationship(
                    "Faction", "CONTROLS", "Location"
                )
                if validation_result.is_valid:
                    created_relationships.append(
                        (faction, validation_result.validated_relationship, location)
                    )
                    logger.info(
                        f"Bootstrap relationship: {faction} {validation_result.validated_relationship} {location}"
                    )

    # Store relationships in the knowledge graph
    relationships_created = 0
    for subj, rel, obj in created_relationships:
        try:
            # Use existing KG infrastructure to create the relationship
            await kg_queries.create_relationship_with_properties(
                subject_name=subj,
                relationship_type=rel,
                object_name=obj,
                properties={
                    "source": "bootstrap",
                    "confidence": 0.8,
                    "chapter_added": 0,
                },
            )
            relationships_created += 1
        except Exception as e:
            from .error_handling import ErrorSeverity, handle_bootstrap_error

            handle_bootstrap_error(
                e,
                f"Bootstrap relationship creation: {subj} {rel} {obj}",
                ErrorSeverity.WARNING,
                {"subject": subj, "relationship": rel, "object": obj},
            )

    logger.info(
        f"Successfully created {relationships_created} bootstrap relationships out of {len(created_relationships)} attempted."
    )


# Dynamic schema refresh function removed - not needed for single-user deployment
