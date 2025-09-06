# initialization/genesis.py
from typing import Any

import structlog

import config
from agents.knowledge_agent import KnowledgeAgent
from data_access import plot_queries
from models import CharacterProfile, WorldItem

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

    plot_outline, _ = await bootstrap_plot_outline(plot_outline)
    character_profiles, _ = await bootstrap_characters(character_profiles, plot_outline)
    world_building, _ = await bootstrap_world(world_building, plot_outline)

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
        plot_outline, character_profiles, world_items_for_kg
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

    # Refresh dynamic schema patterns after bootstrap completion
    await _refresh_dynamic_schema_after_bootstrap()

    return plot_outline, character_profiles, world_items_for_kg


async def _create_bootstrap_relationships(
    character_profiles: dict[str, CharacterProfile],
    world_items: dict[str, dict[str, WorldItem]],
    plot_outline: dict[str, Any],
) -> None:
    """Create basic relationship network using existing validation infrastructure."""
    from core.enhanced_constraints import get_enhanced_relationship_suggestions
    from core.relationship_validator import RelationshipConstraintValidator
    from data_access import kg_queries

    validator = RelationshipConstraintValidator()
    created_relationships = []

    # Get character names and roles
    char_names = list(character_profiles.keys())
    protagonist = plot_outline.get("protagonist_name")
    antagonist = None
    supporting_chars = []

    for name, profile in character_profiles.items():
        role = profile.updates.get("role", "supporting")
        if role == "antagonist":
            antagonist = name
        elif role == "supporting":
            supporting_chars.append(name)

    # 1. Core conflict relationship: protagonist vs antagonist
    if protagonist and antagonist and protagonist != antagonist:
        suggestions = get_enhanced_relationship_suggestions("Character", "Character")
        conflict_rels = [
            rel
            for rel, desc, conf in suggestions
            if any(term in rel.lower() for term in ["rival", "enemy", "opposes"])
        ]
        if conflict_rels:
            rel_type = conflict_rels[0]
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
            suggestions = get_enhanced_relationship_suggestions(
                "Character", "Character"
            )
            ally_rels = [
                rel
                for rel, desc, conf in suggestions
                if any(term in rel.lower() for term in ["ally", "friend", "trusts"])
            ]
            if ally_rels:
                rel_type = ally_rels[0]
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
            from .error_handling import handle_bootstrap_error, ErrorSeverity
            handle_bootstrap_error(
                e,
                f"Bootstrap relationship creation: {subj} {rel} {obj}",
                ErrorSeverity.WARNING,
                {"subject": subj, "relationship": rel, "object": obj}
            )

    logger.info(
        f"Successfully created {relationships_created} bootstrap relationships out of {len(created_relationships)} attempted."
    )


async def _refresh_dynamic_schema_after_bootstrap() -> None:
    """Refresh dynamic schema patterns after bootstrap phase completion."""
    try:
        # Only refresh if dynamic schema is enabled
        if not getattr(config.settings, "ENABLE_DYNAMIC_SCHEMA", True):
            return

        logger.info("Refreshing dynamic schema patterns after bootstrap completion...")

        # Import here to avoid circular dependencies
        from core.dynamic_schema_manager import dynamic_schema_manager

        # Initialize/refresh the dynamic schema system with all the new bootstrap data
        await dynamic_schema_manager.initialize(force_refresh=True)

        # Get status for logging
        status = await dynamic_schema_manager.get_system_status()
        type_patterns = status.get("type_inference", {}).get("total_patterns", 0)
        constraints = status.get("constraints", {}).get("total_constraints", 0)
        schema_info = status.get("schema", {})
        total_nodes = schema_info.get("total_nodes", 0)

        logger.info(
            f"Dynamic schema patterns learned from bootstrap data: "
            f"{type_patterns} type patterns from {total_nodes} nodes, "
            f"{constraints} relationship constraints"
        )

        # This ensures novel-specific patterns are available from chapter 1 onwards
        logger.info(
            "Dynamic schema system ready for chapter generation with bootstrap patterns"
        )

    except Exception as e:
        # Don't fail bootstrap if schema refresh fails
        from .error_handling import handle_bootstrap_error, ErrorSeverity
        handle_bootstrap_error(
            e,
            "Dynamic schema refresh after bootstrap",
            ErrorSeverity.WARNING,
            {"phase": "post_bootstrap_schema_refresh"}
        )
