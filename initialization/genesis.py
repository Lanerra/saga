import json

import structlog
from agents.kg_maintainer_agent import KGMaintainerAgent
from config import settings
from data_access import chapter_queries, plot_queries
from kg_maintainer.models import WorldItem

from .bootstrappers.character_bootstrapper import (
    bootstrap_characters,
    create_default_characters,
)
from .bootstrappers.plot_bootstrapper import bootstrap_plot_outline, create_default_plot
from .bootstrappers.world_bootstrapper import bootstrap_world, create_default_world
from .data_loader import convert_model_to_objects, load_user_supplied_model
from .models import CharacterProfile, PlotOutline, WorldBuilding

logger = structlog.get_logger(__name__)


async def run_genesis_phase() -> tuple[
    PlotOutline,
    dict[str, CharacterProfile],
    WorldBuilding,
    dict[str, int],
]:
    """Execute the initial bootstrapping phase."""
    usage_totals: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def _add_usage(total: dict[str, int], usage: dict[str, int]) -> None:
        for key, val in usage.items():
            total[key] = total.get(key, 0) + val

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
        plot_outline = create_default_plot(settings.DEFAULT_PROTAGONIST_NAME)
        character_profiles = create_default_characters(plot_outline["protagonist_name"])
        world_building = create_default_world()

    plot_outline, plot_usage = await bootstrap_plot_outline(plot_outline)
    if plot_usage:
        _add_usage(usage_totals, plot_usage)
    character_profiles, char_usage = await bootstrap_characters(
        character_profiles, plot_outline
    )
    if char_usage:
        _add_usage(usage_totals, char_usage)
    world_building, world_usage = await bootstrap_world(world_building, plot_outline)
    if world_usage:
        _add_usage(usage_totals, world_usage)

    await plot_queries.save_plot_outline_to_db(
        plot_outline.model_dump(exclude_none=True)
    )
    logger.info("Persisted bootstrapped plot outline to Neo4j.")

    kg_agent = KGMaintainerAgent()
    world_items_for_kg: dict[str, dict[str, WorldItem]] = dict(world_building.items())
    await kg_agent.persist_profiles(
        character_profiles, settings.KG_PREPOPULATION_CHAPTER_NUM, full_sync=True
    )
    await kg_agent.persist_world(
        world_items_for_kg, settings.KG_PREPOPULATION_CHAPTER_NUM, full_sync=True
    )
    logger.info("Knowledge graph pre-population complete (full sync).")

    # New enrichment pass to clarify ambiguous concepts before drafting begins
    await kg_agent.heal_and_enrich_kg()
    logger.info("Initial KG enrichment pass executed.")

    bootstrap_state_text = json.dumps(
        {
            "plot_outline": plot_outline.model_dump(exclude_none=True),
            "characters": {
                name: profile.model_dump(exclude_none=True)
                for name, profile in character_profiles.items()
            },
            "world": {
                cat: {
                    item_name: item.model_dump(exclude_none=True)
                    for item_name, item in items.items()
                }
                for cat, items in world_items_for_kg.items()
            },
        },
        ensure_ascii=False,
    )

    end_state = await kg_agent.generate_chapter_end_state(
        bootstrap_state_text,
        0,
    )
    await chapter_queries.save_chapter_data_to_db(
        0,
        bootstrap_state_text,
        bootstrap_state_text,
        None,
        None,
        False,
        end_state.model_dump(),
    )
    logger.info("Saved chapter 0 end state to Neo4j.")

    return plot_outline, character_profiles, world_items_for_kg, usage_totals
