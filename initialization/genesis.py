from typing import Any, Dict, Tuple
import structlog

import config
from kg_maintainer.models import CharacterProfile, WorldItem
from kg_maintainer_agent import KGMaintainerAgent
from data_access import plot_queries

from .data_loader import load_user_supplied_model, convert_model_to_objects
from .bootstrappers.plot_bootstrapper import create_default_plot, bootstrap_plot_outline
from .bootstrappers.character_bootstrapper import (
    create_default_characters,
    bootstrap_characters,
)
from .bootstrappers.world_bootstrapper import create_default_world, bootstrap_world

logger = structlog.get_logger(__name__)


async def run_genesis_phase() -> Tuple[
    Dict[str, Any],
    Dict[str, CharacterProfile],
    Dict[str, Dict[str, WorldItem]],
    Dict[str, int],
]:
    """Execute the initial bootstrapping phase."""
    usage_totals: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def _add_usage(total: Dict[str, int], usage: Dict[str, int]) -> None:
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
        plot_outline = create_default_plot(config.DEFAULT_PROTAGONIST_NAME)
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

    await plot_queries.save_plot_outline_to_db(plot_outline)
    logger.info("Persisted bootstrapped plot outline to Neo4j.")

    kg_agent = KGMaintainerAgent()
    world_items_for_kg: Dict[str, Dict[str, WorldItem]] = {
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

    return plot_outline, character_profiles, world_items_for_kg, usage_totals
