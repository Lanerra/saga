# initialization/bootstrappers/plot_bootstrapper.py
from collections.abc import Coroutine
from typing import Any

import structlog

import config
import utils

from .common import bootstrap_field

logger = structlog.get_logger(__name__)


def create_default_plot(default_protagonist_name: str) -> dict[str, Any]:
    """Create a default plot outline with placeholders."""
    num_points = config.TARGET_PLOT_POINTS_INITIAL_GENERATION
    return {
        "title": config.DEFAULT_PLOT_OUTLINE_TITLE,
        "protagonist_name": default_protagonist_name,
        "genre": config.CONFIGURED_GENRE,
        "setting": config.CONFIGURED_SETTING_DESCRIPTION,
        "theme": config.CONFIGURED_THEME,
        "summary": config.FILL_IN,
        "logline": config.FILL_IN,
        "inciting_incident": config.FILL_IN,
        "central_conflict": config.FILL_IN,
        "stakes": config.FILL_IN,
        "plot_points": [f"{config.FILL_IN}" for _ in range(num_points)],
        "narrative_style": config.FILL_IN,
        "tone": config.FILL_IN,
        "pacing": config.FILL_IN,
        "is_default": True,
        "source": "default_fallback",
    }


async def bootstrap_plot_outline(
    plot_outline: dict[str, Any],
    character_profiles: dict[str, Any] | None = None,
    world_building: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, int] | None]:
    """Fill missing plot fields via LLM."""
    tasks: dict[str, Coroutine] = {}
    usage_data: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    fields_to_bootstrap = {
        "title": not plot_outline.get("title")
        or utils._is_fill_in(plot_outline.get("title")),
        "protagonist_name": not plot_outline.get("protagonist_name")
        or utils._is_fill_in(plot_outline.get("protagonist_name")),
        "genre": not plot_outline.get("genre")
        or utils._is_fill_in(plot_outline.get("genre")),
        "setting": not plot_outline.get("setting")
        or utils._is_fill_in(plot_outline.get("setting")),
        "theme": not plot_outline.get("theme")
        or utils._is_fill_in(plot_outline.get("theme")),
        "summary": not plot_outline.get("summary")
        or utils._is_fill_in(plot_outline.get("summary")),
        "logline": not plot_outline.get("logline")
        or utils._is_fill_in(plot_outline.get("logline")),
        "inciting_incident": not plot_outline.get("inciting_incident")
        or utils._is_fill_in(plot_outline.get("inciting_incident")),
        "central_conflict": not plot_outline.get("central_conflict")
        or utils._is_fill_in(plot_outline.get("central_conflict")),
        "stakes": not plot_outline.get("stakes")
        or utils._is_fill_in(plot_outline.get("stakes")),
        "narrative_style": not plot_outline.get("narrative_style")
        or utils._is_fill_in(plot_outline.get("narrative_style")),
        "tone": not plot_outline.get("tone")
        or utils._is_fill_in(plot_outline.get("tone")),
        "pacing": not plot_outline.get("pacing")
        or utils._is_fill_in(plot_outline.get("pacing")),
    }

    for field, needed in fields_to_bootstrap.items():
        if needed:
            tasks[field] = bootstrap_field(
                field, plot_outline, "bootstrapper/fill_plot_field.j2"
            )

    plot_points = plot_outline.get("plot_points", [])
    fill_in_count = sum(1 for p in plot_points if utils._is_fill_in(p))
    needed_plot_points = max(
        0,
        config.TARGET_PLOT_POINTS_INITIAL_GENERATION
        - (len(plot_points) - fill_in_count),
    )

    if needed_plot_points > 0:
        # Provide richer interplay-aware context without changing prompts
        pp_context: dict[str, Any] = {
            "plot_outline": plot_outline,
        }
        if isinstance(character_profiles, dict) and character_profiles:
            # Convert CharacterProfile objects to plain dicts for JSON context
            try:
                pp_context["characters"] = {
                    name: (
                        profile.to_dict()
                        if hasattr(profile, "to_dict")
                        else dict(profile)
                    )
                    for name, profile in character_profiles.items()
                }
            except Exception:
                pp_context["characters"] = {
                    name: str(profile) for name, profile in character_profiles.items()
                }
        if isinstance(world_building, dict) and world_building:
            # World items may be objects or dicts; coerce to serializable
            try:
                serial_world: dict[str, Any] = {}
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
                pp_context["world"] = serial_world
            except Exception:
                pp_context["world"] = {
                    k: list(v.keys()) if isinstance(v, dict) else str(v)
                    for k, v in world_building.items()
                }

        tasks["plot_points"] = bootstrap_field(
            "plot_points",
            pp_context,
            "bootstrapper/fill_plot_points.j2",
            is_list=True,
            list_count=needed_plot_points,
        )

    if not tasks:
        return plot_outline, None

    # Process each field sequentially to avoid parallel similar-sounding outputs
    task_keys = list(tasks.keys())
    for field in task_keys:
        value, usage = await tasks[field]
        if usage:
            for k, v in usage.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        usage_data[sub_k] = usage_data.get(sub_k, 0) + sub_v
                else:
                    usage_data[k] = usage_data.get(k, 0) + v
        if field == "plot_points":
            new_points = value
            final_points = [
                p
                for p in plot_outline.get("plot_points", [])
                if not utils._is_fill_in(p)
            ]
            final_points.extend(new_points)
            plot_outline["plot_points"] = final_points[
                : config.TARGET_PLOT_POINTS_INITIAL_GENERATION
            ]
        elif value:
            plot_outline[field] = value

    # Robust post-processing: ensure critical fields are non-empty even if LLM failed
    # Title fallback
    if not plot_outline.get("title") or utils._is_fill_in(plot_outline.get("title")):
        fallback_title = (
            config.DEFAULT_PLOT_OUTLINE_TITLE
            if getattr(config, "DEFAULT_PLOT_OUTLINE_TITLE", "").strip()
            else "Untitled Narrative"
        )
        plot_outline["title"] = fallback_title

    # Summary fallback (minimal but non-empty)
    if not plot_outline.get("summary") or utils._is_fill_in(
        plot_outline.get("summary")
    ):
        protagonist = (
            plot_outline.get("protagonist_name") or config.DEFAULT_PROTAGONIST_NAME
        )
        setting = plot_outline.get("setting") or config.CONFIGURED_SETTING_DESCRIPTION
        logline = plot_outline.get("logline")
        if logline and isinstance(logline, str) and logline.strip():
            summary_fallback = logline.strip()
        else:
            summary_fallback = (
                f"{protagonist} faces rising stakes in {setting}. "
                f"A journey begins that will test conviction and change their world."
            )
        plot_outline["summary"] = summary_fallback

    if usage_data["total_tokens"] > 0:
        plot_outline["is_default"] = False
        plot_outline["source"] = "bootstrapped"
    return plot_outline, usage_data if usage_data["total_tokens"] > 0 else None
