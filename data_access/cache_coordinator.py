"""Centralized cache invalidation for data_access.

This module exists to prevent cache management from being:
- ad-hoc (e.g., tests calling .cache_clear() directly on individual functions), and
- forgotten in write paths (e.g., sync_* / batch-write functions).

Design goals:
- Keep imports light and avoid eager importing heavy modules at package import time.
- Provide narrow, explicit invalidation entrypoints for characters/world/kg.
- Be safe to call even if a cache implementation changes (best-effort cache_clear()).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _best_effort_cache_clear(fn: Any) -> bool:
    """Attempt to clear an async_lru cache on a function.

    Returns:
        True if a cache_clear() existed and was called, else False.
    """
    cache_clear = getattr(fn, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
        return True
    return False


def clear_character_read_caches() -> dict[str, bool]:
    """Clear caches for character read APIs."""
    # Local import to avoid eager module import side effects.
    from data_access import character_queries

    return {
        "get_character_profile_by_name": _best_effort_cache_clear(
            character_queries.get_character_profile_by_name
        ),
        "get_character_profile_by_id": _best_effort_cache_clear(
            character_queries.get_character_profile_by_id
        ),
    }


def clear_world_read_caches() -> dict[str, bool]:
    """Clear caches for world read APIs."""
    from data_access import world_queries

    return {
        "get_world_item_by_id": _best_effort_cache_clear(world_queries.get_world_item_by_id),
    }


def clear_kg_read_caches() -> dict[str, bool]:
    """Clear caches for KG read APIs."""
    from data_access import kg_queries

    return {
        "query_kg_from_db": _best_effort_cache_clear(kg_queries.query_kg_from_db),
        "get_novel_info_property_from_db": _best_effort_cache_clear(
            kg_queries.get_novel_info_property_from_db
        ),
    }


def clear_all_data_access_caches() -> dict[str, dict[str, bool]]:
    """Clear all known data_access caches (primarily for tests / debug tooling)."""
    return {
        "character": clear_character_read_caches(),
        "world": clear_world_read_caches(),
        "kg": clear_kg_read_caches(),
    }