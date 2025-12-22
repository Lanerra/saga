"""Coordinate `data_access` cache invalidation.

This module centralizes invalidation so cache management is:

- explicit (callers do not reach into individual functions to call `cache_clear()`), and
- consistently applied in write paths (so read-through caches do not serve stale data).

Notes:
    - This module avoids eager importing `data_access` submodules to keep package import-time
      side effects and dependency fan-out low.
    - Invalidation is best-effort: if a cached function no longer exposes `cache_clear()`,
      the coordinator reports that it could not clear that cache instead of raising.
"""

from __future__ import annotations

from typing import Any


def _best_effort_cache_clear(fn: Any) -> bool:
    """Clear a function cache if the function exposes `cache_clear()`.

    Args:
        fn: A function object that may or may not expose a `cache_clear()` attribute.

    Returns:
        True if `cache_clear()` existed and was called. False if the function does not expose
        a callable `cache_clear()`.

    Notes:
        This is intentionally best-effort so cache coordination remains stable even if the
        caching implementation changes (e.g., migrating away from `async_lru`).
    """
    cache_clear = getattr(fn, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
        return True
    return False


def clear_character_read_caches() -> dict[str, bool]:
    """Clear caches for character read APIs.

    Returns:
        A mapping of cache-bearing API names to whether their cache was cleared.

    Notes:
        Character read functions are cached (read-through) to reduce repeated Neo4j IO.
        Write paths that mutate character state (for example, `sync_characters()`) should
        call this coordinator to invalidate cached reads.
    """
    # Local import to avoid eager module import side effects.
    from data_access import character_queries

    return {
        "get_character_profile_by_name": _best_effort_cache_clear(character_queries.get_character_profile_by_name),
        "get_character_profile_by_id": _best_effort_cache_clear(character_queries.get_character_profile_by_id),
    }


def clear_world_read_caches() -> dict[str, bool]:
    """Clear caches for world read APIs.

    Returns:
        A mapping of cache-bearing API names to whether their cache was cleared.

    Notes:
        World read functions are cached (read-through). Write paths that upsert world items
        should invalidate these caches to prevent stale reads.
    """
    from data_access import world_queries

    return {
        "get_world_item_by_id": _best_effort_cache_clear(world_queries.get_world_item_by_id),
    }


def clear_kg_read_caches() -> dict[str, bool]:
    """Clear caches for KG read APIs.

    Returns:
        A mapping of cache-bearing API names to whether their cache was cleared.

    Notes:
        KG read functions are cached (read-through). KG write paths (for example,
        `add_kg_triples_batch_to_db()`) should invalidate these caches after successful writes.
    """
    from data_access import kg_queries

    return {
        "query_kg_from_db": _best_effort_cache_clear(kg_queries.query_kg_from_db),
        "get_novel_info_property_from_db": _best_effort_cache_clear(kg_queries.get_novel_info_property_from_db),
    }


def clear_all_data_access_caches() -> dict[str, dict[str, bool]]:
    """Clear all known `data_access` caches.

    Returns:
        A mapping by subsystem ("character", "world", "kg") whose values are the per-function
        cleared/not-cleared mappings.

    Notes:
        This is intended for tests and debug tooling. Production write paths should prefer
        the narrower invalidation entrypoints for the data they mutate.
    """
    return {
        "character": clear_character_read_caches(),
        "world": clear_world_read_caches(),
        "kg": clear_kg_read_caches(),
    }
