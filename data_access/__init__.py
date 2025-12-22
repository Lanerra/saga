# data_access/__init__.py
"""Expose the `data_access` public API via lazy exports.

Notes:
    Import-time behavior:
        This package avoids eager imports of heavy dependencies by lazily resolving exports
        on first access via [`__getattr__()`](data_access/__init__.py:76). This preserves the
        existing import surface (for example, `from data_access import get_world_item_by_id`)
        while reducing import-time fan-out (NumPy/Neo4j/etc.).

    Contract:
        - Only names listed in `_EXPORTS` and `_SUBMODULES` are exposed via lazy resolution.
        - Missing attributes raise `AttributeError` as normal module attribute access would.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Public function exports (attribute name -> (module, attribute))
_EXPORTS: dict[str, tuple[str, str]] = {
    # Plot
    "save_plot_outline_to_db": ("data_access.plot_queries", "save_plot_outline_to_db"),
    "get_plot_outline_from_db": ("data_access.plot_queries", "get_plot_outline_from_db"),
    "append_plot_point": ("data_access.plot_queries", "append_plot_point"),
    "plot_point_exists": ("data_access.plot_queries", "plot_point_exists"),
    "get_last_plot_point_id": ("data_access.plot_queries", "get_last_plot_point_id"),
    # Characters
    "sync_characters": ("data_access.character_queries", "sync_characters"),
    "get_character_profile_by_id": ("data_access.character_queries", "get_character_profile_by_id"),
    "get_character_profile_by_name": ("data_access.character_queries", "get_character_profile_by_name"),
    "resolve_character_name": ("data_access.character_queries", "resolve_character_name"),
    "get_character_info_for_snippet_from_db": (
        "data_access.character_queries",
        "get_character_info_for_snippet_from_db",
    ),
    # World
    "sync_world_items": ("data_access.world_queries", "sync_world_items"),
    "get_world_building": ("data_access.world_queries", "get_world_building"),
    "get_world_elements_for_snippet_from_db": (
        "data_access.world_queries",
        "get_world_elements_for_snippet_from_db",
    ),
    "resolve_world_name": ("data_access.world_queries", "resolve_world_name"),
    "get_world_item_by_name": ("data_access.world_queries", "get_world_item_by_name"),
    "get_world_item_by_id": ("data_access.world_queries", "get_world_item_by_id"),
    # Chapters
    "load_chapter_count_from_db": ("data_access.chapter_queries", "load_chapter_count_from_db"),
    "save_chapter_data_to_db": ("data_access.chapter_queries", "save_chapter_data_to_db"),
    "get_chapter_data_from_db": ("data_access.chapter_queries", "get_chapter_data_from_db"),
    "get_embedding_from_db": ("data_access.chapter_queries", "get_embedding_from_db"),
    # KG
    "add_kg_triples_batch_to_db": ("data_access.kg_queries", "add_kg_triples_batch_to_db"),
    "query_kg_from_db": ("data_access.kg_queries", "query_kg_from_db"),
    "get_most_recent_value_from_db": ("data_access.kg_queries", "get_most_recent_value_from_db"),
    "get_novel_info_property_from_db": ("data_access.kg_queries", "get_novel_info_property_from_db"),
    "get_shortest_path_length_between_entities": (
        "data_access.kg_queries",
        "get_shortest_path_length_between_entities",
    ),
}

# Common submodule conveniences, so `from data_access import chapter_queries` keeps working.
_SUBMODULES: dict[str, str] = {
    "chapter_queries": "data_access.chapter_queries",
    "character_queries": "data_access.character_queries",
    "kg_queries": "data_access.kg_queries",
    "plot_queries": "data_access.plot_queries",
    "world_queries": "data_access.world_queries",
    "cache_coordinator": "data_access.cache_coordinator",
    "cypher_builders": "data_access.cypher_builders",
}


def __getattr__(name: str) -> Any:
    """Resolve public exports and submodules lazily on first attribute access.

    Args:
        name: Attribute name being accessed on the `data_access` package.

    Returns:
        The resolved attribute (a function export or an imported submodule).

    Raises:
        AttributeError: If `name` is not a declared export or submodule.

    Notes:
        This is the mechanism that keeps package import-time dependency fan-out low.
        Callers should not rely on side effects at `import data_access` time; submodules
        are imported only when accessed.
    """
    if name in _EXPORTS:
        module_path, attr_name = _EXPORTS[name]
        module = import_module(module_path)
        return getattr(module, attr_name)

    if name in _SUBMODULES:
        return import_module(_SUBMODULES[name])

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return module attributes for IDEs and introspection.

    Returns:
        A sorted list containing actual module globals plus lazy export/submodule names.
    """
    return sorted(set(list(globals().keys()) + list(_EXPORTS.keys()) + list(_SUBMODULES.keys())))


__all__ = [
    "save_plot_outline_to_db",
    "get_plot_outline_from_db",
    "append_plot_point",
    "plot_point_exists",
    "get_last_plot_point_id",
    "sync_characters",
    "get_character_profile_by_id",
    "get_character_profile_by_name",
    "resolve_character_name",
    "get_character_info_for_snippet_from_db",
    "sync_world_items",
    "get_world_building",
    "get_world_elements_for_snippet_from_db",
    "resolve_world_name",
    "get_world_item_by_name",
    "get_world_item_by_id",
    "load_chapter_count_from_db",
    "save_chapter_data_to_db",
    "get_chapter_data_from_db",
    "get_embedding_from_db",
    "add_kg_triples_batch_to_db",
    "query_kg_from_db",
    "get_most_recent_value_from_db",
    "get_novel_info_property_from_db",
    "get_shortest_path_length_between_entities",
]
