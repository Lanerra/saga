# data_access/__init__.py
# This file makes the data_access directory a Python package.
# You can also use it to expose a simpler API from this package if desired.

from .chapter_queries import (
    find_similar_chapters_in_db,
    get_chapter_data_from_db,
    get_chapters_data_from_db,
    get_embedding_from_db,
    load_chapter_count_from_db,
    save_chapter_data_to_db,
)
from .character_queries import (
    get_character_info_for_snippet_from_db,
    get_character_profile_by_name,
    get_character_profiles_from_db,
    resolve_character_name,
    sync_characters,
)
from .character_queries import (
    sync_full_state_from_object_to_db as sync_characters_full_state_from_object_to_db,
)
from .kg_queries import (
    add_kg_triples_batch_to_db,
    fetch_unresolved_dynamic_relationships,
    get_most_recent_value_from_db,
    get_novel_info_property_from_db,
    get_shortest_path_length_between_entities,
    normalize_existing_relationship_types,
    query_kg_from_db,
    update_dynamic_relationship_type,
)
from .plot_queries import (
    append_plot_point,
    get_last_plot_point_id,
    get_plot_outline_from_db,
    plot_point_exists,
    save_plot_outline_to_db,
)
from .repository import chapter_repository
from .world_queries import (
    get_world_building_from_db,
    get_world_elements_for_snippet_from_db,
    get_world_item_by_id,
    get_world_item_by_name,
    resolve_world_name,
    sync_world_items,
)
from .world_queries import (
    sync_full_state_from_object_to_db as sync_world_full_state_from_object_to_db,
)

__all__ = [
    "save_plot_outline_to_db",
    "get_plot_outline_from_db",
    "append_plot_point",
    "plot_point_exists",
    "get_last_plot_point_id",
    "sync_characters_full_state_from_object_to_db",
    "sync_characters",
    "get_character_profile_by_name",
    "resolve_character_name",
    "get_character_profiles_from_db",
    "get_character_info_for_snippet_from_db",
    "sync_world_full_state_from_object_to_db",
    "sync_world_items",
    "get_world_building_from_db",
    "get_world_elements_for_snippet_from_db",
    "resolve_world_name",
    "get_world_item_by_name",
    "get_world_item_by_id",
    "chapter_repository",
    "load_chapter_count_from_db",
    "save_chapter_data_to_db",
    "get_chapter_data_from_db",
    "get_chapters_data_from_db",
    "get_embedding_from_db",
    "find_similar_chapters_in_db",
    "add_kg_triples_batch_to_db",
    "query_kg_from_db",
    "normalize_existing_relationship_types",
    "get_most_recent_value_from_db",
    "get_novel_info_property_from_db",
    "fetch_unresolved_dynamic_relationships",
    "update_dynamic_relationship_type",
    "get_shortest_path_length_between_entities",
]
