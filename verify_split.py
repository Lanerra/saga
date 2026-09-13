#!/usr/bin/env python3
"""Verify scene_extraction split imports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.langgraph.nodes.scene_extraction import (
    _extract_characters_from_scene,
    _extract_events_from_scene,
    _extract_locations_from_scene,
    _extract_relationships_from_scene,
    extract_from_scene,
    extract_from_scenes,
)
from core.langgraph.nodes.scene_extraction_normalization import (
    consolidate_scene_extractions,
)
from core.langgraph.nodes.scene_extraction_parsing import (
    normalize_dict_items,
    normalize_triple_entities,
    parse_character_updates,
    parse_kg_triples,
    parse_world_updates,
)
from core.langgraph.nodes.scene_extraction_validation import (
    _get_normalized_entity_key,
    _validate_entity_with_spacy,
    load_spacy_model_if_enabled,
)

assert all(
    callable(function)
    for function in (
        extract_from_scenes,
        extract_from_scene,
        _extract_characters_from_scene,
        _extract_locations_from_scene,
        _extract_events_from_scene,
        _extract_relationships_from_scene,
        parse_character_updates,
        parse_world_updates,
        parse_kg_triples,
        normalize_triple_entities,
        normalize_dict_items,
        _validate_entity_with_spacy,
        _get_normalized_entity_key,
        load_spacy_model_if_enabled,
        consolidate_scene_extractions,
    )
)
print("All imports OK")
