# tests/test_scene_extraction_validation.py
"""Integration tests for scene extraction with spaCy validation."""

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import config
from core.langgraph.nodes.scene_extraction_normalization import (
    consolidate_scene_extractions,
)
from core.langgraph.nodes.scene_extraction_validation import (
    _get_normalized_entity_key,
    _validate_entity_with_spacy,
)


@pytest.fixture
def mock_text_processing_service() -> Generator[MagicMock, None, None]:
    """Create a mock text processing service."""
    with patch("core.langgraph.nodes.scene_extraction_validation._get_text_processing_service") as mock_getter:
        mock_service = MagicMock()
        mock_spacy = MagicMock()
        mock_spacy.is_loaded.return_value = True
        mock_spacy.verify_entity_presence.return_value = True

        # Make normalize_entity_name return different values based on input
        def normalize_side_effect(name: str) -> str:
            if name == "John":
                return "john"
            elif name == "The Dark Tower":
                return "dark tower"  # Normalized form
            elif name == "Sword":
                return "sword"
            elif name == "The Castle":
                return "castle"
            else:
                return name.lower()

        mock_spacy.normalize_entity_name.side_effect = normalize_side_effect
        mock_service.spacy_service = mock_spacy
        mock_getter.return_value = mock_service
        yield mock_service


@pytest.fixture(autouse=True, params=[False, True])
def entity_validation_settings(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    effective = config.EffectiveSettings.model_validate({**config.snapshot_settings().model_dump(), "ENABLE_ENTITY_VALIDATION": request.param})
    with config.bind_settings(effective):
        yield


def test_validate_entity_with_spacy_enabled(mock_text_processing_service: MagicMock) -> None:
    """Exact spans are mandatory with either optional NLP setting."""

    result = _validate_entity_with_spacy("John works at Google", "John", {"John"})
    assert _validate_entity_with_spacy("John works at Google", "John") is False

    assert result is True
    assert _validate_entity_with_spacy("John works at Google", "Jane") is False
    mock_text_processing_service.spacy_service.verify_entity_presence.assert_not_called()


def test_validate_entity_with_spacy_disabled(mock_text_processing_service: MagicMock) -> None:
    """Disabling optional NLP cannot admit an absent name."""

    result = _validate_entity_with_spacy("John works at Google", "John", {"John"})

    assert result is True
    assert _validate_entity_with_spacy("John works at Google", "Jane") is False
    mock_text_processing_service.spacy_service.verify_entity_presence.assert_not_called()


def test_validate_entity_with_spacy_not_loaded(mock_text_processing_service: MagicMock) -> None:
    """Test entity validation when spaCy model not loaded."""
    mock_text_processing_service.spacy_service.is_loaded.return_value = False

    result = _validate_entity_with_spacy("John works at Google", "John", {"John"})

    assert result is True  # Should return True when model not loaded
    mock_text_processing_service.spacy_service.verify_entity_presence.assert_not_called()


def test_validate_entity_with_spacy_not_found(mock_text_processing_service: MagicMock) -> None:
    """Test entity validation when entity not found."""
    mock_text_processing_service.spacy_service.verify_entity_presence.return_value = False

    result = _validate_entity_with_spacy("John works at Google", "Jane", {"Jane"})

    assert result is False


def test_get_normalized_entity_key_with_spacy(mock_text_processing_service: MagicMock) -> None:
    """Statistical normalization cannot rewrite exact identity."""

    result = _get_normalized_entity_key("The Dark Tower")

    assert result == "The Dark Tower"
    mock_text_processing_service.spacy_service.normalize_entity_name.assert_not_called()


def test_get_normalized_entity_key_fallback(mock_text_processing_service: MagicMock) -> None:
    """Literal identity needs no fallback normalization."""

    result = _get_normalized_entity_key("The Dark Tower")

    assert result == "The Dark Tower"
    mock_text_processing_service.spacy_service.normalize_entity_name.assert_not_called()


def test_consolidate_scene_extractions_with_spacy(mock_text_processing_service: MagicMock) -> None:
    """Consolidation preserves identity without statistical normalization."""

    scene_results = [
        {
            "characters": [
                {"name": "John", "description": "Main character"},
                {"name": "The Dark Tower", "description": "Location"},
            ],
            "world_items": [
                {"name": "Sword", "description": "Weapon"},
                {"name": "The Castle", "description": "Fortress"},
            ],
            "relationships": [
                {"source_name": "John", "target_name": "The Dark Tower", "relationship_type": "VISITS"},
            ],
        }
    ]

    result = consolidate_scene_extractions(scene_results)


    assert len(result["characters"]) == 2
    assert len(result["world_items"]) == 2
    assert len(result["relationships"]) == 1

    mock_text_processing_service.spacy_service.normalize_entity_name.assert_not_called()


def test_consolidate_scene_extractions_fallback(mock_text_processing_service: MagicMock) -> None:
    """Case-distinct identities must not merge."""

    scene_results = [
        {
            "characters": [
                {"name": "John", "description": "Main character"},
                {"name": "john", "description": "Alternative description"},
            ],
            "world_items": [],
            "relationships": [],
        }
    ]

    result = consolidate_scene_extractions(scene_results)

    assert result["characters"] == scene_results[0]["characters"]

    # Should not call spaCy service when disabled
    mock_text_processing_service.spacy_service.normalize_entity_name.assert_not_called()


def test_consolidate_scene_extractions_keeps_latest_description_and_history(mock_text_processing_service: MagicMock) -> None:
    """Later shorter assertions supersede prose without discarding history."""

    scene_results = [
        {
            "characters": [
                {"name": "John", "description": "Much longer description with more details"},
                {"name": "John", "description": "Short desc"},
            ],
            "world_items": [],
            "relationships": [],
        }
    ]

    result = consolidate_scene_extractions(scene_results)

    assert len(result["characters"]) == 1
    assert result["characters"][0]["description"] == "Short desc"
    assert result["characters"][0]["attributes"]["scene_assertions"] == scene_results[0]["characters"]


def test_consolidate_scene_extractions_relationship_deduplication(mock_text_processing_service: MagicMock) -> None:
    """Case-distinct endpoint triples must remain distinct."""

    scene_results = [
        {
            "characters": [],
            "world_items": [],
            "relationships": [
                {"source_name": "John", "target_name": "The Dark Tower", "relationship_type": "VISITS"},
                {"source_name": "john", "target_name": "dark tower", "relationship_type": "VISITS"},
            ],
        }
    ]

    result = consolidate_scene_extractions(scene_results)

    assert result["relationships"] == scene_results[0]["relationships"]


def test_consolidate_scene_extractions_multiple_scenes(mock_text_processing_service: MagicMock) -> None:
    """Test consolidation across multiple scenes."""

    scene_results = [
        {
            "characters": [
                {"name": "John", "description": "Scene 1 description"},
            ],
            "world_items": [],
            "relationships": [],
        },
        {
            "characters": [
                {"name": "John", "description": "Later"},
            ],
            "world_items": [],
            "relationships": [],
        },
    ]

    result = consolidate_scene_extractions(scene_results)

    # Chronology, not description length, decides the current snapshot.
    assert len(result["characters"]) == 1
    assert result["characters"][0]["description"] == "Later"
    assert result["characters"][0]["attributes"]["scene_assertions"] == [scene["characters"][0] for scene in scene_results]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("category_in_attributes", [False, True])
def test_consolidation_preserves_equal_names_in_distinct_categories(
    reverse: bool, category_in_attributes: bool
) -> None:
    location: dict[str, Any] = {
        "name": "Crossing",
        "type": "Location",
        "description": "A river crossing",
        "first_appearance_chapter": 3,
        "scene_index": 0,
        "attributes": {"id": "location-crossing", "rules": ["Toll required"]},
    }
    event: dict[str, Any] = {
        "name": "crossing",
        "type": "Event",
        "description": "The annual ceremonial crossing of the river",
        "first_appearance_chapter": 3,
        "scene_index": 1,
        "attributes": {"id": "event-crossing", "goals": ["Reach the far bank"]},
    }
    if category_in_attributes:
        location["type"] = event["type"] = "Item"
        location["attributes"]["category"] = "place"
        event["attributes"]["category"] = "battle"
    items = [event, location] if reverse else [location, event]
    scenes = [{"world_items": [item]} for item in items]

    result = consolidate_scene_extractions(scenes)

    assert result == {"characters": [], "world_items": items, "relationships": []}
    for index, item in enumerate(items):
        assert result["world_items"][index] is item
    assert consolidate_scene_extractions(scenes) == result


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("equal_length", [False, True])
def test_consolidation_preserves_world_item_duplicate_selection(
    reverse: bool, equal_length: bool
) -> None:
    first: dict[str, Any] = {
        "name": "Crossing",
        "type": "Location",
        "description": "A ford",
        "first_appearance_chapter": 3,
        "scene_index": 0,
        "attributes": {"id": "first-crossing", "category": "Location", "rules": ["Toll required"]},
    }
    second: dict[str, Any] = {
        "name": "crossing",
        "type": "Location",
        "description": "A gate" if equal_length else "A ford beside the village gate",
        "first_appearance_chapter": 3,
        "scene_index": 1,
        "attributes": {"id": "second-crossing", "category": "place", "key_elements": ["Village gate"]},
    }
    items = [second, first] if reverse else [first, second]


    result = consolidate_scene_extractions([{"world_items": [item]} for item in items])

    assert result == {"characters": [], "world_items": items, "relationships": []}
    for index, item in enumerate(items):
        assert result["world_items"][index] is item


def test_validate_entity_with_spacy_exception_handling(mock_text_processing_service: MagicMock) -> None:
    """A broken optional NLP service cannot change exact span validation."""
    mock_text_processing_service.spacy_service.verify_entity_presence.side_effect = Exception("Test error")


    result = _validate_entity_with_spacy("John works at Google", "John", {"John"})

    assert result is True
    assert _validate_entity_with_spacy("John works at Google", "Jane") is False
    mock_text_processing_service.spacy_service.verify_entity_presence.assert_not_called()


def test_get_normalized_entity_key_exception_handling(mock_text_processing_service: MagicMock) -> None:
    """A broken optional normalizer cannot rewrite identity."""
    mock_text_processing_service.spacy_service.normalize_entity_name.side_effect = Exception("Test error")


    result = _get_normalized_entity_key("The Dark Tower")

    assert result == "The Dark Tower"
    mock_text_processing_service.spacy_service.normalize_entity_name.assert_not_called()
