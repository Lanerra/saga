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


@patch("core.langgraph.nodes.scene_extraction_validation.config")
def test_validate_entity_with_spacy_enabled(mock_config: MagicMock, mock_text_processing_service: MagicMock) -> None:
    """Test entity validation when enabled."""
    mock_config.settings.ENABLE_ENTITY_VALIDATION = True

    result = _validate_entity_with_spacy("John works at Google", "John")

    assert result is True
    mock_text_processing_service.spacy_service.verify_entity_presence.assert_called_once()


@patch("core.langgraph.nodes.scene_extraction_validation.config")
def test_validate_entity_with_spacy_disabled(mock_config: MagicMock, mock_text_processing_service: MagicMock) -> None:
    """Test entity validation when disabled."""
    mock_config.settings.ENABLE_ENTITY_VALIDATION = False

    result = _validate_entity_with_spacy("John works at Google", "John")

    assert result is True
    # Should not call spaCy service when disabled
    mock_text_processing_service.spacy_service.verify_entity_presence.assert_not_called()


@patch("core.langgraph.nodes.scene_extraction_validation.config")
def test_validate_entity_with_spacy_not_loaded(mock_config: MagicMock, mock_text_processing_service: MagicMock) -> None:
    """Test entity validation when spaCy model not loaded."""
    mock_config.settings.ENABLE_ENTITY_VALIDATION = True
    mock_text_processing_service.spacy_service.is_loaded.return_value = False

    result = _validate_entity_with_spacy("John works at Google", "John")

    assert result is True  # Should return True when model not loaded
    mock_text_processing_service.spacy_service.verify_entity_presence.assert_not_called()


@patch("core.langgraph.nodes.scene_extraction_validation.config")
def test_validate_entity_with_spacy_not_found(mock_config: MagicMock, mock_text_processing_service: MagicMock) -> None:
    """Test entity validation when entity not found."""
    mock_config.settings.ENABLE_ENTITY_VALIDATION = True
    mock_text_processing_service.spacy_service.verify_entity_presence.return_value = False

    result = _validate_entity_with_spacy("John works at Google", "Jane")

    assert result is False


@patch("core.langgraph.nodes.scene_extraction_validation.config")
def test_get_normalized_entity_key_with_spacy(mock_config: MagicMock, mock_text_processing_service: MagicMock) -> None:
    """Test entity key normalization with spaCy."""
    mock_config.settings.ENABLE_ENTITY_VALIDATION = True

    result = _get_normalized_entity_key("The Dark Tower")

    assert result == "dark tower"
    mock_text_processing_service.spacy_service.normalize_entity_name.assert_called_once_with("The Dark Tower")


@patch("core.langgraph.nodes.scene_extraction_validation.config")
def test_get_normalized_entity_key_fallback(mock_config: MagicMock, mock_text_processing_service: MagicMock) -> None:
    """Test entity key normalization fallback."""
    mock_config.settings.ENABLE_ENTITY_VALIDATION = False

    result = _get_normalized_entity_key("The Dark Tower")

    assert result == "the dark tower"
    # Should not call spaCy service when disabled
    mock_text_processing_service.spacy_service.normalize_entity_name.assert_not_called()


@patch("core.langgraph.nodes.scene_extraction_validation.config")
def test_consolidate_scene_extractions_with_spacy(mock_config: MagicMock, mock_text_processing_service: MagicMock) -> None:
    """Test scene consolidation using spaCy normalization."""
    mock_config.settings.ENABLE_ENTITY_VALIDATION = True

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

    # Should use spaCy normalization for deduplication
    assert len(result["characters"]) == 2
    assert len(result["world_items"]) == 2
    assert len(result["relationships"]) == 1

    # Verify spaCy was called for normalization
    assert mock_text_processing_service.spacy_service.normalize_entity_name.call_count >= 4


@patch("core.langgraph.nodes.scene_extraction_validation.config")
def test_consolidate_scene_extractions_fallback(mock_config: MagicMock, mock_text_processing_service: MagicMock) -> None:
    """Test scene consolidation with fallback normalization."""
    mock_config.settings.ENABLE_ENTITY_VALIDATION = False

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

    # Should deduplicate using simple case-insensitive matching
    assert len(result["characters"]) == 1
    assert result["characters"][0]["description"] == "Alternative description"

    # Should not call spaCy service when disabled
    mock_text_processing_service.spacy_service.normalize_entity_name.assert_not_called()


@patch("core.langgraph.nodes.scene_extraction_validation.config")
def test_consolidate_scene_extractions_keep_longest_description(mock_config: MagicMock, mock_text_processing_service: MagicMock) -> None:
    """Test that consolidation keeps the longest description."""
    mock_config.settings.ENABLE_ENTITY_VALIDATION = True

    scene_results = [
        {
            "characters": [
                {"name": "John", "description": "Short desc"},
                {"name": "John", "description": "Much longer description with more details"},
            ],
            "world_items": [],
            "relationships": [],
        }
    ]

    result = consolidate_scene_extractions(scene_results)

    assert len(result["characters"]) == 1
    assert result["characters"][0]["description"] == "Much longer description with more details"


@patch("core.langgraph.nodes.scene_extraction_validation.config")
def test_consolidate_scene_extractions_relationship_deduplication(mock_config: MagicMock, mock_text_processing_service: MagicMock) -> None:
    """Test relationship deduplication using spaCy normalization."""
    mock_config.settings.ENABLE_ENTITY_VALIDATION = True

    scene_results = [
        {
            "characters": [],
            "world_items": [],
            "relationships": [
                {"source_name": "John", "target_name": "The Dark Tower", "relationship_type": "VISITS"},
                {"source_name": "john", "target_name": "dark tower", "relationship_type": "VISITS"},  # Should deduplicate
            ],
        }
    ]

    result = consolidate_scene_extractions(scene_results)

    # Should deduplicate relationships using spaCy normalization
    assert len(result["relationships"]) == 1


@patch("core.langgraph.nodes.scene_extraction_validation.config")
def test_consolidate_scene_extractions_multiple_scenes(mock_config: MagicMock, mock_text_processing_service: MagicMock) -> None:
    """Test consolidation across multiple scenes."""
    mock_config.settings.ENABLE_ENTITY_VALIDATION = True

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
                {"name": "John", "description": "Scene 2 longer description"},
            ],
            "world_items": [],
            "relationships": [],
        },
    ]

    result = consolidate_scene_extractions(scene_results)

    # Should merge and keep the longest description
    assert len(result["characters"]) == 1
    assert result["characters"][0]["description"] == "Scene 2 longer description"


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("category_in_attributes", [False, True])
def test_consolidation_preserves_equal_names_in_distinct_categories(
    monkeypatch: pytest.MonkeyPatch, reverse: bool, category_in_attributes: bool
) -> None:
    monkeypatch.setitem(vars(config), "settings", config.settings.model_copy(update={"ENABLE_ENTITY_VALIDATION": False}))
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
    monkeypatch: pytest.MonkeyPatch, reverse: bool, equal_length: bool
) -> None:
    monkeypatch.setitem(vars(config), "settings", config.settings.model_copy(update={"ENABLE_ENTITY_VALIDATION": False}))
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
    selected = items[0] if equal_length else second

    result = consolidate_scene_extractions([{"world_items": [item]} for item in items])

    assert result == {"characters": [], "world_items": [selected], "relationships": []}
    assert result["world_items"][0] is selected


@patch("core.langgraph.nodes.scene_extraction_validation.config")
def test_validate_entity_with_spacy_exception_handling(mock_config: MagicMock, mock_text_processing_service: MagicMock) -> None:
    """Test entity validation exception handling."""
    mock_config.settings.ENABLE_ENTITY_VALIDATION = True
    mock_text_processing_service.spacy_service.verify_entity_presence.side_effect = Exception("Test error")

    # Should fallback to substring matching on exception
    result = _validate_entity_with_spacy("John works at Google", "John")

    assert result is True  # Should use fallback


@patch("core.langgraph.nodes.scene_extraction_validation.config")
def test_get_normalized_entity_key_exception_handling(mock_config: MagicMock, mock_text_processing_service: MagicMock) -> None:
    """Test entity key normalization exception handling."""
    mock_config.settings.ENABLE_ENTITY_VALIDATION = True
    mock_text_processing_service.spacy_service.normalize_entity_name.side_effect = Exception("Test error")

    # Should fallback to simple normalization on exception
    result = _get_normalized_entity_key("The Dark Tower")

    assert result == "the dark tower"  # Should use fallback
