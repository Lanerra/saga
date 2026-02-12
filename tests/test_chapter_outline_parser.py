# tests/test_chapter_outline_parser.py
"""Test for ChapterOutlineParser implementation."""

import json
import os
import tempfile

import pytest

from core.parsers import ChapterOutlineParser


@pytest.fixture
def sample_chapter_outline_data():
    """Create sample chapter outline data for testing."""
    return {
        "chapter_number": 1,
        "act_number": 1,
        "title": "The Beginning",
        "summary": "The hero's journey begins",
        "scene_description": "Eleanor Whitaker patrols the refugee camp's muddy perimeter as mist swallows the pines near the Oconee River swamps. Her lantern flickers over Sarah's bloodstained doll, found where the child vanished.",
        "key_beats": [
            "Eleanor Whitaker discovers Sarah's bloodstained doll at the swamp's edge during her night patrol",
            "James Carter finds charred Confederate musket fragments near the marshes",
            "Thomas Reed discovers claw marks on his cabin porch",
        ],
        "plot_point": "Eleanor's discovery of the doll ignites the hunt, exposing the creature's pattern.",
    }


@pytest.fixture
def temp_chapter_outline_file(sample_chapter_outline_data):
    """Create a temporary chapter outline file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_chapter_outline_data, f)
        temp_file = f.name

    yield temp_file

    # Clean up
    if os.path.exists(temp_file):
        os.unlink(temp_file)


@pytest.mark.asyncio
async def test_chapter_outline_parser_initialization():
    """Test that ChapterOutlineParser initializes correctly."""
    parser = ChapterOutlineParser(chapter_outline_path="test_path.json", chapter_number=1)

    assert parser.chapter_outline_path == "test_path.json"
    assert parser.chapter_number == 1


@pytest.mark.asyncio
async def test_chapter_outline_parser_parse_chapter_outline(temp_chapter_outline_file, sample_chapter_outline_data):
    """Test that parse_chapter_outline correctly reads and parses JSON."""
    parser = ChapterOutlineParser(chapter_outline_path=temp_chapter_outline_file, chapter_number=1)

    result = await parser.parse_chapter_outline()

    assert result == sample_chapter_outline_data


@pytest.mark.asyncio
async def test_chapter_outline_parser_parse_chapter(temp_chapter_outline_file):
    """Test that _parse_chapter correctly parses chapter data."""
    parser = ChapterOutlineParser(chapter_outline_path=temp_chapter_outline_file, chapter_number=1)

    # Parse the chapter outline
    chapter_outline_data = await parser.parse_chapter_outline()

    # Parse the chapter
    chapter = parser._parse_chapter(chapter_outline_data)

    assert chapter.number == 1
    assert chapter.act_number == 1
    assert chapter.title == "The Beginning"
    assert chapter.summary == "The hero's journey begins"
    assert chapter.created_chapter == 1
    assert chapter.is_provisional == False


@pytest.mark.asyncio
async def test_chapter_outline_parser_parse_scenes(temp_chapter_outline_file):
    """Test that _parse_scenes correctly parses scene data."""
    parser = ChapterOutlineParser(chapter_outline_path=temp_chapter_outline_file, chapter_number=1)

    # Parse the chapter outline
    chapter_outline_data = await parser.parse_chapter_outline()

    # Parse the scenes (provide empty list for character_names as we're not using the database)
    scenes = parser._parse_scenes(chapter_outline_data, [])

    # Parser creates ONE scene per chapter from scene_description
    assert len(scenes) == 1
    assert scenes[0].chapter_number == 1
    assert scenes[0].scene_index == 0
    assert scenes[0].title == "Chapter 1 Scene 1"
    assert scenes[0].pov_character == "Eleanor Whitaker"
    assert len(scenes[0].beats) == 3
    assert scenes[0].beats[0] == "Eleanor Whitaker discovers Sarah's bloodstained doll at the swamp's edge during her night patrol"
    assert scenes[0].beats[1] == "James Carter finds charred Confederate musket fragments near the marshes"
    assert scenes[0].beats[2] == "Thomas Reed discovers claw marks on his cabin porch"


@pytest.mark.asyncio
async def test_chapter_outline_parser_parse_scene_events(temp_chapter_outline_file):
    """Test that _parse_scene_events correctly parses event data."""
    parser = ChapterOutlineParser(chapter_outline_path=temp_chapter_outline_file, chapter_number=1)

    # Parse the chapter outline
    chapter_outline_data = await parser.parse_chapter_outline()

    # Parse the scene events (provide empty list for character_names)
    events = parser._parse_scene_events(chapter_outline_data, [])

    # Parser creates ONE event per beat in key_beats list
    assert len(events) == 3
    assert events[0].chapter_number == 1
    assert events[0].act_number == 1
    assert events[0].scene_index == 0
    assert events[0].event_type == "SceneEvent"
    assert events[0].name == "Eleanor Whitaker discovers Sarah's bloodstained doll at the swamp's edge during her night patrol"

    assert events[1].chapter_number == 1
    assert events[1].act_number == 1
    assert events[1].scene_index == 0
    assert events[1].event_type == "SceneEvent"
    assert events[1].name == "James Carter finds charred Confederate musket fragments near the marshes"

    assert events[2].chapter_number == 1
    assert events[2].act_number == 1
    assert events[2].scene_index == 0
    assert events[2].event_type == "SceneEvent"
    assert events[2].name == "Thomas Reed discovers claw marks on his cabin porch"


@pytest.mark.asyncio
async def test_parse_locations_matches_known_locations(temp_chapter_outline_file):
    """_parse_locations returns locations whose names appear in outline text."""
    parser = ChapterOutlineParser(chapter_outline_path=temp_chapter_outline_file, chapter_number=1)
    chapter_outline_data = await parser.parse_chapter_outline()

    known_locations = [
        {"id": "loc-camp", "name": "refugee camp", "description": "A muddy camp"},
        {"id": "loc-swamp", "name": "Oconee River swamps", "description": "Dense swampland"},
        {"id": "loc-castle", "name": "Castle Blackrock", "description": "A dark fortress"},
    ]

    locations = parser._parse_locations(chapter_outline_data, known_locations, chapter_number=1)

    assert len(locations) == 1
    assert locations[0].name in {"refugee camp", "Oconee River swamps"}
    assert locations[0].created_chapter == 1


@pytest.mark.asyncio
async def test_parse_locations_case_insensitive(temp_chapter_outline_file):
    """_parse_locations matches location names regardless of case."""
    parser = ChapterOutlineParser(chapter_outline_path=temp_chapter_outline_file, chapter_number=1)
    chapter_outline_data = await parser.parse_chapter_outline()

    known_locations = [
        {"id": "loc-camp", "name": "Refugee Camp", "description": "A muddy camp"},
    ]

    locations = parser._parse_locations(chapter_outline_data, known_locations, chapter_number=1)

    assert len(locations) == 1
    assert locations[0].name == "Refugee Camp"
    assert locations[0].id == "loc-camp"


@pytest.mark.asyncio
async def test_parse_locations_no_matches(temp_chapter_outline_file):
    """_parse_locations creates a provisional location from scene_description when no known locations match."""
    parser = ChapterOutlineParser(chapter_outline_path=temp_chapter_outline_file, chapter_number=1)
    chapter_outline_data = await parser.parse_chapter_outline()

    known_locations = [
        {"id": "loc-castle", "name": "Castle Blackrock", "description": "A dark fortress"},
    ]

    locations = parser._parse_locations(chapter_outline_data, known_locations, chapter_number=1)

    assert len(locations) == 1
    assert locations[0].is_provisional is True
    assert locations[0].created_chapter == 1
    assert locations[0].category == "Location"
    assert isinstance(locations[0].name, str)
    assert len(locations[0].name) > 0
    assert locations[0].description == chapter_outline_data["scene_description"]


@pytest.mark.asyncio
async def test_parse_locations_empty_known_locations(temp_chapter_outline_file):
    """_parse_locations creates a provisional location from scene_description when known_locations is empty."""
    parser = ChapterOutlineParser(chapter_outline_path=temp_chapter_outline_file, chapter_number=1)
    chapter_outline_data = await parser.parse_chapter_outline()

    locations = parser._parse_locations(chapter_outline_data, [], chapter_number=1)

    assert len(locations) == 1
    assert locations[0].is_provisional is True
    assert locations[0].created_chapter == 1
    assert locations[0].category == "Location"
    assert isinstance(locations[0].name, str)
    assert len(locations[0].name) > 0
    assert locations[0].description == chapter_outline_data["scene_description"]


@pytest.mark.asyncio
async def test_chapter_outline_parser_generate_id():
    """Test that _generate_id generates consistent IDs."""
    parser = ChapterOutlineParser()

    # Generate IDs for the same entity
    id1 = parser._generate_id("Chapter", 1, 0)
    id2 = parser._generate_id("Chapter", 1, 0)

    # IDs should be the same for the same input
    assert id1 == id2

    # Generate IDs for different entities
    id3 = parser._generate_id("Scene", 1, 0)
    id4 = parser._generate_id("Event", 1, 0, 0)

    # IDs should be different for different entity types
    assert id1 != id3
    assert id1 != id4
    assert id3 != id4


@pytest.mark.asyncio
async def test_chapter_outline_parser_missing_scenes(temp_chapter_outline_file):
    """Test that parser handles missing scenes gracefully."""
    # Create a chapter outline without scenes
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"chapter_number": 1, "act_number": 1, "title": "Test Chapter", "summary": "Test summary"}, f)
        temp_file = f.name

    try:
        parser = ChapterOutlineParser(chapter_outline_path=temp_file, chapter_number=1)

        # Parse the chapter outline
        chapter_outline_data = await parser.parse_chapter_outline()

        # Parse scenes (should return empty list)
        scenes = parser._parse_scenes(chapter_outline_data, [])
        assert len(scenes) == 0

        # Parse events (should return empty list)
        events = parser._parse_scene_events(chapter_outline_data, [])
        assert len(events) == 0

        # Parse locations (should return empty list with no known locations)
        locations = parser._parse_locations(chapter_outline_data, [], chapter_number=1)
        assert locations == []

    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


@pytest.mark.asyncio
async def test_chapter_outline_parser_invalid_json():
    """Test that parser handles invalid JSON gracefully."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("invalid json {{{")
        temp_file = f.name

    try:
        parser = ChapterOutlineParser(chapter_outline_path=temp_file, chapter_number=1)

        # Should raise ValueError for invalid JSON
        with pytest.raises(ValueError):
            await parser.parse_chapter_outline()
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


@pytest.mark.asyncio
async def test_chapter_outline_parser_file_not_found():
    """Test that parser handles missing file gracefully."""
    parser = ChapterOutlineParser(chapter_outline_path="/nonexistent/file.json", chapter_number=1)

    # Should raise ValueError for missing file
    with pytest.raises(ValueError):
        await parser.parse_chapter_outline()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
