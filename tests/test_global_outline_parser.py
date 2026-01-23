# tests/test_global_outline_parser.py
"""Test the GlobalOutlineParser implementation."""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.parsers.global_outline_parser import GlobalOutlineParser, MajorPlotPoint


@pytest.fixture
def sample_global_outline():
    """Sample valid global outline JSON."""
    return {
        "act_count": 3,
        "acts": [
            {
                "act_number": 1,
                "title": "Setup",
                "summary": "Introduction of hero",
                "key_events": ["Hero meets mentor"],
                "chapters_start": 1,
                "chapters_end": 7,
            },
            {
                "act_number": 2,
                "title": "Confrontation",
                "summary": "Hero faces challenges",
                "key_events": ["Major battle"],
                "chapters_start": 8,
                "chapters_end": 14,
            },
            {
                "act_number": 3,
                "title": "Resolution",
                "summary": "Hero triumphs",
                "key_events": ["Final victory"],
                "chapters_start": 15,
                "chapters_end": 20,
            },
        ],
        "inciting_incident": "The kingdom is threatened",
        "midpoint": "Hero discovers the truth",
        "climax": "Final confrontation with villain",
        "resolution": "Peace is restored",
        "character_arcs": [
            {
                "character_name": "Hero",
                "starting_state": "Naive",
                "ending_state": "Wise",
                "key_moments": ["First failure", "Learning from mentor"],
            }
        ],
        "thematic_progression": "From innocence to experience",
        "pacing_notes": "Fast-paced action throughout",
    }


@pytest.fixture
def mock_global_outline_file(sample_global_outline):
    """Create a temporary global outline file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_global_outline, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_parse_global_outline_success(mock_global_outline_file):
    """Test successful parsing of global outline."""
    parser = GlobalOutlineParser(global_outline_path=mock_global_outline_file)
    
    result = await parser.parse_global_outline()
    
    assert result is not None
    assert "inciting_incident" in result
    assert "midpoint" in result
    assert "climax" in result
    assert "resolution" in result


@pytest.mark.asyncio
async def test_parse_global_outline_file_not_found():
    """Test error handling when file is not found."""
    parser = GlobalOutlineParser(global_outline_path="/nonexistent/path.json")
    
    with pytest.raises(ValueError, match="Global outline file not found"):
        await parser.parse_global_outline()


@pytest.mark.asyncio
async def test_parse_global_outline_invalid_json():
    """Test error handling when JSON is invalid."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("invalid json {{{")
        temp_path = f.name
    
    try:
        parser = GlobalOutlineParser(global_outline_path=temp_path)
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            await parser.parse_global_outline()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_parse_major_plot_points(sample_global_outline):
    """Test parsing of major plot points."""
    parser = GlobalOutlineParser()
    
    plot_points = parser._parse_major_plot_points(sample_global_outline)
    
    assert len(plot_points) == 4
    assert all(isinstance(pp, MajorPlotPoint) for pp in plot_points)
    
    # Check sequence orders
    sequence_orders = [pp.sequence_order for pp in plot_points]
    assert sequence_orders == [1, 2, 3, 4]


def test_parse_major_plot_points_missing_data():
    """Test error handling when major plot points are missing."""
    parser = GlobalOutlineParser()
    incomplete_data = {
        "inciting_incident": "Test",
        "midpoint": "Test",
        # Missing climax and resolution
    }
    
    with pytest.raises(ValueError, match="Expected 4 major plot points"):
        parser._parse_major_plot_points(incomplete_data)


@pytest.mark.asyncio
async def test_parse_locations(sample_global_outline):
    """Test parsing of locations from narrative text using LLM extraction."""
    parser = GlobalOutlineParser()

    with patch.object(parser, '_extract_world_items_from_outline') as mock_extract:
        from models.kg_models import WorldItem
        mock_extract.return_value = [
            WorldItem(
                id="loc_1",
                name="Blackwater Creek",
                description="A small town",
                category="location",
                created_chapter=0,
                is_provisional=False,
            ),
            WorldItem(
                id="item_1",
                name="Bloodstained doll",
                description="A creepy doll",
                category="object",
                created_chapter=0,
                is_provisional=False,
            ),
        ]

        locations = await parser._parse_locations(sample_global_outline)

        assert len(locations) == 1
        assert locations[0].name is None
        assert locations[0].category == "Location"
        assert "Blackwater Creek" not in locations[0].description


@pytest.mark.asyncio
async def test_parse_items(sample_global_outline):
    """Test parsing of items from narrative text using LLM extraction."""
    parser = GlobalOutlineParser()

    with patch.object(parser, '_extract_world_items_from_outline') as mock_extract:
        from models.kg_models import WorldItem
        mock_extract.return_value = [
            WorldItem(
                id="loc_1",
                name="Blackwater Creek",
                description="A small town",
                category="location",
                created_chapter=0,
                is_provisional=False,
            ),
            WorldItem(
                id="item_1",
                name="Bloodstained doll",
                description="A creepy doll",
                category="object",
                created_chapter=0,
                is_provisional=False,
            ),
        ]

        items = await parser._parse_items(sample_global_outline)

        assert len(items) == 1
        assert items[0].category == "object"
        assert items[0].name == "Bloodstained doll"


def test_parse_character_arcs(sample_global_outline):
    """Test parsing of character arcs."""
    parser = GlobalOutlineParser()
    character_arcs = parser._parse_character_arcs(sample_global_outline)
    
    assert len(character_arcs) == 1
    assert "Hero" in character_arcs
    assert character_arcs["Hero"]["arc_start"] == "Naive"
    assert character_arcs["Hero"]["arc_end"] == "Wise"
    assert len(character_arcs["Hero"]["arc_key_moments"]) == 2


def test_parse_character_arcs_empty():
    """Test parsing of character arcs when none exist."""
    parser = GlobalOutlineParser()
    empty_data = {}
    
    character_arcs = parser._parse_character_arcs(empty_data)
    
    assert len(character_arcs) == 0


def test_generate_event_id():
    """Test event ID generation."""
    parser = GlobalOutlineParser()
    
    id1 = parser._generate_event_id("Test Event", 1)
    id2 = parser._generate_event_id("Test Event", 1)
    
    assert id1 == id2  # Should be deterministic
    assert id1.startswith("event_")
    assert len(id1) > 10  # Should have some length


@pytest.mark.asyncio
async def test_create_major_plot_point_nodes_success():
    """Test successful creation of MajorPlotPoint nodes."""
    parser = GlobalOutlineParser()
    
    # Mock plot points
    plot_points = [
        MajorPlotPoint(
            id="test_id_1",
            name="Test Event 1",
            description="Test description 1",
            sequence_order=1,
        )
    ]
    
    # Mock the database manager
    with patch("core.parsers.global_outline_parser.neo4j_manager.execute_write_query") as mock_query:
        mock_query.return_value = None
        
        result = await parser.create_major_plot_point_nodes(plot_points)
        
        assert result is True
        assert mock_query.call_count == 1


@pytest.mark.asyncio
async def test_create_major_plot_point_nodes_failure():
    """Test error handling in MajorPlotPoint node creation."""
    parser = GlobalOutlineParser()
    
    plot_points = [
        MajorPlotPoint(
            id="test_id_1",
            name="Test Event 1",
            description="Test description 1",
            sequence_order=1,
        )
    ]
    
    # Mock the database manager to raise an exception
    with patch("core.parsers.global_outline_parser.neo4j_manager.execute_write_query") as mock_query:
        mock_query.side_effect = Exception("Database error")
        
        result = await parser.create_major_plot_point_nodes(plot_points)
        
        assert result is False


@pytest.mark.asyncio
async def test_enrich_character_arcs_success():
    """Test successful enrichment of character arcs."""
    parser = GlobalOutlineParser()
    
    # Mock character arcs
    character_arcs = {
        "Hero": {
            "arc_start": "Naive",
            "arc_end": "Wise",
            "arc_key_moments": ["First failure", "Learning from mentor"],
        }
    }
    
    # Mock the database manager
    with patch("core.parsers.global_outline_parser.neo4j_manager.execute_write_query") as mock_query:
        mock_query.return_value = None
        
        result = await parser.enrich_character_arcs(character_arcs)
        
        assert result is True
        assert mock_query.call_count == 1


@pytest.mark.asyncio
async def test_enrich_character_arcs_failure():
    """Test error handling in character arc enrichment."""
    parser = GlobalOutlineParser()
    
    character_arcs = {
        "Hero": {
            "arc_start": "Naive",
            "arc_end": "Wise",
            "arc_key_moments": ["First failure", "Learning from mentor"],
        }
    }
    
    # Mock the database manager to raise an exception
    with patch("core.parsers.global_outline_parser.neo4j_manager.execute_write_query") as mock_query:
        mock_query.side_effect = Exception("Database error")
        
        result = await parser.enrich_character_arcs(character_arcs)
        
        assert result is False


@pytest.mark.asyncio
async def test_parse_and_persist_success(mock_global_outline_file):
    """Test successful parse and persist operation."""
    parser = GlobalOutlineParser(global_outline_path=mock_global_outline_file)

    # Mock all the database operations and LLM extraction
    with patch("core.parsers.global_outline_parser.neo4j_manager.execute_write_query") as mock_query, \
         patch.object(parser, '_extract_world_items_from_outline') as mock_extract:

        mock_query.return_value = None
        mock_extract.return_value = []

        result = await parser.parse_and_persist()

        assert result[0] is True
        assert "Successfully parsed and persisted" in result[1]


@pytest.mark.asyncio
async def test_parse_and_persist_failure(mock_global_outline_file):
    """Test error handling in parse and persist operation."""
    parser = GlobalOutlineParser(global_outline_path=mock_global_outline_file)

    # Mock the database manager to raise an exception and LLM extraction
    with patch("core.parsers.global_outline_parser.neo4j_manager.execute_write_query") as mock_query, \
         patch.object(parser, '_extract_world_items_from_outline') as mock_extract:

        mock_query.side_effect = Exception("Database error")
        mock_extract.return_value = []

        result = await parser.parse_and_persist()

        assert result[0] is False
        assert "Failed" in result[1] or "Error" in result[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
