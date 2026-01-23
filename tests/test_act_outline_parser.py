# tests/test_act_outline_parser.py
"""Test the ActOutlineParser implementation."""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.parsers.act_outline_parser import ActOutlineParser
from models.kg_models import ActKeyEvent


@pytest.fixture
def sample_act_outline():
    """Sample valid act outline JSON."""
    return {
        "format_version": 2,
        "acts": [
            {
                "act_number": 1,
                "total_acts": 3,
                "act_role": "Setup/Introduction",
                "chapters_in_act": 7,
                "sections": {
                    "act_summary": "Introduction of hero and world",
                    "opening_situation": "Hero lives peaceful life",
                    "key_events": [
                        {
                            "sequence": 1,
                            "event": "Hero meets mentor",
                            "cause": "Hero seeks guidance",
                            "effect": "Hero learns about threat"
                        },
                        {
                            "sequence": 2,
                            "event": "Hero discovers secret",
                            "cause": "Mentor reveals truth",
                            "effect": "Hero is motivated to act"
                        },
                        {
                            "sequence": 3,
                            "event": "Hero leaves home",
                            "cause": "Hero feels responsibility",
                            "effect": "Hero begins journey"
                        }
                    ],
                    "character_development": "Hero learns about world",
                    "stakes_and_tension": "First signs of danger appear",
                    "act_ending_turn": "Hero commits to quest",
                    "thematic_thread": "Theme of duty appears",
                    "pacing_notes": "Slow build to action",
                    "locations": [
                        {
                            "name": "Hero's Village",
                            "description": "Small village in the mountains"
                        },
                        {
                            "name": "Mentor's Tower",
                            "description": "Tall tower with ancient knowledge"
                        }
                    ]
                }
            },
            {
                "act_number": 2,
                "total_acts": 3,
                "act_role": "Confrontation/Rising Action",
                "chapters_in_act": 7,
                "sections": {
                    "act_summary": "Hero faces challenges",
                    "opening_situation": "Hero arrives at first challenge",
                    "key_events": [
                        {
                            "sequence": 1,
                            "event": "First battle",
                            "cause": "Hero is attacked",
                            "effect": "Hero learns combat skills"
                        },
                        {
                            "sequence": 2,
                            "event": "Hero finds allies",
                            "cause": "Hero shares story",
                            "effect": "Hero gains support"
                        }
                    ],
                    "character_development": "Hero grows stronger",
                    "stakes_and_tension": "Pressure increases",
                    "act_ending_turn": "Hero faces major obstacle",
                    "thematic_thread": "Theme of sacrifice appears",
                    "pacing_notes": "Faster pace with more action",
                    "locations": [
                        {
                            "name": "Battlefield",
                            "description": "Large open field for combat"
                        }
                    ]
                }
            }
        ]
    }


@pytest.fixture
def mock_act_outline_file(sample_act_outline):
    """Create a temporary act outline file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_act_outline, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_parse_act_outline_success(mock_act_outline_file):
    """Test successful parsing of act outline."""
    parser = ActOutlineParser(act_outline_path=mock_act_outline_file)
    
    result = await parser.parse_act_outline()
    
    assert result is not None
    assert "acts" in result
    assert len(result["acts"]) == 2


@pytest.mark.asyncio
async def test_parse_act_outline_file_not_found():
    """Test error handling when file is not found."""
    parser = ActOutlineParser(act_outline_path="/nonexistent/path.json")
    
    with pytest.raises(ValueError, match="Act outline file not found"):
        await parser.parse_act_outline()


@pytest.mark.asyncio
async def test_parse_act_outline_invalid_json():
    """Test error handling when JSON is invalid."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("invalid json {{{")
        temp_path = f.name
    
    try:
        parser = ActOutlineParser(act_outline_path=temp_path)
        
        with pytest.raises(ValueError, match="Invalid JSON in act outline file"):
            await parser.parse_act_outline()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_parse_act_key_events(sample_act_outline):
    """Test parsing of act key events."""
    parser = ActOutlineParser()
    
    act_events = parser._parse_act_key_events(sample_act_outline)
    
    assert len(act_events) == 5  # 3 from act 1, 2 from act 2
    assert all(isinstance(event, ActKeyEvent) for event in act_events)
    
    # Check first event
    first_event = act_events[0]
    assert first_event.name == "Hero meets mentor"
    assert first_event.act_number == 1
    assert first_event.sequence_in_act == 1
    assert first_event.cause == "Hero seeks guidance"
    assert first_event.effect == "Hero learns about threat"
    assert first_event.event_type == "ActKeyEvent"
    assert first_event.created_chapter == 0
    assert first_event.is_provisional == False


@pytest.mark.asyncio
async def test_parse_location_enrichment(sample_act_outline):
    """Test parsing of location name enrichment."""
    parser = ActOutlineParser()
    
    location_names = parser._parse_location_enrichment(sample_act_outline)
    
    assert len(location_names) == 3  # 2 from act 1, 1 from act 2
    assert "Small village in the mountains" in location_names
    assert "Tall tower with ancient knowledge" in location_names
    assert "Large open field for combat" in location_names
    
    # Check that descriptions map to correct names
    assert location_names["Small village in the mountains"] == "Hero's Village"
    assert location_names["Tall tower with ancient knowledge"] == "Mentor's Tower"
    assert location_names["Large open field for combat"] == "Battlefield"


@pytest.mark.asyncio
async def test_generate_event_id():
    """Test event ID generation."""
    parser = ActOutlineParser()
    
    # Test that same inputs produce same IDs
    id1 = parser._generate_event_id("Test Event", 1, 1)
    id2 = parser._generate_event_id("Test Event", 1, 1)
    assert id1 == id2
    
    # Test that different inputs produce different IDs
    id3 = parser._generate_event_id("Different Event", 1, 1)
    assert id1 != id3
    
    # Test that IDs start with "event_"
    assert id1.startswith("event_")
    assert id2.startswith("event_")
    assert id3.startswith("event_")


@pytest.mark.asyncio
async def test_parse_and_persist_integration(mock_act_outline_file):
    """Test full integration of parse_and_persist method."""
    parser = ActOutlineParser(act_outline_path=mock_act_outline_file)
    
    # Mock the database operations
    with patch.object(parser, 'create_act_key_event_nodes', new_callable=AsyncMock) as mock_create_events, \
         patch.object(parser, 'enrich_location_names', new_callable=AsyncMock) as mock_enrich_locations, \
         patch.object(parser, 'create_event_relationships', new_callable=AsyncMock) as mock_create_relationships:
        
        # Set up mocks to return success
        mock_create_events.return_value = True
        mock_enrich_locations.return_value = True
        mock_create_relationships.return_value = True
        
        # Call the method
        success, message = await parser.parse_and_persist()
        
        # Verify success
        assert success is True
        assert "Successfully parsed and persisted" in message
        
        # Verify that all methods were called
        mock_create_events.assert_called_once()
        mock_enrich_locations.assert_called_once()
        mock_create_relationships.assert_called_once()


@pytest.mark.asyncio
async def test_parse_and_persist_failure(mock_act_outline_file):
    """Test error handling in parse_and_persist method."""
    parser = ActOutlineParser(act_outline_path=mock_act_outline_file)
    
    # Mock the database operations to fail
    with patch.object(parser, 'create_act_key_event_nodes', new_callable=AsyncMock) as mock_create_events:
        mock_create_events.return_value = False
        
        # Call the method
        success, message = await parser.parse_and_persist()
        
        # Verify failure
        assert success is False
        assert "Failed to create ActKeyEvent nodes" in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
