# tests/test_chapter_outline_parser_extended.py
"""Extended tests for ChapterOutlineParser implementation.

This module provides extended tests for the ChapterOutlineParser class
to verify that it correctly implements all schema requirements from
docs/schema-design.md, including:

1. Character lookup functionality
2. ActKeyEvent lookup functionality  
3. FEATURES_CHARACTER relationship creation
4. INVOLVES relationship creation
5. PART_OF relationship creation (SceneEvent → ActKeyEvent)
"""

import json
import tempfile
import os
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from core.parsers import ChapterOutlineParser
from models.kg_models import CharacterProfile


@pytest.fixture
def sample_chapter_outline_with_characters():
    """Create sample chapter outline data with character references for testing."""
    return {
        "chapter_number": 1,
        "act_number": 1,
        "title": "The Beginning",
        "summary": "The hero's journey begins",
        "scenes": [
            {
                "scene_index": 0,
                "title": "Opening Scene",
                "pov_character": "Hero",
                "setting": "A dark forest",
                "plot_point": "The hero enters the forest",
                "conflict": "Danger lurks",
                "outcome": "The hero survives",
                "beats": ["Hero hears rustling", "Shadow appears", "Hero draws weapon"],
                "events": [
                    {
                        "name": "Forest Encounter",
                        "description": "The hero encounters danger",
                        "conflict": "Tension builds",
                        "outcome": "Hero escapes",
                        "pov_character": "Hero"
                    }
                ],
                "location": {
                    "name": "Dark Forest",
                    "description": "A dense, dark forest"
                }
            }
        ]
    }


@pytest.mark.asyncio
async def test_chapter_outline_parser_character_lookup():
    """Test that _get_character_by_name correctly queries characters."""
    parser = ChapterOutlineParser()
    
    # Mock the character query function
    with patch('data_access.character_queries.get_character_profile_by_name', new_callable=AsyncMock) as mock_query:
        mock_query.return_value = CharacterProfile(
            name="Test Character",
            personality_description="Test description",
            traits=["brave", "strong"],
            status="Active"
        )
        
        # Test the character lookup
        result = await parser._get_character_by_name("Test Character")
        
        # Verify the query was called
        assert mock_query.called
        assert mock_query.call_args[0][0] == "Test Character"
        
        # Verify the result
        assert result is not None
        assert result.name == "Test Character"


@pytest.mark.asyncio
async def test_chapter_outline_parser_character_lookup_not_found():
    """Test that _get_character_by_name returns None when character not found."""
    parser = ChapterOutlineParser()
    
    # Mock the character query function to return None
    with patch('data_access.character_queries.get_character_profile_by_name', new_callable=AsyncMock) as mock_query:
        mock_query.return_value = None
        
        # Test the character lookup
        result = await parser._get_character_by_name("NonExistent Character")
        
        # Verify the result is None
        assert result is None


@pytest.mark.asyncio
async def test_chapter_outline_parser_act_key_event_lookup():
    """Test that _get_act_key_event correctly queries ActKeyEvents."""
    parser = ChapterOutlineParser()
    
    # Mock the Neo4j query
    with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
        mock_query.return_value = [
            {
                "e": {
                    "id": "event_123",
                    "name": "Test Event",
                    "description": "Test description",
                    "event_type": "ActKeyEvent",
                    "act_number": 1,
                    "sequence_in_act": 1
                }
            }
        ]
        
        # Test the ActKeyEvent lookup
        result = await parser._get_act_key_event(1, 1)
        
        # Verify the query was called
        assert mock_query.called
        
        # Verify the result
        assert result is not None
        assert result["e"]["id"] == "event_123"


@pytest.mark.asyncio
async def test_chapter_outline_parser_act_key_event_lookup_not_found():
    """Test that _get_act_key_event returns None when event not found."""
    parser = ChapterOutlineParser()
    
    # Mock the Neo4j query to return empty results
    with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
        mock_query.return_value = []
        
        # Test the ActKeyEvent lookup
        result = await parser._get_act_key_event(999, 999)
        
        # Verify the result is None
        assert result is None


@pytest.mark.asyncio
async def test_chapter_outline_parser_features_character_relationship():
    """Test that FEATURES_CHARACTER relationships are created correctly."""
    # Create a temporary chapter outline file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "chapter_number": 1,
            "act_number": 1,
            "title": "Test Chapter",
            "summary": "Test summary",
            "scenes": [
                {
                    "scene_index": 0,
                    "title": "Test Scene",
                    "pov_character": "TestCharacter",
                    "setting": "Test setting",
                    "plot_point": "Test plot point",
                    "conflict": "Test conflict",
                    "outcome": "Test outcome",
                    "beats": ["Test beat 1", "Test beat 2"],
                    "events": [],
                    "location": {
                        "name": "Test Location",
                        "description": "Test description"
                    }
                }
            ]
        }, f)
        temp_file = f.name
    
    try:
        parser = ChapterOutlineParser(chapter_outline_path=temp_file, chapter_number=1)
        
        # Parse the chapter outline
        chapter_outline_data = await parser.parse_chapter_outline()
        
        # Parse the chapter
        chapter = parser._parse_chapter(chapter_outline_data)
        
        # Parse the scenes
        scenes = parser._parse_scenes(chapter_outline_data)
        
        # Parse the events
        events = parser._parse_scene_events(chapter_outline_data)
        
        # Parse the locations
        locations = parser._parse_locations(chapter_outline_data)
        
        # Mock the character lookup to return a character
        with patch('data_access.character_queries.get_character_profile_by_name', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = CharacterProfile(
                name="TestCharacter",
                personality_description="Test description",
                traits=["brave"],
                status="Active"
            )
            
            # Mock the Neo4j write query
            with patch('core.db_manager.neo4j_manager.execute_write_query', new_callable=AsyncMock) as mock_write:
                # Call create_relationships
                await parser.create_relationships([chapter], scenes, events, locations)
                
                # Verify that the FEATURES_CHARACTER relationship query was called
                assert mock_write.called
                
                # Check that the relationship creation query was included
                for call_args in mock_write.call_args_list:
                    query = call_args[0][0]
                    if "FEATURES_CHARACTER" in query:
                        assert "is_pov = true" in query
                        assert "MATCH (c:Character {name: $character_name})" in query
                        break
                else:
                    pytest.fail("FEATURES_CHARACTER relationship query not found")
    
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


@pytest.mark.asyncio
async def test_chapter_outline_parser_involves_relationship():
    """Test that INVOLVES relationships are created correctly."""
    # Create a temporary chapter outline file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "chapter_number": 1,
            "act_number": 1,
            "title": "Test Chapter",
            "summary": "Test summary",
            "scenes": [
                {
                    "scene_index": 0,
                    "title": "Test Scene 1",
                    "pov_character": "TestCharacter",
                    "setting": "Test setting 1",
                    "plot_point": "Test plot point 1",
                    "conflict": "Test conflict 1",
                    "outcome": "Test outcome 1",
                    "beats": ["Test beat 1", "Test beat 2"],
                    "events": [
                        {
                            "name": "Test Event 1",
                            "description": "Test description 1",
                            "conflict": "Test conflict 1",
                            "outcome": "Test outcome 1",
                            "pov_character": "TestCharacter"
                        }
                    ],
                    "location": {
                        "name": "Test Location 1",
                        "description": "Test description 1"
                    }
                },
                {
                    "scene_index": 1,
                    "title": "Test Scene 2",
                    "pov_character": "TestCharacter",
                    "setting": "Test setting 2",
                    "plot_point": "Test plot point 2",
                    "conflict": "Test conflict 2",
                    "outcome": "Test outcome 2",
                    "beats": ["Test beat 3", "Test beat 4"],
                    "events": [
                        {
                            "name": "Test Event 2",
                            "description": "Test description 2",
                            "conflict": "Test conflict 2",
                            "outcome": "Test outcome 2",
                            "pov_character": "TestCharacter"
                        }
                    ],
                    "location": {
                        "name": "Test Location 2",
                        "description": "Test description 2"
                    }
                }
            ]
        }, f)
        temp_file = f.name
    
    try:
        parser = ChapterOutlineParser(chapter_outline_path=temp_file, chapter_number=1)
        
        # Parse the chapter outline
        chapter_outline_data = await parser.parse_chapter_outline()
        
        # Parse the chapter
        chapter = parser._parse_chapter(chapter_outline_data)
        
        # Parse the scenes
        scenes = parser._parse_scenes(chapter_outline_data)
        
        # Parse the events
        events = parser._parse_scene_events(chapter_outline_data)
        
        # Parse the locations
        locations = parser._parse_locations(chapter_outline_data)
        
        # Mock the character lookup to return a character
        with patch('data_access.character_queries.get_character_profile_by_name', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = CharacterProfile(
                name="TestCharacter",
                personality_description="Test description",
                traits=["brave"],
                status="Active"
            )
            
            # Mock the Neo4j write query
            with patch('core.db_manager.neo4j_manager.execute_write_query', new_callable=AsyncMock) as mock_write:
                # Call create_relationships
                await parser.create_relationships([chapter], scenes, events, locations)
                
                # Verify that the INVOLVES relationship query was called
                assert mock_write.called
                
                # Check that the relationship creation query was included
                for call_args in mock_write.call_args_list:
                    query = call_args[0][0]
                    if "INVOLVES" in query:
                        assert "role = \"protagonist\"" in query
                        assert "MATCH (c:Character {name: $character_name})" in query
                        break
                else:
                    pytest.fail("INVOLVES relationship query not found")
    
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


@pytest.mark.asyncio
async def test_chapter_outline_parser_part_of_relationship():
    """Test that PART_OF relationships (SceneEvent → ActKeyEvent) are created correctly."""
    # Create a temporary chapter outline file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "chapter_number": 1,
            "act_number": 1,
            "title": "Test Chapter",
            "summary": "Test summary",
            "scenes": [
                {
                    "scene_index": 0,
                    "title": "Test Scene",
                    "pov_character": "TestCharacter",
                    "setting": "Test setting",
                    "plot_point": "Test plot point",
                    "conflict": "Test conflict",
                    "outcome": "Test outcome",
                    "beats": ["Test beat 1", "Test beat 2"],
                    "events": [
                        {
                            "name": "Test Event",
                            "description": "Test description",
                            "conflict": "Test conflict",
                            "outcome": "Test outcome",
                            "pov_character": "TestCharacter"
                        }
                    ],
                    "location": {
                        "name": "Test Location",
                        "description": "Test description"
                    }
                }
            ]
        }, f)
        temp_file = f.name
    
    try:
        parser = ChapterOutlineParser(chapter_outline_path=temp_file, chapter_number=1)
        
        # Parse the chapter outline
        chapter_outline_data = await parser.parse_chapter_outline()
        
        # Parse the chapter
        chapter = parser._parse_chapter(chapter_outline_data)
        
        # Parse the scenes
        scenes = parser._parse_scenes(chapter_outline_data)
        
        # Parse the events
        events = parser._parse_scene_events(chapter_outline_data)
        
        # Parse the locations
        locations = parser._parse_locations(chapter_outline_data)
        
        # Mock the character lookup to return a character
        with patch('data_access.character_queries.get_character_profile_by_name', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = CharacterProfile(
                name="TestCharacter",
                personality_description="Test description",
                traits=["brave"],
                status="Active"
            )
            
            # Mock the ActKeyEvent lookup to return an event
            with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_read:
                mock_read.return_value = [
                    {
                        "e": {
                            "id": "act_key_event_1",
                            "name": "Test Act Key Event",
                            "description": "Test description",
                            "event_type": "ActKeyEvent",
                            "act_number": 1,
                            "sequence_in_act": 1
                        }
                    }
                ]
                
                # Mock the Neo4j write query
                with patch('core.db_manager.neo4j_manager.execute_write_query', new_callable=AsyncMock) as mock_write:
                    # Call create_relationships
                    await parser.create_relationships([chapter], scenes, events, locations)
                    
                    # Verify that the PART_OF relationship query was called
                    assert mock_write.called
                    
                    # Check that the relationship creation query was included
                    for call_args in mock_write.call_args_list:
                        query = call_args[0][0]
                        if "PART_OF" in query and "ake:Event" in query:
                            assert "MATCH (ake:Event {id: $ake_id})" in query
                            break
                    else:
                        pytest.fail("PART_OF relationship query not found")
    
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


@pytest.mark.asyncio
async def test_chapter_outline_parser_all_relationships_created():
    """Test that all required relationships are created in one pass."""
    # Create a temporary chapter outline file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "chapter_number": 1,
            "act_number": 1,
            "title": "Test Chapter",
            "summary": "Test summary",
            "scenes": [
                {
                    "scene_index": 0,
                    "title": "Test Scene 1",
                    "pov_character": "TestCharacter",
                    "setting": "Test setting 1",
                    "plot_point": "Test plot point 1",
                    "conflict": "Test conflict 1",
                    "outcome": "Test outcome 1",
                    "beats": ["Test beat 1", "Test beat 2"],
                    "events": [
                        {
                            "name": "Test Event 1",
                            "description": "Test description 1",
                            "conflict": "Test conflict 1",
                            "outcome": "Test outcome 1",
                            "pov_character": "TestCharacter"
                        }
                    ],
                    "location": {
                        "name": "Test Location 1",
                        "description": "Test description 1"
                    }
                },
                {
                    "scene_index": 1,
                    "title": "Test Scene 2",
                    "pov_character": "TestCharacter",
                    "setting": "Test setting 2",
                    "plot_point": "Test plot point 2",
                    "conflict": "Test conflict 2",
                    "outcome": "Test outcome 2",
                    "beats": ["Test beat 3", "Test beat 4"],
                    "events": [
                        {
                            "name": "Test Event 2",
                            "description": "Test description 2",
                            "conflict": "Test conflict 2",
                            "outcome": "Test outcome 2",
                            "pov_character": "TestCharacter"
                        }
                    ],
                    "location": {
                        "name": "Test Location 2",
                        "description": "Test description 2"
                    }
                }
            ]
        }, f)
        temp_file = f.name
    
    try:
        parser = ChapterOutlineParser(chapter_outline_path=temp_file, chapter_number=1)
        
        # Parse the chapter outline
        chapter_outline_data = await parser.parse_chapter_outline()
        
        # Parse the chapter
        chapter = parser._parse_chapter(chapter_outline_data)
        
        # Parse the scenes
        scenes = parser._parse_scenes(chapter_outline_data)
        
        # Parse the events
        events = parser._parse_scene_events(chapter_outline_data)
        
        # Parse the locations
        locations = parser._parse_locations(chapter_outline_data)
        
        # Mock the character lookup to return a character
        with patch('data_access.character_queries.get_character_profile_by_name', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = CharacterProfile(
                name="TestCharacter",
                personality_description="Test description",
                traits=["brave"],
                status="Active"
            )
            
            # Mock the ActKeyEvent lookup to return an event
            with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_read:
                mock_read.return_value = [
                    {
                        "e": {
                            "id": "act_key_event_1",
                            "name": "Test Act Key Event",
                            "description": "Test description",
                            "event_type": "ActKeyEvent",
                            "act_number": 1,
                            "sequence_in_act": 1
                        }
                    }
                ]
                
                # Mock the Neo4j write query
                with patch('core.db_manager.neo4j_manager.execute_write_query', new_callable=AsyncMock) as mock_write:
                    # Call create_relationships
                    await parser.create_relationships([chapter], scenes, events, locations)
                    
                    # Verify that the write query was called
                    assert mock_write.called
                    
                    # Collect all queries
                    all_queries = [call_args[0][0] for call_args in mock_write.call_args_list]
                    
                    # Verify all required relationship types are present
                    required_relationships = [
                        "PART_OF",
                        "FOLLOWS",
                        "FEATURES_CHARACTER",
                        "INVOLVES"
                    ]
                    
                    for rel_type in required_relationships:
                        found = False
                        for query in all_queries:
                            if rel_type in query:
                                found = True
                                break
                        assert found, f"Relationship type {rel_type} not found in queries"
    
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
