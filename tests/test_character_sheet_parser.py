# tests/test_character_sheet_parser.py
"""Tests for the CharacterSheetParser class."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.parsers.character_sheet_parser import CharacterSheetParser
from models.kg_models import CharacterProfile


@pytest.fixture
def sample_character_sheets():
    """Create a sample character sheets JSON file for testing."""
    sample_data = {
        "Alice": {
            "name": "Alice",
            "description": "A brave heroine",
            "traits": ["brave", "loyal", "cunning"],
            "status": "Active",
            "relationships": {
                "Bob": {
                    "type": "FRIEND_OF",
                    "description": "Childhood friend"
                }
            }
        },
        "Bob": {
            "name": "Bob",
            "description": "A loyal friend",
            "traits": ["loyal", "trustworthy"],
            "status": "Active",
            "relationships": {
                "Alice": {
                    "type": "FRIEND_OF",
                    "description": "Childhood friend"
                }
            }
        }
    }
    
    # Write to a temporary file
    temp_file = "/tmp/test_character_sheets.json"
    with open(temp_file, 'w') as f:
        json.dump(sample_data, f)
    
    yield temp_file
    
    # Clean up
    if os.path.exists(temp_file):
        os.remove(temp_file)


@pytest.mark.asyncio
class TestCharacterSheetParser:
    """Test the CharacterSheetParser class."""
    
    async def test_parse_character_sheets_success(self, sample_character_sheets):
        """Test successful parsing of character sheets."""
        parser = CharacterSheetParser(character_sheets_path=sample_character_sheets)
        
        characters = await parser.parse_character_sheets()
        
        assert len(characters) == 2
        assert isinstance(characters[0], CharacterProfile)
        assert characters[0].name == "Alice"
        assert "brave" in characters[0].traits
        
    async def test_parse_character_sheets_file_not_found(self):
        """Test handling of missing character sheets file."""
        parser = CharacterSheetParser(character_sheets_path="/nonexistent/file.json")
        
        with pytest.raises(ValueError, match="Character sheets file not found"):
            await parser.parse_character_sheets()
    
    async def test_parse_character_sheets_invalid_json(self, tmp_path):
        """Test handling of invalid JSON in character sheets file."""
        invalid_json_file = tmp_path / "invalid.json"
        invalid_json_file.write_text("{invalid json}")
        
        parser = CharacterSheetParser(character_sheets_path=str(invalid_json_file))
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            await parser.parse_character_sheets()
    
    async def test_parse_character_sheet_missing_fields(self):
        """Test parsing of character sheet with missing required fields."""
        parser = CharacterSheetParser()
        
        # Test missing description
        with pytest.raises(ValueError, match="Missing 'description'"):
            parser._parse_character_sheet("Test", {"name": "Test"})
    
    async def test_parse_character_sheet_invalid_status(self):
        """Test parsing of character sheet with invalid status."""
        parser = CharacterSheetParser()
        
        # Test invalid status (should default to Active)
        character_data = {
            "name": "Test",
            "description": "Test character",
            "traits": ["test"],
            "status": "InvalidStatus"
        }
        
        with patch('core.parsers.character_sheet_parser.logger') as mock_logger:
            result = parser._parse_character_sheet("Test", character_data)
            
            # Check that warning was logged
            assert mock_logger.warning.called
            # Check that status was set to Active
            assert result.status == "Active"
    
    async def test_parse_relationships(self, sample_character_sheets):
        """Test parsing of relationships from character sheets."""
        parser = CharacterSheetParser(character_sheets_path=sample_character_sheets)
        
        characters = await parser.parse_character_sheets()
        relationships = await parser.parse_relationships(characters)
        
        assert len(relationships) == 2
        assert "Alice" in relationships
        assert "Bob" in relationships
        
        # Check that relationships are bidirectional
        assert "Bob" in relationships["Alice"]
        assert "Alice" in relationships["Bob"]
    
    async def test_create_character_nodes(self, sample_character_sheets):
        """Test creation of character nodes in Neo4j."""
        parser = CharacterSheetParser(character_sheets_path=sample_character_sheets)
        
        with patch('core.parsers.character_sheet_parser.sync_characters') as mock_sync:
            mock_sync.return_value = True
            
            characters = await parser.parse_character_sheets()
            result = await parser.create_character_nodes(characters)
            
            assert result is True
            mock_sync.assert_called_once()
    
    async def test_create_character_nodes_failure(self, sample_character_sheets):
        """Test handling of character node creation failure."""
        parser = CharacterSheetParser(character_sheets_path=sample_character_sheets)
        
        with patch('core.parsers.character_sheet_parser.sync_characters') as mock_sync:
            mock_sync.return_value = False
            
            characters = await parser.parse_character_sheets()
            result = await parser.create_character_nodes(characters)
            
            assert result is False
    
    async def test_create_relationships(self, sample_character_sheets):
        """Test creation of relationships in Neo4j."""
        parser = CharacterSheetParser(character_sheets_path=sample_character_sheets)
        
        with patch('core.parsers.character_sheet_parser.neo4j_manager') as mock_manager:
            mock_manager.execute_write_query = AsyncMock(return_value=None)
            
            characters = await parser.parse_character_sheets()
            relationships = await parser.parse_relationships(characters)
            result = await parser.create_relationships(relationships)
            
            assert result is True
            # Check that queries were executed
            assert mock_manager.execute_write_query.call_count == 2
    
    async def test_parse_and_persist_success(self, sample_character_sheets):
        """Test successful parse and persist operation."""
        parser = CharacterSheetParser(character_sheets_path=sample_character_sheets)
        
        with patch('core.parsers.character_sheet_parser.sync_characters') as mock_sync:
            with patch('core.parsers.character_sheet_parser.neo4j_manager') as mock_manager:
                mock_sync.return_value = True
                mock_manager.execute_write_query = AsyncMock(return_value=None)
                
                result, message = await parser.parse_and_persist()
                
                assert result is True
                assert "Successfully parsed and persisted" in message
    
    async def test_parse_and_persist_failure(self, sample_character_sheets):
        """Test handling of parse and persist failure."""
        parser = CharacterSheetParser(character_sheets_path=sample_character_sheets)
        
        with patch('core.parsers.character_sheet_parser.sync_characters') as mock_sync:
            with patch('core.parsers.character_sheet_parser.neo4j_manager') as mock_manager:
                mock_sync.return_value = False
                mock_manager.execute_write_query = AsyncMock(return_value=None)
                
                result, message = await parser.parse_and_persist()
                
                assert result is False
                assert "Failed to create character nodes" in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
