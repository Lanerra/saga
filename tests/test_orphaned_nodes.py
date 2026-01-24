# tests/test_orphaned_nodes.py
"""Test for orphaned node detection in the knowledge graph.

This test file covers:
1. Detection of orphaned Character nodes (no relationships)
2. Detection of orphaned Event nodes (no relationships)
3. Detection of orphaned Location nodes (no relationships)
4. Detection of orphaned Item nodes (no relationships)
5. Detection of orphaned Scene nodes (no relationships)
6. Detection of orphaned Chapter nodes (no relationships)

Based on: docs/schema-design.md - Stage 5: Testing
"""

import pytest
from unittest.mock import MagicMock, patch
from core.db_manager import neo4j_manager
from models.kg_models import CharacterProfile, MajorPlotPoint, ActKeyEvent, SceneEvent, Location, WorldItem, Scene, Chapter


@pytest.mark.asyncio
class TestOrphanedNodeDetection:
    """Test orphaned node detection in the knowledge graph."""

    async def test_detect_orphaned_character_nodes(self):
        """Test detection of orphaned Character nodes."""
        # Mock the database manager
        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            # Mock query to return orphaned characters
            mock_query.return_value = [
                {"id": "char_001", "name": "Alice", "relationships": []},
                {"id": "char_002", "name": "Bob", "relationships": []},
            ]
            
            # Call the detection function
            result = await neo4j_manager.detect_orphaned_characters()
            
            # Verify results
            assert len(result) == 2
            assert result[0]["name"] == "Alice"
            assert result[1]["name"] == "Bob"

    async def test_detect_orphaned_event_nodes(self):
        """Test detection of orphaned Event nodes."""
        # Mock the database manager
        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            # Mock query to return orphaned events
            mock_query.return_value = [
                {"id": "event_001", "name": "Inciting Incident", "relationships": []},
                {"id": "event_002", "name": "Midpoint", "relationships": []},
            ]
            
            # Call the detection function
            result = await neo4j_manager.detect_orphaned_events()
            
            # Verify results
            assert len(result) == 2
            assert result[0]["name"] == "Inciting Incident"
            assert result[1]["name"] == "Midpoint"

    async def test_detect_orphaned_location_nodes(self):
        """Test detection of orphaned Location nodes."""
        # Mock the database manager
        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            # Mock query to return orphaned locations
            mock_query.return_value = [
                {"id": "loc_001", "name": "Castle", "relationships": []},
                {"id": "loc_002", "name": "Forest", "relationships": []},
            ]
            
            # Call the detection function
            result = await neo4j_manager.detect_orphaned_locations()
            
            # Verify results
            assert len(result) == 2
            assert result[0]["name"] == "Castle"
            assert result[1]["name"] == "Forest"

    async def test_detect_orphaned_item_nodes(self):
        """Test detection of orphaned Item nodes."""
        # Mock the database manager
        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            # Mock query to return orphaned items
            mock_query.return_value = [
                {"id": "item_001", "name": "Sword", "relationships": []},
                {"id": "item_002", "name": "Shield", "relationships": []},
            ]
            
            # Call the detection function
            result = await neo4j_manager.detect_orphaned_items()
            
            # Verify results
            assert len(result) == 2
            assert result[0]["name"] == "Sword"
            assert result[1]["name"] == "Shield"

    async def test_detect_orphaned_scene_nodes(self):
        """Test detection of orphaned Scene nodes."""
        # Mock the database manager
        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            # Mock query to return orphaned scenes
            mock_query.return_value = [
                {"id": "scene_001", "title": "Scene 1", "relationships": []},
                {"id": "scene_002", "title": "Scene 2", "relationships": []},
            ]
            
            # Call the detection function
            result = await neo4j_manager.detect_orphaned_scenes()
            
            # Verify results
            assert len(result) == 2
            assert result[0]["title"] == "Scene 1"
            assert result[1]["title"] == "Scene 2"

    async def test_detect_orphaned_chapter_nodes(self):
        """Test detection of orphaned Chapter nodes."""
        # Mock the database manager
        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            # Mock query to return orphaned chapters
            mock_query.return_value = [
                {"id": "chapter_001", "title": "Chapter 1", "relationships": []},
                {"id": "chapter_002", "title": "Chapter 2", "relationships": []},
            ]
            
            # Call the detection function
            result = await neo4j_manager.detect_orphaned_chapters()
            
            # Verify results
            assert len(result) == 2
            assert result[0]["title"] == "Chapter 1"
            assert result[1]["title"] == "Chapter 2"


@pytest.mark.asyncio
class TestOrphanedNodeCleanup:
    """Test cleanup of orphaned nodes."""

    async def test_cleanup_orphaned_character_nodes(self):
        """Test cleanup of orphaned Character nodes."""
        # Mock the database manager
        with patch("core.db_manager.neo4j_manager.execute_write_query") as mock_query:
            # Mock query to return orphaned characters
            mock_query.return_value = [
                {"id": "char_001", "name": "Alice", "relationships": []},
                {"id": "char_002", "name": "Bob", "relationships": []},
            ]
            
            # Call the cleanup function
            result = await neo4j_manager.cleanup_orphaned_characters()
            
            # Verify results
            assert result == 2

    async def test_cleanup_orphaned_event_nodes(self):
        """Test cleanup of orphaned Event nodes."""
        # Mock the database manager
        with patch("core.db_manager.neo4j_manager.execute_write_query") as mock_query:
            # Mock query to return orphaned events
            mock_query.return_value = [
                {"id": "event_001", "name": "Inciting Incident", "relationships": []},
                {"id": "event_002", "name": "Midpoint", "relationships": []},
            ]
            
            # Call the cleanup function
            result = await neo4j_manager.cleanup_orphaned_events()
            
            # Verify results
            assert result == 2

    async def test_cleanup_orphaned_location_nodes(self):
        """Test cleanup of orphaned Location nodes."""
        # Mock the database manager
        with patch("core.db_manager.neo4j_manager.execute_write_query") as mock_query:
            # Mock query to return orphaned locations
            mock_query.return_value = [
                {"id": "loc_001", "name": "Castle", "relationships": []},
                {"id": "loc_002", "name": "Forest", "relationships": []},
            ]
            
            # Call the cleanup function
            result = await neo4j_manager.cleanup_orphaned_locations()
            
            # Verify results
            assert result == 2

    async def test_cleanup_orphaned_item_nodes(self):
        """Test cleanup of orphaned Item nodes."""
        # Mock the database manager
        with patch("core.db_manager.neo4j_manager.execute_write_query") as mock_query:
            # Mock query to return orphaned items
            mock_query.return_value = [
                {"id": "item_001", "name": "Sword", "relationships": []},
                {"id": "item_002", "name": "Shield", "relationships": []},
            ]
            
            # Call the cleanup function
            result = await neo4j_manager.cleanup_orphaned_items()
            
            # Verify results
            assert result == 2

    async def test_cleanup_orphaned_scene_nodes(self):
        """Test cleanup of orphaned Scene nodes."""
        # Mock the database manager
        with patch("core.db_manager.neo4j_manager.execute_write_query") as mock_query:
            # Mock query to return orphaned scenes
            mock_query.return_value = [
                {"id": "scene_001", "title": "Scene 1", "relationships": []},
                {"id": "scene_002", "title": "Scene 2", "relationships": []},
            ]
            
            # Call the cleanup function
            result = await neo4j_manager.cleanup_orphaned_scenes()
            
            # Verify results
            assert result == 2

    async def test_cleanup_orphaned_chapter_nodes(self):
        """Test cleanup of orphaned Chapter nodes."""
        # Mock the database manager
        with patch("core.db_manager.neo4j_manager.execute_write_query") as mock_query:
            # Mock query to return orphaned chapters
            mock_query.return_value = [
                {"id": "chapter_001", "title": "Chapter 1", "relationships": []},
                {"id": "chapter_002", "title": "Chapter 2", "relationships": []},
            ]
            
            # Call the cleanup function
            result = await neo4j_manager.cleanup_orphaned_chapters()
            
            # Verify results
            assert result == 2


@pytest.mark.asyncio
class TestOrphanedNodeEdgeCases:
    """Test edge cases for orphaned node detection."""

    async def test_no_orphaned_nodes(self):
        """Test when no orphaned nodes are found."""
        # Mock the database manager
        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            # Mock query to return no orphaned nodes
            mock_query.return_value = []
            
            # Call the detection function
            result = await neo4j_manager.detect_orphaned_characters()
            
            # Verify results
            assert len(result) == 0

    async def test_all_nodes_have_relationships(self):
        """Test when all nodes have relationships."""
        # Mock the database manager
        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            # Mock query to return nodes with relationships
            mock_query.return_value = [
                {"id": "char_001", "name": "Alice", "relationships": ["FRIENDS_WITH"]},
                {"id": "char_002", "name": "Bob", "relationships": ["FRIENDS_WITH"]},
            ]
            
            # Call the detection function
            result = await neo4j_manager.detect_orphaned_characters()
            
            # Verify results - should return nodes with relationships
            assert len(result) == 2
            assert result[0]["id"] == "char_001"
            assert result[1]["id"] == "char_002"

    async def test_database_error(self):
        """Test when database error occurs."""
        # Mock the database manager
        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            # Mock query to raise an error
            mock_query.side_effect = Exception("Database error")
            
            # Call the detection function
            with pytest.raises(Exception) as excinfo:
                await neo4j_manager.detect_orphaned_characters()
            
            # Verify error message
            assert "Database error" in str(excinfo.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
