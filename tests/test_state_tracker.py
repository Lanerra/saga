# tests/test_state_tracker.py
"""Unit tests for the StateTracker component."""

import asyncio
from unittest.mock import patch

import pytest

# Test imports
from processing.state_tracker import StateTracker


class TestStateTracker:
    """Test suite for StateTracker functionality."""
    
    @pytest.fixture
    async def state_tracker(self):
        """Create a StateTracker instance for testing."""
        tracker = StateTracker()
        # Clear any existing state
        await tracker.clear()
        return tracker
    
    @pytest.fixture
    def mock_config_enabled(self):
        """Mock config with StateTracker enabled."""
        with patch('processing.state_tracker.config') as mock_config:
            mock_config.STATE_TRACKER_ENABLED = True
            mock_config.STATE_TRACKER_SIMILARITY_THRESHOLD = 0.85
            yield mock_config
    
    @pytest.fixture
    def mock_config_disabled(self):
        """Mock config with StateTracker disabled."""
        with patch('processing.state_tracker.config') as mock_config:
            mock_config.STATE_TRACKER_ENABLED = False
            mock_config.STATE_TRACKER_SIMILARITY_THRESHOLD = 0.85
            yield mock_config

    @pytest.mark.asyncio
    async def test_reserve_new_entity(self, state_tracker, mock_config_enabled):
        """Test reserving a new entity."""
        # Test successful reservation
        result = await state_tracker.reserve("TestCharacter", "character", "A brave protagonist")
        assert result is True
        
        # Verify entity is stored
        metadata = await state_tracker.check("TestCharacter")
        assert metadata is not None
        assert metadata["name"] == "TestCharacter"
        assert metadata["type"] == "character"
        assert metadata["description"] == "A brave protagonist"
        assert "timestamp" in metadata
    
    @pytest.mark.asyncio
    async def test_reserve_duplicate_entity(self, state_tracker, mock_config_enabled):
        """Test attempting to reserve an already reserved entity."""
        # Reserve first entity
        await state_tracker.reserve("TestCharacter", "character", "A brave protagonist")
        
        # Attempt to reserve same name
        result = await state_tracker.reserve("TestCharacter", "world_item", "A magic sword")
        assert result is False
        
        # Verify original entity is unchanged
        metadata = await state_tracker.check("TestCharacter")
        assert metadata["type"] == "character"
        assert metadata["description"] == "A brave protagonist"
    
    @pytest.mark.asyncio
    async def test_reserve_disabled(self, state_tracker, mock_config_disabled):
        """Test reservation when StateTracker is disabled."""
        result = await state_tracker.reserve("TestCharacter", "character", "A brave protagonist")
        assert result is True  # Always returns True when disabled
        
        # Check should return None when disabled
        metadata = await state_tracker.check("TestCharacter")
        assert metadata is None
    
    @pytest.mark.asyncio
    async def test_check_existing_entity(self, state_tracker, mock_config_enabled):
        """Test checking an existing entity."""
        # Reserve entity
        await state_tracker.reserve("TestWorld", "world_item", "A mystical place")
        
        # Check entity
        metadata = await state_tracker.check("TestWorld")
        assert metadata is not None
        assert metadata["name"] == "TestWorld"
        assert metadata["type"] == "world_item"
    
    @pytest.mark.asyncio
    async def test_check_nonexistent_entity(self, state_tracker, mock_config_enabled):
        """Test checking a non-existent entity."""
        metadata = await state_tracker.check("NonExistent")
        assert metadata is None
    
    @pytest.mark.asyncio
    async def test_release_entity(self, state_tracker, mock_config_enabled):
        """Test releasing a reserved entity."""
        # Reserve entity
        await state_tracker.reserve("TestCharacter", "character", "A brave protagonist")
        
        # Verify it exists
        metadata = await state_tracker.check("TestCharacter")
        assert metadata is not None
        
        # Release entity
        await state_tracker.release("TestCharacter")
        
        # Verify it's gone
        metadata = await state_tracker.check("TestCharacter")
        assert metadata is None
    
    @pytest.mark.asyncio
    async def test_release_nonexistent_entity(self, state_tracker, mock_config_enabled):
        """Test releasing a non-existent entity (should not error)."""
        # This should not raise an exception
        await state_tracker.release("NonExistent")
    
    @pytest.mark.asyncio
    async def test_get_all_entities(self, state_tracker, mock_config_enabled):
        """Test getting all tracked entities."""
        # Reserve multiple entities
        await state_tracker.reserve("Character1", "character", "First character")
        await state_tracker.reserve("Character2", "character", "Second character")
        await state_tracker.reserve("World1", "world_item", "First world item")
        
        # Get all entities
        all_entities = await state_tracker.get_all()
        assert len(all_entities) == 3
        assert "Character1" in all_entities
        assert "Character2" in all_entities
        assert "World1" in all_entities
    
    @pytest.mark.asyncio
    async def test_clear_entities(self, state_tracker, mock_config_enabled):
        """Test clearing all tracked entities."""
        # Reserve some entities
        await state_tracker.reserve("Character1", "character", "First character")
        await state_tracker.reserve("World1", "world_item", "First world item")
        
        # Verify they exist
        all_entities = await state_tracker.get_all()
        assert len(all_entities) == 2
        
        # Clear all entities
        await state_tracker.clear()
        
        # Verify they're gone
        all_entities = await state_tracker.get_all()
        assert len(all_entities) == 0
    
    @pytest.mark.asyncio
    async def test_has_similar_description_basic(self, state_tracker, mock_config_enabled):
        """Test basic description similarity checking."""
        # Reserve entity with substantial description
        await state_tracker.reserve("TestCharacter", "character", 
                                   "A brave warrior who fights against the dark forces of evil in the realm")
        
        # Test very similar description (should meet threshold)
        similar_name = await state_tracker.has_similar_description(
            "A brave warrior who fights against the dark forces of evil in the realm and protects", 
            "character"
        )
        assert similar_name == "TestCharacter"
        
        # Test dissimilar description
        different_name = await state_tracker.has_similar_description(
            "A small furry creature that loves cheese", 
            "character"
        )
        assert different_name is None
    
    @pytest.mark.asyncio
    async def test_has_similar_description_type_filter(self, state_tracker, mock_config_enabled):
        """Test description similarity with type filtering."""
        # Reserve entities with similar descriptions but different types
        await state_tracker.reserve("Character1", "character", 
                                   "A brave warrior who fights in the great battles")
        await state_tracker.reserve("World1", "world_item", 
                                   "A brave warrior who fights in the great battles")
        
        # Test with character type filter
        similar_name = await state_tracker.has_similar_description(
            "A brave warrior who fights in the great battles", 
            "character"
        )
        assert similar_name == "Character1"
        
        # Test with world_item type filter
        similar_name = await state_tracker.has_similar_description(
            "A brave warrior who fights in the great battles", 
            "world_item"
        )
        assert similar_name == "World1"
    
    @pytest.mark.asyncio
    async def test_has_similar_description_disabled(self, state_tracker, mock_config_disabled):
        """Test description similarity when StateTracker is disabled."""
        result = await state_tracker.has_similar_description("Any description", "character")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_entities_by_type(self, state_tracker, mock_config_enabled):
        """Test getting entities filtered by type."""
        # Reserve mixed entity types
        await state_tracker.reserve("Character1", "character", "First character")
        await state_tracker.reserve("Character2", "character", "Second character")
        await state_tracker.reserve("World1", "world_item", "First world item")
        await state_tracker.reserve("World2", "world_item", "Second world item")
        
        # Get characters only
        characters = await state_tracker.get_entities_by_type("character")
        assert len(characters) == 2
        assert "Character1" in characters
        assert "Character2" in characters
        assert "World1" not in characters
        
        # Get world items only
        world_items = await state_tracker.get_entities_by_type("world_item")
        assert len(world_items) == 2
        assert "World1" in world_items
        assert "World2" in world_items
        assert "Character1" not in world_items
    
    @pytest.mark.asyncio
    async def test_get_recent_entities(self, state_tracker, mock_config_enabled):
        """Test getting entities within time window."""
        # Reserve entities with known timestamps
        await state_tracker.reserve("RecentChar", "character", "Recent character")
        
        # Get recent entities (default 24 hours)
        recent_entities = await state_tracker.get_recent_entities(24)
        assert len(recent_entities) == 1
        assert "RecentChar" in recent_entities
        
        # Test with shorter time window (should still include recent entity)
        recent_entities = await state_tracker.get_recent_entities(1)
        assert len(recent_entities) == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, state_tracker, mock_config_enabled):
        """Test concurrent access to StateTracker (thread safety)."""
        
        async def reserve_entity(name: str, entity_type: str, description: str):
            return await state_tracker.reserve(name, entity_type, description)
        
        # Attempt concurrent reservations of different entities
        tasks = [
            reserve_entity(f"Character{i}", "character", f"Character {i} description")
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All reservations should succeed
        assert all(results)
        
        # Verify all entities were stored
        all_entities = await state_tracker.get_all()
        assert len(all_entities) == 10
    
    @pytest.mark.asyncio
    async def test_concurrent_duplicate_reservations(self, state_tracker, mock_config_enabled):
        """Test concurrent attempts to reserve the same entity name."""
        
        async def reserve_same_entity():
            return await state_tracker.reserve("SameName", "character", "Description")
        
        # Attempt concurrent reservations of the same name
        tasks = [reserve_same_entity() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # Only one should succeed
        successful_reservations = sum(results)
        assert successful_reservations == 1
        
        # Verify only one entity exists
        all_entities = await state_tracker.get_all()
        assert len(all_entities) == 1
        assert "SameName" in all_entities
    
    @pytest.mark.asyncio
    async def test_timestamp_parsing_edge_cases(self, state_tracker, mock_config_enabled):
        """Test edge cases in timestamp parsing."""
        # Reserve entity normally
        await state_tracker.reserve("TestChar", "character", "Test description")
        
        # Modify timestamp to invalid format directly
        state_tracker._entities["TestChar"]["timestamp"] = "invalid-timestamp"
        
        # get_recent_entities should handle invalid timestamps gracefully
        recent_entities = await state_tracker.get_recent_entities(24)
        # Should still work without crashing, though this entity may be excluded
        assert isinstance(recent_entities, dict)
    
    @pytest.mark.asyncio
    async def test_description_similarity_edge_cases(self, state_tracker, mock_config_enabled):
        """Test edge cases in description similarity."""
        # Reserve entity with short description
        await state_tracker.reserve("ShortDesc", "character", "Short")
        
        # Test similarity with short descriptions (should return None)
        similar_name = await state_tracker.has_similar_description("Brief", "character")
        assert similar_name is None
        
        # Reserve entity with empty description
        await state_tracker.reserve("EmptyDesc", "character", "")
        
        # Test similarity with empty descriptions
        similar_name = await state_tracker.has_similar_description("", "character")
        assert similar_name is None


@pytest.mark.integration
class TestStateTrackerIntegration:
    """Integration tests for StateTracker with other SAGA components."""
    
    @pytest.mark.asyncio
    async def test_state_tracker_with_character_validation(self):
        """Test StateTracker integration with character validation patterns."""
        with patch('processing.state_tracker.config') as mock_config:
            mock_config.STATE_TRACKER_ENABLED = True
            mock_config.STATE_TRACKER_SIMILARITY_THRESHOLD = 0.85
            
            tracker = StateTracker()
            await tracker.clear()
            
            # Simulate character bootstrap process
            await tracker.reserve("Alice", "character", "A skilled archer from the northern lands")
            await tracker.reserve("Bob", "character", "A wise mage with ancient knowledge")
            
            # Test conflict detection
            conflict_result = await tracker.reserve("Alice", "world_item", "A magic bow")
            assert conflict_result is False
            
            # Test similarity detection (use exact same words to ensure high similarity)
            similar_name = await tracker.has_similar_description(
                "A skilled archer from the northern lands", 
                "character"
            )
            assert similar_name == "Alice"
    
    @pytest.mark.asyncio
    async def test_state_tracker_with_world_validation(self):
        """Test StateTracker integration with world item validation patterns."""
        with patch('processing.state_tracker.config') as mock_config:
            mock_config.STATE_TRACKER_ENABLED = True
            mock_config.STATE_TRACKER_SIMILARITY_THRESHOLD = 0.85
            
            tracker = StateTracker()
            await tracker.clear()
            
            # Simulate world bootstrap process
            await tracker.reserve("Mystic Forest", "world_item", "A dense forest filled with magical creatures")
            await tracker.reserve("Dragon Peak", "world_item", "A towering mountain where dragons nest")
            
            # Test type filtering
            world_items = await tracker.get_entities_by_type("world_item")
            assert len(world_items) == 2
            assert "Mystic Forest" in world_items
            assert "Dragon Peak" in world_items
            
            # Test no character conflicts
            characters = await tracker.get_entities_by_type("character")
            assert len(characters) == 0