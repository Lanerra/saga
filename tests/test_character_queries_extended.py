"""Extended tests for data_access/character_queries.py to improve coverage."""
import pytest
from unittest.mock import AsyncMock, patch
from data_access import character_queries
from models import CharacterProfile
from models.kg_constants import KG_IS_PROVISIONAL, KG_NODE_CHAPTER_UPDATED

@pytest.mark.asyncio
class TestSyncFullStateExtended:
    """Extended tests for sync_full_state_from_object_to_db."""

    async def test_sync_full_state_complex_logic(self, monkeypatch):
        """Test syncing complex character state with relationships, traits, and events."""
        
        async def mock_read(query, params=None):
            if "RETURN c.name AS name" in query:
                return [{"name": "OldChar"}] # Should be marked deleted
            return []

        async def mock_write(query, params=None):
            return []

        monkeypatch.setattr(character_queries.neo4j_manager, "execute_read_query", AsyncMock(side_effect=mock_read))
        monkeypatch.setattr(character_queries.neo4j_manager, "execute_cypher_batch", AsyncMock(side_effect=mock_write))

        profiles_data = {
            "Alice": {
                "description": "Hero",
                "traits": ["Brave", "Smart"],
                "development_in_chapter_1": "Started journey",
                "source_quality_chapter_1": "provisional_from_unrevised_draft",
                "relationships": {
                    "Bob": {"type": "FRIEND_OF", "description": "Ally", "chapter_added": 1}
                }
            }
        }

        result = await character_queries.sync_full_state_from_object_to_db(profiles_data)
        assert result is True

        calls = character_queries.neo4j_manager.execute_cypher_batch.call_args_list
        assert len(calls) > 0
        statements = calls[0][0][0]
        combined_query_text = " ".join([s[0] for s in statements])

        # Verify key logic blocks
        assert "SET c.is_deleted = TRUE" in combined_query_text # Delete OldChar
        assert "MERGE (c:Character {name: $char_name_val})" in combined_query_text # Merge Alice
        assert "MERGE (ni)-[:HAS_CHARACTER]->(c)" in combined_query_text # Link to NovelInfo
        assert "MERGE (c)-[:HAS_TRAIT]->(t)" in combined_query_text # Traits
        assert "CREATE (dev:DevelopmentEvent)" in combined_query_text # Dev Event
        assert "MERGE (s)-[r:`FRIEND_OF`]->(o)" in combined_query_text # Relationship

    async def test_sync_full_state_invalid_profile(self, monkeypatch):
        """Test handling of invalid profile data."""
        monkeypatch.setattr(character_queries.neo4j_manager, "execute_read_query", AsyncMock(return_value=[]))
        monkeypatch.setattr(character_queries.neo4j_manager, "execute_cypher_batch", AsyncMock(return_value=[]))

        profiles_data = {
            "InvalidChar": "Not a dict" # Should be skipped
        }
        
        result = await character_queries.sync_full_state_from_object_to_db(profiles_data)
        assert result is True # Should not crash

@pytest.mark.asyncio
class TestGetCharacterProfileByNameExtended:
    """Extended tests for get_character_profile_by_name."""

    async def test_get_character_profile_full_details(self, monkeypatch):
        """Test retrieving profile with all sub-elements."""
        
        mock_char_node = {"name": "Alice", "description": "Hero", "id": "1"}
        
        async def mock_read(query, params=None):
            if "RETURN c" in query:
                return [{"c": mock_char_node}]
            if "RETURN t.name AS trait_name" in query:
                return [{"trait_name": "Brave"}]
            if "RETURN target.name AS target_name" in query:
                # Relationships
                return [{"target_name": "Bob", "rel_type": "FRIEND_OF", "rel_props": {"type": "FRIEND_OF", "source_profile_managed": True}}]
            if "RETURN dev.summary AS summary" in query:
                # Dev events
                return [{"summary": "Event1", "chapter": 1, "is_provisional": False}]
            return []

        monkeypatch.setattr(character_queries.neo4j_manager, "execute_read_query", AsyncMock(side_effect=mock_read))

        profile = await character_queries.get_character_profile_by_name("Alice")
        assert profile is not None
        assert profile.name == "Alice"
        assert "Brave" in profile.traits
        assert "Bob" in profile.relationships
        # Verify dev event mapped to dict
        # CharacterProfile model internals might store it or it might be in additional_properties
        # Based on implementation: profile[dev_key] = summary
        # So check if it's accessible.
        # Assuming we can inspect the model or converting to dict works.
        profile_dict = profile.to_dict()
        assert "development_in_chapter_1" in profile_dict
        assert profile_dict["development_in_chapter_1"] == "Event1"
