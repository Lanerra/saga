"""Extended tests for data_access/character_queries.py to improve coverage."""
import pytest
from unittest.mock import AsyncMock, patch
from data_access import character_queries
from models import CharacterProfile
from models.kg_constants import KG_IS_PROVISIONAL, KG_NODE_CHAPTER_UPDATED


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
