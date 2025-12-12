"""Extended tests for data_access/character_queries.py to improve coverage.

These tests are intentionally written against the current single-query implementations
of get_character_profile_by_name()/get_character_profile_by_id(), rather than older
multi-query behavior.
"""

from unittest.mock import AsyncMock

import pytest

from data_access import character_queries


@pytest.mark.asyncio
class TestGetCharacterProfileFetchesNoRowDropping:
    """Regression tests: character row should be preserved even when optional patterns match nothing."""

    async def test_get_character_profile_by_name_full_details(self, monkeypatch):
        """Profile by name returns core fields + optional collections when present."""
        async def mock_read(query, params=None):
            if "MATCH (c:Character {name: $name})" in query:
                return [
                    {
                        "c": {"name": "Alice", "description": "Hero", "id": "1"},
                        "traits": ["Brave"],
                        "relationships": [
                            {
                                "target_name": "Bob",
                                "rel_type": "FRIEND_OF",
                                "rel_props": {"type": "FRIEND_OF", "source_profile_managed": True},
                            }
                        ],
                        "dev_events": [
                            {"summary": "Event1", "chapter": 1, "is_provisional": False}
                        ],
                    }
                ]
            return []

        monkeypatch.setattr(
            character_queries.neo4j_manager,
            "execute_read_query",
            AsyncMock(side_effect=mock_read),
        )

        profile = await character_queries.get_character_profile_by_name("Alice")
        assert profile is not None
        assert profile.name == "Alice"
        assert "Brave" in profile.traits
        assert "Bob" in profile.relationships

        profile_dict = profile.to_dict()
        assert profile_dict["development_in_chapter_1"] == "Event1"

        character_queries.get_character_profile_by_name.cache_clear()

    async def test_get_character_profile_by_name_no_optional_data_still_returns_profile(
        self, monkeypatch
    ):
        """Profile by name should not be dropped when traits/relationships/dev events are absent."""
        async def mock_read(query, params=None):
            if "MATCH (c:Character {name: $name})" in query:
                return [
                    {
                        "c": {"name": "Lonely", "description": "No ties", "id": "99"},
                        "traits": [],
                        "relationships": [],
                        "dev_events": [],
                    }
                ]
            return []

        monkeypatch.setattr(
            character_queries.neo4j_manager,
            "execute_read_query",
            AsyncMock(side_effect=mock_read),
        )

        profile = await character_queries.get_character_profile_by_name("Lonely")
        assert profile is not None
        assert profile.name == "Lonely"
        assert profile.traits == []
        assert profile.relationships == {}

        character_queries.get_character_profile_by_name.cache_clear()

    async def test_get_character_profile_by_id_no_optional_data_still_returns_profile(
        self, monkeypatch
    ):
        """Profile by id should not be dropped when traits/relationships/dev events are absent."""
        async def mock_read(query, params=None):
            if "MATCH (c:Character {id: $character_id})" in query:
                return [
                    {
                        "c": {"name": "Lonely", "description": "No ties", "id": "99"},
                        "traits": [],
                        "relationships": [],
                        "dev_events": [],
                    }
                ]
            return []

        monkeypatch.setattr(
            character_queries.neo4j_manager,
            "execute_read_query",
            AsyncMock(side_effect=mock_read),
        )

        profile = await character_queries.get_character_profile_by_id("99")
        assert profile is not None
        assert profile.name == "Lonely"
        assert profile.traits == []
        assert profile.relationships == {}

        character_queries.get_character_profile_by_id.cache_clear()
