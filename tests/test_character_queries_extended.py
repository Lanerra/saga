# tests/test_character_queries_extended.py
"""Extended tests for data_access/character_queries.py to improve coverage.

These tests are intentionally written against the current single-query implementations
of get_character_profile_by_name()/get_character_profile_by_id(), rather than older
multi-query behavior.
"""

from unittest.mock import AsyncMock

import pytest

from data_access import character_queries
from data_access.cache_coordinator import clear_all_data_access_caches


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
                                # Canonical type is rel_type; rel_props may not include a `type` property.
                                "rel_props": {"source_profile_managed": True},
                            }
                        ],
                        "dev_events": [{"summary": "Event1", "chapter": 1, "is_provisional": False}],
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

        # Single relationship to a target keeps legacy shape: dict[str, Any]
        assert "Bob" in profile.relationships
        assert isinstance(profile.relationships["Bob"], dict)
        assert profile.relationships["Bob"]["type"] == "FRIEND_OF"

        profile_dict = profile.to_dict()
        assert profile_dict["development_in_chapter_1"] == "Event1"

        clear_all_data_access_caches()

    async def test_get_character_profile_by_name_preserves_multi_relationships_same_target(self, monkeypatch):
        """Regression: multiple relationships to the same target must not overwrite each other."""

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
                                "rel_props": {
                                    "source_profile_managed": True,
                                    "description": "Genuinely likes him",
                                },
                            },
                            {
                                "target_name": "Bob",
                                "rel_type": "ENEMY_OF",
                                "rel_props": {
                                    "source_profile_managed": True,
                                    "description": "Also rivals him",
                                },
                            },
                        ],
                        "dev_events": [],
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

        assert "Bob" in profile.relationships
        bob_rels = profile.relationships["Bob"]
        assert isinstance(bob_rels, list)
        assert len(bob_rels) == 2

        # Deterministic ordering: by type then description then chapter_added.
        assert bob_rels[0]["type"] == "ENEMY_OF"
        assert bob_rels[1]["type"] == "FRIEND_OF"

        assert {r["type"] for r in bob_rels} == {"FRIEND_OF", "ENEMY_OF"}

        clear_all_data_access_caches()

    async def test_get_character_profile_by_name_no_optional_data_still_returns_profile(self, monkeypatch):
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

        clear_all_data_access_caches()

    async def test_get_character_profile_by_id_no_optional_data_still_returns_profile(self, monkeypatch):
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

        clear_all_data_access_caches()
