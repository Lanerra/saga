"""Tests for data_access/character_queries.py"""
from unittest.mock import AsyncMock

import pytest

import utils
from data_access import character_queries
from models import CharacterProfile


class TestCharacterNameResolution:
    """Tests for character name resolution."""

    def test_resolve_character_name_exists(self):
        """Test resolving character name that exists."""
        character_queries.CHAR_NAME_TO_CANONICAL.clear()
        character_queries.CHAR_NAME_TO_CANONICAL["alice"] = "Alice"

        result = character_queries.resolve_character_name("alice")
        assert result == "Alice"

    def test_resolve_character_name_missing(self):
        """Test resolving character name that doesn't exist."""
        character_queries.CHAR_NAME_TO_CANONICAL.clear()

        result = character_queries.resolve_character_name("Unknown")
        assert result == "Unknown"

    def test_resolve_character_name_empty(self):
        """Test resolving empty character name."""
        result = character_queries.resolve_character_name("")
        assert result == ""


@pytest.mark.asyncio
class TestGetAllCharacterNames:
    """Tests for getting all character names."""

    async def test_get_all_character_names_empty(self, monkeypatch):
        """Test getting character names when none exist."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            character_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await character_queries.get_all_character_names()
        assert isinstance(result, list)
        assert len(result) == 0

    async def test_get_all_character_names_multiple(self, monkeypatch):
        """Test getting multiple character names."""
        mock_read = AsyncMock(
            return_value=[
                {"name": "Alice"},
                {"name": "Bob"},
                {"name": "Charlie"},
            ]
        )
        monkeypatch.setattr(
            character_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await character_queries.get_all_character_names()
        assert len(result) == 3
        assert "Alice" in result


@pytest.mark.asyncio
class TestGetCharacterInfoForSnippet:
    """Tests for getting character info for snippet."""

    async def test_get_character_info_empty(self, monkeypatch):
        """Test getting character info when none exist."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            character_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await character_queries.get_character_info_for_snippet_from_db("Alice", 10)
        assert result is None

    async def test_get_character_info_found(self, monkeypatch):
        """Test getting character info that exists."""
        character_queries.CHAR_NAME_TO_CANONICAL.clear()
        character_queries.CHAR_NAME_TO_CANONICAL["alice"] = "Alice"

        mock_read = AsyncMock(
            return_value=[
                {
                    "description": "A brave hero",
                    "current_status": "Active",
                    "most_current_dev_event": None,
                    "is_provisional_overall": False,
                }
            ]
        )
        monkeypatch.setattr(
            character_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await character_queries.get_character_info_for_snippet_from_db("Alice", 10)
        assert result is not None
        assert result["description"] == "A brave hero"
        assert result["most_recent_development_note"] == "N/A"


@pytest.mark.asyncio
class TestFindThinCharacters:
    """Tests for finding thin characters for enrichment."""

    async def test_find_thin_characters_empty(self, monkeypatch):
        """Test finding thin characters when none exist."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            character_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await character_queries.find_thin_characters_for_enrichment()
        assert isinstance(result, list)
        assert len(result) == 0

    async def test_find_thin_characters_found(self, monkeypatch):
        """Test finding thin characters that exist."""
        mock_read = AsyncMock(
            return_value=[
                {
                    "name": "Alice",
                    "description": "Brief description",
                }
            ]
        )
        monkeypatch.setattr(
            character_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await character_queries.find_thin_characters_for_enrichment()
        assert len(result) > 0


@pytest.mark.asyncio
class TestSyncCharacters:
    """Tests for syncing characters."""

    async def test_sync_characters_empty(self, monkeypatch):
        """Test syncing empty character list."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(
            character_queries.neo4j_manager, "execute_cypher_batch", mock_execute
        )

        result = await character_queries.sync_characters([], 1)
        assert result is True

    async def test_sync_characters_single(self, monkeypatch):
        """Test syncing single character."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(
            character_queries.neo4j_manager, "execute_cypher_batch", mock_execute
        )

        profile = CharacterProfile.from_dict(
            "Alice", {"description": "A hero", "traits": ["brave"]}
        )
        result = await character_queries.sync_characters([profile], 1)
        assert result is True

    async def test_sync_characters_multiple(self, monkeypatch):
        """Test syncing multiple characters."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(
            character_queries.neo4j_manager, "execute_cypher_batch", mock_execute
        )

        profiles = [
            CharacterProfile.from_dict("Alice", {"description": "A hero", "traits": ["brave"]}),
            CharacterProfile.from_dict("Bob", {"description": "A friend", "traits": ["loyal"]}),
        ]
        result = await character_queries.sync_characters(profiles, 1)
        assert result is True


@pytest.mark.asyncio
class TestGetCharacterProfiles:
    """Tests for getting character profiles."""

    async def test_get_character_profiles_empty(self, monkeypatch):
        """Test getting profiles when none exist."""
        async def fake_read(query, params=None):
            if "RETURN c.name" in query:
                return []
            return []

        monkeypatch.setattr(
            character_queries.neo4j_manager,
            "execute_read_query",
            AsyncMock(side_effect=fake_read),
        )

        result = await character_queries.get_character_profiles()
        assert isinstance(result, list)
        assert len(result) == 0

    async def test_get_character_profiles_single(self, monkeypatch):
        """Test getting single character profile."""
        async def fake_read(query, params=None):
            if "RETURN c.name" in query:
                return [{"name": "Alice"}]
            if "RETURN c" in query:
                return [
                    {
                        "c": {
                            "name": "Alice",
                            "description": "A hero",
                        }
                    }
                ]
            if "HAS_TRAIT" in query:
                return [{"trait_name": "brave"}]
            if "RETURN target.name" in query:
                return []
            if "DEVELOPED_IN_CHAPTER" in query:
                return []
            return []

        monkeypatch.setattr(
            character_queries.neo4j_manager,
            "execute_read_query",
            AsyncMock(side_effect=fake_read),
        )

        result = await character_queries.get_character_profiles()
        assert len(result) == 1
        assert result[0].name == "Alice"


@pytest.mark.asyncio
class TestGetCharactersForChapterContext:
    """Tests for getting characters for chapter context."""

    async def test_get_characters_for_chapter_context_empty(self, monkeypatch):
        """Test getting characters when none exist."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            character_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await character_queries.get_characters_for_chapter_context_native(
            chapter_number=1, limit=10
        )
        assert isinstance(result, list)

    async def test_get_characters_for_chapter_context_found(self, monkeypatch):
        """Test getting characters that exist."""
        async def fake_read(query, params=None):
            if "RETURN c" in query:
                return [
                    {
                        "c": {
                            "name": "Alice",
                            "description": "A hero",
                        }
                    }
                ]
            if "HAS_TRAIT" in query:
                return [{"trait_name": "brave"}]
            if "RETURN target.name" in query:
                return [{"target_name": "Bob", "rel_props": {"type": "FRIEND_OF"}}]
            if "DEVELOPED_IN_CHAPTER" in query:
                return []
            return []

        monkeypatch.setattr(
            character_queries.neo4j_manager,
            "execute_read_query",
            AsyncMock(side_effect=fake_read),
        )

        result = await character_queries.get_characters_for_chapter_context_native(
            chapter_number=1, limit=10
        )
        assert len(result) > 0
        assert result[0].name == "Alice"
