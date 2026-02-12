# tests/test_character_queries.py
"""Tests for data_access/character_queries.py"""

from unittest.mock import AsyncMock, patch

import pytest
from neo4j.exceptions import ServiceUnavailable

import utils
from core.exceptions import DatabaseError
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
        monkeypatch.setattr(character_queries.neo4j_manager, "execute_read_query", mock_read)

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
        monkeypatch.setattr(character_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await character_queries.get_all_character_names()
        assert len(result) == 3
        assert "Alice" in result


@pytest.mark.asyncio
class TestGetCharacterInfoForSnippet:
    """Tests for getting character info for snippet."""

    async def test_get_character_info_empty(self, monkeypatch):
        """Test getting character info when none exist."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(character_queries.neo4j_manager, "execute_read_query", mock_read)

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
        monkeypatch.setattr(character_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await character_queries.get_character_info_for_snippet_from_db("Alice", 10)
        assert result is not None
        assert result["description"] == "A brave hero"
        assert result["most_recent_development_note"] == "N/A"

    async def test_get_character_info_no_optional_data_still_returns(self, monkeypatch):
        """Regression: character row should not be dropped when optional matches find nothing."""
        character_queries.CHAR_NAME_TO_CANONICAL.clear()
        character_queries.CHAR_NAME_TO_CANONICAL["lonely"] = "Lonely"

        mock_read = AsyncMock(
            return_value=[
                {
                    "description": "No ties",
                    "current_status": "Active",
                    "char_is_provisional": False,
                    "dev_events": [],
                    "provisional_rel_count": 0,
                }
            ]
        )
        monkeypatch.setattr(character_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await character_queries.get_character_info_for_snippet_from_db("Lonely", 10)
        assert result is not None
        assert result["description"] == "No ties"
        assert result["current_status"] == "Active"
        assert result["most_recent_development_note"] == "N/A"
        assert result["is_provisional_overall"] is False

    async def test_get_character_info_for_snippet_raises_on_database_error(self):
        """get_character_info_for_snippet_from_db should propagate DatabaseError, not return None."""
        with patch("data_access.character_queries.neo4j_manager") as mock_neo4j:
            mock_neo4j.execute_read_query = AsyncMock(side_effect=ServiceUnavailable("Connection lost"))
            mock_neo4j.connect = AsyncMock()

            with pytest.raises(DatabaseError):
                await character_queries.get_character_info_for_snippet_from_db("Alice", chapter_limit=5)

    async def test_get_character_info_for_snippet_returns_none_on_not_found(self):
        """When character truly doesn't exist, should return None (not an error case)."""
        with patch("data_access.character_queries.neo4j_manager") as mock_neo4j:
            mock_neo4j.execute_read_query = AsyncMock(return_value=[])

            result = await character_queries.get_character_info_for_snippet_from_db("NonExistent", chapter_limit=5)
            assert result is None


@pytest.mark.asyncio
class TestFindThinCharacters:
    """Tests for finding thin characters for enrichment."""

    async def test_find_thin_characters_empty(self, monkeypatch):
        """Test finding thin characters when none exist."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(character_queries.neo4j_manager, "execute_read_query", mock_read)

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
        monkeypatch.setattr(character_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await character_queries.find_thin_characters_for_enrichment()
        assert len(result) == 1


@pytest.mark.asyncio
class TestSyncCharacters:
    """Tests for syncing characters."""

    async def test_sync_characters_empty(self, monkeypatch):
        """Syncing an empty list completes without calling the database."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(character_queries.neo4j_manager, "execute_cypher_batch", mock_execute)

        await character_queries.sync_characters([], 1)
        mock_execute.assert_not_called()

    async def test_sync_characters_single(self, monkeypatch):
        """Syncing a single character persists without error."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(character_queries.neo4j_manager, "execute_cypher_batch", mock_execute)

        profile = CharacterProfile.from_dict("Alice", {"description": "A hero", "traits": ["brave"]})
        await character_queries.sync_characters([profile], 1)

    async def test_sync_characters_updates_name_map(self, monkeypatch):
        """Sync updates the canonical-name map without clearing unrelated entries."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(character_queries.neo4j_manager, "execute_cypher_batch", mock_execute)

        character_queries.CHAR_NAME_TO_CANONICAL.clear()
        character_queries.CHAR_NAME_TO_CANONICAL["existing"] = "Existing"

        profiles = [
            CharacterProfile.from_dict("Alice", {"description": "A hero", "traits": ["brave"]}),
        ]
        await character_queries.sync_characters(profiles, 1)

        # Existing entries are preserved; new entries are added.
        assert character_queries.CHAR_NAME_TO_CANONICAL["existing"] == "Existing"
        assert character_queries.CHAR_NAME_TO_CANONICAL.get(utils._normalize_for_id("Alice")) == "Alice"

    async def test_sync_characters_multiple(self, monkeypatch):
        """Syncing multiple characters persists without error."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(character_queries.neo4j_manager, "execute_cypher_batch", mock_execute)

        profiles = [
            CharacterProfile.from_dict("Alice", {"description": "A hero", "traits": ["brave"]}),
            CharacterProfile.from_dict("Bob", {"description": "A friend", "traits": ["loyal"]}),
        ]
        await character_queries.sync_characters(profiles, 1)


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

        # Seed stale state to ensure fetch rebuilds deterministically.
        character_queries.CHAR_NAME_TO_CANONICAL.clear()
        character_queries.CHAR_NAME_TO_CANONICAL["stale"] = "Stale"

        result = await character_queries.get_character_profiles()
        assert len(result) == 1
        assert result[0].name == "Alice"

        assert "stale" not in character_queries.CHAR_NAME_TO_CANONICAL
        assert character_queries.CHAR_NAME_TO_CANONICAL.get(utils._normalize_for_id("Alice")) == "Alice"


@pytest.mark.asyncio
class TestGetCharactersForChapterContext:
    """Tests for getting characters for chapter context."""

    async def test_get_characters_for_chapter_context_empty(self, monkeypatch):
        """Test getting characters when none exist."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(character_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await character_queries.get_characters_for_chapter_context_native(chapter_number=1, limit=10)
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

        result = await character_queries.get_characters_for_chapter_context_native(chapter_number=1, limit=10)
        assert len(result) == 1
        assert result[0].name == "Alice"


@pytest.mark.asyncio
async def test_character_queries_catch_specific_exceptions():
    """Verify character_queries catches specific exceptions, not Exception."""
    import inspect

    source = inspect.getsource(character_queries)

    assert "except Exception" not in source, "Found broad 'except Exception' handlers"

    assert "from neo4j.exceptions import" in source or "neo4j.exceptions" in source, "Should import specific Neo4j exceptions"
