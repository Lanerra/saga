# tests/test_character_queries.py
"""Tests for data_access/character_queries.py"""

from unittest.mock import AsyncMock

import pytest
from neo4j.exceptions import ServiceUnavailable

from core.exceptions import DatabaseError
from core.service_context import get_services
from data_access import character_queries
from models import CharacterProfile
from tests.fakes.service_context import patch_service


class TestCharacterNameResolution:
    """Tests for character name resolution."""

    def test_resolve_character_name_exists(self) -> None:
        """Test resolving character name that exists."""
        character_queries.CHAR_NAME_TO_CANONICAL.clear()
        character_queries.CHAR_NAME_TO_CANONICAL["alice"] = "Alice"

        result = character_queries.resolve_character_name("alice")
        assert result == "Alice"

    def test_resolve_character_name_missing(self) -> None:
        """Test resolving character name that doesn't exist."""
        character_queries.CHAR_NAME_TO_CANONICAL.clear()

        result = character_queries.resolve_character_name("Unknown")
        assert result == "Unknown"

    def test_resolve_character_name_empty(self) -> None:
        """Test resolving empty character name."""
        result = character_queries.resolve_character_name("")
        assert result == ""


@pytest.mark.asyncio
class TestGetAllCharacterNames:
    """Tests for getting all character names."""

    async def test_get_all_character_names_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting character names when none exist."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(get_services().database, "execute_read_query", mock_read)

        result = await character_queries.get_all_character_names()
        assert isinstance(result, list)
        assert len(result) == 0

    async def test_get_all_character_names_multiple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting multiple character names."""
        mock_read = AsyncMock(
            return_value=[
                {"name": "Alice"},
                {"name": "Bob"},
                {"name": "Charlie"},
            ]
        )
        monkeypatch.setattr(get_services().database, "execute_read_query", mock_read)

        result = await character_queries.get_all_character_names()
        assert len(result) == 3
        assert "Alice" in result


@pytest.mark.asyncio
class TestGetCharacterInfoForSnippet:
    """Tests for getting character info for snippet."""

    async def test_get_character_info_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting character info when none exist."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(get_services().database, "execute_read_query", mock_read)

        result = await character_queries.get_character_info_for_snippet_from_db("Alice", 10)
        assert result is None

    async def test_get_character_info_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        monkeypatch.setattr(get_services().database, "execute_read_query", mock_read)

        result = await character_queries.get_character_info_for_snippet_from_db("Alice", 10)
        assert result is not None
        assert result["description"] == "A brave hero"
        assert result["most_recent_development_note"] == "N/A"

    async def test_get_character_info_no_optional_data_still_returns(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        monkeypatch.setattr(get_services().database, "execute_read_query", mock_read)

        result = await character_queries.get_character_info_for_snippet_from_db("Lonely", 10)
        assert result is not None
        assert result["description"] == "No ties"
        assert result["current_status"] == "Active"
        assert result["most_recent_development_note"] == "N/A"
        assert result["is_provisional_overall"] is False

    async def test_get_character_info_for_snippet_raises_on_database_error(self) -> None:
        """get_character_info_for_snippet_from_db should propagate DatabaseError, not return None."""
        with patch_service('database') as mock_neo4j:
            mock_neo4j.execute_read_query = AsyncMock(side_effect=ServiceUnavailable("Connection lost"))
            mock_neo4j.connect = AsyncMock()

            with pytest.raises(DatabaseError):
                await character_queries.get_character_info_for_snippet_from_db("Alice", chapter_limit=5)

    async def test_get_character_info_for_snippet_returns_none_on_not_found(self) -> None:
        """When character truly doesn't exist, should return None (not an error case)."""
        with patch_service('database') as mock_neo4j:
            mock_neo4j.execute_read_query = AsyncMock(return_value=[])

            result = await character_queries.get_character_info_for_snippet_from_db("NonExistent", chapter_limit=5)
            assert result is None


@pytest.mark.asyncio
class TestFindThinCharacters:
    """Tests for finding thin characters for enrichment."""

    async def test_find_thin_characters_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test finding thin characters when none exist."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(get_services().database, "execute_read_query", mock_read)

        result = await character_queries.find_thin_characters_for_enrichment()
        assert isinstance(result, list)
        assert len(result) == 0

    async def test_find_thin_characters_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test finding thin characters that exist."""
        mock_read = AsyncMock(
            return_value=[
                {
                    "name": "Alice",
                    "description": "Brief description",
                }
            ]
        )
        monkeypatch.setattr(get_services().database, "execute_read_query", mock_read)

        result = await character_queries.find_thin_characters_for_enrichment()
        assert len(result) == 1


@pytest.mark.asyncio
class TestSyncCharacters:
    """Tests for syncing characters."""

    async def test_sync_characters_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Syncing an empty list completes without calling the database."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(get_services().database, "execute_cypher_batch", mock_execute)

        await character_queries.sync_characters([], 1)
        mock_execute.assert_not_called()

    async def test_sync_characters_single(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Syncing a single character persists without error."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(get_services().database, "execute_cypher_batch", mock_execute)

        profile = CharacterProfile.from_dict("Alice", {"description": "A hero", "traits": ["brave"]})
        await character_queries.sync_characters([profile], 1)

    async def test_sync_characters_updates_name_map(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sync updates the canonical-name map without clearing unrelated entries."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(get_services().database, "execute_cypher_batch", mock_execute)

        character_queries.CHAR_NAME_TO_CANONICAL.clear()
        character_queries.CHAR_NAME_TO_CANONICAL["existing"] = "Existing"

        profiles = [
            CharacterProfile.from_dict("Alice", {"description": "A hero", "traits": ["brave"]}),
        ]
        await character_queries.sync_characters(profiles, 1)

        # Existing entries are preserved; new entries are added.
        assert character_queries.CHAR_NAME_TO_CANONICAL["existing"] == "Existing"
        assert character_queries.CHAR_NAME_TO_CANONICAL.get("Alice") == "Alice"
        assert "alice" not in character_queries.CHAR_NAME_TO_CANONICAL

    async def test_sync_characters_multiple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Syncing multiple characters persists without error."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(get_services().database, "execute_cypher_batch", mock_execute)

        profiles = [
            CharacterProfile.from_dict("Alice", {"description": "A hero", "traits": ["brave"]}),
            CharacterProfile.from_dict("Bob", {"description": "A friend", "traits": ["loyal"]}),
        ]
        await character_queries.sync_characters(profiles, 1)


@pytest.mark.asyncio
class TestGetCharacterProfiles:
    """Tests for getting character profiles."""

    async def test_get_character_profiles_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting profiles when none exist."""

        async def fake_read(query: str, params: dict[str, object] | None = None) -> list[dict[str, object]]:
            if "RETURN c.name" in query:
                return []
            return []

        monkeypatch.setattr(
            get_services().database,
            "execute_read_query",
            AsyncMock(side_effect=fake_read),
        )

        result = await character_queries.get_character_profiles()
        assert isinstance(result, list)
        assert len(result) == 0

    async def test_get_character_profiles_single(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting single character profile."""

        async def fake_read(query: str, params: dict[str, object] | None = None) -> list[dict[str, object]]:
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
            get_services().database,
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
        assert character_queries.CHAR_NAME_TO_CANONICAL.get("Alice") == "Alice"
        assert "alice" not in character_queries.CHAR_NAME_TO_CANONICAL


@pytest.mark.asyncio
class TestGetCharactersForChapterContext:
    """Tests for getting characters for chapter context."""

    async def test_get_characters_for_chapter_context_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting characters when none exist."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(get_services().database, "execute_read_query", mock_read)

        result = await character_queries.get_characters_for_chapter_context_native(chapter_number=1, limit=10)
        assert isinstance(result, list)

    async def test_get_characters_for_chapter_context_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting characters that exist."""

        async def fake_read(query: str, params: dict[str, object] | None = None) -> list[dict[str, object]]:
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
            get_services().database,
            "execute_read_query",
            AsyncMock(side_effect=fake_read),
        )

        result = await character_queries.get_characters_for_chapter_context_native(chapter_number=1, limit=10)
        assert len(result) == 1
        assert result[0].name == "Alice"


@pytest.mark.asyncio
async def test_character_queries_catch_specific_exceptions() -> None:
    """Verify character_queries catches specific exceptions, not Exception."""
    import inspect

    source = inspect.getsource(character_queries)

    assert "except Exception" not in source, "Found broad 'except Exception' handlers"

    assert "from neo4j.exceptions import" in source or "neo4j.exceptions" in source, "Should import specific Neo4j exceptions"
