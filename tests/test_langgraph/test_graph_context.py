# tests/test_langgraph/test_graph_context.py
"""
Tests for LangGraph graph context (Step 1.3.1).

Tests the build_context_from_graph function and helper functions.
"""

from unittest.mock import AsyncMock, patch

import pytest

from core.langgraph.graph_context import (
    _get_character_relationships,
    _get_characters_by_names,
    _get_location_details,
    _get_recent_summaries,
    build_context_from_graph,
    get_key_events,
)


@pytest.mark.asyncio
class TestBuildContextFromGraph:
    """Tests for build_context_from_graph function."""

    async def test_build_context_default_params(
        self,
        fake_neo4j,
        mock_character_queries,
        mock_world_queries,
        mock_chapter_queries,
    ):
        """Test building context with default parameters."""
        with patch("core.langgraph.graph_context.character_queries", mock_character_queries):
            with patch("core.langgraph.graph_context.world_queries", mock_world_queries):
                with patch("core.langgraph.graph_context.chapter_queries", mock_chapter_queries):
                    with patch("core.langgraph.graph_context.neo4j_manager", fake_neo4j):
                        context = await build_context_from_graph(current_chapter=5)

                        assert "characters" in context
                        assert "world_items" in context
                        assert "relationships" in context
                        assert "recent_summaries" in context
                        assert "location" in context
                        assert context["location"] is None

    async def test_build_context_with_character_names(
        self,
        fake_neo4j,
        mock_character_queries,
        mock_world_queries,
        mock_chapter_queries,
        sample_character_profile,
    ):
        """Test building context with specific character names."""
        mock_character_queries.get_character_profile_by_name = AsyncMock(return_value=sample_character_profile)

        with patch("core.langgraph.graph_context.character_queries", mock_character_queries):
            with patch("core.langgraph.graph_context.world_queries", mock_world_queries):
                with patch("core.langgraph.graph_context.chapter_queries", mock_chapter_queries):
                    with patch("core.langgraph.graph_context.neo4j_manager", fake_neo4j):
                        context = await build_context_from_graph(
                            current_chapter=5,
                            active_character_names=["Test Hero"],
                        )

                        assert len(context["characters"]) > 0

    async def test_build_context_with_location(
        self,
        fake_neo4j,
        mock_character_queries,
        mock_world_queries,
        mock_chapter_queries,
    ):
        """Test building context with location."""
        fake_neo4j.configure_response(
            r"loc_id",
            [
                {
                    "id": "loc_001",
                    "name": "Castle",
                    "description": "A grand castle",
                    "rules": ["No magic"],
                    "category": "structure",
                }
            ],
        )

        with patch("core.langgraph.graph_context.character_queries", mock_character_queries):
            with patch("core.langgraph.graph_context.world_queries", mock_world_queries):
                with patch("core.langgraph.graph_context.chapter_queries", mock_chapter_queries):
                    with patch("core.langgraph.graph_context.neo4j_manager", fake_neo4j):
                        context = await build_context_from_graph(
                            current_chapter=5,
                            location_id="loc_001",
                        )

                        assert context["location"] is not None
                        assert context["location"]["name"] == "Castle"

    async def test_build_context_handles_errors(self, mock_character_queries, mock_world_queries, mock_chapter_queries):
        """Test that context building handles errors gracefully."""
        # Mock character queries to raise exception
        mock_character_queries.get_characters_for_chapter_context_native.side_effect = Exception("Query error")

        with patch("core.langgraph.graph_context.character_queries", mock_character_queries):
            with patch("core.langgraph.graph_context.world_queries", mock_world_queries):
                with patch("core.langgraph.graph_context.chapter_queries", mock_chapter_queries):
                    context = await build_context_from_graph(current_chapter=5)

                    # Should return empty context structure
                    assert "characters" in context
                    assert "world_items" in context
                    assert context["characters"] == []  # Empty due to error


@pytest.mark.asyncio
class TestGetCharactersByNames:
    """Tests for _get_characters_by_names helper function."""

    async def test_get_existing_characters(self, mock_character_queries, sample_character_profile):
        """Test getting characters that exist."""
        mock_character_queries.get_character_profile_by_name = AsyncMock(return_value=sample_character_profile)

        with patch("core.langgraph.graph_context.character_queries", mock_character_queries):
            characters = await _get_characters_by_names(["Test Hero"])

            assert len(characters) == 1
            assert characters[0].name == "Test Hero"

    async def test_get_nonexistent_characters(self, mock_character_queries):
        """Test getting characters that don't exist."""
        mock_character_queries.get_character_profile_by_name = AsyncMock(return_value=None)

        with patch("core.langgraph.graph_context.character_queries", mock_character_queries):
            characters = await _get_characters_by_names(["Nonexistent"])

            assert len(characters) == 0

    async def test_get_empty_list(self, mock_character_queries):
        """Test with empty character name list."""
        with patch("core.langgraph.graph_context.character_queries", mock_character_queries):
            characters = await _get_characters_by_names([])

            assert len(characters) == 0


@pytest.mark.asyncio
class TestGetCharacterRelationships:
    """Tests for _get_character_relationships helper function."""

    async def test_get_relationships(self, fake_neo4j):
        """Test getting character relationships."""
        fake_neo4j.configure_response(
            r"MATCH.*Character.*Character",
            [
                {
                    "source": "Alice",
                    "target": "Bob",
                    "rel_type": "FRIEND_OF",
                    "description": "Close friends",
                    "confidence": 0.9,
                }
            ],
        )

        with patch("core.langgraph.graph_context.neo4j_manager", fake_neo4j):
            relationships = await _get_character_relationships(["Alice", "Bob"])

            assert len(relationships) == 1
            assert relationships[0]["source"] == "Alice"
            assert relationships[0]["target"] == "Bob"

    async def test_get_relationships_empty_list(self):
        """Test with empty character list."""
        relationships = await _get_character_relationships([])
        assert relationships == []


@pytest.mark.asyncio
class TestGetRecentSummaries:
    """Tests for _get_recent_summaries helper function."""

    async def test_get_recent_summaries(self, mock_chapter_queries):
        """Test getting recent chapter summaries."""
        mock_chapter_queries.get_chapter_content_batch_native.return_value = {
            3: {"summary": "Chapter 3 summary", "text": "...", "is_provisional": False},
            4: {"summary": "Chapter 4 summary", "text": "...", "is_provisional": False},
        }

        with patch("core.langgraph.graph_context.chapter_queries", mock_chapter_queries):
            summaries = await _get_recent_summaries(current_chapter=5, lookback_chapters=3)

            assert len(summaries) == 2
            assert summaries[0]["summary"] == "Chapter 3 summary"

    async def test_get_summaries_with_no_lookback(self, mock_chapter_queries):
        """Test with no lookback (current chapter = 1)."""
        with patch("core.langgraph.graph_context.chapter_queries", mock_chapter_queries):
            summaries = await _get_recent_summaries(current_chapter=1, lookback_chapters=5)

            assert summaries == []


@pytest.mark.asyncio
class TestGetLocationDetails:
    """Tests for _get_location_details helper function."""

    async def test_get_existing_location(self, fake_neo4j):
        """Test getting an existing location."""
        fake_neo4j.configure_response(
            r"loc_id",
            [
                {
                    "id": "loc_001",
                    "name": "Castle",
                    "description": "A grand castle",
                    "rules": ["No magic"],
                    "category": "structure",
                }
            ],
        )

        with patch("core.langgraph.graph_context.neo4j_manager", fake_neo4j):
            location = await _get_location_details("loc_001")

            assert location is not None
            assert location["name"] == "Castle"
            assert location["id"] == "loc_001"

    async def test_get_nonexistent_location(self, fake_neo4j):
        """Test getting a nonexistent location."""
        with patch("core.langgraph.graph_context.neo4j_manager", fake_neo4j):
            location = await _get_location_details("nonexistent")

            assert location is None


@pytest.mark.asyncio
class TestGetKeyEvents:
    """Tests for get_key_events function."""

    async def test_get_key_events(self, fake_neo4j):
        """Test getting key events."""
        fake_neo4j.configure_response(
            r"MATCH.*Event",
            [
                {
                    "description": "Battle at the bridge",
                    "importance": 0.9,
                    "chapter": 3,
                },
                {
                    "description": "Discovery of the artifact",
                    "importance": 0.8,
                    "chapter": 4,
                },
            ],
        )

        with patch("core.langgraph.graph_context.neo4j_manager", fake_neo4j):
            events = await get_key_events(current_chapter=5)

            assert len(events) == 2
            assert events[0]["description"] == "Battle at the bridge"
            assert events[0]["importance"] == 0.9

    async def test_get_events_with_custom_params(self, fake_neo4j):
        """Test getting events with custom parameters."""
        with patch("core.langgraph.graph_context.neo4j_manager", fake_neo4j):
            events = await get_key_events(
                current_chapter=10,
                lookback_chapters=5,
                max_events=10,
            )

            assert events == []

    async def test_get_events_handles_errors(self, mock_neo4j_manager):
        """Test that get_key_events handles errors gracefully."""
        mock_neo4j_manager.execute_read_query.side_effect = Exception("Query error")

        with patch("core.langgraph.graph_context.neo4j_manager", mock_neo4j_manager):
            events = await get_key_events(current_chapter=5)

            assert events == []
