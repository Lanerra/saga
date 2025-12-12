"""Comprehensive tests for data_access/plot_queries.py"""
from unittest.mock import AsyncMock

import pytest

from data_access import plot_queries


@pytest.mark.asyncio
class TestSavePlotOutline:
    """Tests for saving plot outline."""

    async def test_save_plot_outline_basic(self, monkeypatch):
        """Test saving basic plot outline."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(
            plot_queries.neo4j_manager, "execute_cypher_batch", mock_execute
        )

        plot_data = {
            "title": "Test Novel",
            "genre": "Fantasy",
            "acts": [
                {"act_number": 1, "description": "Act 1"},
            ],
        }

        result = await plot_queries.save_plot_outline_to_db(plot_data)
        assert result is True
        mock_execute.assert_called()

    async def test_save_plot_outline_does_not_replace_novelinfo_map(self, monkeypatch):
        """
        Guard against NovelInfo property erasure.

        Regression test for the old `SET ni = $props` behavior; we must use an additive
        merge (`SET ni += $props`) so unrelated NovelInfo keys are preserved.
        """
        executed: list[tuple[str, dict]] = []

        async def fake_batch(statements):
            executed.extend(statements)

        monkeypatch.setattr(
            plot_queries.neo4j_manager,
            "execute_cypher_batch",
            AsyncMock(side_effect=fake_batch),
        )

        result = await plot_queries.save_plot_outline_to_db(
            {
                "title": "Test Novel",
                "genre": "Fantasy",
                # Explicitly omit plot_points to keep this test focused on NovelInfo update.
            }
        )
        assert result is True
        assert executed, "Expected at least the NovelInfo upsert statement to be executed"

        cypher = executed[0][0]
        assert "SET ni += $props" in cypher
        assert "SET ni = $props" not in cypher

    async def test_save_plot_outline_invalid_plot_points_still_saves_novelinfo(
        self, monkeypatch
    ):
        """
        When plot_points is invalid, we should skip plot point sync, but still persist
        NovelInfo properties (no early-return success without writes).
        """
        executed: list[tuple[str, dict]] = []

        async def fake_batch(statements):
            executed.extend(statements)

        monkeypatch.setattr(
            plot_queries.neo4j_manager,
            "execute_cypher_batch",
            AsyncMock(side_effect=fake_batch),
        )

        result = await plot_queries.save_plot_outline_to_db(
            {
                "title": "Test Novel",
                "genre": "Fantasy",
                "plot_points": {"not": "a list"},
            }
        )
        assert result is True
        assert executed, "Expected NovelInfo upsert to run even when plot_points is invalid"

        cypher = executed[0][0]
        assert "SET ni += $props" in cypher

    async def test_save_plot_outline_empty(self, monkeypatch):
        """Test saving empty plot outline."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(
            plot_queries.neo4j_manager, "execute_cypher_batch", mock_execute
        )

        result = await plot_queries.save_plot_outline_to_db({})
        assert result is True

    async def test_save_plot_outline_with_chapters(self, monkeypatch):
        """Test saving plot outline with chapter details."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(
            plot_queries.neo4j_manager, "execute_cypher_batch", mock_execute
        )

        plot_data = {
            "title": "Test Novel",
            "acts": [
                {
                    "act_number": 1,
                    "description": "Act 1",
                    "chapters": [
                        {"chapter_number": 1, "description": "Chapter 1"},
                        {"chapter_number": 2, "description": "Chapter 2"},
                    ],
                },
            ],
        }

        result = await plot_queries.save_plot_outline_to_db(plot_data)
        assert result is True


@pytest.mark.asyncio
class TestGetPlotOutline:
    """Tests for getting plot outline."""

    async def test_get_plot_outline_found(self, monkeypatch):
        """Test getting plot outline when found."""
        async def fake_read(query, params=None):
            if "RETURN ni" in query:
                return [
                    {
                        "ni": {
                            "title": "Test Novel",
                            "genre": "Fantasy",
                        }
                    }
                ]
            return []

        monkeypatch.setattr(
            plot_queries.neo4j_manager,
            "execute_read_query",
            AsyncMock(side_effect=fake_read),
        )

        result = await plot_queries.get_plot_outline_from_db()
        assert isinstance(result, dict)
        assert result.get("title") == "Test Novel"

    async def test_get_plot_outline_empty(self, monkeypatch):
        """Test getting plot outline when empty."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            plot_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await plot_queries.get_plot_outline_from_db()
        assert isinstance(result, dict)
        assert len(result) == 0

    async def test_get_plot_outline_with_acts(self, monkeypatch):
        """Test getting plot outline with acts."""
        async def fake_read(query, params=None):
            if "MATCH (ni:NovelInfo)" in query and "RETURN ni" in query:
                return [
                    {
                        "title": "Test Novel",
                        "genre": "Fantasy",
                    }
                ]
            if "MATCH (act:Act)" in query:
                return [
                    {"act_number": 1, "description": "Act 1"},
                    {"act_number": 2, "description": "Act 2"},
                ]
            return []

        monkeypatch.setattr(
            plot_queries.neo4j_manager,
            "execute_read_query",
            AsyncMock(side_effect=fake_read),
        )

        result = await plot_queries.get_plot_outline_from_db()
        assert isinstance(result, dict)


@pytest.mark.asyncio
class TestPlotPointExists:
    """Tests for checking if plot point exists."""

    async def test_plot_point_exists_true(self, monkeypatch):
        """Test checking plot point that exists."""
        mock_read = AsyncMock(return_value=[{"cnt": 1}])
        monkeypatch.setattr(
            plot_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await plot_queries.plot_point_exists("Test description")
        assert result is True

    async def test_plot_point_exists_false(self, monkeypatch):
        """Test checking plot point that doesn't exist."""
        mock_read = AsyncMock(return_value=[{"cnt": 0}])
        monkeypatch.setattr(
            plot_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await plot_queries.plot_point_exists("Test description")
        assert result is False

    async def test_plot_point_exists_empty_result(self, monkeypatch):
        """Test checking plot point with empty result."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            plot_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await plot_queries.plot_point_exists("Test description")
        assert result is False


@pytest.mark.asyncio
class TestGetLastPlotPointId:
    """Tests for getting last plot point ID."""

    async def test_get_last_plot_point_id_found(self, monkeypatch):
        """Test getting last plot point ID when found."""
        mock_read = AsyncMock(return_value=[{"id": "pp_novel_5"}])
        monkeypatch.setattr(
            plot_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await plot_queries.get_last_plot_point_id()
        assert result == "pp_novel_5"

    async def test_get_last_plot_point_id_none(self, monkeypatch):
        """Test getting last plot point ID when none exist."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            plot_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await plot_queries.get_last_plot_point_id()
        assert result is None

    async def test_get_last_plot_point_id_null(self, monkeypatch):
        """Test getting last plot point ID when null."""
        mock_read = AsyncMock(return_value=[{"id": None}])
        monkeypatch.setattr(
            plot_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await plot_queries.get_last_plot_point_id()
        assert result is None
