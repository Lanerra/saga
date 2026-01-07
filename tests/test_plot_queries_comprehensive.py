# tests/test_plot_queries_comprehensive.py
"""Comprehensive tests for data_access/plot_queries.py"""

from unittest.mock import AsyncMock

import pytest

import config
from data_access import plot_queries


@pytest.mark.asyncio
class TestSavePlotOutline:
    """Tests for saving plot outline."""

    async def test_save_plot_outline_basic(self, monkeypatch):
        """Test saving basic plot outline.

        Ensures structured/list input like `acts` is not silently ignored; it must be
        persisted (round-trippable) via a *_json NovelInfo property.
        """
        executed: list[tuple[str, dict]] = []

        async def fake_batch(statements):
            executed.extend(statements)

        monkeypatch.setattr(
            plot_queries.neo4j_manager,
            "execute_cypher_batch",
            AsyncMock(side_effect=fake_batch),
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
        assert executed

        # First statement is NovelInfo upsert with props
        cypher, params = executed[0]
        assert "MERGE (ni:NovelInfo" in cypher

        # Contract: primitive keys are persisted directly, structured keys are persisted via *_json.
        assert "SET ni += $primitive_props" in cypher
        assert "structured_props" in params
        assert "primitive_props" in params

        primitive_props = params["primitive_props"]
        structured_props = params["structured_props"]

        assert primitive_props.get("title") == "Test Novel"
        assert primitive_props.get("genre") == "Fantasy"

        assert "acts" in structured_props
        assert structured_props["acts"] == [{"act_number": 1, "description": "Act 1"}]

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

        cypher, params = executed[0]

        # Contract: additive merge for primitives (no destructive `SET ni = ...`).
        assert "SET ni += $primitive_props" in cypher
        assert "SET ni = $primitive_props" not in cypher

        # Contract: structured data is persisted via a derived map of *_json keys.
        assert "apoc.map.fromPairs" in cypher
        assert "SET ni += json_props" in cypher

        assert params["primitive_props"]["title"] == "Test Novel"
        assert params["primitive_props"]["genre"] == "Fantasy"

    async def test_save_plot_outline_invalid_plot_points_still_saves_novelinfo(self, monkeypatch):
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
        assert "SET ni += $primitive_props" in cypher

    async def test_save_plot_outline_empty(self, monkeypatch):
        """Test saving empty plot outline."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(plot_queries.neo4j_manager, "execute_cypher_batch", mock_execute)

        result = await plot_queries.save_plot_outline_to_db({})
        assert result is True

    async def test_save_plot_outline_with_chapters(self, monkeypatch):
        """Test saving plot outline with chapter details.

        Ensures nested structures are persisted (via acts_json) instead of being dropped.
        """
        executed: list[tuple[str, dict]] = []

        async def fake_batch(statements):
            executed.extend(statements)

        monkeypatch.setattr(
            plot_queries.neo4j_manager,
            "execute_cypher_batch",
            AsyncMock(side_effect=fake_batch),
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
        assert executed

        cypher, params = executed[0]
        assert "MERGE (ni:NovelInfo" in cypher

        structured_props = params["structured_props"]
        assert "acts" in structured_props

        decoded = structured_props["acts"]
        assert decoded[0]["chapters"][0]["chapter_number"] == 1


@pytest.mark.asyncio
class TestGetPlotOutline:
    """Tests for getting plot outline."""

    async def test_get_plot_outline_found(self, monkeypatch):
        """Test getting plot outline when found.

        Plot points should be returned as structured dicts (not just descriptions),
        and *_json NovelInfo fields should round-trip back to structured keys.
        """

        async def fake_read(query, params=None):
            # NovelInfo query returns `plot_data` (merged primitives + decoded *_json fields).
            if "RETURN apoc.map.merge(primitives, decoded) AS plot_data" in query:
                return [
                    {
                        "plot_data": {
                            "title": "Test Novel",
                            "genre": "Fantasy",
                            "acts": [{"act_number": 1, "description": "Act 1"}],
                        }
                    }
                ]
            if "RETURN pp" in query:
                return [
                    {
                        "pp": {
                            "id": f"pp_{config.MAIN_NOVEL_INFO_NODE_ID}_1",
                            "sequence": 1,
                            "description": "Plot Point 1",
                            "status": "pending",
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
        assert result.get("acts") == [{"act_number": 1, "description": "Act 1"}]

        plot_points = result.get("plot_points")
        assert isinstance(plot_points, list)
        assert plot_points and isinstance(plot_points[0], dict)
        assert plot_points[0]["description"] == "Plot Point 1"
        assert plot_points[0]["sequence"] == 1

    async def test_get_plot_outline_empty(self, monkeypatch):
        """Test getting plot outline when empty."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(plot_queries.neo4j_manager, "execute_read_query", mock_read)

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
        """Test checking plot point that exists (novel-scoped)."""

        async def fake_read(query, params=None):
            assert "MATCH (ni:NovelInfo" in query
            assert (params or {}).get("novel_id") == config.MAIN_NOVEL_INFO_NODE_ID
            assert (params or {}).get("desc") == "Test description"
            return [{"cnt": 1}]

        monkeypatch.setattr(
            plot_queries.neo4j_manager,
            "execute_read_query",
            AsyncMock(side_effect=fake_read),
        )

        result = await plot_queries.plot_point_exists("Test description")
        assert result is True

    async def test_plot_point_exists_false(self, monkeypatch):
        """Test checking plot point that doesn't exist (novel-scoped)."""

        async def fake_read(query, params=None):
            assert "MATCH (ni:NovelInfo" in query
            assert (params or {}).get("novel_id") == config.MAIN_NOVEL_INFO_NODE_ID
            return [{"cnt": 0}]

        monkeypatch.setattr(
            plot_queries.neo4j_manager,
            "execute_read_query",
            AsyncMock(side_effect=fake_read),
        )

        result = await plot_queries.plot_point_exists("Test description")
        assert result is False

    async def test_plot_point_exists_empty_result(self, monkeypatch):
        """Test checking plot point with empty result (novel-scoped)."""

        async def fake_read(query, params=None):
            assert "MATCH (ni:NovelInfo" in query
            assert (params or {}).get("novel_id") == config.MAIN_NOVEL_INFO_NODE_ID
            return []

        monkeypatch.setattr(
            plot_queries.neo4j_manager,
            "execute_read_query",
            AsyncMock(side_effect=fake_read),
        )

        result = await plot_queries.plot_point_exists("Test description")
        assert result is False


@pytest.mark.asyncio
class TestGetLastPlotPointId:
    """Tests for getting last plot point ID."""

    async def test_get_last_plot_point_id_found(self, monkeypatch):
        """Test getting last plot point ID when found (novel-scoped)."""

        async def fake_read(query, params=None):
            assert "MATCH (ni:NovelInfo" in query
            assert (params or {}).get("novel_id") == config.MAIN_NOVEL_INFO_NODE_ID
            return [{"id": "pp_novel_5"}]

        monkeypatch.setattr(
            plot_queries.neo4j_manager,
            "execute_read_query",
            AsyncMock(side_effect=fake_read),
        )

        result = await plot_queries.get_last_plot_point_id()
        assert result == "pp_novel_5"

    async def test_get_last_plot_point_id_none(self, monkeypatch):
        """Test getting last plot point ID when none exist (novel-scoped)."""

        async def fake_read(query, params=None):
            assert "MATCH (ni:NovelInfo" in query
            assert (params or {}).get("novel_id") == config.MAIN_NOVEL_INFO_NODE_ID
            return []

        monkeypatch.setattr(
            plot_queries.neo4j_manager,
            "execute_read_query",
            AsyncMock(side_effect=fake_read),
        )

        result = await plot_queries.get_last_plot_point_id()
        assert result is None

    async def test_get_last_plot_point_id_null(self, monkeypatch):
        """Test getting last plot point ID when null (novel-scoped)."""

        async def fake_read(query, params=None):
            assert "MATCH (ni:NovelInfo" in query
            assert (params or {}).get("novel_id") == config.MAIN_NOVEL_INFO_NODE_ID
            return [{"id": None}]

        monkeypatch.setattr(
            plot_queries.neo4j_manager,
            "execute_read_query",
            AsyncMock(side_effect=fake_read),
        )

        result = await plot_queries.get_last_plot_point_id()
        assert result is None


def test_plot_queries_catch_specific_exceptions():
    """Verify plot_queries catches specific exceptions, not Exception."""
    import inspect

    from data_access import plot_queries

    source = inspect.getsource(plot_queries)

    assert "except Exception" not in source, "Found broad 'except Exception' handlers"
