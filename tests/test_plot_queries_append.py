# tests/test_plot_queries_append.py
from unittest.mock import AsyncMock

import pytest

import config
from data_access import plot_queries


@pytest.mark.asyncio
async def test_append_plot_point_single_write_query_and_atomic_id(monkeypatch):
    """
    Append must:
    - use a single write query (no pre-read for max(sequence))
    - compute the next seq/id inside that write query
    """
    write_calls: list[tuple[str, dict]] = []

    async def fake_write(query, params=None):
        write_calls.append((query, params or {}))

        # Validate concurrency strategy via cypher text (single-query atomic increment)
        assert "SET ni.last_plot_point_seq" in query
        assert "coalesce(ni.last_plot_point_seq, 0) + 1" in query
        assert "RETURN pp_id AS id" in query

        # Validate expected params
        assert (params or {}).get("novel_id") == config.MAIN_NOVEL_INFO_NODE_ID
        assert (params or {}).get("desc") == "New"
        assert (params or {}).get("prev_id") == f"pp_{config.MAIN_NOVEL_INFO_NODE_ID}_2"

        # Simulate DB returning the created id
        return [{"id": f"pp_{config.MAIN_NOVEL_INFO_NODE_ID}_3"}]

    monkeypatch.setattr(
        plot_queries.neo4j_manager,
        "execute_write_query",
        AsyncMock(side_effect=fake_write),
    )
    # Ensure the implementation does not use these older multi-call paths
    monkeypatch.setattr(
        plot_queries.neo4j_manager,
        "execute_read_query",
        AsyncMock(side_effect=AssertionError("append_plot_point must not read max(sequence)")),
    )
    monkeypatch.setattr(
        plot_queries.neo4j_manager,
        "execute_cypher_batch",
        AsyncMock(side_effect=AssertionError("append_plot_point must not batch for id assignment")),
    )

    new_id = await plot_queries.append_plot_point(
        "New", f"pp_{config.MAIN_NOVEL_INFO_NODE_ID}_2"
    )

    assert new_id == f"pp_{config.MAIN_NOVEL_INFO_NODE_ID}_3"
    assert len(write_calls) == 1, "append_plot_point must issue exactly one write query"
