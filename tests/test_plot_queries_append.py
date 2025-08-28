# tests/test_plot_queries_append.py
from unittest.mock import AsyncMock

import pytest

import config
from data_access import plot_queries


@pytest.mark.asyncio
async def test_append_plot_point(monkeypatch):
    async def fake_read(query, params=None):
        assert "max(pp.sequence)" in query
        return [{"max_seq": 2}]

    executed = []

    async def fake_batch(statements):
        executed.extend(statements)

    monkeypatch.setattr(
        plot_queries.neo4j_manager,
        "execute_read_query",
        AsyncMock(side_effect=fake_read),
    )
    monkeypatch.setattr(
        plot_queries.neo4j_manager,
        "execute_cypher_batch",
        AsyncMock(side_effect=fake_batch),
    )

    new_id = await plot_queries.append_plot_point(
        "New", f"pp_{config.MAIN_NOVEL_INFO_NODE_ID}_2"
    )

    assert new_id == f"pp_{config.MAIN_NOVEL_INFO_NODE_ID}_3"
    assert executed
