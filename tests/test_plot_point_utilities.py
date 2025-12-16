# tests/test_plot_point_utilities.py
from unittest.mock import AsyncMock

import pytest

import config
from data_access import plot_queries


@pytest.mark.asyncio
async def test_plot_point_exists(monkeypatch):
    async def fake_read(query, params=None):
        assert "MATCH (ni:NovelInfo" in query
        assert (params or {}).get("novel_id") == config.MAIN_NOVEL_INFO_NODE_ID
        assert (params or {}).get("desc") == "a"
        return [{"cnt": 1}]

    monkeypatch.setattr(
        plot_queries.neo4j_manager,
        "execute_read_query",
        AsyncMock(side_effect=fake_read),
    )
    assert await plot_queries.plot_point_exists("a")


@pytest.mark.asyncio
async def test_get_last_plot_point_id(monkeypatch):
    async def fake_read(query, params=None):
        assert "MATCH (ni:NovelInfo" in query
        assert (params or {}).get("novel_id") == config.MAIN_NOVEL_INFO_NODE_ID
        return [{"id": "pp_1"}]

    monkeypatch.setattr(
        plot_queries.neo4j_manager,
        "execute_read_query",
        AsyncMock(side_effect=fake_read),
    )
    result = await plot_queries.get_last_plot_point_id()
    assert result == "pp_1"
