from unittest.mock import AsyncMock

import pytest

from data_access import plot_queries


@pytest.mark.asyncio
async def test_plot_point_exists(monkeypatch):
    monkeypatch.setattr(
        plot_queries.neo4j_manager,
        "execute_read_query",
        AsyncMock(return_value=[{"cnt": 1}]),
    )
    assert await plot_queries.plot_point_exists("a")


@pytest.mark.asyncio
async def test_get_last_plot_point_id(monkeypatch):
    monkeypatch.setattr(
        plot_queries.neo4j_manager,
        "execute_read_query",
        AsyncMock(return_value=[{"id": "pp_1"}]),
    )
    result = await plot_queries.get_last_plot_point_id()
    assert result == "pp_1"
