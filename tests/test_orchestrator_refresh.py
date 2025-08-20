from unittest.mock import AsyncMock

import pytest

import utils
from data_access import plot_queries
from orchestration.nana_orchestrator import NANA_Orchestrator


@pytest.fixture
def orchestrator(monkeypatch):
    monkeypatch.setattr(utils, "load_spacy_model_if_needed", lambda: None)
    orch = NANA_Orchestrator()
    monkeypatch.setattr(orch, "_update_rich_display", lambda *a, **k: None)
    return orch


@pytest.mark.asyncio
async def test_refresh_plot_outline(orchestrator, monkeypatch):
    monkeypatch.setattr(
        plot_queries,
        "get_plot_outline_from_db",
        AsyncMock(return_value={"plot_points": ["a", "b"]}),
    )
    await orchestrator.refresh_plot_outline()
    assert orchestrator.plot_outline["plot_points"] == ["a", "b"]
