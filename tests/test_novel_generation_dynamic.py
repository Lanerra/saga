from unittest.mock import AsyncMock

import pytest

import config
import utils
from core.db_manager import neo4j_manager
from data_access import chapter_queries
from orchestration.nana_orchestrator import NANA_Orchestrator


@pytest.mark.asyncio
async def test_dynamic_chapter_adjustment(monkeypatch):
    monkeypatch.setattr(utils, "load_spacy_model_if_needed", lambda: None)
    monkeypatch.setattr(config, "CHAPTERS_PER_RUN", 3)

    orch = NANA_Orchestrator()
    monkeypatch.setattr(orch, "_update_rich_display", lambda *a, **k: None)

    orch.plot_outline = {"title": "T", "plot_points": ["p1", "p2"]}
    orch.chapter_count = 0

    calls = []

    async def fake_run(ch):
        calls.append(ch)
        if ch == 1:
            orch.plot_outline["plot_points"].append("p3")
        orch.chapter_count = ch
        return "text"

    monkeypatch.setattr(
        orch, "run_chapter_generation_process", AsyncMock(side_effect=fake_run)
    )
    monkeypatch.setattr(neo4j_manager, "connect", AsyncMock())
    monkeypatch.setattr(neo4j_manager, "create_db_schema", AsyncMock())
    monkeypatch.setattr(neo4j_manager, "close", AsyncMock())
    monkeypatch.setattr(orch.kg_maintainer_agent, "load_schema_from_db", AsyncMock())
    monkeypatch.setattr(orch, "async_init_orchestrator", AsyncMock())
    monkeypatch.setattr(orch, "perform_initial_setup", AsyncMock(return_value=True))
    monkeypatch.setattr(orch.kg_maintainer_agent, "heal_and_enrich_kg", AsyncMock())
    monkeypatch.setattr(orch, "refresh_plot_outline", AsyncMock())
    monkeypatch.setattr(
        chapter_queries, "load_chapter_count_from_db", AsyncMock(return_value=3)
    )
    monkeypatch.setattr(orch.display, "start", lambda: None)
    monkeypatch.setattr(orch.display, "stop", AsyncMock())
    monkeypatch.setattr(orch, "_validate_critical_configs", lambda: True)

    await orch.run_novel_generation_loop()

    assert calls == [1, 2, 3]
    assert orch.chapter_count == 3
