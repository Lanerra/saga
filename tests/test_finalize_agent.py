import asyncio

import numpy as np
import pytest

# Tests for deprecated FinalizeAgent - functionality now in KnowledgeAgent
from agents.knowledge_agent import KnowledgeAgent
from models import CharacterProfile, WorldItem


class DummyKGAgent(KnowledgeAgent):
    pass


@pytest.mark.asyncio
async def test_finalize_chapter_success(monkeypatch):
    kg_agent = DummyKGAgent()
    # Tests for deprecated FinalizeAgent - functionality now in KnowledgeAgent
    # This test file is deprecated and should be removed in a future version
    agent = kg_agent  # Using KnowledgeAgent directly

    async def fake_summary(text: str, num: int):
        return "sum", {"prompt_tokens": 1}

    async def fake_embedding(text: str):
        return np.array([0.1, 0.2], dtype=np.float32)

    async def fake_extract(*_args, **_kwargs):
        return (
            '{"character_updates": {"Alice": {"description": "Hero"}}, "world_updates": {"Places": {"Town": {"description": "Nice"}}}, "kg_triples": ["A|b|c"]}',
            {"total_tokens": 2},
        )

    save_mock = asyncio.Future()
    save_mock.set_result(None)

    monkeypatch.setattr(kg_agent, "summarize_chapter", fake_summary)
    monkeypatch.setattr(
        "core.llm_interface.llm_service.async_get_embedding", fake_embedding
    )
    monkeypatch.setattr(kg_agent, "_llm_extract_updates", fake_extract)
    monkeypatch.setattr(kg_agent, "persist_profiles", lambda *a, **k: save_mock)
    monkeypatch.setattr(kg_agent, "persist_world", lambda *a, **k: save_mock)
    monkeypatch.setattr(
        "data_access.kg_queries.add_kg_triples_batch_to_db", lambda *a, **k: save_mock
    )
    monkeypatch.setattr(
        "data_access.chapter_queries.save_chapter_data_to_db", lambda *a, **k: save_mock
    )

    # Using KnowledgeAgent's extract_and_merge_knowledge instead of deprecated FinalizeAgent
    result = await agent.extract_and_merge_knowledge({}, {}, {}, 1, "text", "raw")
    assert result["summary"] == "sum"
    assert np.allclose(result["embedding"], np.array([0.1, 0.2], dtype=np.float32))
    assert result["kg_usage"] == {"total_tokens": 2}


@pytest.mark.asyncio
async def test_finalize_chapter_validation_failure(monkeypatch):
    kg_agent = DummyKGAgent()
    # Tests for deprecated FinalizeAgent - functionality now in KnowledgeAgent
    # This test file is deprecated and should be removed in a future version
    agent = kg_agent  # Using KnowledgeAgent directly

    async def fake_summary(text: str, num: int):
        return "sum", {"prompt_tokens": 1}

    async def fake_embedding(text: str):
        return np.array([0.1, 0.2], dtype=np.float32)

    async def fake_extract(*_args, **_kwargs):
        return (
            '{"character_updates": {"": {"description": "bad"}}, "world_updates": {}, "kg_triples": []}',
            {"total_tokens": 2},
        )

    save_mock = asyncio.Future()
    save_mock.set_result(None)

    monkeypatch.setattr(kg_agent, "summarize_chapter", fake_summary)
    monkeypatch.setattr(
        "core.llm_interface.llm_service.async_get_embedding", fake_embedding
    )
    monkeypatch.setattr(kg_agent, "_llm_extract_updates", fake_extract)
    profiles_called: dict[str, CharacterProfile] = {}
    world_called: dict[str, dict[str, WorldItem]] = {}

    async def persist_profiles(profiles, chapter):
        profiles_called.update(profiles)

    async def persist_world(world, chapter):
        world_called.update(world)

    monkeypatch.setattr(kg_agent, "persist_profiles", persist_profiles)
    monkeypatch.setattr(kg_agent, "persist_world", persist_world)
    monkeypatch.setattr(
        "data_access.kg_queries.add_kg_triples_batch_to_db", lambda *a, **k: save_mock
    )
    monkeypatch.setattr(
        "data_access.chapter_queries.save_chapter_data_to_db", lambda *a, **k: save_mock
    )

    # Using KnowledgeAgent's extract_and_merge_knowledge instead of deprecated FinalizeAgent
    result = await agent.extract_and_merge_knowledge({}, {}, {}, 1, "text", None)
    assert profiles_called == {}
    assert world_called == {}
    assert result["kg_usage"] == {"total_tokens": 2}
