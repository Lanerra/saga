import asyncio

import pytest

from agents.knowledge_agent import KnowledgeAgent
from kg_maintainer.models import CharacterProfile


class DummyLLM:
    async def async_call_llm(self, *args, **kwargs):
        return (
            '{"character_updates": {"Alice": {"traits": ["brave"], "development_in_chapter_1": "Did stuff"}}, "world_updates": {}, "kg_triples": ["Alice | visited | Town"]}',
            {"total_tokens": 10},
        )


llm_service_mock = DummyLLM()


def test_extract_and_merge(monkeypatch):
    agent = KnowledgeAgent()
    monkeypatch.setattr(
        agent,
        "_llm_extract_updates",
        lambda props, text, num: llm_service_mock.async_call_llm(),
    )
    monkeypatch.setattr(
        agent, "persist_profiles", lambda profiles, chapter: asyncio.sleep(0)
    )
    monkeypatch.setattr(agent, "persist_world", lambda world, chapter: asyncio.sleep(0))
    monkeypatch.setattr(
        "data_access.kg_queries.add_kg_triples_batch_to_db",
        lambda triples: asyncio.sleep(0),
    )

    plot_outline = {}
    character_profiles = {"Alice": CharacterProfile(name="Alice", description="Old")}
    world_building = {}

    usage = asyncio.run(
        agent.extract_and_merge_knowledge(
            plot_outline,
            character_profiles,
            world_building,
            1,
            "text",
        )
    )
    assert usage == {"total_tokens": 10}
    assert character_profiles["Alice"].traits == ["brave"]


@pytest.mark.asyncio
async def test_summarize_chapter_json(monkeypatch):
    agent = KnowledgeAgent()

    async def _fake_llm(*args, **kwargs):
        return '{"summary": "Short"}', {"prompt_tokens": 1}

    monkeypatch.setattr(
        "agents.knowledge_agent.llm_service.async_call_llm",
        _fake_llm,
    )

    summary, usage = await agent.summarize_chapter("x" * 6000, 1)
    assert summary == "Short"
    assert usage == {"prompt_tokens": 1}
