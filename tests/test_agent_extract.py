# tests/test_agent_extract.py
import asyncio

import pytest

from agents.knowledge_agent import KnowledgeAgent
from models import CharacterProfile, WorldItem


class DummyLLM:
    async def async_call_llm(self, *args, **kwargs):
        return (
            '{"character_updates": {"Alice": {"traits": ["brave"], "development_in_chapter_1": "Did stuff"}}, "world_updates": {}, "kg_triples": ["Alice | visited | Town"]}',
            {"total_tokens": 10},
        )


llm_service_mock = DummyLLM()


def test_extract_and_merge(monkeypatch):
    agent = KnowledgeAgent()

    # Patch LLM extraction to return controlled JSON and usage
    monkeypatch.setattr(
        agent,
        "_llm_extract_updates",
        lambda props, text, num: llm_service_mock.async_call_llm(),
    )

    # Patch KG persistence to avoid DB access in unit test
    async def _noop_persist_entities(characters, world_items, chapter_number):
        return None

    monkeypatch.setattr(
        "agents.knowledge_agent.knowledge_graph_service.persist_entities",
        _noop_persist_entities,
    )

    # Patch triple persistence to a no-op
    monkeypatch.setattr(
        "data_access.kg_queries.add_kg_triples_batch_to_db",
        lambda triples, chapter_number, is_from_flawed: asyncio.sleep(0),
    )

    plot_outline = {}
    # Updated API: pass lists of models
    characters = [CharacterProfile(name="Alice", description="Old")]
    world_items: list[WorldItem] = []

    usage = asyncio.run(
        agent.extract_and_merge_knowledge(
            plot_outline,
            characters,
            world_items,
            1,
            "text",
        )
    )
    assert usage == {"total_tokens": 10}
    # Traits should be updated in-place for the corresponding character
    assert any(c.name == "Alice" and c.traits == ["brave"] for c in characters)


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
