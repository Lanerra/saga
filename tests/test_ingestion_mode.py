# tests/test_ingestion_mode.py
import pytest

from agents.knowledge_agent import KnowledgeAgent
from agents.narrative_agent import NarrativeAgent
from core.llm_interface_refactored import llm_service
from utils.ingestion_utils import split_text_into_chapters


def test_split_text_into_chapters_basic():
    text = "p1\n\npara2 longer\n\npara3\n\npara4"
    chapters = split_text_into_chapters(text, max_chars=20)
    assert chapters == ["p1\n\npara2 longer", "para3\n\npara4"]


@pytest.mark.asyncio
async def test_ingest_and_finalize_chunk_delegates(monkeypatch):
    agent = KnowledgeAgent()

    async def fake_finalize(*args, **kwargs):
        return {"summary": "done"}

    monkeypatch.setattr(agent, "extract_and_merge_knowledge", fake_finalize)
    result = await agent.ingest_and_finalize_chunk({}, {}, {}, 1, "text")
    assert result["summary"] == "done"


@pytest.mark.asyncio
async def test_plan_continuation_parses(monkeypatch):
    agent = NarrativeAgent()

    async def fake_llm(*_a, **_k):
        return '["a", "b"]', {"total_tokens": 1}

    monkeypatch.setattr(llm_service, "async_call_llm", fake_llm)
    points, usage = await agent.plan_continuation("sum", 2)
    assert points == ["a", "b"]
    assert usage == {"total_tokens": 1}
