import pytest

from agents.drafting_agent import DraftingAgent
from core.llm_interface import llm_service
from kg_maintainer.models import SceneDetail


@pytest.mark.asyncio
async def test_draft_chapter_whole_chapter(monkeypatch):
    agent = DraftingAgent()

    async def fake_call_llm(**kwargs):
        return "chapter text", {"total_tokens": 5}

    monkeypatch.setattr(llm_service, "async_call_llm", fake_call_llm)

    plot_outline = {"title": "Test", "genre": "F"}
    draft, raw, usage = await agent.draft_chapter(
        plot_outline,
        1,
        "focus",
        "context",
        None,
    )

    assert draft == "chapter text"
    assert raw == "chapter text"
    assert usage == {"total_tokens": 5}


@pytest.mark.asyncio
async def test_draft_chapter_scene_mode(monkeypatch):
    agent = DraftingAgent()
    responses = [
        ("scene1", {"prompt_tokens": 1}),
        ("scene2", {"prompt_tokens": 1}),
    ]

    async def fake_call_llm(**kwargs):
        return responses.pop(0)

    monkeypatch.setattr(llm_service, "async_call_llm", fake_call_llm)

    scenes = [
        SceneDetail(scene_number=1, summary="a"),
        SceneDetail(scene_number=2, summary="b"),
    ]

    draft, raw, usage = await agent.draft_chapter(
        {"title": "Test", "genre": "F"},
        2,
        "focus",
        "context",
        scenes,
    )

    assert draft == "scene1\n\nscene2"
    assert "scene1" in raw and "scene2" in raw
    assert usage["prompt_tokens"] == 2


@pytest.mark.asyncio
async def test_draft_chapter_caching(monkeypatch):
    agent = DraftingAgent()
    call_count = 0

    async def fake_call_llm(**_kwargs):
        nonlocal call_count
        call_count += 1
        return "scene", {"prompt_tokens": 1}

    monkeypatch.setattr(llm_service, "async_call_llm", fake_call_llm)

    scenes = [SceneDetail(scene_number=1, summary="a")]

    await agent.draft_chapter({"title": "T", "genre": "F"}, 3, "focus", "ctx", scenes)
    await agent.draft_chapter({"title": "T", "genre": "F"}, 3, "focus", "ctx", scenes)

    assert call_count == 1
    agent._cached_llm_call.cache_clear()
