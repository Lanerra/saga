# tests/test_similarity_segments.py
import numpy as np
import pytest

from utils import similarity


@pytest.mark.asyncio
async def test_find_semantically_closest_segment_basic(monkeypatch):
    embeddings = {
        "query": np.array([1.0, 0.0], dtype=np.float32),
        "aaa": np.array([0.0, 1.0], dtype=np.float32),
        "bbb": np.array([0.5, 0.5], dtype=np.float32),
    }

    async def fake_embed(text: str):
        return embeddings[text]

    monkeypatch.setattr(similarity.llm_service, "async_get_embedding", fake_embed)
    monkeypatch.setattr(
        similarity,
        "get_text_segments",
        lambda doc, segment_type: [("aaa", 0, 3), ("bbb", 4, 7)],
    )

    result = await similarity.find_semantically_closest_segment("aaa bbb", "query")
    assert result == (4, 7, pytest.approx(0.707106, rel=1e-4))


@pytest.mark.asyncio
async def test_find_semantically_closest_segment_no_segments(monkeypatch):
    async def fake_embed(text: str):
        return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr(similarity.llm_service, "async_get_embedding", fake_embed)
    monkeypatch.setattr(similarity, "get_text_segments", lambda doc, st: [])

    result = await similarity.find_semantically_closest_segment("doc", "query")
    assert result is None


@pytest.mark.asyncio
async def test_find_semantically_closest_segment_query_embedding_none(monkeypatch):
    async def fake_embed(text: str):
        return None

    monkeypatch.setattr(similarity.llm_service, "async_get_embedding", fake_embed)
    monkeypatch.setattr(similarity, "get_text_segments", lambda doc, st: [("x", 0, 1)])

    result = await similarity.find_semantically_closest_segment("doc", "query")
    assert result is None
