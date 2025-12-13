# tests/test_llm_interface_content_extraction.py
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
import pytest

import config
from core.llm_interface_refactored import CompletionService, EmbeddingService


class _DummyCompletionClient:
    def __init__(self) -> None:
        self.called = False

    async def get_completion(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.called = True
        # Simulate provider that returns reasoning_content instead of content
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "qwen3-4b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "Reasoned output JSON",
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }

    async def get_streaming_completion(self, *args: Any, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        # Yield two chunks that stream reasoning_content in the delta
        async def _gen() -> AsyncGenerator[dict[str, Any], None]:
            yield {"choices": [{"delta": {"reasoning_content": "Hello "}, "finish_reason": None}]}
            yield {"choices": [{"delta": {"reasoning_content": "World"}, "finish_reason": None}]}

        # Make this an async generator function by yielding from the inner generator
        async for item in _gen():
            yield item


class _DummyTextProcessor:
    class _Cleaner:
        def clean_response(self, text: str) -> str:
            return text.strip()

    def __init__(self) -> None:
        self.response_cleaner = self._Cleaner()

    def get_combined_statistics(self) -> dict[str, Any]:
        return {}


@pytest.mark.asyncio
async def test_extracts_reasoning_content_when_missing_content() -> None:
    svc = CompletionService(_DummyCompletionClient(), _DummyTextProcessor())  # type: ignore
    text, usage = await svc.get_completion("model", "prompt")
    assert text == "Reasoned output JSON"
    assert usage and usage.get("total_tokens") == 3


@pytest.mark.asyncio
async def test_streaming_delta_reasoning_content_accumulates() -> None:
    # Streaming API removed; skip if method not available
    if not hasattr(CompletionService, "get_streaming_completion"):
        pytest.skip("Streaming completion is not supported in the current API")
    svc = CompletionService(_DummyCompletionClient(), _DummyTextProcessor())  # type: ignore
    text, usage = await svc.get_streaming_completion("model", "prompt")  # type: ignore
    assert text == "Hello World"


def test_embedding_fallback_accepts_numeric_list_under_non_embedding_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Exercise the fallback branch in EmbeddingService._extract_and_validate_embedding():

    - Response does NOT include "embedding" key
    - Response includes a numeric list under some other key
    - Should validate and return a numpy array without raising TypeError
    """
    # Keep the test fast and deterministic by shrinking the expected dim.
    monkeypatch.setattr(config, "EXPECTED_EMBEDDING_DIM", 3)
    monkeypatch.setattr(config, "EMBEDDING_DTYPE", np.float32)

    class _DummyEmbeddingClient:
        pass

    svc = EmbeddingService(_DummyEmbeddingClient())  # type: ignore[arg-type]

    response = {"vector": [1, 2.5, 3]}
    embedding = svc._extract_and_validate_embedding(response)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (3,)
    assert embedding.dtype == np.float32
