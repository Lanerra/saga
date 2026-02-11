# tests/test_llm_interface_content_extraction.py
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
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "provider-with-content-parts",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}],
                        "reasoning_content": "this should be ignored",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }

    # NOTE:
    # Streaming was removed from the public LLM surface (CORE-006 remediation),
    # so we intentionally do not provide a get_streaming_completion() helper here.


class _DummyTextProcessor:
    class _Cleaner:
        def clean_response(self, text: str) -> str:
            return text.strip()

    def __init__(self) -> None:
        self.response_cleaner = self._Cleaner()

    def clean_text_with_spacy(self, text: str, aggressive: bool = False) -> str:
        return text

    def get_combined_statistics(self) -> dict[str, Any]:
        return {}


@pytest.mark.asyncio
async def test_extracts_text_from_list_of_parts_content_and_ignores_reasoning_content() -> None:
    svc = CompletionService(_DummyCompletionClient(), _DummyTextProcessor())  # type: ignore
    text, usage = await svc.get_completion("model", "prompt")
    assert text == "Hello world"
    assert usage and usage.get("total_tokens") == 3


@pytest.mark.asyncio
async def test_get_completion_strict_raises_on_missing_model_or_prompt() -> None:
    """
    CORE-007: CompletionService.get_completion() must not return ambiguous sentinels
    on input validation failures by default.
    """
    from core.exceptions import LLMServiceError

    svc = CompletionService(_DummyCompletionClient(), _DummyTextProcessor())  # type: ignore

    with pytest.raises(LLMServiceError):
        await svc.get_completion("", "prompt")

    with pytest.raises(LLMServiceError):
        await svc.get_completion("model", "")


@pytest.mark.asyncio
async def test_get_completion_non_strict_returns_legacy_sentinel_on_error() -> None:
    """
    CORE-007 compatibility: strict=False preserves the legacy ("", None) sentinel.
    """

    class _ExplodingClient:
        async def get_completion(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("provider down")

    svc = CompletionService(_ExplodingClient(), _DummyTextProcessor())  # type: ignore
    text, usage = await svc.get_completion("model", "prompt", strict=False)

    assert text == ""
    assert usage is None


@pytest.mark.asyncio
async def test_reasoning_content_is_not_used_when_content_is_missing() -> None:
    class _ReasoningOnlyClient:
        async def get_completion(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
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
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }

    svc = CompletionService(_ReasoningOnlyClient(), _DummyTextProcessor())  # type: ignore
    text, usage = await svc.get_completion("model", "prompt")
    assert text == ""
    assert usage and usage.get("total_tokens") == 3


def test_streaming_api_surface_removed() -> None:
    """
    CORE-006: Streaming support was removed; ensure we don't expose misleading streaming APIs.
    """
    assert not hasattr(CompletionService, "get_streaming_completion")


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


@pytest.mark.asyncio
async def test_get_embedding_truncates_to_configured_max_before_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "EMBEDDING_MAX_INPUT_TOKENS", 5)
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "dummy-embed")
    monkeypatch.setattr(config, "EXPECTED_EMBEDDING_DIM", 3)
    monkeypatch.setattr(config, "EMBEDDING_DTYPE", np.float32)

    import core.llm_interface_refactored as llm_interface_refactored

    truncate_calls: dict[str, object] = {}

    def fake_truncate_text_by_tokens(
        *,
        text: str,
        model_name: str,
        max_tokens: int,
        truncation_marker: str = "\n... (truncated)",
    ) -> str:
        truncate_calls["text"] = text
        truncate_calls["model_name"] = model_name
        truncate_calls["max_tokens"] = max_tokens
        truncate_calls["truncation_marker"] = truncation_marker
        return "TRUNCATED"

    monkeypatch.setattr(llm_interface_refactored, "truncate_text_by_tokens", fake_truncate_text_by_tokens)

    class _CapturingEmbeddingClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def get_embedding(self, text: str, model: str) -> dict[str, Any]:
            self.calls.append((text, model))
            return {"embedding": [0.1, 0.2, 0.3]}

    client = _CapturingEmbeddingClient()
    svc = EmbeddingService(client)  # type: ignore[arg-type]

    embedding = await svc.get_embedding("LONG INPUT TEXT")
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (3,)

    assert truncate_calls == {
        "text": "LONG INPUT TEXT",
        "model_name": "dummy-embed",
        "max_tokens": 5,
        "truncation_marker": "\n... (truncated)",
    }
    assert client.calls == [("TRUNCATED", "dummy-embed")]


@pytest.mark.asyncio
async def test_get_embedding_exact_limit_can_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "EMBEDDING_MAX_INPUT_TOKENS", 5)
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "dummy-embed")
    monkeypatch.setattr(config, "EXPECTED_EMBEDDING_DIM", 3)
    monkeypatch.setattr(config, "EMBEDDING_DTYPE", np.float32)

    import core.llm_interface_refactored as llm_interface_refactored

    def fake_truncate_text_by_tokens(
        *,
        text: str,
        model_name: str,
        max_tokens: int,
        truncation_marker: str = "\n... (truncated)",
    ) -> str:
        return text

    monkeypatch.setattr(llm_interface_refactored, "truncate_text_by_tokens", fake_truncate_text_by_tokens)

    class _CapturingEmbeddingClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def get_embedding(self, text: str, model: str) -> dict[str, Any]:
            self.calls.append((text, model))
            return {"embedding": [0.1, 0.2, 0.3]}

    client = _CapturingEmbeddingClient()
    svc = EmbeddingService(client)  # type: ignore[arg-type]

    await svc.get_embedding("SHORT")
    assert client.calls == [("SHORT", "dummy-embed")]


@pytest.mark.asyncio
async def test_get_embedding_empty_input_short_circuits_without_request() -> None:
    class _ExplodingEmbeddingClient:
        async def get_embedding(self, text: str, model: str) -> dict[str, Any]:
            raise AssertionError("Embedding client must not be called for empty input")

    svc = EmbeddingService(_ExplodingEmbeddingClient())  # type: ignore[arg-type]

    assert await svc.get_embedding("") is None
    assert await svc.get_embedding("   ") is None
