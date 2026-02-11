import asyncio

import httpx
import pytest

import config
from core.http_client_service import (
    CompletionHTTPClient,
    EmbeddingHTTPClient,
    HTTPClientService,
)


def _make_service(transport: httpx.MockTransport) -> HTTPClientService:
    service = HTTPClientService.__new__(HTTPClientService)
    service._client = httpx.AsyncClient(transport=transport)
    service._semaphore = asyncio.Semaphore(10)
    service.request_count = 0
    service._stats = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "retry_attempts": 0,
    }
    return service


def _make_transport(handler):
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
class TestHTTPClientService:
    async def test_successful_post_returns_response_and_increments_request_count(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "ok"})

        service = _make_service(_make_transport(handler))
        response = await service.post_json("http://example.com/api", {"key": "value"})

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert service.request_count == 1
        await service.aclose()

    async def test_statistics_track_total_and_successful_requests(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        service = _make_service(_make_transport(handler))
        await service.post_json("http://example.com/a", {})
        await service.post_json("http://example.com/b", {})

        statistics = service.get_statistics()
        assert statistics["total_requests"] == 2
        assert statistics["successful_requests"] == 2
        assert statistics["failed_requests"] == 0
        await service.aclose()

    async def test_4xx_client_error_breaks_without_retrying(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config, "LLM_RETRY_ATTEMPTS", 3)
        monkeypatch.setattr(config, "LLM_RETRY_DELAY_SECONDS", 0)

        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(400, json={"error": "bad request"})

        service = _make_service(_make_transport(handler))
        with pytest.raises(httpx.HTTPStatusError) as exception_info:
            await service.post_json("http://example.com/api", {})

        assert exception_info.value.response.status_code == 400
        assert call_count == 1
        await service.aclose()

    async def test_429_rate_limit_triggers_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config, "LLM_RETRY_ATTEMPTS", 2)
        monkeypatch.setattr(config, "LLM_RETRY_DELAY_SECONDS", 0)

        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(429, json={"error": "rate limited"})
            return httpx.Response(200, json={"ok": True})

        service = _make_service(_make_transport(handler))
        response = await service.post_json("http://example.com/api", {})

        assert response.status_code == 200
        assert call_count == 2
        await service.aclose()

    async def test_5xx_server_error_triggers_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config, "LLM_RETRY_ATTEMPTS", 2)
        monkeypatch.setattr(config, "LLM_RETRY_DELAY_SECONDS", 0)

        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(500, json={"error": "internal server error"})
            return httpx.Response(200, json={"ok": True})

        service = _make_service(_make_transport(handler))
        response = await service.post_json("http://example.com/api", {})

        assert response.status_code == 200
        assert call_count == 2
        await service.aclose()

    async def test_all_retries_exhausted_raises_last_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config, "LLM_RETRY_ATTEMPTS", 2)
        monkeypatch.setattr(config, "LLM_RETRY_DELAY_SECONDS", 0)

        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(500, json={"error": "server error"})

        service = _make_service(_make_transport(handler))
        with pytest.raises(httpx.HTTPStatusError) as exception_info:
            await service.post_json("http://example.com/api", {})

        assert exception_info.value.response.status_code == 500
        assert call_count == 2
        assert service._stats["failed_requests"] == 1
        await service.aclose()

    async def test_get_statistics_computes_success_rate(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        service = _make_service(_make_transport(handler))
        await service.post_json("http://example.com/a", {})
        await service.post_json("http://example.com/b", {})

        statistics = service.get_statistics()
        assert statistics["success_rate"] == 100.0
        assert statistics["failure_rate"] == 0.0
        assert statistics["avg_retries_per_request"] == 0.0

    async def test_get_statistics_with_zero_requests_returns_zero_rates(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        service = _make_service(_make_transport(handler))
        statistics = service.get_statistics()

        assert statistics["success_rate"] == 0
        assert statistics["failure_rate"] == 0
        assert statistics["avg_retries_per_request"] == 0
        assert statistics["total_requests"] == 0
        await service.aclose()


@pytest.mark.asyncio
class TestEmbeddingHTTPClient:
    async def test_empty_text_raises_value_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        service = _make_service(_make_transport(handler))
        client = EmbeddingHTTPClient(service)

        with pytest.raises(ValueError, match="Text must be a non-empty string"):
            await client.get_embedding("", "nomic-embed-text:latest")
        await service.aclose()

    async def test_whitespace_only_text_raises_value_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        service = _make_service(_make_transport(handler))
        client = EmbeddingHTTPClient(service)

        with pytest.raises(ValueError, match="Text must be a non-empty string"):
            await client.get_embedding("   \n\t  ", "nomic-embed-text:latest")
        await service.aclose()

    async def test_valid_text_sends_correct_payload_to_correct_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config, "EMBEDDING_API_BASE", "http://fake-embedding-host:11434")

        captured_request = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured_request["url"] = str(request.url)
            captured_request["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})

        service = _make_service(_make_transport(handler))
        client = EmbeddingHTTPClient(service)
        result = await client.get_embedding("  hello world  ", "nomic-embed-text:latest")

        assert captured_request["url"] == "http://fake-embedding-host:11434/api/embeddings"
        assert captured_request["payload"] == {"model": "nomic-embed-text:latest", "prompt": "hello world"}
        assert result == {"embedding": [0.1, 0.2, 0.3]}
        await service.aclose()


@pytest.mark.asyncio
class TestCompletionHTTPClient:
    async def test_sends_correct_payload_with_auth_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config, "OPENAI_API_BASE", "http://fake-llm-host:8080/v1")
        monkeypatch.setattr(config, "OPENAI_API_KEY", "test-api-key-123")
        monkeypatch.setattr(config, "LLM_TOP_P", 0.9)

        captured_request = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured_request["url"] = str(request.url)
            captured_request["headers"] = dict(request.headers)
            captured_request["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "response"}}]})

        service = _make_service(_make_transport(handler))
        client = CompletionHTTPClient(service)
        messages = [{"role": "user", "content": "hello"}]
        result = await client.get_completion("test-model", messages, 0.7, 1024)

        assert captured_request["url"] == "http://fake-llm-host:8080/v1/chat/completions"
        assert captured_request["headers"]["authorization"] == "Bearer test-api-key-123"
        assert captured_request["headers"]["content-type"] == "application/json"
        assert captured_request["payload"] == {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1024,
            "stream": False,
        }
        assert result == {"choices": [{"message": {"content": "response"}}]}
        await service.aclose()

    async def test_extra_kwargs_merged_into_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config, "OPENAI_API_BASE", "http://fake-llm-host:8080/v1")
        monkeypatch.setattr(config, "OPENAI_API_KEY", "key")
        monkeypatch.setattr(config, "LLM_TOP_P", 0.95)

        captured_payload = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured_payload.update(json.loads(request.content))
            return httpx.Response(200, json={"choices": []})

        service = _make_service(_make_transport(handler))
        client = CompletionHTTPClient(service)
        messages = [{"role": "user", "content": "hi"}]
        await client.get_completion(
            "test-model",
            messages,
            0.5,
            512,
            frequency_penalty=0.3,
            presence_penalty=0.1,
        )

        assert captured_payload["frequency_penalty"] == 0.3
        assert captured_payload["presence_penalty"] == 0.1
        assert captured_payload["model"] == "test-model"
        assert captured_payload["temperature"] == 0.5
        assert captured_payload["max_tokens"] == 512
        await service.aclose()
