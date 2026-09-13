"""Exercise serialized requests, retries, fallbacks and strict response admission."""
import json
from typing import Any

import httpx
import pytest

import config
from core.exceptions import LLMServiceError
from core.http_client_service import HTTPClientService
from core.llm_interface_refactored import create_llm_service


@pytest.mark.parametrize("failure", ["programming", "redirect"])
async def test_nontransient_transport_failures_do_not_retry(failure: str) -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if failure == "programming":
            raise RuntimeError("programming-canary")
        return httpx.Response(307, headers={"Location": "https://elsewhere.invalid"})

    settings = config.snapshot_settings().model_copy(update={"LLM_RETRY_ATTEMPTS": 3, "LLM_RETRY_DELAY_SECONDS": 0.001})
    client = HTTPClientService(configuration=settings, client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    try:
        with pytest.raises(RuntimeError if failure == "programming" else httpx.HTTPStatusError):
            await client.post_json("https://synthetic.invalid", {"prompt": "synthetic"})
        assert calls == 1
    finally:
        await client.aclose()


@pytest.mark.parametrize("raw", [
    '{"choices": [], "choices": [{"message": {"content": "answer"}}]}',
    '{"choices": [{"message": {"content": "first", "content": "second"}}]}',
])
async def test_duplicate_provider_envelope_keys_are_rejected(raw: str) -> None:
    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=raw, headers={"Content-Type": "application/json"})
    ))))
    try:
        with pytest.raises(LLMServiceError):
            await service.async_call_llm("synthetic", "prompt")
    finally:
        await service.aclose()


async def test_sampler_schema_and_budget_survive_http_retry_and_model_fallback() -> None:
    bodies: list[dict[str, Any]] = []
    schema = {"type": "json_schema", "json_schema": {"name": "example", "strict": False, "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}}}

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) <= 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer": "you’d know"}', "reasoning_content": "not-an-answer"}, "finish_reason": "stop"}]})

    settings = config.snapshot_settings().model_copy(update={"LLM_RETRY_ATTEMPTS": 2, "LLM_RETRY_DELAY_SECONDS": 0.001})
    service = create_llm_service(HTTPClientService(configuration=settings, client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    try:
        answer, _ = await service.async_call_llm_json_object("primary", "prompt", temperature=0.1, allow_fallback=True, response_format=schema, reject_duplicate_keys=True)
        assert answer == {"answer": "you’d know"}
        assert len(bodies) == 3
        assert bodies[0] == bodies[1]
        assert bodies[2] == {**bodies[0], "model": settings.MEDIUM_MODEL}
        assert all(body["temperature"] == 1.0 and body["max_tokens"] == 65536 and body["response_format"] == schema for body in bodies)
    finally:
        await service.aclose()


async def test_json_retry_preserves_sampler_schema_budget_and_answer_strings() -> None:
    bodies: list[dict[str, Any]] = []
    text = {"answer": "Iven's ledger’s ```inscription``` <think>literal</think>"}

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        content = "invalid JSON" if len(bodies) == 1 else json.dumps(text)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    try:
        answer, _ = await service.async_call_llm_json_object("primary", "prompt", temperature=0.3, max_attempts=2)
        assert answer == text
        assert len(bodies) == 2
        assert bodies[0] == bodies[1]
        assert bodies[0]["temperature"] == 1.0
        assert bodies[0]["max_tokens"] == 65536
    finally:
        await service.aclose()
