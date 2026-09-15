"""Mechanical budgets, not native-model tokenization or narrative quality."""
import json
from typing import Any

import httpx
import pytest
import tiktoken

import config
from core.http_client_service import CompletionHTTPClient, HTTPClientService
from core.text_processing_service import TokenizerService


@pytest.mark.asyncio
@pytest.mark.parametrize('extra', [{}, {'response_format': {'type': 'json_schema', 'json_schema': {'description': '海' * 300}}}])
@pytest.mark.run_settings(MAX_CONTEXT_TOKENS=220)
async def test_complete_request_rejects_unfit_system_and_options(extra: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={'choices': [{'message': {'content': 'answer'}}]})

    service = HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    try:
        with pytest.raises(ValueError, match='budget'):
            await CompletionHTTPClient(service).get_completion('synthetic', [{'role': 'system', 'content': 's' * 100}, {'role': 'user', 'content': 'u'}], 0.5, 100, **extra)
        assert requests == []
    finally:
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('options', [{'stream': True}, {'n': 2}, {'max_completion_tokens': 100000}, {'extra_body': {'messages': []}}])
async def test_options_cannot_override_request_contract(options: dict[str, Any]) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={'choices': [{'message': {'content': 'answer'}}]})

    service = HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    try:
        with pytest.raises(ValueError):
            await CompletionHTTPClient(service).get_completion('synthetic', [{'role': 'user', 'content': 'u'}], 0.5, 10, **options)
        assert requests == []
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_exact_serialized_boundary_and_no_input_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = [{'role': 'system', 'content': '海'}, {'role': 'user', 'content': 'u'}]
    payload = {'model': 'synthetic', 'messages': messages, 'temperature': 1.0, 'top_p': config.LLM_TOP_P, 'max_tokens': 10, 'stream': False}
    encoder = tiktoken.get_encoding(config.TIKTOKEN_DEFAULT_ENCODING)
    measured = len(encoder.encode(json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False), disallowed_special=()))
    # Explicit framing allowance on top of the entire serialized payload.
    limit = measured + 8 * len(messages) + 3 + 10
    requests: list[httpx.Request] = []
    for budget, admitted in [(limit, True), (limit - 1, False)]:
        effective = config.EffectiveSettings.model_validate({**config.snapshot_settings().model_dump(), "MAX_CONTEXT_TOKENS": budget})
        requests.clear()

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={'choices': [{'message': {'content': 'answer'}}]})

        service = HTTPClientService(configuration=effective, client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        try:
            if admitted:
                await CompletionHTTPClient(service).get_completion('synthetic', messages, 0.5, 10)
                assert json.loads(requests[0].content) == payload
                assert len(encoder.encode(requests[0].content.decode(), disallowed_special=())) == measured
            else:
                with pytest.raises(ValueError, match='budget'):
                    await CompletionHTTPClient(service).get_completion('synthetic', messages, 0.5, 10)
                assert requests == []
            assert messages == [{'role': 'system', 'content': '海'}, {'role': 'user', 'content': 'u'}]
        finally:
            await service.aclose()


@pytest.mark.parametrize('budget', [0, 1, 2, 3, 4, 8])
def test_truncation_remeasures_unicode_and_marker(budget: int) -> None:
    tokenizer = TokenizerService()
    value = tokenizer.truncate_text_by_tokens('🌙海' * 20, 'synthetic', budget)
    assert tokenizer.count_tokens(value, 'synthetic') <= budget
    assert '\ufffd' not in value


def test_missing_encoding_uses_conservative_byte_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TokenizerService()
    monkeypatch.setattr(tokenizer, 'get_tokenizer', lambda model: None)
    assert tokenizer.count_tokens('海', 'synthetic') == 3
    assert tokenizer.truncate_text_by_tokens('海' * 20, 'synthetic', 1) == ''
