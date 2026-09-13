"""Absolute provider deadlines using real tasks and deterministic transports."""
import asyncio
from typing import Any

import httpx
import pytest

from core.http_client_service import HTTPClientService
from core.llm_interface_refactored import create_llm_service


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['transport', 'backoff', 'semaphore'])
@pytest.mark.run_settings(HTTPX_TIMEOUT=0.02, LLM_RETRY_DELAY_SECONDS=1.0)
async def test_deadline_cancels_entire_http_operation(phase: str, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    cancelled = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if phase == 'backoff':
            return httpx.Response(429)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError('unreachable')

    service = HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    if phase == 'semaphore':
        service._semaphore = asyncio.Semaphore(0)
    operation = asyncio.create_task(service.post_json('https://synthetic.invalid', {}))
    try:
        done, _ = await asyncio.wait({operation}, timeout=0.3)
        assert done == {operation}, 'total deadline did not stop operation'
        with pytest.raises(TimeoutError):
            await operation
        assert attempts == (0 if phase == 'semaphore' else 1)
        assert cancelled.is_set() == (phase == 'transport')
    finally:
        operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['fallback', 'object', 'array'])
@pytest.mark.run_settings(HTTPX_TIMEOUT=0.04, LLM_RETRY_ATTEMPTS=1)
async def test_composed_operations_share_one_deadline(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0.025)
        if kind == 'fallback' and attempts == 1:
            return httpx.Response(400)
        answer = 'invalid-json' if attempts == 1 else ('{}' if kind == 'object' else '[]')
        return httpx.Response(200, json={'choices': [{'message': {'content': answer}}]})

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    try:
        arguments: dict[str, Any] = {'model_name': 'synthetic', 'prompt': 'synthetic', 'auto_clean_response': False}
        with pytest.raises(TimeoutError):
            if kind == 'fallback':
                await service.async_call_llm(**arguments, allow_fallback=True)
            elif kind == 'object':
                await service.async_call_llm_json_object(**arguments)
            else:
                await service.async_call_llm_json_array(**arguments)
        assert attempts == 2
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_caller_cancellation_propagates_and_releases_semaphore() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError('unreachable')

    service = HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    operation = asyncio.create_task(service.post_json('https://synthetic.invalid', {}))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert cancelled.is_set()
        assert service._semaphore._value == service.configuration.MAX_CONCURRENT_LLM_CALLS
    finally:
        await service.aclose()
