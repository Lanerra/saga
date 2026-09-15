"""Retained utility preference/cancellation contract; no production caller found."""
import asyncio

import httpx
import pytest

import config
from core.http_client_service import HTTPClientService
from core.llm_interface_refactored import create_llm_service
from core.service_context import RunServices, get_services, inject_services
from processing.text_deduplicator import TextDeduplicator


@pytest.mark.parametrize("semantic", [False, True])
@pytest.mark.parametrize("prefer_newer", [False, True])
async def test_duplicate_preference_preserves_selected_source_text(semantic: bool, prefer_newer: bool) -> None:
    first, last = ("He kept the ledger.", "She held the book.") if semantic else ("The ledger is closed.", "THE LEDGER IS CLOSED!")
    original = first + "\n\n" + last
    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"embedding": [0.25] * config.EXPECTED_EMBEDDING_DIM})
    ))))
    try:
        with inject_services(RunServices(service, get_services().database)):
            result, removed = await TextDeduplicator(min_segment_length_chars=1, use_semantic_comparison=semantic, prefer_newer=prefer_newer).deduplicate(original)
        assert result == (last if prefer_newer else first)
        assert removed == len(original) - len(result)
    finally:
        await service.aclose()


@pytest.mark.parametrize("failure", [asyncio.CancelledError, TimeoutError])
async def test_deduplication_propagates_cancellation_and_deadline(failure: type[BaseException]) -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        raise failure()

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    try:
        with inject_services(RunServices(service, get_services().database)):
            with pytest.raises(failure):
                await TextDeduplicator(min_segment_length_chars=1, use_semantic_comparison=True).deduplicate("A unique first paragraph.\n\nA separate last paragraph.")
    finally:
        await service.aclose()
