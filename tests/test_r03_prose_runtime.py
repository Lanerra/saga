"""Synthetic prose/transport regressions; no provider or narrative-quality claim."""
import asyncio
import inspect
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import spacy

import config
from core.exceptions import LLMServiceError
from core.http_client_service import HTTPClientService, completion_content, prepare_completion_payload
from core.langgraph.content_manager import ContentManager, get_scene_drafts
from core.langgraph.nodes.assemble_chapter_node import assemble_chapter
from core.langgraph.nodes.scene_generation_node import draft_scene
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import create_llm_service
from core.service_context import RunServices, get_services, inject_services
from core.spacy_service import SpacyService, get_spacy_service
from core.text_processing_service import ResponseCleaningService
from tests.test_generation_failure_contract import seeded_state


@pytest.mark.parametrize("text", [
    "Iven's ledger's clasp wouldn't open; you'd think he'd know.",
    "Iven’s ledger’s clasp wouldn’t open; you’d think he’d know.",
    '“Don’t,” she said. “It’s mine.” (He knew.) Well—perhaps... no.',
    "The sailors' maps showed a well-lit room at 3:15.",
])
def test_conservative_cleanup_preserves_source_spacing(text: str) -> None:
    service = SpacyService()
    service._nlp = spacy.blank("en")
    assert service.clean_text(text) == text


@pytest.mark.parametrize("loaded", [False, True])
def test_conservative_cleanup_preserves_paragraphs_without_model_dependence(loaded: bool) -> None:
    service = SpacyService()
    if loaded:
        service._nlp = spacy.blank("en")
    assert service.clean_text("  Iven's\tledger.\r\n\r\nYou'd  know.  ") == "Iven's ledger.\n\nYou'd know."


@pytest.mark.parametrize("text", [
    '<think>reasoning-canary',
    '<analysis>reasoning-canary',
    '<think>outer<think>inner</think>reasoning-canary',
    '<think>reasoning-canary</analysis>',
    'Answer: <think>reasoning-canary',
    '```text\n<think>reasoning-canary\n```',
])
def test_incomplete_reasoning_cannot_become_answer(text: str) -> None:
    with pytest.raises(ValueError, match="reasoning"):
        ResponseCleaningService().clean_response(text)


@pytest.mark.parametrize("text", [
    'She read the sign.\nAnswer: you’d know.\nThen she left.',
    'The inscription was ```ledger``` and nothing else.',
    'She opened the book.\nChapter 2: The Ledger\nIt was blank.',
    'First paragraph.\n\n\nSecond paragraph.',
    json.dumps({"quote": "<think>literal inscription</think>", "code": "```ledger```"}),
])
def test_cleanup_does_not_rewrite_answer_interior(text: str) -> None:
    assert ResponseCleaningService().clean_response(text) == text


@pytest.mark.parametrize("text", [
    '<think>hidden</think>\nIven’s ledger.',
    '<analysis>hidden</analysis>\nIven’s ledger.',
    'hidden\n</think>\nIven’s ledger.',
    '```text\nIven’s ledger.\n```',
    'Here is your text:\nIven’s ledger.',
    'Answer: ```text\n<think>hidden</think>\nIven’s ledger.\n```',
])
def test_explicit_outer_artifacts_are_removed(text: str) -> None:
    assert ResponseCleaningService().clean_response(text) == 'Iven’s ledger.'


def test_primary_payload_admission_applies_global_sampler() -> None:
    settings = config.snapshot_settings()
    payload = prepare_completion_payload(settings, "synthetic", [{"role": "user", "content": "prompt"}], 0.1, 65536)
    assert payload["temperature"] == 1.0
    assert payload["max_tokens"] == 65536
    assert settings.MAX_CONTEXT_TOKENS == 131072
    assert settings.STRUCTURED_OUTPUT_STRICT is False


@pytest.mark.parametrize("finish", ["length", "content_filter", "tool_calls", "function_call"])
@pytest.mark.parametrize("parts", [False, True])
def test_unsuccessful_provider_completion_is_not_an_answer(finish: str, parts: bool) -> None:
    settings = config.snapshot_settings().model_copy(update={"COMPLETION_CONTENT_FORMAT": "text_parts" if parts else "string"})
    content: Any = [{"type": "text", "text": "partial prose"}] if parts else "partial prose"
    with pytest.raises(ValueError, match="completion"):
        completion_content({"choices": [{"message": {"content": content}, "finish_reason": finish}]}, settings)


def test_explicit_provider_refusal_is_not_prose() -> None:
    with pytest.raises(ValueError, match="completion"):
        completion_content({"choices": [{"message": {"content": "not prose", "refusal": "refused"}, "finish_reason": "stop"}]}, config.snapshot_settings())


@pytest.mark.parametrize("fallback", [False, True])
async def test_no_answer_after_cleanup_is_a_failure(fallback: bool) -> None:
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>hidden</think>"}, "finish_reason": "stop"}]})

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    try:
        with pytest.raises(LLMServiceError):
            await service.async_call_llm("synthetic", "prompt", allow_fallback=fallback)
        assert len(requests) == (2 if fallback else 1)
        assert service.get_combined_statistics()["completion_service"]["completions_successful"] == 0
    finally:
        await service.aclose()


@pytest.mark.parametrize("loaded", [False, True])
async def test_real_scene_retention_and_assembly_preserve_prose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loaded: bool) -> None:
    prose = 'Iven’s ledger’s clasp wouldn’t open; you’d think he’d know.\n\n“Don’t,” she said.  “It’s mine.”'
    monkeypatch.setattr(get_spacy_service(), "_nlp", spacy.blank("en") if loaded else None)
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": prose, "reasoning_content": "reasoning-canary"}, "finish_reason": "stop"}]})

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    state = seeded_state(tmp_path)
    original = dict(state)
    try:
        with inject_services(RunServices(service, get_services().database)):
            update = await draft_scene(state)
            assert not update.get("has_fatal_error")
            combined: NarrativeState = {**state, **update}
            manager = ContentManager(str(tmp_path))
            assert get_scene_drafts(combined, manager) == [prose]
            assembly = await assemble_chapter(combined)
            assert assembly["draft_ref"] is not None
            assert manager.load_text(assembly["draft_ref"]) == prose
        assert state == original
        assert requests[0]["temperature"] == 1.0
        assert requests[0]["max_tokens"] == 65536
        assert Path(inspect.getfile(draft_scene)).resolve() == Path(__file__).resolve().parents[1] / "core/langgraph/nodes/scene_generation_node.py"
    finally:
        await service.aclose()


async def test_raw_response_opt_out_preserves_exact_provider_text() -> None:
    text = '  <think>literal author text</think>\n\n\nIven\'s  ledger.  '
    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": text}, "finish_reason": "stop"}]})
    ))))
    try:
        result, _ = await service.async_call_llm("synthetic", "prompt", auto_clean_response=False, spacy_cleanup=False)
        assert result == text
    finally:
        await service.aclose()


async def test_cancellation_never_uses_fallback() -> None:
    calls = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError()

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    try:
        with pytest.raises(asyncio.CancelledError):
            await service.async_call_llm("synthetic", "prompt", allow_fallback=True)
        assert calls == 1
    finally:
        await service.aclose()
