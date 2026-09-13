"""Strict planning and context failure boundaries with synthetic transports."""
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import config
from core.http_client_service import HTTPClientService
from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.context_retrieval_node import retrieve_context
from core.langgraph.nodes.context_scene_retrieval import _summarize_scene_text, get_semantic_context
from core.langgraph.nodes.scene_planning_node import _parse_scene_plan_json_from_llm_response
from core.langgraph.nodes.summary_node import ChapterSummaryContractError, _parse_summary_response, summarize_chapter
from core.llm_interface_refactored import create_llm_service
from core.service_context import RunServices, get_services, inject_services
from tests.test_generation_failure_contract import SCENE as LEGACY_SCENE
from tests.test_generation_failure_contract import seeded_state

SCENE = {**LEGACY_SCENE, "beats": ["Hero opens the door"]}


def test_valid_scene_plan_preserves_exact_provider_fields() -> None:
    assert _parse_scene_plan_json_from_llm_response(json.dumps([SCENE])) == [SCENE]


@pytest.mark.parametrize("field,value", [
    ("title", 17), ("pov_character", None), ("setting", {}), ("plot_point", []),
    ("conflict", False), ("outcome", "  "), ("beats", "one beat"), ("beats", [17]),
    ("characters", [None]),
])
def test_scene_plan_fields_are_validated_before_retention(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match="Scene plan contract violation"):
        _parse_scene_plan_json_from_llm_response(json.dumps([{**SCENE, field: value}]))


def test_scene_plan_duplicate_keys_are_not_silently_overwritten() -> None:
    text = json.dumps([SCENE]).replace('"title":', '"title": "conflicting-canary", "title":', 1)
    with pytest.raises(ValueError, match="Scene plan contract violation"):
        _parse_scene_plan_json_from_llm_response(text)


@pytest.mark.parametrize("text", [
    'reasoning-canary {"summary": "not an answer"}',
    '{"summary": "first", "summary": "second"}',
    '[{"summary": "nested, wrong root"}]',
    '```json\n{"summary": "fenced"}\n```',
    '{"summary": "first"} {"summary": "second"}',
])
def test_summary_contract_does_not_salvage_reasoning_or_ambiguous_json(text: str) -> None:
    with pytest.raises(ChapterSummaryContractError):
        _parse_summary_response(text)


@pytest.mark.parametrize("scene_index", [-1, 1, True, 0.5])
async def test_invalid_context_index_fails_before_retrieval(tmp_path: Path, scene_index: Any) -> None:
    state = seeded_state(tmp_path)
    state["current_scene_index"] = scene_index
    result = await retrieve_context(state)
    assert result["has_fatal_error"] is True
    assert result["error_node"] == "retrieve_context"
    assert result["hybrid_context_ref"] is None


@pytest.mark.parametrize("node", ["scene_summary", "semantic", "chapter_summary"])
async def test_optional_context_does_not_swallow_total_provider_deadline(tmp_path: Path, node: str) -> None:
    calls = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()
        raise AssertionError("deadline must cancel transport")

    settings = config.snapshot_settings().model_copy(update={"HTTPX_TIMEOUT": 0.03})
    service = create_llm_service(HTTPClientService(configuration=settings, client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    try:
        with inject_services(RunServices(service, get_services().database)):
            with pytest.raises(TimeoutError):
                if node == "scene_summary":
                    await _summarize_scene_text("source", "title", "synthetic", 100)
                elif node == "semantic":
                    await get_semantic_context({}, "query", 2, "synthetic", ContentManager(str(tmp_path)))
                else:
                    manager = ContentManager(str(tmp_path))
                    await summarize_chapter({"project_dir": str(tmp_path), "current_chapter": 1, "draft_ref": manager.save_text("source", "draft", "chapter_1", 1)})
        assert calls == 1
    finally:
        await service.aclose()
