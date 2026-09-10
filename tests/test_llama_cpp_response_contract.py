"""Frozen synthetic llama.cpp exchange replay; never a live-provider quality test."""
import hashlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import spacy
from pydantic import ValidationError

from config.settings import EffectiveSettings
from core.db_manager import Neo4jManagerSingleton
from core.exceptions import LLMServiceError
from core.http_client_service import CompletionHTTPClient, CompletionResponse, HTTPClientService, PartsCompletionResponse, completion_content
from core.langgraph.content_manager import ContentManager, get_scene_drafts
from core.langgraph.nodes.scene_generation_node import draft_scene
from core.langgraph.state import NarrativeState, create_initial_state
from core.llm_interface_refactored import RefactoredLLMService, create_llm_service
from core.service_context import RunServices, inject_services
from core.spacy_service import get_spacy_service
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager
from tests.offline import boundary_key

FIXTURES = Path(__file__).parent / "fixtures"
RESPONSE_BYTES = (FIXTURES / "llama_cpp_response.json").read_bytes()
RECORDED_REQUEST = json.loads((FIXTURES / "llama_cpp_request.json").read_bytes())["request"]
COUNT_FIELDS = ("cache_n", "prompt_n", "predicted_n", "draft_n", "draft_n_accepted")
MEASUREMENT_FIELDS = ("prompt_ms", "prompt_per_token_ms", "prompt_per_second", "predicted_ms", "predicted_per_token_ms", "predicted_per_second")
# The production conservative spaCy cleanup inserts spaces at these token boundaries.
PERSISTED_DRAFT = "The brass key slides home. I turn it. The mechanism grinds, then releases. Behind me, Iven 's shoes scratch the flagstones— he waits, outside. The door swings. Cold air rolls out to meet me. I cross the threshold, the brass still warm in my palm. Dawn is hours off. After that, the ferry."


def test_frozen_response_schema_preserves_metadata() -> None:
    assert hashlib.sha256(RESPONSE_BYTES).hexdigest() == "c3baabe50393acf11e9aa9155259fc728744ab241c221dc997de22bdf26e735f"
    response = json.loads(RESPONSE_BYTES)
    validated = CompletionResponse.model_validate_json(RESPONSE_BYTES)
    assert validated.model_dump(exclude_unset=True) == response
    assert completion_content(response, EffectiveSettings(_env_file=None)) == response["choices"][0]["message"]["content"]


@pytest.mark.parametrize("format_name", ["text", "text_parts"])
@pytest.mark.parametrize("field", COUNT_FIELDS + ("cached_tokens",))
@pytest.mark.parametrize("value", [-1, 1.5, "1", True, None])
def test_metadata_counts_are_strict_nonnegative_integers(format_name: str, field: str, value: Any) -> None:
    response = json.loads(RESPONSE_BYTES)
    parent = response["usage"]["prompt_tokens_details"] if field == "cached_tokens" else response["timings"]
    parent[field] = value
    if format_name == "text_parts":
        response["choices"][0]["message"]["content"] = [{"type": "text", "text": "Synthetic prose."}]
    with pytest.raises(ValidationError) as caught:
        (PartsCompletionResponse if format_name == "text_parts" else CompletionResponse).model_validate(response)
    assert [error["loc"] for error in caught.value.errors()] == [("usage", "prompt_tokens_details", field) if field == "cached_tokens" else ("timings", field)]


@pytest.mark.parametrize("field", MEASUREMENT_FIELDS)
@pytest.mark.parametrize("value", [-0.1, "1.0", True, None, float("nan"), float("inf"), float("-inf")])
def test_metadata_measurements_are_strict_finite_nonnegative_numbers(field: str, value: Any) -> None:
    response = json.loads(RESPONSE_BYTES)
    response["timings"][field] = value
    with pytest.raises(ValidationError) as caught:
        CompletionResponse.model_validate(response)
    assert [error["loc"] for error in caught.value.errors()] == [("timings", field)]


@pytest.mark.parametrize("metadata", ["absent", "null", "zero"])
@pytest.mark.parametrize("format_name", ["text", "text_parts"])
def test_optional_metadata_and_zero_values(metadata: str, format_name: str) -> None:
    response = json.loads(RESPONSE_BYTES)
    if metadata == "absent":
        del response["timings"]
        del response["usage"]["prompt_tokens_details"]
    elif metadata == "null":
        response["timings"] = None
        response["usage"]["prompt_tokens_details"] = None
    else:
        response["timings"] = {field: 0 for field in COUNT_FIELDS + MEASUREMENT_FIELDS}
    expected = response["choices"][0]["message"]["content"]
    if format_name == "text_parts":
        response["choices"][0]["message"]["content"] = [{"type": "text", "text": expected}]
    assert completion_content(response, EffectiveSettings(_env_file=None, COMPLETION_CONTENT_FORMAT=format_name)) == expected


def seeded_scene(directory: Path) -> NarrativeState:
    state = create_initial_state(project_id="synthetic-replay", title="The Spare Key", genre="Mystery", theme="Trust", setting="Island archive", target_word_count=120, total_chapters=1, project_dir=str(directory), protagonist_name="Mara")
    state["narrative_style"] = "First-person present, spare concrete sentences; no access to other minds."
    state["narrative_model"] = "3.6-a3b"
    scene = {"title": "Signal 1", "pov_character": "Mara", "setting": "Island archive", "characters": ["Mara"], "plot_point": "Enter the obstacle", "conflict": "Time is running out", "outcome": "Iven waits outside while I cross the threshold.", "beats": ["I turn the brass key in the archive door"]}
    manager = ContentManager(str(directory))
    state["chapter_plan_ref"] = manager.save_json([scene, scene], "chapter_plan", "chapter_1", 1)
    state["chapter_plan_scene_count"] = 2
    state["hybrid_context_ref"] = manager.save_text("Mara has the brass key. The archive door is locked. Iven waits outside. The ferry leaves at dawn.", "hybrid_context", "chapter_1", 1)
    return state


@pytest.mark.parametrize("route", ["completion", "draft"])
@pytest.mark.parametrize("defect", [None, "unknown_timing", "missing_timing", "wrong_details", "unknown_details", "bad_count", "bad_measurement", "reasoning_only", "empty_content", "wrong_content", "multiple_choices", "invalid_usage", "unknown_top_level"])
async def test_recorded_completion_and_draft_path(route: str, defect: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, request: pytest.FixtureRequest) -> None:
    root = Path(__file__).resolve().parents[1]
    for symbol, path in ((CompletionResponse, "core/http_client_service.py"), (CompletionHTTPClient, "core/http_client_service.py"), (RefactoredLLMService, "core/llm_interface_refactored.py"), (draft_scene, "core/langgraph/nodes/scene_generation_node.py"), (ContentManager, "core/langgraph/content_manager.py")):
        assert Path(inspect.getfile(symbol)).resolve() == root / path
    boundary = request.config.stash[boundary_key]
    assert boundary.active is True
    attempts_before = list(boundary.attempts)
    pipeline = spacy.blank("en")
    pipeline.add_pipe("sentencizer")
    monkeypatch.setattr(get_spacy_service(), "_nlp", pipeline)
    caplog.set_level(logging.DEBUG)
    response = json.loads(RESPONSE_BYTES)
    expected = response["choices"][0]["message"]["content"]
    if defect == "unknown_timing":
        response["timings"]["unknown"] = 1
    elif defect == "missing_timing":
        del response["timings"]["prompt_n"]
    elif defect == "wrong_details":
        response["usage"]["prompt_tokens_details"] = []
    elif defect == "unknown_details":
        response["usage"]["prompt_tokens_details"]["unknown"] = 1
    elif defect == "bad_count":
        response["timings"]["cache_n"] = True
    elif defect == "bad_measurement":
        response["timings"]["prompt_ms"] = -1.0
    elif defect == "reasoning_only":
        del response["choices"][0]["message"]["content"]
    elif defect == "empty_content":
        response["choices"][0]["message"]["content"] = "   "
    elif defect == "wrong_content":
        response["choices"][0]["message"]["content"] = 17
    elif defect == "multiple_choices":
        response["choices"] *= 2
    elif defect == "invalid_usage":
        response["usage"]["completion_tokens"] = "428"
    elif defect == "unknown_top_level":
        response["unknown"] = "response-canary"
    wire_bytes = RESPONSE_BYTES if defect is None else json.dumps(response).encode()
    requests: list[httpx.Request] = []

    def replay(outgoing: httpx.Request) -> httpx.Response:
        requests.append(outgoing)
        assert outgoing.method == "POST"
        assert str(outgoing.url) == "https://recorded.invalid/v1/chat/completions"
        assert outgoing.headers["authorization"] == "Bearer completion-canary"
        assert "embedding-canary" not in str(outgoing.headers)
        assert json.loads(outgoing.content) == RECORDED_REQUEST
        return httpx.Response(200, content=wire_bytes, headers={"Content-Type": "application/json"})

    effective = EffectiveSettings(_env_file=None, OPENAI_API_BASE="https://recorded.invalid/v1", OPENAI_API_KEY="completion-canary", EMBEDDING_API_KEY="embedding-canary", MAX_GENERATION_TOKENS=4096, TEMPERATURE_OVERRIDE=None, LLM_TOP_P=0.95, LLM_RETRY_ATTEMPTS=1, HTTPX_TIMEOUT=120)
    transport = HTTPClientService(configuration=effective, client=httpx.AsyncClient(transport=httpx.MockTransport(replay)))
    service = create_llm_service(transport)
    database = FakeNeo4jManager()
    try:
        with inject_services(RunServices(service, cast(Neo4jManagerSingleton, database))):
            arguments: dict[str, Any] = {"model_name": RECORDED_REQUEST["model"], "prompt": RECORDED_REQUEST["messages"][1]["content"], "system_prompt": RECORDED_REQUEST["messages"][0]["content"], "temperature": 0.7, "max_tokens": 4096, "auto_clean_response": False}
            if route == "completion":
                if defect is None:
                    text, usage = await service.async_call_llm(**arguments)
                    assert text == expected
                    assert usage == {**response["usage"], "finish_reason": "stop"}
                else:
                    with pytest.raises(LLMServiceError):
                        await service.async_call_llm(**arguments)
            else:
                state = seeded_scene(tmp_path)
                original = dict(state)
                result = await draft_scene(state)
                assert state == original
                if defect is None:
                    assert set(result) == {"scene_drafts_ref", "current_scene_index", "current_node"}
                    assert result["current_scene_index"] == 1
                    assert get_scene_drafts(result, ContentManager(str(tmp_path))) == [PERSISTED_DRAFT]
                else:
                    assert result["has_fatal_error"] is True
                    assert result["error_node"] == "draft_scene"
                    assert "scene_drafts_ref" not in result
                    assert ContentManager(str(tmp_path)).get_latest_version("scenes", "chapter_1") == 0
            assert len(requests) == 1
            assert transport.request_count == 1
            assert database.executed_queries == []
            assert boundary.attempts == attempts_before
            for canary in ("The brass key slides home", "The user wants me to write", "I turn the brass key in the archive door", "completion-canary", "embedding-canary", "response-canary"):
                assert canary not in caplog.text
    finally:
        await service.aclose()
    assert transport._client.is_closed
    assert hashlib.sha256(RESPONSE_BYTES).hexdigest() == "c3baabe50393acf11e9aa9155259fc728744ab241c221dc997de22bdf26e735f"
