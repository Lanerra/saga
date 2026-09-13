"""Synthetic positive controls for configuration-test integration contracts."""

import inspect
import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import spacy

import config
from config.settings import EffectiveSettings
from core.embedding_contract import validate_embedding
from core.http_client_service import CompletionHTTPClient, HTTPClientService
from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization.all_chapter_outlines_node import generate_all_chapter_outlines
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import RefactoredLLMService
from core.service_context import get_services
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager
from tests.offline import boundary_key


def test_default_run_is_bound_and_guarded(request: pytest.FixtureRequest) -> None:
    services = get_services()
    assert config.snapshot_settings() is services.configuration
    assert cast(RefactoredLLMService, services.language_model).configuration is services.configuration
    assert request.config.stash[boundary_key].active is True
    with pytest.raises(OSError, match="Synthetic missing spaCy model"):
        spacy.load("en_core_web_lg")
    assert Path(inspect.getfile(HTTPClientService)).resolve() == Path(__file__).resolve().parents[1] / "core/http_client_service.py"


@pytest.mark.unbound_settings
async def test_unbound_marker_keeps_offline_guards(request: pytest.FixtureRequest) -> None:
    with pytest.raises(RuntimeError, match="active run context"):
        get_services()
    assert not isinstance(config.get_settings(), EffectiveSettings)
    assert request.config.stash[boundary_key].active is True
    with pytest.raises(OSError, match="Synthetic missing spaCy model"):
        spacy.load("en_core_web_lg")


@pytest.mark.run_settings(EXPECTED_EMBEDDING_DIM=2, NEO4J_VECTOR_DIMENSIONS=2, EMBEDDING_MODEL="synthetic-two", EMBEDDING_DTYPE="float32")
async def test_explicit_run_settings_bind_provider_and_embedding_contract() -> None:
    services = get_services()
    assert config.settings is services.configuration
    assert cast(RefactoredLLMService, services.language_model).configuration is services.configuration
    assert config.snapshot_settings().EXPECTED_EMBEDDING_DIM == 2
    assert validate_embedding([0.25, 0.75], model="synthetic-two").tolist() == [0.25, 0.75]
    with pytest.raises(ValueError, match="dimensions"):
        validate_embedding([0.25], model="synthetic-two")
    with pytest.raises(ValueError, match="model identity"):
        validate_embedding([0.25, 0.75], model="wrong-model")


@pytest.mark.run_settings(MEDIUM_MODEL="synthetic-fixture-medium")
async def test_offline_commit_provider_uses_matching_run_contract(offline_commit_providers: FakeNeo4jManager) -> None:
    services = get_services()
    assert services.database is offline_commit_providers
    assert services.configuration is config.snapshot_settings()
    assert services.configuration.EXPECTED_EMBEDDING_DIM == 2
    assert services.configuration.MEDIUM_MODEL == "synthetic-fixture-medium"
    vectors = await services.language_model.async_get_embeddings_batch(["First", "Second"])
    assert [validate_embedding(vector, model=config.EMBEDDING_MODEL).tolist() for vector in vectors] == [[0.25, 0.75], [0.25, 0.75]]
    assert offline_commit_providers.executed_queries == []


@pytest.mark.parametrize("override,expected", [(1.0, 1.0), (None, 0.7)])
async def test_sampler_override_has_explicit_disabled_control(override: float | None, expected: float) -> None:
    payloads: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Synthetic answer"}, "finish_reason": "stop"}]})

    configuration = EffectiveSettings(_env_file=None, TEMPERATURE_OVERRIDE=override)
    service = HTTPClientService(configuration=configuration, client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    try:
        await CompletionHTTPClient(service).get_completion("synthetic-model", [{"role": "user", "content": "Synthetic prompt"}], 0.7, 65536)
    finally:
        await service.aclose()
    assert payloads == [{"model": "synthetic-model", "messages": [{"role": "user", "content": "Synthetic prompt"}], "temperature": expected, "top_p": configuration.LLM_TOP_P, "max_tokens": 65536, "stream": False}]
    assert configuration.STRUCTURED_OUTPUT_STRICT is False
    assert configuration.MAX_CONTEXT_TOKENS == 131072


@pytest.mark.run_settings(GENERATE_ALL_CHAPTER_OUTLINES_AT_INIT=True)
async def test_enabled_outline_flag_produces_real_retained_outline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outline = {"scene_description": "Mara opens the archive.", "key_beats": ["Mara finds the key", "Mara opens the door", "Mara reads the ledger"], "plot_point": "Mara learns the truth"}
    calls: list[dict[str, Any]] = []

    async def respond(**options: Any) -> tuple[str, dict[str, int]]:
        calls.append(options)
        return json.dumps(outline), {}

    monkeypatch.setattr(get_services().language_model, "async_call_llm", respond)
    state: NarrativeState = {"project_dir": str(tmp_path), "total_chapters": 1, "target_word_count": 100}
    result = await generate_all_chapter_outlines(state)
    assert result["initialization_step"] == "all_chapter_outlines_complete"
    assert result["last_error"] is None
    assert len(calls) == 1
    reference = result["chapter_outlines_ref"]
    assert reference is not None
    saved = ContentManager(str(tmp_path)).load_json(reference)
    assert isinstance(saved, dict)
    assert set(saved) == {"1"}
    assert {name: saved["1"][name] for name in outline} == outline
    assert saved["1"]["chapter_number"] == 1
    assert saved["1"]["act_number"] == 1


def test_settings_do_not_leak_from_preceding_scopes() -> None:
    configuration = config.snapshot_settings()
    assert configuration.TEMPERATURE_OVERRIDE == 1.0
    assert configuration.EMBEDDING_MODEL == "nomic-embed-text:latest"
    assert configuration.EXPECTED_EMBEDDING_DIM == 768
    assert configuration.OPENAI_API_BASE == "http://127.0.0.1:9/v1"
    assert not set(EffectiveSettings.model_fields).intersection(vars(config))
