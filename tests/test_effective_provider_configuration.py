"""Run configuration and wire contracts with synthetic transports only."""
import logging
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import ValidationError

import config
from config.settings import EffectiveSettings, SagaSettings
from core.db_manager import Neo4jManagerSingleton
from core.http_client_service import CompletionHTTPClient, EmbeddingHTTPClient, HTTPClientService
from core.llm_interface_refactored import RefactoredLLMService, create_llm_service


@pytest.mark.parametrize('field', ['LLM_RETRY_ATTEMPTS', 'JSON_PARSE_RETRY_ATTEMPTS', 'MAX_CONCURRENT_LLM_CALLS', 'HTTPX_TIMEOUT', 'LLM_RETRY_DELAY_SECONDS'])
@pytest.mark.parametrize('value', [0, -1])
def test_nonpositive_controls_are_rejected(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        SagaSettings(_env_file=None, **cast(dict[str, Any], {field: value}))


@pytest.mark.parametrize('field', ['HTTPX_TIMEOUT', 'LLM_RETRY_DELAY_SECONDS'])
@pytest.mark.parametrize('value', [float('inf'), float('nan')])
def test_nonfinite_controls_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        SagaSettings(_env_file=None, **cast(dict[str, Any], {field: value}))


def test_credentials_are_private() -> None:
    settings = SagaSettings(_env_file=None, OPENAI_API_KEY='completion-secret-canary', EMBEDDING_API_KEY='embedding-secret-canary', NEO4J_PASSWORD='database-secret-canary')
    for secret in ('completion-secret-canary', 'embedding-secret-canary', 'database-secret-canary'):
        assert secret not in repr(settings)
        assert secret not in str(settings)


@pytest.mark.asyncio
@pytest.mark.parametrize('embedding_key', ['embedding-secret-canary', ''])
@pytest.mark.unbound_settings
async def test_credentials_are_endpoint_isolated(monkeypatch: pytest.MonkeyPatch, embedding_key: str) -> None:
    monkeypatch.setitem(vars(config), 'OPENAI_API_BASE', 'https://completion.invalid/v1')
    monkeypatch.setitem(vars(config), 'EMBEDDING_API_BASE', 'https://embedding.invalid')
    monkeypatch.setitem(vars(config), 'OPENAI_API_KEY', 'completion-secret-canary')
    monkeypatch.setitem(vars(config), 'EMBEDDING_API_KEY', embedding_key)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == 'embedding.invalid':
            return httpx.Response(200, json={'embedding': [0.5]})
        return httpx.Response(200, json={'choices': [{'message': {'content': 'answer'}}]})

    service = HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    try:
        await EmbeddingHTTPClient(service).get_embedding('prompt-canary', 'embedding-model')
        await CompletionHTTPClient(service).get_completion('completion-model', [{'role': 'user', 'content': 'prompt-canary'}], 0.5, 10)
        assert [(str(request.url), request.headers.get('authorization')) for request in requests] == [
            ('https://embedding.invalid/api/embeddings', 'Bearer embedding-secret-canary' if embedding_key else None),
            ('https://completion.invalid/v1/chat/completions', 'Bearer completion-secret-canary'),
        ]
    finally:
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.unbound_settings
async def test_active_provider_snapshot_survives_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(vars(config), 'OPENAI_API_BASE', 'https://first.invalid/v1')
    monkeypatch.setitem(vars(config), 'OPENAI_API_KEY', 'first-secret-canary')
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={'choices': [{'message': {'content': 'answer'}}]})

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    monkeypatch.setitem(vars(config), 'OPENAI_API_BASE', 'https://later.invalid/v1')
    monkeypatch.setitem(vars(config), 'OPENAI_API_KEY', 'later-secret-canary')
    try:
        assert await service.async_call_llm('model', 'prompt-canary', auto_clean_response=False) == ('answer', {})
        assert [(str(request.url), request.headers['authorization']) for request in requests] == [('https://first.invalid/v1/chat/completions', 'Bearer first-secret-canary')]
    finally:
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('body', [{'vector': [0.5]}, {'embedding': [[0.5]]}, {'embedding': ['0.5']}, {'embedding': [True]}, {'embedding': [0.5], 'unexpected': 'response-canary'}])
async def test_embedding_wire_schema_rejects_alternates(body: dict[str, Any]) -> None:
    service = HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))))
    try:
        with pytest.raises(ValueError):
            await EmbeddingHTTPClient(service).get_embedding('prompt-canary', 'model')
    finally:
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('body', [{'choices': []}, {'choices': [{'content': 'response-canary'}]}, {'choices': [{'message': {'reasoning_content': 'response-canary'}}]}, {'choices': [{'message': {'content': 17}}]}, {'choices': [{'message': {'content': 'valid'}, 'finish_reason': 'response-canary'}]}])
async def test_completion_wire_schema_rejects_alternates(body: dict[str, Any]) -> None:
    service = HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))))
    try:
        with pytest.raises(ValueError):
            await CompletionHTTPClient(service).get_completion('model', [{'role': 'user', 'content': 'prompt-canary'}], 0.5, 10)
    finally:
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [200, 400])
async def test_provider_operational_logs_exclude_canaries(caplog: pytest.LogCaptureFixture, status: int) -> None:
    caplog.set_level(logging.DEBUG)
    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(status, json={'error': 'response-canary prompt-canary credential-canary'})))))
    try:
        await service.async_call_llm('model', 'prompt-canary', strict=False)
        await service.async_get_embedding('prompt-canary')
        for canary in ('prompt-canary', 'response-canary', 'credential-canary'):
            assert canary not in caplog.text
    finally:
        await service.aclose()


@pytest.mark.unbound_settings
def test_reload_uses_real_file_below_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest) -> None:
    from pydantic_settings import DotEnvSettingsSource

    from config.loader import reload_settings
    from tests.offline import boundary_key

    boundary = request.config.stash[boundary_key]
    original_reader = next(value for owner, name, value in boundary.patches._setattr if owner is DotEnvSettingsSource and name == '_read_env_files')
    monkeypatch.setattr(DotEnvSettingsSource, '_read_env_files', original_reader)
    synthetic_file = tmp_path / 'synthetic-settings.ini'
    synthetic_file.write_text('OPENAI_API_KEY=file-secret-canary\nOPENAI_API_BASE=https://file.invalid/v1\nTEMPERATURE_OVERRIDE=0.25\n')
    handlers = list(logging.getLogger().handlers)
    try:
        with monkeypatch.context() as environment:
            environment.setenv('OPENAI_API_KEY', 'process-secret-canary')
            environment.setenv('OPENAI_API_BASE', 'https://process.invalid/v1')
            assert reload_settings(env_file=synthetic_file) is True
            with config.bind_settings(config.snapshot_settings()):
                assert config.OPENAI_API_BASE == 'https://process.invalid/v1'
                assert config.settings.OPENAI_API_KEY.get_secret_value() == 'process-secret-canary'
                assert config.Temperatures.OVERRIDE == 0.25
            assert logging.getLogger().handlers == handlers
            synthetic_file.write_text('LLM_RETRY_ATTEMPTS=0\n')
            with pytest.raises(ValidationError):
                reload_settings(env_file=synthetic_file)
            assert config.snapshot_settings().TEMPERATURE_OVERRIDE == 0.25
    finally:
        reload_settings(env_file=None)


@pytest.mark.asyncio
@pytest.mark.unbound_settings
async def test_managed_run_binds_one_immutable_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.service_context import managed_services
    from tests.fakes.fake_neo4j_manager import FakeNeo4jManager

    try:
        with monkeypatch.context() as environment:
            environment.setenv('TEMPERATURE_OVERRIDE', '0.2')
            config.reload(env_file=None)
            async with managed_services(database=cast(Neo4jManagerSingleton, FakeNeo4jManager())) as services:
                assert config.settings is services.configuration
                assert cast(RefactoredLLMService, services.language_model).configuration is services.configuration
                with pytest.raises(ValidationError):
                    services.configuration.HTTPX_TIMEOUT = 0  # type: ignore[misc]
                with pytest.raises(ValidationError):
                    services.configuration.validation.ENABLE_VALIDATION = False  # type: ignore[misc]
                with pytest.raises(TypeError):
                    services.configuration.relationship_normalization.SIMILARITY_THRESHOLDS['DEFAULT'] = 0.1  # type: ignore[index]
                environment.setenv('TEMPERATURE_OVERRIDE', '0.8')
                config.reload(env_file=None)
                assert config.Temperatures.OVERRIDE == 0.2
            async with managed_services(database=cast(Neo4jManagerSingleton, FakeNeo4jManager())) as services:
                assert config.settings is services.configuration
                assert config.Temperatures.OVERRIDE == 0.8
    finally:
        config.reload(env_file=None)


@pytest.mark.asyncio
async def test_narrative_parser_embedding_failure_logs_no_text(caplog: pytest.LogCaptureFixture) -> None:
    from core.parsers.narrative_enrichment_parser import NarrativeEnrichmentParser
    from core.service_context import RunServices, inject_services
    from tests.fakes.fake_neo4j_manager import FakeNeo4jManager

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(400, json={'error': 'response-canary'})))))
    try:
        with inject_services(RunServices(service, cast(Neo4jManagerSingleton, FakeNeo4jManager()))):
            parser = NarrativeEnrichmentParser('synthetic narrative')
            assert await parser._generate_embedding_vector('prompt-canary') is None
            assert 'prompt-canary' not in caplog.text
            assert 'response-canary' not in caplog.text
    finally:
        await service.aclose()


@pytest.mark.parametrize('field', ['LLM_RETRY_ATTEMPTS', 'JSON_PARSE_RETRY_ATTEMPTS', 'MAX_CONCURRENT_LLM_CALLS', 'HTTPX_TIMEOUT'])
@pytest.mark.unbound_settings
def test_invalid_controls_fail_before_client_allocation(monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    monkeypatch.setitem(vars(config), field, 0)
    allocations = []

    def allocate(**kwargs: Any) -> httpx.AsyncClient:
        allocations.append(kwargs)
        raise AssertionError('Must validate before allocating')

    monkeypatch.setattr(httpx, 'AsyncClient', allocate)
    with pytest.raises(ValidationError):
        HTTPClientService()
    assert allocations == []


@pytest.mark.asyncio
async def test_embedding_redirect_cannot_forward_dedicated_credential() -> None:
    requests: list[httpx.Request] = []

    def redirect(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(307, headers={'Location': 'https://other.invalid/stolen'})

    effective = EffectiveSettings(_env_file=None, EMBEDDING_API_KEY='dedicated-secret-canary', EMBEDDING_API_BASE='https://embedding.invalid', LLM_RETRY_ATTEMPTS=1)
    service = HTTPClientService(configuration=effective, client=httpx.AsyncClient(transport=httpx.MockTransport(redirect), follow_redirects=True))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await EmbeddingHTTPClient(service).get_embedding('prompt-canary', 'model')
        assert len(requests) == 1
        assert requests[0].url.host == 'embedding.invalid'
        assert requests[0].headers['Authorization'] == 'Bearer dedicated-secret-canary'
    finally:
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('format_name', ['text', 'text_parts'])
async def test_configured_completion_formats_over_transport(format_name: str, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    content: Any = 'response-canary' if format_name == 'text' else [{'type': 'text', 'text': 'response-canary'}]
    body = {'choices': [{'message': {'content': content, 'reasoning_content': 'private-reasoning-canary'}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
    configuration = EffectiveSettings(_env_file=None, COMPLETION_CONTENT_FORMAT=format_name, OPENAI_API_KEY='completion-secret-canary')
    service = create_llm_service(HTTPClientService(configuration=configuration, client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)))))
    try:
        response, usage = await service.async_call_llm('model', 'prompt-canary', auto_clean_response=False)
        assert response == 'response-canary'
        assert usage is not None and usage['total_tokens'] == 2
        for canary in ('response-canary', 'prompt-canary', 'private-reasoning-canary', 'completion-secret-canary'):
            assert canary not in caplog.text
    finally:
        await service.aclose()


def test_state_defaults_use_bound_run_snapshot(tmp_path: Path) -> None:
    from core.langgraph.state import create_initial_state

    effective = EffectiveSettings(_env_file=None, SMALL_MODEL='run-small', MEDIUM_MODEL='run-medium', LARGE_MODEL='run-large', NARRATIVE_MODEL='run-narrative', DEFAULT_NARRATIVE_STYLE='run-style')
    with config.bind_settings(effective):
        state = create_initial_state(project_id='synthetic', title='Synthetic', genre='genre', theme='theme', setting='setting', target_word_count=100, total_chapters=1, project_dir=str(tmp_path), protagonist_name='Hero')
    assert state['small_model'] == state['extraction_model'] == 'run-small'
    assert state['medium_model'] == state['revision_model'] == 'run-medium'
    assert state['large_model'] == 'run-large'
    assert state['narrative_model'] == 'run-narrative'
    assert state['narrative_style'] == 'run-style'


def test_validation_uses_bound_run_snapshot() -> None:
    from core.langgraph.nodes.validation_node import _is_plot_stagnant

    effective = EffectiveSettings(_env_file=None, PLOT_STAGNATION_MIN_WORD_COUNT=0, PLOT_STAGNATION_MIN_ENTITIES=0)
    with config.bind_settings(effective):
        assert _is_plot_stagnant({'draft_word_count': 0}, {}, []) is False


@pytest.mark.asyncio
async def test_outline_flag_uses_bound_run_snapshot() -> None:
    from core.langgraph.initialization.all_chapter_outlines_node import generate_all_chapter_outlines

    with config.bind_settings(EffectiveSettings(_env_file=None, GENERATE_ALL_CHAPTER_OUTLINES_AT_INIT=False)):
        assert await generate_all_chapter_outlines({}) == {
            'current_node': 'all_chapter_outlines',
            'initialization_step': 'all_chapter_outlines_failed',
            'has_fatal_error': True,
            'last_error': 'Initialization requires GENERATE_ALL_CHAPTER_OUTLINES_AT_INIT=True; on-demand-only initialization is unsupported',
        }


def test_schema_validator_uses_bound_run_snapshot() -> None:
    from core.schema_validator import schema_validator

    effective = EffectiveSettings(_env_file=None, schema_enforcement={'ENFORCE_SCHEMA_VALIDATION': False})
    with config.bind_settings(effective):
        assert schema_validator.validate_entity_type('SyntheticUnknown') == (True, 'SyntheticUnknown', None)


@pytest.mark.asyncio
@pytest.mark.parametrize('attempts', [1, 2, 3])
@pytest.mark.parametrize('route', ['world', 'possessions', 'event_items'])
async def test_parser_honors_configured_attempts(attempts: int, route: str, caplog: pytest.LogCaptureFixture) -> None:
    import json

    from core.parsers.act_outline_parser import ActOutlineParser
    from core.parsers.global_outline_parser import GlobalOutlineParser
    from core.service_context import RunServices, inject_services
    from models.kg_models import ActKeyEvent
    from tests.fakes.fake_neo4j_manager import FakeNeo4jManager

    effective = EffectiveSettings(_env_file=None, JSON_PARSE_RETRY_ATTEMPTS=attempts)
    count = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        content = 'response-canary'
        if count == attempts:
            if route == 'world':
                content = '[]'
            elif route == 'possessions':
                content = json.dumps({'possessions': [{'character': 'response-canary', 'item': 'item'}]})
            else:
                content = json.dumps({'featured_items': [{'item': 'response-canary', 'role': 'featured'}]})
        return httpx.Response(200, json={'choices': [{'message': {'content': content}}]})

    service = create_llm_service(HTTPClientService(configuration=effective, client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    try:
        with inject_services(RunServices(service, cast(Neo4jManagerSingleton, FakeNeo4jManager()))):
            if route == 'world':
                assert await GlobalOutlineParser()._extract_world_items_from_outline({'raw_text': 'prompt-canary'}) == []
            elif route == 'possessions':
                assert await GlobalOutlineParser()._extract_item_possessions({'raw_text': 'prompt-canary'}, [], []) == {'response-canary': 'item'}
            else:
                event = ActKeyEvent(id='event', name='event', description='prompt-canary', act_number=1, sequence_in_act=1, cause='cause', effect='effect')
                assert await ActOutlineParser()._extract_event_item_involvements([event], []) == {'event': [('response-canary', 'featured')]}
            assert count == attempts
            assert 'response-canary' not in caplog.text
            assert 'prompt-canary' not in caplog.text
    finally:
        await service.aclose()

