# tests/conftest.py
import asyncio
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pytest

import config
from config.settings import EffectiveSettings
from core.service_context import get_services
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager
from tests.fakes.strict_commit_recorder import StrictCommitRecorder
from tests.offline import boundary_key

# Ensure repository root is on PYTHONPATH for tests
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


@contextmanager
def synthetic_run(configuration: EffectiveSettings) -> Generator[None, None, None]:
    import httpx

    from core.db_manager import Neo4jManagerSingleton
    from core.http_client_service import HTTPClientService
    from core.llm_interface_refactored import create_llm_service
    from core.service_context import RunServices, inject_services

    def unexpected_request(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Supply an explicit synthetic provider for this case")

    client = httpx.AsyncClient(transport=httpx.MockTransport(unexpected_request))
    language_model = create_llm_service(HTTPClientService(configuration=configuration, client=client))
    with inject_services(RunServices(language_model, Neo4jManagerSingleton())):
        try:
            yield
        finally:
            asyncio.run(language_model.aclose())


@pytest.fixture(autouse=True)
def run_service_context(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    if request.node.get_closest_marker("unbound_settings") is not None:
        yield
        return
    marker = request.node.get_closest_marker("run_settings")
    configuration = EffectiveSettings(_env_file=None, **marker.kwargs) if marker is not None else config.snapshot_settings()
    with synthetic_run(configuration):
        yield


@pytest.fixture
def owned_graph_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.db_manager import neo4j_manager
    from data_access.cache_coordinator import clear_all_data_access_caches
    from tests.fakes.graph_ownership import PROJECT_ID, OwnershipDriver

    monkeypatch.setattr(neo4j_manager, "_project_id", None)
    monkeypatch.setattr(neo4j_manager, "_database", None)
    monkeypatch.setattr(neo4j_manager, "_uri", None)
    neo4j_manager.bind_project(PROJECT_ID)
    monkeypatch.setattr(neo4j_manager, "driver", OwnershipDriver())
    clear_all_data_access_caches()


@pytest.fixture
def offline_graph_reads(monkeypatch: pytest.MonkeyPatch) -> FakeNeo4jManager:
    database = FakeNeo4jManager()
    monkeypatch.setattr(get_services().database, 'execute_read_query', database.execute_read_query)
    return database


@pytest.fixture
def offline_commit_providers(monkeypatch: pytest.MonkeyPatch, run_service_context: None) -> Generator[FakeNeo4jManager, None, None]:
    enclosing = config.snapshot_settings()
    values = {name: getattr(enclosing, name) for name in EffectiveSettings.model_fields}
    configuration = EffectiveSettings(_env_file=None, **{**values, "EXPECTED_EMBEDDING_DIM": 2})
    with synthetic_run(configuration):
        database = StrictCommitRecorder()
        monkeypatch.setattr(get_services(), 'database', database)

        async def embedding_batch(texts: list[str]) -> list[list[float]]:
            return [[0.25, 0.75] for text in texts]

        monkeypatch.setattr(get_services().language_model, 'async_get_embeddings_batch', embedding_batch)
        yield database


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Include explicitly marked integration cases. Fixtures must configure disposable dependencies; unit cases remain offline.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-integration"):
        return
    excluded = [item for item in items if item.get_closest_marker("integration") is not None]
    items[:] = [item for item in items if item.get_closest_marker("integration") is None]
    config.hook.pytest_deselected(items=excluded)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Generator[None, Any, Any]:
    boundary = item.config.stash[boundary_key]
    boundary.current_test = item.nodeid
    boundary.active = item.get_closest_marker("integration") is None
    try:
        return (yield)
    finally:
        boundary.active = True
        boundary.current_test = "between tests"


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    boundary = session.config.stash[boundary_key]
    if boundary.attempts and exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: Any) -> None:
    boundary = terminalreporter.config.stash[boundary_key]
    terminalreporter.write_line(f"SAGA unit boundary rejected attempts: {boundary.attempts!r}")


@pytest.fixture(autouse=True)
def offline_language_assets(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("integration") is not None:
        return
    import spacy
    import tiktoken

    encoder = tiktoken.Encoding(name="synthetic-byte", pat_str=r"(?s:.)", mergeable_ranks={bytes([number]): number for number in range(256)}, special_tokens={})
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda name: encoder)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: encoder)

    def missing_model(name: str) -> None:
        raise OSError(f"Synthetic missing spaCy model: {name}")

    monkeypatch.setattr(spacy, "load", missing_model)
