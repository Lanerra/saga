"""Per-run ownership at the real service and consumer boundaries."""
import asyncio
from pathlib import Path
from typing import Any

import pytest


class ExampleLanguageModel:
    def __init__(self) -> None:
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


class ExampleDatabase:
    def __init__(self) -> None:
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_sequential_run_ownership(outcome: str) -> None:
    from core.service_context import get_services, managed_services

    created: list[ExampleLanguageModel] = []
    database: Any = ExampleDatabase()

    def factory() -> Any:
        service = ExampleLanguageModel()
        created.append(service)
        return service

    async def run() -> None:
        async with managed_services(language_model_factory=factory, database=database) as services:
            assert get_services() is services
            assert services.language_model is created[-1]
            assert services.database is database
            if outcome == "error":
                raise ValueError("synthetic failure")
            if outcome == "cancel":
                raise asyncio.CancelledError

    for _ in range(2):
        if outcome == "error":
            with pytest.raises(ValueError, match="synthetic failure"):
                await run()
        elif outcome == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await run()
        else:
            await run()
    assert created[0] is not created[1]
    assert [service.closed for service in created] == [1, 1]
    assert database.closed == 2


async def test_database_closes_when_language_model_creation_fails() -> None:
    from core.service_context import managed_services

    database: Any = ExampleDatabase()

    def factory() -> Any:
        raise ValueError("creation failed")

    with pytest.raises(ValueError, match="creation failed"):
        async with managed_services(language_model_factory=factory, database=database):
            pytest.fail("Must not enter")
    assert database.closed == 1


async def test_parser_consumes_injected_embedding_interface(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx
    import numpy as np

    import config
    from core.parsers.narrative_enrichment_parser import NarrativeEnrichmentParser
    from core.service_context import RunServices, inject_services

    class Embeddings(ExampleLanguageModel):
        async def async_get_embedding(self, text: str) -> np.ndarray:
            assert text == "synthetic injection canary"
            return np.array([0.25, 0.75])

    async def unexpected_transport(*arguments: Any, **keywords: Any) -> None:
        raise AssertionError("Injected parser must not use a global HTTP client")

    monkeypatch.setattr(config, "EXPECTED_EMBEDDING_DIM", 2)
    monkeypatch.setattr(httpx.AsyncClient, "post", unexpected_transport)
    language_model: Any = Embeddings()
    database: Any = ExampleDatabase()
    with inject_services(RunServices(language_model, database)):
        result = await NarrativeEnrichmentParser("synthetic")._generate_embedding_vector("synthetic injection canary")
    assert result == [0.25, 0.75]


async def test_data_access_consumes_injected_database() -> None:
    from core.service_context import RunServices, inject_services
    from data_access.character_queries import get_all_character_names

    class Database(ExampleDatabase):
        async def execute_read_query(self, query: str) -> list[dict[str, str]]:
            assert query == "MATCH (c:Character) RETURN c.name AS name ORDER BY c.name"
            return [{"name": "Synthetic Character"}]

    language_model: Any = ExampleLanguageModel()
    database: Any = Database()
    with inject_services(RunServices(language_model, database)):
        assert await get_all_character_names() == ["Synthetic Character"]


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_real_clients_recreated_and_closed_through_langgraph(outcome: str) -> None:
    import httpx
    from langgraph.constants import END
    from langgraph.graph.state import StateGraph

    from core.http_client_service import HTTPClientService
    from core.llm_interface_refactored import create_llm_service
    from core.service_context import LanguageModel, get_services, managed_services

    class Transport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.calls = 0
            self.closes = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.calls += 1
            assert request.url.path == "/v1/chat/completions"
            return httpx.Response(200, json={"choices": [{"message": {"content": "Synthetic reply"}}]})

        async def aclose(self) -> None:
            self.closes += 1

    transports: list[Transport] = []
    clients: list[httpx.AsyncClient] = []
    services: list[LanguageModel] = []

    def factory() -> LanguageModel:
        transport = Transport()
        client = httpx.AsyncClient(transport=transport)
        service = create_llm_service(HTTPClientService(client=client))
        transports.append(transport)
        clients.append(client)
        services.append(service)
        return service

    async def consumer(state: dict[str, str]) -> dict[str, str]:
        assert get_services().language_model is services[-1]
        text, _ = await get_services().language_model.async_call_llm("synthetic", state["prompt"], auto_clean_response=False)
        return {"reply": text}

    builder = StateGraph(dict)
    builder.add_node("consumer", consumer)
    builder.set_entry_point("consumer")
    builder.add_edge("consumer", END)
    graph = builder.compile()
    database: Any = ExampleDatabase()

    async def run() -> None:
        async with managed_services(language_model_factory=factory, database=database):
            assert await graph.ainvoke({"prompt": "Synthetic prompt"}) == {"reply": "Synthetic reply"}
            if outcome == "error":
                raise ValueError("workflow failure")
            if outcome == "cancel":
                raise asyncio.CancelledError

    for _ in range(2):
        if outcome == "error":
            with pytest.raises(ValueError, match="workflow failure"):
                await run()
        elif outcome == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await run()
        else:
            await run()
    assert services[0] is not services[1]
    assert clients[0] is not clients[1]
    assert [(transport.calls, transport.closes) for transport in transports] == [(1, 1), (1, 1)]
    assert [client.is_closed for client in clients] == [True, True]
    assert database.closed == 2


def test_expired_task_context_cannot_resolve_services() -> None:
    from contextvars import Context, copy_context

    from core.service_context import RunServices, get_services, inject_services

    with pytest.raises(RuntimeError, match="active run context"):
        Context().run(get_services)
    language_model: Any = ExampleLanguageModel()
    database: Any = ExampleDatabase()
    with inject_services(RunServices(language_model, database)):
        inherited = copy_context()
        assert inherited.run(get_services).language_model is language_model
    with pytest.raises(RuntimeError, match="active run context"):
        inherited.run(get_services)


async def test_closed_owned_services_cannot_be_reused() -> None:
    from core.service_context import inject_services, managed_services

    language_model: Any = ExampleLanguageModel()
    database: Any = ExampleDatabase()
    async with managed_services(language_model_factory=lambda: language_model, database=database) as services:
        pass
    with pytest.raises(RuntimeError, match="closed"):
        with inject_services(services):
            pytest.fail("Closed services must not be borrowed")


def test_interface_import_does_not_allocate_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy

    import httpx

    import core.llm_interface_refactored as interface

    def unexpected_client(*arguments: Any, **keywords: Any) -> None:
        raise AssertionError("Interface import allocated an HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", unexpected_client)
    loaded = runpy.run_path(str(interface.__file__))
    assert "llm_service" not in loaded


async def test_close_failure_still_closes_database() -> None:
    from core.service_context import managed_services

    class FailingClose(ExampleLanguageModel):
        async def aclose(self) -> None:
            await super().aclose()
            raise RuntimeError("close failed")

    language_model: Any = FailingClose()
    database: Any = ExampleDatabase()
    with pytest.raises(RuntimeError, match="close failed"):
        async with managed_services(language_model_factory=lambda: language_model, database=database):
            pass
    assert language_model.closed == 1
    assert database.closed == 1


async def test_quick_bootstrap_borrows_one_owner_until_schema_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    import config
    from core.graph_ownership import load_graph_project_id
    from core.project_manager import ProjectManager
    from core.service_context import get_services, managed_services
    from main import run_quick_mode

    events: list[str] = []

    class Metadata(ExampleLanguageModel):
        async def async_call_llm(self, **arguments: Any) -> tuple[str, dict[str, int]]:
            assert get_services().language_model is self
            events.append("metadata")
            return json.dumps({
                "title": "Synthetic Run", "genre": "Fantasy", "theme": "Discovery", "setting": "Harbor",
                "protagonist_name": "Ada", "narrative_style": config.DEFAULT_NARRATIVE_STYLE, "total_chapters": 1,
                "target_word_count": config.TARGET_WORD_COUNT,
            }), {}

        async def aclose(self) -> None:
            events.append("language_model_closed")
            await super().aclose()

    class Database(ExampleDatabase):
        def bind_project(self, project_id: str) -> None:
            assert project_id == load_graph_project_id(tmp_path / "synthetic_run")
            events.append("bound")

        async def connect(self) -> None:
            events.append("connected")

        async def create_db_schema(self) -> None:
            events.append("schema")
            raise ValueError("Synthetic missing schema")

        async def close(self) -> None:
            events.append("database_closed")
            await super().close()

    monkeypatch.setattr(ProjectManager, "projects_root", tmp_path)
    language_model: Any = Metadata()
    database: Any = Database()
    with pytest.raises(ValueError, match="Synthetic missing schema"):
        async with managed_services(language_model_factory=lambda: language_model, database=database) as services:
            await run_quick_mode("Synthetic premise", services=services)
    assert events == ["metadata", "bound", "connected", "schema", "language_model_closed", "database_closed"]
    assert ProjectManager.load_config(tmp_path / "synthetic_run").title == "Synthetic Run"
    assert language_model.closed == database.closed == 1


@pytest.mark.parametrize("outcome", ["accepted", "schema_failure", "cancel"])
async def test_parser_command_bootstraps_frozen_receipt_with_one_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    import config
    from core.langgraph.initialization.staged_import import RECEIPT_QUERY, InitializationImport
    from core.parser_runner import run_parser_command
    from core.service_context import managed_services
    from tests.test_staged_initialization import example_state

    events: list[str] = []
    identity = ""
    state = example_state(tmp_path)

    class Outlines(ExampleLanguageModel):
        async def async_call_llm(self, **arguments: Any) -> tuple[str, dict[str, int]]:
            return "[]", {}

        async def aclose(self) -> None:
            events.append("language_model_closed")
            await super().aclose()

    class Database(ExampleDatabase):
        def bind_project(self, project_id: str) -> None:
            assert project_id == state["graph_project_id"]
            events.append("bound")

        async def connect(self) -> None:
            events.append("connected")

        async def create_db_schema(self) -> None:
            events.append("schema")
            if outcome == "schema_failure":
                raise ValueError("Synthetic missing schema")
            if outcome == "cancel":
                raise asyncio.CancelledError

        async def execute_read_query(self, query: str, parameters: dict[str, str]) -> list[dict[str, str]]:
            assert query == RECEIPT_QUERY
            assert parameters == {"project_id": state["graph_project_id"]}
            events.append("receipt")
            return [{"identity": identity}]

        async def close(self) -> None:
            events.append("database_closed")
            await super().close()

    monkeypatch.setattr(config, "ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    language_model: Any = Outlines()
    database: Any = Database()

    async def run() -> dict[str, tuple[bool, str]]:
        nonlocal identity
        async with managed_services(language_model_factory=lambda: language_model, database=database) as services:
            plan = await InitializationImport(str(tmp_path)).prepare(state)
            identity = plan.identity
            return await run_parser_command(str(tmp_path), None, services=services)

    if outcome == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await run()
    else:
        result = await run()
        if outcome == "accepted":
            assert result == {"initialization": (True, f"Accepted initialization {identity}")}
        else:
            assert result == {"initialization": (False, "Initialization import failed: Synthetic missing schema")}
    expected = ["bound", "connected", "schema"]
    if outcome == "accepted":
        expected += ["bound", "receipt"]
        assert (tmp_path / ".saga/initialization" / f"{identity}-accepted").read_text() == identity
    else:
        assert not (tmp_path / ".saga/initialization" / f"{identity}-accepted").exists()
    assert events == [*expected, "language_model_closed", "database_closed"]
    assert language_model.closed == database.closed == 1
