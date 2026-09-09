from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from neo4j import Driver

from core.db_manager import Neo4jManagerSingleton
from core.exceptions import DatabaseConnectionError
from core.service_context import get_services


class RecordingTransaction:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def run(self, query: str, parameters: Any = None) -> list[Any]:
        self.queries.append(query)
        return []

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def closed(self) -> bool:
        return False


class RecordingSession:
    def __init__(self, transaction: RecordingTransaction) -> None:
        self.transaction = transaction

    def __enter__(self) -> "RecordingSession":
        return self

    def __exit__(self, *arguments: Any) -> None:
        pass

    def begin_transaction(self) -> RecordingTransaction:
        return self.transaction

    def execute_read(self, callback: Any, *arguments: Any) -> Any:
        return callback(self.transaction, *arguments)

    execute_write = execute_read


class RecordingDriver:
    def __init__(self) -> None:
        self.transaction = RecordingTransaction()

    def session(self, **arguments: Any) -> RecordingSession:
        return RecordingSession(self.transaction)


def isolated_manager() -> Neo4jManagerSingleton:
    manager = object.__new__(Neo4jManagerSingleton)
    manager._initialized_flag = False
    Neo4jManagerSingleton.__init__(manager)
    return manager


@pytest.mark.parametrize("operation", ["read", "write", "batch", "transaction", "schema"])
async def test_unbound_manager_rejects_active_operations(operation: str) -> None:
    manager = isolated_manager()
    driver = RecordingDriver()
    manager.driver = cast(Driver, driver)
    with pytest.raises(DatabaseConnectionError, match="project"):
        if operation == "read":
            await manager.execute_read_query("MATCH (c:Chapter {number: 1}) RETURN c")
        elif operation == "write":
            await manager.execute_write_query("MERGE (c:Chapter {number: 1})")
        elif operation == "batch":
            await manager.execute_cypher_batch([("MERGE (c:Chapter {number: 1})", {})])
        elif operation == "transaction":
            await manager.execute_in_transaction(lambda transaction: transaction.run("MATCH (n) DETACH DELETE n"))
        else:
            manager._execute_schema_batch(["CREATE INDEX chapter_name FOR (n:Chapter) ON (n.name)"])
    assert driver.transaction.queries == []


def test_binding_cannot_switch_projects() -> None:
    manager = isolated_manager()
    manager.bind_project("11111111-1111-4111-8111-111111111111")
    with pytest.raises(DatabaseConnectionError, match="project"):
        manager.bind_project("22222222-2222-4222-8222-222222222222")


@pytest.mark.parametrize("operation", ["capability", "property"])
async def test_unbound_metadata_cache_is_not_admitted(operation: str) -> None:
    import time

    manager = isolated_manager()
    manager._apoc_available_cache = True
    manager._property_keys_cache = {"name"}
    manager._property_keys_cache_ts = time.monotonic()
    with pytest.raises(DatabaseConnectionError, match="project"):
        if operation == "capability":
            await manager.is_apoc_available()
        else:
            await manager.has_property_key("name")


@pytest.mark.parametrize("operation", ["capability", "property"])
async def test_metadata_cache_rejects_restored_owner_mismatch(operation: str) -> None:
    import time

    from tests.fakes.graph_ownership import PROJECT_ID, OwnershipDriver

    manager = isolated_manager()
    manager.bind_project(PROJECT_ID)
    driver = OwnershipDriver()
    driver.transaction.owner = "22222222-2222-4222-8222-222222222222"
    manager.driver = cast(Driver, driver)
    manager._apoc_available_cache = True
    manager._property_keys_cache = {"name"}
    manager._property_keys_cache_ts = time.monotonic()
    with pytest.raises(DatabaseConnectionError, match="project"):
        if operation == "capability":
            await manager.is_apoc_available()
        else:
            await manager.has_property_key("name")


@pytest.mark.parametrize("module_name,function_name,arguments", [
    ("character_queries", "get_character_profile_by_name", ("Missing",)),
    ("character_queries", "get_character_profile_by_id", ("missing-character",)),
    ("world_queries", "get_world_item_by_id", ("missing-location",)),
    ("kg_queries", "query_kg_from_db", ("Missing",)),
    ("kg_queries", "get_novel_info_property_from_db", ("theme",)),
    ("plot_queries", "get_plot_outline_from_db", ()),
])
async def test_cached_graph_reads_reject_owner_change(module_name: str, function_name: str, arguments: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib import import_module

    from tests.fakes.graph_ownership import PROJECT_ID, OwnershipDriver

    manager = isolated_manager()
    manager.bind_project(PROJECT_ID)
    driver = OwnershipDriver()
    driver.transaction.payload = RecordingTransaction()
    manager.driver = cast(Driver, driver)
    module = import_module(f"data_access.{module_name}")
    monkeypatch.setattr(get_services(), 'database', manager)
    monkeypatch.setattr(get_services(), 'database', manager)
    function = getattr(module, function_name)
    function.cache_clear()
    await function(*arguments)
    driver.transaction.owner = "22222222-2222-4222-8222-222222222222"
    with pytest.raises(DatabaseConnectionError, match="project"):
        await function(*arguments)
    function.cache_clear()


async def test_close_retains_binding_and_rejects_target_change(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.fakes.graph_ownership import PROJECT_ID

    manager = isolated_manager()
    manager.bind_project(PROJECT_ID)
    await manager.close()
    assert manager.require_project_binding() == PROJECT_ID
    monkeypatch.setattr("config.NEO4J_DATABASE", "other-story")
    with pytest.raises(DatabaseConnectionError, match="target changed"):
        await manager.execute_read_query("MATCH (n) RETURN n")


@pytest.mark.parametrize("identity", ["", "Same Story", "11111111-1111-4111-8111-11111111111A"])
def test_malformed_identity_is_not_replaced(tmp_path: Path, identity: str) -> None:
    from core.graph_ownership import load_graph_project_id

    (tmp_path / "graph-project-id").write_text(identity)
    with pytest.raises(ValueError):
        load_graph_project_id(tmp_path)
    assert (tmp_path / "graph-project-id").read_text() == identity


def test_concurrent_identity_creation_returns_one_durable_identity(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from core.graph_ownership import load_graph_project_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        identities = list(executor.map(load_graph_project_id, [tmp_path] * 8))
    assert identities == [load_graph_project_id(tmp_path)] * 8


async def test_bootstrap_binds_before_connection_and_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.graph_ownership import load_graph_project_id
    from orchestration.langgraph_orchestrator import LangGraphOrchestrator

    events: list[str] = []

    class BootstrapManager:
        def bind_project(self, identity: str) -> None:
            assert identity == load_graph_project_id(tmp_path)
            events.append("bind")

        async def connect(self) -> None:
            assert events == ["bind"]
            events.append("connect")

        async def create_db_schema(self) -> None:
            assert events == ["bind", "connect"]
            events.append("schema")

    orchestrator = object.__new__(LangGraphOrchestrator)
    orchestrator.project_dir = tmp_path
    monkeypatch.setattr(get_services(), 'database', BootstrapManager())
    await orchestrator._ensure_neo4j_connection()
    assert events == ["bind", "connect", "schema"]


def test_equal_display_names_have_distinct_durable_graph_identities(tmp_path: Path) -> None:
    from core.graph_ownership import load_graph_project_id

    first = tmp_path / "first" / "Same Story"
    second = tmp_path / "second" / "Same Story"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    first_identity = load_graph_project_id(first)
    second_identity = load_graph_project_id(second)
    assert str(UUID(first_identity)) == first_identity
    assert first_identity != second_identity
    assert load_graph_project_id(first) == first_identity
    restored = tmp_path / "restored"
    first.rename(restored)
    assert load_graph_project_id(restored) == first_identity


@pytest.mark.parametrize("version", [True, False, 1.0, "1", None, 2])
def test_marker_version_requires_exact_integer(version: Any) -> None:
    from neo4j import ManagedTransaction

    from core.graph_ownership import GraphOwnershipError, assert_graph_owner
    from tests.fakes.graph_ownership import PROJECT_ID

    class MarkerTransaction:
        def run(self, query: str) -> list[dict[str, Any]]:
            return [{"key": "exclusive", "project_id": PROJECT_ID, "version": version}]

    with pytest.raises(GraphOwnershipError):
        assert_graph_owner(cast(ManagedTransaction, MarkerTransaction()), PROJECT_ID)


@pytest.mark.parametrize("owned,defect", [(owned, defect) for owned in [False, True] for defect in ["wrong_label", "wrong_property", "wrong_type", "wrong_entity", "index_only", "offline", "wrong_backing"]] + [(True, "missing")])
def test_claim_rejects_ineffective_schema_without_changes(owned: bool, defect: str) -> None:
    from core.graph_ownership import GraphOwnershipError
    from tests.fakes.graph_ownership import PROJECT_ID, OwnershipDriver, OwnershipRows

    constraint: dict[str, Any] = {"name": "saga_graph_owner_unique", "type": "UNIQUENESS", "entityType": "NODE", "labelsOrTypes": ["SagaGraphOwner"], "properties": ["key"], "ownedIndex": "saga_graph_owner_unique"}
    index: dict[str, Any] = {"name": "saga_graph_owner_unique", "type": "RANGE", "entityType": "NODE", "labelsOrTypes": ["SagaGraphOwner"], "properties": ["key"], "state": "ONLINE", "owningConstraint": "saga_graph_owner_unique"}
    constraints = [constraint]
    indexes = [index]
    if defect == "wrong_label":
        constraint["labelsOrTypes"] = ["Unrelated"]
    elif defect == "wrong_property":
        constraint["properties"] = ["other"]
    elif defect == "wrong_type":
        constraint["type"] = "NODE_PROPERTY_EXISTENCE"
    elif defect == "wrong_entity":
        constraint["entityType"] = "RELATIONSHIP"
    elif defect == "index_only":
        constraints = []
        index["owningConstraint"] = None
    elif defect == "missing":
        constraints = []
        indexes = []
    elif defect == "offline":
        index["state"] = "POPULATING"
    else:
        constraint["ownedIndex"] = "other"

    class SchemaRows(OwnershipRows):
        def single(self, *, strict: bool) -> dict[str, Any]:
            assert len(self) == 1
            return self[0]

        def consume(self) -> None:
            pass

    class SchemaDriver(OwnershipDriver):
        def __init__(self) -> None:
            super().__init__()
            self.changes: list[str] = []

        def run(self, query: str) -> Any:
            if query.startswith("SHOW CONSTRAINTS"):
                return SchemaRows(constraints)
            if query.startswith("SHOW INDEXES"):
                return SchemaRows(indexes)
            if query.startswith("MATCH (owner:"):
                return super().run(query) if owned else SchemaRows()
            if query == "MATCH (n) RETURN count(n) AS count":
                return SchemaRows([{"count": 0}])
            self.changes.append(query)
            return SchemaRows()

        def execute_write(self, callback: Any, *arguments: Any) -> None:
            self.changes.append("claim")

    manager = isolated_manager()
    manager.bind_project(PROJECT_ID)
    driver = SchemaDriver()
    manager.driver = cast(Driver, driver)
    with pytest.raises(GraphOwnershipError, match="constraint"):
        manager._sync_claim_project()
    assert driver.changes == []
