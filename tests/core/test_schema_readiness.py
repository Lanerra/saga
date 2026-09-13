"""Production write admission with a narrow catalog transport."""
import json
from pathlib import Path
from typing import Any, cast

import pytest
from neo4j import Driver

from core.db_manager import Neo4jManagerSingleton
from core.exceptions import DatabaseError
from core.graph_ownership import OWNER_QUERY
from core.schema_readiness import CONSTRAINT_QUERY, FUNCTION_QUERY, INDEX_QUERY, PROCEDURE_QUERY
from tests.fakes.graph_ownership import PROJECT_ID, OwnershipRows
from tests.fakes.schema_catalog import schema_catalog


class InitializationRows(OwnershipRows):
    def single(self, *, strict: bool) -> dict[str, Any]:
        assert strict and len(self) == 1
        return self[0]


class CatalogTransport:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self.rolled_back = False
        self.catalog: dict[str, list[dict[str, Any]]] = {}

    def run(self, query: str, parameters: Any = None) -> OwnershipRows:
        if query == OWNER_QUERY:
            return OwnershipRows([{"key": "exclusive", "project_id": PROJECT_ID, "version": 1}])
        if query == "MATCH (owner:SagaGraphOwner {key: 'exclusive', project_id: $project_id}) SET owner.version = owner.version":
            assert parameters == {"project_id": PROJECT_ID}
            return OwnershipRows([])
        if query in schema_catalog():
            return OwnershipRows(self.catalog.get(query, []))
        if query == "CALL db.awaitIndexes(60)":
            return OwnershipRows([])
        if query.startswith("CREATE "):
            self.writes.append(query)
            return OwnershipRows([])
        raise AssertionError(query)

    def session(self, **arguments: Any) -> "CatalogTransport":
        return self

    def __enter__(self) -> "CatalogTransport":
        return self

    def __exit__(self, *arguments: Any) -> None:
        pass

    def begin_transaction(self) -> "CatalogTransport":
        return self

    def execute_write(self, callback: Any, *arguments: Any) -> Any:
        return callback(self, *arguments)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        self.rolled_back = True

    def closed(self) -> bool:
        return False


def manager_with_catalog(catalog: CatalogTransport) -> Neo4jManagerSingleton:
    manager = object.__new__(Neo4jManagerSingleton)
    manager._initialized_flag = False
    Neo4jManagerSingleton.__init__(manager)
    manager.bind_project(PROJECT_ID)
    manager.driver = cast(Driver, catalog)
    return manager


@pytest.mark.parametrize("operation", ["query", "batch", "callback"])
async def test_absent_schema_rejects_before_payload(operation: str) -> None:
    catalog = CatalogTransport()
    manager = manager_with_catalog(catalog)
    with pytest.raises(DatabaseError, match="[Ss]chema|prerequisite"):
        if operation == "query":
            await manager.execute_write_query("CREATE (:Chapter {number: 1})")
        elif operation == "batch":
            await manager.execute_cypher_batch([("CREATE (:Chapter {number: 1})", {})])
        else:
            await manager.execute_in_transaction(lambda transaction: transaction.run("CREATE (:Chapter {number: 1})"))
    assert catalog.writes == []


@pytest.mark.parametrize("catalog_query,field,value", [
    (CONSTRAINT_QUERY, "type", "NODE_PROPERTY_EXISTENCE"),
    (CONSTRAINT_QUERY, "labelsOrTypes", ["Decoy"]),
    (CONSTRAINT_QUERY, "properties", ["other"]),
    (CONSTRAINT_QUERY, "ownedIndex", "other"),
    (INDEX_QUERY, "state", "POPULATING"),
    (INDEX_QUERY, "state", "FAILED"),
    (INDEX_QUERY, "owningConstraint", None),
    (INDEX_QUERY, "type", "TEXT"),
])
async def test_ineffective_constraint_blocks_payload(catalog_query: str, field: str, value: Any) -> None:
    catalog = CatalogTransport()
    catalog.catalog = schema_catalog()
    row = next(row for row in catalog.catalog[catalog_query] if row["name"] == "character_id_unique")
    row[field] = value
    with pytest.raises(DatabaseError, match="schema prerequisites"):
        await manager_with_catalog(catalog).execute_write_query("CREATE (:Character {id: 'ada'})")
    assert catalog.writes == []


@pytest.mark.parametrize("field,value", [
    ("type", "RANGE"), ("labelsOrTypes", ["Character"]), ("properties", ["other"]),
    ("state", "POPULATING"), ("state", "FAILED"),
    ("options", {"indexConfig": {"vector.dimensions": 1, "vector.similarity_function": "COSINE"}}),
    ("options", {"indexConfig": {"vector.dimensions": True, "vector.similarity_function": "COSINE"}}),
    ("options", {"indexConfig": {"vector.dimensions": 768, "vector.similarity_function": "EUCLIDEAN"}}),
])
async def test_ineffective_vector_blocks_payload(field: str, value: Any) -> None:
    catalog = CatalogTransport()
    catalog.catalog = schema_catalog()
    catalog.catalog[INDEX_QUERY][-1][field] = value
    with pytest.raises(DatabaseError, match="schema prerequisites"):
        await manager_with_catalog(catalog).execute_write_query("CREATE (:Chapter {number: 1})")
    assert catalog.writes == []


@pytest.mark.parametrize("catalog_query,name", [(PROCEDURE_QUERY, "apoc.merge.relationship"), (FUNCTION_QUERY, "apoc.util.sha256")])
async def test_version_alone_cannot_admit_missing_capability(catalog_query: str, name: str) -> None:
    catalog = CatalogTransport()
    catalog.catalog = schema_catalog()
    catalog.catalog[catalog_query] = [row for row in catalog.catalog[catalog_query] if row["name"] != name]
    with pytest.raises(DatabaseError, match="executable graph prerequisites"):
        await manager_with_catalog(catalog).execute_write_query("CREATE (:Character {id: 'ada'})")
    assert catalog.writes == []


async def test_effective_catalog_admits_payload() -> None:
    catalog = CatalogTransport()
    catalog.catalog = schema_catalog()
    await manager_with_catalog(catalog).execute_write_query("CREATE (:Character {id: 'ada'})")
    assert catalog.writes == ["CREATE (:Character {id: 'ada'})"]


class InitializationTransport(CatalogTransport):
    def __init__(self) -> None:
        super().__init__()
        self.catalog = schema_catalog()
        self.identity: str | None = None
        self.payloads: list[tuple[str, dict[str, Any]]] = []
        self.expected_payloads: list[tuple[str, dict[str, Any]]] = []
        self.entity_ids: set[str] = set()

    def execute_read(self, callback: Any, *arguments: Any) -> Any:
        return callback(self, *arguments)

    def run(self, query: str, parameters: Any = None, **keywords: Any) -> OwnershipRows:
        arguments = parameters if parameters is not None else keywords
        if "RETURN owner.initialization_plan AS identity" in query:
            return InitializationRows([{"identity": self.identity}])
        if "WHERE NOT n:SagaGraphOwner" in query:
            return InitializationRows([{"count": 0}])
        if "RETURN count(n) AS count" in query:
            assert arguments["identity"] in self.entity_ids
            return InitializationRows([{"count": 1}])
        if "SET owner.initialization_plan = $identity" in query:
            self.identity = arguments["identity"]
            return OwnershipRows([])
        if (query, arguments) in self.expected_payloads:
            self.payloads.append((query, arguments))
            return OwnershipRows([])
        return super().run(query, parameters)


@pytest.mark.parametrize("installed", ["fresh", "event_name_unique", "custom_event_name", "custom_event_key", "unrelated"])
async def test_catalog_acceptance_checks_effective_event_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, installed: str) -> None:
    from core.langgraph.content_manager import ContentManager
    from core.langgraph.initialization.staged_import import InitializationImport
    from core.schema_readiness import CONSTRAINTS, required_statements
    from core.service_context import get_services
    from tests.test_initialization_catalog import SyntheticSelector
    from tests.test_staged_initialization import example_state, with_catalog

    state = example_state(tmp_path)
    state["graph_project_id"] = PROJECT_ID
    manager = ContentManager(str(tmp_path))
    assert state["global_outline_ref"] is not None
    outline = manager.load_json_strict(state["global_outline_ref"])
    outline["inciting_incident"] = outline["midpoint"] = "Ada returns to the weather station " * 20
    state["global_outline_ref"] = manager.save_json(outline, "global_outline", "duplicate", version=2)
    selector = SyntheticSelector()
    monkeypatch.setattr(get_services().language_model, "async_call_llm", selector)
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    importer = InitializationImport(str(tmp_path))
    plan = await importer.prepare(with_catalog(state))
    events = [json.loads(entity.payload) for entity in plan.entities if entity.label == "Event" and json.loads(entity.payload)["name"] == outline["midpoint"]]
    assert len(events) == 2 and events[0]["id"] != events[1]["id"]
    transport = InitializationTransport()
    transport.expected_payloads = [(statement.query, json.loads(statement.parameters)) for statement in plan.statements]
    transport.entity_ids = {entity.identity for entity in plan.entities}
    if installed != "fresh":
        transport.catalog[CONSTRAINT_QUERY].append({
            "name": installed, "type": "NODE_KEY" if installed == "custom_event_key" else "UNIQUENESS", "entityType": "NODE",
            "labelsOrTypes": ["Event" if installed != "unrelated" else "UserLabel"], "properties": ["name"], "ownedIndex": installed,
        })
    monkeypatch.setattr(get_services(), "database", manager_with_catalog(transport))
    calls = len(selector.prompts)
    if installed not in {"fresh", "unrelated"}:
        with pytest.raises(DatabaseError) as failure:
            await importer.accept(plan.identity)
        assert "Catalog initialization v2" in str(failure.value.__cause__)
        assert installed in str(failure.value.__cause__)
        assert transport.payloads == [] and transport.identity is None
        assert transport.rolled_back is True
    else:
        assert ("event_id_unique", "Event", ("id",)) in CONSTRAINTS
        assert not any(label == "Event" and properties == ("name",) for _, label, properties in CONSTRAINTS)
        assert any("`event_id_unique`" in statement for statement in required_statements())
        assert await importer.accept(plan.identity) == plan
        assert transport.payloads == [(statement.query, json.loads(statement.parameters)) for statement in plan.statements]
        assert any("relationship:HAPPENS_BEFORE" in query for query, _ in transport.payloads)
        accepted = list(transport.payloads)
        edited = tmp_path / "world/history.yaml"
        edited.write_text("Accepted author edit\n")
        assert await importer.accept(plan.identity) == plan
        assert transport.payloads == accepted
        assert edited.read_text() == "Accepted author edit\n"
    assert len(selector.prompts) == calls


async def test_missing_event_id_constraint_still_blocks_catalog_payload() -> None:
    transport = CatalogTransport()
    transport.catalog = schema_catalog()
    transport.catalog[CONSTRAINT_QUERY] = [row for row in transport.catalog[CONSTRAINT_QUERY] if row["name"] != "event_id_unique"]
    with pytest.raises(DatabaseError, match="schema prerequisites"):
        await manager_with_catalog(transport).execute_write_query("CREATE (:Event {id: 'event_a'})")
    assert transport.writes == []


async def test_bootstrap_never_installs_event_name_uniqueness() -> None:
    transport = InitializationTransport()
    await manager_with_catalog(transport)._create_constraints_and_indexes()
    assert any("event_id_unique" in query for query in transport.writes)
    assert not any("event_name_unique" in query for query in transport.writes)
