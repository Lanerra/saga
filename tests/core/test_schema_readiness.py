"""Production write admission with a narrow catalog transport."""
from typing import Any, cast

import pytest
from neo4j import Driver

from core.db_manager import Neo4jManagerSingleton
from core.exceptions import DatabaseError
from core.graph_ownership import OWNER_QUERY
from core.schema_readiness import CONSTRAINT_QUERY, FUNCTION_QUERY, INDEX_QUERY, PROCEDURE_QUERY
from tests.fakes.graph_ownership import PROJECT_ID, OwnershipRows
from tests.fakes.schema_catalog import schema_catalog


class CatalogTransport:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self.rolled_back = False
        self.catalog: dict[str, list[dict[str, Any]]] = {}

    def run(self, query: str, parameters: Any = None) -> OwnershipRows:
        if query == OWNER_QUERY:
            return OwnershipRows([{"key": "exclusive", "project_id": PROJECT_ID, "version": 1}])
        if query.startswith("SHOW "):
            return OwnershipRows(self.catalog.get(query, []))
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
