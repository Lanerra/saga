"""Ownership metadata only; arbitrary story queries fail instead of returning success."""
from typing import Any

from core.graph_ownership import OWNER_CONSTRAINT_QUERY, OWNER_INDEX_QUERY, OWNER_QUERY
from tests.fakes.schema_catalog import schema_catalog

PROJECT_ID = "11111111-1111-4111-8111-111111111111"


class OwnershipRows(list[dict[str, Any]]):
    def data(self) -> list[dict[str, Any]]:
        return list(self)

    def consume(self) -> None:
        pass


class OwnershipTransaction:
    def __init__(self, payload: Any = None) -> None:
        self.payload = payload
        self.owner = PROJECT_ID
        self.catalog = schema_catalog()

    def run(self, query: str, parameters: Any = None) -> Any:
        if query in self.catalog:
            return OwnershipRows(self.catalog[query])
        if query == "CALL db.awaitIndexes(60)":
            return OwnershipRows([])
        if query == OWNER_QUERY:
            return OwnershipRows([{"key": "exclusive", "project_id": self.owner, "version": 1}])
        if query == "RETURN 1 AS ownership_verified":
            return [{"ownership_verified": 1}]
        if self.payload is not None:
            if parameters is None:
                return self.payload.run(query)
            return self.payload.run(query, parameters)
        raise AssertionError(f"Ownership fake does not implement story query: {query}")

    def __getattr__(self, name: str) -> Any:
        if self.payload is None:
            raise AttributeError(name)
        return getattr(self.payload, name)


class OwnershipDriver:
    def __init__(self) -> None:
        self.transaction = OwnershipTransaction()

    def session(self, **arguments: Any) -> "OwnershipDriver":
        return self

    def __enter__(self) -> "OwnershipDriver":
        return self

    def __exit__(self, *arguments: Any) -> None:
        pass

    def run(self, query: str) -> Any:
        if query == OWNER_CONSTRAINT_QUERY:
            return OwnershipRows([{"name": "saga_graph_owner_unique", "type": "UNIQUENESS", "entityType": "NODE", "labelsOrTypes": ["SagaGraphOwner"], "properties": ["key"], "ownedIndex": "saga_graph_owner_unique"}])
        if query == OWNER_INDEX_QUERY:
            return OwnershipRows([{"name": "saga_graph_owner_unique", "type": "RANGE", "entityType": "NODE", "labelsOrTypes": ["SagaGraphOwner"], "properties": ["key"], "state": "ONLINE", "owningConstraint": "saga_graph_owner_unique"}])
        return self.transaction.run(query)

    def execute_read(self, callback: Any, *arguments: Any) -> Any:
        return callback(self.transaction, *arguments)

    def close(self) -> None:
        pass
