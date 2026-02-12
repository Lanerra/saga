# tests/fakes/fake_neo4j_manager.py
"""In-memory fake of Neo4jManagerSingleton for testing without a real database.

This fake records all executed queries and supports configurable responses
via regex pattern matching. It replaces patch-based mocking of neo4j_manager
with a lightweight, injectable alternative.
"""

from __future__ import annotations

import re
from typing import Any


class FakeNeo4jManager:
    """In-memory fake implementing the Neo4jManagerSingleton public async API.

    Provides query recording, configurable read responses, and assertion helpers
    for verifying database interactions in tests.
    """

    def __init__(self) -> None:
        self.executed_queries: list[tuple[str, dict[str, Any] | None]] = []
        self.batch_statements: list[list[tuple[str, dict[str, Any]]]] = []
        self._configured_responses: list[tuple[re.Pattern[str], list[dict[str, Any]]]] = []
        self._apoc_available: bool = True

    def configure_response(self, query_pattern: str, response: list[dict[str, Any]]) -> None:
        """Register a regex pattern that returns the given response for matching read queries."""
        self._configured_responses.append((re.compile(query_pattern, re.IGNORECASE), response))

    async def execute_read_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.executed_queries.append((query, parameters))
        for pattern, response in self._configured_responses:
            if pattern.search(query):
                return response
        return []

    async def execute_write_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.executed_queries.append((query, parameters))
        for pattern, response in self._configured_responses:
            if pattern.search(query):
                return response
        return []

    async def execute_cypher_batch(self, cypher_statements_with_params: list[tuple[str, dict[str, Any]]]) -> None:
        self.batch_statements.append(list(cypher_statements_with_params))
        for query, params in cypher_statements_with_params:
            self.executed_queries.append((query, params))

    async def execute_in_transaction(
        self,
        transaction_func: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return None

    async def is_apoc_available(self, *, log_warning_once: bool = False) -> bool:
        return self._apoc_available

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def refresh_property_keys_cache(self) -> set[str]:
        return set()

    async def has_property_key(self, key: str, max_age_seconds: int = 300) -> bool:
        return False

    @staticmethod
    def embedding_to_list(embedding: Any) -> list[float]:
        """Convert an embedding to a plain list (mirrors the real manager)."""
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        if isinstance(embedding, list):
            return embedding
        return list(embedding)

    def assert_query_executed(self, pattern: str) -> tuple[str, dict[str, Any] | None]:
        """Assert that at least one executed query matches the given regex pattern.

        Returns the first matching (query, parameters) tuple.
        Raises AssertionError if no match is found.
        """
        compiled = re.compile(pattern, re.IGNORECASE)
        for query, params in self.executed_queries:
            if compiled.search(query):
                return query, params
        executed_summary = "\n".join(f"  {q[:120]}" for q, _ in self.executed_queries) or "  (none)"
        raise AssertionError(f"No executed query matched pattern {pattern!r}.\n" f"Executed queries:\n{executed_summary}")

    def assert_batch_contains(self, pattern: str) -> tuple[str, dict[str, Any]]:
        """Assert that at least one batch contained a statement matching the pattern.

        Returns the first matching (query, parameters) tuple.
        Raises AssertionError if no match is found.
        """
        compiled = re.compile(pattern, re.IGNORECASE)
        for batch in self.batch_statements:
            for query, params in batch:
                if compiled.search(query):
                    return query, params
        raise AssertionError(f"No batch statement matched pattern {pattern!r}.")

    def get_executed_queries(self) -> list[tuple[str, dict[str, Any] | None]]:
        """Return all recorded queries."""
        return list(self.executed_queries)

    def reset(self) -> None:
        """Clear all recorded queries and configured responses."""
        self.executed_queries.clear()
        self.batch_statements.clear()
        self._configured_responses.clear()
