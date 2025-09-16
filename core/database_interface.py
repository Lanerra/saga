# core/database_interface.py
"""Minimal Neo4j database wrapper."""

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class Neo4jDatabase:
    """Thin wrapper around the shared ``neo4j_manager`` instance."""

    def __init__(self, manager: Any | None = None) -> None:
        # Import locally to avoid circular imports for modules that use both
        # ``core.db_manager`` and this wrapper.
        if manager is None:
            from core.db_manager import neo4j_manager

            manager = neo4j_manager

        self._manager = manager
        logger.debug(
            "Neo4jDatabase wrapper initialised", manager=type(self._manager).__name__
        )

    async def connect(self) -> None:
        """Establish a connection to Neo4j."""
        await self._manager.connect()

    async def close(self) -> None:
        """Close the Neo4j driver."""
        await self._manager.close()

    async def execute_read_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a read query and return the raw Neo4j records."""
        return await self._manager.execute_read_query(query, parameters)

    async def execute_write_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a write query and return the raw Neo4j records."""
        return await self._manager.execute_write_query(query, parameters)

    async def execute_cypher_batch(
        self, cypher_statements_with_params: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Execute a batch of Cypher statements in a single transaction."""
        await self._manager.execute_cypher_batch(cypher_statements_with_params)
