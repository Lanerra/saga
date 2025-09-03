# core/db_manager.py
# core_db/base_db_manager.py
from typing import Any

import numpy as np
import structlog
from models.kg_constants import NODE_LABELS, RELATIONSHIP_TYPES

import config
from core.exceptions import (
    DatabaseConnectionError,
    DatabaseTransactionError,
    handle_database_error
)
from neo4j import (  # type: ignore
    AsyncDriver,
    AsyncGraphDatabase,
    AsyncManagedTransaction,
)
from neo4j.exceptions import ServiceUnavailable  # type: ignore

logger = structlog.get_logger(__name__)


class Neo4jManagerSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Neo4jManagerSingleton, cls).__new__(cls)
            cls._instance._initialized_flag = False
        return cls._instance

    def __init__(self):
        if self._initialized_flag:
            return

        self.logger = structlog.get_logger(__name__)
        self.driver: AsyncDriver | None = None
        self._initialized_flag = True
        self.logger.info(
            "Neo4jManagerSingleton initialized. Call connect() to establish connection."
        )

    async def connect(self):
        if self.driver:
            self.logger.info(
                "Existing driver instance found. Attempting to close it before creating a new connection."
            )
            try:
                await self.driver.close()
            except Exception as close_error:
                self.logger.warning(
                    f"Error closing existing driver (it might have been already closed or invalid): {close_error}"
                )
            finally:
                self.driver = None

        try:
            self.driver = AsyncGraphDatabase.driver(
                config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
            )
            await self.driver.verify_connectivity()
            self.logger.info(f"Successfully connected to Neo4j at {config.NEO4J_URI}")
        except ServiceUnavailable as e:
            self.logger.critical(
                "Neo4j service unavailable",
                uri=config.NEO4J_URI,
                error=str(e)
            )
            self.driver = None
            raise DatabaseConnectionError(
                "Neo4j database is not available",
                details={
                    "uri": config.NEO4J_URI,
                    "original_error": str(e),
                    "suggestion": "Ensure the Neo4j database is running and accessible"
                }
            )
        except Exception as e:
            self.logger.critical(
                "Unexpected error during Neo4j connection",
                uri=config.NEO4J_URI,
                error=str(e),
                exc_info=True
            )
            self.driver = None
            raise handle_database_error("connection", e, uri=config.NEO4J_URI)

    async def close(self):
        if self.driver:
            try:
                await self.driver.close()
                self.logger.info("Neo4j driver closed.")
            except Exception as e:
                self.logger.error(
                    f"Error while closing Neo4j driver: {e}", exc_info=True
                )
            finally:
                self.driver = None
        else:
            self.logger.info("No active Neo4j driver to close (driver was None).")

    async def _ensure_connected(self):
        if self.driver is None:
            self.logger.info("Driver is None, attempting to connect.")
            await self.connect()

        if self.driver is None:
            raise DatabaseConnectionError(
                "Neo4j driver not initialized",
                details={
                    "suggestion": "Call connect() method first to establish database connection"
                }
            )

    async def _execute_query_tx(
        self,
        tx: AsyncManagedTransaction,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.logger.debug(f"Executing Cypher query: {query} with params: {parameters}")
        result_cursor = await tx.run(query, parameters)
        return await result_cursor.data()

    async def execute_read_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        await self._ensure_connected()
        async with self.driver.session(database=config.NEO4J_DATABASE) as session:  # type: ignore
            return await session.execute_read(self._execute_query_tx, query, parameters)

    async def execute_write_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        await self._ensure_connected()
        async with self.driver.session(database=config.NEO4J_DATABASE) as session:  # type: ignore
            return await session.execute_write(
                self._execute_query_tx, query, parameters
            )

    async def execute_cypher_batch(
        self, cypher_statements_with_params: list[tuple[str, dict[str, Any]]]
    ):
        if not cypher_statements_with_params:
            self.logger.info("execute_cypher_batch: No statements to execute.")
            return

        await self._ensure_connected()
        async with self.driver.session(database=config.NEO4J_DATABASE) as session:  # type: ignore
            tx = await session.begin_transaction()
            try:
                for query, params in cypher_statements_with_params:
                    self.logger.debug(f"Batch Cypher: {query} with params {params}")
                    await tx.run(query, params)  # type: ignore
                await tx.commit()  # type: ignore
                self.logger.info(
                    f"Successfully executed batch of {len(cypher_statements_with_params)} Cypher statements."
                )
            except Exception as e:
                self.logger.error(
                    "Error in Cypher batch execution",
                    batch_size=len(cypher_statements_with_params),
                    error=str(e),
                    exc_info=True,
                )
                # Use the same clean rollback pattern as _execute_schema_batch
                if tx and not tx.closed():  # type: ignore
                    await tx.rollback()  # type: ignore
                raise DatabaseTransactionError(
                    "Batch Cypher execution failed",
                    details={
                        "batch_size": len(cypher_statements_with_params),
                        "original_error": str(e),
                        "operation": "batch_execution"
                    }
                )

    async def create_db_schema(self) -> None:
        """Create and verify Neo4j indexes and constraints in separate phases to avoid transaction conflicts.

        Phase 1: Schema-only operations (constraints, indexes)
        Phase 2: Data operations (relationship and node type placeholders)
        """
        self.logger.info(
            "Creating/verifying Neo4j schema elements (phased execution)..."
        )

        # Phase 1: Schema-only operations (constraints and indexes)
        await self._create_constraints_and_indexes()

        # Phase 2: Data operations (type placeholders)
        await self._create_type_placeholders()

        self.logger.info(
            "Neo4j schema (indexes, constraints, labels, relationship types, vector index) verification process complete."
        )

    async def _create_constraints_and_indexes(self) -> None:
        """Create constraints, indexes, and vector index in schema-only transactions."""
        self.logger.info("Phase 1: Creating constraints and indexes...")

        core_constraints_queries = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT novelInfo_id_unique IF NOT EXISTS FOR (n:NovelInfo) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT chapter_number_unique IF NOT EXISTS FOR (c:Chapter) REQUIRE c.number IS UNIQUE",
            "CREATE CONSTRAINT character_name_unique IF NOT EXISTS FOR (char:Character) REQUIRE char.name IS UNIQUE",
            "CREATE CONSTRAINT worldElement_id_unique IF NOT EXISTS FOR (we:WorldElement) REQUIRE we.id IS UNIQUE",
            "CREATE CONSTRAINT worldContainer_id_unique IF NOT EXISTS FOR (wc:WorldContainer) REQUIRE wc.id IS UNIQUE",
            "CREATE CONSTRAINT trait_name_unique IF NOT EXISTS FOR (t:Trait) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT plotPoint_id_unique IF NOT EXISTS FOR (pp:PlotPoint) REQUIRE pp.id IS UNIQUE",
            "CREATE CONSTRAINT valueNode_value_type_unique IF NOT EXISTS FOR (vn:ValueNode) REQUIRE (vn.value, vn.type) IS UNIQUE",
            "CREATE CONSTRAINT developmentEvent_id_unique IF NOT EXISTS FOR (dev:DevelopmentEvent) REQUIRE dev.id IS UNIQUE",
            "CREATE CONSTRAINT worldElaborationEvent_id_unique IF NOT EXISTS FOR (elab:WorldElaborationEvent) REQUIRE elab.id IS UNIQUE",
        ]

        index_queries = [
            "CREATE INDEX entity_name_property_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_is_provisional_idx IF NOT EXISTS FOR (e:Entity) ON (e.is_provisional)",
            "CREATE INDEX entity_is_deleted_idx IF NOT EXISTS FOR (e:Entity) ON (e.is_deleted)",
            "CREATE INDEX plotPoint_sequence IF NOT EXISTS FOR (pp:PlotPoint) ON (pp.sequence)",
            "CREATE INDEX developmentEvent_chapter_updated IF NOT EXISTS FOR (d:DevelopmentEvent) ON (d.chapter_updated)",
            "CREATE INDEX worldElaborationEvent_chapter_updated IF NOT EXISTS FOR (we:WorldElaborationEvent) ON (we.chapter_updated)",
            "CREATE INDEX worldElement_category IF NOT EXISTS FOR (we:WorldElement) ON (we.category)",
            "CREATE INDEX worldElement_name_property_idx IF NOT EXISTS FOR (we:WorldElement) ON (we.name)",
            "CREATE INDEX chapter_is_provisional IF NOT EXISTS FOR (c:`Chapter`) ON (c.is_provisional)",
        ]

        vector_index_query = f"""
        CREATE VECTOR INDEX {config.NEO4J_VECTOR_INDEX_NAME} IF NOT EXISTS
        FOR (c:Chapter) ON (c.embedding_vector)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {config.NEO4J_VECTOR_DIMENSIONS},
            `vector.similarity_function`: '{config.NEO4J_VECTOR_SIMILARITY_FUNCTION}'
        }}}}
        """

        schema_only_queries = (
            core_constraints_queries + index_queries + [vector_index_query]
        )

        try:
            await self._execute_schema_batch(schema_only_queries)
            self.logger.info(
                f"Phase 1 complete: Successfully created {len(schema_only_queries)} schema elements."
            )
        except Exception as e:
            self.logger.error(
                f"Phase 1 batch failed: {e}. Attempting individual operations...",
                exc_info=True,
            )
            await self._execute_schema_individually(schema_only_queries)

    async def _create_type_placeholders(self) -> None:
        """Create relationship type and node label placeholders in separate data transactions."""
        self.logger.info("Phase 2: Creating type placeholders...")

        # Ensure schema tokens exist to avoid Neo4j warnings when
        # matching on types that have not been used yet. Creating
        # and immediately deleting a dummy node/relationship is sufficient.
        # Note: Since RELATIONSHIP_TYPES and NODE_LABELS are constants from our codebase,
        # injection risk is minimal, but we use this approach for security best practice.
        relationship_type_queries = []
        for rel_type in RELATIONSHIP_TYPES:
            # Use string formatting safely with validated constants
            query = (
                f"CREATE (a:__RelTypePlaceholder)-[:{rel_type}]->"
                f"(b:__RelTypePlaceholder) WITH a,b DETACH DELETE a,b"
            )
            relationship_type_queries.append(query)
        
        node_label_queries = []
        for label in NODE_LABELS:
            # Use backticks to handle labels with special characters safely
            query = f"CREATE (a:`{label}`) WITH a DELETE a"
            node_label_queries.append(query)

        data_operations = relationship_type_queries + node_label_queries
        data_ops_with_params: list[tuple[str, dict[str, Any]]] = [
            (query, {}) for query in data_operations
        ]

        try:
            await self.execute_cypher_batch(data_ops_with_params)
            self.logger.info(
                f"Phase 2 complete: Successfully created {len(data_operations)} type placeholders."
            )
        except Exception as e:
            self.logger.error(
                f"Phase 2 batch failed: {e}. Attempting individual operations...",
                exc_info=True,
            )
            for query_text in data_operations:
                try:
                    await self.execute_write_query(query_text)
                    self.logger.debug(
                        "Phase 2 fallback: Successfully created type placeholder."
                    )
                except Exception as individual_e:
                    self.logger.warning(
                        f"Phase 2 fallback: Failed to create type placeholder: {individual_e}"
                    )

    async def _execute_schema_batch(self, queries: list[str]) -> None:
        """Execute schema operations in isolation (no data writes)."""
        await self._ensure_connected()
        async with self.driver.session(database=config.NEO4J_DATABASE) as session:  # type: ignore
            tx = await session.begin_transaction()
            try:
                for query in queries:
                    self.logger.debug(f"Schema operation: {query[:100]}...")
                    await tx.run(query)
                await tx.commit()
            except Exception:
                if tx and not tx.closed():  # type: ignore
                    await tx.rollback()  # type: ignore
                raise

    async def _execute_schema_individually(self, queries: list[str]) -> None:
        """Fallback: Execute schema operations individually."""
        for query_text in queries:
            try:
                await self.execute_write_query(query_text)
                self.logger.info(
                    f"Fallback: Successfully applied schema operation: '{query_text[:100]}...'"
                )
            except Exception as individual_e:
                self.logger.warning(
                    f"Fallback: Failed to apply schema operation '{query_text[:100]}...': {individual_e}"
                )

    def embedding_to_list(self, embedding: np.ndarray | None) -> list[float] | None:
        if embedding is None:
            return None
        if not isinstance(embedding, np.ndarray):
            self.logger.warning(
                f"Attempting to convert non-numpy array to list for Neo4j: {type(embedding)}"
            )
            if hasattr(embedding, "tolist"):
                return embedding.tolist()  # type: ignore
            self.logger.error(
                f"Cannot convert type {type(embedding)} to list for Neo4j."
            )
            return None
        return embedding.astype(np.float32).tolist()

    def list_to_embedding(
        self, embedding_list: list[float | int] | None
    ) -> np.ndarray | None:
        if embedding_list is None:
            return None
        try:
            return np.array(embedding_list, dtype=config.EMBEDDING_DTYPE)
        except Exception as e:
            self.logger.error(
                f"Error converting list to numpy embedding: {e}", exc_info=True
            )
            return None


neo4j_manager = Neo4jManagerSingleton()
