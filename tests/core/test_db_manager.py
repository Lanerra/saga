# tests/core/test_db_manager.py
"""Comprehensive tests for core/db_manager.py"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from neo4j.exceptions import ServiceUnavailable

from core.db_manager import Neo4jManagerSingleton, neo4j_manager
from core.exceptions import (
    DatabaseConnectionError,
    DatabaseError,
    DatabaseTransactionError,
)


@pytest.fixture(autouse=True)
def _restore_global_neo4j_manager_singleton_state():
    """
    Prevent state leakage from these unit tests into the rest of the suite.

    This module intentionally mutates the `Neo4jManagerSingleton` (it is a true singleton),
    setting `driver` to MagicMock instances and modifying caches. If we don't restore that
    state, later tests that expect a real Neo4j connection (e.g. first-name matching) can
    fail in an order-dependent way.

    We snapshot the singleton's mutable fields before each test in this module and restore
    them afterwards.
    """
    manager = Neo4jManagerSingleton()

    original_driver = manager.driver
    original_property_keys_cache = manager._property_keys_cache
    original_property_keys_cache_ts = manager._property_keys_cache_ts

    # CORE-010: APOC capability detection is cached once per process; tests that
    # mutate those caches must not leak to other modules/tests.
    original_apoc_available_cache = manager._apoc_available_cache
    original_apoc_availability_warning_logged = manager._apoc_availability_warning_logged

    yield

    manager.driver = original_driver
    manager._property_keys_cache = original_property_keys_cache
    manager._property_keys_cache_ts = original_property_keys_cache_ts
    manager._apoc_available_cache = original_apoc_available_cache
    manager._apoc_availability_warning_logged = original_apoc_availability_warning_logged


class TestNeo4jManagerSingleton:
    """Singleton pattern and initialization"""

    def test_singleton_pattern(self):
        """Multiple instantiations return the same instance"""
        instance1 = Neo4jManagerSingleton()
        instance2 = Neo4jManagerSingleton()
        assert instance1 is instance2

    def test_initialization_only_once(self):
        """Initialization occurs only once despite multiple instantiations"""
        manager = Neo4jManagerSingleton()
        assert manager._initialized_flag is True

        initial_driver = manager.driver
        manager2 = Neo4jManagerSingleton()
        assert manager2 is manager
        assert manager2.driver is initial_driver

    def test_initial_state(self):
        """Manager starts with expected initial state"""
        manager = Neo4jManagerSingleton()
        assert manager._property_keys_cache is None
        assert manager._property_keys_cache_ts is None


@pytest.mark.asyncio
class TestConnection:
    """Connection establishment and teardown"""

    async def test_connect_success(self, monkeypatch):
        """Successful connection to Neo4j"""
        mock_driver = MagicMock()
        mock_driver.verify_connectivity = MagicMock()

        mock_graph_database = MagicMock()
        mock_graph_database.driver.return_value = mock_driver

        monkeypatch.setattr("core.db_manager.GraphDatabase", mock_graph_database)
        monkeypatch.setattr(Neo4jManagerSingleton, "_sync_probe_apoc_version", lambda _self: "5.0.0")

        manager = Neo4jManagerSingleton()
        await manager.connect()

        assert manager.driver is mock_driver
        mock_driver.verify_connectivity.assert_called_once()

    async def test_connect_service_unavailable(self, monkeypatch):
        """Connection fails when Neo4j service is unavailable"""
        mock_graph_database = MagicMock()
        mock_graph_database.driver.side_effect = ServiceUnavailable("Service down")

        monkeypatch.setattr("core.db_manager.GraphDatabase", mock_graph_database)

        manager = Neo4jManagerSingleton()
        manager.driver = None

        with pytest.raises(DatabaseConnectionError) as exc_info:
            await manager.connect()

        assert "Neo4j database is not available" in str(exc_info.value)
        assert manager.driver is None

    async def test_connect_unexpected_error(self, monkeypatch):
        """Connection fails with unexpected error"""
        mock_graph_database = MagicMock()
        mock_graph_database.driver.side_effect = RuntimeError("Unexpected error")

        monkeypatch.setattr("core.db_manager.GraphDatabase", mock_graph_database)

        manager = Neo4jManagerSingleton()
        manager.driver = None

        with pytest.raises(DatabaseError):
            await manager.connect()

        assert manager.driver is None

    async def test_connect_closes_existing_driver(self, monkeypatch):
        """Connect closes existing driver before creating new one"""
        old_driver = MagicMock()
        old_driver.close = MagicMock()

        new_driver = MagicMock()
        new_driver.verify_connectivity = MagicMock()

        mock_graph_database = MagicMock()
        mock_graph_database.driver.return_value = new_driver

        monkeypatch.setattr("core.db_manager.GraphDatabase", mock_graph_database)
        monkeypatch.setattr(Neo4jManagerSingleton, "_sync_probe_apoc_version", lambda _self: "5.0.0")

        manager = Neo4jManagerSingleton()
        manager.driver = old_driver

        await manager.connect()

        old_driver.close.assert_called_once()
        assert manager.driver is new_driver

    async def test_close_with_active_driver(self, monkeypatch):
        """Close properly closes active driver"""
        mock_driver = MagicMock()
        mock_driver.close = MagicMock()

        manager = Neo4jManagerSingleton()
        manager.driver = mock_driver

        await manager.close()

        mock_driver.close.assert_called_once()
        assert manager.driver is None

    async def test_close_with_no_driver(self):
        """Close handles case when driver is None"""
        manager = Neo4jManagerSingleton()
        manager.driver = None

        await manager.close()

        assert manager.driver is None

    async def test_close_with_error(self, monkeypatch):
        """Close handles errors during driver closure"""
        mock_driver = MagicMock()
        mock_driver.close = MagicMock(side_effect=RuntimeError("Close error"))

        manager = Neo4jManagerSingleton()
        manager.driver = mock_driver

        await manager.close()

        assert manager.driver is None

    async def test_ensure_connected_when_disconnected(self, monkeypatch):
        """_ensure_connected connects when driver is None"""

        async def mock_connect(self):
            self.driver = MagicMock()

        monkeypatch.setattr(Neo4jManagerSingleton, "connect", mock_connect)

        manager = Neo4jManagerSingleton()
        manager.driver = None

        await manager._ensure_connected()

        assert manager.driver is not None

    async def test_ensure_connected_when_already_connected(self, monkeypatch):
        """_ensure_connected does nothing when driver exists"""
        mock_connect = AsyncMock()
        monkeypatch.setattr(Neo4jManagerSingleton, "connect", mock_connect)

        manager = Neo4jManagerSingleton()
        manager.driver = MagicMock()

        await manager._ensure_connected()

        mock_connect.assert_not_called()

    async def test_ensure_connected_fails_after_connect_attempt(self, monkeypatch):
        """_ensure_connected raises error if connect fails to set driver"""

        async def mock_connect_fail(self):
            pass

        monkeypatch.setattr(Neo4jManagerSingleton, "connect", mock_connect_fail)

        manager = Neo4jManagerSingleton()
        manager.driver = None

        with pytest.raises(DatabaseConnectionError) as exc_info:
            await manager._ensure_connected()

        assert "Neo4j driver not initialized" in str(exc_info.value)


class TestSyncConnection:
    """Synchronous connection checks"""

    def test_ensure_connected_sync_with_driver(self):
        """_ensure_connected_sync succeeds when driver exists"""
        manager = Neo4jManagerSingleton()
        manager.driver = MagicMock()

        manager._ensure_connected_sync()

    def test_ensure_connected_sync_without_driver(self):
        """_ensure_connected_sync raises error when driver is None"""
        manager = Neo4jManagerSingleton()
        manager.driver = None

        with pytest.raises(DatabaseConnectionError) as exc_info:
            manager._ensure_connected_sync()

        assert "Neo4j driver not initialized" in str(exc_info.value)


class TestSyncQueryExecution:
    """Synchronous query execution methods"""

    def test_sync_execute_query_tx(self):
        """_sync_execute_query_tx executes query in transaction"""
        manager = Neo4jManagerSingleton()

        mock_tx = MagicMock()
        mock_record1 = MagicMock()
        mock_record1.__iter__ = lambda self: iter([("name", "Alice")])
        mock_record2 = MagicMock()
        mock_record2.__iter__ = lambda self: iter([("name", "Bob")])

        mock_tx.run.return_value = [mock_record1, mock_record2]

        result = manager._sync_execute_query_tx(mock_tx, "MATCH (n) RETURN n", {})

        assert len(result) == 2
        mock_tx.run.assert_called_once_with("MATCH (n) RETURN n", {})

    def test_sync_execute_read_query(self):
        """_sync_execute_read_query uses execute_read (Neo4j v5+ API)"""
        manager = Neo4jManagerSingleton()

        mock_session = MagicMock()
        mock_session.execute_read.return_value = [{"name": "Alice"}]

        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        manager.driver = mock_driver

        result = manager._sync_execute_read_query("MATCH (n) RETURN n", {})

        assert result == [{"name": "Alice"}]
        mock_session.execute_read.assert_called_once_with(manager._sync_execute_query_tx, "MATCH (n) RETURN n", {})

    def test_sync_execute_write_query(self):
        """_sync_execute_write_query uses execute_write (Neo4j v5+ API)"""
        manager = Neo4jManagerSingleton()

        mock_session = MagicMock()
        mock_session.execute_write.return_value = [{"created": 1}]

        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        manager.driver = mock_driver

        result = manager._sync_execute_write_query("CREATE (n:Node)", {})

        assert result == [{"created": 1}]
        mock_session.execute_write.assert_called_once_with(manager._sync_execute_query_tx, "CREATE (n:Node)", {})

    def test_sync_execute_cypher_batch_empty(self):
        """_sync_execute_cypher_batch handles empty statement list"""
        manager = Neo4jManagerSingleton()
        manager.driver = MagicMock()

        manager._sync_execute_cypher_batch([])

    def test_sync_execute_cypher_batch_success(self):
        """_sync_execute_cypher_batch commits transaction successfully"""
        manager = Neo4jManagerSingleton()

        mock_tx = MagicMock()
        mock_tx.closed.return_value = False
        mock_tx.run = MagicMock()
        mock_tx.commit = MagicMock()

        mock_session = MagicMock()
        mock_session.begin_transaction.return_value = mock_tx

        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        manager.driver = mock_driver

        statements = [
            ("CREATE (n:Node {id: $id})", {"id": "1"}),
            ("CREATE (n:Node {id: $id})", {"id": "2"}),
        ]

        manager._sync_execute_cypher_batch(statements)

        assert mock_tx.run.call_count == 2
        mock_tx.commit.assert_called_once()

    def test_sync_execute_cypher_batch_statement_error(self):
        """_sync_execute_cypher_batch handles statement-level errors"""
        manager = Neo4jManagerSingleton()

        mock_error = Exception("Statement error")
        mock_error.code = "Neo.ClientError.Statement.SyntaxError"
        mock_error.message = "Invalid syntax"

        mock_tx = MagicMock()
        mock_tx.closed.return_value = False
        mock_tx.run.side_effect = mock_error
        mock_tx.rollback = MagicMock()

        mock_session = MagicMock()
        mock_session.begin_transaction.return_value = mock_tx

        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        manager.driver = mock_driver

        statements = [("INVALID QUERY", {})]

        with pytest.raises(DatabaseTransactionError):
            manager._sync_execute_cypher_batch(statements)

    def test_sync_execute_cypher_batch_transaction_error(self):
        """_sync_execute_cypher_batch raises DatabaseTransactionError on failure"""
        manager = Neo4jManagerSingleton()

        mock_error = RuntimeError("Transaction failed")

        mock_tx = MagicMock()
        mock_tx.closed.return_value = False
        mock_tx.run.side_effect = mock_error
        mock_tx.rollback = MagicMock()

        mock_session = MagicMock()
        mock_session.begin_transaction.return_value = mock_tx

        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        manager.driver = mock_driver

        statements = [("CREATE (n:Node)", {})]

        with pytest.raises(DatabaseTransactionError) as exc_info:
            manager._sync_execute_cypher_batch(statements)

        assert "Batch Cypher execution failed" in str(exc_info.value)
        mock_tx.rollback.assert_called_once()


@pytest.mark.asyncio
class TestQueryExecution:
    """Async query execution methods"""

    async def test_execute_read_query_success(self, monkeypatch):
        """Execute read query successfully"""
        mock_result = [{"name": "Alice"}, {"name": "Bob"}]

        async def mock_sync_read(query, params):
            return mock_result

        manager = Neo4jManagerSingleton()
        manager.driver = MagicMock()

        monkeypatch.setattr(manager, "_sync_execute_read_query", lambda q, p: mock_result)

        result = await manager.execute_read_query("MATCH (n) RETURN n", {"limit": 10})

        assert result == mock_result

    async def test_execute_write_query_success(self, monkeypatch):
        """Execute write query successfully"""
        mock_result = [{"created": 1}]

        manager = Neo4jManagerSingleton()
        manager.driver = MagicMock()

        monkeypatch.setattr(manager, "_sync_execute_write_query", lambda q, p: mock_result)

        result = await manager.execute_write_query("CREATE (n:Node)", {})

        assert result == mock_result

    async def test_execute_cypher_batch_empty(self, monkeypatch):
        """Execute empty batch does nothing"""
        manager = Neo4jManagerSingleton()
        manager.driver = MagicMock()

        called = []

        def mock_batch(statements):
            called.append(True)

        monkeypatch.setattr(manager, "_sync_execute_cypher_batch", mock_batch)

        await manager.execute_cypher_batch([])

        assert len(called) == 1

    async def test_execute_cypher_batch_with_statements(self, monkeypatch):
        """Execute batch with multiple statements"""
        statements = [
            ("CREATE (n:Node {id: $id})", {"id": "1"}),
            ("CREATE (n:Node {id: $id})", {"id": "2"}),
        ]

        manager = Neo4jManagerSingleton()
        manager.driver = MagicMock()

        executed_statements = []

        def mock_batch(stmts):
            executed_statements.extend(stmts)

        monkeypatch.setattr(manager, "_sync_execute_cypher_batch", mock_batch)

        await manager.execute_cypher_batch(statements)

        assert len(executed_statements) == 2


@pytest.mark.asyncio
class TestTransactionManagement:
    """Transaction execution with automatic rollback"""

    async def test_execute_in_transaction_success(self, monkeypatch):
        """Execute transaction successfully"""
        manager = Neo4jManagerSingleton()

        mock_tx = MagicMock()
        mock_tx.closed.return_value = False
        mock_tx.commit = MagicMock()

        mock_session = MagicMock()
        mock_session.begin_transaction.return_value = mock_tx
        mock_session.__enter__ = lambda self: self
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        manager.driver = mock_driver

        def transaction_func(tx, value):
            return value * 2

        async def mock_to_thread(func):
            return func()

        monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)

        result = await manager.execute_in_transaction(transaction_func, 5)

        assert result == 10
        mock_tx.commit.assert_called_once()

    async def test_execute_in_transaction_rollback_on_error(self, monkeypatch):
        """Transaction rolls back on error"""
        manager = Neo4jManagerSingleton()

        mock_tx = MagicMock()
        mock_tx.closed.return_value = False
        mock_tx.rollback = MagicMock()

        mock_session = MagicMock()
        mock_session.begin_transaction.return_value = mock_tx
        mock_session.__enter__ = lambda self: self
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        manager.driver = mock_driver

        def failing_transaction(tx):
            raise RuntimeError("Transaction failed")

        with pytest.raises(DatabaseTransactionError) as exc_info:
            await manager.execute_in_transaction(failing_transaction)

        assert "Transaction failed and was rolled back" in str(exc_info.value)
        mock_tx.rollback.assert_called_once()

    async def test_execute_in_transaction_with_kwargs(self, monkeypatch):
        """Transaction executes with keyword arguments"""
        manager = Neo4jManagerSingleton()

        mock_tx = MagicMock()
        mock_tx.closed.return_value = False
        mock_tx.commit = MagicMock()

        mock_session = MagicMock()
        mock_session.begin_transaction.return_value = mock_tx
        mock_session.__enter__ = lambda self: self
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        manager.driver = mock_driver

        def transaction_func(tx, value, multiplier=2):
            return value * multiplier

        async def mock_to_thread(func):
            return func()

        monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)

        result = await manager.execute_in_transaction(transaction_func, 5, multiplier=3)

        assert result == 15
        mock_tx.commit.assert_called_once()


@pytest.mark.asyncio
class TestPropertyKeyCache:
    """Property key caching functionality"""

    async def test_refresh_property_keys_cache_success(self, monkeypatch):
        """Refresh property keys cache successfully"""
        manager = Neo4jManagerSingleton()
        manager.driver = MagicMock()

        mock_results = [
            {"propertyKey": "name"},
            {"propertyKey": "description"},
            {"propertyKey": "created_ts"},
        ]

        monkeypatch.setattr(manager, "_sync_execute_read_query", lambda q, p: mock_results)

        keys = await manager.refresh_property_keys_cache()

        assert keys == {"name", "description", "created_ts"}
        assert manager._property_keys_cache == keys
        assert manager._property_keys_cache_ts is not None

    async def test_refresh_property_keys_cache_alternative_field_names(self, monkeypatch):
        """Refresh handles alternative field names"""
        manager = Neo4jManagerSingleton()
        manager.driver = MagicMock()

        mock_results = [
            {"propertyName": "name"},
            {"property": "description"},
        ]

        monkeypatch.setattr(manager, "_sync_execute_read_query", lambda q, p: mock_results)

        keys = await manager.refresh_property_keys_cache()

        assert keys == {"name", "description"}

    async def test_refresh_property_keys_cache_error(self, monkeypatch):
        """Refresh handles errors gracefully"""
        manager = Neo4jManagerSingleton()
        manager.driver = MagicMock()

        def mock_failing_query(q, p):
            raise RuntimeError("Query failed")

        monkeypatch.setattr(manager, "_sync_execute_read_query", mock_failing_query)

        keys = await manager.refresh_property_keys_cache()

        assert keys == set()
        assert manager._property_keys_cache == set()
        assert manager._property_keys_cache_ts is None

    async def test_has_property_key_cache_hit(self, monkeypatch):
        """has_property_key uses cached values"""
        manager = Neo4jManagerSingleton()
        manager._property_keys_cache = {"name", "description"}
        manager._property_keys_cache_ts = 1000.0

        mock_time = MagicMock()
        mock_time.monotonic.return_value = 1100.0

        with patch("time.monotonic", return_value=1100.0):
            result = await manager.has_property_key("name")

        assert result is True

    async def test_has_property_key_cache_miss(self, monkeypatch):
        """has_property_key returns False for missing key"""
        manager = Neo4jManagerSingleton()
        manager._property_keys_cache = {"name", "description"}
        manager._property_keys_cache_ts = 1000.0

        with patch("time.monotonic", return_value=1100.0):
            result = await manager.has_property_key("unknown_key")

        assert result is False

    async def test_has_property_key_cache_expired(self, monkeypatch):
        """has_property_key refreshes expired cache"""
        manager = Neo4jManagerSingleton()
        manager.driver = MagicMock()
        manager._property_keys_cache = {"old_key"}
        manager._property_keys_cache_ts = 1000.0

        mock_results = [{"propertyKey": "new_key"}]

        monkeypatch.setattr(manager, "_sync_execute_read_query", lambda q, p: mock_results)

        with patch("time.monotonic", return_value=1400.0):
            result = await manager.has_property_key("new_key", max_age_seconds=300)

        assert result is True
        assert manager._property_keys_cache == {"new_key"}

    async def test_has_property_key_no_cache(self, monkeypatch):
        """has_property_key initializes cache when None"""
        manager = Neo4jManagerSingleton()
        manager.driver = MagicMock()
        manager._property_keys_cache = None

        mock_results = [{"propertyKey": "name"}]

        monkeypatch.setattr(manager, "_sync_execute_read_query", lambda q, p: mock_results)

        result = await manager.has_property_key("name")

        assert result is True
        assert manager._property_keys_cache == {"name"}


@pytest.mark.asyncio
class TestApocCapabilityDetection:
    """APOC capability detection / caching (CORE-010)."""

    async def test_is_apoc_available_returns_false_and_is_cached_when_unavailable(self) -> None:
        manager = Neo4jManagerSingleton()

        # Reset caches explicitly for determinism within this unit test.
        manager._apoc_available_cache = None
        manager._apoc_availability_warning_logged = False

        # Avoid any real connectivity attempts by patching execute_read_query directly.
        with patch.object(
            manager,
            "execute_read_query",
            new=AsyncMock(side_effect=RuntimeError("Procedure not found")),
        ) as exec_read:
            out1 = await manager.is_apoc_available()
            out2 = await manager.is_apoc_available()

        assert out1 is False
        assert out2 is False

        # Cached once per process => only one underlying probe.
        assert exec_read.await_count == 1


@pytest.mark.asyncio
class TestSchemaCreation:
    """Schema creation and management"""

    async def test_create_db_schema_phases(self, monkeypatch):
        """create_db_schema executes both phases"""
        manager = Neo4jManagerSingleton()

        phase1_called = []
        phase2_called = []

        async def mock_phase1():
            phase1_called.append(True)

        async def mock_phase2():
            phase2_called.append(True)

        monkeypatch.setattr(manager, "_create_constraints_and_indexes", mock_phase1)
        monkeypatch.setattr(manager, "_create_type_placeholders", mock_phase2)

        await manager.create_db_schema()

        assert len(phase1_called) == 1
        assert len(phase2_called) == 1

    async def test_create_constraints_and_indexes_batch_success(self, monkeypatch):
        """Constraints and indexes created via batch (offloaded via asyncio.to_thread)"""
        manager = Neo4jManagerSingleton()

        executed_queries: list[str] = []

        def mock_execute_batch(queries: list[str]) -> None:
            executed_queries.extend(queries)

        to_thread_calls: list[tuple[object, tuple[object, ...], dict[str, object], int]] = []

        async def mock_to_thread(func, *args, **kwargs):
            # Record the calling thread id (the event loop thread for this test),
            # and execute inline to keep this unit test deterministic.
            to_thread_calls.append((func, args, kwargs, threading.get_ident()))
            return func(*args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
        monkeypatch.setattr(manager, "_execute_schema_batch", mock_execute_batch)
        monkeypatch.setattr(manager, "execute_write_query", AsyncMock())

        await manager._create_constraints_and_indexes()

        assert len(executed_queries) > 0
        assert len(to_thread_calls) >= 1
        # Ensure the schema batch path uses asyncio.to_thread(...) rather than running directly on the loop.
        assert to_thread_calls[0][0] is manager._execute_schema_batch

    async def test_create_constraints_and_indexes_batch_failure_fallback(self, monkeypatch):
        """Falls back to individual execution on batch failure (batch runs via asyncio.to_thread)"""
        manager = Neo4jManagerSingleton()

        def mock_batch_fail(queries: list[str]) -> None:
            raise RuntimeError("Batch failed")

        individual_calls: list[list[str]] = []

        async def mock_individual(queries: list[str]) -> None:
            individual_calls.append(queries)

        async def mock_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
        monkeypatch.setattr(manager, "_execute_schema_batch", mock_batch_fail)
        monkeypatch.setattr(manager, "_execute_schema_individually", mock_individual)
        monkeypatch.setattr(manager, "execute_write_query", AsyncMock())

        await manager._create_constraints_and_indexes()

        assert len(individual_calls) > 0

    async def test_create_constraints_and_indexes_vector_index_error(self, monkeypatch):
        """Vector index creation error is logged but doesn't fail"""
        manager = Neo4jManagerSingleton()

        def mock_batch_success(queries: list[str]) -> None:
            return None

        async def mock_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def mock_vector_fail(query):
            raise RuntimeError("Vector index not supported")

        monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
        monkeypatch.setattr(manager, "_execute_schema_batch", mock_batch_success)
        monkeypatch.setattr(manager, "execute_write_query", mock_vector_fail)

        await manager._create_constraints_and_indexes()

    async def test_create_constraints_and_indexes_schema_batch_tx_run_off_event_loop_thread(self, monkeypatch):
        """tx.run for schema batch does not execute on the asyncio event loop thread (CORE-001)"""
        manager = Neo4jManagerSingleton()

        event_loop_thread_id = threading.get_ident()
        tx_run_thread_ids: list[int] = []

        def record_tx_run_thread(query: str) -> None:
            tx_run_thread_ids.append(threading.get_ident())

        mock_tx = MagicMock()
        mock_tx.closed.return_value = False
        mock_tx.run.side_effect = record_tx_run_thread
        mock_tx.commit = MagicMock()

        mock_session = MagicMock()
        mock_session.begin_transaction.return_value = mock_tx

        mock_driver = MagicMock()
        # driver.session(...) returns a context manager; its __enter__ yields the session object.
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        manager.driver = mock_driver
        # Avoid real vector index execution; this test focuses on the schema batch.
        monkeypatch.setattr(manager, "execute_write_query", AsyncMock())

        await manager._create_constraints_and_indexes()

        assert len(tx_run_thread_ids) > 0
        # The schema batch is offloaded via asyncio.to_thread, so tx.run must not run on the event loop thread.
        assert all(tid != event_loop_thread_id for tid in tx_run_thread_ids)

    async def test_create_type_placeholders_success(self, monkeypatch):
        """Type placeholders created successfully"""
        manager = Neo4jManagerSingleton()

        executed = []

        async def mock_batch(statements):
            executed.extend(statements)

        monkeypatch.setattr(manager, "execute_cypher_batch", mock_batch)

        await manager._create_type_placeholders()

        assert len(executed) > 0

    async def test_create_type_placeholders_fallback(self, monkeypatch):
        """Type placeholders fall back to individual execution"""
        manager = Neo4jManagerSingleton()

        async def mock_batch_fail(statements):
            raise RuntimeError("Batch failed")

        individual_calls = []

        async def mock_individual(query):
            individual_calls.append(query)
            raise RuntimeError("Individual also fails")

        monkeypatch.setattr(manager, "execute_cypher_batch", mock_batch_fail)
        monkeypatch.setattr(manager, "execute_write_query", mock_individual)

        await manager._create_type_placeholders()

        assert len(individual_calls) > 0

    async def test_execute_schema_individually_success(self, monkeypatch):
        """Schema queries executed individually"""
        manager = Neo4jManagerSingleton()

        executed = []

        async def mock_write(query):
            executed.append(query)

        monkeypatch.setattr(manager, "execute_write_query", mock_write)

        queries = ["QUERY1", "QUERY2", "QUERY3"]

        await manager._execute_schema_individually(queries)

        assert executed == queries

    async def test_execute_schema_individually_continues_on_error(self, monkeypatch):
        """Schema individual execution continues on error"""
        manager = Neo4jManagerSingleton()

        executed = []

        async def mock_write(query):
            executed.append(query)
            if query == "QUERY2":
                raise RuntimeError("Query2 failed")

        monkeypatch.setattr(manager, "execute_write_query", mock_write)

        queries = ["QUERY1", "QUERY2", "QUERY3"]

        await manager._execute_schema_individually(queries)

        assert executed == queries


class TestSyncSchemaOperations:
    """Synchronous schema operations"""

    def test_execute_schema_batch_success(self):
        """Schema batch executes and commits"""
        manager = Neo4jManagerSingleton()

        mock_tx = MagicMock()
        mock_tx.closed.return_value = False
        mock_tx.run = MagicMock()
        mock_tx.commit = MagicMock()

        mock_session = MagicMock()
        mock_session.begin_transaction.return_value = mock_tx
        mock_session.__enter__ = lambda self: self
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        manager.driver = mock_driver

        queries = [
            "CREATE CONSTRAINT test1 IF NOT EXISTS FOR (n:Test) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX test2 IF NOT EXISTS FOR (n:Test) ON (n.name)",
        ]

        manager._execute_schema_batch(queries)

        assert mock_tx.run.call_count == 2
        mock_tx.commit.assert_called_once()

    def test_execute_schema_batch_rollback_on_error(self):
        """Schema batch rolls back on error"""
        manager = Neo4jManagerSingleton()

        mock_tx = MagicMock()
        mock_tx.closed.return_value = False
        mock_tx.run.side_effect = RuntimeError("Query failed")
        mock_tx.rollback = MagicMock()

        mock_session = MagicMock()
        mock_session.begin_transaction.return_value = mock_tx
        mock_session.__enter__ = lambda self: self
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        manager.driver = mock_driver

        queries = ["CREATE CONSTRAINT invalid"]

        with pytest.raises(RuntimeError):
            manager._execute_schema_batch(queries)

        mock_tx.rollback.assert_called_once()


class TestEmbeddingConversion:
    """Embedding conversion utilities"""

    def test_embedding_to_list_numpy_array(self):
        """Convert numpy array to list"""
        manager = Neo4jManagerSingleton()

        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = manager.embedding_to_list(embedding)

        assert isinstance(result, list)
        assert len(result) == 3
        assert result == [1.0, 2.0, 3.0]

    def test_embedding_to_list_none(self):
        """Convert None returns None"""
        manager = Neo4jManagerSingleton()

        result = manager.embedding_to_list(None)

        assert result is None

    def test_embedding_to_list_scalar_array(self):
        """Convert scalar (0-d) array to single-element list"""
        manager = Neo4jManagerSingleton()

        embedding = np.array(5.0)
        result = manager.embedding_to_list(embedding)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result == [5.0]

    def test_embedding_to_list_with_tolist_method(self):
        """Convert object with tolist method"""
        manager = Neo4jManagerSingleton()

        class FakeEmbedding:
            def tolist(self):
                return [1.0, 2.0]

        embedding = FakeEmbedding()
        result = manager.embedding_to_list(embedding)

        assert result == [1.0, 2.0]

    def test_embedding_to_list_invalid_type(self):
        """Invalid type returns None"""
        manager = Neo4jManagerSingleton()

        result = manager.embedding_to_list("invalid")

        assert result is None

    def test_list_to_embedding_success(self):
        """Convert list to numpy array"""
        manager = Neo4jManagerSingleton()

        embedding_list = [1.0, 2.0, 3.0]
        result = manager.list_to_embedding(embedding_list)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_list_to_embedding_with_integers(self):
        """Convert list with integers to numpy array"""
        manager = Neo4jManagerSingleton()

        embedding_list = [1, 2, 3]
        result = manager.list_to_embedding(embedding_list)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_list_to_embedding_none(self):
        """Convert None returns None"""
        manager = Neo4jManagerSingleton()

        result = manager.list_to_embedding(None)

        assert result is None

    def test_list_to_embedding_error(self, monkeypatch):
        """Conversion error returns None"""
        manager = Neo4jManagerSingleton()

        def mock_array_fail(*args, **kwargs):
            raise ValueError("Conversion failed")

        monkeypatch.setattr(np, "array", mock_array_fail)

        result = manager.list_to_embedding([1.0, 2.0])

        assert result is None


class TestSingletonInstance:
    """Module-level singleton instance"""

    def test_neo4j_manager_is_singleton(self):
        """neo4j_manager is singleton instance"""
        assert isinstance(neo4j_manager, Neo4jManagerSingleton)

    def test_neo4j_manager_same_as_new_instance(self):
        """neo4j_manager is same instance as newly created"""
        new_instance = Neo4jManagerSingleton()
        assert neo4j_manager is new_instance
