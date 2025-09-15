# core/database_interface.py
"""
Database Interface for unified dependency injection in SAGA.

This module provides a protocol interface for database operations and implements
a wrapper service for the existing Neo4j singleton. It temporarily uses the existing
singleton while providing a DI interface, preparing for eventual singleton elimination.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import asynccontextmanager

import structlog
import numpy as np

logger = structlog.get_logger(__name__)


@dataclass
class DatabaseMetrics:
    """Database operation metrics."""
    total_read_queries: int = 0
    total_write_queries: int = 0
    total_batch_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_query_time: float = 0.0
    connection_count: int = 0
    last_operation_time: Optional[datetime] = None
    
    def record_operation(self, operation_type: str, success: bool, execution_time: float):
        """Record a database operation."""
        if operation_type == "read":
            self.total_read_queries += 1
        elif operation_type == "write":
            self.total_write_queries += 1
        elif operation_type == "batch":
            self.total_batch_operations += 1
        
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
        
        self.total_query_time += execution_time
        self.last_operation_time = datetime.utcnow()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        total_ops = self.successful_operations + self.failed_operations
        return (self.successful_operations / total_ops * 100) if total_ops > 0 else 0.0
    
    @property
    def average_query_time(self) -> float:
        """Calculate average query time in seconds."""
        total_ops = self.successful_operations + self.failed_operations
        return (self.total_query_time / total_ops) if total_ops > 0 else 0.0


@runtime_checkable
class DatabaseInterface(Protocol):
    """Protocol defining the interface for database operations."""
    
    async def connect(self) -> None:
        """Establish database connection."""
        ...
    
    async def close(self) -> None:
        """Close database connection."""
        ...
    
    async def execute_read_query(
        self, query: str, parameters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Execute a read query and return results."""
        ...
    
    async def execute_write_query(
        self, query: str, parameters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Execute a write query and return results."""
        ...
    
    async def execute_cypher_batch(
        self, cypher_statements_with_params: List[tuple[str, Dict[str, Any]]]
    ) -> None:
        """Execute a batch of Cypher statements in a transaction."""
        ...
    
    async def create_db_schema(self) -> None:
        """Create and verify database schema (indexes, constraints, etc.)."""
        ...
    
    def embedding_to_list(self, embedding: np.ndarray | None) -> List[float] | None:
        """Convert numpy embedding to list for database storage."""
        ...
    
    def list_to_embedding(
        self, embedding_list: List[float | int] | None
    ) -> np.ndarray | None:
        """Convert list from database to numpy embedding."""
        ...
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get database connection information."""
        ...
    
    def get_metrics(self) -> DatabaseMetrics:
        """Get database operation metrics."""
        ...


@runtime_checkable
class DatabaseHealthCheckInterface(Protocol):
    """Protocol for database health checking."""
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a database health check."""
        ...
    
    async def verify_connectivity(self) -> bool:
        """Verify database connectivity."""
        ...


class Neo4jDatabaseService:
    """
    Database service wrapper implementing DatabaseInterface.
    
    This service wraps the existing Neo4jManagerSingleton to provide
    a dependency-injectable interface while maintaining backward compatibility.
    Eventually, this will replace the singleton pattern entirely.
    """
    
    def __init__(self):
        # Import here to avoid circular dependencies
        from core.db_manager import neo4j_manager
        
        self._neo4j_manager = neo4j_manager
        self._metrics = DatabaseMetrics()
        self._is_connected = False
        self._last_health_check: Optional[datetime] = None
        self._health_check_cache: Optional[Dict[str, Any]] = None
        
        logger.info("Neo4j database service initialized")
    
    async def connect(self) -> None:
        """Establish database connection."""
        start_time = time.time()
        try:
            await self._neo4j_manager.connect()
            self._is_connected = True
            self._metrics.connection_count += 1
            
            execution_time = time.time() - start_time
            self._metrics.record_operation("connect", True, execution_time)
            
            logger.info("Database connection established")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._metrics.record_operation("connect", False, execution_time)
            logger.error(f"Failed to establish database connection: {e}", exc_info=True)
            raise
    
    async def close(self) -> None:
        """Close database connection."""
        start_time = time.time()
        try:
            await self._neo4j_manager.close()
            self._is_connected = False
            
            execution_time = time.time() - start_time
            self._metrics.record_operation("close", True, execution_time)
            
            logger.info("Database connection closed")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._metrics.record_operation("close", False, execution_time)
            logger.error(f"Error closing database connection: {e}", exc_info=True)
            raise
    
    async def execute_read_query(
        self, query: str, parameters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Execute a read query and return results."""
        start_time = time.time()
        try:
            result = await self._neo4j_manager.execute_read_query(query, parameters)
            
            execution_time = time.time() - start_time
            self._metrics.record_operation("read", True, execution_time)
            
            logger.debug(f"Read query executed in {execution_time:.3f}s, returned {len(result)} records")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._metrics.record_operation("read", False, execution_time)
            logger.error(f"Read query failed: {e}", exc_info=True)
            raise
    
    async def execute_write_query(
        self, query: str, parameters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Execute a write query and return results."""
        start_time = time.time()
        try:
            result = await self._neo4j_manager.execute_write_query(query, parameters)
            
            execution_time = time.time() - start_time
            self._metrics.record_operation("write", True, execution_time)
            
            logger.debug(f"Write query executed in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._metrics.record_operation("write", False, execution_time)
            logger.error(f"Write query failed: {e}", exc_info=True)
            raise
    
    async def execute_cypher_batch(
        self, cypher_statements_with_params: List[tuple[str, Dict[str, Any]]]
    ) -> None:
        """Execute a batch of Cypher statements in a transaction."""
        start_time = time.time()
        try:
            await self._neo4j_manager.execute_cypher_batch(cypher_statements_with_params)
            
            execution_time = time.time() - start_time
            self._metrics.record_operation("batch", True, execution_time)
            
            logger.debug(f"Batch of {len(cypher_statements_with_params)} statements executed in {execution_time:.3f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._metrics.record_operation("batch", False, execution_time)
            logger.error(f"Batch execution failed: {e}", exc_info=True)
            raise
    
    async def create_db_schema(self) -> None:
        """Create and verify database schema (indexes, constraints, etc.)."""
        start_time = time.time()
        try:
            await self._neo4j_manager.create_db_schema()
            
            execution_time = time.time() - start_time
            self._metrics.record_operation("schema", True, execution_time)
            
            logger.info(f"Database schema created/verified in {execution_time:.3f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._metrics.record_operation("schema", False, execution_time)
            logger.error(f"Schema creation failed: {e}", exc_info=True)
            raise
    
    def embedding_to_list(self, embedding: np.ndarray | None) -> List[float] | None:
        """Convert numpy embedding to list for database storage."""
        return self._neo4j_manager.embedding_to_list(embedding)
    
    def list_to_embedding(
        self, embedding_list: List[float | int] | None
    ) -> np.ndarray | None:
        """Convert list from database to numpy embedding."""
        return self._neo4j_manager.list_to_embedding(embedding_list)
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get database connection information."""
        # Import config here to avoid circular dependencies
        import config
        
        return {
            "uri": config.NEO4J_URI,
            "database": config.NEO4J_DATABASE,
            "is_connected": self._is_connected,
            "driver_available": self._neo4j_manager.driver is not None,
            "service_type": "Neo4j",
            "wrapper_type": "Neo4jDatabaseService"
        }
    
    def get_metrics(self) -> DatabaseMetrics:
        """Get database operation metrics."""
        return self._metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a database health check."""
        # Cache health check results for 30 seconds to avoid excessive checking
        now = datetime.utcnow()
        if (self._last_health_check and 
            (now - self._last_health_check).total_seconds() < 30 and 
            self._health_check_cache):
            return self._health_check_cache
        
        health_info = {
            "healthy": False,
            "timestamp": now.isoformat(),
            "checks": {}
        }
        
        try:
            # Check 1: Driver availability
            health_info["checks"]["driver_available"] = self._neo4j_manager.driver is not None
            
            # Check 2: Connection test
            if self._neo4j_manager.driver:
                try:
                    await asyncio.to_thread(self._neo4j_manager.driver.verify_connectivity)
                    health_info["checks"]["connectivity"] = True
                except Exception as e:
                    health_info["checks"]["connectivity"] = False
                    health_info["checks"]["connectivity_error"] = str(e)
            else:
                health_info["checks"]["connectivity"] = False
                health_info["checks"]["connectivity_error"] = "No driver available"
            
            # Check 3: Simple query test
            try:
                result = await self.execute_read_query("RETURN 1 as test")
                health_info["checks"]["query_test"] = len(result) == 1 and result[0].get("test") == 1
            except Exception as e:
                health_info["checks"]["query_test"] = False
                health_info["checks"]["query_error"] = str(e)
            
            # Check 4: Metrics validation
            health_info["checks"]["metrics_available"] = True
            health_info["metrics_summary"] = {
                "total_operations": self._metrics.successful_operations + self._metrics.failed_operations,
                "success_rate": self._metrics.success_rate,
                "average_query_time": self._metrics.average_query_time
            }
            
            # Overall health determination
            health_info["healthy"] = all([
                health_info["checks"].get("driver_available", False),
                health_info["checks"].get("connectivity", False),
                health_info["checks"].get("query_test", False)
            ])
            
        except Exception as e:
            health_info["checks"]["health_check_error"] = str(e)
            health_info["healthy"] = False
            logger.error(f"Health check failed: {e}", exc_info=True)
        
        # Cache the results
        self._last_health_check = now
        self._health_check_cache = health_info
        
        return health_info
    
    async def verify_connectivity(self) -> bool:
        """Verify database connectivity."""
        try:
            if not self._neo4j_manager.driver:
                return False
            
            await asyncio.to_thread(self._neo4j_manager.driver.verify_connectivity)
            return True
        except Exception as e:
            logger.debug(f"Connectivity verification failed: {e}")
            return False
    
    @asynccontextmanager
    async def transaction_context(self):
        """Context manager for database transactions (future enhancement)."""
        # For now, this is a placeholder. Individual queries in Neo4j are automatically transactional.
        # This could be enhanced to support explicit transaction management.
        try:
            yield self
        except Exception as e:
            logger.error(f"Transaction context error: {e}", exc_info=True)
            raise
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information for monitoring (ServiceInterface compliance)."""
        return {
            "service_name": "Neo4jDatabaseService",
            "service_type": "database",
            "backend": "Neo4j",
            "connection_info": self.get_connection_info(),
            "metrics": {
                "total_read_queries": self._metrics.total_read_queries,
                "total_write_queries": self._metrics.total_write_queries,
                "total_batch_operations": self._metrics.total_batch_operations,
                "success_rate": self._metrics.success_rate,
                "average_query_time": self._metrics.average_query_time,
                "last_operation": self._metrics.last_operation_time.isoformat() 
                    if self._metrics.last_operation_time else None
            },
            "is_connected": self._is_connected,
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None
        }
    
    async def initialize(self) -> None:
        """Initialize the database service (called by lifecycle manager)."""
        try:
            await self.connect()
            logger.info("Database service initialized successfully")
        except Exception as e:
            logger.error(f"Database service initialization failed: {e}", exc_info=True)
            raise
    
    async def dispose(self) -> None:
        """Dispose of the database service (called by lifecycle manager)."""
        try:
            await self.close()
            logger.info("Database service disposed successfully")
        except Exception as e:
            logger.error(f"Database service disposal failed: {e}", exc_info=True)
            # Don't re-raise during disposal to avoid cascading failures


class DatabaseServiceFactory:
    """Factory for creating database service instances."""
    
    @staticmethod
    def create_neo4j_service() -> Neo4jDatabaseService:
        """Create a Neo4j database service instance."""
        return Neo4jDatabaseService()
    
    @staticmethod
    def create_service(db_type: str = "neo4j") -> DatabaseInterface:
        """Create a database service instance of the specified type."""
        if db_type.lower() == "neo4j":
            return DatabaseServiceFactory.create_neo4j_service()
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
