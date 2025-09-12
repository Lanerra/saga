# tests/service_mocks.py
"""
Unified testing framework for SAGA dependency injection system.

This module provides centralized mock implementations, isolated test service contexts,
and utilities to reduce test complexity and eliminate the need for extensive monkeypatching.
"""

import asyncio
import threading
from contextlib import contextmanager, asynccontextmanager
from typing import Any, Dict, List, Optional, Callable, AsyncIterable, Union, Type
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from datetime import datetime
import inspect

import structlog
import numpy as np

# Import the core DI components
from core.service_registry import ServiceRegistry, ServiceLifecycle
from core.service_lifecycle import ServiceLifecycleManager
from core.database_interface import DatabaseInterface, DatabaseMetrics
from core.validation_service_provider import ValidationServiceInterface, TypeInferenceServiceInterface

logger = structlog.get_logger(__name__)


@dataclass
class MockServiceConfig:
    """Configuration for mock services."""
    service_name: str
    interface: Optional[Type] = None
    lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON
    dependencies: List[str] = field(default_factory=list)
    mock_data: Dict[str, Any] = field(default_factory=dict)
    behavior_overrides: Dict[str, Callable] = field(default_factory=dict)


class MockDatabaseService:
    """Mock database service for testing."""
    
    def __init__(self, mock_data: Dict[str, Any] = None):
        self.mock_data = mock_data or {}
        self.metrics = DatabaseMetrics()
        self.is_connected = False
        self.query_history: List[Dict[str, Any]] = []
        self.call_count = 0
        self.should_fail = False
        self.failure_message = "Mock database failure"
        
    async def connect(self) -> None:
        """Mock connect method."""
        self.call_count += 1
        if self.should_fail:
            raise ConnectionError(self.failure_message)
        self.is_connected = True
        self.metrics.connection_count += 1
    
    async def close(self) -> None:
        """Mock close method."""
        self.call_count += 1
        self.is_connected = False
    
    async def execute_read_query(
        self, query: str, parameters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Mock read query execution."""
        self.call_count += 1
        self.query_history.append({
            "type": "read",
            "query": query,
            "parameters": parameters,
            "timestamp": datetime.utcnow()
        })
        
        if self.should_fail:
            raise Exception(self.failure_message)
        
        self.metrics.record_operation("read", True, 0.001)
        
        # Return mock data based on query patterns
        if "RETURN 1 as test" in query:
            return [{"test": 1}]
        elif "labels" in query.lower():
            return self.mock_data.get("labels", [{"label": "Character"}, {"label": "Location"}])
        elif "relationships" in query.lower():
            return self.mock_data.get("relationships", [{"type": "KNOWS"}, {"type": "LOCATED_IN"}])
        else:
            return self.mock_data.get("default_read_result", [])
    
    async def execute_write_query(
        self, query: str, parameters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Mock write query execution."""
        self.call_count += 1
        self.query_history.append({
            "type": "write",
            "query": query,
            "parameters": parameters,
            "timestamp": datetime.utcnow()
        })
        
        if self.should_fail:
            raise Exception(self.failure_message)
        
        self.metrics.record_operation("write", True, 0.002)
        return self.mock_data.get("default_write_result", [])
    
    async def execute_cypher_batch(
        self, cypher_statements_with_params: List[tuple[str, Dict[str, Any]]]
    ) -> None:
        """Mock batch execution."""
        self.call_count += 1
        for query, params in cypher_statements_with_params:
            self.query_history.append({
                "type": "batch",
                "query": query,
                "parameters": params,
                "timestamp": datetime.utcnow()
            })
        
        if self.should_fail:
            raise Exception(self.failure_message)
        
        self.metrics.record_operation("batch", True, 0.005)
    
    async def create_db_schema(self) -> None:
        """Mock schema creation."""
        self.call_count += 1
        if self.should_fail:
            raise Exception(self.failure_message)
    
    def embedding_to_list(self, embedding: np.ndarray | None) -> List[float] | None:
        """Mock embedding conversion."""
        if embedding is None:
            return None
        return [0.1, 0.2, 0.3]  # Simple mock embedding
    
    def list_to_embedding(
        self, embedding_list: List[float | int] | None
    ) -> np.ndarray | None:
        """Mock embedding conversion."""
        if embedding_list is None:
            return None
        return np.array(embedding_list, dtype=np.float32)
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Mock connection info."""
        return {
            "uri": "bolt://localhost:7687",
            "database": "test_db",
            "is_connected": self.is_connected,
            "service_type": "MockNeo4j"
        }
    
    def get_metrics(self) -> DatabaseMetrics:
        """Get mock metrics."""
        return self.metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Mock health check."""
        return {
            "healthy": not self.should_fail,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "connectivity": not self.should_fail,
                "query_test": not self.should_fail
            }
        }
    
    async def verify_connectivity(self) -> bool:
        """Mock connectivity verification."""
        return not self.should_fail
    
    def get_service_info(self) -> Dict[str, Any]:
        """Mock service info."""
        return {
            "service_name": "MockDatabaseService",
            "service_type": "database",
            "call_count": self.call_count,
            "query_history_length": len(self.query_history)
        }
    
    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.query_history.clear()
        self.should_fail = False
        self.is_connected = False
        self.metrics = DatabaseMetrics()


class MockTypeInferenceService:
    """Mock type inference service for testing."""
    
    def __init__(self, mock_data: Dict[str, Any] = None):
        self.mock_data = mock_data or {}
        self.call_count = 0
        self.inference_history: List[Dict[str, Any]] = []
        self.should_fail = False
        self.failure_message = "Mock type inference failure"
        
        # Default type mappings
        self.default_types = {
            "subject": "Character",
            "object": "Location",
            "literal": "ValueNode"
        }
    
    def infer_subject_type(self, subject_info: Dict[str, Any]) -> str:
        """Mock subject type inference."""
        self.call_count += 1
        self.inference_history.append({
            "method": "infer_subject_type",
            "input": subject_info,
            "timestamp": datetime.utcnow()
        })
        
        if self.should_fail:
            raise Exception(self.failure_message)
        
        # Use custom mapping if provided
        subject_name = subject_info.get("name", "").lower()
        custom_mapping = self.mock_data.get("subject_type_mapping", {})
        
        for pattern, type_name in custom_mapping.items():
            if pattern in subject_name:
                return type_name
        
        return self.mock_data.get("default_subject_type", self.default_types["subject"])
    
    def infer_object_type(
        self, object_info: Dict[str, Any], is_literal: bool = False
    ) -> str:
        """Mock object type inference."""
        self.call_count += 1
        self.inference_history.append({
            "method": "infer_object_type",
            "input": object_info,
            "is_literal": is_literal,
            "timestamp": datetime.utcnow()
        })
        
        if self.should_fail:
            raise Exception(self.failure_message)
        
        if is_literal:
            return self.mock_data.get("literal_type", self.default_types["literal"])
        
        object_name = object_info.get("name", "").lower()
        custom_mapping = self.mock_data.get("object_type_mapping", {})
        
        for pattern, type_name in custom_mapping.items():
            if pattern in object_name:
                return type_name
        
        return self.mock_data.get("default_object_type", self.default_types["object"])
    
    def get_service_info(self) -> Dict[str, Any]:
        """Mock service info."""
        return {
            "service_name": "MockTypeInferenceService",
            "service_type": "type_inference",
            "call_count": self.call_count,
            "inference_history_length": len(self.inference_history)
        }
    
    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.inference_history.clear()
        self.should_fail = False


class MockValidationService:
    """Mock validation service for testing."""
    
    def __init__(self, mock_data: Dict[str, Any] = None):
        self.mock_data = mock_data or {}
        self.call_count = 0
        self.validation_history: List[Dict[str, Any]] = []
        self.should_fail = False
        self.failure_message = "Mock validation failure"
        
        # Default validation behavior - always pass unless overridden
        self.default_validation_result = {
            "is_valid": True,
            "confidence": 0.9,
            "message": "Mock validation passed"
        }
    
    def validate_relationship(
        self,
        subject_type: str,
        predicate: str,
        object_type: str,
        context: Dict[str, Any] = None,
    ) -> Any:
        """Mock relationship validation."""
        self.call_count += 1
        self.validation_history.append({
            "method": "validate_relationship",
            "subject_type": subject_type,
            "predicate": predicate,
            "object_type": object_type,
            "context": context,
            "timestamp": datetime.utcnow()
        })
        
        if self.should_fail:
            raise Exception(self.failure_message)
        
        # Check for custom validation rules
        validation_key = f"{subject_type}_{predicate}_{object_type}"
        custom_result = self.mock_data.get("validation_overrides", {}).get(validation_key)
        
        if custom_result:
            return custom_result
        
        return self.mock_data.get("default_validation_result", self.default_validation_result)
    
    def validate_triple(self, triple_dict: Dict[str, Any]) -> Any:
        """Mock triple validation."""
        self.call_count += 1
        self.validation_history.append({
            "method": "validate_triple",
            "triple": triple_dict,
            "timestamp": datetime.utcnow()
        })
        
        if self.should_fail:
            raise Exception(self.failure_message)
        
        return self.mock_data.get("default_triple_validation", self.default_validation_result)
    
    def validate_batch(self, triples: List[Dict[str, Any]]) -> List[Any]:
        """Mock batch validation."""
        self.call_count += 1
        self.validation_history.append({
            "method": "validate_batch",
            "batch_size": len(triples),
            "timestamp": datetime.utcnow()
        })
        
        if self.should_fail:
            raise Exception(self.failure_message)
        
        # Return validation result for each triple
        return [self.default_validation_result] * len(triples)
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Mock validation statistics."""
        return {
            "total_validations": self.call_count,
            "validation_history_length": len(self.validation_history)
        }
    
    def get_service_info(self) -> Dict[str, Any]:
        """Mock service info."""
        return {
            "service_name": "MockValidationService",
            "service_type": "validation",
            "call_count": self.call_count
        }
    
    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.validation_history.clear()
        self.should_fail = False


class TestServiceRegistry:
    """
    Isolated service registry for testing.
    
    Provides a clean service registry instance for each test,
    preventing test interference and reducing setup complexity.
    """
    
    def __init__(self):
        self.registry = ServiceRegistry()
        self.lifecycle_manager = ServiceLifecycleManager(self.registry)
        self.mock_services: Dict[str, Any] = {}
        self.original_services: Dict[str, Any] = {}
        
    def register_mock_service(
        self, 
        name: str, 
        mock_instance: Any,
        interface: Type = None,
        lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON
    ):
        """Register a mock service instance."""
        self.mock_services[name] = mock_instance
        self.registry.register_instance(name, mock_instance, interface)
        return self
    
    def register_mock_database(self, mock_data: Dict[str, Any] = None) -> MockDatabaseService:
        """Register a mock database service."""
        mock_db = MockDatabaseService(mock_data)
        self.register_mock_service("database_service", mock_db, DatabaseInterface)
        return mock_db
    
    def register_mock_type_inference(self, mock_data: Dict[str, Any] = None) -> MockTypeInferenceService:
        """Register a mock type inference service."""
        mock_service = MockTypeInferenceService(mock_data)
        self.register_mock_service("type_inference_service", mock_service, TypeInferenceServiceInterface)
        return mock_service
    
    def register_mock_validation(self, mock_data: Dict[str, Any] = None) -> MockValidationService:
        """Register a mock validation service."""
        mock_service = MockValidationService(mock_data)
        self.register_mock_service("validation_service", mock_service, ValidationServiceInterface)
        return mock_service
    
    def setup_common_mocks(self) -> Dict[str, Any]:
        """Setup commonly used mock services."""
        mocks = {
            "database": self.register_mock_database(),
            "type_inference": self.register_mock_type_inference(), 
            "validation": self.register_mock_validation()
        }
        return mocks
    
    def get_service(self, name: str) -> Any:
        """Get a service from the test registry."""
        return self.registry.resolve(name)
    
    def reset_all_mocks(self):
        """Reset all mock services to clean state."""
        for mock_service in self.mock_services.values():
            if hasattr(mock_service, 'reset'):
                mock_service.reset()
    
    async def initialize_services(self) -> bool:
        """Initialize all test services."""
        return await self.lifecycle_manager.initialize_services()
    
    async def shutdown_services(self) -> bool:
        """Shutdown all test services."""
        return await self.lifecycle_manager.shutdown_services()
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get test registry status."""
        return {
            "registered_services": list(self.registry._services.keys()),
            "mock_services": list(self.mock_services.keys()),
            "registry_metrics": self.registry.get_registry_metrics()
        }


class TestContextManager:
    """
    Context manager for isolated test environments.
    
    Provides automatic setup/teardown of test services and
    prevents test interference through proper isolation.
    """
    
    def __init__(self, auto_setup_mocks: bool = True):
        self.auto_setup_mocks = auto_setup_mocks
        self.test_registry: Optional[TestServiceRegistry] = None
        self.patches: List[Any] = []
        self.mock_services: Dict[str, Any] = {}
        
    @contextmanager
    def test_service_context(self, mock_configs: List[MockServiceConfig] = None):
        """Context manager for test services."""
        self.test_registry = TestServiceRegistry()
        
        try:
            # Setup mock services
            if self.auto_setup_mocks:
                self.mock_services = self.test_registry.setup_common_mocks()
            
            # Setup custom mock configurations
            if mock_configs:
                for config in mock_configs:
                    if config.service_name == "database_service":
                        mock_service = self.test_registry.register_mock_database(config.mock_data)
                    elif config.service_name == "type_inference_service":
                        mock_service = self.test_registry.register_mock_type_inference(config.mock_data)
                    elif config.service_name == "validation_service":
                        mock_service = self.test_registry.register_mock_validation(config.mock_data)
                    else:
                        # Generic mock service
                        mock_service = Mock()
                        for method_name, behavior in config.behavior_overrides.items():
                            setattr(mock_service, method_name, behavior)
                        self.test_registry.register_mock_service(
                            config.service_name, mock_service, config.interface, config.lifecycle
                        )
                    
                    self.mock_services[config.service_name] = mock_service
            
            # Patch global service registry access to use test registry
            with patch('core.service_registry._registry', self.test_registry.registry):
                with patch('core.service_registry.get_service_registry', lambda: self.test_registry.registry):
                    yield self.test_registry, self.mock_services
                    
        finally:
            # Cleanup
            if self.test_registry:
                self.test_registry.reset_all_mocks()
            
            # Clear patches
            for patch_obj in self.patches:
                if hasattr(patch_obj, 'stop'):
                    patch_obj.stop()
            self.patches.clear()
    
    @asynccontextmanager
    async def async_test_service_context(self, mock_configs: List[MockServiceConfig] = None):
        """Async context manager for test services."""
        with self.test_service_context(mock_configs) as (test_registry, mock_services):
            try:
                # Initialize services
                await test_registry.initialize_services()
                yield test_registry, mock_services
            finally:
                # Shutdown services
                await test_registry.shutdown_services()


# Convenience functions for common test scenarios
def create_test_registry() -> TestServiceRegistry:
    """Create a new test service registry."""
    return TestServiceRegistry()


def create_mock_database(mock_data: Dict[str, Any] = None) -> MockDatabaseService:
    """Create a mock database service."""
    return MockDatabaseService(mock_data)


def create_mock_type_inference(mock_data: Dict[str, Any] = None) -> MockTypeInferenceService:
    """Create a mock type inference service."""
    return MockTypeInferenceService(mock_data)


def create_mock_validation(mock_data: Dict[str, Any] = None) -> MockValidationService:
    """Create a mock validation service."""
    return MockValidationService(mock_data)


@contextmanager
def mock_service_context(
    database_data: Dict[str, Any] = None,
    type_inference_data: Dict[str, Any] = None,
    validation_data: Dict[str, Any] = None,
    additional_mocks: Dict[str, MockServiceConfig] = None
):
    """
    Simplified context manager for common mock scenarios.
    
    This replaces the need for extensive monkeypatching in tests.
    """
    context_manager = TestContextManager()
    
    mock_configs = []
    
    if database_data is not None:
        mock_configs.append(MockServiceConfig(
            service_name="database_service",
            interface=DatabaseInterface,
            mock_data=database_data
        ))
    
    if type_inference_data is not None:
        mock_configs.append(MockServiceConfig(
            service_name="type_inference_service", 
            interface=TypeInferenceServiceInterface,
            mock_data=type_inference_data
        ))
    
    if validation_data is not None:
        mock_configs.append(MockServiceConfig(
            service_name="validation_service",
            interface=ValidationServiceInterface,
            mock_data=validation_data
        ))
    
    if additional_mocks:
        for name, config in additional_mocks.items():
            config.service_name = name
            mock_configs.append(config)
    
    with context_manager.test_service_context(mock_configs) as (test_registry, mock_services):
        yield test_registry, mock_services


@asynccontextmanager
async def async_mock_service_context(
    database_data: Dict[str, Any] = None,
    type_inference_data: Dict[str, Any] = None,
    validation_data: Dict[str, Any] = None,
    additional_mocks: Dict[str, MockServiceConfig] = None
):
    """
    Async version of mock_service_context.
    """
    context_manager = TestContextManager()
    
    mock_configs = []
    
    if database_data is not None:
        mock_configs.append(MockServiceConfig(
            service_name="database_service",
            interface=DatabaseInterface,
            mock_data=database_data
        ))
    
    if type_inference_data is not None:
        mock_configs.append(MockServiceConfig(
            service_name="type_inference_service",
            interface=TypeInferenceServiceInterface, 
            mock_data=type_inference_data
        ))
    
    if validation_data is not None:
        mock_configs.append(MockServiceConfig(
            service_name="validation_service",
            interface=ValidationServiceInterface,
            mock_data=validation_data
        ))
    
    if additional_mocks:
        for name, config in additional_mocks.items():
            config.service_name = name
            mock_configs.append(config)
    
    async with context_manager.async_test_service_context(mock_configs) as (test_registry, mock_services):
        yield test_registry, mock_services


# Legacy compatibility helpers for gradual migration
def patch_validation_service_provider(mock_service: Any):
    """Helper to patch validation service provider for legacy tests."""
    return patch('core.validation_service_provider.get_service_provider', 
                return_value=Mock(
                    get_type_inference_service=Mock(return_value=mock_service),
                    get_validation_service=Mock(return_value=mock_service)
                ))


def patch_database_manager(mock_db: MockDatabaseService):
    """Helper to patch database manager for legacy tests."""
    return patch('core.db_manager.neo4j_manager', mock_db)


# Test utilities
def assert_service_called(mock_service: Any, method_name: str, min_times: int = 1):
    """Assert that a mock service method was called minimum number of times."""
    if hasattr(mock_service, 'call_count'):
        assert mock_service.call_count >= min_times, f"Service was called {mock_service.call_count} times, expected at least {min_times}"
    elif hasattr(mock_service, method_name):
        method = getattr(mock_service, method_name)
        if hasattr(method, 'call_count'):
            assert method.call_count >= min_times, f"Method {method_name} was called {method.call_count} times, expected at least {min_times}"


def get_service_call_history(mock_service: Any, method_name: str = None) -> List[Dict[str, Any]]:
    """Get call history from a mock service."""
    if hasattr(mock_service, 'validation_history'):
        return mock_service.validation_history
    elif hasattr(mock_service, 'inference_history'):
        return mock_service.inference_history
    elif hasattr(mock_service, 'query_history'):
        return mock_service.query_history
    else:
        return []


logger.info("Service mocks module loaded - ready to reduce test complexity!")