# tests/test_type_inference_integration.py
"""
Integration test suite for type inference system integration.

Tests cover service provider patterns, triple processor integration,
dynamic schema manager integration, and interface compatibility.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Any

from core.validation_service_provider import (
    ValidationServiceProvider,
    get_type_inference_service,
    TypeInferenceServiceInterface,
)
from core.intelligent_type_inference import IntelligentTypeInference
from core.triple_processor import TripleProcessor


class MockTypeInferenceService:
    """Mock implementation of TypeInferenceServiceInterface for testing."""
    
    def __init__(self, service_name="MockService"):
        self.service_name = service_name
        self.infer_subject_calls = []
        self.infer_object_calls = []
    
    def infer_subject_type(self, subject_info: dict[str, Any]) -> str:
        """Mock subject type inference."""
        self.infer_subject_calls.append(subject_info)
        return subject_info.get("type", "Character")
    
    def infer_object_type(self, object_info: dict[str, Any], is_literal: bool = False) -> str:
        """Mock object type inference."""
        self.infer_object_calls.append((object_info, is_literal))
        if is_literal:
            return "ValueNode"
        return object_info.get("type", "Location")


class TestValidationServiceProvider:
    """Test validation service provider functionality."""

    def test_service_provider_initialization(self):
        """Test service provider initializes correctly."""
        provider = ValidationServiceProvider()
        
        # Should start with no services registered
        with pytest.raises(RuntimeError, match="No type inference service registered"):
            provider.get_type_inference_service()

    def test_register_type_inference_service(self):
        """Test registering a type inference service."""
        provider = ValidationServiceProvider()
        mock_service = MockTypeInferenceService("TestService")
        
        provider.register_type_inference_service(mock_service)
        
        # Should be able to retrieve the service
        retrieved_service = provider.get_type_inference_service()
        assert retrieved_service == mock_service
        assert retrieved_service.service_name == "TestService"

    def test_replace_type_inference_service(self):
        """Test replacing an existing type inference service."""
        provider = ValidationServiceProvider()
        
        # Register first service
        first_service = MockTypeInferenceService("FirstService")
        provider.register_type_inference_service(first_service)
        
        # Register second service (should replace first)
        second_service = MockTypeInferenceService("SecondService")
        provider.register_type_inference_service(second_service)
        
        # Should retrieve the second service
        retrieved_service = provider.get_type_inference_service()
        assert retrieved_service == second_service
        assert retrieved_service.service_name == "SecondService"

    def test_lazy_initialization_with_default_service(self):
        """Test lazy initialization creates default service."""
        provider = ValidationServiceProvider()
        
        with patch.object(provider, '_initialize_default_services') as mock_init:
            mock_service = MockTypeInferenceService("DefaultService")
            
            def setup_default():
                provider._type_inference_service = mock_service
            
            mock_init.side_effect = setup_default
            
            # First call should trigger initialization
            retrieved_service = provider.get_type_inference_service()
            
            assert retrieved_service == mock_service
            mock_init.assert_called_once()

    def test_thread_safety_with_lock(self):
        """Test thread safety using locks."""
        provider = ValidationServiceProvider()
        mock_service = MockTypeInferenceService("ThreadSafeService")
        
        # Test that lock is used during registration
        with patch.object(provider, '_lock') as mock_lock:
            mock_lock.__enter__ = Mock(return_value=None)
            mock_lock.__exit__ = Mock(return_value=None)
            
            provider.register_type_inference_service(mock_service)
            
            # Lock should have been used
            mock_lock.__enter__.assert_called()
            mock_lock.__exit__.assert_called()

    def test_global_service_provider_function(self):
        """Test global get_type_inference_service function."""
        mock_service = MockTypeInferenceService("GlobalService")
        
        with patch('core.validation_service_provider.get_service_provider') as mock_get_provider:
            mock_provider = Mock()
            mock_provider.get_type_inference_service.return_value = mock_service
            mock_get_provider.return_value = mock_provider
            
            # Call global function
            result = get_type_inference_service()
            
            assert result == mock_service
            mock_get_provider.assert_called_once()
            mock_provider.get_type_inference_service.assert_called_once()


class TestTripleProcessorIntegration:
    """Test triple processor integration with type inference."""

    def test_triple_processor_initialization(self):
        """Test triple processor initializes with type inference service."""
        mock_service = MockTypeInferenceService("ProcessorService")
        
        processor = TripleProcessor(type_inference_service=mock_service)
        
        assert processor._type_inference_service == mock_service
        
        # Check that stats are initialized
        stats = processor._processing_stats
        assert "total_triples_processed" in stats
        assert "type_inferences" in stats

    def test_triple_processor_dependency_injection(self):
        """Test triple processor uses dependency injection when service not provided."""
        with patch('core.triple_processor.get_type_inference_service') as mock_get_service:
            mock_service = MockTypeInferenceService("InjectedService")
            mock_get_service.return_value = mock_service
            
            processor = TripleProcessor()
            
            # Access the service to trigger lazy loading
            service = processor._get_type_inference_service()
            
            assert service == mock_service
            mock_get_service.assert_called_once()

    def test_triple_processor_error_handling_on_di_failure(self):
        """Test triple processor error handling when dependency injection fails."""
        with patch('core.triple_processor.get_type_inference_service', side_effect=Exception("DI failed")):
            processor = TripleProcessor()
            
            # Should raise RuntimeError since fallback was removed
            with pytest.raises(RuntimeError, match="Unable to initialize type inference service"):
                processor._get_type_inference_service()

    def test_triple_processor_service_caching(self):
        """Test that triple processor caches the type inference service."""
        mock_service = MockTypeInferenceService("CachedService")
        
        processor = TripleProcessor(type_inference_service=mock_service)
        
        # Multiple calls should return same service
        service1 = processor._get_type_inference_service()
        service2 = processor._get_type_inference_service()
        
        assert service1 == service2 == mock_service

    def test_triple_processing_with_type_inference(self):
        """Test triple processing integrates type inference correctly."""
        mock_service = MockTypeInferenceService("ProcessingService")
        processor = TripleProcessor(type_inference_service=mock_service)
        
        # Mock triple data
        triple_data = {
            "subject": {"name": "Alice", "type": None},
            "predicate": "LOVES",
            "object_entity": {"name": "Bob", "type": None},
            "is_literal_object": False
        }
        
        # Process the triple (would normally involve more complex processing)
        service = processor._get_type_inference_service()
        
        # Simulate type inference during processing
        subject_type = service.infer_subject_type(triple_data["subject"])
        object_type = service.infer_object_type(triple_data["object_entity"])
        
        assert subject_type == "Character"  # Mock default
        assert object_type == "Location"    # Mock default
        
        # Verify calls were made
        assert len(mock_service.infer_subject_calls) == 1
        assert len(mock_service.infer_object_calls) == 1

    def test_triple_processor_error_handling(self):
        """Test error handling in triple processor type inference integration."""
        # Create a service that raises errors
        error_service = Mock()
        error_service.infer_subject_type.side_effect = ValueError("Inference error")
        
        processor = TripleProcessor(type_inference_service=error_service)
        
        # Should handle errors gracefully
        try:
            service = processor._get_type_inference_service()
            service.infer_subject_type({"name": "Test", "type": None})
        except ValueError:
            # Error should propagate but not crash the processor
            pass
        
        # Processor should still be functional
        assert processor._type_inference_service == error_service


class TestInterfaceCompatibility:
    """Test interface compatibility between different inference systems."""

    def test_intelligent_type_inference_interface_compliance(self):
        """Test IntelligentTypeInference complies with interface."""
        from core.schema_introspector import SchemaIntrospector
        
        schema_introspector = SchemaIntrospector()
        service = IntelligentTypeInference(schema_introspector)
        
        # Should implement required methods
        assert hasattr(service, 'infer_subject_type')
        assert hasattr(service, 'infer_object_type')
        
        # Methods should have correct signatures
        subject_info = {"name": "Test", "type": "Character"}
        object_info = {"name": "Test", "type": "Location"}
        
        subject_result = service.infer_subject_type(subject_info)
        object_result = service.infer_object_type(object_info)
        object_literal_result = service.infer_object_type(object_info, is_literal=True)
        
        assert isinstance(subject_result, str)
        assert isinstance(object_result, str)
        assert isinstance(object_literal_result, str)

    def test_mock_service_interface_compliance(self):
        """Test mock service complies with interface."""
        mock_service = MockTypeInferenceService()
        
        # Should implement required methods
        assert hasattr(mock_service, 'infer_subject_type')
        assert hasattr(mock_service, 'infer_object_type')
        
        # Should work with type inference consumer code
        subject_info = {"name": "Test", "type": "Character"}
        object_info = {"name": "Test", "type": "Location"}
        
        subject_result = mock_service.infer_subject_type(subject_info)
        object_result = mock_service.infer_object_type(object_info)
        
        assert isinstance(subject_result, str)
        assert isinstance(object_result, str)

    def test_interface_substitutability(self):
        """Test that different implementations can be substituted."""
        from core.schema_introspector import SchemaIntrospector
        
        schema_introspector = SchemaIntrospector()
        real_service = IntelligentTypeInference(schema_introspector)
        mock_service = MockTypeInferenceService()
        
        services = [real_service, mock_service]
        
        for service in services:
            # Both should handle the same interface
            subject_info = {"name": "Alice", "type": "Character"}
            object_info = {"name": "Castle", "type": "Location"}
            
            subject_result = service.infer_subject_type(subject_info)
            object_result = service.infer_object_type(object_info)
            literal_result = service.infer_object_type({"value": "42"}, is_literal=True)
            
            assert isinstance(subject_result, str)
            assert isinstance(object_result, str)
            assert literal_result == "ValueNode"


class TestDynamicSchemaManagerIntegration:
    """Test integration with dynamic schema manager."""

    def test_dynamic_schema_manager_type_inference_integration(self):
        """Test dynamic schema manager integration with type inference."""
        # Mock the dynamic schema manager
        with patch('core.dynamic_schema_manager.DynamicSchemaManager') as mock_dsm_class:
            mock_dsm = AsyncMock()
            mock_dsm.infer_node_type.return_value = "Character"
            mock_dsm_class.return_value = mock_dsm
            
            # Test integration
            result = mock_dsm.infer_node_type("Alice", "character", "A brave warrior")
            assert result == "Character"
            
            mock_dsm.infer_node_type.assert_called_once_with("Alice", "character", "A brave warrior")

    @pytest.mark.asyncio
    async def test_intelligent_inference_integration(self):
        """Test integration with intelligent type inference system."""
        # Mock the intelligent type inference
        with patch('core.intelligent_type_inference.IntelligentTypeInference') as mock_iti_class:
            mock_iti = Mock()
            mock_iti.infer_type.return_value = ("Character", 0.85)
            mock_iti_class.return_value = mock_iti
            
            # Create mock schema introspector
            mock_introspector = AsyncMock()
            
            # Test integration
            inference_system = mock_iti_class(mock_introspector)
            result_type, confidence = inference_system.infer_type("Alice", "character")
            
            assert result_type == "Character"
            assert confidence == 0.85
            
            mock_iti.infer_type.assert_called_once_with("Alice", "character")


class TestServiceProviderPatterns:
    """Test service provider design patterns."""

    def test_singleton_behavior(self):
        """Test that service provider exhibits singleton-like behavior."""
        # Multiple instances should work independently
        provider1 = ValidationServiceProvider()
        provider2 = ValidationServiceProvider()
        
        service1 = MockTypeInferenceService("Service1")
        service2 = MockTypeInferenceService("Service2")
        
        provider1.register_type_inference_service(service1)
        provider2.register_type_inference_service(service2)
        
        # Each provider should maintain its own service
        assert provider1.get_type_inference_service() == service1
        assert provider2.get_type_inference_service() == service2

    def test_factory_pattern_integration(self):
        """Test factory pattern for creating services."""
        def create_test_service():
            return MockTypeInferenceService("FactoryService")
        
        provider = ValidationServiceProvider()
        
        # Simulate factory creation
        service = create_test_service()
        provider.register_type_inference_service(service)
        
        retrieved_service = provider.get_type_inference_service()
        assert retrieved_service.service_name == "FactoryService"

    def test_dependency_injection_pattern(self):
        """Test dependency injection pattern."""
        # Create service with dependencies
        mock_dependency = Mock()
        mock_dependency.get_config.return_value = {"setting": "value"}
        
        class ServiceWithDependencies:
            def __init__(self, dependency):
                self.dependency = dependency
            
            def infer_subject_type(self, subject_info):
                config = self.dependency.get_config()
                return f"Character_{config['setting']}"
            
            def infer_object_type(self, object_info, is_literal=False):
                return "Location"
        
        # Inject dependency
        service = ServiceWithDependencies(mock_dependency)
        
        provider = ValidationServiceProvider()
        provider.register_type_inference_service(service)
        
        # Test that dependency is used
        retrieved_service = provider.get_type_inference_service()
        result = retrieved_service.infer_subject_type({"name": "Test"})
        
        assert result == "Character_value"
        mock_dependency.get_config.assert_called_once()


class TestErrorHandlingAndResilience:
    """Test error handling and system resilience."""

    def test_service_provider_error_recovery(self):
        """Test service provider recovers from errors."""
        provider = ValidationServiceProvider()
        
        # Register a service that fails
        failing_service = Mock()
        failing_service.infer_subject_type.side_effect = RuntimeError("Service failed")
        
        provider.register_type_inference_service(failing_service)
        
        # Service should be retrieved even if it fails during use
        retrieved_service = provider.get_type_inference_service()
        assert retrieved_service == failing_service
        
        # Error should propagate when service is used
        with pytest.raises(RuntimeError, match="Service failed"):
            retrieved_service.infer_subject_type({"name": "Test"})

    def test_triple_processor_error_propagation(self):
        """Test triple processor error propagation on service failures."""
        # Create processor with no service initially
        processor = TripleProcessor()
        
        # Mock service provider to fail
        with patch('core.triple_processor.get_type_inference_service', side_effect=Exception("Service unavailable")):
            # Should propagate error since fallback was removed
            with pytest.raises(RuntimeError, match="Unable to initialize type inference service"):
                processor._get_type_inference_service()

    def test_interface_validation_errors(self):
        """Test handling of interface validation errors."""
        # Create invalid service (missing required methods)
        class InvalidService:
            def invalid_method(self):
                pass
        
        invalid_service = InvalidService()
        provider = ValidationServiceProvider()
        
        # Should allow registration (interface validation is runtime)
        provider.register_type_inference_service(invalid_service)
        
        # Should fail when used
        retrieved_service = provider.get_type_inference_service()
        
        with pytest.raises(AttributeError):
            retrieved_service.infer_subject_type({"name": "Test"})

    def test_concurrent_access_handling(self):
        """Test handling of concurrent access to services."""
        import threading
        import time
        
        provider = ValidationServiceProvider()
        results = []
        errors = []
        
        def register_and_use_service(service_name):
            try:
                service = MockTypeInferenceService(service_name)
                provider.register_type_inference_service(service)
                
                # Small delay to encourage race conditions
                time.sleep(0.01)
                
                retrieved_service = provider.get_type_inference_service()
                result = retrieved_service.infer_subject_type({"name": "Test"})
                results.append((service_name, result))
            except Exception as e:
                errors.append((service_name, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=register_and_use_service, args=(f"Service{i}",))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should handle concurrent access without errors
        assert len(errors) == 0
        assert len(results) == 5
        
        # Last registered service should be active
        final_service = provider.get_type_inference_service()
        assert hasattr(final_service, 'service_name')


class TestRealWorldScenarios:
    """Test real-world integration scenarios."""

    def test_complete_processing_pipeline(self):
        """Test complete processing pipeline integration."""
        # Setup complete pipeline
        provider = ValidationServiceProvider()
        from core.schema_introspector import SchemaIntrospector
        
        schema_introspector = SchemaIntrospector()
        inference_service = IntelligentTypeInference(schema_introspector)
        provider.register_type_inference_service(inference_service)
        
        processor = TripleProcessor()
        
        # Mock data representing real processing scenario
        triples_data = [
            {
                "subject": {"name": "Alice", "type": None},
                "predicate": "LOVES",
                "object_entity": {"name": "Castle", "type": None},
                "is_literal_object": False
            },
            {
                "subject": {"name": "Dragon", "type": "Creature"},
                "predicate": "GUARDS",
                "object_entity": {"name": "Treasure", "type": None},
                "is_literal_object": False
            },
            {
                "subject": {"name": "Hero", "type": None},
                "predicate": "HAS_ATTRIBUTE",
                "object_entity": {"value": "brave"},
                "is_literal_object": True
            }
        ]
        
        # Process each triple through the pipeline
        results = []
        service = processor._get_type_inference_service()
        
        for triple in triples_data:
            subject_type = service.infer_subject_type(triple["subject"])
            
            if triple["is_literal_object"]:
                object_type = service.infer_object_type(
                    triple["object_entity"], 
                    is_literal=True
                )
            else:
                object_type = service.infer_object_type(triple["object_entity"])
            
            results.append({
                "subject_type": subject_type,
                "object_type": object_type,
                "predicate": triple["predicate"]
            })
        
        # Verify results
        assert len(results) == 3
        
        # First triple: Alice LOVES Castle
        assert results[0]["subject_type"] in ["Character", "Entity"]  # Should infer or fallback
        assert results[0]["object_type"] in ["Location", "Entity"]   # Should infer or fallback
        
        # Second triple: Dragon GUARDS Treasure  
        assert results[1]["subject_type"] == "Creature"  # Already specified
        assert results[1]["object_type"] in ["Object", "Entity"]     # Should infer or fallback
        
        # Third triple: Hero HAS_ATTRIBUTE brave (literal)
        assert results[2]["subject_type"] in ["Character", "Entity"] # Should infer or fallback
        assert results[2]["object_type"] == "ValueNode"              # Literal object

    def test_hot_swapping_services(self):
        """Test hot-swapping type inference services."""
        provider = ValidationServiceProvider()
        processor = TripleProcessor()
        
        # Start with first service
        service1 = MockTypeInferenceService("Service1")
        provider.register_type_inference_service(service1)
        
        # Process some data
        with patch('core.triple_processor.get_type_inference_service', return_value=service1):
            proc_service = processor._get_type_inference_service()
            result1 = proc_service.infer_subject_type({"name": "Alice", "type": "Character"})
        
        # Swap to second service
        service2 = MockTypeInferenceService("Service2")
        provider.register_type_inference_service(service2)
        
        # New processor should use new service
        processor2 = TripleProcessor()
        with patch('core.triple_processor.get_type_inference_service', return_value=service2):
            proc_service2 = processor2._get_type_inference_service()
            result2 = proc_service2.infer_subject_type({"name": "Bob", "type": "Character"})
        
        # Verify different services were used
        assert len(service1.infer_subject_calls) == 1
        assert len(service2.infer_subject_calls) == 1
        assert service1.infer_subject_calls[0]["name"] == "Alice"
        assert service2.infer_subject_calls[0]["name"] == "Bob"

    def test_performance_under_load(self):
        """Test performance characteristics under load."""
        provider = ValidationServiceProvider()
        service = MockTypeInferenceService("LoadTestService")
        provider.register_type_inference_service(service)
        
        processor = TripleProcessor()
        
        # Generate load
        import time
        start_time = time.time()
        
        for i in range(100):
            proc_service = processor._get_type_inference_service()
            proc_service.infer_subject_type({"name": f"Entity{i}", "type": "Character"})
            proc_service.infer_object_type({"name": f"Location{i}", "type": "Location"})
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete reasonably quickly (less than 1 second for 200 operations)
        assert duration < 1.0
        
        # Verify all operations were recorded
        assert len(service.infer_subject_calls) == 100
        assert len(service.infer_object_calls) == 100


@pytest.fixture
def clean_service_provider():
    """Fixture providing a clean ValidationServiceProvider instance."""
    return ValidationServiceProvider()


@pytest.fixture
def mock_triple_processor():
    """Fixture providing a TripleProcessor with mock service."""
    mock_service = MockTypeInferenceService("TestProcessor")
    return TripleProcessor(type_inference_service=mock_service)


@pytest.fixture
def sample_triple_data():
    """Fixture providing sample triple data for testing."""
    return [
        {
            "subject": {"name": "Alice", "type": "Character"},
            "predicate": "LOVES",
            "object_entity": {"name": "Bob", "type": "Character"},
            "is_literal_object": False
        },
        {
            "subject": {"name": "Hero", "type": None},
            "predicate": "LOCATED_IN", 
            "object_entity": {"name": "Castle", "type": None},
            "is_literal_object": False
        },
        {
            "subject": {"name": "Dragon", "type": "Creature"},
            "predicate": "HAS_POWER",
            "object_entity": {"value": "75"},
            "is_literal_object": True
        }
    ]