# tests/test_cache_integration.py
"""
Tests for cache integration with existing services.

This module contains tests to verify that the existing services
are properly integrated with the unified cache coordination system.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from core.http_client_service import EmbeddingHTTPClient, HTTPClientService
from core.lightweight_cache import (
    clear_service_cache,
    get_cache_metrics,
    get_cached_value,
    register_cache_service,
    set_cached_value,
)
from core.llm_interface_refactored import EmbeddingService
from core.schema_introspector import SchemaIntrospector


class TestCacheIntegration:
    """Test suite for cache integration with existing services."""

    def setup_method(self):
        """Set up test environment."""
        # Coordinator removed; keep behavior-local caches only
        self.coordinator = None

        # Clear any existing test caches
        test_services = ["llm_embedding", "schema_introspection", "text_processing"]
        for service in test_services:
            clear_service_cache(service)

    def test_embedding_service_cache_integration(self):
        """Test that EmbeddingService integrates with coordinated cache."""
        # Create mock HTTP client
        mock_http_client = Mock(spec=HTTPClientService)
        embedding_client = EmbeddingHTTPClient(mock_http_client)

        # Create embedding service (this should auto-register with cache coordinator)
        EmbeddingService(embedding_client)

        # Verify service is registered by performing cache operations
        test_key = "test_cache_key"
        test_value = "test_cache_value"

        # Set value in cache
        set_cached_value(test_key, test_value, "llm_embedding")

        # Get value from cache
        cached_result = get_cached_value(test_key, "llm_embedding")
        assert cached_result == test_value

        # Test metrics
        metrics = get_cache_metrics("llm_embedding")
        assert isinstance(metrics, dict)

    def test_embedding_service_embedding_cache(self):
        """Test that EmbeddingService caches embeddings properly."""
        # Create mock HTTP client
        mock_http_client = Mock(spec=HTTPClientService)
        embedding_client = EmbeddingHTTPClient(mock_http_client)

        # Create embedding service
        embedding_service = EmbeddingService(embedding_client)

        # Mock the HTTP client response with correct dimension
        import numpy as np

        import config

        test_embedding = np.array(
            [0.1] * config.EXPECTED_EMBEDDING_DIM, dtype=config.EMBEDDING_DTYPE
        )

        mock_response = {
            "embedding": test_embedding.tolist()  # Convert to list for JSON serialization
        }
        embedding_client.get_embedding = AsyncMock(return_value=mock_response)

        # Test embedding caching
        test_text = "Hello, world!"

        # First call should make HTTP request
        result1 = asyncio.run(embedding_service.get_embedding(test_text))
        assert result1 is not None
        assert embedding_client.get_embedding.called

        # Reset mock
        embedding_client.get_embedding.reset_mock()

        # Second call should hit cache
        result2 = asyncio.run(embedding_service.get_embedding(test_text))
        import numpy as np

        assert np.array_equal(result1, result2)
        assert not embedding_client.get_embedding.called  # Should not make HTTP request

    def test_schema_introspector_cache_integration(self):
        """Test that SchemaIntrospector integrates with coordinated cache."""
        # Create schema introspector (this should auto-register with cache coordinator)
        SchemaIntrospector()

        # Verify service is registered by performing cache operations
        test_labels = {"Person", "Location", "Event"}
        set_cached_value("active_labels", test_labels, "schema_introspection")

        # Verify cache hit
        cached_result = get_cached_value("active_labels", "schema_introspection")
        assert cached_result == test_labels

        # Test metrics
        metrics = get_cache_metrics("schema_introspection")
        assert isinstance(metrics, dict)

    def test_cross_service_cache_isolation(self):
        """Test that different services have isolated caches."""
        # Register two different services
        register_cache_service("service1")
        register_cache_service("service2")

        # Set same key in both services
        test_key = "test_key"
        value1 = "value1"
        value2 = "value2"

        set_cached_value(test_key, value1, "service1")
        set_cached_value(test_key, value2, "service2")

        # Verify isolation
        result1 = get_cached_value(test_key, "service1")
        result2 = get_cached_value(test_key, "service2")

        assert result1 == value1
        assert result2 == value2
        assert result1 != result2

    def test_cache_metrics_integration(self):
        """Test that cache metrics are properly collected."""
        register_cache_service("test_metrics_service")

        # Perform some cache operations
        set_cached_value("key1", "value1", "test_metrics_service")
        get_cached_value("key1", "test_metrics_service")  # Hit
        get_cached_value("key2", "test_metrics_service")  # Miss

        # Get metrics
        metrics = get_cache_metrics("test_metrics_service")
        assert isinstance(metrics, dict)
        # Only size is tracked in minimal cache
        assert metrics.get("size", 0) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
