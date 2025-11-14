# tests/test_service_integration.py
#!/usr/bin/env python3
"""
Integration test to verify that services work with the new lightweight cache.
"""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from core.http_client_service import EmbeddingHTTPClient, HTTPClientService
from core.lightweight_cache import get_cache_metrics, get_cached_value, set_cached_value
from core.llm_interface_refactored import EmbeddingService
from core.schema_introspector import SchemaIntrospector


def test_embedding_service_cache():
    """Test that EmbeddingService works with lightweight cache."""
    print("Testing EmbeddingService cache integration...")

    # Create dummy embedding service wired to a basic HTTP client
    # (actual network calls are not exercised in this test)
    mock_http_client = HTTPClientService()
    embedding_client = EmbeddingHTTPClient(mock_http_client)
    EmbeddingService(embedding_client)

    # Test that the service registered itself
    metrics = get_cache_metrics("llm_embedding")
    assert isinstance(metrics, dict), "Cache metrics should be available"
    print("✓ Embedding service registered successfully")

    print("Embedding service cache test passed!")


def test_schema_introspector_cache():
    """Test that SchemaIntrospector works with lightweight cache."""
    print("Testing SchemaIntrospector cache integration...")

    # Create schema introspector
    SchemaIntrospector()

    # Test cache operations
    test_labels = {"Person", "Location", "Event"}
    set_cached_value("active_labels", test_labels, "schema_introspection")

    cached_result = get_cached_value("active_labels", "schema_introspection")
    assert cached_result == test_labels, "Cached value should match"

    # Test metrics
    metrics = get_cache_metrics("schema_introspection")
    assert isinstance(metrics, dict), "Cache metrics should be available"
    print("✓ Schema introspector cache operations work")

    print("Schema introspector cache test passed!")


def test_cross_service_isolation():
    """Test that different services have isolated caches."""
    print("Testing cross-service cache isolation...")

    # Set same key in different services
    key = "test_key"
    value1 = "value1"
    value2 = "value2"

    set_cached_value(key, value1, "llm_embedding")
    set_cached_value(key, value2, "schema_introspection")

    # Verify isolation
    result1 = get_cached_value(key, "llm_embedding")
    result2 = get_cached_value(key, "schema_introspection")

    assert result1 == value1, f"llm_embedding should have {value1}"
    assert result2 == value2, f"schema_introspection should have {value2}"
    assert result1 != result2, "Services should have isolated caches"
    print("✓ Cross-service isolation works")

    print("Cross-service isolation test passed!")


if __name__ == "__main__":
    test_embedding_service_cache()
    test_schema_introspector_cache()
    test_cross_service_isolation()
    print("\n🎉 All integration tests passed!")
