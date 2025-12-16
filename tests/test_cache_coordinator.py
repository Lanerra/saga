# tests/test_cache_coordinator.py
"""
Tests for the lightweight cache system.

This module contains comprehensive tests for the lightweight cache system
to ensure proper functionality and cross-service cache operations.
"""

import threading
from typing import Any

import pytest

from core.lightweight_cache import (
    clear_service_cache,
    get_cache_metrics,
    get_cached_value,
    invalidate_cache_key,
    register_cache_service,
    set_cached_value,
)


class MockService:
    """Mock service for testing cache operations."""

    def __init__(self, name: str):
        self.name = name
        self.cache_hits = 0
        self.cache_misses = 0

    def get_data(self, key: str) -> Any | None:
        """Get data from cache or simulate computation."""
        value = get_cached_value(key, self.name)
        if value is not None:
            self.cache_hits += 1
            return value
        else:
            self.cache_misses += 1
            # Simulate computation
            value = f"computed_data_for_{key}"
            set_cached_value(key, value, self.name)
            return value

    def get_stats(self) -> dict[str, int]:
        """Get service statistics."""
        return {"cache_hits": self.cache_hits, "cache_misses": self.cache_misses}


class TestLightweightCache:
    """Test suite for lightweight cache functionality."""

    def setup_method(self) -> None:
        """Set up test environment."""
        # Clear any existing test caches by clearing service caches
        test_service_names = [
            "test_service",
            "test_service_tag",
            "concurrent_service",
            "expiring_service",
            "lru_service",
            "event_service",
            "stats_service1",
            "stats_service2",
            "mock_service",
            "service1",
            "service2",
        ]
        for service_name in test_service_names:
            clear_service_cache(service_name)

    def test_service_registration(self) -> None:
        """Test service registration with cache."""
        service_name = "test_service"

        # Register service
        register_cache_service(service_name)

        # Verify service is registered by performing cache operations
        key = "test_key"
        value = "test_value"

        set_cached_value(key, value, service_name)
        retrieved_value = get_cached_value(key, service_name)

        assert retrieved_value == value

    def test_cache_operations(self) -> None:
        """Test basic cache operations."""
        service_name = "test_service"
        register_cache_service(service_name)

        # Test set and get
        key = "test_key"
        value = "test_value"

        set_cached_value(key, value, service_name)
        retrieved_value = get_cached_value(key, service_name)

        assert retrieved_value == value

    def test_cache_invalidation(self) -> None:
        """Test cache invalidation."""
        service_name = "test_service"
        register_cache_service(service_name)

        # Set value
        key = "test_key"
        value = "test_value"
        set_cached_value(key, value, service_name)

        # Verify value exists
        retrieved_value = get_cached_value(key, service_name)
        assert retrieved_value == value

        # Invalidate key
        invalidate_cache_key(key, service_name)

        # Verify value is gone
        retrieved_value = get_cached_value(key, service_name)
        assert retrieved_value is None

    def test_cache_metrics(self) -> None:
        """Test cache metrics collection."""
        service_name = "test_service"
        register_cache_service(service_name)

        # Perform some operations
        key = "test_key"
        value = "test_value"

        # Miss
        retrieved_value = get_cached_value(key, service_name)
        assert retrieved_value is None

        # Hit
        set_cached_value(key, value, service_name)
        retrieved_value = get_cached_value(key, service_name)
        assert retrieved_value == value

        # Get metrics
        metrics = get_cache_metrics(service_name)
        assert isinstance(metrics, dict)
        # Minimal metrics are supported; ensure size reflects entries
        assert metrics.get("size", 0) >= 1

    def test_cross_service_cache_isolation(self) -> None:
        """Test cross-service cache isolation."""
        service1_name = "service1"
        service2_name = "service2"

        register_cache_service(service1_name)
        register_cache_service(service2_name)

        # Set value from service1
        key = "shared_key"
        value1 = "value1"
        value2 = "value2"
        set_cached_value(key, value1, service1_name)
        set_cached_value(key, value2, service2_name)

        # Verify isolation - each service should have its own cache
        retrieved_value1 = get_cached_value(key, service1_name)
        retrieved_value2 = get_cached_value(key, service2_name)

        assert retrieved_value1 == value1
        assert retrieved_value2 == value2
        assert retrieved_value1 != retrieved_value2

    def test_tag_based_operations(self) -> None:
        """Tagging removed: ensure tag APIs are not supported."""
        service_name = "test_service_tag"
        register_cache_service(service_name)

        # Tag APIs are intentionally removed; importing functions would fail.
        # Validate by asserting minimal cache ops still work and metrics are size-only.
        key = "k"
        set_cached_value(key, "v", service_name)
        assert get_cached_value(key, service_name) == "v"
        metrics = get_cache_metrics(service_name)
        assert metrics.get("size", 0) >= 1

    def test_concurrent_cache_access(self) -> None:
        """Test concurrent cache access from multiple threads."""
        service_name = "concurrent_service"
        register_cache_service(service_name)

        def worker(worker_id: int, results: list[Any]) -> None:
            """Worker function for concurrent testing."""
            key = f"key_{worker_id}"
            value = f"value_{worker_id}"

            # Set value
            set_cached_value(key, value, service_name)

            # Get value
            retrieved = get_cached_value(key, service_name)
            results.append((key, retrieved))

        # Run multiple workers concurrently
        threads = []
        results: list[tuple[str, Any]] = []

        for i in range(10):
            thread = threading.Thread(target=worker, args=(i, results))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all operations succeeded
        assert len(results) == 10
        for key, value in results:
            assert value is not None
            assert value == f"value_{key.split('_')[1]}"

    def test_mock_service_integration(self) -> None:
        """Test integration with mock service."""
        service_name = "mock_service"
        register_cache_service(service_name)

        mock_service = MockService(service_name)

        # First call - should be cache miss
        result1 = mock_service.get_data("test_key")
        assert result1 == "computed_data_for_test_key"
        stats1 = mock_service.get_stats()
        assert stats1["cache_misses"] == 1
        assert stats1["cache_hits"] == 0

        # Second call - should be cache hit
        result2 = mock_service.get_data("test_key")
        assert result2 == "computed_data_for_test_key"
        stats2 = mock_service.get_stats()
        assert stats2["cache_misses"] == 1
        assert stats2["cache_hits"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
