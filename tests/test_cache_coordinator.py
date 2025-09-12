# tests/test_cache_coordinator.py
"""
Tests for the unified cache coordination system.

This module contains comprehensive tests for the cache coordination system
to ensure proper functionality, coordination, and cross-service cache sharing.
"""

import asyncio
import threading
import time
from typing import Any, Dict, List, Optional
import pytest

from core.cache_coordinator import (
    CacheCoordinator, 
    get_cache_coordinator,
    register_cache_service,
    get_cached_value,
    set_cached_value,
    invalidate_cache_key,
    invalidate_by_tag,
    get_cache_metrics
)
from core.cache_entry import CachePolicy, CacheInvalidationEvent
from core.cache_manager import get_cache_manager
from core.service_registry import get_service_registry


class MockService:
    """Mock service for testing cache coordination."""
    
    def __init__(self, name: str):
        self.name = name
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_data(self, key: str) -> Optional[Any]:
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
    
    def get_stats(self) -> Dict[str, int]:
        """Get service statistics."""
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses
        }


class TestCacheCoordinator:
    """Test suite for CacheCoordinator functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        # Clear any existing state
        self.coordinator = get_cache_coordinator()
        self.cache_manager = get_cache_manager()
        self.service_registry = get_service_registry()
        
        # Clear caches
        self.cache_manager.clear_all_caches()
        
        # Reset policy manager to ensure clean state
        from core.cache_policies import _policy_manager
        _policy_manager._policies = {}
        _policy_manager._setup_default_policies()
        
        # Remove any existing test caches to ensure fresh creation
        test_service_names = [
            "test_service", "test_service_tag", "concurrent_service", 
            "expiring_service", "lru_service", "event_service",
            "stats_service1", "stats_service2", "mock_service",
            "service1", "service2"
        ]
        for service_name in test_service_names:
            self.cache_manager.remove_cache(service_name)
    
    def test_cache_coordinator_initialization(self):
        """Test that cache coordinator initializes correctly."""
        coordinator = CacheCoordinator()
        assert coordinator is not None
        
        # Test global coordinator access
        global_coordinator = get_cache_coordinator()
        assert global_coordinator is not None
        assert isinstance(global_coordinator, CacheCoordinator)
    
    def test_service_registration(self):
        """Test service registration with cache coordinator."""
        service_name = "test_service"
        policy = CachePolicy(max_size=50, ttl_seconds=300)
        
        # Register service
        register_cache_service(service_name, policy)
        
        # Verify cache was created
        cache = self.cache_manager.get_service_cache(service_name)
        assert cache is not None
        assert cache.policy.max_size == 50
        assert cache.policy.ttl_seconds == 300
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        service_name = "test_service"
        register_cache_service(service_name)
        
        # Test set and get
        key = "test_key"
        value = "test_value"
        
        set_cached_value(key, value, service_name)
        retrieved_value = get_cached_value(key, service_name)
        
        assert retrieved_value == value
    
    def test_cache_invalidation(self):
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
    
    def test_cache_metrics(self):
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
        assert service_name in metrics
        service_metrics = metrics[service_name]
        
        assert service_metrics.hits >= 1
        assert service_metrics.misses >= 1
    
    def test_cross_service_cache_sharing(self):
        """Test cross-service cache sharing capabilities."""
        service1_name = "service1"
        service2_name = "service2"
        
        register_cache_service(service1_name)
        register_cache_service(service2_name)
        
        # Set value from service1
        key = "shared_key"
        value = "shared_value"
        set_cached_value(key, value, service1_name)
        
        # Retrieve from service2 (should be separate caches)
        retrieved_value = get_cached_value(key, service2_name)
        assert retrieved_value is None
        
        # But service1 should still have it
        retrieved_value = get_cached_value(key, service1_name)
        assert retrieved_value == value
    
    def test_tag_based_invalidation(self):
        """Test tag-based cache invalidation."""
        service_name = "test_service_tag"
        register_cache_service(service_name)
        
        # Set values with tags
        key1 = "key1"
        key2 = "key2"
        value1 = "value1"
        value2 = "value2"
        
        set_cached_value(key1, value1, service_name)
        set_cached_value(key2, value2, service_name)
        
        # Add tags
        from core.cache_coordinator import _cache_coordinator
        coordinator = _cache_coordinator
        coordinator.add_tags(key1, {"tag1", "common"}, service_name)
        coordinator.add_tags(key2, {"tag2", "common"}, service_name)
        
        # Verify tags were added by checking via cache manager directly
        cache_manager = get_cache_manager()
        cache = cache_manager.get_service_cache(service_name)
        assert cache is not None
        
        # Test initial state
        assert get_cached_value(key1, service_name) == value1
        assert get_cached_value(key2, service_name) == value2
        
        # Invalidate by specific tag
        invalidate_by_tag("tag1", service_name)
        
        # key1 should be gone, key2 should remain
        assert get_cached_value(key1, service_name) is None
        assert get_cached_value(key2, service_name) == value2
        
        # Invalidate by common tag
        set_cached_value(key1, value1, service_name)  # Restore key1
        coordinator.add_tags(key1, {"tag1", "common"}, service_name)  # Re-add tags
        invalidate_by_tag("common", service_name)
        
        # Both should be gone
        assert get_cached_value(key1, service_name) is None
        assert get_cached_value(key2, service_name) is None
    
    def test_cache_policy_application(self):
        """Test cache policy application."""
        service_name = "test_service"
        default_policy = CachePolicy(max_size=100, ttl_seconds=600)
        
        register_cache_service(service_name, default_policy)
        
        # Verify default policy
        cache = self.cache_manager.get_service_cache(service_name)
        assert cache.policy.max_size == 100
        assert cache.policy.ttl_seconds == 600
        
        # Update policy
        new_policy = CachePolicy(max_size=50, ttl_seconds=300)
        from core.cache_coordinator import _cache_coordinator
        coordinator = _cache_coordinator
        coordinator._cache_manager.apply_cache_policy(service_name, new_policy)
        
        # Verify updated policy
        cache = self.cache_manager.get_service_cache(service_name)
        assert cache.policy.max_size == 50
        assert cache.policy.ttl_seconds == 300
    
    def test_concurrent_cache_access(self):
        """Test concurrent cache access from multiple threads."""
        service_name = "concurrent_service"
        register_cache_service(service_name)
        
        def worker(worker_id: int, results: List[Any]):
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
        results = []
        
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
    
    def test_cache_expiration(self):
        """Test cache entry expiration."""
        service_name = "expiring_service"
        policy = CachePolicy(max_size=10, ttl_seconds=1)  # 1 second TTL
        register_cache_service(service_name, policy)
        
        key = "expiring_key"
        value = "expiring_value"
        
        # Set value with short TTL
        set_cached_value(key, value, service_name, ttl=1)
        
        # Should be available immediately
        retrieved = get_cached_value(key, service_name)
        assert retrieved == value
        
        # Wait for expiration (background cleanup runs every 1 second, so wait a bit longer)
        time.sleep(3)
        
        # Should be expired
        retrieved = get_cached_value(key, service_name)
        assert retrieved is None
    
    def test_cache_eviction_policies(self):
        """Test different cache eviction policies."""
        # Test LRU eviction
        service_name = "lru_service"
        policy = CachePolicy(max_size=3, eviction_strategy="LRU")
        register_cache_service(service_name, policy)
        
        # Fill cache to capacity
        for i in range(3):
            set_cached_value(f"key_{i}", f"value_{i}", service_name)
        
        # Access first key to make it recently used
        get_cached_value("key_0", service_name)
        
        # Add new key - should evict LRU (key_1)
        set_cached_value("key_3", "value_3", service_name)
        
        # key_0 should still exist (recently used)
        assert get_cached_value("key_0", service_name) == "value_0"
        
        # key_1 should be evicted
        assert get_cached_value("key_1", service_name) is None
        
        # key_2 and key_3 should exist
        assert get_cached_value("key_2", service_name) == "value_2"
        assert get_cached_value("key_3", service_name) == "value_3"
    
    def test_invalidation_event_handling(self):
        """Test cache invalidation event handling."""
        service_name = "event_service"
        register_cache_service(service_name)
        
        events_received = []
        
        def event_handler(event):
            """Test event handler."""
            events_received.append(event)
        
        # Subscribe to events
        from core.cache_coordinator import _cache_coordinator
        coordinator = _cache_coordinator
        coordinator.subscribe_to_events(service_name, event_handler)
        
        # Trigger invalidation
        key = "test_key"
        value = "test_value"
        set_cached_value(key, value, service_name)
        invalidate_cache_key(key, service_name)
        
        # Verify event was received
        assert len(events_received) >= 1
        event = events_received[0]
        assert isinstance(event, CacheInvalidationEvent)
        assert event.cache_key == key
        assert event.service_origin == service_name
    
    def test_cross_service_stats(self):
        """Test cross-service cache statistics."""
        service1_name = "stats_service1"
        service2_name = "stats_service2"
        
        register_cache_service(service1_name)
        register_cache_service(service2_name)
        
        # Perform operations on both services
        set_cached_value("key1", "value1", service1_name)
        set_cached_value("key2", "value2", service2_name)
        
        get_cached_value("key1", service1_name)
        get_cached_value("key2", service2_name)
        
        # Get cross-service stats
        from core.cache_coordinator import _cache_coordinator
        coordinator = _cache_coordinator
        stats = coordinator.get_cross_service_stats()
        
        assert service1_name in stats
        assert service2_name in stats
        assert stats[service1_name]["total_entries"] >= 1
        assert stats[service2_name]["total_entries"] >= 1
    
    def test_mock_service_integration(self):
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
