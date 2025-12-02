# tests/test_lightweight_cache_basic.py
#!/usr/bin/env python3
"""
Basic test to verify lightweight cache functionality.
"""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.lightweight_cache import (
    get_cache_metrics,
    get_cached_value,
    invalidate_cache_key,
    register_cache_service,
    set_cached_value,
)


def test_basic_cache_operations() -> None:
    """Test basic cache operations."""
    print("Testing basic cache operations...")

    # Register a service
    service_name = "test_service"
    register_cache_service(service_name)

    # Test set and get
    key = "test_key"
    value = "test_value"

    set_cached_value(key, value, service_name)
    retrieved_value = get_cached_value(key, service_name)

    assert retrieved_value == value, f"Expected {value}, got {retrieved_value}"
    print("✓ Set/Get test passed")

    # Test cache invalidation
    invalidate_cache_key(key, service_name)
    retrieved_value = get_cached_value(key, service_name)

    assert retrieved_value is None, f"Expected None, got {retrieved_value}"
    print("✓ Invalidation test passed")

    # Test metrics
    metrics = get_cache_metrics(service_name)
    assert isinstance(metrics, dict), f"Expected dict, got {type(metrics)}"
    print("✓ Metrics test passed")

    print("All basic tests passed!")


if __name__ == "__main__":
    test_basic_cache_operations()
