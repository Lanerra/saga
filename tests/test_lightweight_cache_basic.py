# tests/test_lightweight_cache_basic.py
#!/usr/bin/env python3
"""
Basic test to verify lightweight cache functionality.
"""

import os
import sys
import threading

import pytest

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.lightweight_cache as lightweight_cache_module
from core.lightweight_cache import (
    get_cache_metrics,
    get_cache_size,
    get_cached_value,
    invalidate_cache_key,
    register_cache_service,
    set_cached_value,
)


def test_basic_cache_operations() -> None:
    """Test basic cache operations."""
    service_name = "test_service"
    register_cache_service(service_name)

    key = "test_key"
    value = "test_value"

    set_cached_value(key, value, service_name)
    retrieved_value = get_cached_value(key, service_name)
    assert retrieved_value == value

    invalidate_cache_key(key, service_name)
    retrieved_value = get_cached_value(key, service_name)
    assert retrieved_value is None

    metrics = get_cache_metrics(service_name)
    assert isinstance(metrics, dict)


def test_cache_ttl_expiry_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    service_name = "test_service_ttl"

    # Control time precisely so we don't rely on sleeps.
    now = {"t": 100.0}

    def fake_now() -> float:
        return now["t"]

    monkeypatch.setattr(lightweight_cache_module, "_now", fake_now)

    register_cache_service(service_name, maxsize=8)

    set_cached_value("k", "v", service_name, ttl=5.0)
    assert get_cached_value("k", service_name) == "v"
    assert get_cache_size(service_name) == 1

    # Before expiration
    now["t"] = 104.999
    assert get_cached_value("k", service_name) == "v"
    assert get_cache_size(service_name) == 1

    # After expiration: key should be treated as absent and purged on access/size.
    now["t"] = 105.001
    assert get_cached_value("k", service_name) is None
    assert get_cache_size(service_name) == 0


def test_cache_bounded_growth_lru_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    service_name = "test_service_bounded"

    # Avoid any chance of TTL-based purging affecting this test.
    monkeypatch.setattr(lightweight_cache_module, "_now", lambda: 1000.0)

    register_cache_service(service_name, maxsize=2)

    set_cached_value("a", "A", service_name)
    set_cached_value("b", "B", service_name)
    assert get_cached_value("a", service_name) == "A"
    assert get_cached_value("b", service_name) == "B"
    assert get_cache_size(service_name) == 2

    # Touch "a" so that "b" becomes the LRU.
    assert get_cached_value("a", service_name) == "A"

    # Adding a third item should evict the LRU ("b").
    set_cached_value("c", "C", service_name)
    assert get_cache_size(service_name) == 2

    assert get_cached_value("a", service_name) == "A"
    assert get_cached_value("b", service_name) is None
    assert get_cached_value("c", service_name) == "C"


def test_cache_thread_safety_sanity(monkeypatch: pytest.MonkeyPatch) -> None:
    service_name = "test_service_threads"

    monkeypatch.setattr(lightweight_cache_module, "_now", lambda: 2000.0)
    register_cache_service(service_name, maxsize=64)

    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(200):
                key = f"k{(n * 1000) + i}"
                set_cached_value(key, i, service_name, ttl=60.0)
                _ = get_cached_value(key, service_name)
        except BaseException as e:  # pragma: no cover - should not happen
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


if __name__ == "__main__":
    test_basic_cache_operations()
