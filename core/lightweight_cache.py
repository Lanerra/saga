# core/lightweight_cache.py
"""
Ultra-simple per-service in-memory cache.

This cache is intended for small, process-local memoization within the `core`
package. It is *not* coordinated across processes and has no tag-based
invalidation.

Remediation (core audit item 12):
- TTL is enforced (no longer ignored).
- Caches are bounded by default (LRU eviction).
- Basic thread safety is provided via locks.

Public API is kept stable; optional parameters were added to allow per-service
configuration without forcing widespread call-site changes.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


def _now() -> float:
    """Time source for TTL evaluation (monotonic for correctness).

    This is a dedicated function to make TTL behavior easy to test via monkeypatch.
    """
    return time.monotonic()


DEFAULT_MAXSIZE: int = 1024
"""Default per-service cache bound to prevent unbounded growth."""


@dataclass(frozen=True)
class _CacheEntry:
    value: Any
    expires_at: float | None


class _ServiceCache:
    """Per-service cache with TTL + LRU eviction + a lock."""

    def __init__(self, *, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError(f"maxsize must be > 0, got {maxsize}")
        self.maxsize = maxsize
        self._lock = threading.Lock()
        self._data: OrderedDict[str, _CacheEntry] = OrderedDict()

    def _purge_expired_locked(self, now: float) -> None:
        if not self._data:
            return

        expired_keys: list[str] = []
        for key, entry in self._data.items():
            if entry.expires_at is not None and entry.expires_at <= now:
                expired_keys.append(key)

        for key in expired_keys:
            self._data.pop(key, None)

    def _evict_lru_locked(self) -> None:
        while len(self._data) > self.maxsize:
            # pop the least-recently-used key (front of OrderedDict)
            self._data.popitem(last=False)

    def get(self, key: str) -> Any | None:
        now = _now()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None

            if entry.expires_at is not None and entry.expires_at <= now:
                self._data.pop(key, None)
                return None

            # LRU: reading counts as use
            self._data.move_to_end(key)
            return entry.value

    def set(self, key: str, value: Any, *, ttl: float | None) -> None:
        now = _now()
        expires_at: float | None = None
        if ttl is not None:
            # Treat ttl as seconds. ttl <= 0 means "immediately expired".
            expires_at = now + ttl

        with self._lock:
            self._purge_expired_locked(now)
            self._data[key] = _CacheEntry(value=value, expires_at=expires_at)
            self._data.move_to_end(key)
            self._evict_lru_locked()

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def size(self) -> int:
        now = _now()
        with self._lock:
            self._purge_expired_locked(now)
            return len(self._data)


_SERVICE_CACHES: dict[str, _ServiceCache] = {}
_SERVICE_CACHES_LOCK = threading.Lock()


def _ensure_service(service_name: str, *, maxsize: int | None = None) -> _ServiceCache:
    with _SERVICE_CACHES_LOCK:
        cache = _SERVICE_CACHES.get(service_name)
        if cache is None:
            cache = _ServiceCache(maxsize=maxsize or DEFAULT_MAXSIZE)
            _SERVICE_CACHES[service_name] = cache
            return cache

        if maxsize is not None and maxsize != cache.maxsize:
            # Best-effort compatibility: allow configuration updates without
            # forcing a "stop the world" refactor.
            cache.maxsize = maxsize

        return cache


def get_cached_value(key: str, service_name: str) -> Any | None:
    """Return cached value or None if not found / expired."""
    return _ensure_service(service_name).get(key)


def set_cached_value(key: str, value: Any, service_name: str, ttl: float | None = None) -> None:
    """Set cached value with optional TTL (seconds).

    Args:
        ttl: Time-to-live in seconds. If None, the value does not expire by time.
    """
    _ensure_service(service_name).set(key, value, ttl=ttl)


def invalidate_cache_key(key: str, service_name: str) -> None:
    """Delete a key if present."""
    _ensure_service(service_name).invalidate(key)


def get_cache_metrics(service_name: str | None = None) -> dict[str, Any]:
    """Return minimal cache metrics.

    Note: this is intentionally light-weight to keep compatibility with existing
    consumers that only expect a dict.
    """
    if service_name:
        cache = _ensure_service(service_name)
        return {"size": cache.size(), "maxsize": cache.maxsize}

    # totals across services
    with _SERVICE_CACHES_LOCK:
        services = list(_SERVICE_CACHES.items())

    total_entries = sum(cache.size() for _, cache in services)
    return {"total_services": len(services), "total_entries": total_entries}


def get_cache_size(service_name: str) -> int:
    """Return number of non-expired entries for the service."""
    return _ensure_service(service_name).size()


def register_cache_service(service_name: str, maxsize: int | None = None) -> None:
    """Ensure a cache exists for a service (optional configuration).

    Args:
        maxsize: Per-service max entries. Defaults to `DEFAULT_MAXSIZE`.
    """
    _ensure_service(service_name, maxsize=maxsize)


def clear_service_cache(service_name: str) -> None:
    """Clear all cache entries for a service."""
    _ensure_service(service_name).clear()
