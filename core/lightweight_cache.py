# core/lightweight_cache.py
"""
Ultra-simple per-service in-memory cache.

Provides a minimal API used across SAGA without coordination, tags, or metrics.
"""

from typing import Any

_SERVICE_CACHES: dict[str, dict[str, Any]] = {}


def _ensure_service(service_name: str) -> dict[str, Any]:
    cache = _SERVICE_CACHES.get(service_name)
    if cache is None:
        cache = {}
        _SERVICE_CACHES[service_name] = cache
    return cache


def get_cached_value(key: str, service_name: str) -> Any | None:
    """Return cached value or None if not found."""
    return _ensure_service(service_name).get(key)


def set_cached_value(key: str, value: Any, service_name: str, ttl: int | None = None) -> None:
    """Set cached value (ttl ignored)."""
    _ensure_service(service_name)[key] = value


def invalidate_cache_key(key: str, service_name: str) -> None:
    """Delete a key if present."""
    _ensure_service(service_name).pop(key, None)


def get_cache_metrics(service_name: str | None = None) -> dict[str, Any]:
    """Return minimal size info only (compatibility)."""
    if service_name:
        return {"size": len(_ensure_service(service_name))}
    return {"total_services": len(_SERVICE_CACHES)}


def get_cache_size(service_name: str) -> int:
    """Return number of entries for the service."""
    return len(_ensure_service(service_name))


def register_cache_service(service_name: str) -> None:
    """No-op: services are created on demand."""
    _ensure_service(service_name)


def clear_service_cache(service_name: str) -> None:
    """Clear all cache entries for a service."""
    _SERVICE_CACHES[service_name] = {}
