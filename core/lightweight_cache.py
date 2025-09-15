# core/lightweight_cache.py
"""
Lightweight cache implementation using LRU cache decorators.

This module provides a simple, lightweight caching solution using functools.lru_cache
and async_lru decorators, replacing the heavyweight unified cache coordination system.
"""

import asyncio
import functools
from collections import defaultdict
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Service-specific caches using LRU
_service_caches = {}
_service_async_caches = {}
_cache_stats = defaultdict(lambda: defaultdict(int))
_service_tags = defaultdict(lambda: defaultdict(set))  # service -> key -> tags


def _get_sync_cache(service_name: str, maxsize: int = 128):
    """Get or create a sync LRU cache for a service."""
    if service_name not in _service_caches:

        @functools.lru_cache(maxsize=maxsize)
        def _cache():
            return {}

        _service_caches[service_name] = _cache
        # Initialize the cache
        _cache()
    return _service_caches[service_name]()


def _get_async_cache(service_name: str, maxsize: int = 128):
    """Get or create an async LRU cache for a service."""
    if service_name not in _service_async_caches:
        from async_lru import alru_cache

        @alru_cache(maxsize=maxsize)
        async def _cache():
            return {}

        _service_async_caches[service_name] = _cache
        # Initialize the cache
        asyncio.run(_cache())
    return _service_async_caches[service_name]()


def get_cached_value(key: str, service_name: str) -> Any | None:
    """
    Get cached value using LRU cache.

    Args:
        key: Cache key to retrieve
        service_name: Name of the requesting service

    Returns:
        Cached value or None if not found
    """
    try:
        cache = _get_sync_cache(service_name)
        value = cache.get(key)
        if value is not None:
            _cache_stats[service_name]["hits"] += 1
        else:
            _cache_stats[service_name]["misses"] += 1
        return value
    except Exception as e:
        logger.error(f"Error getting cached value: {e}")
        return None


def set_cached_value(
    key: str, value: Any, service_name: str, ttl: int | None = None
) -> None:
    """
    Set cached value using LRU cache.

    Args:
        key: Cache key
        value: Value to cache
        service_name: Name of the requesting service
        ttl: Time to live in seconds (ignored in LRU cache, kept for API compatibility)
    """
    try:
        cache = _get_sync_cache(service_name)
        cache[key] = value
        _cache_stats[service_name]["sets"] += 1
    except Exception as e:
        logger.error(f"Error setting cached value: {e}")


def invalidate_cache_key(key: str, service_name: str) -> None:
    """
    Invalidate cache key by deleting it from the cache.

    Args:
        key: Cache key to invalidate
        service_name: Name of the service
    """
    try:
        cache = _get_sync_cache(service_name)
        if key in cache:
            del cache[key]
            _cache_stats[service_name]["invalidations"] += 1

            # Remove tags for this key
            if key in _service_tags[service_name]:
                del _service_tags[service_name][key]
    except Exception as e:
        logger.error(f"Error invalidating cache key: {e}")


def get_cache_metrics(service_name: str | None = None) -> dict[str, Any]:
    """
    Get basic cache metrics.

    Args:
        service_name: Specific service name (optional)

    Returns:
        Dictionary of cache metrics
    """
    if service_name:
        return dict(_cache_stats[service_name])
    return {service: dict(stats) for service, stats in _cache_stats.items()}


def register_cache_service(service_name: str) -> None:
    """
    Register a cache service (simple initialization with LRU).

    Args:
        service_name: Name of the service to register
    """
    try:
        _get_sync_cache(service_name)
        logger.info(f"Cache service registered: {service_name}")
    except Exception as e:
        logger.error(f"Error registering cache service: {e}")


def invalidate_by_tag(tag: str, service_name: str | None = None) -> None:
    """
    Invalidate cache entries by tag.

    Args:
        tag: Tag to invalidate
        service_name: Specific service name (if None, invalidates across all services)
    """
    try:
        if service_name:
            # Invalidate for specific service
            keys_to_invalidate = []
            for key, tags in _service_tags[service_name].items():
                if tag in tags:
                    keys_to_invalidate.append(key)

            for key in keys_to_invalidate:
                invalidate_cache_key(key, service_name)
        else:
            # Invalidate across all services
            for svc_name in _service_tags.keys():
                keys_to_invalidate = []
                for key, tags in _service_tags[svc_name].items():
                    if tag in tags:
                        keys_to_invalidate.append(key)

                for key in keys_to_invalidate:
                    invalidate_cache_key(key, svc_name)
    except Exception as e:
        logger.error(f"Error invalidating by tag: {e}")


def add_tags(key: str, tags: set[str], service_name: str) -> None:
    """
    Add tags to a cache entry.

    Args:
        key: Cache key
        tags: Tags to add
        service_name: Name of the service
    """
    try:
        _service_tags[service_name][key].update(tags)
    except Exception as e:
        logger.error(f"Error adding tags: {e}")


def get_by_tag(tag: str, service_name: str) -> dict[str, Any]:
    """
    Get cache entries by tag.

    Args:
        tag: Tag to search for
        service_name: Name of the service

    Returns:
        Dictionary of key-value pairs for entries with the tag
    """
    try:
        result = {}
        cache = _get_sync_cache(service_name)

        for key, key_tags in _service_tags[service_name].items():
            if tag in key_tags and key in cache:
                result[key] = cache[key]

        return result
    except Exception as e:
        logger.error(f"Error getting by tag: {e}")
        return {}


def clear_service_cache(service_name: str) -> None:
    """
    Clear all cache entries for a specific service.

    Args:
        service_name: Name of the service whose cache to clear
    """
    try:
        if service_name in _service_caches:
            # Clear the cache by recreating it
            @functools.lru_cache(maxsize=128)
            def _new_cache():
                return {}

            _service_caches[service_name] = _new_cache
            _new_cache()  # Initialize

            # Clear stats and tags
            _cache_stats[service_name].clear()
            _service_tags[service_name].clear()

        logger.info(f"Cleared cache for service: {service_name}")
    except Exception as e:
        logger.error(f"Error clearing service cache: {e}")


# Backward compatibility functions that match the old API
def get_cache_coordinator():
    """Get cache coordinator (returns None for lightweight implementation)."""
    return None


def initialize_cache_coordinator():
    """Initialize cache coordinator (no-op for lightweight implementation)."""
    pass


def broadcast_cache_invalidation(event):
    """Broadcast cache invalidation (no-op for lightweight implementation)."""
    pass


def subscribe_to_cache_events(service_name: str, callback):
    """Subscribe to cache events (no-op for lightweight implementation)."""
    pass
