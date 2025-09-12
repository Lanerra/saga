# core/cache_manager.py
"""
Unified cache management interface for the SAGA system.

This module provides the main cache management interface that coordinates
all cache operations across services and implements the unified caching strategy.
"""

import asyncio
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime

from core.cache_entry import CacheEntry, CacheMetrics, CachePolicy
from core.cache_policies import get_policy


class UnifiedCache:
    """
    Unified cache implementation with policy-based configuration.
    
    Provides a consistent cache interface that can be used across all services
    while implementing coordinated caching policies and metrics.
    """
    
    def __init__(self, service_name: str, policy: Optional[CachePolicy] = None):
        """
        Initialize unified cache.
        
        Args:
            service_name: Name of the service this cache belongs to
            policy: Cache policy to use (defaults to service-specific policy)
        """
        self.service_name = service_name
        self.policy = policy or get_policy(service_name)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._metrics = CacheMetrics()
        self._lock = threading.RLock()
        self._invalidation_callbacks: List[Callable] = []
        self._tags_index: Dict[str, Set[str]] = {}  # tag -> set of cache keys
        
        # Start background cleanup task if TTL is configured
        if self.policy.ttl_seconds:
            self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background task for cleaning up expired entries."""
        def cleanup_expired_entries():
            import structlog
            logger = structlog.get_logger(__name__)
            logger.debug("Starting cache cleanup thread")
            while True:
                time.sleep(1)  # Check every second for testing
                logger.debug("Running cache cleanup")
                self._cleanup_expired()
                logger.debug("Cache cleanup completed")
        
        cleanup_thread = threading.Thread(target=cleanup_expired_entries, daemon=True)
        cleanup_thread.start()
        # Store reference to thread for debugging
        self._cleanup_thread = cleanup_thread
    
    def _cleanup_expired(self):
        """Remove expired cache entries."""
        expired_keys = []
        with self._lock:
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key, reason="ttl_expired")
    
    def _evict_lru_entry(self):
        """Evict the least recently used entry."""
        if self._cache:
            lru_key = next(iter(self._cache))
            self._remove_entry(lru_key, reason="eviction")
            self._metrics.evictions += 1
    
    def _evict_lfu_entry(self):
        """Evict the least frequently used entry."""
        if self._cache:
            lfu_key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
            self._remove_entry(lfu_key, reason="eviction")
            self._metrics.evictions += 1
    
    def _evict_fifo_entry(self):
        """Evict the first in, first out entry."""
        if self._cache:
            fifo_key = next(iter(self._cache))
            self._remove_entry(fifo_key, reason="eviction")
            self._metrics.evictions += 1
    
    def _evict_entry_by_policy(self):
        """Evict an entry based on the configured eviction strategy."""
        if self.policy.eviction_strategy == "LRU":
            self._evict_lru_entry()
        elif self.policy.eviction_strategy == "LFU":
            self._evict_lfu_entry()
        elif self.policy.eviction_strategy == "FIFO":
            self._evict_fifo_entry()
    
    def _update_metrics_for_get(self, hit: bool):
        """Update metrics for cache get operations."""
        with self._lock:
            if hit:
                self._metrics.hits += 1
            else:
                self._metrics.misses += 1
            self._metrics.update_hit_rate()
    
    def _update_metrics_for_put(self, entry_size: int):
        """Update metrics for cache put operations."""
        with self._lock:
            self._metrics.total_entries = len(self._cache)
            self._metrics.memory_usage_bytes += entry_size
            self._metrics.update_average_entry_size()
    
    def _update_metrics_for_remove(self, entry_size: int):
        """Update metrics for cache remove operations."""
        with self._lock:
            self._metrics.total_entries = len(self._cache)
            self._metrics.memory_usage_bytes -= entry_size
            if self._metrics.memory_usage_bytes < 0:
                self._metrics.memory_usage_bytes = 0
            self._metrics.update_average_entry_size()
    
    def _remove_entry(self, key: str, reason: str = "manual"):
        """Remove an entry and update indices."""
        if key in self._cache:
            entry = self._cache.pop(key)
            
            # Remove from tags index
            for tag in entry.tags:
                if tag in self._tags_index and key in self._tags_index[tag]:
                    self._tags_index[tag].remove(key)
                    if not self._tags_index[tag]:
                        del self._tags_index[tag]
            
            # Update metrics
            # Handle numpy arrays and other complex types
            try:
                entry_size = len(str(entry.value)) if entry.value is not None else 0
            except (ValueError, TypeError):
                # For numpy arrays and other complex types, estimate size
                try:
                    import numpy as np
                    if isinstance(entry.value, np.ndarray):
                        entry_size = entry.value.nbytes
                    else:
                        entry_size = len(repr(entry.value)) if entry.value is not None else 0
                except ImportError:
                    entry_size = len(repr(entry.value)) if entry.value is not None else 0
            self._update_metrics_for_remove(entry_size)
            
            # Call invalidation callbacks
            for callback in self._invalidation_callbacks:
                try:
                    callback(key, reason)
                except Exception:
                    pass  # Don't let callback errors break cache operations
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check if expired
                if entry.is_expired():
                    self._remove_entry(key, reason="ttl_expired")
                    self._update_metrics_for_get(hit=False)
                    return None
                
                # Update access tracking
                entry.update_access()
                
                # Move to end for LRU
                self._cache.move_to_end(key)
                
                self._update_metrics_for_get(hit=True)
                return entry.value
            else:
                self._update_metrics_for_get(hit=False)
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Put a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (overrides policy TTL)
        """
        with self._lock:
            # Check if cache is at capacity
            if len(self._cache) >= self.policy.max_size:
                self._evict_entry_by_policy()
            
            # Determine TTL
            effective_ttl = ttl if ttl is not None else self.policy.ttl_seconds
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                ttl=effective_ttl,
                access_count=1,
                service_origin=self.service_name
            )
            
            # Store entry
            self._cache[key] = entry
            self._cache.move_to_end(key)  # Move to end for LRU tracking
            
            # Update metrics
            # Handle numpy arrays and other complex types
            try:
                entry_size = len(str(value)) if value is not None else 0
            except (ValueError, TypeError):
                # For numpy arrays and other complex types, estimate size
                try:
                    import numpy as np
                    if isinstance(value, np.ndarray):
                        entry_size = value.nbytes
                    else:
                        entry_size = len(repr(value)) if value is not None else 0
                except ImportError:
                    entry_size = len(repr(value)) if value is not None else 0
            self._update_metrics_for_put(entry_size)
    
    def remove(self, key: str) -> None:
        """
        Remove a key from cache.
        
        Args:
            key: Cache key to remove
        """
        with self._lock:
            self._remove_entry(key, reason="manual")
    
    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            keys_to_remove = list(self._cache.keys())
            for key in keys_to_remove:
                self._remove_entry(key, reason="clear")
    
    def get_metrics(self) -> CacheMetrics:
        """
        Get cache metrics.
        
        Returns:
            Current cache metrics
        """
        with self._lock:
            self._metrics.update_cache_age()
            return CacheMetrics(
                hits=self._metrics.hits,
                misses=self._metrics.misses,
                evictions=self._metrics.evictions,
                total_entries=len(self._cache),
                memory_usage_bytes=self._metrics.memory_usage_bytes,
                hit_rate=self._metrics.hit_rate,
                average_entry_size_bytes=self._metrics.average_entry_size_bytes,
                cache_age_seconds=self._metrics.cache_age_seconds
            )
    
    def set_invalidation_callback(self, callback: Callable) -> None:
        """
        Set callback for cache invalidation events.
        
        Args:
            callback: Function to call when entries are invalidated
        """
        with self._lock:
            self._invalidation_callbacks.append(callback)
    
    def add_tags(self, key: str, tags: Set[str]) -> None:
        """
        Add tags to a cache entry for grouping and bulk operations.
        
        Args:
            key: Cache key
            tags: Tags to add
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry.tags.update(tags)
                
                # Update tags index
                for tag in tags:
                    if tag not in self._tags_index:
                        self._tags_index[tag] = set()
                    self._tags_index[tag].add(key)
    
    def get_by_tag(self, tag: str) -> Dict[str, Any]:
        """
        Get all cache entries with a specific tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            Dictionary of key-value pairs for entries with the tag
        """
        with self._lock:
            if tag not in self._tags_index:
                return {}
            
            result = {}
            keys_to_remove = []
            
            for key in self._tags_index[tag]:
                if key in self._cache:
                    entry = self._cache[key]
                    if not entry.is_expired():
                        result[key] = entry.value
                    else:
                        keys_to_remove.append(key)
            
            # Clean up expired entries from tag index
            for key in keys_to_remove:
                self._tags_index[tag].remove(key)
                if not self._tags_index[tag]:
                    del self._tags_index[tag]
            
            return result
    
    def invalidate_by_tag(self, tag: str) -> None:
        """
        Invalidate all cache entries with a specific tag.
        
        Args:
            tag: Tag to invalidate
        """
        with self._lock:
            if tag in self._tags_index:
                keys_to_invalidate = list(self._tags_index[tag])
                for key in keys_to_invalidate:
                    self._remove_entry(key, reason="tag_invalidation")
    
    def get_cache_size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)
    
    def get_cache_keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())


class CacheManager:
    """
    Main cache manager that coordinates all cache operations across services.
    
    Provides a unified interface for managing caches across all services
    and handles cross-service cache coordination.
    """
    
    def __init__(self):
        """Initialize cache manager."""
        self._caches: Dict[str, UnifiedCache] = {}
        self._lock = threading.RLock()
        self._event_callbacks: List[Callable] = []
    
    def create_cache(self, service_name: str, policy: Optional[CachePolicy] = None) -> UnifiedCache:
        """
        Create a cache for a service.
        
        Args:
            service_name: Name of the service
            policy: Cache policy to use
            
        Returns:
            Unified cache instance
        """
        with self._lock:
            if service_name not in self._caches:
                cache = UnifiedCache(service_name, policy)
                self._caches[service_name] = cache
            elif policy is not None:
                # Update existing cache policy if new policy provided
                self._caches[service_name].policy = policy
            return self._caches[service_name]
    
    def remove_cache(self, service_name: str) -> bool:
        """
        Remove a cache for a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            True if cache was removed, False if it didn't exist
        """
        with self._lock:
            if service_name in self._caches:
                cache = self._caches.pop(service_name)
                cache.clear()
                return True
            return False
    
    def get_service_cache(self, service_name: str) -> Optional[UnifiedCache]:
        """
        Get cache for a specific service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Cache for the service or None if not found
        """
        with self._lock:
            return self._caches.get(service_name)
    
    def apply_cache_policy(self, service_name: str, policy: CachePolicy) -> None:
        """
        Apply a cache policy to a service's cache.
        
        Args:
            service_name: Name of the service
            policy: Cache policy to apply
        """
        with self._lock:
            if service_name in self._caches:
                self._caches[service_name].policy = policy
    
    def get_cross_service_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get cache statistics across all services.
        
        Returns:
            Dictionary of service names to their cache statistics
        """
        with self._lock:
            stats = {}
            for service_name, cache in self._caches.items():
                metrics = cache.get_metrics()
                stats[service_name] = {
                    "hits": metrics.hits,
                    "misses": metrics.misses,
                    "hit_rate": metrics.hit_rate,
                    "total_entries": metrics.total_entries,
                    "evictions": metrics.evictions,
                    "memory_usage_bytes": metrics.memory_usage_bytes
                }
            return stats
    
    def clear_all_caches(self) -> None:
        """Clear all caches across all services."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
    
    def get_total_memory_usage(self) -> int:
        """Get total memory usage across all caches."""
        with self._lock:
            return sum(cache.get_metrics().memory_usage_bytes for cache in self._caches.values())
    
    def get_total_entries(self) -> int:
        """Get total number of entries across all caches."""
        with self._lock:
            return sum(cache.get_cache_size() for cache in self._caches.values())


# Global cache manager instance
_cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    return _cache_manager


def create_cache(service_name: str, policy: Optional[CachePolicy] = None) -> UnifiedCache:
    """Create a cache for a service."""
    return _cache_manager.create_cache(service_name, policy)


def get_service_cache(service_name: str) -> Optional[UnifiedCache]:
    """Get cache for a specific service."""
    return _cache_manager.get_service_cache(service_name)
