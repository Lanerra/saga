# core/cache_coordinator.py
"""
Main cache coordinator for the unified cache coordination system.

This module provides the central coordination service that manages
cache operations across all services, handles invalidation events,
and enables cross-service cache sharing.
"""

import asyncio
import threading
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime

from core.cache_entry import CacheEntry, CacheInvalidationEvent, CacheMetrics
from core.cache_manager import CacheManager, get_cache_manager
from core.cache_policies import get_policy_manager, CachePolicy


class CacheCoordinator:
    """
    Central cache coordinator that manages cache operations across services.
    
    Coordinates cache operations, handles invalidation events, and enables
    cross-service cache sharing and consistency.
    """
    
    def __init__(self):
        """Initialize cache coordinator."""
        self._cache_manager = get_cache_manager()
        self._policy_manager = get_policy_manager()
        self._event_subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        self._invalidation_lock = threading.RLock()
        self._service_registration_callbacks: List[Callable] = []
    
    def register_service(self, service_name: str, policy: Optional[CachePolicy] = None) -> None:
        """
        Register a service with the cache coordinator.
        
        Args:
            service_name: Name of the service to register
            policy: Cache policy for the service (optional)
        """
        with self._lock:
            # Create cache for the service
            self._cache_manager.create_cache(service_name, policy)
            
            # Initialize event subscribers list for this service
            if service_name not in self._event_subscribers:
                self._event_subscribers[service_name] = []
            
            # Notify registration callbacks
            for callback in self._service_registration_callbacks:
                try:
                    callback(service_name)
                except Exception:
                    pass
            
            # Log service registration
            from structlog import get_logger
            logger = get_logger(__name__)
            logger.info(f"Service registered with cache coordinator: {service_name}")
    
    def get(self, key: str, service_name: str) -> Optional[Any]:
        """
        Get a cached value for a service.
        
        Args:
            key: Cache key to retrieve
            service_name: Name of the requesting service
            
        Returns:
            Cached value or None if not found
        """
        cache = self._cache_manager.get_service_cache(service_name)
        if cache:
            return cache.get(key)
        return None
    
    def set(self, key: str, value: Any, service_name: str, ttl: Optional[int] = None) -> None:
        """
        Set a cached value for a service.
        
        Args:
            key: Cache key
            value: Value to cache
            service_name: Name of the requesting service
            ttl: Time to live in seconds (optional)
        """
        cache = self._cache_manager.get_service_cache(service_name)
        if cache:
            cache.put(key, value, ttl)
    
    def invalidate(self, key: str, service_name: str, cascade: bool = True) -> None:
        """
        Invalidate a cache entry.
        
        Args:
            key: Cache key to invalidate
            service_name: Name of the service requesting invalidation
            cascade: Whether to cascade invalidation to dependencies
        """
        with self._invalidation_lock:
            cache = self._cache_manager.get_service_cache(service_name)
            if cache:
                cache.remove(key)
                
                # Create invalidation event
                event = CacheInvalidationEvent(
                    cache_key=key,
                    service_origin=service_name,
                    reason="manual",
                    cascade_to_dependencies=cascade
                )
                
                # Broadcast invalidation event
                self.handle_invalidation_event(event)
    
    def get_metrics(self, service_name: Optional[str] = None) -> Dict[str, CacheMetrics]:
        """
        Get cache metrics for services.
        
        Args:
            service_name: Specific service name (optional, if None returns all)
            
        Returns:
            Dictionary of service names to their cache metrics
        """
        if service_name:
            cache = self._cache_manager.get_service_cache(service_name)
            if cache:
                return {service_name: cache.get_metrics()}
            return {}
        else:
            # Return metrics for all services
            metrics = {}
            for svc_name in self._cache_manager._caches.keys():
                cache = self._cache_manager.get_service_cache(svc_name)
                if cache:
                    metrics[svc_name] = cache.get_metrics()
            return metrics
    
    def handle_invalidation_event(self, event: CacheInvalidationEvent) -> None:
        """
        Handle a cache invalidation event.
        
        Args:
            event: Cache invalidation event to process
        """
        with self._invalidation_lock:
            # Notify subscribers for the originating service
            if event.service_origin in self._event_subscribers:
                for callback in self._event_subscribers[event.service_origin]:
                    try:
                        callback(event)
                    except Exception:
                        pass
            
            # Cascade invalidation if requested
            if event.cascade_to_dependencies and event.cache_key:
                self._cascade_invalidation(event)
            
            # Broadcast to all services if it's a cross-service event
            if event.tags:
                self._broadcast_tag_invalidation(event)
    
    def _cascade_invalidation(self, event: CacheInvalidationEvent) -> None:
        """
        Cascade invalidation to dependent cache entries.
        
        Args:
            event: Cache invalidation event that triggered cascading
        """
        # This would implement dependency-based invalidation
        # For now, we'll just log that cascading would occur
        from structlog import get_logger
        logger = get_logger(__name__)
        logger.debug(f"Cascading invalidation for key: {event.cache_key}")
    
    def _broadcast_tag_invalidation(self, event: CacheInvalidationEvent) -> None:
        """
        Broadcast tag-based invalidation to all services.
        
        Args:
            event: Cache invalidation event with tags
        """
        if event.tags:
            for service_name, cache in self._cache_manager._caches.items():
                if service_name != event.service_origin:  # Don't invalidate originating service again
                    for tag in event.tags:
                        cache.invalidate_by_tag(tag)
    
    def sync_shared_entries(self) -> None:
        """Synchronize shared cache entries across services."""
        # This would implement cross-service cache sharing
        # For Phase 1, we'll just log that synchronization would occur
        from structlog import get_logger
        logger = get_logger(__name__)
        logger.debug("Synchronizing shared cache entries across services")
    
    def subscribe_to_events(self, service_name: str, callback: Callable) -> None:
        """
        Subscribe to cache events for a service.
        
        Args:
            service_name: Name of the service subscribing
            callback: Callback function to receive events
        """
        with self._lock:
            if service_name not in self._event_subscribers:
                self._event_subscribers[service_name] = []
            self._event_subscribers[service_name].append(callback)
    
    def add_service_registration_callback(self, callback: Callable) -> None:
        """
        Add a callback for service registration events.
        
        Args:
            callback: Callback function to receive service registration events
        """
        with self._lock:
            self._service_registration_callbacks.append(callback)
    
    def get_cross_service_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cross-service cache statistics.
        
        Returns:
            Dictionary of cross-service cache statistics
        """
        return self._cache_manager.get_cross_service_cache_stats()
    
    def clear_service_cache(self, service_name: str) -> None:
        """
        Clear all cache entries for a specific service.
        
        Args:
            service_name: Name of the service whose cache to clear
        """
        cache = self._cache_manager.get_service_cache(service_name)
        if cache:
            cache.clear()
    
    def invalidate_by_tag(self, tag: str, service_name: Optional[str] = None) -> None:
        """
        Invalidate cache entries by tag.
        
        Args:
            tag: Tag to invalidate
            service_name: Specific service name (optional, if None invalidates across all)
        """
        if service_name:
            cache = self._cache_manager.get_service_cache(service_name)
            if cache:
                cache.invalidate_by_tag(tag)
        else:
            # Invalidate across all services
            for cache in self._cache_manager._caches.values():
                cache.invalidate_by_tag(tag)
    
    def add_tags(self, key: str, tags: Set[str], service_name: str) -> None:
        """
        Add tags to a cache entry.
        
        Args:
            key: Cache key
            tags: Tags to add
            service_name: Name of the service
        """
        cache = self._cache_manager.get_service_cache(service_name)
        if cache:
            cache.add_tags(key, tags)
    
    def get_by_tag(self, tag: str, service_name: str) -> Dict[str, Any]:
        """
        Get cache entries by tag.
        
        Args:
            tag: Tag to search for
            service_name: Name of the service
            
        Returns:
            Dictionary of key-value pairs for entries with the tag
        """
        cache = self._cache_manager.get_service_cache(service_name)
        if cache:
            return cache.get_by_tag(tag)
        return {}


# Global cache coordinator instance
_cache_coordinator = CacheCoordinator()


def get_cache_coordinator() -> CacheCoordinator:
    """Get the global cache coordinator instance."""
    return _cache_coordinator


def initialize_cache_coordinator() -> CacheCoordinator:
    """Initialize and return the cache coordinator."""
    return _cache_coordinator


def register_cache_service(service_name: str, cache_config: Optional[CachePolicy] = None) -> None:
    """
    Register a cache service with the coordinator.
    
    Args:
        service_name: Name of the service to register
        cache_config: Cache policy configuration (optional)
    """
    _cache_coordinator.register_service(service_name, cache_config)


def get_cached_value(key: str, service_name: str) -> Optional[Any]:
    """
    Get a cached value from a service's cache.
    
    Args:
        key: Cache key to retrieve
        service_name: Name of the service
        
    Returns:
        Cached value or None if not found
    """
    return _cache_coordinator.get(key, service_name)


def set_cached_value(key: str, value: Any, service_name: str, ttl: Optional[int] = None) -> None:
    """
    Set a cached value in a service's cache.
    
    Args:
        key: Cache key
        value: Value to cache
        service_name: Name of the service
        ttl: Time to live in seconds (optional)
    """
    _cache_coordinator.set(key, value, service_name, ttl)


def invalidate_cache_key(key: str, service_name: str, cascade: bool = True) -> None:
    """
    Invalidate a cache key.
    
    Args:
        key: Cache key to invalidate
        service_name: Name of the service
        cascade: Whether to cascade invalidation to dependencies
    """
    _cache_coordinator.invalidate(key, service_name, cascade)


def invalidate_by_tag(tag: str, service_name: Optional[str] = None) -> None:
    """
    Invalidate cache entries by tag.
    
    Args:
        tag: Tag to invalidate
        service_name: Specific service name (optional)
    """
    _cache_coordinator.invalidate_by_tag(tag, service_name)


def get_cache_metrics(service_name: Optional[str] = None) -> Dict[str, CacheMetrics]:
    """
    Get cache metrics.
    
    Args:
        service_name: Specific service name (optional)
        
    Returns:
        Dictionary of service names to their cache metrics
    """
    return _cache_coordinator.get_metrics(service_name)


def clear_service_cache(service_name: str) -> None:
    """
    Clear all cache entries for a service.
    
    Args:
        service_name: Name of the service whose cache to clear
    """
    _cache_coordinator.clear_service_cache(service_name)


def broadcast_cache_invalidation(event: CacheInvalidationEvent) -> None:
    """
    Broadcast a cache invalidation event.
    
    Args:
        event: Cache invalidation event to broadcast
    """
    _cache_coordinator.handle_invalidation_event(event)


def subscribe_to_cache_events(service_name: str, callback: Callable) -> None:
    """
    Subscribe to cache events.
    
    Args:
        service_name: Name of the service subscribing
        callback: Callback function to receive events
    """
    _cache_coordinator.subscribe_to_events(service_name, callback)
