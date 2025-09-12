# core/cache_entry.py
"""
Cache entry and metrics models for the unified cache coordination system.

This module defines the core data structures used by the cache coordination system
to track cache entries, metrics, and policies across services.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Set
import time


@dataclass
class CacheEntry:
    """
    Represents a cached entry with metadata for coordination and invalidation.
    
    This class extends the basic cache entry concept to include cross-service
    coordination metadata like tags, dependencies, and service origin.
    """
    
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    ttl: Optional[int]  # Time to live in seconds
    access_count: int
    service_origin: str  # Which service created this entry
    tags: Set[str] = field(default_factory=set)  # For cross-service cache grouping
    dependencies: Set[str] = field(default_factory=set)  # Cache keys this entry depends on
    
    def __post_init__(self):
        """Initialize access count and timestamps."""
        if self.access_count is None:
            self.access_count = 1
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.last_accessed is None:
            self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired based on TTL."""
        if self.ttl is None:
            return False
        # Use UTC timestamp for consistency
        current_time = datetime.utcnow().timestamp()
        expiration_time = self.created_at.timestamp() + self.ttl
        is_expired = current_time > expiration_time
        import structlog
        logger = structlog.get_logger(__name__)
        logger.debug(f"Checking expiration - current: {current_time}, created: {self.created_at.timestamp()}, ttl: {self.ttl}, expiration: {expiration_time}, expired: {is_expired}")
        return is_expired
    
    def update_access(self):
        """Update access timestamp and increment access count."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


@dataclass
class CacheMetrics:
    """
    Comprehensive metrics for cache performance monitoring.
    
    Tracks hits, misses, evictions, and performance characteristics
    to enable optimization and monitoring of cache behavior.
    """
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0
    memory_usage_bytes: int = 0
    hit_rate: float = 0.0
    average_entry_size_bytes: int = 0
    cache_age_seconds: int = 0
    
    def update_hit_rate(self):
        """Calculate and update the cache hit rate."""
        total_requests = self.hits + self.misses
        self.hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0
    
    def update_average_entry_size(self):
        """Calculate and update the average entry size."""
        self.average_entry_size_bytes = (
            self.memory_usage_bytes // self.total_entries
        ) if self.total_entries > 0 else 0
    
    def update_cache_age(self):
        """Update the cache age in seconds."""
        self.cache_age_seconds = int(time.time())


@dataclass
class CachePolicy:
    """
    Configuration for cache behavior and coordination policies.
    
    Defines how caches should behave in terms of size, eviction, TTL,
    and cross-service sharing policies.
    """
    
    max_size: int = 1000
    ttl_seconds: Optional[int] = None
    eviction_strategy: str = "LRU"  # "LRU", "LFU", "FIFO"
    shared_across_services: bool = False
    replication_factor: int = 1  # For distributed caching
    
    def __post_init__(self):
        """Validate policy parameters."""
        if self.max_size <= 0:
            raise ValueError("max_size must be positive")
        if self.ttl_seconds is not None and self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive or None")
        if self.eviction_strategy not in ["LRU", "LFU", "FIFO"]:
            raise ValueError("eviction_strategy must be 'LRU', 'LFU', or 'FIFO'")
        if self.replication_factor < 1:
            raise ValueError("replication_factor must be at least 1")


@dataclass
class CacheInvalidationEvent:
    """
    Represents a cache invalidation event for coordination.
    
    Used to propagate invalidation events across services and
    trigger cascading invalidations based on dependencies.
    """
    
    cache_key: Optional[str]
    service_origin: str
    tags: Set[str] = field(default_factory=set)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reason: str = "manual"  # "manual", "ttl_expired", "dependency_changed", "system_event"
    cascade_to_dependencies: bool = True
    
    def __post_init__(self):
        """Validate event parameters."""
        valid_reasons = ["manual", "ttl_expired", "dependency_changed", "system_event"]
        if self.reason not in valid_reasons:
            raise ValueError(f"reason must be one of {valid_reasons}")
