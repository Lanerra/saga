# core/cache_policies.py
"""
Cache policy definitions and management for the unified cache coordination system.

This module provides predefined cache policies and policy management functionality
for different service types and use cases.
"""

from typing import Dict, Optional
from core.cache_entry import CachePolicy


class CachePolicyManager:
    """
    Manages cache policies for different services and use cases.
    
    Provides centralized policy management with default policies
    and service-specific overrides.
    """
    
    def __init__(self):
        """Initialize policy manager with default policies."""
        self._policies: Dict[str, CachePolicy] = {}
        self._default_policy = CachePolicy()
        self._setup_default_policies()
    
    def _setup_default_policies(self):
        """Set up default policies for common service types."""
        # Text processing service policies
        self._policies["text_processing"] = CachePolicy(
            max_size=100,
            ttl_seconds=3600,  # 1 hour
            eviction_strategy="LRU",
            shared_across_services=True
        )
        
        # LLM embedding service policies
        self._policies["llm_embedding"] = CachePolicy(
            max_size=128,  # Matches existing config
            ttl_seconds=720,  # 2 hours
            eviction_strategy="LRU",
            shared_across_services=True
        )
        
        # Schema introspection policies
        self._policies["schema_introspection"] = CachePolicy(
            max_size=10,
            ttl_seconds=300,  # 5 minutes (matches existing TTL)
            eviction_strategy="LRU",
            shared_across_services=True
        )
        
        # General service policies
        self._policies["default"] = CachePolicy(
            max_size=100,
            ttl_seconds=1800,  # 30 minutes
            eviction_strategy="LRU",
            shared_across_services=False
        )
    
    def get_policy(self, service_name: str) -> CachePolicy:
        """
        Get cache policy for a specific service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Cache policy for the service, or default policy if not found
        """
        return self._policies.get(service_name, self._default_policy)
    
    def register_policy(self, service_name: str, policy: CachePolicy):
        """
        Register a custom policy for a service.
        
        Args:
            service_name: Name of the service
            policy: Cache policy to register
        """
        self._policies[service_name] = policy
    
    def update_policy(self, service_name: str, **policy_updates) -> CachePolicy:
        """
        Update an existing policy with new values.
        
        Args:
            service_name: Name of the service
            **policy_updates: Policy attributes to update
            
        Returns:
            Updated policy
        """
        current_policy = self.get_policy(service_name)
        
        # Create updated policy with new values
        updated_policy = CachePolicy(
            max_size=policy_updates.get('max_size', current_policy.max_size),
            ttl_seconds=policy_updates.get('ttl_seconds', current_policy.ttl_seconds),
            eviction_strategy=policy_updates.get('eviction_strategy', current_policy.eviction_strategy),
            shared_across_services=policy_updates.get('shared_across_services', current_policy.shared_across_services),
            replication_factor=policy_updates.get('replication_factor', current_policy.replication_factor)
        )
        
        self._policies[service_name] = updated_policy
        return updated_policy
    
    def get_all_policies(self) -> Dict[str, CachePolicy]:
        """Get all registered policies."""
        return self._policies.copy()


# Global policy manager instance
_policy_manager = CachePolicyManager()


def get_policy_manager() -> CachePolicyManager:
    """Get the global cache policy manager instance."""
    return _policy_manager


def get_policy(service_name: str) -> CachePolicy:
    """Get cache policy for a service."""
    return _policy_manager.get_policy(service_name)


def register_policy(service_name: str, policy: CachePolicy):
    """Register a custom policy for a service."""
    _policy_manager.register_policy(service_name, policy)
