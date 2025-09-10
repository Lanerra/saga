# core/service_registry.py
"""
Service Registry for unified dependency injection across SAGA.

This module provides a thread-safe service registry that manages service lifecycle,
dependencies, and performance monitoring. It replaces the mixed architecture patterns
with a consistent dependency injection system.
"""

import asyncio
import functools
import threading
import time
import weakref
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class ServiceLifecycle(Enum):
    """Service lifecycle patterns."""
    SINGLETON = "singleton"  # One instance per registry
    FACTORY = "factory"      # New instance per request
    SCOPED = "scoped"        # One instance per scope/context


class ServiceStatus(Enum):
    """Service registration status."""
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    DISPOSED = "disposed"


@dataclass
class ServiceMetrics:
    """Performance metrics for service operations."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_creation_time: float = 0.0
    last_request_time: Optional[datetime] = None
    error_count: int = 0
    
    def record_request(self, success: bool, creation_time: float = 0.0):
        """Record a service request."""
        self.total_requests += 1
        self.last_request_time = datetime.utcnow()
        
        if success:
            self.successful_requests += 1
            self.total_creation_time += creation_time
        else:
            self.failed_requests += 1
            self.error_count += 1
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        return (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0.0
    
    @property
    def average_creation_time(self) -> float:
        """Calculate average creation time in seconds."""
        return (self.total_creation_time / self.successful_requests) if self.successful_requests > 0 else 0.0


@dataclass
class ServiceRegistration:
    """Service registration information."""
    name: str
    factory: Callable[..., Any]
    lifecycle: ServiceLifecycle
    dependencies: List[str] = field(default_factory=list)
    interface: Optional[type] = None
    instance: Optional[Any] = None
    status: ServiceStatus = ServiceStatus.REGISTERED
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)
    error_message: Optional[str] = None
    registration_time: datetime = field(default_factory=datetime.utcnow)


class ServiceInterface(Protocol):
    """Base protocol for all services."""
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information for monitoring."""
        ...


class ServiceRegistry:
    """
    Thread-safe service registry with dependency injection.
    
    Provides unified service management across SAGA with support for:
    - Multiple lifecycle patterns (singleton, factory, scoped)
    - Dependency resolution with proper initialization order
    - Performance monitoring and metrics
    - Thread-safe operations
    - Service lifecycle management
    """
    
    def __init__(self):
        self._services: Dict[str, ServiceRegistration] = {}
        self._instances: Dict[str, Any] = {}  # For singleton instances
        self._scoped_instances: Dict[str, Dict[str, Any]] = {}  # scope_id -> service_name -> instance
        self._initialization_order: List[str] = []
        self._dependency_graph: Dict[str, Set[str]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        self._initialization_lock = threading.RLock()
        
        # Async support
        self._async_lock = asyncio.Lock()
        
        # Context management
        self._current_scope: Optional[str] = None
        self._scope_stack: List[str] = []
        
        # Performance tracking
        self._registry_metrics = {
            "total_registrations": 0,
            "total_resolutions": 0,
            "failed_resolutions": 0,
            "circular_dependency_errors": 0,
            "creation_time_total": 0.0
        }
        
        logger.info("Service registry initialized")
    
    def register_singleton(
        self, 
        name: str, 
        factory: Callable[..., T], 
        dependencies: List[str] = None,
        interface: type = None
    ) -> 'ServiceRegistry':
        """Register a singleton service."""
        return self._register_service(
            name, factory, ServiceLifecycle.SINGLETON, dependencies or [], interface
        )
    
    def register_factory(
        self, 
        name: str, 
        factory: Callable[..., T], 
        dependencies: List[str] = None,
        interface: type = None
    ) -> 'ServiceRegistry':
        """Register a factory service (new instance per request)."""
        return self._register_service(
            name, factory, ServiceLifecycle.FACTORY, dependencies or [], interface
        )
    
    def register_scoped(
        self, 
        name: str, 
        factory: Callable[..., T], 
        dependencies: List[str] = None,
        interface: type = None
    ) -> 'ServiceRegistry':
        """Register a scoped service (one instance per scope)."""
        return self._register_service(
            name, factory, ServiceLifecycle.SCOPED, dependencies or [], interface
        )
    
    def register_instance(self, name: str, instance: T, interface: type = None) -> 'ServiceRegistry':
        """Register an existing instance as a singleton."""
        with self._lock:
            registration = ServiceRegistration(
                name=name,
                factory=lambda: instance,
                lifecycle=ServiceLifecycle.SINGLETON,
                dependencies=[],
                interface=interface,
                instance=instance,
                status=ServiceStatus.READY
            )
            
            self._services[name] = registration
            self._instances[name] = instance
            self._registry_metrics["total_registrations"] += 1
            
            logger.info(f"Registered instance service: {name}")
            return self
    
    def _register_service(
        self, 
        name: str, 
        factory: Callable[..., T], 
        lifecycle: ServiceLifecycle,
        dependencies: List[str],
        interface: type = None
    ) -> 'ServiceRegistry':
        """Internal service registration."""
        with self._lock:
            # Validate dependencies exist or will be registered
            for dep in dependencies:
                if dep not in self._services:
                    logger.warning(f"Service {name} depends on unregistered service: {dep}")
            
            registration = ServiceRegistration(
                name=name,
                factory=factory,
                lifecycle=lifecycle,
                dependencies=dependencies,
                interface=interface
            )
            
            self._services[name] = registration
            self._dependency_graph[name] = set(dependencies)
            self._registry_metrics["total_registrations"] += 1
            
            logger.info(f"Registered {lifecycle.value} service: {name} with dependencies: {dependencies}")
            return self
    
    def resolve(self, name: str) -> Any:
        """Resolve a service by name."""
        start_time = time.time()
        
        try:
            with self._lock:
                if name not in self._services:
                    self._registry_metrics["failed_resolutions"] += 1
                    raise ValueError(f"Service not registered: {name}")
                
                registration = self._services[name]
                
                # Check for singleton instance
                if registration.lifecycle == ServiceLifecycle.SINGLETON:
                    if name in self._instances:
                        instance = self._instances[name]
                        registration.metrics.record_request(True)
                        self._registry_metrics["total_resolutions"] += 1
                        return instance
                
                # Check for scoped instance
                elif registration.lifecycle == ServiceLifecycle.SCOPED:
                    scope_id = self._current_scope or "default"
                    if (scope_id in self._scoped_instances and 
                        name in self._scoped_instances[scope_id]):
                        instance = self._scoped_instances[scope_id][name]
                        registration.metrics.record_request(True)
                        self._registry_metrics["total_resolutions"] += 1
                        return instance
                
                # Create new instance
                instance = self._create_instance(name)
                creation_time = time.time() - start_time
                
                # Store based on lifecycle
                if registration.lifecycle == ServiceLifecycle.SINGLETON:
                    self._instances[name] = instance
                    registration.instance = instance
                elif registration.lifecycle == ServiceLifecycle.SCOPED:
                    scope_id = self._current_scope or "default"
                    if scope_id not in self._scoped_instances:
                        self._scoped_instances[scope_id] = {}
                    self._scoped_instances[scope_id][name] = instance
                
                registration.status = ServiceStatus.READY
                registration.metrics.record_request(True, creation_time)
                self._registry_metrics["total_resolutions"] += 1
                self._registry_metrics["creation_time_total"] += creation_time
                
                return instance
                
        except Exception as e:
            if name in self._services:
                self._services[name].status = ServiceStatus.ERROR
                self._services[name].error_message = str(e)
                self._services[name].metrics.record_request(False)
            
            self._registry_metrics["failed_resolutions"] += 1
            logger.error(f"Failed to resolve service {name}: {e}", exc_info=True)
            raise
    
    def _create_instance(self, name: str) -> Any:
        """Create a service instance with dependency injection."""
        registration = self._services[name]
        
        if registration.status == ServiceStatus.INITIALIZING:
            self._registry_metrics["circular_dependency_errors"] += 1
            raise RuntimeError(f"Circular dependency detected for service: {name}")
        
        registration.status = ServiceStatus.INITIALIZING
        
        try:
            # Resolve dependencies
            dependencies = {}
            for dep_name in registration.dependencies:
                dependencies[dep_name] = self.resolve(dep_name)
            
            # Create instance
            if dependencies:
                # Try to inject dependencies as keyword arguments
                try:
                    instance = registration.factory(**dependencies)
                except TypeError:
                    # Fallback to positional arguments in dependency order
                    dep_values = [dependencies[dep] for dep in registration.dependencies]
                    instance = registration.factory(*dep_values)
            else:
                instance = registration.factory()
            
            logger.debug(f"Created instance for service: {name}")
            return instance
            
        except Exception as e:
            registration.status = ServiceStatus.ERROR
            registration.error_message = str(e)
            logger.error(f"Failed to create instance for service {name}: {e}", exc_info=True)
            raise
    
    @contextmanager
    def service_scope(self, scope_id: str = None):
        """Context manager for scoped service instances."""
        if scope_id is None:
            scope_id = f"scope_{int(time.time() * 1000000)}"
        
        with self._lock:
            self._scope_stack.append(self._current_scope)
            self._current_scope = scope_id
            
            if scope_id not in self._scoped_instances:
                self._scoped_instances[scope_id] = {}
        
        try:
            yield scope_id
        finally:
            with self._lock:
                # Clean up scoped instances
                if scope_id in self._scoped_instances:
                    for instance in self._scoped_instances[scope_id].values():
                        if hasattr(instance, 'dispose'):
                            try:
                                instance.dispose()
                            except Exception as e:
                                logger.warning(f"Error disposing scoped instance: {e}")
                    
                    del self._scoped_instances[scope_id]
                
                # Restore previous scope
                self._current_scope = self._scope_stack.pop() if self._scope_stack else None
    
    def is_registered(self, name: str) -> bool:
        """Check if a service is registered."""
        with self._lock:
            return name in self._services
    
    def get_service_info(self, name: str) -> Dict[str, Any]:
        """Get information about a registered service."""
        with self._lock:
            if name not in self._services:
                raise ValueError(f"Service not registered: {name}")
            
            registration = self._services[name]
            return {
                "name": registration.name,
                "lifecycle": registration.lifecycle.value,
                "status": registration.status.value,
                "dependencies": registration.dependencies,
                "interface": registration.interface.__name__ if registration.interface else None,
                "has_instance": registration.instance is not None,
                "registration_time": registration.registration_time.isoformat(),
                "error_message": registration.error_message,
                "metrics": {
                    "total_requests": registration.metrics.total_requests,
                    "success_rate": registration.metrics.success_rate,
                    "average_creation_time": registration.metrics.average_creation_time,
                    "last_request_time": registration.metrics.last_request_time.isoformat() 
                        if registration.metrics.last_request_time else None
                }
            }
    
    def get_registry_metrics(self) -> Dict[str, Any]:
        """Get overall registry performance metrics."""
        with self._lock:
            avg_creation_time = (
                self._registry_metrics["creation_time_total"] / 
                max(1, self._registry_metrics["total_resolutions"] - self._registry_metrics["failed_resolutions"])
            )
            
            return {
                "total_services": len(self._services),
                "total_registrations": self._registry_metrics["total_registrations"],
                "total_resolutions": self._registry_metrics["total_resolutions"],
                "failed_resolutions": self._registry_metrics["failed_resolutions"],
                "circular_dependency_errors": self._registry_metrics["circular_dependency_errors"],
                "success_rate": (
                    (self._registry_metrics["total_resolutions"] - self._registry_metrics["failed_resolutions"]) /
                    max(1, self._registry_metrics["total_resolutions"]) * 100
                ),
                "average_creation_time": avg_creation_time,
                "active_singletons": len(self._instances),
                "active_scopes": len(self._scoped_instances),
                "services_by_lifecycle": {
                    lifecycle.value: sum(1 for s in self._services.values() if s.lifecycle == lifecycle)
                    for lifecycle in ServiceLifecycle
                }
            }
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get the service dependency graph."""
        with self._lock:
            return {name: list(deps) for name, deps in self._dependency_graph.items()}
    
    def validate_dependencies(self) -> List[str]:
        """Validate all service dependencies and return any issues."""
        issues = []
        
        with self._lock:
            # Check for missing dependencies
            for service_name, registration in self._services.items():
                for dep_name in registration.dependencies:
                    if dep_name not in self._services:
                        issues.append(f"Service '{service_name}' depends on unregistered service '{dep_name}'")
            
            # Check for circular dependencies
            def has_circular_dependency(service: str, visited: Set[str], path: List[str]) -> bool:
                if service in path:
                    cycle = " -> ".join(path[path.index(service):] + [service])
                    issues.append(f"Circular dependency detected: {cycle}")
                    return True
                
                if service in visited or service not in self._dependency_graph:
                    return False
                
                visited.add(service)
                path.append(service)
                
                for dep in self._dependency_graph[service]:
                    if has_circular_dependency(dep, visited, path):
                        return True
                
                path.pop()
                return False
            
            visited = set()
            for service_name in self._services:
                if service_name not in visited:
                    has_circular_dependency(service_name, visited, [])
        
        return issues
    
    def dispose(self):
        """Dispose of the registry and clean up resources."""
        with self._lock:
            # Dispose singleton instances
            for instance in self._instances.values():
                if hasattr(instance, 'dispose'):
                    try:
                        instance.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing singleton instance: {e}")
            
            # Dispose scoped instances
            for scope_instances in self._scoped_instances.values():
                for instance in scope_instances.values():
                    if hasattr(instance, 'dispose'):
                        try:
                            instance.dispose()
                        except Exception as e:
                            logger.warning(f"Error disposing scoped instance: {e}")
            
            # Clear all data
            self._services.clear()
            self._instances.clear()
            self._scoped_instances.clear()
            self._dependency_graph.clear()
            
            logger.info("Service registry disposed")


# Global service registry instance
_registry = ServiceRegistry()


def get_service_registry() -> ServiceRegistry:
    """Get the global service registry instance."""
    return _registry


def register_singleton(name: str, factory: Callable[..., T], dependencies: List[str] = None, interface: type = None) -> ServiceRegistry:
    """Register a singleton service with the global registry."""
    return _registry.register_singleton(name, factory, dependencies, interface)


def register_factory(name: str, factory: Callable[..., T], dependencies: List[str] = None, interface: type = None) -> ServiceRegistry:
    """Register a factory service with the global registry."""
    return _registry.register_factory(name, factory, dependencies, interface)


def register_scoped(name: str, factory: Callable[..., T], dependencies: List[str] = None, interface: type = None) -> ServiceRegistry:
    """Register a scoped service with the global registry."""
    return _registry.register_scoped(name, factory, dependencies, interface)


def register_instance(name: str, instance: T, interface: type = None) -> ServiceRegistry:
    """Register an existing instance with the global registry."""
    return _registry.register_instance(name, instance, interface)


def resolve(name: str) -> Any:
    """Resolve a service from the global registry."""
    return _registry.resolve(name)


def service_scope(scope_id: str = None):
    """Create a service scope context."""
    return _registry.service_scope(scope_id)


# Type-safe resolution helpers
def resolve_typed(service_type: type[T], name: str = None) -> T:
    """Resolve a service with type safety."""
    service_name = name or service_type.__name__
    return resolve(service_name)