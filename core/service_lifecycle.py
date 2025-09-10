# core/service_lifecycle.py
"""
Service Lifecycle Manager for SAGA dependency injection system.

This module manages service initialization in dependency order and handles
clean shutdown in reverse dependency order. It supports both sync and async
service initialization patterns.
"""

import asyncio
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Awaitable, Union
from concurrent.futures import ThreadPoolExecutor, Future

import structlog

from core.service_registry import ServiceRegistry, get_service_registry

logger = structlog.get_logger(__name__)


class InitializationPhase(Enum):
    """Service initialization phases."""
    NOT_STARTED = "not_started"
    DEPENDENCIES_RESOLVED = "dependencies_resolved"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    FAILED = "failed"
    DISPOSED = "disposed"


class ShutdownPhase(Enum):
    """Service shutdown phases."""
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    DISPOSED = "disposed"
    FAILED = "failed"


@dataclass
class ServiceLifecycleInfo:
    """Lifecycle information for a service."""
    name: str
    phase: InitializationPhase = InitializationPhase.NOT_STARTED
    shutdown_phase: ShutdownPhase = ShutdownPhase.RUNNING
    dependencies: List[str] = field(default_factory=list)
    dependents: Set[str] = field(default_factory=set)
    initialization_time: Optional[float] = None
    error_message: Optional[str] = None
    initialized_at: Optional[datetime] = None
    disposal_time: Optional[float] = None
    disposed_at: Optional[datetime] = None


class ServiceHealthCheck:
    """Service health check result."""
    
    def __init__(self, service_name: str, healthy: bool, message: str = "", details: Dict[str, Any] = None):
        self.service_name = service_name
        self.healthy = healthy
        self.message = message
        self.details = details or {}
        self.checked_at = datetime.utcnow()


class ServiceLifecycleManager:
    """
    Manages service lifecycle with dependency-ordered initialization and shutdown.
    
    Features:
    - Dependency-ordered initialization
    - Reverse-order shutdown 
    - Async service initialization support
    - Health checking
    - Graceful error handling and recovery
    - Performance monitoring
    """
    
    def __init__(self, service_registry: ServiceRegistry = None):
        self._registry = service_registry or get_service_registry()
        self._lifecycle_info: Dict[str, ServiceLifecycleInfo] = {}
        self._initialization_order: List[str] = []
        self._shutdown_order: List[str] = []
        
        # State management
        self._is_initialized = False
        self._is_shutting_down = False
        self._initialization_lock = threading.RLock()
        self._shutdown_lock = threading.RLock()
        
        # Async support
        self._async_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="lifecycle-")
        
        # Performance tracking
        self._metrics = {
            "total_initialization_time": 0.0,
            "services_initialized": 0,
            "initialization_failures": 0,
            "shutdown_time": 0.0,
            "services_disposed": 0,
            "disposal_failures": 0,
            "health_check_count": 0,
            "healthy_services": 0
        }
        
        logger.info("Service lifecycle manager initialized")
    
    def register_service_lifecycle(
        self, 
        service_name: str, 
        dependencies: List[str] = None,
        initialization_callback: Optional[Callable[[Any], Union[None, Awaitable[None]]]] = None,
        disposal_callback: Optional[Callable[[Any], Union[None, Awaitable[None]]]] = None,
        health_check: Optional[Callable[[Any], Union[ServiceHealthCheck, Awaitable[ServiceHealthCheck]]]] = None
    ):
        """Register lifecycle callbacks for a service."""
        with self._initialization_lock:
            if service_name not in self._lifecycle_info:
                self._lifecycle_info[service_name] = ServiceLifecycleInfo(
                    name=service_name,
                    dependencies=dependencies or []
                )
            
            lifecycle_info = self._lifecycle_info[service_name]
            lifecycle_info.dependencies = dependencies or []
            
            # Store callbacks as attributes on the lifecycle info
            if initialization_callback:
                setattr(lifecycle_info, '_init_callback', initialization_callback)
            if disposal_callback:
                setattr(lifecycle_info, '_disposal_callback', disposal_callback)
            if health_check:
                setattr(lifecycle_info, '_health_check', health_check)
            
            # Update dependency graph
            for dep in lifecycle_info.dependencies:
                if dep not in self._lifecycle_info:
                    self._lifecycle_info[dep] = ServiceLifecycleInfo(name=dep)
                self._lifecycle_info[dep].dependents.add(service_name)
    
    async def initialize_services(self, service_names: List[str] = None) -> bool:
        """
        Initialize services in dependency order.
        
        Args:
            service_names: Specific services to initialize. If None, initialize all registered services.
            
        Returns:
            True if all services initialized successfully, False otherwise.
        """
        async with self._async_lock:
            if self._is_initialized:
                logger.warning("Services already initialized")
                return True
            
            if self._is_shutting_down:
                raise RuntimeError("Cannot initialize services during shutdown")
            
            start_time = time.time()
            logger.info("Starting service initialization...")
            
            try:
                # Determine which services to initialize
                target_services = service_names or list(self._registry._services.keys())
                
                # Build complete dependency graph for target services
                all_required_services = self._get_transitive_dependencies(target_services)
                
                # Calculate initialization order
                initialization_order = self._calculate_initialization_order(all_required_services)
                
                # Initialize services in order
                success_count = 0
                for service_name in initialization_order:
                    try:
                        success = await self._initialize_single_service(service_name)
                        if success:
                            success_count += 1
                            self._metrics["services_initialized"] += 1
                        else:
                            self._metrics["initialization_failures"] += 1
                            logger.error(f"Failed to initialize service: {service_name}")
                    except Exception as e:
                        self._metrics["initialization_failures"] += 1
                        logger.error(f"Exception initializing service {service_name}: {e}", exc_info=True)
                        # Continue with other services for resilience
                
                total_time = time.time() - start_time
                self._metrics["total_initialization_time"] = total_time
                self._initialization_order = initialization_order
                self._shutdown_order = list(reversed(initialization_order))
                
                # Mark as initialized if we got at least some services
                self._is_initialized = success_count > 0
                
                logger.info(
                    f"Service initialization completed in {total_time:.2f}s. "
                    f"Successful: {success_count}/{len(initialization_order)}"
                )
                
                return success_count == len(initialization_order)
                
            except Exception as e:
                logger.error(f"Service initialization failed: {e}", exc_info=True)
                return False
    
    async def _initialize_single_service(self, service_name: str) -> bool:
        """Initialize a single service."""
        if service_name not in self._lifecycle_info:
            self._lifecycle_info[service_name] = ServiceLifecycleInfo(name=service_name)
        
        lifecycle_info = self._lifecycle_info[service_name]
        
        if lifecycle_info.phase == InitializationPhase.INITIALIZED:
            return True
        
        if lifecycle_info.phase == InitializationPhase.FAILED:
            logger.warning(f"Skipping failed service: {service_name}")
            return False
        
        try:
            lifecycle_info.phase = InitializationPhase.INITIALIZING
            start_time = time.time()
            
            # Ensure service is registered and resolved
            if not self._registry.is_registered(service_name):
                logger.warning(f"Service not registered in registry: {service_name}")
                return False
            
            # Resolve the service instance (this triggers creation)
            service_instance = self._registry.resolve(service_name)
            
            # Call initialization callback if provided
            init_callback = getattr(lifecycle_info, '_init_callback', None)
            if init_callback:
                if asyncio.iscoroutinefunction(init_callback):
                    await init_callback(service_instance)
                else:
                    # Run sync callback in executor to avoid blocking
                    await asyncio.get_event_loop().run_in_executor(
                        self._executor, init_callback, service_instance
                    )
            
            # Special handling for services with async initialize methods
            if hasattr(service_instance, 'initialize') and asyncio.iscoroutinefunction(service_instance.initialize):
                await service_instance.initialize()
            elif hasattr(service_instance, 'initialize'):
                # Run sync initialize in executor
                await asyncio.get_event_loop().run_in_executor(
                    self._executor, service_instance.initialize
                )
            
            initialization_time = time.time() - start_time
            lifecycle_info.initialization_time = initialization_time
            lifecycle_info.initialized_at = datetime.utcnow()
            lifecycle_info.phase = InitializationPhase.INITIALIZED
            
            logger.debug(f"Initialized service {service_name} in {initialization_time:.3f}s")
            return True
            
        except Exception as e:
            lifecycle_info.phase = InitializationPhase.FAILED
            lifecycle_info.error_message = str(e)
            logger.error(f"Failed to initialize service {service_name}: {e}", exc_info=True)
            return False
    
    def _get_transitive_dependencies(self, service_names: List[str]) -> Set[str]:
        """Get all transitive dependencies for given services."""
        required_services = set()
        to_process = deque(service_names)
        
        while to_process:
            service_name = to_process.popleft()
            if service_name in required_services:
                continue
            
            required_services.add(service_name)
            
            # Add dependencies from registry
            if self._registry.is_registered(service_name):
                service_info = self._registry.get_service_info(service_name)
                for dep in service_info.get("dependencies", []):
                    if dep not in required_services:
                        to_process.append(dep)
            
            # Add dependencies from lifecycle info
            if service_name in self._lifecycle_info:
                for dep in self._lifecycle_info[service_name].dependencies:
                    if dep not in required_services:
                        to_process.append(dep)
        
        return required_services
    
    def _calculate_initialization_order(self, service_names: Set[str]) -> List[str]:
        """Calculate dependency-ordered initialization sequence using topological sort."""
        # Build dependency graph
        graph = defaultdict(set)
        in_degree = defaultdict(int)
        
        for service_name in service_names:
            dependencies = []
            
            # Get dependencies from registry
            if self._registry.is_registered(service_name):
                service_info = self._registry.get_service_info(service_name)
                dependencies.extend(service_info.get("dependencies", []))
            
            # Get dependencies from lifecycle info
            if service_name in self._lifecycle_info:
                dependencies.extend(self._lifecycle_info[service_name].dependencies)
            
            # Remove duplicates
            dependencies = list(set(dependencies))
            
            for dep in dependencies:
                if dep in service_names:  # Only include deps that are being initialized
                    graph[dep].add(service_name)
                    in_degree[service_name] += 1
            
            # Ensure service is in in_degree map
            if service_name not in in_degree:
                in_degree[service_name] = 0
        
        # Topological sort
        queue = deque([service for service in service_names if in_degree[service] == 0])
        initialization_order = []
        
        while queue:
            service = queue.popleft()
            initialization_order.append(service)
            
            for dependent in graph[service]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check for circular dependencies
        if len(initialization_order) != len(service_names):
            remaining = service_names - set(initialization_order)
            logger.error(f"Circular dependency detected. Remaining services: {remaining}")
            # Add remaining services anyway (they'll likely fail to initialize)
            initialization_order.extend(remaining)
        
        return initialization_order
    
    async def shutdown_services(self, graceful_timeout: float = 30.0) -> bool:
        """
        Shutdown services in reverse dependency order.
        
        Args:
            graceful_timeout: Maximum time to wait for graceful shutdown
            
        Returns:
            True if all services shut down successfully
        """
        async with self._async_lock:
            if self._is_shutting_down:
                logger.warning("Shutdown already in progress")
                return True
            
            if not self._is_initialized:
                logger.info("Services not initialized, nothing to shutdown")
                return True
            
            self._is_shutting_down = True
            start_time = time.time()
            
            logger.info("Starting service shutdown...")
            
            try:
                success_count = 0
                
                # Shutdown in reverse initialization order
                for service_name in self._shutdown_order:
                    try:
                        if service_name in self._lifecycle_info:
                            lifecycle_info = self._lifecycle_info[service_name]
                            
                            if lifecycle_info.phase == InitializationPhase.INITIALIZED:
                                success = await self._shutdown_single_service(service_name, graceful_timeout)
                                if success:
                                    success_count += 1
                                    self._metrics["services_disposed"] += 1
                                else:
                                    self._metrics["disposal_failures"] += 1
                    except Exception as e:
                        self._metrics["disposal_failures"] += 1
                        logger.error(f"Exception shutting down service {service_name}: {e}", exc_info=True)
                
                total_time = time.time() - start_time
                self._metrics["shutdown_time"] = total_time
                
                logger.info(
                    f"Service shutdown completed in {total_time:.2f}s. "
                    f"Successful: {success_count}/{len(self._shutdown_order)}"
                )
                
                return success_count == len(self._shutdown_order)
                
            finally:
                self._is_shutting_down = False
                self._is_initialized = False
    
    async def _shutdown_single_service(self, service_name: str, timeout: float) -> bool:
        """Shutdown a single service."""
        lifecycle_info = self._lifecycle_info[service_name]
        lifecycle_info.shutdown_phase = ShutdownPhase.SHUTTING_DOWN
        
        try:
            start_time = time.time()
            
            # Get service instance
            if self._registry.is_registered(service_name):
                try:
                    service_instance = self._registry.resolve(service_name)
                except Exception:
                    # Service might already be disposed
                    lifecycle_info.shutdown_phase = ShutdownPhase.DISPOSED
                    return True
                
                # Call disposal callback if provided
                disposal_callback = getattr(lifecycle_info, '_disposal_callback', None)
                if disposal_callback:
                    try:
                        if asyncio.iscoroutinefunction(disposal_callback):
                            await asyncio.wait_for(disposal_callback(service_instance), timeout=timeout)
                        else:
                            # Run sync callback in executor with timeout
                            future = asyncio.get_event_loop().run_in_executor(
                                self._executor, disposal_callback, service_instance
                            )
                            await asyncio.wait_for(future, timeout=timeout)
                    except asyncio.TimeoutError:
                        logger.warning(f"Disposal callback timeout for service: {service_name}")
                
                # Call dispose method if available
                if hasattr(service_instance, 'dispose'):
                    try:
                        if asyncio.iscoroutinefunction(service_instance.dispose):
                            await asyncio.wait_for(service_instance.dispose(), timeout=timeout)
                        else:
                            future = asyncio.get_event_loop().run_in_executor(
                                self._executor, service_instance.dispose
                            )
                            await asyncio.wait_for(future, timeout=timeout)
                    except asyncio.TimeoutError:
                        logger.warning(f"Dispose timeout for service: {service_name}")
                    except Exception as e:
                        logger.warning(f"Error disposing service {service_name}: {e}")
            
            disposal_time = time.time() - start_time
            lifecycle_info.disposal_time = disposal_time
            lifecycle_info.disposed_at = datetime.utcnow()
            lifecycle_info.shutdown_phase = ShutdownPhase.DISPOSED
            
            logger.debug(f"Disposed service {service_name} in {disposal_time:.3f}s")
            return True
            
        except Exception as e:
            lifecycle_info.shutdown_phase = ShutdownPhase.FAILED
            lifecycle_info.error_message = str(e)
            logger.error(f"Failed to dispose service {service_name}: {e}", exc_info=True)
            return False
    
    async def health_check(self, service_names: List[str] = None) -> Dict[str, ServiceHealthCheck]:
        """Perform health checks on services."""
        target_services = service_names or [
            name for name, info in self._lifecycle_info.items() 
            if info.phase == InitializationPhase.INITIALIZED
        ]
        
        health_results = {}
        
        for service_name in target_services:
            self._metrics["health_check_count"] += 1
            
            try:
                if service_name not in self._lifecycle_info:
                    health_results[service_name] = ServiceHealthCheck(
                        service_name, False, "Service not registered in lifecycle manager"
                    )
                    continue
                
                lifecycle_info = self._lifecycle_info[service_name]
                
                if lifecycle_info.phase != InitializationPhase.INITIALIZED:
                    health_results[service_name] = ServiceHealthCheck(
                        service_name, False, f"Service not initialized (phase: {lifecycle_info.phase.value})"
                    )
                    continue
                
                # Check if service is still available in registry
                if not self._registry.is_registered(service_name):
                    health_results[service_name] = ServiceHealthCheck(
                        service_name, False, "Service not found in registry"
                    )
                    continue
                
                # Get service instance and run health check
                try:
                    service_instance = self._registry.resolve(service_name)
                    
                    # Use custom health check if provided
                    health_check_func = getattr(lifecycle_info, '_health_check', None)
                    if health_check_func:
                        if asyncio.iscoroutinefunction(health_check_func):
                            health_result = await health_check_func(service_instance)
                        else:
                            health_result = await asyncio.get_event_loop().run_in_executor(
                                self._executor, health_check_func, service_instance
                            )
                    else:
                        # Default health check - just verify instance exists and has no obvious issues
                        health_result = ServiceHealthCheck(service_name, True, "Service instance available")
                    
                    health_results[service_name] = health_result
                    
                    if health_result.healthy:
                        self._metrics["healthy_services"] += 1
                    
                except Exception as e:
                    health_results[service_name] = ServiceHealthCheck(
                        service_name, False, f"Health check failed: {str(e)}"
                    )
                
            except Exception as e:
                health_results[service_name] = ServiceHealthCheck(
                    service_name, False, f"Health check error: {str(e)}"
                )
        
        return health_results
    
    def get_lifecycle_status(self) -> Dict[str, Any]:
        """Get overall lifecycle manager status."""
        total_services = len(self._lifecycle_info)
        initialized_services = sum(
            1 for info in self._lifecycle_info.values() 
            if info.phase == InitializationPhase.INITIALIZED
        )
        failed_services = sum(
            1 for info in self._lifecycle_info.values() 
            if info.phase == InitializationPhase.FAILED
        )
        
        return {
            "is_initialized": self._is_initialized,
            "is_shutting_down": self._is_shutting_down,
            "total_services": total_services,
            "initialized_services": initialized_services,
            "failed_services": failed_services,
            "initialization_order": self._initialization_order,
            "shutdown_order": self._shutdown_order,
            "metrics": self._metrics,
            "services": {
                name: {
                    "phase": info.phase.value,
                    "shutdown_phase": info.shutdown_phase.value,
                    "initialization_time": info.initialization_time,
                    "disposal_time": info.disposal_time,
                    "error_message": info.error_message,
                    "initialized_at": info.initialized_at.isoformat() if info.initialized_at else None,
                    "disposed_at": info.disposed_at.isoformat() if info.disposed_at else None
                }
                for name, info in self._lifecycle_info.items()
            }
        }
    
    @asynccontextmanager
    async def lifecycle_context(self, services: List[str] = None):
        """Context manager for automatic service lifecycle management."""
        try:
            success = await self.initialize_services(services)
            if not success:
                logger.warning("Some services failed to initialize")
            yield self
        finally:
            await self.shutdown_services()
    
    def dispose(self):
        """Dispose of the lifecycle manager."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
        logger.info("Service lifecycle manager disposed")


# Global lifecycle manager instance
_lifecycle_manager = ServiceLifecycleManager()


def get_lifecycle_manager() -> ServiceLifecycleManager:
    """Get the global service lifecycle manager."""
    return _lifecycle_manager


async def initialize_services(service_names: List[str] = None) -> bool:
    """Initialize services using the global lifecycle manager."""
    return await _lifecycle_manager.initialize_services(service_names)


async def shutdown_services(graceful_timeout: float = 30.0) -> bool:
    """Shutdown services using the global lifecycle manager."""
    return await _lifecycle_manager.shutdown_services(graceful_timeout)


def lifecycle_context(services: List[str] = None):
    """Create a lifecycle context using the global manager."""
    return _lifecycle_manager.lifecycle_context(services)