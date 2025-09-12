# core/dynamic_schema_manager.py
"""
Unified dynamic schema management system for SAGA.

This is the main interface that coordinates schema introspection, intelligent type inference,
and adaptive constraint validation, replacing static mappings with dynamic, data-driven
schema understanding.

UPDATED: Now uses unified dependency injection system for better service management,
testability, and consistent architecture across SAGA.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

import structlog

# Import the new DI system
from core.service_registry import resolve, register_singleton
from core.database_interface import DatabaseInterface

logger = structlog.get_logger(__name__)


class DynamicSchemaManager:
    """
    Unified schema management with automatic learning and updates.

    This class coordinates all dynamic schema components and provides a single
    interface for type inference and relationship validation.
    
    UPDATED: Now uses service registry for dependency injection instead of
    creating component instances directly.
    """

    def __init__(
        self,
        database_service: Optional[DatabaseInterface] = None,
        schema_introspector=None,
        type_inference_service=None,
        constraint_system=None
    ):
        """
        Initialize the dynamic schema manager.
        
        Args:
            database_service: Database service instance (injected via DI)
            schema_introspector: Schema introspection service (injected via DI)
            type_inference_service: Type inference service (injected via DI)
            constraint_system: Constraint system service (injected via DI)
        """
        # Service dependencies (will be resolved via DI if not provided)
        self._database_service = database_service
        self._schema_introspector = schema_introspector
        self._type_inference_service = type_inference_service
        self._constraint_system = constraint_system
        
        # Track which services were resolved via service registry
        self._services_from_registry = {
            "database": False,
            "introspector": False,
            "type_inference": False,
            "constraints": False
        }

        # State tracking
        self.is_initialized = False
        self.initialization_in_progress = False
        self.last_full_update = None
        self._init_lock = asyncio.Lock()

        # Configuration
        self.enable_fallback = (
            True  # Always fall back to static methods if dynamic fails
        )
        self.auto_refresh_enabled = True
        self.max_cache_age_minutes = 60
        self.learning_enabled = True
        
        # Service resolution statistics
        self._service_stats = {
            "registry_resolutions": 0,
            "fallback_creations": 0,
            "initialization_attempts": 0,
            "successful_initializations": 0
        }

    async def initialize(self, force_refresh: bool = False):
        """
        Initialize the dynamic schema system with learning from existing data.
        
        Now resolves service dependencies via service registry first, with fallback
        to direct instantiation for backward compatibility.
        """
        async with self._init_lock:
            if self.is_initialized and not force_refresh:
                return

            if self.initialization_in_progress:
                logger.debug("Initialization already in progress, waiting...")
                while self.initialization_in_progress:
                    await asyncio.sleep(0.1)
                return

            self.initialization_in_progress = True
            self._service_stats["initialization_attempts"] += 1

            try:
                logger.info("Initializing dynamic schema system...")
                start_time = datetime.utcnow()

                # Resolve service dependencies via DI
                await self._resolve_service_dependencies()

                # Initialize components in parallel for better performance
                tasks = []

                if self.learning_enabled and self._type_inference_service and self._constraint_system:
                    # Learn patterns from existing data
                    if hasattr(self._type_inference_service, 'learn_from_existing_data'):
                        tasks.append(self._type_inference_service.learn_from_existing_data())
                    if hasattr(self._constraint_system, 'learn_constraints_from_data'):
                        tasks.append(self._constraint_system.learn_constraints_from_data())

                # Warm up the introspector cache
                if self._schema_introspector:
                    if hasattr(self._schema_introspector, 'get_active_labels'):
                        tasks.append(self._schema_introspector.get_active_labels())
                    if hasattr(self._schema_introspector, 'get_active_relationship_types'):
                        tasks.append(self._schema_introspector.get_active_relationship_types())

                # Execute all initialization tasks
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Log any exceptions from initialization tasks
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            logger.warning(f"Initialization task {i} failed: {result}")

                self.is_initialized = True
                self.last_full_update = datetime.utcnow()
                self._service_stats["successful_initializations"] += 1

                init_time = (datetime.utcnow() - start_time).total_seconds()
                logger.info(f"Dynamic schema system initialized in {init_time:.2f}s")

                # Log summary
                await self._log_initialization_summary()

            except Exception as e:
                logger.error(
                    f"Failed to initialize dynamic schema system: {e}", exc_info=True
                )
                self.is_initialized = False
            finally:
                self.initialization_in_progress = False
    
    async def _resolve_service_dependencies(self):
        """Resolve service dependencies via service registry with fallbacks."""
        
        # Resolve database service
        if self._database_service is None:
            try:
                self._database_service = resolve("database_service")
                self._services_from_registry["database"] = True
                self._service_stats["registry_resolutions"] += 1
                logger.debug("Database service resolved via registry")
            except Exception as e:
                logger.debug(f"Database service not in registry ({e}), will use fallback if needed")
        
        # Resolve schema introspector
        if self._schema_introspector is None:
            try:
                self._schema_introspector = resolve("schema_introspector")
                self._services_from_registry["introspector"] = True
                self._service_stats["registry_resolutions"] += 1
                logger.debug("Schema introspector resolved via registry")
            except Exception:
                # Fallback: create instance directly
                try:
                    from core.schema_introspector import SchemaIntrospector
                    self._schema_introspector = SchemaIntrospector()
                    self._service_stats["fallback_creations"] += 1
                    logger.debug("Schema introspector created directly (fallback)")
                except Exception as e:
                    logger.warning(f"Failed to create schema introspector: {e}")
        
        # Resolve type inference service
        if self._type_inference_service is None:
            try:
                self._type_inference_service = resolve("type_inference_service")
                self._services_from_registry["type_inference"] = True
                self._service_stats["registry_resolutions"] += 1
                logger.debug("Type inference service resolved via registry")
            except Exception:
                # Fallback: create instance directly
                try:
                    from core.intelligent_type_inference import IntelligentTypeInference
                    self._type_inference_service = IntelligentTypeInference(self._schema_introspector)
                    self._service_stats["fallback_creations"] += 1
                    logger.debug("Type inference service created directly (fallback)")
                except Exception as e:
                    logger.warning(f"Failed to create type inference service: {e}")
        
        # Resolve constraint system
        if self._constraint_system is None:
            try:
                self._constraint_system = resolve("constraint_system")
                self._services_from_registry["constraints"] = True
                self._service_stats["registry_resolutions"] += 1
                logger.debug("Constraint system resolved via registry")
            except Exception:
                # Fallback: create instance directly
                try:
                    from core.adaptive_constraint_system import AdaptiveConstraintSystem
                    self._constraint_system = AdaptiveConstraintSystem(self._schema_introspector)
                    self._service_stats["fallback_creations"] += 1
                    logger.debug("Constraint system created directly (fallback)")
                except Exception as e:
                    logger.warning(f"Failed to create constraint system: {e}")

    async def _log_initialization_summary(self):
        """Log a summary of what was learned during initialization."""
        try:
            # Get summaries from components (with safety checks)
            pattern_summary = {}
            constraint_summary = {}
            
            if self._type_inference_service and hasattr(self._type_inference_service, 'get_pattern_summary'):
                pattern_summary = self._type_inference_service.get_pattern_summary()
            
            if self._constraint_system and hasattr(self._constraint_system, 'get_constraint_summary'):
                constraint_summary = self._constraint_system.get_constraint_summary()

            logger.info(
                f"Type inference: {pattern_summary.get('total_patterns', 0)} patterns learned"
            )
            logger.info(
                f"Constraints: {constraint_summary.get('total_constraints', 0)} relationship constraints learned"
            )
            
            # Log service resolution statistics
            registry_count = sum(1 for used in self._services_from_registry.values() if used)
            logger.info(f"Services resolved: {registry_count}/4 via registry, {self._service_stats['fallback_creations']} via fallback")

        except Exception as e:
            logger.debug(f"Failed to log initialization summary: {e}")

    async def ensure_initialized(self):
        """Ensure the system is initialized before use."""
        if not self.is_initialized and not self.initialization_in_progress:
            await self.initialize()

    async def refresh_if_needed(self, max_age_minutes: int = None):
        """Refresh schema knowledge if it's stale."""
        if not self.auto_refresh_enabled:
            return

        max_age = max_age_minutes or self.max_cache_age_minutes

        if not self.last_full_update:
            await self.initialize()
            return

        age = datetime.utcnow() - self.last_full_update
        if age.total_seconds() > (max_age * 60):
            logger.info(
                f"Schema data is {age.total_seconds() / 60:.1f} minutes old, refreshing..."
            )
            await self.initialize(force_refresh=True)

    async def infer_node_type(
        self, name: str, category: str = "", description: str = ""
    ) -> str:
        """
        Smart type inference combining learned patterns with fallback to static methods.

        This is the main interface that replaces the static _infer_specific_node_type function.
        """
        if not name or not name.strip():
            return "Entity"

        name = name.strip()
        category = category.strip() if category else ""
        description = description.strip() if description else ""

        try:
            # Ensure system is initialized
            await self.ensure_initialized()

            # Refresh data if needed
            await self.refresh_if_needed()

            # Try dynamic inference first
            if self.learning_enabled and self.is_initialized and self._type_inference_service:
                inferred_type, confidence = self._type_inference_service.infer_type(
                    name, category, description
                )

                # Use dynamic inference if confidence is high enough
                if confidence >= 0.5:  # Configurable threshold
                    logger.debug(
                        f"Dynamic inference: '{name}' -> '{inferred_type}' (confidence: {confidence:.3f})"
                    )
                    return inferred_type
                elif confidence > 0.2:
                    # Medium confidence - use as hint but still try fallback
                    logger.debug(
                        f"Dynamic inference low confidence: '{name}' -> '{inferred_type}' (confidence: {confidence:.3f})"
                    )

                    # Try our own type inference system with category-only inference
                    if category:
                        try:
                            # Use the unified inference system for category-based inference
                            category_inferred_type, category_confidence = self._type_inference_service.infer_type(
                                name="", category=category, description=""
                            )
                            if category_confidence >= 0.3 and category_inferred_type != "Entity":
                                logger.debug(
                                    f"Category fallback inference: '{category}' -> '{category_inferred_type}' (confidence: {category_confidence:.3f})"
                                )
                                return category_inferred_type
                        except Exception as e:
                            logger.debug(f"Category fallback inference failed: {e}")

                    # Return dynamic inference even with medium confidence
                    return inferred_type

        except Exception as e:
            logger.warning(f"Dynamic type inference failed for '{name}': {e}")

        # Fallback to static inference if enabled
        if self.enable_fallback:
            try:
                # Import here to avoid circular dependency
                from data_access.kg_queries import _infer_specific_node_type_static

                result = _infer_specific_node_type_static(name, category, "Entity")
                logger.debug(f"Static fallback inference: '{name}' -> '{result}'")
                return result
            except ImportError:
                # If static method doesn't exist yet, try unified inference system
                try:
                    # Use our own unified inference system as final fallback
                    if self._type_inference_service:
                        fallback_type, fallback_confidence = self._type_inference_service.infer_type(
                            name, category, description
                        )
                    else:
                        fallback_type, fallback_confidence = "Entity", 0.0
                    
                    # Accept any non-Entity result from our unified system
                    if fallback_type != "Entity":
                        logger.debug(
                            f"Unified fallback inference: '{name}' -> '{fallback_type}' (confidence: {fallback_confidence:.3f})"
                        )
                        return fallback_type
                        
                except Exception as e:
                    logger.debug(f"Unified fallback inference failed: {e}")
            except Exception as e:
                logger.warning(f"Static fallback inference failed for '{name}': {e}")

        # Final fallback
        return "Entity"

    async def validate_relationship(
        self, subject_type: str, relationship_type: str, object_type: str
    ) -> tuple[bool, float, str]:
        """
        Creative writing-friendly relationship validation - NEVER REJECTS relationships.

        Returns:
            Tuple of (is_always_valid, confidence, explanation)
        """
        try:
            # Ensure system is initialized
            await self.ensure_initialized()
            await self.refresh_if_needed()

            # Try adaptive constraint validation first - now always returns True for creative flexibility
            if self.learning_enabled and self.is_initialized and self._constraint_system:
                is_valid, confidence, reason = (
                    self._constraint_system.validate_relationship(
                        subject_type, relationship_type, object_type
                    )
                )

                # Adaptive system now always returns True with varying confidence
                logger.debug(
                    f"Creative validation: {subject_type}->{relationship_type}->{object_type} (conf: {confidence:.3f}): {reason}"
                )
                return is_valid, confidence, reason

        except Exception as e:
            logger.warning(f"Dynamic constraint validation failed: {e}")

        # NO MORE RULE-BASED FALLBACK - rules are too rigid for creative writing!
        # Instead, be encouraging and permissive

        # Always allow - creative writing needs maximum flexibility
        return (
            True,
            0.5,
            f"Creative relationship: {subject_type} {relationship_type} {object_type} - exploring narrative possibilities",
        )

    async def suggest_relationship_types(
        self, subject_type: str, object_type: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Get suggested valid relationship types between two node types."""
        try:
            await self.ensure_initialized()
            await self.refresh_if_needed()

            if self.learning_enabled and self.is_initialized and self._constraint_system:
                suggestions = self._constraint_system.suggest_relationship_types(
                    subject_type, object_type, limit
                )
                if suggestions:
                    return suggestions

        except Exception as e:
            logger.warning(f"Failed to get relationship suggestions: {e}")

        # Fallback to rule-based suggestions if available
        if self.enable_fallback:
            try:
                from core.relationship_constraints import (
                    get_all_valid_relationships_for_node_pair,
                )

                valid_rels = get_all_valid_relationships_for_node_pair(
                    subject_type, object_type
                )
                return [
                    {
                        "relationship_type": rel,
                        "confidence": 0.7,
                        "sample_size": 0,
                        "examples": [],
                        "source": "rule_based",
                    }
                    for rel in valid_rels[:limit]
                ]
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Rule-based suggestions failed: {e}")

        return []

    async def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive status of the dynamic schema system."""
        try:
            await self.ensure_initialized()

            # Get component statuses (with safety checks)
            pattern_summary = {}
            constraint_summary = {}
            schema_summary = {}
            
            if self.is_initialized and self._type_inference_service and hasattr(self._type_inference_service, 'get_pattern_summary'):
                pattern_summary = self._type_inference_service.get_pattern_summary()
            
            if self.is_initialized and self._constraint_system and hasattr(self._constraint_system, 'get_constraint_summary'):
                constraint_summary = self._constraint_system.get_constraint_summary()
            
            if self.is_initialized and self._schema_introspector and hasattr(self._schema_introspector, 'get_schema_summary'):
                schema_summary = await self._schema_introspector.get_schema_summary()

            return {
                "system": {
                    "initialized": self.is_initialized,
                    "learning_enabled": self.learning_enabled,
                    "auto_refresh_enabled": self.auto_refresh_enabled,
                    "fallback_enabled": self.enable_fallback,
                    "last_update": self.last_full_update.isoformat()
                    if self.last_full_update
                    else None,
                },
                "services": {
                    "from_registry": self._services_from_registry,
                    "registry_resolutions": self._service_stats["registry_resolutions"],
                    "fallback_creations": self._service_stats["fallback_creations"],
                    "available_services": {
                        "database": self._database_service is not None,
                        "introspector": self._schema_introspector is not None,
                        "type_inference": self._type_inference_service is not None,
                        "constraints": self._constraint_system is not None
                    }
                },
                "type_inference": pattern_summary,
                "constraints": constraint_summary,
                "schema": schema_summary,
                "performance": {
                    "cache_age_minutes": self.max_cache_age_minutes,
                    "initialization_in_progress": self.initialization_in_progress,
                    "initialization_attempts": self._service_stats["initialization_attempts"],
                    "successful_initializations": self._service_stats["successful_initializations"],
                },
            }

        except Exception as e:
            return {
                "error": str(e),
                "system": {
                    "initialized": self.is_initialized,
                    "learning_enabled": self.learning_enabled,
                },
            }

    async def invalidate_all_caches(self):
        """Force invalidation of all caches for fresh data."""
        try:
            if self._schema_introspector and hasattr(self._schema_introspector, 'invalidate_cache'):
                await self._schema_introspector.invalidate_cache()
                logger.info("All schema caches invalidated")
            else:
                logger.warning("Schema introspector not available for cache invalidation")
        except Exception as e:
            logger.error(f"Failed to invalidate caches: {e}")

    def configure(
        self,
        learning_enabled: bool = None,
        auto_refresh_enabled: bool = None,
        enable_fallback: bool = None,
        max_cache_age_minutes: int = None,
    ):
        """Configure the dynamic schema system behavior."""
        if learning_enabled is not None:
            self.learning_enabled = learning_enabled
        if auto_refresh_enabled is not None:
            self.auto_refresh_enabled = auto_refresh_enabled
        if enable_fallback is not None:
            self.enable_fallback = enable_fallback
        if max_cache_age_minutes is not None:
            self.max_cache_age_minutes = max_cache_age_minutes

        logger.info(
            f"Dynamic schema system configured: "
            f"learning={self.learning_enabled}, "
            f"auto_refresh={self.auto_refresh_enabled}, "
            f"fallback={self.enable_fallback}"
        )
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get service information for monitoring (ServiceInterface compliance).
        """
        return {
            "service_name": "DynamicSchemaManager",
            "service_type": "schema_management",
            "is_initialized": self.is_initialized,
            "services_from_registry": self._services_from_registry,
            "service_statistics": self._service_stats,
            "configuration": {
                "learning_enabled": self.learning_enabled,
                "auto_refresh_enabled": self.auto_refresh_enabled,
                "enable_fallback": self.enable_fallback,
                "max_cache_age_minutes": self.max_cache_age_minutes
            },
            "last_update": self.last_full_update.isoformat() if self.last_full_update else None
        }
    
    async def dispose(self):
        """
        Dispose of the dynamic schema manager (called by lifecycle manager).
        """
        self.is_initialized = False
        self.initialization_in_progress = False
        
        # Don't dispose of injected services - let the service registry handle that
        # Just clear our references
        self._database_service = None
        self._schema_introspector = None
        self._type_inference_service = None
        self._constraint_system = None
        
        logger.info("Dynamic schema manager disposed")


# Service registration for dependency injection
def register_dynamic_schema_manager_service():
    """Register the dynamic schema manager with the service registry."""
    register_singleton(
        name="dynamic_schema_manager",
        factory=lambda: DynamicSchemaManager(),
        dependencies=["database_service"],  # Other dependencies are optional
        interface=DynamicSchemaManager
    )
    logger.info("Dynamic schema manager service registered")


def get_dynamic_schema_manager() -> DynamicSchemaManager:
    """
    Get a dynamic schema manager instance from the service registry.
    
    Returns:
        DynamicSchemaManager instance
    """
    try:
        return resolve("dynamic_schema_manager")
    except ValueError:
        # Fallback: create instance directly for backward compatibility
        logger.debug("Dynamic schema manager not in service registry, creating new instance")
        return DynamicSchemaManager()


# Global instance for easy access across the application (backward compatibility)
# This will eventually be replaced by service registry usage
dynamic_schema_manager = DynamicSchemaManager()
