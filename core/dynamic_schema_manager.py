# core/dynamic_schema_manager.py
"""
Unified dynamic schema management system for SAGA.

This is the main interface that coordinates schema introspection, intelligent type inference,
and adaptive constraint validation, replacing static mappings with dynamic, data-driven
schema understanding.
"""

import asyncio
import structlog
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import config
from core.schema_introspector import SchemaIntrospector
from core.intelligent_type_inference import IntelligentTypeInference
from core.adaptive_constraint_system import AdaptiveConstraintSystem

logger = structlog.get_logger(__name__)


class DynamicSchemaManager:
    """
    Unified schema management with automatic learning and updates.
    
    This class coordinates all dynamic schema components and provides a single
    interface for type inference and relationship validation.
    """
    
    def __init__(self):
        # Core components
        self.introspector = SchemaIntrospector()
        self.type_inference = IntelligentTypeInference(self.introspector)
        self.constraint_system = AdaptiveConstraintSystem(self.introspector)
        
        # State tracking
        self.is_initialized = False
        self.initialization_in_progress = False
        self.last_full_update = None
        self._init_lock = asyncio.Lock()
        
        # Configuration
        self.enable_fallback = True  # Always fall back to static methods if dynamic fails
        self.auto_refresh_enabled = True
        self.max_cache_age_minutes = 60
        self.learning_enabled = True
    
    async def initialize(self, force_refresh: bool = False):
        """Initialize the dynamic schema system with learning from existing data."""
        async with self._init_lock:
            if self.is_initialized and not force_refresh:
                return
            
            if self.initialization_in_progress:
                logger.debug("Initialization already in progress, waiting...")
                while self.initialization_in_progress:
                    await asyncio.sleep(0.1)
                return
            
            self.initialization_in_progress = True
            
            try:
                logger.info("Initializing dynamic schema system...")
                start_time = datetime.utcnow()
                
                # Initialize components in parallel for better performance
                tasks = []
                
                if self.learning_enabled:
                    # Learn patterns from existing data
                    tasks.append(self.type_inference.learn_from_existing_data())
                    tasks.append(self.constraint_system.learn_constraints_from_data())
                
                # Warm up the introspector cache
                tasks.append(self.introspector.get_active_labels())
                tasks.append(self.introspector.get_active_relationship_types())
                
                # Execute all initialization tasks
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                self.is_initialized = True
                self.last_full_update = datetime.utcnow()
                
                init_time = (datetime.utcnow() - start_time).total_seconds()
                logger.info(f"Dynamic schema system initialized in {init_time:.2f}s")
                
                # Log summary
                await self._log_initialization_summary()
                
            except Exception as e:
                logger.error(f"Failed to initialize dynamic schema system: {e}", exc_info=True)
                self.is_initialized = False
            finally:
                self.initialization_in_progress = False
    
    async def _log_initialization_summary(self):
        """Log a summary of what was learned during initialization."""
        try:
            # Get summaries from components
            pattern_summary = self.type_inference.get_pattern_summary()
            constraint_summary = self.constraint_system.get_constraint_summary()
            
            logger.info(f"Type inference: {pattern_summary.get('total_patterns', 0)} patterns learned")
            logger.info(f"Constraints: {constraint_summary.get('total_constraints', 0)} relationship constraints learned")
            
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
            logger.info(f"Schema data is {age.total_seconds() / 60:.1f} minutes old, refreshing...")
            await self.initialize(force_refresh=True)
    
    async def infer_node_type(self, name: str, category: str = "", description: str = "") -> str:
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
            if self.learning_enabled and self.is_initialized:
                inferred_type, confidence = self.type_inference.infer_type(name, category, description)
                
                # Use dynamic inference if confidence is high enough
                if confidence >= 0.5:  # Configurable threshold
                    logger.debug(f"Dynamic inference: '{name}' -> '{inferred_type}' (confidence: {confidence:.3f})")
                    return inferred_type
                elif confidence > 0.2:
                    # Medium confidence - use as hint but still try fallback
                    logger.debug(f"Dynamic inference low confidence: '{name}' -> '{inferred_type}' (confidence: {confidence:.3f})")
                    
                    # Try enhanced category mapping as secondary option
                    if category:
                        try:
                            from core.enhanced_node_taxonomy import infer_node_type_from_category
                            category_type = infer_node_type_from_category(category)
                            if category_type != "WorldElement":
                                return category_type
                        except ImportError:
                            pass
                    
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
                # If static method doesn't exist yet, try enhanced taxonomy
                try:
                    from core.enhanced_node_taxonomy import infer_node_type_from_name, infer_node_type_from_category
                    
                    # Try category first
                    if category:
                        category_type = infer_node_type_from_category(category)
                        if category_type != "WorldElement":
                            return category_type
                    
                    # Try name-based inference
                    name_type = infer_node_type_from_name(name, f"{category} {description}".strip())
                    if name_type != "Entity":
                        return name_type
                        
                except ImportError:
                    pass
            except Exception as e:
                logger.warning(f"Static fallback inference failed for '{name}': {e}")
        
        # Final fallback
        return "Entity"
    
    async def validate_relationship(
        self, subject_type: str, relationship_type: str, object_type: str
    ) -> Tuple[bool, float, str]:
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
            if self.learning_enabled and self.is_initialized:
                is_valid, confidence, reason = self.constraint_system.validate_relationship(
                    subject_type, relationship_type, object_type
                )
                
                # Adaptive system now always returns True with varying confidence
                logger.debug(f"Creative validation: {subject_type}->{relationship_type}->{object_type} (conf: {confidence:.3f}): {reason}")
                return is_valid, confidence, reason
            
        except Exception as e:
            logger.warning(f"Dynamic constraint validation failed: {e}")
        
        # NO MORE RULE-BASED FALLBACK - rules are too rigid for creative writing!
        # Instead, be encouraging and permissive
        
        # Always allow - creative writing needs maximum flexibility
        return True, 0.5, f"Creative relationship: {subject_type} {relationship_type} {object_type} - exploring narrative possibilities"
    
    async def suggest_relationship_types(
        self, subject_type: str, object_type: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get suggested valid relationship types between two node types."""
        try:
            await self.ensure_initialized()
            await self.refresh_if_needed()
            
            if self.learning_enabled and self.is_initialized:
                suggestions = self.constraint_system.suggest_relationship_types(
                    subject_type, object_type, limit
                )
                if suggestions:
                    return suggestions
            
        except Exception as e:
            logger.warning(f"Failed to get relationship suggestions: {e}")
        
        # Fallback to rule-based suggestions if available
        if self.enable_fallback:
            try:
                from core.relationship_constraints import get_all_valid_relationships_for_node_pair
                valid_rels = get_all_valid_relationships_for_node_pair(subject_type, object_type)
                return [
                    {
                        "relationship_type": rel,
                        "confidence": 0.7,
                        "sample_size": 0,
                        "examples": [],
                        "source": "rule_based"
                    }
                    for rel in valid_rels[:limit]
                ]
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Rule-based suggestions failed: {e}")
        
        return []
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the dynamic schema system."""
        try:
            await self.ensure_initialized()
            
            # Get component statuses
            pattern_summary = self.type_inference.get_pattern_summary() if self.is_initialized else {}
            constraint_summary = self.constraint_system.get_constraint_summary() if self.is_initialized else {}
            
            # Get schema summary from introspector
            schema_summary = await self.introspector.get_schema_summary() if self.is_initialized else {}
            
            return {
                "system": {
                    "initialized": self.is_initialized,
                    "learning_enabled": self.learning_enabled,
                    "auto_refresh_enabled": self.auto_refresh_enabled,
                    "fallback_enabled": self.enable_fallback,
                    "last_update": self.last_full_update.isoformat() if self.last_full_update else None,
                },
                "type_inference": pattern_summary,
                "constraints": constraint_summary,
                "schema": schema_summary,
                "performance": {
                    "cache_age_minutes": self.max_cache_age_minutes,
                    "initialization_in_progress": self.initialization_in_progress,
                }
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "system": {
                    "initialized": self.is_initialized,
                    "learning_enabled": self.learning_enabled,
                }
            }
    
    async def invalidate_all_caches(self):
        """Force invalidation of all caches for fresh data."""
        try:
            await self.introspector.invalidate_cache()
            logger.info("All schema caches invalidated")
        except Exception as e:
            logger.error(f"Failed to invalidate caches: {e}")
    
    def configure(
        self,
        learning_enabled: bool = None,
        auto_refresh_enabled: bool = None,
        enable_fallback: bool = None,
        max_cache_age_minutes: int = None
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
        
        logger.info(f"Dynamic schema system configured: "
                   f"learning={self.learning_enabled}, "
                   f"auto_refresh={self.auto_refresh_enabled}, "
                   f"fallback={self.enable_fallback}")


# Global instance for easy access across the application
dynamic_schema_manager = DynamicSchemaManager()