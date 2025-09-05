# core/schema_initialization.py
"""
Initialization and management for the dynamic schema system.

This module provides startup integration and management utilities for the
dynamic schema system, ensuring proper initialization and configuration.
"""

import asyncio
import logging

import config
from core.dynamic_schema_manager import dynamic_schema_manager

logger = logging.getLogger(__name__)


async def initialize_dynamic_schema_system(
    force_refresh: bool = False, enable_learning: bool = None, timeout_seconds: int = 30
) -> bool:
    """
    Initialize the dynamic schema system during application startup.

    Args:
        force_refresh: Force refresh of all cached data
        enable_learning: Override the learning configuration
        timeout_seconds: Maximum time to wait for initialization

    Returns:
        True if initialization was successful, False otherwise
    """
    if not config.settings.ENABLE_DYNAMIC_SCHEMA:
        logger.info("Dynamic schema system disabled via configuration")
        return True

    try:
        logger.info("Initializing dynamic schema system...")

        # Configure the schema manager based on settings
        dynamic_schema_manager.configure(
            learning_enabled=enable_learning
            if enable_learning is not None
            else config.settings.DYNAMIC_SCHEMA_LEARNING_ENABLED,
            auto_refresh_enabled=config.settings.DYNAMIC_SCHEMA_AUTO_REFRESH,
            enable_fallback=config.settings.DYNAMIC_SCHEMA_FALLBACK_ENABLED,
            max_cache_age_minutes=config.settings.DYNAMIC_SCHEMA_CACHE_TTL_MINUTES,
        )

        # Configure component-specific settings
        if hasattr(dynamic_schema_manager, "type_inference"):
            dynamic_schema_manager.type_inference.confidence_threshold = (
                config.settings.DYNAMIC_TYPE_INFERENCE_CONFIDENCE_THRESHOLD
            )
            dynamic_schema_manager.type_inference.min_pattern_frequency = (
                config.settings.DYNAMIC_TYPE_PATTERN_MIN_FREQUENCY
            )

        if hasattr(dynamic_schema_manager, "constraint_system"):
            dynamic_schema_manager.constraint_system.min_confidence = (
                config.settings.DYNAMIC_CONSTRAINT_CONFIDENCE_THRESHOLD
            )
            dynamic_schema_manager.constraint_system.min_samples = (
                config.settings.DYNAMIC_CONSTRAINT_MIN_SAMPLES
            )

        if hasattr(dynamic_schema_manager, "introspector"):
            dynamic_schema_manager.introspector.cache_ttl = (
                config.settings.SCHEMA_INTROSPECTION_CACHE_TTL_MINUTES * 60
            )

        # Initialize with timeout to prevent hanging startup
        try:
            await asyncio.wait_for(
                dynamic_schema_manager.initialize(force_refresh=force_refresh),
                timeout=timeout_seconds,
            )

            # Log initialization status
            status = await dynamic_schema_manager.get_system_status()
            logger.info("Dynamic schema system initialized successfully")

            # Log summary statistics
            if status.get("system", {}).get("initialized"):
                type_info = status.get("type_inference", {})
                constraint_info = status.get("constraints", {})
                logger.info(
                    f"Type patterns learned: {type_info.get('total_patterns', 0)}"
                )
                logger.info(
                    f"Relationship constraints learned: {constraint_info.get('total_constraints', 0)}"
                )

            return True

        except TimeoutError:
            logger.warning(
                f"Dynamic schema initialization timed out after {timeout_seconds}s - continuing with fallback"
            )
            return False

    except Exception as e:
        logger.error(f"Failed to initialize dynamic schema system: {e}", exc_info=True)
        return False


async def refresh_dynamic_schema_system() -> bool:
    """
    Refresh the dynamic schema system (useful for maintenance or after data changes).

    Returns:
        True if refresh was successful, False otherwise
    """
    if not config.settings.ENABLE_DYNAMIC_SCHEMA:
        logger.debug("Dynamic schema system disabled - skipping refresh")
        return True

    try:
        logger.info("Refreshing dynamic schema system...")
        await dynamic_schema_manager.invalidate_all_caches()
        await dynamic_schema_manager.initialize(force_refresh=True)
        logger.info("Dynamic schema system refreshed successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to refresh dynamic schema system: {e}", exc_info=True)
        return False


async def get_schema_health_check() -> dict:
    """
    Get health check information for the dynamic schema system.

    Returns:
        Dictionary with health check information
    """
    try:
        if not config.settings.ENABLE_DYNAMIC_SCHEMA:
            return {
                "status": "disabled",
                "healthy": True,
                "message": "Dynamic schema system is disabled via configuration",
            }

        # Get system status
        status = await dynamic_schema_manager.get_system_status()

        # Determine health
        system_info = status.get("system", {})
        is_initialized = system_info.get("initialized", False)
        has_error = "error" in status

        if has_error:
            return {
                "status": "error",
                "healthy": False,
                "message": f"System error: {status.get('error', 'Unknown error')}",
                "details": status,
            }
        elif not is_initialized:
            return {
                "status": "not_initialized",
                "healthy": False,
                "message": "Dynamic schema system not initialized",
                "details": status,
            }
        else:
            # Check if we have reasonable data
            type_patterns = status.get("type_inference", {}).get("total_patterns", 0)
            constraints = status.get("constraints", {}).get("total_constraints", 0)

            if type_patterns == 0 and constraints == 0:
                health_status = "warning"
                healthy = True  # Still functional, just no learned data
                message = (
                    "System initialized but no patterns learned (may be empty database)"
                )
            else:
                health_status = "healthy"
                healthy = True
                message = f"System healthy with {type_patterns} type patterns and {constraints} constraints"

            return {
                "status": health_status,
                "healthy": healthy,
                "message": message,
                "details": {
                    "type_patterns": type_patterns,
                    "constraints": constraints,
                    "last_update": system_info.get("last_update"),
                    "learning_enabled": system_info.get("learning_enabled"),
                    "fallback_enabled": system_info.get("fallback_enabled"),
                },
            }

    except Exception as e:
        return {
            "status": "error",
            "healthy": False,
            "message": f"Health check failed: {e}",
            "error": str(e),
        }


def setup_dynamic_schema_logging():
    """Configure logging for the dynamic schema system components."""
    # Set appropriate log levels for schema components
    schema_loggers = [
        "core.schema_introspector",
        "core.intelligent_type_inference",
        "core.adaptive_constraint_system",
        "core.dynamic_schema_manager",
    ]

    for logger_name in schema_loggers:
        component_logger = logging.getLogger(logger_name)
        # Set to INFO level for schema components, or DEBUG if main log level is DEBUG
        if config.settings.LOG_LEVEL_STR == "DEBUG":
            component_logger.setLevel(logging.DEBUG)
        else:
            component_logger.setLevel(logging.INFO)


# Convenience function for integration into main.py or other startup scripts
async def startup_dynamic_schema_integration(timeout_seconds: int = 30) -> bool:
    """
    Complete startup integration for dynamic schema system.

    This function handles all aspects of startup integration:
    - Logging setup
    - System initialization
    - Error handling and fallback

    Args:
        timeout_seconds: Maximum time to wait for initialization

    Returns:
        True if successful (or safely fell back), False if critical failure
    """
    try:
        # Setup logging
        setup_dynamic_schema_logging()

        # Initialize the system
        success = await initialize_dynamic_schema_system(
            timeout_seconds=timeout_seconds
        )

        if success:
            logger.info("Dynamic schema integration completed successfully")
        else:
            logger.warning(
                "Dynamic schema initialization failed - falling back to static methods"
            )

        return True  # Always return True - fallback is acceptable

    except Exception as e:
        logger.error(
            f"Critical error in dynamic schema startup integration: {e}", exc_info=True
        )
        return False


# Background refresh task (optional - can be scheduled in orchestrator)
async def background_refresh_task(refresh_interval_hours: int = 6):
    """
    Background task to periodically refresh schema data.

    Args:
        refresh_interval_hours: Hours between refreshes
    """
    if not config.settings.DYNAMIC_SCHEMA_AUTO_REFRESH:
        logger.debug("Auto-refresh disabled - background refresh task stopping")
        return

    while True:
        try:
            await asyncio.sleep(
                refresh_interval_hours * 3600
            )  # Convert hours to seconds

            if config.settings.ENABLE_DYNAMIC_SCHEMA:
                logger.info("Running scheduled schema refresh...")
                await refresh_dynamic_schema_system()
            else:
                logger.debug("Dynamic schema disabled - skipping scheduled refresh")

        except asyncio.CancelledError:
            logger.info("Schema refresh background task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in schema refresh background task: {e}", exc_info=True)
            # Continue running despite errors
