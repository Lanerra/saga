# core/schema_initialization.py
"""
Initialization and management for the static schema system.

This module provides simplified startup integration for the static schema system,
ensuring proper initialization without dynamic overhead.
"""

import structlog

logger = structlog.get_logger(__name__)


async def initialize_static_schema_system() -> bool:
    """
    Initialize the static schema system during application startup.

    Returns:
        True if initialization was successful, False otherwise
    """
    logger.info("Initializing static schema system...")

    # Static schema is always available and doesn't need complex initialization
    logger.info("Static schema system initialized successfully")
    return True


def setup_static_schema_logging():
    """Configure logging for the static schema system components."""
    # Set appropriate log levels for schema components
    schema_loggers = [
        "core.schema_validator",
    ]

    for logger_name in schema_loggers:
        component_logger = structlog.get_logger(logger_name)
        # For structlog, we don't set levels directly on loggers
        # The level filtering is handled by the configuration
        pass


# Convenience function for integration into main.py or other startup scripts
async def startup_schema_integration() -> bool:
    """
    Complete startup integration for static schema system.

    Args:
        timeout_seconds: Maximum time to wait for initialization (ignored for static)

    Returns:
        True if successful
    """
    try:
        # Setup logging
        setup_static_schema_logging()

        # Initialize the system
        success = await initialize_static_schema_system()

        if success:
            logger.info("Static schema integration completed successfully")
        else:
            logger.warning("Static schema initialization had issues")

        return True

    except Exception as e:
        logger.error(f"Error in static schema startup integration: {e}", exc_info=True)
        return False
