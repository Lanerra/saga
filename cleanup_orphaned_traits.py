#!/usr/bin/env python3
"""
Utility script to clean up orphaned Trait nodes in the Neo4j database.

This script removes Trait nodes that have no incoming HAS_TRAIT relationships,
which can occur after migrating from the old property-based trait storage to
the new Trait node architecture.

Usage:
    python cleanup_orphaned_traits.py
"""

import asyncio

import structlog

from core.db_manager import neo4j_manager

logger = structlog.get_logger(__name__)


async def main() -> None:
    """Run the orphaned trait cleanup."""
    logger.info("Starting orphaned Trait cleanup...")

    try:
        # Connect to Neo4j
        await neo4j_manager.connect()

        # Run cleanup
        deleted_count = await neo4j_manager.cleanup_orphaned_traits()

        logger.info(
            "Orphaned Trait cleanup complete",
            deleted_count=deleted_count,
        )

        if deleted_count > 0:
            print(f"✓ Cleaned up {deleted_count} orphaned Trait nodes")
        else:
            print("✓ No orphaned Trait nodes found")

    except Exception as e:
        logger.error("Error during cleanup", error=str(e), exc_info=True)
        print(f"✗ Error during cleanup: {e}")
        return

    finally:
        await neo4j_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
