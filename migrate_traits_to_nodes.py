#!/usr/bin/env python3
"""
Migration script to convert trait properties to Trait nodes.

This script migrates traits from the old property-based storage (c.traits = [...])
to the new Trait node architecture with HAS_TRAIT relationships.

This migration:
1. Finds all Character and WorldItem nodes with trait properties
2. Creates Trait nodes for each trait
3. Creates HAS_TRAIT relationships
4. Removes the old trait properties

Usage:
    python migrate_traits_to_nodes.py [--dry-run]
"""

import argparse
import asyncio

import structlog

from core.db_manager import neo4j_manager

logger = structlog.get_logger(__name__)


async def migrate_character_traits(dry_run: bool = False) -> int:
    """
    Migrate Character node trait properties to Trait nodes.

    Args:
        dry_run: If True, only report what would be migrated without making changes

    Returns:
        Number of Character nodes migrated
    """
    # First, check how many characters have trait properties
    check_query = """
    MATCH (c:Character)
    WHERE c.traits IS NOT NULL AND size(c.traits) > 0
    RETURN count(c) AS character_count,
           sum(size(c.traits)) AS total_traits
    """

    result = await neo4j_manager.execute_read_query(check_query)
    if not result or not result[0]:
        logger.info("No Character nodes with trait properties found")
        return 0

    character_count = result[0].get("character_count", 0)
    total_traits = result[0].get("total_traits", 0)

    logger.info(
        "Found Character nodes with trait properties",
        character_count=character_count,
        total_traits=total_traits,
    )

    if dry_run:
        print(
            f"[DRY RUN] Would migrate {total_traits} traits from {character_count} Character nodes"
        )
        return character_count

    # Perform the migration
    migrate_query = """
    MATCH (c:Character)
    WHERE c.traits IS NOT NULL AND size(c.traits) > 0
    WITH c, c.traits AS old_traits
    UNWIND old_traits AS trait_name
    MERGE (t:Trait:Entity {name: trait_name})
    ON CREATE SET
        t.description = '',
        t.created_at = timestamp(),
        t.created_chapter = coalesce(c.created_chapter, 0)
    MERGE (c)-[ht:HAS_TRAIT]->(t)
    ON CREATE SET
        ht.chapter_added = coalesce(c.created_chapter, 0),
        ht.last_updated = timestamp()
    WITH DISTINCT c
    SET c.traits = null
    RETURN count(c) AS migrated_count
    """

    result = await neo4j_manager.execute_write_query(migrate_query)
    migrated_count = result[0].get("migrated_count", 0) if result else 0

    logger.info(
        "Migrated Character trait properties to Trait nodes",
        migrated_count=migrated_count,
    )

    return migrated_count


async def migrate_world_item_traits(dry_run: bool = False) -> int:
    """
    Migrate WorldItem/Entity node trait properties to Trait nodes.

    Args:
        dry_run: If True, only report what would be migrated without making changes

    Returns:
        Number of WorldItem/Entity nodes migrated
    """
    # Check for world items with trait properties
    check_query = """
    MATCH (w:Entity)
    WHERE (w:Object OR w:Artifact OR w:Location OR w:Document OR w:Item OR w:Relic)
      AND w.traits IS NOT NULL AND size(w.traits) > 0
    RETURN count(w) AS item_count,
           sum(size(w.traits)) AS total_traits
    """

    result = await neo4j_manager.execute_read_query(check_query)
    if not result or not result[0]:
        logger.info("No WorldItem nodes with trait properties found")
        return 0

    item_count = result[0].get("item_count", 0)
    total_traits = result[0].get("total_traits", 0)

    logger.info(
        "Found WorldItem nodes with trait properties",
        item_count=item_count,
        total_traits=total_traits,
    )

    if dry_run:
        print(
            f"[DRY RUN] Would migrate {total_traits} traits from {item_count} WorldItem nodes"
        )
        return item_count

    # Perform the migration
    migrate_query = """
    MATCH (w:Entity)
    WHERE (w:Object OR w:Artifact OR w:Location OR w:Document OR w:Item OR w:Relic)
      AND w.traits IS NOT NULL AND size(w.traits) > 0
    WITH w, w.traits AS old_traits
    UNWIND old_traits AS trait_name
    MERGE (t:Trait:Entity {name: trait_name})
    ON CREATE SET
        t.description = '',
        t.created_at = timestamp(),
        t.created_chapter = coalesce(w.created_chapter, 0)
    MERGE (w)-[ht:HAS_TRAIT]->(t)
    ON CREATE SET
        ht.chapter_added = coalesce(w.created_chapter, 0),
        ht.last_updated = timestamp()
    WITH DISTINCT w
    SET w.traits = null
    RETURN count(w) AS migrated_count
    """

    result = await neo4j_manager.execute_write_query(migrate_query)
    migrated_count = result[0].get("migrated_count", 0) if result else 0

    logger.info(
        "Migrated WorldItem trait properties to Trait nodes",
        migrated_count=migrated_count,
    )

    return migrated_count


async def main() -> None:
    """Run the trait migration."""
    parser = argparse.ArgumentParser(
        description="Migrate trait properties to Trait nodes"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )
    args = parser.parse_args()

    logger.info(
        "Starting trait property migration",
        dry_run=args.dry_run,
    )

    try:
        # Connect to Neo4j
        await neo4j_manager.connect()

        # Migrate character traits
        print("\n📊 Checking Character nodes...")
        char_count = await migrate_character_traits(dry_run=args.dry_run)

        # Migrate world item traits
        print("\n📊 Checking WorldItem nodes...")
        item_count = await migrate_world_item_traits(dry_run=args.dry_run)

        # Summary
        total_migrated = char_count + item_count

        if args.dry_run:
            print(f"\n[DRY RUN] Total nodes that would be migrated: {total_migrated}")
            print("\nRe-run without --dry-run to perform the migration")
        else:
            print(f"\n✓ Migration complete!")
            print(f"  - Migrated {char_count} Character nodes")
            print(f"  - Migrated {item_count} WorldItem nodes")
            print(f"  - Total: {total_migrated} nodes")

            if total_migrated > 0:
                print("\nNow cleaning up orphaned traits...")
                deleted_count = await neo4j_manager.cleanup_orphaned_traits()
                print(f"✓ Cleaned up {deleted_count} orphaned Trait nodes")

    except Exception as e:
        logger.error("Error during migration", error=str(e), exc_info=True)
        print(f"\n✗ Error during migration: {e}")
        return

    finally:
        await neo4j_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
