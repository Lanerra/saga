#!/usr/bin/env python3
"""
Migration script to fix RELATIONSHIP type bloat.

This script converts relationships with the literal type 'RELATIONSHIP'
(which have their semantic type stored in a 'type' property) into
properly-typed relationships (KNOWS, LOCATED_IN, etc.).

Background:
-----------
Prior to this fix, native_builders.py was creating all relationships with
the Neo4j type :RELATIONSHIP and storing the actual semantic type
(KNOWS, LOVES, LOCATED_IN, etc.) as a property called 'type'.

This caused:
1. 74+ instances of the generic RELATIONSHIP type
2. Loss of semantic information in the graph structure
3. Inefficient queries that can't filter by relationship type

This migration:
1. Finds all :RELATIONSHIP relationships that have a 'type' property
2. Creates new properly-typed relationships using the 'type' property value
3. Copies all other properties to the new relationship
4. Deletes the old RELATIONSHIP relationships

Usage:
------
    python migrate_relationship_types.py [--dry-run]

Options:
    --dry-run    Show what would be migrated without making changes
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import structlog

from core.db_manager import neo4j_manager

logger = structlog.get_logger(__name__)


async def count_relationship_types() -> dict[str, int]:
    """Count all relationship types in the database."""
    query = """
    MATCH ()-[r]->()
    RETURN type(r) as rel_type, count(r) as count
    ORDER BY count DESC
    """

    results = await neo4j_manager.execute_read_query(query)

    type_counts = {}
    for record in results:
        type_counts[record["rel_type"]] = record["count"]

    return type_counts


async def find_broken_relationships() -> list[dict]:
    """Find all RELATIONSHIP type relationships that need migration."""
    query = """
    MATCH (s)-[r:RELATIONSHIP]->(t)
    WHERE r.type IS NOT NULL AND r.type <> ''
    RETURN
        id(r) as rel_id,
        r.type as semantic_type,
        labels(s) as source_labels,
        s.name as source_name,
        labels(t) as target_labels,
        t.name as target_name,
        properties(r) as props
    LIMIT 1000
    """

    results = await neo4j_manager.execute_read_query(query)

    relationships = []
    for record in results:
        relationships.append({
            "rel_id": record["rel_id"],
            "semantic_type": record["semantic_type"],
            "source_labels": record["source_labels"],
            "source_name": record["source_name"],
            "target_labels": record["target_labels"],
            "target_name": record["target_name"],
            "props": record["props"],
        })

    return relationships


async def migrate_relationships(dry_run: bool = False) -> int:
    """
    Migrate RELATIONSHIP type relationships to proper semantic types.

    Args:
        dry_run: If True, only show what would be migrated without making changes

    Returns:
        Number of relationships migrated
    """
    # Find all broken relationships
    broken_rels = await find_broken_relationships()

    if not broken_rels:
        logger.info("No RELATIONSHIP type relationships found to migrate")
        return 0

    logger.info(
        f"Found {len(broken_rels)} RELATIONSHIP type relationships to migrate",
        dry_run=dry_run,
    )

    if dry_run:
        # Show sample of what would be migrated
        for rel in broken_rels[:10]:
            logger.info(
                "Would migrate",
                semantic_type=rel["semantic_type"],
                source=f"{rel['source_labels'][0] if rel['source_labels'] else 'Unknown'}:{rel['source_name']}",
                target=f"{rel['target_labels'][0] if rel['target_labels'] else 'Unknown'}:{rel['target_name']}",
            )
        if len(broken_rels) > 10:
            logger.info(f"... and {len(broken_rels) - 10} more")
        return len(broken_rels)

    # Migrate in batches
    migration_query = """
    MATCH (s)-[old_r:RELATIONSHIP]->(t)
    WHERE id(old_r) = $rel_id
      AND old_r.type IS NOT NULL
      AND old_r.type <> ''

    WITH s, old_r, t, old_r.type as semantic_type, properties(old_r) as old_props

    // Create new properly-typed relationship using apoc
    CALL apoc.merge.relationship(
        s,
        semantic_type,
        {},  // No properties for matching
        apoc.map.removeKey(old_props, 'type'),  // Remove 'type' property, copy all others
        apoc.map.removeKey(old_props, 'type'),  // Same for onMatch
        t
    ) YIELD rel as new_r

    // Delete old RELATIONSHIP
    DELETE old_r

    RETURN semantic_type, count(new_r) as migrated
    """

    migrated_count = 0
    failed_count = 0

    for rel in broken_rels:
        try:
            results = await neo4j_manager.execute_write_query(
                migration_query,
                {"rel_id": rel["rel_id"]},
            )

            if results:
                migrated_count += 1
                if migrated_count % 100 == 0:
                    logger.info(f"Migrated {migrated_count}/{len(broken_rels)} relationships...")
        except Exception as e:
            failed_count += 1
            logger.error(
                "Failed to migrate relationship",
                rel_id=rel["rel_id"],
                semantic_type=rel["semantic_type"],
                error=str(e),
            )

    logger.info(
        "Migration complete",
        migrated=migrated_count,
        failed=failed_count,
        total=len(broken_rels),
    )

    return migrated_count


async def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description="Migrate RELATIONSHIP type bloat to proper semantic types"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )
    args = parser.parse_args()

    logger.info("Starting relationship type migration", dry_run=args.dry_run)

    # Show current state
    logger.info("Counting relationship types before migration...")
    type_counts_before = await count_relationship_types()

    logger.info("Relationship type distribution (top 10):")
    for rel_type, count in list(type_counts_before.items())[:10]:
        logger.info(f"  {rel_type}: {count}")

    # Run migration
    migrated = await migrate_relationships(dry_run=args.dry_run)

    if args.dry_run:
        logger.info(f"Dry run complete. Would migrate {migrated} relationships.")
        return

    # Show new state
    if migrated > 0:
        logger.info("Counting relationship types after migration...")
        type_counts_after = await count_relationship_types()

        logger.info("Relationship type distribution after migration (top 20):")
        for rel_type, count in list(type_counts_after.items())[:20]:
            logger.info(f"  {rel_type}: {count}")

        # Calculate reduction in RELATIONSHIP type
        before_count = type_counts_before.get("RELATIONSHIP", 0)
        after_count = type_counts_after.get("RELATIONSHIP", 0)
        reduction = before_count - after_count

        logger.info(
            "Migration summary",
            relationship_type_before=before_count,
            relationship_type_after=after_count,
            reduction=reduction,
            migrated_to_semantic_types=migrated,
        )


if __name__ == "__main__":
    asyncio.run(main())
