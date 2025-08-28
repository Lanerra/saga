# tools/relationship_migration.py
"""
Migration tools for fixing existing invalid relationships in the knowledge graph.

This module provides utilities to:
1. Audit existing relationships for constraint violations
2. Automatically fix invalid relationships where possible
3. Report on relationships that need manual attention
4. Backup and restore relationship data during migration
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from core.db_manager import neo4j_manager
from core.relationship_constraints import validate_relationship_semantics
from core.relationship_validator import (
    RelationshipConstraintValidator,
)

logger = logging.getLogger(__name__)


class RelationshipMigrationTool:
    """Main tool for migrating and fixing relationship constraints in the knowledge graph."""

    def __init__(self):
        self.validator = RelationshipConstraintValidator()
        self.migration_stats = {
            "total_relationships": 0,
            "valid_relationships": 0,
            "corrected_relationships": 0,
            "deleted_relationships": 0,
            "failed_corrections": 0,
            "backup_created": False,
        }
        self.backup_file: Path | None = None

    async def audit_existing_relationships(
        self, limit: int | None = None
    ) -> dict[str, Any]:
        """
        Audit all existing relationships in the knowledge graph for constraint violations.

        Args:
            limit: Optional limit on number of relationships to check

        Returns:
            Dictionary containing audit results and statistics
        """
        logger.info("Starting relationship constraint audit...")

        # Query all relationships with their node types
        query = """
        MATCH (s)-[r]->(o)
        WHERE s:Entity AND o:Entity
        RETURN 
            elementId(r) as rel_id,
            s.name as subject_name,
            labels(s) as subject_labels,
            type(r) as relationship_type,
            o.name as object_name,
            labels(o) as object_labels,
            properties(r) as rel_properties
        ORDER BY s.name, type(r), o.name
        """ + (f" LIMIT {limit}" if limit else "")

        try:
            results = await neo4j_manager.execute_read_query(query)
            logger.info(f"Found {len(results)} relationships to audit")

            audit_results = {
                "total_checked": len(results),
                "valid": [],
                "invalid": [],
                "correctable": [],
                "uncorrectable": [],
            }

            for record in results:
                rel_id = record["rel_id"]
                relationship_type = record["relationship_type"]
                subject_labels = record["subject_labels"]
                object_labels = record["object_labels"]

                # Extract primary node types (skip 'Entity' base label)
                subject_type = self._extract_primary_node_type(subject_labels)
                object_type = self._extract_primary_node_type(object_labels)

                # Validate the relationship
                is_valid, errors = validate_relationship_semantics(
                    subject_type, relationship_type, object_type
                )

                relationship_data = {
                    "rel_id": rel_id,
                    "subject_name": record["subject_name"],
                    "subject_type": subject_type,
                    "relationship_type": relationship_type,
                    "object_name": record["object_name"],
                    "object_type": object_type,
                    "properties": record["rel_properties"],
                    "errors": errors,
                }

                if is_valid:
                    audit_results["valid"].append(relationship_data)
                else:
                    # Try to find a correction
                    correction_result = self.validator.validate_relationship(
                        subject_type, relationship_type, object_type
                    )

                    if (
                        correction_result.is_valid
                        and correction_result.validated_relationship
                        != relationship_type
                    ):
                        relationship_data["suggested_correction"] = (
                            correction_result.validated_relationship
                        )
                        relationship_data["correction_confidence"] = (
                            correction_result.confidence
                        )
                        audit_results["correctable"].append(relationship_data)
                    else:
                        audit_results["uncorrectable"].append(relationship_data)
                        audit_results["invalid"].append(relationship_data)

            # Generate audit summary
            summary = self._generate_audit_summary(audit_results)
            audit_results["summary"] = summary

            logger.info(f"Audit complete: {summary}")
            return audit_results

        except Exception as e:
            logger.error(f"Error during relationship audit: {e}", exc_info=True)
            raise

    def _extract_primary_node_type(self, labels: list[str]) -> str:
        """Extract the primary node type from a list of labels, excluding 'Entity'."""
        for label in labels:
            if label != "Entity":
                return label
        return "Entity"  # Fallback

    def _generate_audit_summary(self, audit_results: dict[str, Any]) -> dict[str, Any]:
        """Generate a summary of audit results."""
        total = audit_results["total_checked"]
        valid = len(audit_results["valid"])
        correctable = len(audit_results["correctable"])
        uncorrectable = len(audit_results["uncorrectable"])

        return {
            "total_relationships": total,
            "valid_relationships": valid,
            "invalid_relationships": total - valid,
            "correctable_relationships": correctable,
            "uncorrectable_relationships": uncorrectable,
            "valid_percentage": (valid / total * 100) if total > 0 else 0,
            "correctable_percentage": (correctable / total * 100) if total > 0 else 0,
        }

    async def create_backup(self, backup_dir: str = "backups") -> Path:
        """
        Create a backup of all relationships before migration.

        Args:
            backup_dir: Directory to store the backup file

        Returns:
            Path to the created backup file
        """
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path / f"relationships_backup_{timestamp}.json"

        logger.info(f"Creating relationship backup at {backup_file}")

        # Query all relationships with full data
        query = """
        MATCH (s)-[r]->(o)
        WHERE s:Entity AND o:Entity
        RETURN 
            elementId(r) as rel_id,
            s.name as subject_name,
            labels(s) as subject_labels,
            s.id as subject_id,
            type(r) as relationship_type,
            properties(r) as rel_properties,
            o.name as object_name,
            labels(o) as object_labels,
            o.id as object_id
        ORDER BY s.name, type(r), o.name
        """

        try:
            results = await neo4j_manager.execute_read_query(query)

            backup_data = {
                "backup_timestamp": timestamp,
                "total_relationships": len(results),
                "relationships": [dict(record) for record in results],
            }

            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, default=str)

            self.backup_file = backup_file
            self.migration_stats["backup_created"] = True

            logger.info(f"Backup created with {len(results)} relationships")
            return backup_file

        except Exception as e:
            logger.error(f"Error creating backup: {e}", exc_info=True)
            raise

    async def fix_correctable_relationships(
        self,
        audit_results: dict[str, Any],
        dry_run: bool = True,
        min_confidence: float = 0.5,
    ) -> dict[str, Any]:
        """
        Fix relationships that can be automatically corrected.

        Args:
            audit_results: Results from audit_existing_relationships
            dry_run: If True, only simulate fixes without making changes
            min_confidence: Minimum confidence threshold for automatic corrections

        Returns:
            Dictionary containing fix results and statistics
        """
        correctable = audit_results.get("correctable", [])
        logger.info(
            f"Processing {len(correctable)} correctable relationships (dry_run={dry_run})"
        )

        fix_results = {
            "processed": 0,
            "successful_fixes": 0,
            "skipped_low_confidence": 0,
            "failed_fixes": 0,
            "fixes": [],
        }

        for rel_data in correctable:
            rel_id = rel_data["rel_id"]
            current_type = rel_data["relationship_type"]
            suggested_type = rel_data["suggested_correction"]
            confidence = rel_data["correction_confidence"]

            fix_results["processed"] += 1

            # Skip low confidence corrections
            if confidence < min_confidence:
                fix_results["skipped_low_confidence"] += 1
                logger.debug(
                    f"Skipping low confidence correction: {current_type} -> {suggested_type} "
                    f"(confidence: {confidence:.2f})"
                )
                continue

            fix_record = {
                "rel_id": rel_id,
                "subject": f"{rel_data['subject_type']}:{rel_data['subject_name']}",
                "original_type": current_type,
                "corrected_type": suggested_type,
                "confidence": confidence,
                "success": False,
            }

            if not dry_run:
                try:
                    # Update the relationship type
                    update_query = """
                    MATCH ()-[r]->()
                    WHERE elementId(r) = $rel_id
                    SET r.migration_original_type = $original_type,
                        r.migration_timestamp = timestamp(),
                        r.migration_confidence = $confidence
                    """

                    # For DYNAMIC_REL relationships, update the type property
                    if current_type == "DYNAMIC_REL":
                        update_query += ", r.type = $new_type"
                    else:
                        # For typed relationships, we need to recreate them
                        # This is more complex and should be done carefully
                        logger.warning(
                            f"Typed relationship correction not yet implemented: "
                            f"{current_type} -> {suggested_type}"
                        )
                        continue

                    params = {
                        "rel_id": rel_id,
                        "original_type": current_type,
                        "new_type": suggested_type,
                        "confidence": confidence,
                    }

                    await neo4j_manager.execute_write_query(update_query, params)

                    fix_record["success"] = True
                    fix_results["successful_fixes"] += 1

                    logger.info(
                        f"Fixed relationship: {rel_data['subject_name']} "
                        f"{current_type} -> {suggested_type} {rel_data['object_name']} "
                        f"(confidence: {confidence:.2f})"
                    )

                except Exception as e:
                    fix_record["error"] = str(e)
                    fix_results["failed_fixes"] += 1
                    logger.error(
                        f"Failed to fix relationship {rel_id}: {e}", exc_info=True
                    )
            else:
                # Dry run - just log what would be done
                logger.info(
                    f"[DRY RUN] Would fix: {rel_data['subject_name']} "
                    f"{current_type} -> {suggested_type} {rel_data['object_name']} "
                    f"(confidence: {confidence:.2f})"
                )
                fix_record["success"] = True  # Assume success for dry run
                fix_results["successful_fixes"] += 1

            fix_results["fixes"].append(fix_record)

        logger.info(f"Fix operation complete: {fix_results}")
        return fix_results

    async def delete_uncorrectable_relationships(
        self,
        audit_results: dict[str, Any],
        dry_run: bool = True,
        require_confirmation: bool = True,
    ) -> dict[str, Any]:
        """
        Delete relationships that cannot be automatically corrected.

        Args:
            audit_results: Results from audit_existing_relationships
            dry_run: If True, only simulate deletions without making changes
            require_confirmation: If True, require explicit confirmation for each deletion

        Returns:
            Dictionary containing deletion results
        """
        uncorrectable = audit_results.get("uncorrectable", [])
        logger.info(
            f"Processing {len(uncorrectable)} uncorrectable relationships (dry_run={dry_run})"
        )

        deletion_results = {
            "processed": 0,
            "successful_deletions": 0,
            "skipped_by_user": 0,
            "failed_deletions": 0,
            "deletions": [],
        }

        for rel_data in uncorrectable:
            rel_id = rel_data["rel_id"]
            relationship_desc = (
                f"{rel_data['subject_type']}:{rel_data['subject_name']} "
                f"{rel_data['relationship_type']} "
                f"{rel_data['object_type']}:{rel_data['object_name']}"
            )

            deletion_results["processed"] += 1

            deletion_record = {
                "rel_id": rel_id,
                "relationship": relationship_desc,
                "errors": rel_data["errors"],
                "success": False,
            }

            # Ask for confirmation if required
            if require_confirmation and not dry_run:
                print(f"\nUncorrectable relationship: {relationship_desc}")
                print(f"Errors: {', '.join(rel_data['errors'])}")
                response = input("Delete this relationship? (y/n/q to quit): ").lower()

                if response == "q":
                    logger.info("User requested to quit deletion process")
                    break
                elif response != "y":
                    deletion_results["skipped_by_user"] += 1
                    logger.info(f"Skipped deletion of: {relationship_desc}")
                    continue

            if not dry_run:
                try:
                    # Delete the relationship
                    delete_query = """
                    MATCH ()-[r]->()
                    WHERE elementId(r) = $rel_id
                    DELETE r
                    """

                    await neo4j_manager.execute_write_query(
                        delete_query, {"rel_id": rel_id}
                    )

                    deletion_record["success"] = True
                    deletion_results["successful_deletions"] += 1

                    logger.info(
                        f"Deleted uncorrectable relationship: {relationship_desc}"
                    )

                except Exception as e:
                    deletion_record["error"] = str(e)
                    deletion_results["failed_deletions"] += 1
                    logger.error(
                        f"Failed to delete relationship {rel_id}: {e}", exc_info=True
                    )
            else:
                # Dry run - just log what would be done
                logger.info(f"[DRY RUN] Would delete: {relationship_desc}")
                deletion_record["success"] = True
                deletion_results["successful_deletions"] += 1

            deletion_results["deletions"].append(deletion_record)

        logger.info(f"Deletion operation complete: {deletion_results}")
        return deletion_results

    async def restore_from_backup(self, backup_file: Path) -> dict[str, Any]:
        """
        Restore relationships from a backup file.

        Warning: This will delete all current relationships and restore from backup.
        Use with extreme caution.

        Args:
            backup_file: Path to the backup file

        Returns:
            Dictionary containing restore results
        """
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")

        logger.warning(f"DESTRUCTIVE OPERATION: Restoring from backup {backup_file}")

        # Load backup data
        with open(backup_file, encoding="utf-8") as f:
            backup_data = json.load(f)

        relationships = backup_data["relationships"]
        logger.info(f"Loaded backup with {len(relationships)} relationships")

        restore_results = {
            "backup_file": str(backup_file),
            "relationships_in_backup": len(relationships),
            "current_relationships_deleted": 0,
            "relationships_restored": 0,
            "failed_restorations": 0,
        }

        try:
            # First, delete all current relationships
            delete_query = "MATCH ()-[r]->() DELETE r"
            delete_result = await neo4j_manager.execute_write_query(delete_query)
            restore_results["current_relationships_deleted"] = (
                delete_result[0].get("deletions", 0) if delete_result else 0
            )

            logger.info(
                f"Deleted {restore_results['current_relationships_deleted']} existing relationships"
            )

            # Restore relationships from backup
            for rel_data in relationships:
                try:
                    # Reconstruct the relationship
                    restore_query = """
                    MATCH (s {name: $subject_name}), (o {name: $object_name})
                    WHERE $subject_id IN [s.id, s.name] AND $object_id IN [o.id, o.name]
                    CREATE (s)-[r:%s]->(o)
                    SET r = $rel_properties
                    """ % rel_data["relationship_type"]

                    params = {
                        "subject_name": rel_data["subject_name"],
                        "subject_id": rel_data["subject_id"],
                        "object_name": rel_data["object_name"],
                        "object_id": rel_data["object_id"],
                        "rel_properties": rel_data["rel_properties"],
                    }

                    await neo4j_manager.execute_write_query(restore_query, params)
                    restore_results["relationships_restored"] += 1

                except Exception as e:
                    restore_results["failed_restorations"] += 1
                    logger.error(f"Failed to restore relationship: {rel_data}: {e}")

            logger.info(f"Restore complete: {restore_results}")
            return restore_results

        except Exception as e:
            logger.error(f"Critical error during restore: {e}", exc_info=True)
            raise

    async def run_full_migration(
        self,
        dry_run: bool = True,
        min_confidence: float = 0.5,
        create_backup: bool = True,
        interactive: bool = True,
    ) -> dict[str, Any]:
        """
        Run the complete relationship migration process.

        Args:
            dry_run: If True, only simulate changes
            min_confidence: Minimum confidence for automatic corrections
            create_backup: Whether to create a backup before making changes
            interactive: Whether to ask for user confirmation on deletions

        Returns:
            Complete migration results
        """
        migration_results = {
            "started_at": datetime.now().isoformat(),
            "dry_run": dry_run,
            "steps": {},
        }

        try:
            # Step 1: Create backup if requested
            if create_backup and not dry_run:
                backup_file = await self.create_backup()
                migration_results["steps"]["backup"] = {"backup_file": str(backup_file)}

            # Step 2: Audit existing relationships
            logger.info("Step 1: Auditing existing relationships...")
            audit_results = await self.audit_existing_relationships()
            migration_results["steps"]["audit"] = audit_results["summary"]

            # Step 3: Fix correctable relationships
            logger.info("Step 2: Fixing correctable relationships...")
            fix_results = await self.fix_correctable_relationships(
                audit_results, dry_run=dry_run, min_confidence=min_confidence
            )
            migration_results["steps"]["corrections"] = fix_results

            # Step 4: Handle uncorrectable relationships
            if not dry_run and interactive:
                logger.info("Step 3: Handling uncorrectable relationships...")
                deletion_results = await self.delete_uncorrectable_relationships(
                    audit_results, dry_run=dry_run, require_confirmation=interactive
                )
                migration_results["steps"]["deletions"] = deletion_results

            migration_results["completed_at"] = datetime.now().isoformat()
            migration_results["success"] = True

            logger.info(f"Migration complete: {migration_results}")
            return migration_results

        except Exception as e:
            migration_results["error"] = str(e)
            migration_results["success"] = False
            logger.error(f"Migration failed: {e}", exc_info=True)
            raise


async def main():
    """Command-line interface for the migration tool."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate relationships to enforce constraints"
    )
    parser.add_argument(
        "--audit-only", action="store_true", help="Only audit, don't fix"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate changes without applying them"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for corrections",
    )
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup")
    parser.add_argument(
        "--non-interactive", action="store_true", help="Run without user prompts"
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of relationships to process"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    migration_tool = RelationshipMigrationTool()

    try:
        if args.audit_only:
            # Only run audit
            results = await migration_tool.audit_existing_relationships(
                limit=args.limit
            )
            print("\n=== AUDIT RESULTS ===")
            print(json.dumps(results["summary"], indent=2))
        else:
            # Run full migration
            results = await migration_tool.run_full_migration(
                dry_run=args.dry_run,
                min_confidence=args.min_confidence,
                create_backup=not args.no_backup,
                interactive=not args.non_interactive,
            )
            print("\n=== MIGRATION RESULTS ===")
            print(json.dumps(results, indent=2, default=str))

    except Exception as e:
        logger.error(f"Migration tool failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    asyncio.run(main())
