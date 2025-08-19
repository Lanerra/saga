# reset_neo4j.py
import argparse
import asyncio  # Required to call async methods
import logging  # Added logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any  # Added for type hints

import config  # To get default URI, user, pass if not provided via args
from core.db_manager import Neo4jManagerSingleton  # Use the singleton

# Configure a basic logger for this script
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Create an instance of the manager to use its methods
neo4j_manager_instance = Neo4jManagerSingleton()


async def reset_neo4j_database_async(uri, user, password, confirm=False):
    """
    Asynchronously resets a Neo4j database by:
    1. Removing all nodes and relationships.
    2. Dropping ALL user-defined constraints.
    3. Dropping ALL user-defined indexes.
    """
    if not confirm:
        response = input(
            "⚠️ WARNING: This will delete ALL data, ALL user-defined constraints, and ALL user-defined indexes "
            "in the Neo4j database. This is a destructive operation. Continue? (y/N): "
        )
        if response.lower() not in ["y", "yes"]:
            print("Operation cancelled.")
            return False

    effective_uri = uri or config.NEO4J_URI
    effective_user = user or config.NEO4J_USER
    effective_password = password or config.NEO4J_PASSWORD

    original_uri, original_user, original_pass = (
        config.NEO4J_URI,
        config.NEO4J_USER,
        config.NEO4J_PASSWORD,
    )
    config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD = (
        effective_uri,
        effective_user,
        effective_password,
    )

    try:
        # --- START: Added Connection Retry Logic ---
        max_retries = 5
        retry_delay = 5  # seconds
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Connecting to Neo4j database at {effective_uri} (Attempt {attempt + 1}/{max_retries})..."
                )
                await neo4j_manager_instance.connect()
                logger.info("Successfully connected to Neo4j.")
                break  # Exit loop on successful connection
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Connection failed: {e}. Retrying in {retry_delay} seconds..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("Could not connect to Neo4j after multiple retries.")
                    raise  # Re-raise the last exception if all retries fail
        # --- END: Added Connection Retry Logic ---

        async with neo4j_manager_instance.driver.session(
            database=config.NEO4J_DATABASE
        ) as session:  # type: ignore
            result = await session.run("MATCH (n) RETURN count(n) as count")
            single_result = await result.single()
            node_count = single_result["count"] if single_result else 0
            logger.info(f"Current database has {node_count} nodes.")

        logger.info("Resetting database data (nodes and relationships)...")
        start_time = time.time()

        nodes_deleted_total = 0
        while True:
            async with neo4j_manager_instance.driver.session(
                database=config.NEO4J_DATABASE
            ) as session:  # type: ignore
                tx = await session.begin_transaction()
                result = await tx.run(
                    "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) as deleted_nodes_batch"
                )
                single_result_batch = await result.single()
                await tx.commit()
                deleted_in_batch = (
                    single_result_batch["deleted_nodes_batch"]
                    if single_result_batch
                    else 0
                )
                nodes_deleted_total += deleted_in_batch
                if deleted_in_batch == 0:
                    break
                logger.info(f"   Deleted {deleted_in_batch} nodes in this batch...")
        logger.info(f"   Total {nodes_deleted_total} nodes deleted.")

        logger.info("Attempting to drop ALL user-defined constraints...")
        async with neo4j_manager_instance.driver.session(
            database=config.NEO4J_DATABASE
        ) as session:  # type: ignore
            constraints_result = await session.run("SHOW CONSTRAINTS YIELD name")
            constraints_to_drop: list[str] = [
                record["name"]
                for record in await constraints_result.data()
                if record["name"]
            ]

            if not constraints_to_drop:
                logger.info("   No user-defined constraints found to drop.")
            else:
                for constraint_name in constraints_to_drop:
                    try:
                        logger.info(
                            f"   Attempting to drop constraint: {constraint_name}"
                        )
                        tx = await session.begin_transaction()
                        await tx.run(f"DROP CONSTRAINT {constraint_name} IF EXISTS")
                        await tx.commit()
                        logger.info(
                            f"      Dropped constraint '{constraint_name}' (or it didn't exist)."
                        )
                    except Exception as e_constraint:
                        if tx and not tx.closed():  # type: ignore
                            await tx.rollback()
                        logger.warning(
                            f"   Note: Could not drop constraint '{constraint_name}': {e_constraint}"
                        )

        logger.info(
            "Attempting to drop ALL user-defined indexes (excluding system indexes if identifiable)..."
        )
        async with neo4j_manager_instance.driver.session(
            database=config.NEO4J_DATABASE
        ) as session:  # type: ignore
            # Query for indexes, trying to filter out system ones if possible (type might not always be 'SYSTEM_LOOKUP')
            # The most reliable way is usually by name patterns if system indexes have those, or by excluding known types.
            # For now, we will attempt to drop all that are not of type 'LOOKUP' for node_label_property which could be old schema index.
            # And also not 'RANGE' or 'POINT' unless we are sure SAGA doesn't use them (it mostly uses BTREE for properties and VECTOR).
            # A simpler approach for a full reset is to try dropping all and let `IF EXISTS` handle it.
            indexes_result = await session.run("SHOW INDEXES YIELD name, type")
            indexes_to_drop_info: list[dict[str, Any]] = await indexes_result.data()

            if not indexes_to_drop_info:
                logger.info("   No user-defined indexes found to drop.")
            else:
                for index_info in indexes_to_drop_info:
                    index_name = index_info.get("name")
                    index_type = index_info.get(
                        "type", ""
                    ).upper()  # VECTOR, BTREE, TEXT, FULLTEXT, POINT, RANGE, LOOKUP

                    # Heuristic: Avoid dropping system/lookup indexes that Neo4j might manage internally for constraints.
                    # Typically, SAGA creates BTREE (for property indexes) and VECTOR indexes.
                    # Other types might be from older versions or manual creation.
                    # `tokenLookup` is often for fulltext schema indexes.
                    if (
                        index_name
                        and "tokenLookup" not in index_name.lower()
                        and "system" not in index_type.lower()
                    ):
                        try:
                            logger.info(
                                f"   Attempting to drop index: {index_name} (type: {index_type})"
                            )
                            tx = await session.begin_transaction()
                            await tx.run(f"DROP INDEX {index_name} IF EXISTS")
                            await tx.commit()
                            logger.info(
                                f"      Dropped index '{index_name}' (or it didn't exist)."
                            )
                        except Exception as e_index:
                            if tx and not tx.closed():  # type: ignore
                                await tx.rollback()
                            logger.warning(
                                f"   Note: Could not drop index '{index_name}': {e_index}"
                            )
                    elif index_name:
                        logger.info(
                            f"   Skipping potential system/lookup index: {index_name} (type: {index_type})"
                        )

        elapsed_time = time.time() - start_time
        # Run migration to fix any legacy WorldElements with missing core fields
        logger.info("Running migration for legacy WorldElements...")
        migration_script = (
            Path(__file__).parent
            / "initialization"
            / "migrate_legacy_world_elements.py"
        )
        if migration_script.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(migration_script)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                logger.info(f"Migration output: {result.stdout}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Migration failed: {e.stderr}")
        else:
            logger.warning(f"Migration script not found at {migration_script}")

        logger.info(
            f"✅ Database data, all user-defined constraints, and relevant user-defined indexes reset/dropped in {elapsed_time:.2f} seconds."
        )
        logger.info(
            "   The SAGA system will attempt to recreate its necessary schema on the next run."
        )

        return True

    except Exception as e:
        logger.error(f"❌ Error resetting database: {e}", exc_info=True)
        return False

    finally:
        if neo4j_manager_instance.driver:
            await neo4j_manager_instance.close()
            logger.info("Connection closed.")
        config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD = (
            original_uri,
            original_user,
            original_pass,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reset a Neo4j database by removing all data, user constraints, and user indexes."
    )
    parser.add_argument(
        "--uri",
        default=None,
        help=f"Neo4j connection URI (default: {config.NEO4J_URI})",
    )
    parser.add_argument(
        "--user", default=None, help=f"Neo4j username (default: {config.NEO4J_USER})"
    )
    parser.add_argument(
        "--password",
        default=None,
        help=f"Neo4j password (default: {config.NEO4J_PASSWORD})",
    )
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    asyncio.run(
        reset_neo4j_database_async(
            uri=args.uri, user=args.user, password=args.password, confirm=args.force
        )
    )
