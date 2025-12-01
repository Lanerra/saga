# reset_neo4j.py
import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path

import structlog

import config
from core.db_manager import Neo4jManagerSingleton

structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%m/%d/%Y, %H:%M"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

neo4j_manager_instance = Neo4jManagerSingleton()


async def reset_neo4j_database_async(uri, user, password, confirm=False):
    if not confirm:
        response = input(
            "⚠️ WARNING: This will delete ALL data, ALL user‑defined constraints, "
            "and ALL user‑defined indexes in the Neo4j database. This is a destructive "
            "operation. Continue? (y/N): "
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
        # Retry logic for connection
        max_retries = 5
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Connecting to Neo4j database at {effective_uri} "
                    f"(Attempt {attempt + 1}/{max_retries})..."
                )
                await neo4j_manager_instance.connect()
                logger.info("Successfully connected to Neo4j.")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Connection failed: {e}. Retrying in {retry_delay} seconds..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("Could not connect to Neo4j after multiple retries.")
                    raise

        # Get current node count using public helper
        result = await neo4j_manager_instance.execute_read_query(
            "MATCH (n) RETURN count(n) as count"
        )
        node_count = result[0]["count"] if result else 0
        logger.info(f"Current database has {node_count} nodes.")

        logger.info("Resetting database data (nodes and relationships)...")
        start_time = time.time()

        # Deletion logic
        async def _delete_all_nodes() -> bool:
            max_retries = 3
            backoff = 2
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"Delete attempt {attempt}/{max_retries}")
                    assert neo4j_manager_instance.driver is not None
                    async with neo4j_manager_instance.driver.session(
                        database=config.NEO4J_DATABASE
                    ) as session:
                        result = await session.run(
                            "MATCH (n) DETACH DELETE n RETURN count(n) as deleted_nodes_total"
                        )
                        single_result = await result.single()
                        nodes_deleted_total = single_result.get(
                            "deleted_nodes_total", 0
                        )
                        logger.info(f"  Total {nodes_deleted_total} nodes deleted.")
                    async with neo4j_manager_instance.driver.session(
                        database=config.NEO4J_DATABASE
                    ) as session:
                        rv = await session.run(
                            "MATCH (n) RETURN count(n) as remaining_nodes"
                        )
                        rem_res = await rv.single()
                        remaining = rem_res.get("remaining_nodes", 0)
                        if remaining == 0:
                            return True
                        logger.warning(
                            f"  {remaining} nodes still remain after deletion attempt {attempt}"
                        )
                    if attempt == max_retries:
                        return False
                    await asyncio.sleep(backoff)
                except Exception as exc:
                    logger.error(
                        f"  Exception on delete attempt {attempt}: {exc}", exc_info=True
                    )
                    if attempt == max_retries:
                        return False
                    await asyncio.sleep(backoff)
            return False

        deletion_success = await _delete_all_nodes()
        if not deletion_success:
            logger.critical(
                "Abort: could not delete all nodes after retries. Leaving database intact."
            )
            raise RuntimeError("Failed to wipe all nodes")

        # Drop constraints
        logger.info("Attempting to drop ALL user‑defined constraints...")
        assert neo4j_manager_instance.driver is not None
        async with neo4j_manager_instance.driver.session(
            database=config.NEO4J_DATABASE
        ) as session:
            constraints_result = await session.run("SHOW CONSTRAINTS YIELD name")
            constraints_to_drop = [
                record["name"]
                for record in await constraints_result.data()
                if record["name"]
            ]
            if not constraints_to_drop:
                logger.info("   No user‑defined constraints found to drop.")
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
                        if tx and not tx.closed():  # type: ignore[union-attr]
                            await tx.rollback()
                        logger.warning(
                            f"   Note: Could not drop constraint '{constraint_name}': {e_constraint}"
                        )

        # Drop indexes
        logger.info(
            "Attempting to drop ALL user‑defined indexes (excluding system indexes if identifiable)..."
        )
        assert neo4j_manager_instance.driver is not None
        async with neo4j_manager_instance.driver.session(
            database=config.NEO4J_DATABASE
        ) as session:
            indexes_result = await session.run("SHOW INDEXES YIELD name, type")
            indexes_to_drop_info = await indexes_result.data()
            if not indexes_to_drop_info:
                logger.info("   No user‑defined indexes found to drop.")
            else:
                for index_info in indexes_to_drop_info:
                    index_name = index_info.get("name")
                    index_type = index_info.get("type", "").upper()
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
                            if tx and not tx.closed():  # type: ignore[union-attr]
                                await tx.rollback()
                            logger.warning(
                                f"   Note: Could not drop index '{index_name}': {e_index}"
                            )
                    elif index_name:
                        logger.info(
                            f"   Skipping potential system/lookup index: {index_name} (type: {index_type})"
                        )

        # Migration script
        elapsed_time = time.time() - start_time
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
            f"✅ Database data, all user‑defined constraints, and relevant user‑defined indexes reset/dropped in {elapsed_time:.2f} seconds."
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
        # Reset config values
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
