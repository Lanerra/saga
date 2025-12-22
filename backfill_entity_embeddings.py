import argparse
import asyncio
import time
from typing import Any

import numpy as np
import structlog

import config
from core.db_manager import Neo4jManagerSingleton
from core.entity_embedding_service import (
    compute_entity_embedding_text,
    compute_entity_embedding_text_hash,
)
from core.llm_interface_refactored import llm_service

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


async def _fetch_character_candidates(
    neo4j_manager: Neo4jManagerSingleton,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    query = f"""
    MATCH (c:Character)
    WHERE c.name IS NOT NULL
      AND (
        c.`{config.ENTITY_EMBEDDING_TEXT_HASH_PROPERTY}` IS NULL
        OR c.`{config.ENTITY_EMBEDDING_VECTOR_PROPERTY}` IS NULL
      )
    RETURN
      c.name AS name,
      coalesce(toString(c.description), '') AS description
    ORDER BY toLower(toString(c.name))
    LIMIT $limit
    """
    results = await neo4j_manager.execute_read_query(query, {"limit": int(limit)})
    return [r for r in results if r and r.get("name")]


async def _fetch_world_item_candidates(
    neo4j_manager: Neo4jManagerSingleton,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    query = f"""
    MATCH (w)
    WHERE (w:Location OR w:Item OR w:Event)
      AND w.id IS NOT NULL
      AND (
        w.`{config.ENTITY_EMBEDDING_TEXT_HASH_PROPERTY}` IS NULL
        OR w.`{config.ENTITY_EMBEDDING_VECTOR_PROPERTY}` IS NULL
      )
    RETURN
      w.id AS id,
      coalesce(toString(w.name), '') AS name,
      coalesce(toString(w.category), '') AS category,
      coalesce(toString(w.description), '') AS description
    ORDER BY toLower(toString(w.name)), toString(w.id)
    LIMIT $limit
    """
    results = await neo4j_manager.execute_read_query(query, {"limit": int(limit)})
    return [r for r in results if r and r.get("id")]


async def _write_character_embeddings(
    neo4j_manager: Neo4jManagerSingleton,
    *,
    records: list[dict[str, Any]],
    vectors: list[list[float]],
    text_hashes: list[str],
) -> None:
    query = f"""
    UNWIND $rows AS row
    MATCH (c:Character {{name: row.name}})
    SET c.`{config.ENTITY_EMBEDDING_VECTOR_PROPERTY}` = row.vector,
        c.`{config.ENTITY_EMBEDDING_TEXT_HASH_PROPERTY}` = row.text_hash,
        c.`{config.ENTITY_EMBEDDING_MODEL_PROPERTY}` = $model,
        c.last_updated = timestamp()
    """
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "name": record["name"],
                "vector": vectors[index],
                "text_hash": text_hashes[index],
            }
        )
    await neo4j_manager.execute_write_query(
        query,
        {
            "rows": rows,
            "model": config.EMBEDDING_MODEL,
        },
    )


async def _write_world_item_embeddings(
    neo4j_manager: Neo4jManagerSingleton,
    *,
    records: list[dict[str, Any]],
    vectors: list[list[float]],
    text_hashes: list[str],
) -> None:
    query = f"""
    UNWIND $rows AS row
    MATCH (w)
    WHERE (w:Location OR w:Item OR w:Event)
      AND w.id = row.id
    SET w.`{config.ENTITY_EMBEDDING_VECTOR_PROPERTY}` = row.vector,
        w.`{config.ENTITY_EMBEDDING_TEXT_HASH_PROPERTY}` = row.text_hash,
        w.`{config.ENTITY_EMBEDDING_MODEL_PROPERTY}` = $model,
        w.last_updated = timestamp()
    """
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "id": record["id"],
                "vector": vectors[index],
                "text_hash": text_hashes[index],
            }
        )
    await neo4j_manager.execute_write_query(
        query,
        {
            "rows": rows,
            "model": config.EMBEDDING_MODEL,
        },
    )


async def _backfill_batch(
    neo4j_manager: Neo4jManagerSingleton,
    *,
    character_records: list[dict[str, Any]],
    world_item_records: list[dict[str, Any]],
    dry_run: bool,
) -> int:
    embedding_texts: list[str] = []
    embedding_targets: list[dict[str, Any]] = []

    for record in character_records:
        embedding_text = compute_entity_embedding_text(
            name=str(record["name"]),
            category="",
            description=str(record.get("description") or ""),
        )
        embedding_targets.append({"kind": "character", "record": record, "text": embedding_text})
        embedding_texts.append(embedding_text)

    for record in world_item_records:
        embedding_text = compute_entity_embedding_text(
            name=str(record.get("name") or ""),
            category=str(record.get("category") or ""),
            description=str(record.get("description") or ""),
        )
        embedding_targets.append({"kind": "world_item", "record": record, "text": embedding_text})
        embedding_texts.append(embedding_text)

    if not embedding_targets:
        return 0

    embeddings = await llm_service.async_get_embeddings_batch(embedding_texts)
    if len(embeddings) != len(embedding_targets):
        raise ValueError("embedding batch result length mismatch")

    character_vectors: list[list[float]] = []
    character_hashes: list[str] = []
    character_records_to_write: list[dict[str, Any]] = []

    world_vectors: list[list[float]] = []
    world_hashes: list[str] = []
    world_records_to_write: list[dict[str, Any]] = []

    for index, target in enumerate(embedding_targets):
        embedding = embeddings[index]
        if embedding is None:
            continue

        embedding_array = embedding if isinstance(embedding, np.ndarray) else np.array(embedding)
        vector_list = neo4j_manager.embedding_to_list(embedding_array)
        if not vector_list:
            continue

        text = target["text"]
        text_hash = compute_entity_embedding_text_hash(text)

        if target["kind"] == "character":
            character_records_to_write.append(target["record"])
            character_vectors.append(vector_list)
            character_hashes.append(text_hash)
            continue

        if target["kind"] == "world_item":
            world_records_to_write.append(target["record"])
            world_vectors.append(vector_list)
            world_hashes.append(text_hash)
            continue

        raise ValueError("unsupported embedding target kind")

    if dry_run:
        return len(character_records_to_write) + len(world_records_to_write)

    if character_records_to_write:
        await _write_character_embeddings(
            neo4j_manager,
            records=character_records_to_write,
            vectors=character_vectors,
            text_hashes=character_hashes,
        )

    if world_records_to_write:
        await _write_world_item_embeddings(
            neo4j_manager,
            records=world_records_to_write,
            vectors=world_vectors,
            text_hashes=world_hashes,
        )

    return len(character_records_to_write) + len(world_records_to_write)


async def backfill_entity_embeddings_async(
    *,
    uri: str | None,
    user: str | None,
    password: str | None,
    limit: int,
    batch_size: int,
    dry_run: bool,
) -> None:
    manager = Neo4jManagerSingleton()

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
        logger.info("Connecting to Neo4j", uri=effective_uri, user=effective_user)
        await manager.connect()

        logger.info("Ensuring schema (includes vector indexes)")
        await manager.create_db_schema()

        remaining = int(limit)
        total_backfilled = 0
        start_time = time.time()

        while remaining > 0:
            fetch_limit = min(batch_size, remaining)

            character_records = await _fetch_character_candidates(manager, limit=fetch_limit)
            world_item_records = await _fetch_world_item_candidates(manager, limit=fetch_limit)

            if not character_records and not world_item_records:
                break

            backfilled = await _backfill_batch(
                manager,
                character_records=character_records,
                world_item_records=world_item_records,
                dry_run=dry_run,
            )

            total_backfilled += backfilled
            remaining -= fetch_limit

            logger.info(
                "Backfill batch complete",
                batch_size=fetch_limit,
                backfilled=backfilled,
                total_backfilled=total_backfilled,
                remaining_budget=remaining,
                dry_run=dry_run,
            )

            if backfilled == 0:
                break

        elapsed = time.time() - start_time
        logger.info("Entity embedding backfill complete", total_backfilled=total_backfilled, seconds=f"{elapsed:.2f}", dry_run=dry_run)

    finally:
        if manager.driver:
            await manager.close()
        config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD = (
            original_uri,
            original_user,
            original_pass,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill Neo4j entity embeddings for Character/Location/Item/Event nodes.")
    parser.add_argument("--uri", default=None, help=f"Neo4j connection URI (default: {config.NEO4J_URI})")
    parser.add_argument("--user", default=None, help=f"Neo4j username (default: {config.NEO4J_USER})")
    parser.add_argument("--password", default=None, help=f"Neo4j password (default: {config.NEO4J_PASSWORD})")
    parser.add_argument("--limit", type=int, default=5_000, help="Maximum number of entities to process")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for embedding generation and writes")
    parser.add_argument("--dry-run", action="store_true", help="Compute embeddings and report counts without writing")

    args = parser.parse_args()

    asyncio.run(
        backfill_entity_embeddings_async(
            uri=args.uri,
            user=args.user,
            password=args.password,
            limit=args.limit,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    )
