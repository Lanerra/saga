# core/entity_embedding_service.py
"""Persist per-entity embedding vectors for semantic operations.

This module computes embedding inputs for knowledge-graph entities and builds
Cypher statements that update embedding properties when the entity's embedding
text has changed.

Notes:
    This module performs Neo4j reads (to compare stored text hashes) and LLM I/O
    (to generate embedding vectors) when embedding persistence is enabled.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

import config
from core.db_manager import neo4j_manager
from core.llm_interface_refactored import llm_service

if TYPE_CHECKING:
    from models.kg_models import CharacterProfile, WorldItem

logger = structlog.get_logger(__name__)


def compute_entity_embedding_text(*, name: str, description: str, category: str) -> str:
    """Build the canonical input text for an entity embedding.

    Args:
        name: Entity name.
        description: Entity description text.
        category: Optional category/subtype descriptor (empty for characters).

    Returns:
        Concatenated text used as the embedding input.
    """
    parts = []
    if isinstance(name, str) and name.strip():
        parts.append(name.strip())
    if isinstance(category, str) and category.strip():
        parts.append(category.strip())
    if isinstance(description, str) and description.strip():
        parts.append(description.strip())
    return "\n".join(parts)


def compute_entity_embedding_text_hash(text: str) -> str:
    """Hash embedding input text for change detection.

    Args:
        text: Embedding input text.

    Returns:
        Stable SHA1 hash used to detect whether an embedding needs recomputation.

    Raises:
        ValueError: If `text` is empty.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("entity embedding text must be a non-empty string")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


async def build_entity_embedding_update_statements(
    *,
    characters: list[CharacterProfile],
    world_items: list[WorldItem],
) -> list[tuple[str, dict[str, Any]]]:
    """Build Cypher statements that update entity embedding properties.

    The returned statements are intended to be executed in the same batch
    transaction as other entity upserts.

    Args:
        characters: Character profiles to consider for embedding updates.
        world_items: World items (locations/items/events) to consider for embedding updates.

    Returns:
        List of `(cypher, params)` tuples to update embeddings for entities whose
        stored text hash differs from the newly computed hash.

    Raises:
        ValueError: If embedding persistence is enabled but required config properties
            are missing, or if the embedding batch returns a mismatched result length.

    Notes:
        This function performs Neo4j reads to fetch existing text hashes and LLM I/O
        to generate embeddings for changed entities.
    """
    if not config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE:
        return []

    vector_property = config.ENTITY_EMBEDDING_VECTOR_PROPERTY
    text_hash_property = config.ENTITY_EMBEDDING_TEXT_HASH_PROPERTY
    model_property = config.ENTITY_EMBEDDING_MODEL_PROPERTY

    if not vector_property or not text_hash_property or not model_property:
        raise ValueError("entity embedding property configuration is missing")

    statements: list[tuple[str, dict[str, Any]]] = []

    character_names = sorted({c.name for c in characters if c and c.name})
    world_item_ids = sorted({w.id for w in world_items if w and w.id})

    character_existing_hash: dict[str, str] = {}
    world_existing_hash: dict[str, str] = {}

    if character_names:
        query = f"""
        MATCH (c:Character)
        WHERE c.name IN $names
        RETURN c.name AS key, c.`{text_hash_property}` AS existing_hash
        """
        results = await neo4j_manager.execute_read_query(query, {"names": character_names})

        character_existing_hash = {}
        for record in results or []:
            if not isinstance(record, dict):
                continue
            key = record.get("key")
            if not isinstance(key, str) or not key:
                continue
            existing_hash = record.get("existing_hash")
            if not isinstance(existing_hash, str) or not existing_hash:
                continue
            character_existing_hash[key] = existing_hash

    if world_item_ids:
        query = f"""
        MATCH (w)
        WHERE (w:Location OR w:Item OR w:Event)
          AND w.id IN $ids
        RETURN w.id AS key, w.`{text_hash_property}` AS existing_hash
        """
        results = await neo4j_manager.execute_read_query(query, {"ids": world_item_ids})

        world_existing_hash = {}
        for record in results or []:
            if not isinstance(record, dict):
                continue
            key = record.get("key")
            if not isinstance(key, str) or not key:
                continue
            existing_hash = record.get("existing_hash")
            if not isinstance(existing_hash, str) or not existing_hash:
                continue
            world_existing_hash[key] = existing_hash

    embedding_inputs: list[dict[str, Any]] = []
    embedding_texts: list[str] = []

    for char in characters:
        name = char.name
        description = char.description or ""
        embedding_text = compute_entity_embedding_text(name=name, category="", description=description)
        embedding_hash = compute_entity_embedding_text_hash(embedding_text)
        existing_hash = character_existing_hash.get(name)

        if existing_hash == embedding_hash:
            continue

        embedding_inputs.append(
            {
                "entity_kind": "character",
                "node_key": name,
                "embedding_hash": embedding_hash,
                "embedding_text": embedding_text,
            }
        )
        embedding_texts.append(embedding_text)

    for item in world_items:
        stable_id = item.id
        name = item.name or ""
        category = item.category or ""
        description = item.description or ""
        embedding_text = compute_entity_embedding_text(name=name, category=category, description=description)
        embedding_hash = compute_entity_embedding_text_hash(embedding_text)
        existing_hash = world_existing_hash.get(stable_id)

        if existing_hash == embedding_hash:
            continue

        embedding_inputs.append(
            {
                "entity_kind": "world_item",
                "node_key": stable_id,
                "embedding_hash": embedding_hash,
                "embedding_text": embedding_text,
            }
        )
        embedding_texts.append(embedding_text)

    if not embedding_inputs:
        return []

    embeddings = await llm_service.async_get_embeddings_batch(embedding_texts)

    if len(embeddings) != len(embedding_inputs):
        raise ValueError("embedding batch result length mismatch")

    for index, embedding_input in enumerate(embedding_inputs):
        embedding = embeddings[index]
        if embedding is None:
            logger.warning(
                "entity embedding generation returned None",
                entity_kind=embedding_input["entity_kind"],
                node_key=embedding_input["node_key"],
            )
            continue

        embedding_array = embedding if isinstance(embedding, np.ndarray) else np.array(embedding)
        embedding_list = neo4j_manager.embedding_to_list(embedding_array)
        if not embedding_list:
            continue

        if embedding_input["entity_kind"] == "character":
            cypher = f"""
            MATCH (c:Character {{name: $name}})
            SET c.`{vector_property}` = $vector,
                c.`{text_hash_property}` = $text_hash,
                c.`{model_property}` = $model,
                c.last_updated = timestamp()
            """
            params = {
                "name": embedding_input["node_key"],
                "vector": embedding_list,
                "text_hash": embedding_input["embedding_hash"],
                "model": config.EMBEDDING_MODEL,
            }
            statements.append((cypher, params))
            continue

        if embedding_input["entity_kind"] == "world_item":
            cypher = f"""
            MATCH (w)
            WHERE (w:Location OR w:Item OR w:Event)
              AND w.id = $id
            SET w.`{vector_property}` = $vector,
                w.`{text_hash_property}` = $text_hash,
                w.`{model_property}` = $model,
                w.last_updated = timestamp()
            """
            params = {
                "id": embedding_input["node_key"],
                "vector": embedding_list,
                "text_hash": embedding_input["embedding_hash"],
                "model": config.EMBEDDING_MODEL,
            }
            statements.append((cypher, params))
            continue

        raise ValueError("unsupported entity_kind for embedding update")

    logger.info(
        "Built entity embedding update statements",
        statements=len(statements),
        candidates=len(embedding_inputs),
        characters=len(character_names),
        world_items=len(world_item_ids),
    )

    return statements
