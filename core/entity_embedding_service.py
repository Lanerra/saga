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

import structlog

import config
from core.embedding_contract import embedding_identity, validate_embedding
from core.service_context import get_services
from models.kg_constants import WORLD_ITEM_CANONICAL_LABELS
from utils import classify_category_label

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

    candidates: list[dict[str, Any]] = []
    for char in characters:
        candidates.append({"label": "Character", "id": char.id, "name": char.name,
                           "category": "", "description": char.personality_description or ""})
    for item in world_items:
        label = classify_category_label(item.category)
        if label not in WORLD_ITEM_CANONICAL_LABELS:
            label = "Item"
        candidates.append({"label": label, "id": item.id, "name": item.name,
                           "category": item.category or "", "description": item.description or ""})
    if not candidates:
        return []

    for index, candidate in enumerate(candidates):
        if not isinstance(candidate["name"], str) or not candidate["name"].strip():
            raise ValueError("Invalid canonical entity name")
        identifier = candidate["id"]
        if not isinstance(identifier, str) or (identifier and not identifier.strip()):
            raise ValueError("Invalid canonical entity ID")
        candidate["id"] = identifier or None
        candidate["index"] = index

    # The empty model ID denotes name-only resolution, as in the native upsert.
    # Resolve again inside the write transaction so first creation is supported.
    match_candidates = """
        OPTIONAL MATCH (candidate)
        WHERE entity.label IN labels(candidate)
          AND CASE WHEN entity.id IS NOT NULL THEN candidate.id = entity.id
                   ELSE toLower(trim(candidate.name)) = toLower(trim(entity.name)) END
        WITH entity, collect(candidate) AS candidates
    """
    query = f"""
        UNWIND $entities AS entity
        {match_candidates}
        CALL apoc.util.validate(size(candidates) > 1, 'Ambiguous canonical entity', [])
        WITH entity, head(candidates) AS found
        CALL apoc.util.validate(found IS NOT NULL AND (found.id IS NULL OR trim(found.id) = ''),
                                'Canonical entity has no stable ID', [])
        RETURN entity.index AS key, found.id AS id, found.`{text_hash_property}` AS existing_hash,
               found.`{model_property}` AS existing_model,
               found.`{model_property}_identity` AS existing_identity,
               found.`{vector_property}` AS existing_vector
    """
    results = await get_services().database.execute_read_query(query, {"entities": candidates})
    existing = {record["key"]: record for record in results}
    embedding_inputs: list[dict[str, Any]] = []
    embedding_texts: list[str] = []
    for candidate in candidates:
        embedding_text = compute_entity_embedding_text(name=candidate["name"], category=candidate["category"], description=candidate["description"])
        embedding_hash = compute_entity_embedding_text_hash(embedding_text)
        record = existing.get(candidate["index"], {})
        if record.get("existing_hash") == embedding_hash and record.get("existing_model") == config.EMBEDDING_MODEL and record.get("existing_identity") == embedding_identity():
            validate_embedding(record.get("existing_vector"), model=record["existing_model"])
            continue
        embedding_inputs.append({
            "identity": {"label": candidate["label"], "id": record.get("id") or candidate["id"], "name": candidate["name"]},
            "embedding_hash": embedding_hash,
        })
        embedding_texts.append(embedding_text)

    if not embedding_inputs:
        return []

    embeddings = await get_services().language_model.async_get_embeddings_batch(embedding_texts)

    if len(embeddings) != len(embedding_inputs):
        raise ValueError("embedding batch result length mismatch")

    for index, embedding_input in enumerate(embedding_inputs):
        embedding = embeddings[index]
        if embedding is None:
            logger.warning(
                "entity embedding generation returned None",
                identity=embedding_input["identity"],
            )
            continue

        embedding_list = validate_embedding(embedding, model=config.EMBEDDING_MODEL).tolist()

        cypher = f"""
            WITH $identity AS entity
            {match_candidates}
            CALL apoc.util.validate(size(candidates) <> 1, 'Embedding target must resolve exactly once', [])
            WITH head(candidates) AS node
            CALL apoc.util.validate(node.id IS NULL OR trim(node.id) = '', 'Canonical entity has no stable ID', [])
            SET node.`{vector_property}` = $vector,
                node.`{text_hash_property}` = $text_hash,
                node.`{model_property}` = $model,
                node.`{model_property}_identity` = $embedding_identity,
                node.updated_ts = timestamp()
        """
        params = {
            "identity": embedding_input["identity"],
            "vector": embedding_list,
            "text_hash": embedding_input["embedding_hash"],
            "model": config.EMBEDDING_MODEL,
            "embedding_identity": embedding_identity(),
        }
        statements.append((cypher, params))

    logger.info(
        "Built entity embedding update statements",
        statements=len(statements),
        candidates=len(embedding_inputs),
        characters=len(characters),
        world_items=len(world_items),
    )

    return statements
