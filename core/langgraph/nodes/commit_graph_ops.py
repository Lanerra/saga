# core/langgraph/nodes/commit_graph_ops.py
"""
Graph-operation helpers for chapter node and embedding aggregation.

Extracted from commit_node.py as part of the module-split refactor (I4).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

import config
from core.embedding_contract import validate_embedding
from data_access import chapter_queries

logger = structlog.get_logger(__name__)


def _build_chapter_node_statement(
    chapter_number: int,
    summary: str | None,
    embedding: list[float] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Build Cypher statement for chapter node creation.

    NOTE:
    This MUST use the canonical Chapter persistence semantics so Chapter nodes always
    have schema-required identity (`Chapter.id`) and we never create "number-only"
    Chapter nodes.

    Args:
        chapter_number: Chapter number
        summary: Optional chapter summary
        embedding: Optional embedding vector

    Returns:
        Tuple of (cypher_query, parameters)
    """
    query, parameters = chapter_queries.build_chapter_upsert_statement(
        chapter_number=chapter_number,
        summary=summary,
        embedding_vector=embedding,
        embedding_model=config.EMBEDDING_MODEL,
        is_provisional=False,  # Chapter draft is validated before commit in the workflow
    )

    logger.debug(
        "_build_chapter_node_statement: built statement (canonical chapter upsert)",
        chapter=chapter_number,
        chapter_id=parameters.get("chapter_id_param"),
    )

    return (query, parameters)


def _aggregate_scene_embeddings_to_chapter(
    scene_embeddings: list[list[float]] | dict[str, list[float]],
) -> list[float]:
    """
    Aggregate scene-level embeddings into a single chapter embedding.

    Strategy: Average all scene embeddings to create a representative chapter embedding.
    This provides semantic coverage of the entire chapter while being computationally efficient.

    Args:
        scene_embeddings: List or dict of scene embedding vectors

    Returns:
        Single chapter embedding vector (averaged from all scenes)
    """
    if not scene_embeddings:
        return []

    # Handle both list and dict formats
    if isinstance(scene_embeddings, dict):
        embeddings_list = list(scene_embeddings.values())
    else:
        embeddings_list = scene_embeddings

    if not embeddings_list:
        return []

    # Convert to numpy array for efficient computation
    embeddings_array = np.array([validate_embedding(vector, model=config.EMBEDDING_MODEL) for vector in embeddings_list], dtype=np.float64)

    # Average across scenes (axis=0)
    chapter_embedding = validate_embedding(np.mean(embeddings_array, axis=0), model=config.EMBEDDING_MODEL).tolist()

    return chapter_embedding
