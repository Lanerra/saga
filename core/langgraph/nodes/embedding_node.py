# core/langgraph/nodes/embedding_node.py
"""Generate and externalize embeddings for chapter drafts.

This module defines the embedding node used by the LangGraph workflow. Embeddings
enable semantic retrieval and similarity comparisons in downstream steps.
"""

from typing import Any

import numpy as np
import structlog

from core.langgraph.content_manager import (
    ContentManager,
    get_draft_text,
    get_scene_drafts,
    save_scene_embeddings,
)
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service

logger = structlog.get_logger(__name__)


async def generate_embedding(state: NarrativeState) -> NarrativeState:
    """Generate an embedding for the current chapter draft.

    This node reads the current draft (preferring the externalized `draft_ref`),
    requests an embedding from the LLM provider, and externalizes the embedding to
    reduce state size.

    Args:
        state: Workflow state.

    Returns:
        Updated state containing:
        - embedding_ref: Content reference for the persisted embedding vector.
        - current_node: `"generate_embedding"`.

        If the draft is missing/empty, returns an update with `generated_embedding`
        set to `None` and does not fail the workflow.
        If embedding generation fails, returns an update with `last_error` populated
        but does not set `has_fatal_error`.

    Notes:
        This node performs network I/O to generate embeddings and filesystem I/O to
        persist the result via `ContentManager.save_binary()`.
    """
    content_manager = ContentManager(state.get("project_dir", ""))
    draft_text = get_draft_text(state, content_manager)
    chapter_num = state.get("current_chapter")

    logger.info("generate_embedding: starting embedding generation", chapter=chapter_num)

    if not draft_text or not draft_text.strip():
        logger.warning(
            "generate_embedding: no draft text found, skipping embedding generation",
            chapter=chapter_num,
        )
        return {
            "generated_embedding": None,
            "current_node": "generate_embedding",
        }

    try:
        # Generate embedding using the refactored LLM service
        embedding = await llm_service.async_get_embedding(draft_text)

        if embedding is None:
            logger.error(
                "generate_embedding: failed to generate embedding (returned None)",
                chapter=chapter_num,
            )
            return {
                "generated_embedding": None,
                "current_node": "generate_embedding",
            }

        # Convert numpy array to list for JSON serialization in state
        if isinstance(embedding, np.ndarray):
            embedding_list = embedding.tolist()
        else:
            embedding_list = list(embedding)

        # Initialize content manager for external storage
        content_manager = ContentManager(state.get("project_dir", ""))

        # Get current version (for revision tracking)
        current_version = content_manager.get_latest_version("embedding", f"chapter_{chapter_num}") + 1

        # Externalize generated_embedding to reduce state bloat
        embedding_ref = content_manager.save_binary(
            embedding_list,
            "embedding",
            f"chapter_{chapter_num}",
            current_version,
        )

        logger.info(
            "generate_embedding: successfully generated and externalized embedding",
            chapter=chapter_num,
            dimensions=len(embedding_list),
            version=current_version,
            size=embedding_ref["size_bytes"],
        )

        return {
            "embedding_ref": embedding_ref,
            "current_node": "generate_embedding",
        }

    except Exception as e:
        logger.error(
            "generate_embedding: error during embedding generation",
            error=str(e),
            chapter=chapter_num,
            exc_info=True,
        )
        # We don't want to fail the whole workflow for a missing embedding,
        # but we should log it as a non-fatal error
        return {
            "generated_embedding": None,
            "current_node": "generate_embedding",
            "last_error": f"Embedding generation failed: {e}",
            # Note: Not setting has_fatal_error=True as this is optional
        }


async def generate_scene_embeddings(state: NarrativeState) -> dict[str, Any]:
    """Generate embeddings for each scene draft in the current chapter.

    This node loads scene drafts from `scene_drafts_ref`, requests embeddings in a
    single batch, and persists the resulting list-of-vectors as a single per-chapter
    artifact.

    Returns a partial state update containing:
    - scene_embeddings_ref: Content reference for persisted scene embedding vectors
    - current_node: "generate_scene_embeddings"

    If no scenes are available, this node sets `last_error` and does not set
    `has_fatal_error`.
    """
    chapter_number = state.get("current_chapter", 1)
    if not isinstance(chapter_number, int) or isinstance(chapter_number, bool) or chapter_number <= 0:
        raise ValueError("generate_scene_embeddings expected current_chapter to be a positive int; " f"got {chapter_number!r}")

    content_manager = ContentManager(state.get("project_dir", ""))

    if not state.get("scene_drafts_ref"):
        return {
            "current_node": "generate_scene_embeddings",
            "last_error": "Scene embedding generation skipped: scene_drafts_ref is missing",
        }

    scene_drafts = get_scene_drafts(state, content_manager)

    if len(scene_drafts) == 0:
        return {
            "current_node": "generate_scene_embeddings",
            "last_error": "Scene embedding generation skipped: no scene drafts found",
        }

    embeddings = await llm_service.async_get_embeddings_batch(scene_drafts)

    if len(embeddings) != len(scene_drafts):
        raise ValueError("scene embedding batch result length mismatch")

    scene_embeddings: list[list[float]] = []
    for scene_index, embedding in enumerate(embeddings):
        if embedding is None:
            raise ValueError(f"scene embedding generation returned None for scene_index={scene_index}")

        if isinstance(embedding, np.ndarray):
            embedding_values = embedding.tolist()
        else:
            embedding_values = list(embedding)

        vector: list[float] = []
        for value_index, value in enumerate(embedding_values):
            if isinstance(value, bool):
                raise TypeError("scene embedding values must be numeric (bool is not allowed); " f"scene_index={scene_index}, value_index={value_index}")
            if not isinstance(value, int | float):
                raise TypeError("scene embedding values must be numeric; " f"scene_index={scene_index}, value_index={value_index}, got {type(value)}")
            vector.append(float(value))

        scene_embeddings.append(vector)

    current_version = content_manager.get_latest_version("scene_embeddings", f"chapter_{chapter_number}") + 1
    scene_embeddings_ref = save_scene_embeddings(
        content_manager,
        scene_embeddings,
        chapter_number,
        current_version,
    )

    return {
        "scene_embeddings_ref": scene_embeddings_ref,
        "current_node": "generate_scene_embeddings",
    }
