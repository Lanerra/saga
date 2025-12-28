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
    require_project_dir,
    save_scene_embeddings,
)
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service

logger = structlog.get_logger(__name__)



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

    content_manager = ContentManager(require_project_dir(state))

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
