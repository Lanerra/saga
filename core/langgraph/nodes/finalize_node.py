# core/langgraph/nodes/finalize_node.py
"""Persist the finalized chapter as durable artifacts.

This module defines the finalization node that persists the generated chapter to
the filesystem and to Neo4j, then clears large transient state fields.

Notes:
    Durable canonical prose and compatibility mirrors are mandatory before the
    Neo4j write. A checksum-bound filesystem acceptance receipt follows graph
    acknowledgement. Cross-store restart reconciliation belongs to the workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import structlog

import config
from core.langgraph.chapter_lifecycle import ChapterLifecycle
from core.langgraph.content_manager import (
    ContentManager,
    get_draft_text,
    get_previous_summaries,
    load_embedding,
    load_scene_embeddings,
    require_project_dir,
)
from core.langgraph.manuscript import ManuscriptReceipt, ManuscriptStore
from core.langgraph.nodes.commit_graph_ops import _aggregate_scene_embeddings_to_chapter
from core.langgraph.quality_policy import acceptance_decision, announce_acceptance, validation_decision
from core.langgraph.state import NarrativeState
from core.service_context import get_services
from data_access.chapter_queries import save_finalized_chapter_to_db as save_chapter_data_to_db

logger = structlog.get_logger(__name__)


async def finalize_chapter(state: NarrativeState) -> NarrativeState:
    """Finalize the chapter and persist it to durable storage.

    This node writes a canonical chapter file, resolves an embedding (prefer a
    scene aggregate, then `embedding_ref`, otherwise compute a fallback), and persists chapter
    metadata to Neo4j.

    Args:
        state: Workflow state.

    Returns:
        Updated state with:
        - extracted_entities / extracted_relationships cleared (already persisted)
        - quality scores, findings, and revision controls preserved
        - needs_revision reset to `False`
        - current_node set to `"finalize"`

        On fatal errors (missing draft, publication, or Neo4j failure), returns
        a state with `has_fatal_error` set and `last_error` populated.

    Notes:
        - Filesystem failures block finalization and preserve the draft reference.
        - Export reads retained canonical bytes through the accepted receipt,
          never through the replaceable Markdown/plain-text compatibility mirrors.
        - Graph acknowledgement without a receipt requires restart reconciliation;
          prepared artifacts remain recoverable and are not automatically accepted.
        - This node performs I/O (filesystem + Neo4j) and may compute an embedding
          if no upstream embedding is available.
    """
    logger.info(
        "finalize_chapter: starting finalization",
        chapter=state.get("current_chapter", 1),
    )

    if "lifecycle_version" in state:
        try:
            return await ChapterLifecycle(state).stage().publish()
        except Exception as error:
            return {"current_node": "finalize", "error_node": "finalize", "has_fatal_error": True, "last_error": f"Lifecycle publication requires reconciliation: {error}"}

    project_dir = require_project_dir(state)

    # Initialize content manager for reading externalized content
    content_manager = ContentManager(project_dir)

    from core.exceptions import MissingDraftReferenceError

    try:
        draft_text = get_draft_text(state, content_manager)
    except MissingDraftReferenceError as error:
        error_msg = str(error)
        logger.error("finalize_chapter: fatal error", error=error_msg)
        return {
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "finalize",
            "current_node": "finalize",
        }

    # Validate we have text to finalize
    if not draft_text:
        error_msg = "No draft text available for finalization"
        logger.error("finalize_chapter: fatal error", error=error_msg)
        return {
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "finalize",
            "current_node": "finalize",
        }

    chapter_number = state.get("current_chapter", 1)

    try:
        from core.langgraph.nodes.quality_assurance_node import assess_graph_quality

        validation_decision(state)
        quality = acceptance_decision({**state, "graph_quality_check": await assess_graph_quality(state)})
    except ValueError as error:
        return {"current_node": "finalize", "error_node": "finalize", "has_fatal_error": True, "last_error": str(error)}

    try:
        receipt = await _save_chapter_to_filesystem(
            chapter_number=chapter_number,
            text=draft_text,
            project_dir=project_dir,
        )
        from core.langgraph.chapter_lifecycle import canonical_bytes
        from utils.file_io import ContainedFiles

        files = ContainedFiles(Path(project_dir), durable=True)
        quality_path = receipt.artifact_path.removesuffix(".md") + ".quality.json"
        quality_bytes = canonical_bytes({"manuscript": receipt.model_dump(), "quality": quality})
        if not files.exists(quality_path):
            files.write_bytes(quality_path, quality_bytes)
        if files.read_bytes(quality_path) != quality_bytes:
            raise ValueError("Quality receipt conflict; explicit revalidation with a new manuscript version required")
    except Exception as e:
        error_msg = f"Error saving chapter to filesystem: {str(e)}"
        logger.error(
            "finalize_chapter: filesystem save failed",
            chapter=chapter_number,
            error=str(e),
            exc_info=True,
        )
        return {
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "finalize",
            "current_node": "finalize",
        }

    # Preserve the same producer priority as graph commit and staged enrichment.
    try:
        embedding = None
        embedding_ref = state.get("embedding_ref")

        scene_embeddings_ref = state.get("scene_embeddings_ref")
        if scene_embeddings_ref:
            vectors = load_scene_embeddings(content_manager, scene_embeddings_ref)
            embedding = np.asarray(_aggregate_scene_embeddings_to_chapter(vectors), dtype=config.EMBEDDING_DTYPE)
        elif embedding_ref:
            embedding_list = load_embedding(content_manager, embedding_ref)
            embedding = np.array(embedding_list, dtype=np.float32)
            logger.info(
                "finalize_chapter: reusing embedding from embedding_ref",
                chapter=chapter_number,
                embedding_shape=embedding.shape,
                embedding_ref_path=embedding_ref.get("path") if isinstance(embedding_ref, dict) else None,
            )
        else:
            embedding = await get_services().language_model.async_get_embedding(draft_text)
            logger.info(
                "finalize_chapter: embedding generated (fallback)",
                chapter=chapter_number,
                embedding_shape=embedding.shape if embedding is not None else None,
            )
    except Exception as e:
        logger.error(
            "finalize_chapter: embedding resolution failed",
            chapter=chapter_number,
            error=str(e),
            exc_info=True,
        )
        embedding = None
        # Continue without embedding (non-critical)

    try:
        current_summary = state.get("current_summary")
        if current_summary is None:
            previous_summaries = get_previous_summaries(state, content_manager)
            current_summary = previous_summaries[-1] if previous_summaries else None

        await save_chapter_data_to_db(
            chapter_number=chapter_number,
            summary=current_summary,
            embedding_array=embedding,
            embedding_model=config.EMBEDDING_MODEL,
            is_provisional=False,
        )

        logger.info(
            "finalize_chapter: chapter saved to Neo4j",
            chapter=chapter_number,
        )
    except Exception as e:
        error_msg = f"Failed to finalize chapter: {str(e)}"
        logger.error(
            "finalize_chapter: fatal error - Neo4j save failed",
            error=error_msg,
            chapter=chapter_number,
            exc_info=True,
        )
        # This is critical - return error state
        return {
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "finalize",
            "current_node": "finalize",
        }

    try:
        ManuscriptStore(Path(project_dir)).accept(receipt)
    except Exception as error:
        logger.error("finalize_chapter: acceptance publication failed", chapter=chapter_number, error=str(error), exc_info=True)
        return {
            "last_error": f"Manuscript acceptance requires reconciliation: {error}",
            "has_fatal_error": True,
            "error_node": "finalize",
            "current_node": "finalize",
        }

    announce_acceptance(quality, quality_path)
    # Retain quality evidence when clearing transient extraction state.
    logger.info(
        "finalize_chapter: finalization complete",
        chapter=chapter_number,
        word_count=state.get("draft_word_count", 0),
    )

    return cast(
        NarrativeState,
        {
            "needs_revision": False,
            "current_node": "finalize",
            "last_error": None,
            "extracted_entities": {},
            "extracted_relationships": [],
        },
    )


async def _save_chapter_to_filesystem(
    chapter_number: int,
    text: str,
    project_dir: str,
) -> ManuscriptReceipt:
    """Durably prepare retained Markdown and both compatibility mirrors.

    The returned receipt is not accepted until the graph acknowledges finalization.
    Interrupted mirrors can be rebuilt with ManuscriptStore.recover(chapter_number).
    """
    return ManuscriptStore(Path(project_dir)).prepare(chapter_number, text)


__all__ = ["finalize_chapter"]
