# core/langgraph/nodes/finalize_node.py
"""Persist the finalized chapter as durable artifacts.

This module defines the finalization node that persists the generated chapter to
the filesystem and to Neo4j, then clears large transient state fields.

Notes:
    This node performs filesystem I/O and Neo4j writes. Filesystem writes are
    best-effort; Neo4j persistence is treated as the source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import structlog

from core.langgraph.content_manager import (
    ContentManager,
    get_draft_text,
    get_previous_summaries,
    load_embedding,
)
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from data_access.chapter_queries import save_chapter_data_to_db
from utils.file_io import write_text_file

logger = structlog.get_logger(__name__)


async def finalize_chapter(state: NarrativeState) -> NarrativeState:
    """Finalize the chapter and persist it to durable storage.

    This node writes a canonical chapter file, resolves an embedding (prefer an
    upstream `embedding_ref`, otherwise compute a fallback), and persists chapter
    metadata to Neo4j.

    Args:
        state: Workflow state.

    Returns:
        Updated state with:
        - extracted_entities / extracted_relationships cleared (already persisted)
        - contradictions cleared
        - needs_revision reset to `False`
        - current_node set to `"finalize"`

        On fatal errors (missing draft text, or Neo4j persistence failure), returns
        a state with `has_fatal_error` set and `last_error` populated.

    Notes:
        - Filesystem writes are best-effort: failures are logged and do not block
          Neo4j persistence.
        - Neo4j persistence is treated as the source of truth; failures are fatal.
        - This node performs I/O (filesystem + Neo4j) and may compute an embedding
          if no upstream embedding is available.
    """
    logger.info(
        "finalize_chapter: starting finalization",
        chapter=state.get("current_chapter", 1),
    )

    # Initialize content manager for reading externalized content
    content_manager = ContentManager(state.get("project_dir", ""))

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

    # Step 1: Save to filesystem
    try:
        await _save_chapter_to_filesystem(
            chapter_number=chapter_number,
            text=draft_text,
            project_dir=state.get("project_dir", "output"),
        )
    except Exception as e:
        error_msg = f"Error saving chapter to filesystem: {str(e)}"
        logger.error(
            "finalize_chapter: filesystem save failed",
            chapter=chapter_number,
            error=str(e),
            exc_info=True,
        )
        # Continue with Neo4j save even if filesystem fails
        # (Neo4j is source of truth)

    # Step 2: Get or generate embedding (exactly once per chapter when embedding node is present)
    #
    # Preferred behavior:
    # - If an upstream embedding node ran, it should have stored `embedding_ref` in state.
    #   We load and reuse that here (no recompute).
    # - If no embedding is available (e.g., embedding node absent), we compute as a fallback.
    try:
        embedding = None
        embedding_ref = state.get("embedding_ref")

        if embedding_ref:
            embedding_list = load_embedding(content_manager, embedding_ref)
            embedding = np.array(embedding_list, dtype=np.float32)
            logger.info(
                "finalize_chapter: reusing embedding from embedding_ref",
                chapter=chapter_number,
                embedding_shape=embedding.shape,
                embedding_ref_path=embedding_ref.get("path") if isinstance(embedding_ref, dict) else None,
            )
        else:
            embedding = await llm_service.async_get_embedding(draft_text)
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

    # Step 3: Save to Neo4j
    try:
        current_summary = state.get("current_summary")
        if current_summary is None:
            previous_summaries = get_previous_summaries(state, content_manager)
            current_summary = previous_summaries[-1] if previous_summaries else None

        await save_chapter_data_to_db(
            chapter_number=chapter_number,
            summary=current_summary,
            embedding_array=embedding,
            is_provisional=False,  # Final chapter, not provisional
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

    # Step 4: Clean up temporary state (clear extraction artifacts and counters)
    logger.info(
        "finalize_chapter: finalization complete",
        chapter=chapter_number,
        word_count=state.get("draft_word_count", 0),
    )

    return cast(
        NarrativeState,
        {
            "extracted_entities": {},
            "extracted_relationships": [],
            "contradictions": [],
            "iteration_count": 0,
            "needs_revision": False,
            "current_node": "finalize",
            "last_error": None,
        },
    )


async def _save_chapter_to_filesystem(
    chapter_number: int,
    text: str,
    project_dir: str,
) -> None:
    """Write the finalized chapter to the project filesystem.

    The canonical artifact is a Markdown file with YAML front matter. A plain-text
    mirror is also written for legacy consumers.

    Canonical artifact:
        chapters/chapter_{chapter_number:03d}.md

    Legacy mirror:
        chapters/chapter_{chapter_number:03d}.txt

    Args:
        chapter_number: Chapter number used for filenames and metadata.
        text: Finalized chapter prose.
        project_dir: Base project directory containing the `chapters/` folder.
    """
    from datetime import datetime

    # Create chapters directory if it doesn't exist (handled implicitly by helpers,
    # but we keep the directory Path for clarity and logging).
    chapters_dir = Path(project_dir) / "chapters"

    # Compute metadata
    word_count = len(text.split())
    # NOTE: Title/pov_character are intentionally not sourced from state here to
    # avoid tight coupling; they can be injected in a future refactor.
    title = f"Chapter {chapter_number}"
    generated_at = datetime.utcnow().isoformat()
    version = 1

    # Build Markdown with YAML front matter
    front_matter_lines = [
        "---",
        f"chapter: {chapter_number}",
        f"title: {title}",
        f"word_count: {word_count}",
        f"generated_at: {generated_at}",
        f"version: {version}",
        "---",
        "",
    ]
    markdown_content = "\n".join(front_matter_lines) + text

    # Canonical .md path
    md_file = chapters_dir / f"chapter_{chapter_number:03d}.md"
    # Legacy .txt path (plain body only)
    txt_file = chapters_dir / f"chapter_{chapter_number:03d}.txt"

    try:
        # Write canonical Markdown artifact
        write_text_file(md_file, markdown_content)

        # Write legacy .txt mirror for existing consumers/tests
        write_text_file(txt_file, text)

        logger.info(
            "finalize_chapter: chapter saved to filesystem",
            chapter=chapter_number,
            md_path=str(md_file),
            txt_path=str(txt_file),
            word_count=word_count,
        )
    except Exception as e:
        logger.error(
            "finalize_chapter: failed to write chapter files",
            chapter=chapter_number,
            md_path=str(md_file),
            txt_path=str(txt_file),
            error=str(e),
            exc_info=True,
        )
        raise


__all__ = ["finalize_chapter"]
