"""
Finalization node for LangGraph workflow.

This module contains the chapter finalization logic for the LangGraph-based
narrative generation workflow.

Migration Reference: docs/phase2_migration_plan.md - Step 2.4

Source Code Ported From:
- orchestration/nana_orchestrator.py: Chapter persistence logic
- agents/narrative_agent.py: File saving patterns
"""

from __future__ import annotations

from pathlib import Path

import structlog

from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from data_access.chapter_queries import save_chapter_data_to_db
from utils.file_io import write_text_file

logger = structlog.get_logger(__name__)


async def finalize_chapter(state: NarrativeState) -> NarrativeState:
    """
    Finalize chapter by persisting to filesystem and Neo4j.

    This is the main LangGraph node function for chapter finalization.
    It performs the final steps after generation, revision, and extraction:
    1. Save chapter text to filesystem
    2. Generate embedding for the chapter
    3. Persist everything to Neo4j
    4. Clean up temporary state fields

    PORTED FROM: Orchestrator chapter persistence logic

    Process Flow:
    1. Validate draft text exists
    2. Save chapter text to filesystem (project_dir/chapters/)
    3. Generate embedding vector for the chapter
    4. Save all data to Neo4j (text, summary, embedding)
    5. Clean up temporary extraction fields
    6. Mark chapter as complete

    Args:
        state: Current narrative state containing finalized draft_text

    Returns:
        Updated state with chapter marked as complete and cleanup done
    """
    logger.info(
        "finalize_chapter: starting finalization",
        chapter=state["current_chapter"],
    )

    # Validate we have text to finalize
    if not state.get("draft_text"):
        error_msg = "No draft text available for finalization"
        logger.error("finalize_chapter: fatal error", error=error_msg)
        return {
            **state,
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "finalize",
            "current_node": "finalize",
        }

    chapter_number = state["current_chapter"]
    draft_text = state["draft_text"]

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

    # Step 2: Generate embedding
    try:
        embedding = await llm_service.async_get_embedding(draft_text)
        logger.info(
            "finalize_chapter: embedding generated",
            chapter=chapter_number,
            embedding_shape=embedding.shape if embedding is not None else None,
        )
    except Exception as e:
        logger.error(
            "finalize_chapter: embedding generation failed",
            chapter=chapter_number,
            error=str(e),
            exc_info=True,
        )
        embedding = None
        # Continue without embedding (non-critical)

    # Step 3: Save to Neo4j
    try:
        # Get summary from previous_chapter_summaries if available
        summaries = state.get("previous_chapter_summaries", [])
        current_summary = summaries[-1] if summaries else None

        await save_chapter_data_to_db(
            chapter_number=chapter_number,
            text=draft_text,
            raw_llm_output=draft_text,  # Final text is the raw output
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
            **state,
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "finalize",
            "current_node": "finalize",
        }

    # Step 4: Clean up temporary state
    # Remove large temporary data that's no longer needed
    cleaned_state = {
        **state,
        # Clear extraction results (now committed to Neo4j)
        "extracted_entities": {},
        "extracted_relationships": [],
        # Clear contradictions (chapter is finalized)
        "contradictions": [],
        # Clear iteration tracking
        "iteration_count": 0,
        "needs_revision": False,
        # Update status
        "current_node": "finalize",
        "last_error": None,
    }

    logger.info(
        "finalize_chapter: finalization complete",
        chapter=chapter_number,
        word_count=state.get("draft_word_count", 0),
    )

    return cleaned_state


async def _save_chapter_to_filesystem(
    chapter_number: int,
    text: str,
    project_dir: str,
) -> None:
    """
    Save finalized chapter to filesystem.

    Writes the canonical Markdown artifact with YAML front matter and a
    legacy .txt mirror for backward compatibility.

    Canonical artifact:
        chapters/chapter_{chapter_number:03d}.md

    Legacy compatibility:
        chapters/chapter_{chapter_number:03d}.txt
        (plain text only, no front matter; maintained temporarily)

    Args:
        chapter_number: Chapter number (used for filename and metadata)
        text: Finalized chapter prose
        project_dir: Base project directory (e.g., "output/my-novel")
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
