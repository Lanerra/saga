# core/langgraph/initialization/all_chapter_outlines_node.py
"""Generate skeleton outlines for all chapters during initialization.

This module defines the initialization node that generates high-level structural
outlines for all chapters upfront. These skeleton outlines establish the narrative
framework and are later enriched with scene-level details during chapter generation.
"""

from __future__ import annotations

from typing import Any

import structlog

from core.langgraph.content_manager import (
    ContentManager,
    require_project_dir,
)
from core.langgraph.initialization.chapter_outline_node import (
    _determine_act_for_chapter,
    _generate_single_chapter_outline,
)
from core.langgraph.state import NarrativeState

logger = structlog.get_logger(__name__)


async def generate_all_chapter_outlines(state: NarrativeState) -> NarrativeState:
    """Generate skeleton outlines for all chapters during initialization.

    Args:
        state: Workflow state containing initialization artifacts.

    Returns:
        Updated state containing:
        - chapter_outlines_ref: Externalized skeleton outlines for all chapters.
        - initialization_step: Progress marker.

    Notes:
        This node generates high-level structural outlines for all chapters
        without detailed scene planning. These skeleton outlines are versioned
        as v0 and will be enriched during chapter generation (v1+).

        Uses the Hybrid Approach:
        - Generates structural framework during init (act-level context only)
        - Enrichment happens on-demand with recent summaries and graph state
        - Reuses existing `_generate_single_chapter_outline` logic

        Can be disabled via GENERATE_ALL_CHAPTER_OUTLINES_AT_INIT config parameter
        to fall back to on-demand generation.
    """
    from config.settings import settings

    if not settings.GENERATE_ALL_CHAPTER_OUTLINES_AT_INIT:
        logger.info("generate_all_chapter_outlines: skipping (disabled by config), " "will generate on-demand")
        return {
            **state,
            "initialization_step": "all_chapter_outlines_skipped",
        }

    total_chapters = state.get("total_chapters", 20)

    logger.info(
        "generate_all_chapter_outlines: starting skeleton outline generation",
        total_chapters=total_chapters,
        title=state.get("title", ""),
    )

    content_manager = ContentManager(require_project_dir(state))

    all_outlines = {}

    for chapter_number in range(1, total_chapters + 1):
        act_number = _determine_act_for_chapter(state, chapter_number)

        logger.info(
            "generate_all_chapter_outlines: generating skeleton outline",
            chapter=chapter_number,
            act=act_number,
        )

        chapter_outline = await _generate_single_chapter_outline(
            state=state,
            chapter_number=chapter_number,
            act_number=act_number,
        )

        if not chapter_outline:
            logger.warning(
                "generate_all_chapter_outlines: failed to generate outline, continuing",
                chapter=chapter_number,
            )
            continue

        chapter_outline["generated_at"] = "initialization"
        chapter_outline["version"] = 0

        all_outlines[chapter_number] = chapter_outline

    if not all_outlines:
        error_msg = "Failed to generate any chapter outlines"
        logger.error("generate_all_chapter_outlines: no outlines generated")
        return {
            **state,
            "last_error": error_msg,
            "initialization_step": "all_chapter_outlines_failed",
        }

    outlines_for_storage: dict[str, Any] = {str(chapter): outline for chapter, outline in all_outlines.items()}

    chapter_outlines_ref = content_manager.save_json(
        outlines_for_storage,
        "chapter_outlines",
        "all",
        version=0,
    )

    logger.info(
        "generate_all_chapter_outlines: skeleton outlines generated",
        total_chapters=total_chapters,
        successful_outlines=len(all_outlines),
        version=0,
        size=chapter_outlines_ref["size_bytes"],
    )

    return {
        **state,
        "chapter_outlines_ref": chapter_outlines_ref,
        "last_error": None,
        "initialization_step": "all_chapter_outlines_complete",
    }


__all__ = ["generate_all_chapter_outlines"]
