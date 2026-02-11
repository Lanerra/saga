# core/langgraph/nodes/quality_assurance_node.py
"""Run periodic quality checks against the knowledge graph.

This module defines a QA node that runs on a configurable cadence to detect
consistency issues and optionally apply low-risk cleanup actions (for example,
deduplicating relationships).

Notes:
    Unlike the healing node, which enriches provisional entities, this node focuses
    on detecting narrative inconsistencies and surfacing metrics for monitoring.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import structlog

import config
from core.langgraph.state import NarrativeState
from data_access.kg_queries import (
    consolidate_similar_relationships,
    deduplicate_relationships,
    find_contradictory_trait_characters,
)
from models.kg_constants import CONTRADICTORY_TRAIT_PAIRS

logger = structlog.get_logger(__name__)


async def check_quality(state: NarrativeState) -> NarrativeState:
    """Run quality assurance checks on the knowledge graph.

    Args:
        state: Workflow state.

    Returns:
        Updated state containing:
        - qa_results: Dict of detected issues and applied automatic fixes.
        - qa_history: Append-only history entries with timestamps.
        - last_qa_chapter: Chapter number when the node last executed checks.
        - total_qa_issues / total_qa_fixes: Running totals.
        - current_node: `"check_quality"`.

        If QA is disabled or checks are not yet due (based on frequency), returns an
        update with `current_node` set and does not perform any I/O.

    Notes:
        This node performs Neo4j reads (and may perform writes when relationship
        deduplication/consolidation is enabled). Failures in individual checks are
        logged and do not fail the workflow.
    """
    current_chapter = state.get("current_chapter", 1)
    qa_enabled = config.settings.ENABLE_QA_CHECKS

    if not qa_enabled:
        logger.info("check_quality: QA checks disabled")
        return {"current_node": "check_quality"}

    qa_frequency = config.settings.QA_CHECK_FREQUENCY
    last_qa_chapter = cast(int, state.get("last_qa_chapter", 0))

    if current_chapter - last_qa_chapter < qa_frequency:
        logger.debug(
            "check_quality: Skipping QA checks (not yet due)",
            chapter=current_chapter,
            frequency=qa_frequency,
            last_check=last_qa_chapter,
        )
        return {"current_node": "check_quality"}

    logger.info("check_quality: Starting quality assurance checks", chapter=current_chapter)

    issues_found = 0
    relationships_deduplicated = 0
    relationships_consolidated = 0

    qa_results: dict[str, Any] = {
        "contradictory_traits": [],
        "relationships_deduplicated": relationships_deduplicated,
        "relationships_consolidated": relationships_consolidated,
        "issues_found": issues_found,
    }

    if config.settings.QA_CHECK_CONTRADICTORY_TRAITS:
        try:
            contradictory_traits = await find_contradictory_trait_characters(CONTRADICTORY_TRAIT_PAIRS)
            qa_results["contradictory_traits"] = contradictory_traits
            issues_found += len(contradictory_traits)

            if contradictory_traits:
                logger.warning(
                    "check_quality: Found characters with contradictory traits",
                    count=len(contradictory_traits),
                    examples=[f"{c['character_name']}: {c['trait1']} vs {c['trait2']}" for c in contradictory_traits[:3]],
                )

        except Exception as e:
            logger.error(
                "check_quality: Error checking contradictory traits",
                error=str(e),
                exc_info=True,
            )

    if config.settings.QA_DEDUPLICATE_RELATIONSHIPS:
        try:
            dedupe_count = await deduplicate_relationships()
            relationships_deduplicated = int(dedupe_count)

            if relationships_deduplicated > 0:
                logger.info(
                    "check_quality: Deduplicated relationships",
                    count=relationships_deduplicated,
                )

        except Exception as e:
            logger.error(
                "check_quality: Error deduplicating relationships",
                error=str(e),
                exc_info=True,
            )

    if config.settings.QA_CONSOLIDATE_RELATIONSHIPS:
        try:
            consolidate_count = await consolidate_similar_relationships()
            relationships_consolidated = int(consolidate_count)

            if relationships_consolidated > 0:
                logger.info(
                    "check_quality: Consolidated relationships",
                    count=relationships_consolidated,
                )

        except Exception as e:
            logger.error(
                "check_quality: Error consolidating relationships",
                error=str(e),
                exc_info=True,
            )

    qa_results["issues_found"] = issues_found
    qa_results["relationships_deduplicated"] = relationships_deduplicated
    qa_results["relationships_consolidated"] = relationships_consolidated

    qa_history = list(cast(list[dict[str, Any]], state.get("qa_history", [])))
    qa_history.append(
        {
            "chapter": current_chapter,
            "timestamp": datetime.now(UTC).isoformat(),
            "results": qa_results,
        }
    )

    total_issues = issues_found
    total_fixes = relationships_deduplicated + relationships_consolidated

    logger.info(
        "check_quality: Quality assurance complete",
        chapter=current_chapter,
        issues_found=total_issues,
        automatic_fixes=total_fixes,
    )

    total_qa_issues = cast(int, state.get("total_qa_issues", 0))
    total_qa_fixes = cast(int, state.get("total_qa_fixes", 0))

    return {
        "current_node": "check_quality",
        "last_error": None,
        "last_qa_chapter": current_chapter,
        "qa_results": qa_results,
        "qa_history": qa_history,
        "total_qa_issues": total_qa_issues + total_issues,
        "total_qa_fixes": total_qa_fixes + total_fixes,
    }
