# core/langgraph/nodes/quality_assurance_node.py
"""
Quality Assurance Node for SAGA LangGraph Workflow.

This node runs periodically to detect narrative consistency issues:
1. Contradictory character traits
2. Post-mortem character activity
3. Duplicate relationship consolidation
4. Relationship type normalization

Unlike graph_healing_node which focuses on provisional node enrichment,
this node focuses on consistency checking and error detection.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

import config
from core.langgraph.state import NarrativeState
from data_access.kg_queries import (
    consolidate_similar_relationships,
    deduplicate_relationships,
    find_contradictory_trait_characters,
    find_post_mortem_activity,
)

logger = structlog.get_logger(__name__)

CONTRADICTORY_TRAIT_PAIRS = [
    ("Brave", "Cowardly"),
    ("Honest", "Deceitful"),
    ("Kind", "Cruel"),
    ("Loyal", "Treacherous"),
    ("Optimistic", "Pessimistic"),
    ("Calm", "Anxious"),
    ("Confident", "Insecure"),
    ("Generous", "Selfish"),
    ("Patient", "Impatient"),
    ("Humble", "Arrogant"),
]


async def check_quality(state: NarrativeState) -> NarrativeState:
    """
    Run quality assurance checks on the knowledge graph.

    This node performs consistency checks that help identify
    potential narrative errors or contradictions that may have
    accumulated during generation.

    Checks performed:
    1. Contradictory traits (e.g., Brave + Cowardly)
    2. Post-mortem activity (dead characters doing things)
    3. Duplicate relationships
    4. Relationship type normalization

    Args:
        state: Current narrative state

    Returns:
        Updated state with QA results
    """
    current_chapter = state.get("current_chapter", 1)
    qa_enabled = config.settings.ENABLE_QA_CHECKS

    if not qa_enabled:
        logger.info("check_quality: QA checks disabled")
        return {**state, "current_node": "check_quality"}

    qa_frequency = config.settings.QA_CHECK_FREQUENCY
    last_qa_chapter = state.get("last_qa_chapter", 0)

    if current_chapter - last_qa_chapter < qa_frequency:
        logger.debug(
            "check_quality: Skipping QA checks (not yet due)",
            chapter=current_chapter,
            frequency=qa_frequency,
            last_check=last_qa_chapter,
        )
        return {**state, "current_node": "check_quality"}

    logger.info("check_quality: Starting quality assurance checks", chapter=current_chapter)

    qa_results = {
        "contradictory_traits": [],
        "post_mortem_activities": [],
        "relationships_deduplicated": 0,
        "relationships_consolidated": 0,
        "issues_found": 0,
    }

    if config.settings.QA_CHECK_CONTRADICTORY_TRAITS:
        try:
            contradictory_traits = await find_contradictory_trait_characters(CONTRADICTORY_TRAIT_PAIRS)
            qa_results["contradictory_traits"] = contradictory_traits
            qa_results["issues_found"] += len(contradictory_traits)

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

    if config.settings.QA_CHECK_POST_MORTEM_ACTIVITY:
        try:
            post_mortem = await find_post_mortem_activity()
            qa_results["post_mortem_activities"] = post_mortem
            qa_results["issues_found"] += len(post_mortem)

            if post_mortem:
                logger.warning(
                    "check_quality: Found post-mortem character activity",
                    count=len(post_mortem),
                    examples=[f"{a['character_name']} (died ch. {a['death_chapter']})" for a in post_mortem[:3]],
                )

        except Exception as e:
            logger.error(
                "check_quality: Error checking post-mortem activity",
                error=str(e),
                exc_info=True,
            )

    if config.settings.QA_DEDUPLICATE_RELATIONSHIPS:
        try:
            dedupe_count = await deduplicate_relationships()
            qa_results["relationships_deduplicated"] = dedupe_count

            if dedupe_count > 0:
                logger.info(
                    "check_quality: Deduplicated relationships",
                    count=dedupe_count,
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
            qa_results["relationships_consolidated"] = consolidate_count

            if consolidate_count > 0:
                logger.info(
                    "check_quality: Consolidated relationships",
                    count=consolidate_count,
                )

        except Exception as e:
            logger.error(
                "check_quality: Error consolidating relationships",
                error=str(e),
                exc_info=True,
            )

    qa_history = state.get("qa_history", [])
    qa_history.append(
        {
            "chapter": current_chapter,
            "timestamp": datetime.now(UTC).isoformat(),
            "results": qa_results,
        }
    )

    total_issues = qa_results["issues_found"]
    total_fixes = qa_results["relationships_deduplicated"] + qa_results["relationships_consolidated"]

    logger.info(
        "check_quality: Quality assurance complete",
        chapter=current_chapter,
        issues_found=total_issues,
        automatic_fixes=total_fixes,
    )

    return {
        **state,
        "current_node": "check_quality",
        "last_error": None,
        "last_qa_chapter": current_chapter,
        "qa_results": qa_results,
        "qa_history": qa_history,
        "total_qa_issues": state.get("total_qa_issues", 0) + total_issues,
        "total_qa_fixes": state.get("total_qa_fixes", 0) + total_fixes,
    }
