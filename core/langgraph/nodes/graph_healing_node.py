# core/langgraph/nodes/graph_healing_node.py
"""
Graph Healing Node for SAGA LangGraph Workflow.

This node runs after chapter finalization to heal provisional nodes
and merge semantically similar entities in the knowledge graph.
"""

from __future__ import annotations

import structlog

from core.graph_healing_service import graph_healing_service
from core.langgraph.state import NarrativeState

logger = structlog.get_logger(__name__)


async def heal_graph(state: NarrativeState) -> NarrativeState:
    """
    Heal the knowledge graph by enriching provisional nodes and merging duplicates.

    This node runs after each chapter finalization to:
    1. Identify provisional nodes that need enrichment
    2. Calculate confidence scores based on accumulated evidence
    3. Enrich nodes with missing attributes using LLM inference
    4. Graduate nodes from provisional status when confidence is high
    5. Find and merge semantically similar entities

    Args:
        state: Current narrative state

    Returns:
        Updated state with healing results
    """
    current_chapter = state.get("current_chapter", 1)
    model = state.get("extraction_model", "qwen3-a3b")

    logger.info(
        "heal_graph: Starting graph healing",
        chapter=current_chapter,
    )

    try:
        # Run the healing process
        results = await graph_healing_service.heal_graph(current_chapter=current_chapter, model=model)

        # Explicit observability for non-fatal healing degradations (LANGGRAPH-025).
        # Even if the workflow continues, warnings must be visible in logs/state.
        healing_warnings = results.get("warnings", []) or []
        if healing_warnings:
            logger.warning(
                "heal_graph: Healing completed with warnings",
                chapter=current_chapter,
                warnings=healing_warnings,
                apoc_available=results.get("apoc_available"),
            )

        # Update healing history
        healing_history = state.get("healing_history", [])
        healing_history.append(results)

        # Calculate running totals
        total_graduated = state.get("nodes_graduated", 0) + results["nodes_graduated"]
        total_merged = state.get("nodes_merged", 0) + results["nodes_merged"]
        total_enriched = state.get("nodes_enriched", 0) + results["nodes_enriched"]
        total_removed = state.get("nodes_removed", 0) + results.get("nodes_removed", 0)

        # Avoid negative metrics: some services return "remaining", others return "total".
        # We clamp to ensure the state is never misleading for dashboards/consumers.
        provisional_raw = results.get("provisional_count", 0)
        graduated_this_run = results.get("nodes_graduated", 0)
        provisional_remaining = max(0, provisional_raw - graduated_this_run)

        # Extract merge candidates for potential user review.
        # `merge_candidates` must be populated consistently (was previously always empty).
        merge_candidates: list[dict[str, object]] = []
        pending_merges: list[dict[str, object]] = []
        auto_approved_merges: list[dict[str, object]] = []

        for action in results.get("actions", []):
            if action.get("type") == "merge":
                merge_info = {
                    "primary": action.get("primary"),
                    "duplicate": action.get("duplicate"),
                    "similarity": action.get("similarity"),
                }
                merge_candidates.append(merge_info)

                if action.get("auto_approved"):
                    auto_approved_merges.append(merge_info)
                else:
                    pending_merges.append(merge_info)

        logger.info(
            "heal_graph: Graph healing complete",
            chapter=current_chapter,
            graduated=results["nodes_graduated"],
            enriched=results["nodes_enriched"],
            merged=results["nodes_merged"],
            removed=results.get("nodes_removed", 0),
            provisional_remaining=provisional_remaining,
        )

        return {
            **state,
            "current_node": "heal_graph",
            "last_error": None,
            "last_healing_chapter": current_chapter,
            "provisional_count": provisional_remaining,
            "nodes_graduated": total_graduated,
            "nodes_merged": total_merged,
            "nodes_enriched": total_enriched,
            "nodes_removed": total_removed,
            "merge_candidates": merge_candidates,
            "pending_merges": pending_merges,
            "auto_approved_merges": auto_approved_merges,
            "healing_history": healing_history,
            # Snapshot for callers/tests so warnings are not "silent degradation".
            "last_healing_warnings": healing_warnings,
            "last_apoc_available": results.get("apoc_available"),
        }

    except Exception as e:
        import traceback

        logger.error(
            "heal_graph: Error during graph healing",
            chapter=current_chapter,
            error=str(e),
            traceback=traceback.format_exc(),
        )

        # Don't fail the workflow for healing errors
        return {
            **state,
            "current_node": "heal_graph",
            "last_error": f"Graph healing warning: {str(e)}",
            "last_healing_chapter": current_chapter,
        }


__all__ = ["heal_graph"]
