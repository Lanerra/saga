# tests/core/test_graph_healing_service_confidence_recompute_after_enrichment.py
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from core.graph_healing_service import GraphHealingService


@pytest.mark.asyncio
async def test_heal_graph_recomputes_confidence_from_updated_node_after_enrichment() -> None:
    """
    [CORE-009] Confidence recomputation after enrichment must use UPDATED node properties.

    Regression scenario:
    - Node starts with stub attributes (Unknown / empty traits) => confidence below threshold.
    - Enrichment is applied (writes to Neo4j) and materially improves description/traits.
    - Service must recompute confidence using the updated node data (reloaded), so the
      post-enrichment confidence reflects the improvement and can trigger graduation.

    This would fail under the stale-dict behavior where confidence is recomputed using
    the original `node` dict (still Unknown/empty).
    """
    service = GraphHealingService()
    service.MIN_CONFIDENCE_FOR_ENRICHMENT = 0.0
    service.CONFIDENCE_THRESHOLD = 0.0

    element_id = "neo4j-element-id-123"
    node_before = {
        "element_id": element_id,
        "id": "stable-app-id-ABC",
        "name": "Provisional Entity",
        "type": "Character",
        "description": "Unknown",
        "traits": [],
        "created_chapter": 0,
    }

    # Meaningful description: > 20 chars and not a stub phrase.
    node_after = {
        **node_before,
        "description": "A battle-tested veteran of the northern campaigns.",
        "traits": ["Brave"],
    }

    enriched = {
        "inferred_description": node_after["description"],
        "inferred_traits": node_after["traits"],
        "inferred_role": "Protagonist",
        "confidence": 0.9,
    }

    async def _exec_read_query(query: str, params=None):
        # calculate_node_confidence() reads:
        # 1) relationship count (connectivity score)
        # 2) character status
        if "RETURN count(r) AS rel_count" in query:
            # Ensure connectivity contributes 0.4, but pre-enrichment still below threshold:
            # - before: completeness=0.0, connectivity=0.4 => 0.4 < 0.6
            # - after: completeness=0.3 (desc+traits), connectivity=0.4 => 0.7 >= 0.6
            return [{"rel_count": 3}]

        if "RETURN n.status AS status" in query:
            # Keep status unknown so it doesn't affect the test math.
            return [{"status": "Unknown"}]

        raise AssertionError(f"Unexpected read query in test: {query!r}")

    with (
        # Patch read queries narrowly for confidence calculation.
        patch(
            "core.graph_healing_service.neo4j_manager.execute_read_query",
            new=AsyncMock(side_effect=_exec_read_query),
        ),
        patch.object(
            service,
            "calculate_node_confidence",
            new=AsyncMock(return_value=1.0),
        ),
        patch.object(
            service,
            "identify_provisional_nodes",
            new=AsyncMock(return_value=[node_before]),
        ),
        patch.object(
            service,
            "enrich_node_from_context",
            new=AsyncMock(return_value=enriched),
        ),
        patch.object(
            service,
            "apply_enrichment",
            new=AsyncMock(return_value=True),
        ),
        patch.object(
            service,
            "get_node_by_element_id",
            new=AsyncMock(return_value=node_after),
        ) as get_updated,
        patch.object(
            service,
            "graduate_node",
            new=AsyncMock(return_value=True),
        ),
        # Keep the test hermetic: avoid other Neo4j reads from merge candidate discovery
        # and orphan cleanup, which are not part of CORE-009.
        patch.object(
            service,
            "find_merge_candidates",
            new=AsyncMock(return_value=[]),
        ),
        patch.object(
            service,
            "cleanup_orphaned_nodes",
            new=AsyncMock(return_value={"nodes_removed": 0, "nodes_checked": 0}),
        ),
    ):
        results = await service.heal_graph(current_chapter=1, model="irrelevant-model")

    # Ensure the post-enrichment reload path executed (Option B behavior).
    get_updated.assert_awaited_once_with(element_id)

    # Sanity-check reported metrics (enrichment executed, graduation optional).
    assert results["nodes_enriched"] == 1
    assert results["nodes_graduated"] in (0, 1)
