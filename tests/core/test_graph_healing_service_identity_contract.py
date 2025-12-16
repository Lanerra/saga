# tests/core/test_graph_healing_service_identity_contract.py
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from core.graph_healing_service import GraphHealingService


@pytest.mark.asyncio
async def test_enrich_node_from_context_uses_stable_id_for_boundary_call() -> None:
    """
    [CORE-008] / remediation plan 9.2 item 7

    Graph healing may use Neo4j internal `elementId()` for internal DB operations, but any
    cross-module boundary call (e.g., `data_access.kg_queries.get_chapter_context_for_entity`)
    must use the application-stable `n.id`.

    This test ensures `enrich_node_from_context()` passes `node["id"]` (stable id),
    not `node["element_id"]` (Neo4j internal id).
    """
    service = GraphHealingService()

    node = {
        "element_id": "neo4j-internal-element-id-123",
        "id": "stable-app-id-ABC",
        "name": "Provisional Entity",
        "type": "Character",
        "description": "Unknown",
        "traits": [],
        "created_chapter": 1,
    }

    with patch(
        "data_access.kg_queries.get_chapter_context_for_entity",
        new=AsyncMock(return_value=[]),
    ) as get_ctx:
        enriched = await service.enrich_node_from_context(node, model="irrelevant-model")

    # Short-circuits because no mentions; we only care about the identity boundary behavior.
    assert enriched == {}

    get_ctx.assert_awaited_once()
    _args, kwargs = get_ctx.await_args

    # Stable id should be used for boundary calls when available.
    assert kwargs.get("entity_id") == "stable-app-id-ABC"
    # Regression guard: elementId must not be passed as entity_id.
    assert kwargs.get("entity_id") != "neo4j-internal-element-id-123"


@pytest.mark.asyncio
async def test_identify_provisional_nodes_returns_stable_id_field() -> None:
    """
    [CORE-008] / remediation plan 9.2 item 7

    The provisional-node selection query must return both:
    - `elementId(n) AS element_id` (internal-only)
    - `n.id AS id` (stable id threaded through the healing pipeline)
    """
    service = GraphHealingService()

    with patch(
        "core.graph_healing_service.neo4j_manager.execute_read_query",
        new=AsyncMock(return_value=[]),
    ) as exec_read:
        out = await service.identify_provisional_nodes()

    assert out == []

    exec_read.assert_awaited_once()
    query_arg = exec_read.await_args.args[0]

    assert "elementId(n) AS element_id" in query_arg
    assert "n.id AS id" in query_arg
