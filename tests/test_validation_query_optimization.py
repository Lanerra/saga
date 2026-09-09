# tests/test_validation_query_optimization.py
"""Test validation query optimization - ensuring combined query reduces Neo4j round-trips.

NOTE: This test file has been updated to reflect the current implementation where only
relationship evolution validation is performed. Timeline and world rules validation were
removed in commit fab0d10.
"""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.langgraph.subgraphs.validation import (
    _check_relationship_evolution,
    _fetch_validation_data,
)
from tests.fakes.service_context import patch_service
from tests.test_prior_accepted_canon import accepted_row


@pytest.mark.asyncio
async def test_fetch_validation_data_combined_query(tmp_path: Path) -> None:
    """Test that _fetch_validation_data makes a single query and returns structured data."""

    state, row = accepted_row(tmp_path, 1, "FRIENDS_WITH")
    _, second = accepted_row(tmp_path, 2, "HATES")
    state["current_chapter"] = 3
    mock_results = [row, second]

    with patch_service('database') as mock_manager:
        mock_manager.execute_read_query = AsyncMock(return_value=mock_results)
        mock_manager.require_project_binding.return_value = state["graph_project_id"]

        result = await _fetch_validation_data(state)

        # Verify single query was made
        assert mock_manager.execute_read_query.call_count == 1

        # Verify structured data - only relationships are returned now
        assert "relationships" in result
        assert len(result["relationships"]) == 1

        # Verify relationship data
        assert ("Alice", "Bob") in result["relationships"]
        assert result["relationships"][("Alice", "Bob")] == [{"rel_type": "FRIENDS_WITH", "first_chapter": 1}, {"rel_type": "HATES", "first_chapter": 2}]


@pytest.mark.asyncio
async def test_check_relationship_evolution_uses_prefetched_data() -> None:
    """Test that _check_relationship_evolution uses pre-fetched data instead of querying Neo4j."""

    extracted_relationships = [type("Relationship", (), {"source_name": "Alice", "target_name": "Bob", "relationship_type": "LOVES"})()]

    existing_relationships = {("Alice", "Bob"): [{"rel_type": "HATES", "first_chapter": 1}]}

    with patch_service('database') as mock_manager:
        # Should not call execute_read_query when existing_relationships is provided
        result = await _check_relationship_evolution(extracted_relationships=extracted_relationships, current_chapter=2, existing_relationships=existing_relationships)

        assert mock_manager.execute_read_query.call_count == 0
        assert len(result) == 1


@pytest.mark.asyncio
async def test_validation_subgraph_reduces_queries(tmp_path: Path) -> None:
    """Read prior canon once while comparing real candidate artifacts."""
    from core.langgraph.subgraphs.validation import detect_contradictions

    _, row = accepted_row(tmp_path, 1, "HATES", channel="profile")
    state, _ = accepted_row(tmp_path, 3, "LOVES", channel="repeated")
    with patch_service('database') as manager:
        manager.execute_read_query = AsyncMock(return_value=[row])
        manager.require_project_binding.return_value = state["graph_project_id"]
        result = await detect_contradictions(state)
        assert manager.execute_read_query.call_count == 1
    assert [(item.type, item.conflicting_chapters) for item in result["contradictions"]] == [("relationship", [1, 3])]
    assert result["needs_revision"] is False
