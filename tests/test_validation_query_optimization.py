# tests/test_validation_query_optimization.py
"""Test validation query optimization - ensuring combined query reduces Neo4j round-trips.

NOTE: This test file has been updated to reflect the current implementation where only
relationship evolution validation is performed. Timeline and world rules validation were
removed in commit fab0d10.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.langgraph.subgraphs.validation import (
    _check_relationship_evolution,
    _fetch_validation_data,
)


@pytest.mark.asyncio
async def test_fetch_validation_data_combined_query():
    """Test that _fetch_validation_data makes a single query and returns structured data."""

    # Mock Neo4j response with relationship data only
    mock_results = [
        # Relationships
        {"source_name": "Alice", "target_name": "Bob", "rel_type": "FRIENDS_WITH", "chapter": 1},
        {"source_name": "Alice", "target_name": "Charlie", "rel_type": "HATES", "chapter": 2},
    ]

    with patch("core.langgraph.subgraphs.validation.neo4j_manager") as mock_manager:
        mock_manager.execute_read_query = AsyncMock(return_value=mock_results)

        result = await _fetch_validation_data(current_chapter=3)

        # Verify single query was made
        assert mock_manager.execute_read_query.call_count == 1

        # Verify structured data - only relationships are returned now
        assert "relationships" in result
        assert len(result["relationships"]) == 2

        # Verify relationship data
        assert ("Alice", "Bob") in result["relationships"]
        assert result["relationships"][("Alice", "Bob")]["rel_type"] == "FRIENDS_WITH"
        assert result["relationships"][("Alice", "Bob")]["first_chapter"] == 1


@pytest.mark.asyncio
async def test_check_relationship_evolution_uses_prefetched_data():
    """Test that _check_relationship_evolution uses pre-fetched data instead of querying Neo4j."""

    extracted_relationships = [type("Relationship", (), {"source_name": "Alice", "target_name": "Bob", "relationship_type": "LOVES"})()]

    existing_relationships = {("Alice", "Bob"): {"rel_type": "HATES", "first_chapter": 1}}

    with patch("core.langgraph.subgraphs.validation.neo4j_manager") as mock_manager:
        # Should not call execute_read_query when existing_relationships is provided
        result = await _check_relationship_evolution(extracted_relationships=extracted_relationships, current_chapter=2, existing_relationships=existing_relationships)

        assert mock_manager.execute_read_query.call_count == 0
        assert len(result) == 1


@pytest.mark.asyncio
async def test_validation_subgraph_reduces_queries():
    """Test that the validation subgraph now makes only one Neo4j query for relationships."""

    from core.langgraph.subgraphs.validation import detect_contradictions

    # Mock state with extracted data
    mock_state = {
        "current_chapter": 3,
        "extracted_events": [],  # No longer used
        "extracted_world_rules": [],  # No longer used
        "extracted_relationships": [type("Relationship", (), {"source_name": "Alice", "target_name": "Bob", "relationship_type": "LOVES"})()],
        "current_world_rules": [],  # No longer used
    }

    # Mock Neo4j to return relationship data only
    mock_validation_data = {
        "relationships": {("Alice", "Bob"): {"rel_type": "HATES", "first_chapter": 1}},
    }

    with patch("core.langgraph.subgraphs.validation.neo4j_manager") as mock_manager:
        with patch("core.langgraph.subgraphs.validation._fetch_validation_data") as mock_fetch:
            mock_fetch.return_value = mock_validation_data

            # Mock content manager and require_project_dir
            with patch("core.langgraph.subgraphs.validation.ContentManager") as mock_cm_class:
                mock_cm = MagicMock()
                mock_cm_class.return_value = mock_cm

                with patch("core.langgraph.subgraphs.validation.require_project_dir") as mock_require_dir:
                    mock_require_dir.return_value = "/tmp/test"

                    # Mock get_extracted_relationships
                    with patch("core.langgraph.subgraphs.validation.get_extracted_relationships") as mock_get_rels:
                        mock_get_rels.return_value = mock_state["extracted_relationships"]

                        result = await detect_contradictions(mock_state)

                        # Should only call _fetch_validation_data once (which makes one query)
                        assert mock_fetch.call_count == 1

                        # Should not make additional queries for relationships
                        assert mock_manager.execute_read_query.call_count == 0

                        # Should have detected contradictions
                        assert "contradictions" in result
