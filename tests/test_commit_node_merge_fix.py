# tests/test_commit_node_merge_fix.py
"""Test for the commit_node apoc.merge.node IndexEntryConflictException fix.

This test verifies that when creating relationships with entities that already exist
in the database (by name), the commit_node correctly finds and uses those existing nodes
instead of trying to create new ones with conflicting IDs.
"""

import pytest

from core.langgraph.nodes.commit_node import _build_relationship_statements
from core.langgraph.state import ExtractedRelationship
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager


@pytest.mark.asyncio
async def test_build_relationship_statements_with_existing_entity_by_name(offline_graph_reads: FakeNeo4jManager) -> None:
    """Test that relationships can be created when entities already exist by name.

    This simulates the error scenario where:
    - A Location node with name "Heron" already exists (ID 316)
    - We're trying to create a relationship from this Location to an Item
    - The system should find the existing node by name instead of trying to merge by ID
    """

    relationships = [
        ExtractedRelationship(
            source_name="Heron",
            target_name="Tug",
            relationship_type="IS_TYPE_OF",
            description="Heron is a type of Tug.",
            chapter=2,
            confidence=0.8,
            source_type="Location",
            target_type="Item",
        )
    ]

    # Call the function to build statements
    statements = await _build_relationship_statements(
        relationships=relationships,
        char_entities=[],
        world_entities=[],
        char_mappings={},
        world_mappings={},
        chapter=2,
        is_from_flawed_draft=False,
    )

    # Verify we have statements (first is delete query, second is the actual relationship creation)
    assert len(statements) >= 2

    # Get the query and params for the relationship creation (skip the delete query at index 0)
    query, params = statements[1]

    # Regression: Neo4j does not support parameterized labels like `:{$subject_label}`
    assert ":{$" not in query

    # Endpoint resolution belongs to the same transaction as the relationship write.
    assert "CALL apoc.merge.node" in query
    assert "RETURN node AS s" in query
    assert "RETURN node AS o" in query
    assert len(statements) == 2
    assert params["subject_id"] is None
    assert offline_graph_reads.executed_queries == []
    assert "coalesce(supplied_id, found.id, apoc.util.sha256" in query
    assert "Ambiguous canonical entity" in query

    assert "CALL apoc.merge.relationship" in query
    assert "RETURN rel" in query


@pytest.mark.asyncio
async def test_build_relationship_statements_with_new_entities(offline_graph_reads: FakeNeo4jManager) -> None:
    """Test that new entities are created correctly when they don't exist."""
    # Mock relationship data with new entities
    relationships = [
        ExtractedRelationship(
            source_name="NewLocation",
            target_name="NewItem",
            relationship_type="IS_TYPE_OF",
            description="NewLocation is a type of NewItem.",
            chapter=2,
            confidence=0.8,
            source_type="Location",
            target_type="Item",
        )
    ]

    # Call the function to build statements
    statements = await _build_relationship_statements(
        relationships=relationships,
        char_entities=[],
        world_entities=[],
        char_mappings={},
        world_mappings={},
        chapter=2,
        is_from_flawed_draft=False,
    )

    # Verify we have statements (first is delete query, second is the actual relationship creation)
    assert len(statements) >= 2

    # Get the query and params for the relationship creation (skip the delete query at index 0)
    query, params = statements[1]

    # Regression: Neo4j does not support parameterized labels like `:{$subject_label}`
    assert ":{$" not in query

    # Verify the query uses apoc.merge.node upserts for both endpoints
    assert "CALL apoc.merge.node" in query
    assert "RETURN node AS s" in query
    assert "RETURN node AS o" in query
    assert len(statements) == 2
    assert params["subject_id"] is None
    assert params["object_id"] is None
    assert "coalesce(supplied_id, found.id, apoc.util.sha256" in query

    assert "CALL apoc.merge.relationship" in query
    assert "RETURN rel" in query


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
