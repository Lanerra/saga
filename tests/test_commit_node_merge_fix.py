# tests/test_commit_node_merge_fix.py
"""Test for the commit_node apoc.merge.node IndexEntryConflictException fix.

This test verifies that when creating relationships with entities that already exist
in the database (by name), the commit_node correctly finds and uses those existing nodes
instead of trying to create new ones with conflicting IDs.
"""

import pytest

from core.langgraph.nodes.commit_node import _build_relationship_statements
from core.langgraph.state import ExtractedRelationship


@pytest.mark.asyncio
async def test_build_relationship_statements_with_existing_entity_by_name():
    """Test that relationships can be created when entities already exist by name.

    This simulates the error scenario where:
    - A Location node with name "Heron" already exists (ID 316)
    - We're trying to create a relationship from this Location to an Item
    - The system should find the existing node by name instead of trying to merge by ID
    """
    # Mock relationship data that would cause the original error
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

    # Verify the query uses apoc.merge.node upserts for both endpoints (single-flow Cypher; no UNION/RETURN mid-query)
    assert "CALL apoc.merge.node" in query
    assert "YIELD node AS s" in query
    assert "YIELD node AS o" in query

    # Verify stable id assignment occurs
    assert "SET s.id = coalesce" in query
    assert "SET o.id = coalesce" in query

    # Verify relationship merge and explicit end-of-query return
    assert "CALL apoc.merge.relationship" in query
    assert "RETURN rel" in query

    print("✓ Test passed: Query correctly handles existing entities by name")


@pytest.mark.asyncio
async def test_build_relationship_statements_with_new_entities():
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
    assert "YIELD node AS s" in query
    assert "YIELD node AS o" in query

    # Verify it creates/ensures ids via randomUUID() fallback
    assert "randomUUID()" in query
    assert "SET s.id = coalesce" in query
    assert "SET o.id = coalesce" in query

    # Verify relationship merge and explicit end-of-query return
    assert "CALL apoc.merge.relationship" in query
    assert "RETURN rel" in query

    print("✓ Test passed: New entities are created correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
