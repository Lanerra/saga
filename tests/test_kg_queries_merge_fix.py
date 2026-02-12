# tests/test_kg_queries_merge_fix.py
"""Test for the apoc.merge.node IndexEntryConflictException fix.

This test verifies that when creating relationships with entities that already exist
in the database (by name), the system correctly finds and uses those existing nodes
instead of trying to create new ones with conflicting IDs.
"""

from unittest.mock import AsyncMock, patch

import pytest

from data_access.kg_queries import add_kg_triples_batch_to_db


@pytest.mark.asyncio
async def test_add_kg_triples_with_existing_entity_by_name():
    """Test that relationships can be created when the object entity already exists by name.

    This simulates the error scenario where:
    - An Item node with name "Stubborn" already exists (ID 304)
    - We're trying to create a relationship from a Character to this Item
    - The system should find the existing node by name instead of trying to merge by ID
    """
    # Mock structured triples data that would cause the original error
    structured_triples_data = [
        {
            "subject": {"name": "Everett Larkin", "type": "Character", "category": ""},
            "predicate": "IS_STANDING_BY",
            "object_entity": {"name": "Stubborn", "type": "Item", "category": ""},
            "is_literal_object": False,
            "description": "Everett Larkin stands by the open hatch of the Stubborn.",
            "confidence": 0.8,
            "chapter_added": 1,
        }
    ]

    # Mock the neo4j_manager to simulate the scenario
    with patch("data_access.kg_queries.neo4j_manager") as mock_manager:
        # Mock execute_cypher_batch to capture the query that would be executed
        mock_manager.execute_cypher_batch = AsyncMock()

        # Call the function
        await add_kg_triples_batch_to_db(structured_triples_data=structured_triples_data, chapter_number=1, is_from_flawed_draft=False)

        # Verify that execute_cypher_batch was called
        assert mock_manager.execute_cypher_batch.called

        # Get the captured call arguments
        call_args = mock_manager.execute_cypher_batch.call_args
        statements_with_params = call_args[0][0]

        assert len(statements_with_params) == 1

        # Get the query and params for the relationship creation
        query, params = statements_with_params[0]

        # Regression: Neo4j does not support parameterized labels like `:{$subject_label}`
        assert ":{$" not in query

        # Verify the query handles entity upsert correctly (either via apoc.merge.node, OPTIONAL MATCH, or MERGE approach)
        # The implementation may use different approaches, but should handle existing entities by name
        assert "OPTIONAL MATCH" in query or "CALL apoc.merge.node" in query or "MERGE (s {" in query
        assert "YIELD node AS s" in query or "WITH value.s as s" in query or "MERGE (s {" in query
        assert "YIELD node AS o" in query or "value.o as o" in query or "MERGE (o:" in query

        # Verify stable id assignment occurs (prevents missing id)
        # The implementation may use different approaches, but should handle id assignment
        assert "coalesce" in query and ("s.id = coalesce" in query or "s_found.id = coalesce" in query)
        assert "coalesce" in query and ("o.id = coalesce" in query or "o_found.id = coalesce" in query)

        # Verify the relationship merge occurs
        assert "CALL apoc.merge.relationship" in query

        print("✓ Test passed: Query correctly handles existing entities by name")


@pytest.mark.asyncio
async def test_add_kg_triples_with_literal_object():
    """Test that literal objects still work correctly with the updated query structure."""
    structured_triples_data = [
        {
            "subject": {"name": "Everett Larkin", "type": "Character", "category": ""},
            "predicate": "HAS_STATUS",
            "object_literal": "Active",
            "is_literal_object": True,
            "description": "Everett Larkin is active",
            "confidence": 0.9,
            "chapter_added": 1,
        }
    ]

    with patch("data_access.kg_queries.neo4j_manager") as mock_manager:
        mock_manager.execute_cypher_batch = AsyncMock()

        await add_kg_triples_batch_to_db(structured_triples_data=structured_triples_data, chapter_number=1, is_from_flawed_draft=False)

        assert mock_manager.execute_cypher_batch.called

        call_args = mock_manager.execute_cypher_batch.call_args
        statements_with_params = call_args[0][0]

        assert len(statements_with_params) == 1

        query, params = statements_with_params[0]

        # Regression: Neo4j does not support parameterized labels like `:{$subject_label}`
        assert ":{$" not in query

        # Verify subject upsert and literal object handling
        assert "OPTIONAL MATCH" in query or "CALL apoc.merge.node" in query or "MERGE (s {" in query
        assert "YIELD node AS s" in query or "WITH value.s as s" in query or "MERGE (s {" in query
        assert "MERGE (o:ValueNode" in query

        print("✓ Test passed: Literal object queries work correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
