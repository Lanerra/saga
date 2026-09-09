# tests/test_kg_queries_merge_fix.py
"""Test for the apoc.merge.node IndexEntryConflictException fix.

This test verifies that when creating relationships with entities that already exist
in the database (by name), the system correctly finds and uses those existing nodes
instead of trying to create new ones with conflicting IDs.
"""

from unittest.mock import AsyncMock

import pytest

from data_access.kg_queries import add_kg_triples_batch_to_db
from tests.fakes.service_context import patch_service


@pytest.mark.asyncio
async def test_add_kg_triples_with_existing_entity_by_name() -> None:
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
    with patch_service('database') as mock_manager:
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

        assert "CALL apoc.merge.node" in query
        assert "RETURN node AS s" in query
        assert "RETURN node AS o" in query
        assert "coalesce(supplied_id, found.id, apoc.util.sha256" in query
        assert params["subject_id"] is None
        assert params["object_id"] is None
        assert params["assertion_origin"] == "import"

        assert "CALL apoc.merge.relationship" in query


@pytest.mark.asyncio
async def test_add_kg_triples_with_literal_object() -> None:
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

    with patch_service('database') as mock_manager:
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
        assert "CALL apoc.merge.node" in query
        assert "RETURN node AS s" in query
        assert "MERGE (o:ValueNode" in query


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
