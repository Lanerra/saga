# tests/test_character_labeling.py
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from core.db_manager import (
    Neo4jManagerSingleton,
)  # Needed for type hinting if neo4j_manager is mocked

# Assuming data_access.kg_queries is the path. Adjust if necessary based on project structure.
from data_access.kg_queries import (
    _get_cypher_labels,
    add_kg_triples_batch_to_db,
    query_kg_from_db,
)


# Test cases for _get_cypher_labels
@pytest.mark.parametrize(
    "entity_type, expected_labels",
    [
        ("Character", ":Character"),
        (
            "character",
            ":Character",
        ),  # Test case-insensitivity for "Character"
        (
            "Person",
            ":Character",
        ),  # Person should get normalized to Character
        # "person" removed as lowercase types are rejected by strict schema validation
        ("Location", ":Location"),
        ("Event", ":Event"),
        ("  Item ", ":Item"),  # Test stripping whitespace
        # Removed complex and invalid types as they are now rejected by schema enforcement
        # Empty/None types are now rejected by strict schema validation
    ],
)
def test_get_cypher_labels_various_types(entity_type, expected_labels):
    assert _get_cypher_labels(entity_type) == expected_labels


def test_get_cypher_labels_character_is_primary():
    # Ensure if type is "Character", it doesn't become :Character:Character:Entity
    assert _get_cypher_labels("Character") == ":Character"
    # Ensure if type is "Person", it is normalized to Character
    assert _get_cypher_labels("Person") == ":Character"


# Mocking Neo4j interactions for add_kg_triples_batch_to_db and query_kg_from_db
@pytest.fixture
def mock_neo4j_manager():
    with patch("data_access.kg_queries.neo4j_manager", spec=Neo4jManagerSingleton) as mock_manager:
        mock_manager.execute_cypher_batch = AsyncMock(return_value=None)
        # Simplistic mock for query_kg_from_db, will be updated by test logic
        mock_manager.execute_read_query = AsyncMock(return_value=[])
        yield mock_manager


# Store for captured statements by the mock
captured_statements_for_tests: list[tuple[str, dict[str, Any]]] = []


async def capture_statements_mock(
    statements: list[tuple[str, dict[str, Any]]],
):
    captured_statements_for_tests.clear()
    captured_statements_for_tests.extend(statements)
    return None


@pytest.mark.asyncio
async def test_add_entities_with_character_labeling(mock_neo4j_manager):
    captured_statements_for_tests.clear()
    # Override the mock for execute_cypher_batch for this test to capture statements
    mock_neo4j_manager.execute_cypher_batch = AsyncMock(side_effect=capture_statements_mock)

    triples_data = [
        # Scenario 1: Explicit Character type
        {
            "subject": {"name": "Alice", "type": "Character"},
            "predicate": "IS_A",
            "object_literal": "Protagonist",
            "is_literal_object": True,
        },
        # Scenario 2: Person type, should also get Character label
        {
            "subject": {"name": "Bob", "type": "Person"},
            "predicate": "WORKS_AS",
            "object_literal": "Engineer",
            "is_literal_object": True,
        },
        # Scenario 3: Other type
        {
            "subject": {"name": "Castle", "type": "Location"},
            "predicate": "IS_NEAR",
            "object_literal": "Forest",
            "is_literal_object": True,
        },
        # Scenario 4: Character as object
        {
            "subject": {"name": "Story1", "type": "Concept"},  # Narrative -> Concept
            "predicate": "FEATURES",
            "object_entity": {"name": "Charles", "type": "Character"},
        },
        # Scenario 5: Person as object
        {
            "subject": {
                "name": "ProjectX",
                "type": "Organization",
            },  # Project -> Organization
            "predicate": "MANAGED_BY",
            "object_entity": {"name": "Diana", "type": "Person"},
        },
    ]

    await add_kg_triples_batch_to_db(triples_data, chapter_number=1, is_from_flawed_draft=False)

    # Debug: Print captured statements
    # for i, (query, params) in enumerate(captured_statements_for_tests):
    #     print(f"Statement {i}:")
    #     print(f"  Query: {query.strip()}")
    #     print(f"  Params: {params}")
    #     print("-" * 20)

    # Verify generated Cypher for Alice (Character)
    alice_statement_found = False
    for query, params in captured_statements_for_tests:
        if params.get("subject_name_param") == "Alice":
            # Contract: we use constraint-safe merges (either MERGE with ID or apoc.merge.node).
            # When subject_id is available (for Characters), it uses MERGE with ID.
            # Otherwise, it uses apoc.merge.node with name.
            assert any(merge_type in query for merge_type in ["CALL apoc.merge.node", "MERGE (s {"])
            assert params.get("subject_label") == "Character"
            assert params.get("subject_name_param") == "Alice"
            alice_statement_found = True
            break
    assert alice_statement_found, "Cypher statement for Alice as Character not found or incorrect."

    # Verify generated Cypher for Bob (Person -> Character)
    bob_statement_found = False
    for query, params in captured_statements_for_tests:
        if params.get("subject_name_param") == "Bob":
            # Contract: type normalization updates subject_label to canonical "Character".
            assert any(merge_type in query for merge_type in ["CALL apoc.merge.node", "MERGE (s {"])
            assert params.get("subject_label") == "Character"
            assert params.get("subject_name_param") == "Bob"
            bob_statement_found = True
            break
    assert bob_statement_found, "Cypher statement for Bob as Person->Character not found or incorrect."

    # Verify generated Cypher for Castle (Location)
    castle_statement_found = False
    for query, params in captured_statements_for_tests:
        if params.get("subject_name_param") == "Castle":
            assert "CALL apoc.merge.node" in query
            assert params.get("subject_label") == "Location"
            assert params.get("subject_name_param") == "Castle"
            castle_statement_found = True
            break
    assert castle_statement_found, "Cypher statement for Castle as Location not found or incorrect."

    # Verify Charles (Object, Character)
    charles_statement_found = False
    for query, params in captured_statements_for_tests:
        if params.get("object_name_param") == "Charles":
            # Object entities use either apoc.merge.node or apoc.do.when depending on whether ID is available
            assert any(merge_type in query for merge_type in ["CALL apoc.merge.node", "CALL apoc.do.when"])
            assert params.get("object_label") == "Character"
            assert params.get("object_name_param") == "Charles"
            charles_statement_found = True
            break
    assert charles_statement_found, "Cypher statement for Charles as Character (object) not found or incorrect."

    # Verify Diana (Object, Person -> Character)
    diana_statement_found = False
    for query, params in captured_statements_for_tests:
        if params.get("object_name_param") == "Diana":
            # Object entities use either apoc.merge.node or apoc.do.when depending on whether ID is available
            assert any(merge_type in query for merge_type in ["CALL apoc.merge.node", "CALL apoc.do.when"])
            assert params.get("object_label") == "Character"
            assert params.get("object_name_param") == "Diana"
            diana_statement_found = True
            break
    assert diana_statement_found, "Cypher statement for Diana as Person->Character (object) not found or incorrect."


@pytest.mark.asyncio
async def test_query_retrieves_all_character_types(mock_neo4j_manager):
    """query_kg_from_db constructs correct Cypher for subject and unbounded queries."""
    captured_query_string = ""
    captured_query_params: dict[str, Any] = {}

    async def capture_read_query(query: str, params: dict[str, Any]):
        nonlocal captured_query_string, captured_query_params
        captured_query_string = query
        captured_query_params = params
        return []

    mock_neo4j_manager.execute_read_query = AsyncMock(side_effect=capture_read_query)

    await query_kg_from_db(subject="Alice", predicate="IS_A")
    assert "s.name = $subject_param" in captured_query_string
    assert captured_query_params.get("subject_param") == "Alice"
    assert "MATCH (s)-[r:" in captured_query_string

    captured_query_string = ""
    captured_query_params = {}
    await query_kg_from_db(include_provisional=True, allow_unbounded_scan=True)
    assert "MATCH (s)-[r" in captured_query_string
    assert captured_query_params == {}
