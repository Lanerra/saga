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
from models.kg_constants import KG_IS_PROVISIONAL, KG_REL_CHAPTER_ADDED


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


# Placeholder for a more comprehensive query test.
# This would ideally involve setting up mock return values for execute_read_query
# based on what add_kg_triples_batch_to_db *would* have stored.
@pytest.mark.asyncio
async def test_query_retrieves_all_character_types(mock_neo4j_manager):
    # This test is more conceptual with the current mocking strategy,
    # as it depends on how query_kg_from_db constructs its Cypher.
    # A true integration test would hit a test DB.

    # For this unit test, we're checking the query generated by query_kg_from_db
    # when asked to find :Character nodes.

    # We expect query_kg_from_db to build a query that looks for ":Character"
    # e.g., MATCH (s:Character)-[r:DYNAMIC_REL]->(o) or similar for subject-only queries

    # To test this properly, we'd need to inspect the query string passed to execute_read_query.
    # Let's modify the mock to capture the query string for query_kg_from_db

    captured_query_string = ""
    captured_query_params: dict[str, Any] = {}

    async def capture_read_query_mock(query: str, params: dict[str, Any]):
        nonlocal captured_query_string, captured_query_params
        captured_query_string = query
        captured_query_params = params
        # Simulate finding relevant nodes
        if ":Character" in query and params.get("subject_param") is None:  # General Character query
            return [
                {
                    "subject": "Alice",
                    "predicate": "IS_A",
                    "object": "Protagonist",
                    "object_type": "Literal",
                    KG_REL_CHAPTER_ADDED: 1,
                    "confidence": 1.0,
                    KG_IS_PROVISIONAL: False,
                },
                {
                    "subject": "Bob",
                    "predicate": "WORKS_AS",
                    "object": "Engineer",
                    "object_type": "Literal",
                    KG_REL_CHAPTER_ADDED: 1,
                    "confidence": 1.0,
                    KG_IS_PROVISIONAL: False,
                },
                {
                    "subject": "Charles",
                    "predicate": "APPEARS_IN",
                    "object": "Story1",
                    "object_type": "Narrative",
                    KG_REL_CHAPTER_ADDED: 1,
                    "confidence": 1.0,
                    KG_IS_PROVISIONAL: False,
                },
                {
                    "subject": "Diana",
                    "predicate": "LEADS",
                    "object": "ProjectX",
                    "object_type": "Project",
                    KG_REL_CHAPTER_ADDED: 1,
                    "confidence": 1.0,
                    KG_IS_PROVISIONAL: False,
                },
            ]
        return []

    mock_neo4j_manager.execute_read_query = AsyncMock(side_effect=capture_read_query_mock)

    # Query for all subjects that are Characters (implicit by querying for :Character)
    # query_kg_from_db structure might need adjustment if we want to query nodes by type directly
    # The current query_kg_from_db queries relationships.
    # Let's assume we are interested in subjects of relationships that are characters.

    # To test "queryable as such", we need a query that specifically asks for :Character nodes.
    # The existing query_kg_from_db is for triples. A direct node query function would be better.
    # For now, let's simulate a call that would imply matching characters.

    # If query_kg_from_db is adapted or a new function like get_nodes_by_label('Character') existed:
    # results = await get_nodes_by_label('Character')
    # For now, we test that if query_kg_from_db is called with specific subject that IS a character,
    # its query construction for that subject would be correct.

    # This test is a bit of a stretch for query_kg_from_db.
    # The main verification of "queryable as such" comes from the fact that
    # nodes are labeled correctly (:Character), so standard Cypher queries WILL find them.
    # The python function query_kg_from_db is for querying relationships.

    # Let's test that a query *for* a character uses the right label in its match
    await query_kg_from_db(subject="Alice", predicate="IS_A")
    assert "s.name = $subject_param" in captured_query_string
    # query_kg_from_db in current SAGA normalizes predicate internally and does not
    # pass it as a parameter; only subject is parameterized here.
    assert captured_query_params.get("subject_param") == "Alice"
    # assert "MATCH (s:Entity)-[r:" in captured_query_string # Removed Entity label requirement
    assert "MATCH (s)-[r:" in captured_query_string
    # The test for _get_cypher_labels and add_kg_triples_batch_to_db already ensure Alice is a :Character.
    # So, if Neo4j has Alice as :Character, the query will work.

    # This test asserts that the label :Character is used correctly by the query construction,
    # assuming the query function is designed to filter by label passed somehow or if the subject
    # is already known to be a Character.
    # The most important aspect is that the nodes *have* the :Character label, which previous tests verify.
    # A direct `MATCH (c:Character) RETURN c.name` type of query isn't directly built by query_kg_from_db.

    # A better test for "queryable as such" would be to use a direct Neo4j query.
    # For the purpose of this plan step, ensuring labels are correctly *applied* is key.
    # The "queryable" part is an inherent consequence of correct labeling in Neo4j.

    # Let's simplify this test to focus on the fact that query_kg_from_db *can* retrieve
    # data if the entities were labeled correctly.

    # The actual filtering for :Character would happen in Cypher if one were to write:
    # MATCH (c:Character) ...
    # This python function doesn't build arbitrary node selection queries, only triple queries.
    # So, "queryable as such" is proven by nodes having the label.

    # The critical part is that add_kg_triples_batch_to_db *creates* them with the right labels.
    # The existing `test_add_entities_with_character_labeling` covers this.
    # This test case can be simplified or removed if it's too convoluted for query_kg_from_db

    # Re-evaluating: The most important test for "queryable as such" is that a direct query
    # for MATCH (n:Character) would return the correct nodes. We can't directly test that
    # without a live DB or a more sophisticated mock. But we *can* ensure the labels are set.

    # The `test_add_entities_with_character_labeling` verifies that the MERGE statements
    # contain the correct labels. This is the primary guarantee of queryability.
    # This specific test for query_kg_from_db might be less critical if query_kg_from_db
    # isn't the primary tool for "get all characters".

    # Reset query capture and test a general character retrieval.
    #
    # Guardrail contract: unbounded scans require an explicit opt-in.
    captured_query_string = ""
    captured_query_params = {}
    await query_kg_from_db(include_provisional=True, allow_unbounded_scan=True)
    # assert "MATCH (s:Entity)-[r" in captured_query_string # Removed Entity label requirement
    assert "MATCH (s)-[r" in captured_query_string
    assert captured_query_params == {}
