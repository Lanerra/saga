# tests/test_character_labeling.py
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.db_manager import Neo4jManagerSingleton
from data_access.kg_queries import (
    _get_cypher_labels,
    add_kg_triples_batch_to_db,
    query_kg_from_db,
)
from tests.fakes.service_context import patch_service

pytestmark = pytest.mark.usefixtures("owned_graph_cache")


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
def test_get_cypher_labels_various_types(entity_type: str, expected_labels: str) -> None:
    assert _get_cypher_labels(entity_type) == expected_labels


def test_get_cypher_labels_character_is_primary() -> None:
    # Ensure if type is "Character", it doesn't become :Character:Character:Entity
    assert _get_cypher_labels("Character") == ":Character"
    # Ensure if type is "Person", it is normalized to Character
    assert _get_cypher_labels("Person") == ":Character"


# Mocking Neo4j interactions for add_kg_triples_batch_to_db and query_kg_from_db
@pytest.fixture
def mock_neo4j_manager() -> Iterator[MagicMock]:
    with patch_service('database', spec=Neo4jManagerSingleton) as mock_manager:
        mock_manager.execute_cypher_batch = AsyncMock(return_value=None)
        # Simplistic mock for query_kg_from_db, will be updated by test logic
        mock_manager.execute_read_query = AsyncMock(return_value=[])
        yield mock_manager


# Store for captured statements by the mock
captured_statements_for_tests: list[tuple[str, dict[str, Any]]] = []


async def capture_statements_mock(
    statements: list[tuple[str, dict[str, Any]]],
) -> None:
    captured_statements_for_tests.clear()
    captured_statements_for_tests.extend(statements)
    return None


@pytest.mark.asyncio
async def test_add_entities_with_character_labeling(mock_neo4j_manager: MagicMock) -> None:
    captured_statements_for_tests.clear()
    # Override the mock for execute_cypher_batch for this test to capture statements
    mock_neo4j_manager.execute_cypher_batch = AsyncMock(side_effect=capture_statements_mock)

    triples_data: list[dict[str, object]] = [
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

    assert len(captured_statements_for_tests) == 5
    for endpoint, name, label in [("subject", "Alice", "Character"), ("subject", "Bob", "Character"),
                                  ("subject", "Castle", "Location"), ("object", "Charles", "Character"), ("object", "Diana", "Character")]:
        matching = [(query, parameters) for query, parameters in captured_statements_for_tests if parameters[f"{endpoint}_name"] == name]
        assert len(matching) == 1
        query, parameters = matching[0]
        assert parameters[f"{endpoint}_label"] == label
        assert "CALL apoc.merge.node" in query
        assert "Ambiguous canonical entity" in query


@pytest.mark.asyncio
async def test_query_retrieves_all_character_types(mock_neo4j_manager: MagicMock) -> None:
    """query_kg_from_db constructs correct Cypher for subject and unbounded queries."""
    captured_query_string = ""
    captured_query_params: dict[str, Any] = {}

    async def capture_read_query(query: str, params: dict[str, Any]) -> list[dict[str, object]]:
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
