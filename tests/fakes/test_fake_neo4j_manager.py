# tests/fakes/test_fake_neo4j_manager.py
"""Validate FakeNeo4jManager behavior."""

import numpy as np
import pytest

from tests.fakes.fake_neo4j_manager import FakeNeo4jManager


@pytest.fixture
def fake() -> FakeNeo4jManager:
    return FakeNeo4jManager()


class TestQueryRecording:
    async def test_records_read_queries(self, fake: FakeNeo4jManager) -> None:
        await fake.execute_read_query("MATCH (n) RETURN n", {"limit": 10})
        assert len(fake.executed_queries) == 1
        query, params = fake.executed_queries[0]
        assert query == "MATCH (n) RETURN n"
        assert params == {"limit": 10}

    async def test_records_write_queries(self, fake: FakeNeo4jManager) -> None:
        await fake.execute_write_query("CREATE (n:Test {name: $name})", {"name": "Alice"})
        assert len(fake.executed_queries) == 1
        query, params = fake.executed_queries[0]
        assert query == "CREATE (n:Test {name: $name})"
        assert params == {"name": "Alice"}

    async def test_records_batch_statements(self, fake: FakeNeo4jManager) -> None:
        statements = [
            ("MERGE (n:Chapter {number: $number})", {"number": 1}),
            ("MERGE (n:Character {name: $name})", {"name": "Alice"}),
        ]
        await fake.execute_cypher_batch(statements)
        assert len(fake.batch_statements) == 1
        assert len(fake.batch_statements[0]) == 2
        assert len(fake.executed_queries) == 2

    async def test_records_multiple_batches_separately(self, fake: FakeNeo4jManager) -> None:
        await fake.execute_cypher_batch([("Q1", {})])
        await fake.execute_cypher_batch([("Q2", {})])
        assert len(fake.batch_statements) == 2


class TestConfigurableResponses:
    async def test_returns_configured_response_for_matching_read(self, fake: FakeNeo4jManager) -> None:
        fake.configure_response(r"MATCH.*Character", [{"name": "Alice"}, {"name": "Bob"}])
        result = await fake.execute_read_query("MATCH (c:Character) RETURN c.name AS name")
        assert result == [{"name": "Alice"}, {"name": "Bob"}]

    async def test_returns_empty_list_when_no_pattern_matches(self, fake: FakeNeo4jManager) -> None:
        fake.configure_response(r"Character", [{"name": "Alice"}])
        result = await fake.execute_read_query("MATCH (n:Location) RETURN n")
        assert result == []

    async def test_returns_first_matching_response(self, fake: FakeNeo4jManager) -> None:
        fake.configure_response(r"Character", [{"name": "first"}])
        fake.configure_response(r"Character", [{"name": "second"}])
        result = await fake.execute_read_query("MATCH (c:Character) RETURN c")
        assert result == [{"name": "first"}]

    async def test_matches_case_insensitively(self, fake: FakeNeo4jManager) -> None:
        fake.configure_response(r"character", [{"name": "Alice"}])
        result = await fake.execute_read_query("MATCH (c:CHARACTER) RETURN c")
        assert result == [{"name": "Alice"}]

    async def test_write_queries_return_configured_responses(self, fake: FakeNeo4jManager) -> None:
        fake.configure_response(r"MERGE.*Chapter", [{"id": "chapter_1"}])
        result = await fake.execute_write_query("MERGE (c:Chapter {number: 1}) RETURN c")
        assert result == [{"id": "chapter_1"}]


class TestAssertionHelpers:
    async def test_assert_query_executed_succeeds_on_match(self, fake: FakeNeo4jManager) -> None:
        await fake.execute_read_query("MATCH (c:Character) RETURN c")
        query, _params = fake.assert_query_executed(r"Character")
        assert "Character" in query

    async def test_assert_query_executed_raises_on_no_match(self, fake: FakeNeo4jManager) -> None:
        await fake.execute_read_query("MATCH (c:Character) RETURN c")
        with pytest.raises(AssertionError, match=r"No executed query matched"):
            fake.assert_query_executed(r"Location")

    async def test_assert_batch_contains_succeeds_on_match(self, fake: FakeNeo4jManager) -> None:
        await fake.execute_cypher_batch([("MERGE (c:Chapter {number: $n})", {"n": 1})])
        _query, params = fake.assert_batch_contains(r"Chapter")
        assert params == {"n": 1}

    async def test_assert_batch_contains_raises_on_no_match(self, fake: FakeNeo4jManager) -> None:
        await fake.execute_cypher_batch([("MERGE (c:Chapter)", {})])
        with pytest.raises(AssertionError, match=r"No batch statement matched"):
            fake.assert_batch_contains(r"Location")


class TestReset:
    async def test_clears_all_state(self, fake: FakeNeo4jManager) -> None:
        fake.configure_response(r"test", [{"x": 1}])
        await fake.execute_read_query("test query")
        await fake.execute_cypher_batch([("batch query", {})])

        fake.reset()

        assert fake.executed_queries == []
        assert fake.batch_statements == []
        assert fake._configured_responses == []


class TestCapabilityMethods:
    async def test_apoc_available_returns_true_by_default(self, fake: FakeNeo4jManager) -> None:
        assert await fake.is_apoc_available() is True

    async def test_apoc_available_respects_override(self, fake: FakeNeo4jManager) -> None:
        fake._apoc_available = False
        assert await fake.is_apoc_available() is False

    async def test_connect_and_close_are_no_ops(self, fake: FakeNeo4jManager) -> None:
        await fake.connect()
        await fake.close()
        assert fake.executed_queries == []

    def test_embedding_to_list_converts_numpy_array(self, fake: FakeNeo4jManager) -> None:
        embedding = np.array([0.1, 0.2, 0.3])
        result = fake.embedding_to_list(embedding)
        assert result == [0.1, 0.2, 0.3]

    def test_embedding_to_list_passes_through_plain_list(self, fake: FakeNeo4jManager) -> None:
        result = fake.embedding_to_list([0.1, 0.2])
        assert result == [0.1, 0.2]
