# tests/test_kg_heal.py
from unittest.mock import AsyncMock

import pytest

from data_access import kg_queries


@pytest.mark.asyncio
async def test_normalize_existing_relationship_types(monkeypatch):
    async def fake_read(query, params=None):
        return [{"t": "knows"}, {"t": "Is_Friend_of"}, {"t": None}]

    executed = []

    async def fake_batch(statements):
        executed.extend(statements)

    monkeypatch.setattr(
        kg_queries.neo4j_manager, "execute_read_query", AsyncMock(side_effect=fake_read)
    )
    monkeypatch.setattr(
        kg_queries.neo4j_manager,
        "execute_cypher_batch",
        AsyncMock(side_effect=fake_batch),
    )

    await kg_queries.normalize_existing_relationship_types()

    assert executed
    normalized = {params["new"] for _, params in executed}
    assert "KNOWS" in normalized
    assert "IS_FRIEND_OF" in normalized


@pytest.mark.asyncio
async def test_promote_dynamic_relationships(monkeypatch):
    async def fake_types():
        return ["KNOWS", "ALLY_OF"]

    async def fake_write(query, params=None):
        assert params == {"valid_types": ["KNOWS", "ALLY_OF"]}
        return [{"promoted": 2}]

    monkeypatch.setattr(
        kg_queries,
        "get_defined_relationship_types",
        AsyncMock(side_effect=fake_types),
    )
    monkeypatch.setattr(
        kg_queries.neo4j_manager,
        "execute_write_query",
        AsyncMock(side_effect=fake_write),
    )

    promoted = await kg_queries.promote_dynamic_relationships()
    assert promoted == 2


@pytest.mark.asyncio
async def test_deduplicate_relationships(monkeypatch):
    async def fake_write(query, params=None):
        return [{"removed": 1}]

    monkeypatch.setattr(
        kg_queries.neo4j_manager,
        "execute_write_query",
        AsyncMock(side_effect=fake_write),
    )

    removed = await kg_queries.deduplicate_relationships()
    assert removed == 1


@pytest.mark.asyncio
async def test_fetch_unresolved_dynamic_relationships(monkeypatch):
    sample = [
        {
            "rel_id": 1,
            "subject": "Alice",
            "subject_labels": ["Character"],
            "subject_desc": "brave hero",
            "object": "Bob",
            "object_labels": ["Character"],
            "object_desc": "villain",
            "type": "UNKNOWN",
        }
    ]

    async def fake_read(query, params=None):
        assert "UNKNOWN" in query
        return sample

    monkeypatch.setattr(
        kg_queries.neo4j_manager,
        "execute_read_query",
        AsyncMock(side_effect=fake_read),
    )

    results = await kg_queries.fetch_unresolved_dynamic_relationships()
    assert results == sample


@pytest.mark.asyncio
async def test_update_dynamic_relationship_type(monkeypatch):
    executed = []

    async def fake_write(query, params=None):
        executed.append(params)

    monkeypatch.setattr(
        kg_queries.neo4j_manager,
        "execute_write_query",
        AsyncMock(side_effect=fake_write),
    )

    await kg_queries.update_dynamic_relationship_type(5, "ALLY_OF")
    assert executed and executed[0]["id"] == 5 and executed[0]["type"] == "ALLY_OF"


@pytest.mark.asyncio
async def test_get_shortest_path_length_between_entities(monkeypatch):
    captured = {}

    async def fake_read(query, params=None):
        captured["query"] = query
        captured["params"] = params
        return [{"len": 2}]

    monkeypatch.setattr(
        kg_queries.neo4j_manager,
        "execute_read_query",
        AsyncMock(side_effect=fake_read),
    )

    length = await kg_queries.get_shortest_path_length_between_entities(
        "Alice",
        "Bob",
        max_depth=3,
    )
    assert length == 2
    assert "[*..3]" in captured["query"]
    assert captured["params"] == {"name1": "Alice", "name2": "Bob"}
