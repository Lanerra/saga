import pytest

from prompt_data_getters import get_planning_context_from_kg


@pytest.mark.asyncio
async def test_get_planning_context_from_kg(monkeypatch):
    async def fake_get_novel_info_property_from_db(key):
        return {"theme": "courage", "central_conflict": "war"}.get(key)

    async def fake_get_most_recent_value_from_db(
        subject, predicate, chapter_limit=None, include_provisional=False
    ):
        if subject == "Alice" and predicate == "status_is":
            return "Alive"
        if subject == "Alice" and predicate == "located_in":
            return "Town"
        return None

    async def fake_query_kg_from_db(*args, **kwargs):
        return [{"predicate": "ALLY_OF", "object": "Bob"}]

    monkeypatch.setattr(
        "data_access.kg_queries.get_novel_info_property_from_db",
        fake_get_novel_info_property_from_db,
    )
    monkeypatch.setattr(
        "data_access.kg_queries.get_most_recent_value_from_db",
        fake_get_most_recent_value_from_db,
    )
    monkeypatch.setattr(
        "data_access.kg_queries.query_kg_from_db",
        fake_query_kg_from_db,
    )

    result = await get_planning_context_from_kg(
        {"protagonist_name": "Alice"}, 3, max_facts_per_char=3
    )

    assert "Novel theme: courage" in result
    assert "Alice's status is: Alive" in result
    assert "relationship (ALLY OF)" in result
