# tests/test_data_access_single_fetch.py
from unittest.mock import AsyncMock

import pytest

import utils
from data_access import character_queries, world_queries
from kg_constants import KG_NODE_CREATED_CHAPTER
from models import WorldItem


@pytest.mark.asyncio
async def test_get_character_profile_by_name(monkeypatch):
    async def fake_read(query, params=None):
        if "RETURN c" in query:
            return [
                {
                    "c": {
                        "name": "Alice",
                        "description": "hero",
                        "status": "active",
                        "created_ts": 1,
                    }
                }
            ]
        if "HAS_TRAIT_ASPECT" in query:
            return [{"trait_name": "brave"}]
        if "DYNAMIC_REL" in query:
            return [{"target_name": "Bob", "rel_props": {"type": "KNOWS"}}]
        if "DEVELOPED_IN_CHAPTER" in query:
            return [
                {
                    "summary": "growth",
                    "chapter": 1,
                    "is_provisional": False,
                    "dev_id": "d1",
                }
            ]
        return []

    monkeypatch.setattr(
        character_queries.neo4j_manager,
        "execute_read_query",
        AsyncMock(side_effect=fake_read),
    )

    profile = await character_queries.get_character_profile_by_name("Alice")
    assert profile
    assert profile.name == "Alice"
    assert profile.traits == ["brave"]
    assert profile.relationships["Bob"]["type"] == "KNOWS"
    assert profile.updates["development_in_chapter_1"] == "growth"

    character_queries.get_character_profile_by_name.cache_clear()


@pytest.mark.asyncio
async def test_get_world_item_by_id(monkeypatch):
    async def fake_read(query, params=None):
        if "RETURN we" in query:
            return [
                {
                    "we": {
                        "id": "places_city",
                        "name": "City",
                        "category": "places",
                        KG_NODE_CREATED_CHAPTER: 1,
                    }
                }
            ]
        if "HAS_GOAL" in query:
            return [{"item_value": "Thrive"}]
        if (
            "HAS_RULE" in query
            or "HAS_KEY_ELEMENT" in query
            or "HAS_TRAIT_ASPECT" in query
        ):
            return []
        if "ELABORATED_IN_CHAPTER" in query:
            return [{"summary": "history", "chapter": 2, "is_provisional": False}]
        return []

    monkeypatch.setattr(
        world_queries.neo4j_manager,
        "execute_read_query",
        AsyncMock(side_effect=fake_read),
    )

    item = await world_queries.get_world_item_by_id("places_city")
    assert item
    assert item.name == "City"
    assert item.category == "places"
    assert item.goals == ["Thrive"]
    assert item.additional_properties["elaboration_in_chapter_2"] == "history"

    world_queries.get_world_item_by_id.cache_clear()


@pytest.mark.asyncio
async def test_get_world_item_by_id_fallback(monkeypatch):
    async def fake_read(query, params=None):
        if params and params.get("id") == "places_city":
            return [
                {
                    "we": {
                        "id": "places_city",
                        "name": "City",
                        "category": "places",
                        KG_NODE_CREATED_CHAPTER: 1,
                    }
                }
            ]
        return []

    world_queries.WORLD_NAME_TO_ID.clear()
    world_queries.WORLD_NAME_TO_ID[utils._normalize_for_id("City")] = "places_city"
    monkeypatch.setattr(
        world_queries.neo4j_manager,
        "execute_read_query",
        AsyncMock(side_effect=fake_read),
    )

    item = await world_queries.get_world_item_by_id("City")
    assert item
    assert item.id == "places_city"

    world_queries.get_world_item_by_id.cache_clear()


@pytest.mark.asyncio
async def test_sync_world_items_populates_name_to_id(monkeypatch):
    world_item = WorldItem.from_dict("Places", "City", {"description": "desc"})
    world_data = {"Places": {"City": world_item}}

    monkeypatch.setattr(
        world_queries,
        "generate_world_element_node_cypher",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        world_queries.neo4j_manager,
        "execute_cypher_batch",
        AsyncMock(return_value=None),
    )

    world_queries.WORLD_NAME_TO_ID.clear()
    await world_queries.sync_world_items(world_data, 1)
    assert (
        world_queries.WORLD_NAME_TO_ID[utils._normalize_for_id("City")] == world_item.id
    )


@pytest.mark.asyncio
async def test_get_world_building_from_db_populates_name_to_id(monkeypatch):
    async def fake_read(query, params=None):
        if "RETURN wc" in query:
            return [{"wc": {"overview_description": "desc"}}]
        if "RETURN we" in query:
            return [
                {
                    "we": {
                        "id": "places_city",
                        "name": "City",
                        "category": "places",
                        "created_ts": 1,
                        KG_NODE_CREATED_CHAPTER: 1,
                    }
                }
            ]
        return []

    monkeypatch.setattr(
        world_queries.neo4j_manager,
        "execute_read_query",
        AsyncMock(side_effect=fake_read),
    )

    world_queries.WORLD_NAME_TO_ID.clear()
    world_data = await world_queries.get_world_building_from_db()
    assert world_data["places"]["City"].id == "places_city"
    assert (
        world_queries.WORLD_NAME_TO_ID[utils._normalize_for_id("City")] == "places_city"
    )
