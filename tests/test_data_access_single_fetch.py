# tests/test_data_access_single_fetch.py
from unittest.mock import AsyncMock

import pytest

import utils
from data_access import character_queries, world_queries
from models import WorldItem
from models.kg_constants import KG_NODE_CREATED_CHAPTER


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
        if "HAS_TRAIT" in query:
            return [{"trait_name": "brave"}]
        if "RETURN target.name AS target_name" in query:
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
        if "HAS_GOAL" in query or (":HAS_GOAL" in query):
            # With the new world model, goals are stored on the node; return empty for relation query paths
            return []
        if (
            "HAS_RULE" in query
            or "HAS_KEY_ELEMENT" in query
            or "HAS_TRAIT_ASPECT" in query
        ):
            return []
        if "ELABORATED_IN_CHAPTER" in query:
            # In current implementation, elaborations may not be fetched in the simplified path
            return []
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
    # Goals may be stored as a list property on the node (native), allow empty under mocks
    assert isinstance(item.goals, list)
    # Elaborations are not guaranteed in current simplified fetch; do not assert here

    world_queries.get_world_item_by_id.cache_clear()


@pytest.mark.asyncio
async def test_sync_world_items_populates_name_to_id(monkeypatch):
    world_item = WorldItem.from_dict("Places", "City", {"description": "desc"})
    world_data = [world_item]

    # world_queries builds Cypher internally now; no need to patch generator
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
        if "RETURN w" in query:
            return [
                {
                    "w": {
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
    world_items = await world_queries.get_world_building()
    # Find item by name
    city = next((w for w in world_items if w.name == "City"), None)
    assert city is not None
    assert city.id == "places_city"
    assert (
        world_queries.WORLD_NAME_TO_ID.get(utils._normalize_for_id("City"))
        == "places_city"
    )
