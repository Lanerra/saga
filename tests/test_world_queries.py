"""Tests for data_access/world_queries.py"""
from unittest.mock import AsyncMock

import pytest

import utils
from data_access import world_queries
from models import WorldItem
from models.kg_constants import KG_NODE_CREATED_CHAPTER


class TestWorldNameResolution:
    """Tests for world name resolution."""

    def test_resolve_world_name_exists(self):
        """Test resolving world name that exists."""
        world_queries.WORLD_NAME_TO_ID.clear()
        world_queries.WORLD_NAME_TO_ID["castle"] = "locations_castle"

        result = world_queries.resolve_world_name("Castle")
        assert result == "locations_castle"

    def test_resolve_world_name_missing(self):
        """Test resolving world name that doesn't exist."""
        world_queries.WORLD_NAME_TO_ID.clear()

        result = world_queries.resolve_world_name("Unknown")
        assert result is None

    def test_resolve_world_name_empty(self):
        """Test resolving empty world name."""
        result = world_queries.resolve_world_name("")
        assert result is None


class TestWorldItemByName:
    """Tests for getting world items by name."""

    def test_get_world_item_by_name_found(self):
        """Test getting world item by name when found."""
        world_queries.WORLD_NAME_TO_ID.clear()
        world_queries.WORLD_NAME_TO_ID["castle"] = "locations_castle"

        castle_item = WorldItem.from_dict("Locations", "Castle", {"description": "A castle"})
        castle_item.id = "locations_castle"
        world_data = {
            "Locations": {
                "Castle": castle_item
            }
        }

        result = world_queries.get_world_item_by_name(world_data, "Castle")
        assert result is not None
        assert result.name == "Castle"

    def test_get_world_item_by_name_not_found(self):
        """Test getting world item by name when not found."""
        world_queries.WORLD_NAME_TO_ID.clear()
        world_data = {
            "Locations": {}
        }

        result = world_queries.get_world_item_by_name(world_data, "Unknown")
        assert result is None

    def test_get_world_item_by_name_case_insensitive(self):
        """Test getting world item by name case insensitive."""
        world_queries.WORLD_NAME_TO_ID.clear()
        world_queries.WORLD_NAME_TO_ID["castle"] = "locations_castle"

        castle_item = WorldItem.from_dict("Locations", "Castle", {"description": "A castle"})
        castle_item.id = "locations_castle"
        world_data = {
            "Locations": {
                "Castle": castle_item
            }
        }

        result = world_queries.get_world_item_by_name(world_data, "castle")
        assert result is not None


@pytest.mark.asyncio
class TestSyncWorldItems:
    """Tests for syncing world items to database."""

    async def test_sync_world_items_empty(self, monkeypatch):
        """Test syncing empty world items list."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_cypher_batch", mock_execute
        )

        world_queries.WORLD_NAME_TO_ID.clear()
        result = await world_queries.sync_world_items([], 1)
        assert result is True

    async def test_sync_world_items_single_item(self, monkeypatch):
        """Test syncing single world item."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_cypher_batch", mock_execute
        )

        from unittest.mock import MagicMock
        mock_builder = MagicMock()
        mock_builder.batch_world_item_upsert_cypher.return_value = [("MERGE (w:Location)", {})]
        monkeypatch.setattr("data_access.world_queries.NativeCypherBuilder", lambda: mock_builder)

        world_queries.WORLD_NAME_TO_ID.clear()
        item = WorldItem.from_dict("Locations", "Castle", {"description": "A castle"})
        result = await world_queries.sync_world_items([item], 1)

        assert result is True
        assert utils._normalize_for_id("Castle") in world_queries.WORLD_NAME_TO_ID

    async def test_sync_world_items_multiple(self, monkeypatch):
        """Test syncing multiple world items."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_cypher_batch", mock_execute
        )

        from unittest.mock import MagicMock
        mock_builder = MagicMock()
        mock_builder.batch_world_item_upsert_cypher.return_value = [
            ("MERGE (w:Location)", {}),
            ("MERGE (w:Item)", {}),
        ]
        monkeypatch.setattr("data_access.world_queries.NativeCypherBuilder", lambda: mock_builder)

        world_queries.WORLD_NAME_TO_ID.clear()
        items = [
            WorldItem.from_dict("Locations", "Castle", {"description": "A castle"}),
            WorldItem.from_dict("Items", "Sword", {"description": "A sword"}),
        ]
        result = await world_queries.sync_world_items(items, 1)

        assert result is True
        assert len(world_queries.WORLD_NAME_TO_ID) == 2


@pytest.mark.asyncio
class TestGetWorldItemById:
    """Tests for getting world item by ID."""

    async def test_get_world_item_by_id_found(self, monkeypatch):
        """Test getting world item by ID when found."""
        mock_read = AsyncMock(
            return_value=[
                {
                    "we": {
                        "id": "locations_castle",
                        "name": "Castle",
                        "category": "Locations",
                        KG_NODE_CREATED_CHAPTER: 1,
                    }
                }
            ]
        )
        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await world_queries.get_world_item_by_id("locations_castle")
        assert result is not None
        assert result.name == "Castle"

    async def test_get_world_item_by_id_not_found(self, monkeypatch):
        """Test getting world item by ID when not found."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await world_queries.get_world_item_by_id("unknown_id")
        assert result is None

    async def test_get_world_item_by_id_with_fallback(self, monkeypatch):
        """Test getting world item by ID with name fallback."""
        world_queries.WORLD_NAME_TO_ID.clear()
        world_queries.WORLD_NAME_TO_ID["castle"] = "locations_castle"

        mock_read = AsyncMock(
            return_value=[
                {
                    "we": {
                        "id": "locations_castle",
                        "name": "Castle",
                        "category": "Locations",
                        KG_NODE_CREATED_CHAPTER: 1,
                    }
                }
            ]
        )
        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await world_queries.get_world_item_by_id("Castle")
        assert result is not None


@pytest.mark.asyncio
class TestGetWorldBuilding:
    """Tests for getting all world building items."""

    async def test_get_world_building_empty(self, monkeypatch):
        """Test getting world building when empty."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_read_query", mock_read
        )

        world_queries.WORLD_NAME_TO_ID.clear()
        result = await world_queries.get_world_building()
        assert isinstance(result, list)
        assert len(result) == 0

    async def test_get_world_building_with_items(self, monkeypatch):
        """Test getting world building with items."""
        async def fake_read(query, params=None):
            if "RETURN wc" in query:
                return [{"wc": {"overview_description": "World overview"}}]
            if "RETURN w" in query:
                return [
                    {
                        "w": {
                            "id": "locations_castle",
                            "name": "Castle",
                            "category": "Locations",
                            KG_NODE_CREATED_CHAPTER: 1,
                        }
                    },
                    {
                        "w": {
                            "id": "items_sword",
                            "name": "Sword",
                            "category": "Items",
                            KG_NODE_CREATED_CHAPTER: 1,
                        }
                    },
                ]
            return []

        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_read_query", AsyncMock(side_effect=fake_read)
        )

        world_queries.WORLD_NAME_TO_ID.clear()
        result = await world_queries.get_world_building()
        assert len(result) == 2
        assert world_queries.WORLD_NAME_TO_ID["castle"] == "locations_castle"
        assert world_queries.WORLD_NAME_TO_ID["sword"] == "items_sword"


@pytest.mark.asyncio
class TestGetWorldElementsForSnippet:
    """Tests for getting world elements for snippet."""

    async def test_get_world_elements_for_snippet_empty(self, monkeypatch):
        """Test getting world elements when none exist."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await world_queries.get_world_elements_for_snippet_from_db("Locations", 10, 5)
        assert isinstance(result, list)

    async def test_get_world_elements_for_snippet_found(self, monkeypatch):
        """Test getting world elements that exist."""
        mock_read = AsyncMock(
            return_value=[
                {
                    "name": "Castle",
                    "description": "A grand castle",
                    "is_provisional": False,
                }
            ]
        )
        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await world_queries.get_world_elements_for_snippet_from_db("Locations", 10, 5)
        assert len(result) > 0


@pytest.mark.asyncio
class TestFindThinWorldElements:
    """Tests for finding thin world elements for enrichment."""

    async def test_find_thin_world_elements_empty(self, monkeypatch):
        """Test finding thin world elements when none exist."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await world_queries.find_thin_world_elements_for_enrichment()
        assert isinstance(result, list)
        assert len(result) == 0

    async def test_find_thin_world_elements_found(self, monkeypatch):
        """Test finding thin world elements that exist."""
        mock_read = AsyncMock(
            return_value=[
                {
                    "id": "locations_castle",
                    "name": "Castle",
                    "description": "A castle",
                }
            ]
        )
        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await world_queries.find_thin_world_elements_for_enrichment()
        assert len(result) > 0


@pytest.mark.asyncio
class TestGetWorldItemsForChapterContext:
    """Tests for getting world items for chapter context."""

    async def test_get_world_items_for_chapter_context_empty(self, monkeypatch):
        """Test getting world items when none exist."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await world_queries.get_world_items_for_chapter_context_native(
            chapter_number=1, limit=10
        )
        assert isinstance(result, list)

    async def test_get_world_items_for_chapter_context_found(self, monkeypatch):
        """Test getting world items that exist."""
        mock_read = AsyncMock(
            return_value=[
                {
                    "w": {
                        "id": "locations_castle",
                        "name": "Castle",
                        "category": "Locations",
                        KG_NODE_CREATED_CHAPTER: 1,
                    }
                }
            ]
        )
        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await world_queries.get_world_items_for_chapter_context_native(
            chapter_number=1, limit=10
        )
        assert len(result) > 0


@pytest.mark.asyncio
class TestGetBootstrapWorldElements:
    """Tests for getting bootstrap world elements."""

    async def test_get_bootstrap_world_elements_empty(self, monkeypatch):
        """Test getting bootstrap world elements when none exist."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await world_queries.get_bootstrap_world_elements()
        assert isinstance(result, list)

    async def test_get_bootstrap_world_elements_found(self, monkeypatch):
        """Test getting bootstrap world elements that exist."""
        mock_read = AsyncMock(
            return_value=[
                {
                    "w": {
                        "id": "locations_castle",
                        "name": "Castle",
                        "category": "Locations",
                        KG_NODE_CREATED_CHAPTER: 0,
                    }
                }
            ]
        )
        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await world_queries.get_bootstrap_world_elements()
        assert isinstance(result, list)


@pytest.mark.asyncio
class TestSyncFullStateFromObject:
    """Tests for syncing full state from object."""

    async def test_sync_full_state_empty(self, monkeypatch):
        """Test syncing empty world data."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_cypher_batch", mock_execute
        )
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await world_queries.sync_full_state_from_object_to_db({})
        assert result is True

    async def test_sync_full_state_with_items(self, monkeypatch):
        """Test syncing world data with items."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_cypher_batch", mock_execute
        )
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            world_queries.neo4j_manager, "execute_read_query", mock_read
        )

        world_data = {
            "Locations": {
                "Castle": {
                    "description": "A grand castle",
                    "rules": ["No running"],
                }
            }
        }

        result = await world_queries.sync_full_state_from_object_to_db(world_data)
        assert result is True
        mock_execute.assert_called()
