# tests/test_world_queries_extended.py
"""Extended tests for data_access/world_queries.py to improve coverage."""

from unittest.mock import AsyncMock, patch

import pytest

from core.exceptions import ValidationError
from data_access import world_queries
from models.kg_constants import KG_IS_PROVISIONAL, KG_NODE_CREATED_CHAPTER


@pytest.mark.asyncio
class TestGetWorldItemByIdExtended:
    """Extended tests for get_world_item_by_id."""

    async def test_get_world_item_by_id_missing_fields_raises(self, monkeypatch):
        """Missing category/name in DB record raises ValidationError."""
        world_queries.get_world_item_by_id.cache_clear()

        mock_node = {
            "id": "loc_missing",
            "description": "Incomplete",
        }

        async def mock_read(query, params=None):
            if "RETURN we" in query:
                return [{"we": mock_node}]
            return []

        monkeypatch.setattr(world_queries.neo4j_manager, "execute_read_query", AsyncMock(side_effect=mock_read))

        with pytest.raises(ValidationError):
            await world_queries.get_world_item_by_id("loc_missing")

    async def test_get_world_item_by_id_with_complex_props(self, monkeypatch):
        """Test retrieving item with list properties and elaborations."""

        mock_node = {"id": "loc_complex", "name": "Complex Loc", "category": "Locations", "description": "Desc", KG_NODE_CREATED_CHAPTER: 1, KG_IS_PROVISIONAL: True}

        async def mock_read(query, params=None):
            if "RETURN we" in query:
                return [{"we": mock_node}]
            if "v.value AS item_value" in query:
                return [{"item_value": "Goal1"}]
            if "RETURN t.name AS trait_name" in query:
                return [{"trait_name": "Trait1"}]
            if "RETURN elab.summary" in query:
                return [{"summary": "Elab1", "chapter": 2, "is_provisional": True}]
            return []

        monkeypatch.setattr(world_queries.neo4j_manager, "execute_read_query", AsyncMock(side_effect=mock_read))

        item = await world_queries.get_world_item_by_id("loc_complex")

        assert item is not None
        assert item.is_provisional is True


@pytest.mark.asyncio
class TestGetBootstrapWorldElementsExtended:
    """Extended tests for get_bootstrap_world_elements."""

    async def test_get_bootstrap_elements_filtering(self, monkeypatch):
        """Test that bootstrap elements are filtered correctly."""

        # Mock 2 nodes: one valid, one with FILL_IN
        valid_node = {"id": "valid", "name": "Valid", "category": "Loc", "description": "Good desc", "created_chapter": 0, "source": "bootstrap"}
        # Use a specific marker for this test
        TEST_MARKER = "[TEST_FILL_IN]"

        invalid_node = {"id": "invalid", "name": "Invalid", "category": "Loc", "description": f"Has {TEST_MARKER}", "created_chapter": 0, "source": "bootstrap"}

        async def mock_read(query, params=None):
            return [{"we": valid_node}, {"we": invalid_node}]

        monkeypatch.setattr(world_queries.neo4j_manager, "execute_read_query", AsyncMock(side_effect=mock_read))

        # Patch config.FILL_IN in world_queries to match our test marker
        with patch("data_access.world_queries.config.FILL_IN", TEST_MARKER):
            results = await world_queries.get_bootstrap_world_elements()

        assert len(results) == 1
        assert results[0].name == "Valid"
