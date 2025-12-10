"""Extended tests for data_access/world_queries.py to improve coverage."""
import pytest
from unittest.mock import AsyncMock, patch
from data_access import world_queries
from models import WorldItem
from models.kg_constants import KG_IS_PROVISIONAL, KG_NODE_CREATED_CHAPTER

@pytest.mark.asyncio
class TestSyncFullStateExtended:
    """Extended tests for sync_full_state_from_object_to_db."""

    async def test_sync_full_state_complex(self, monkeypatch):
        """Test syncing complex world state including overview, deletions, and elaborations."""
        
        # Mock DB responses
        async def mock_read(query, params=None):
            if "RETURN we.id AS id" in query: # Fetch existing IDs
                return [{"id": "locations_old_castle"}] # This one should be deleted
            return []

        async def mock_write(query, params=None):
            return []

        monkeypatch.setattr(world_queries.neo4j_manager, "execute_read_query", AsyncMock(side_effect=mock_read))
        monkeypatch.setattr(world_queries.neo4j_manager, "execute_cypher_batch", AsyncMock(side_effect=mock_write))

        world_data = {
            "_overview_": {
                "description": "Global overview",
                "source_quality_chapter_0": "provisional_from_unrevised_draft"
            },
            "Locations": {
                "New Castle": {
                    "description": "A new castle",
                    "goals": ["Defend"],
                    "elaboration_in_chapter_5": "Expanded details",
                    "source_quality_chapter_5": "provisional_from_unrevised_draft"
                }
            }
        }

        result = await world_queries.sync_full_state_from_object_to_db(world_data)
        assert result is True

        # Verify batch calls
        calls = world_queries.neo4j_manager.execute_cypher_batch.call_args_list
        assert len(calls) > 0
        statements = calls[0][0][0] # First arg of first call
        
        # Check for specific queries in the batch
        query_texts = [s[0] for s in statements]
        combined_query_text = " ".join(query_texts)

        assert "MERGE (wc {id: $id_val})" in combined_query_text # Overview
        assert "SET we.is_deleted = TRUE" in combined_query_text # Deletion
        assert "MERGE (we {id: $id_val})" in combined_query_text # New Item
        assert "MERGE (wc)-[:CONTAINS_ELEMENT]->(we)" in combined_query_text # Connection
        assert "MERGE (we)-[:HAS_GOAL]->(v)" in combined_query_text # Goals
        assert "CREATE (elab:WorldElaborationEvent)" in combined_query_text # Elaboration

    async def test_sync_full_state_validation_error(self, monkeypatch):
        """Test that invalid items are skipped/logged but don't crash."""
        monkeypatch.setattr(world_queries.neo4j_manager, "execute_read_query", AsyncMock(return_value=[]))
        monkeypatch.setattr(world_queries.neo4j_manager, "execute_cypher_batch", AsyncMock(return_value=[]))

        world_data = {
            "Locations": {
                "_skip_me": {}, # Should be skipped by name check
                "Valid": {"description": "ok"}
            }
        }
        
        result = await world_queries.sync_full_state_from_object_to_db(world_data)
        assert result is True


@pytest.mark.asyncio
class TestGetWorldItemByIdExtended:
    """Extended tests for get_world_item_by_id."""

    async def test_get_world_item_by_id_missing_fields(self, monkeypatch):
        """Test handling of world item with missing core fields in DB."""
        
        mock_node = {
            "id": "loc_missing",
            # Missing name and category
            "description": "Incomplete"
        }
        
        async def mock_read(query, params=None):
            if "RETURN we" in query:
                return [{"we": mock_node}]
            return []

        monkeypatch.setattr(world_queries.neo4j_manager, "execute_read_query", AsyncMock(side_effect=mock_read))

        # Should log warning but try to fix it
        # utils.validate_world_item_fields handles corrections:
        # If category missing -> "other", name -> "unnamed_element"
        
        result = await world_queries.get_world_item_by_id("loc_missing")
        assert result is not None
        assert result.category == "other"
        assert result.name == "unnamed_element"

    async def test_get_world_item_by_id_with_complex_props(self, monkeypatch):
        """Test retrieving item with list properties and elaborations."""
        
        mock_node = {
            "id": "loc_complex",
            "name": "Complex Loc",
            "category": "Locations",
            "description": "Desc",
            KG_NODE_CREATED_CHAPTER: 1,
            KG_IS_PROVISIONAL: True
        }
        
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
        valid_node = {
            "id": "valid", "name": "Valid", "category": "Loc", "description": "Good desc", 
            "created_chapter": 0, "source": "bootstrap"
        }
        # Use a specific marker for this test
        TEST_MARKER = "[TEST_FILL_IN]"
        
        invalid_node = {
            "id": "invalid", "name": "Invalid", "category": "Loc", "description": f"Has {TEST_MARKER}",
            "created_chapter": 0, "source": "bootstrap"
        }
        
        async def mock_read(query, params=None):
            return [{"we": valid_node}, {"we": invalid_node}]

        monkeypatch.setattr(world_queries.neo4j_manager, "execute_read_query", AsyncMock(side_effect=mock_read))
        
        # Patch config.FILL_IN in world_queries to match our test marker
        with patch("data_access.world_queries.config.FILL_IN", TEST_MARKER):
            results = await world_queries.get_bootstrap_world_elements()
        
        assert len(results) == 1
        assert results[0].name == "Valid"
