"""
Tests for LangGraph commit node (Step 1.2.1).

Tests the commit_to_graph node and its helper functions.
"""

import pytest
from unittest.mock import AsyncMock, patch

from core.langgraph.nodes.commit_node import (
    commit_to_graph,
    _convert_to_character_profiles,
    _convert_to_world_items,
)
from core.langgraph.state import ExtractedEntity


@pytest.mark.asyncio
class TestCommitToGraph:
    """Tests for commit_to_graph node function."""

    async def test_commit_with_no_entities(
        self,
        sample_initial_state,
        mock_knowledge_graph_service,
        mock_chapter_queries,
    ):
        """Test commit with no extracted entities."""
        state = sample_initial_state
        state["extracted_entities"] = {}
        state["extracted_relationships"] = []

        with patch("core.langgraph.nodes.commit_node.knowledge_graph_service", mock_knowledge_graph_service):
            with patch("core.langgraph.nodes.commit_node.chapter_queries", mock_chapter_queries):
                result = await commit_to_graph(state)

                assert result["current_node"] == "commit_to_graph"
                assert result["last_error"] is None

    async def test_commit_with_entities_and_relationships(
        self,
        sample_state_with_extraction,
        mock_knowledge_graph_service,
        mock_chapter_queries,
        mock_kg_queries,
    ):
        """Test commit with entities and relationships."""
        state = sample_state_with_extraction

        with patch("core.langgraph.nodes.commit_node.knowledge_graph_service", mock_knowledge_graph_service):
            with patch("core.langgraph.nodes.commit_node.chapter_queries", mock_chapter_queries):
                with patch("core.langgraph.nodes.commit_node.kg_queries", mock_kg_queries):
                    with patch("core.langgraph.nodes.commit_node.check_entity_similarity", new=AsyncMock(return_value=None)):
                        result = await commit_to_graph(state)

                        assert result["current_node"] == "commit_to_graph"
                        assert result["last_error"] is None

                        # Verify services were called
                        assert mock_knowledge_graph_service.persist_entities.called
                        assert mock_kg_queries.add_kg_triples_batch_to_db.called
                        assert mock_chapter_queries.save_chapter_data_to_db.called

    async def test_commit_handles_errors_gracefully(
        self,
        sample_state_with_extraction,
        mock_knowledge_graph_service,
    ):
        """Test that commit handles errors gracefully."""
        state = sample_state_with_extraction

        # Mock service to raise exception
        mock_knowledge_graph_service.persist_entities.side_effect = Exception("Database error")

        with patch("core.langgraph.nodes.commit_node.knowledge_graph_service", mock_knowledge_graph_service):
            with patch("core.langgraph.nodes.commit_node.check_entity_similarity", new=AsyncMock(return_value=None)):
                result = await commit_to_graph(state)

                assert result["current_node"] == "commit_to_graph"
                assert result["last_error"] is not None
                assert "Database error" in result["last_error"]


class TestConvertToCharacterProfiles:
    """Tests for _convert_to_character_profiles function."""

    def test_convert_character_entities(self):
        """Test converting ExtractedEntity to CharacterProfile."""
        entities = [
            ExtractedEntity(
                name="Alice",
                type="character",
                description="A brave warrior",
                first_appearance_chapter=1,
                attributes={
                    "brave": "",
                    "loyal": "",
                    "status": "alive",
                },
            ),
        ]

        name_mappings = {"Alice": "Alice"}  # No deduplication
        profiles = _convert_to_character_profiles(entities, name_mappings, 1)

        assert len(profiles) == 1
        assert profiles[0].name == "Alice"
        assert profiles[0].description == "A brave warrior"
        assert profiles[0].created_chapter == 1

    def test_convert_with_deduplication_mapping(self):
        """Test conversion applies deduplication mappings."""
        entities = [
            ExtractedEntity(
                name="NewCharacter",
                type="character",
                description="Test",
                first_appearance_chapter=1,
                attributes={},
            ),
        ]

        name_mappings = {"NewCharacter": "ExistingCharacter"}
        profiles = _convert_to_character_profiles(entities, name_mappings, 1)

        assert len(profiles) == 1
        assert profiles[0].name == "ExistingCharacter"  # Mapped name

    def test_convert_empty_list(self):
        """Test converting empty entity list."""
        profiles = _convert_to_character_profiles([], {}, 1)
        assert profiles == []


class TestConvertToWorldItems:
    """Tests for _convert_to_world_items function."""

    def test_convert_world_item_entities(self):
        """Test converting ExtractedEntity to WorldItem."""
        entities = [
            ExtractedEntity(
                name="Magic Sword",
                type="object",
                description="A legendary blade",
                first_appearance_chapter=1,
                attributes={
                    "category": "artifact",
                    "goals": ["Defeat evil"],
                    "rules": ["Only worthy can wield"],
                },
            ),
        ]

        id_mappings = {"Magic Sword": "artifact_001"}
        items = _convert_to_world_items(entities, id_mappings, 1)

        assert len(items) == 1
        assert items[0].name == "Magic Sword"
        assert items[0].id == "artifact_001"
        assert items[0].category == "artifact"
        assert items[0].description == "A legendary blade"

    def test_convert_with_deduplication_mapping(self):
        """Test conversion applies deduplication mappings."""
        entities = [
            ExtractedEntity(
                name="Castle",
                type="location",
                description="A grand castle",
                first_appearance_chapter=1,
                attributes={"category": "structure"},
            ),
        ]

        id_mappings = {"Castle": "existing_castle_id"}
        items = _convert_to_world_items(entities, id_mappings, 1)

        assert len(items) == 1
        assert items[0].id == "existing_castle_id"  # Mapped ID

    def test_convert_handles_list_attributes(self):
        """Test conversion handles list attributes correctly."""
        entities = [
            ExtractedEntity(
                name="Test Location",
                type="location",
                description="Test",
                first_appearance_chapter=1,
                attributes={
                    "category": "location",
                    "goals": ["goal1", "goal2"],
                    "rules": ["rule1"],
                    "key_elements": ["element1", "element2", "element3"],
                },
            ),
        ]

        id_mappings = {"Test Location": "loc_001"}
        items = _convert_to_world_items(entities, id_mappings, 1)

        assert len(items) == 1
        assert len(items[0].goals) == 2
        assert len(items[0].rules) == 1
        assert len(items[0].key_elements) == 3

    def test_convert_empty_list(self):
        """Test converting empty entity list."""
        items = _convert_to_world_items([], {}, 1)
        assert items == []


@pytest.mark.asyncio
class TestDeduplication:
    """Tests for deduplication logic in commit node."""

    async def test_character_deduplication_no_duplicates(
        self,
        sample_state_with_extraction,
        mock_knowledge_graph_service,
        mock_chapter_queries,
    ):
        """Test character deduplication when no duplicates found."""
        state = sample_state_with_extraction

        # Mock no duplicates found
        with patch("core.langgraph.nodes.commit_node.check_entity_similarity") as mock_check:
            mock_check.return_value = None

            with patch("core.langgraph.nodes.commit_node.knowledge_graph_service", mock_knowledge_graph_service):
                with patch("core.langgraph.nodes.commit_node.chapter_queries", mock_chapter_queries):
                    with patch("core.langgraph.nodes.commit_node.kg_queries"):
                        result = await commit_to_graph(state)

                        assert result["last_error"] is None
                        # Verify character names weren't changed
                        assert mock_check.called

    async def test_world_item_deduplication(
        self,
        sample_state_with_extraction,
        mock_knowledge_graph_service,
        mock_chapter_queries,
    ):
        """Test world item deduplication."""
        state = sample_state_with_extraction

        with patch("core.langgraph.nodes.commit_node.check_entity_similarity") as mock_check:
            mock_check.return_value = None

            with patch("core.langgraph.nodes.commit_node.knowledge_graph_service", mock_knowledge_graph_service):
                with patch("core.langgraph.nodes.commit_node.chapter_queries", mock_chapter_queries):
                    with patch("core.langgraph.nodes.commit_node.kg_queries"):
                        result = await commit_to_graph(state)

                        assert result["last_error"] is None
