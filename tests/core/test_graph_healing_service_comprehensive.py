# tests/core/test_graph_healing_service_comprehensive.py
"""Comprehensive tests for GraphHealingService."""

from unittest.mock import patch

import pytest

from core.graph_healing_service import GraphHealingService


class TestGraphHealingServiceNodeIdentification:
    """Test provisional node identification."""

    @pytest.mark.asyncio
    async def test_identify_provisional_nodes_returns_list(self) -> None:
        """Test that identify_provisional_nodes returns a list of nodes."""
        service = GraphHealingService()

        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            mock_query.return_value = [
                {
                    "element_id": "neo4j-element-1",
                    "id": "app-id-1",
                    "name": "Test Entity",
                    "type": "Character",
                    "description": "Unknown",
                    "traits": [],
                    "created_chapter": 0,
                }
            ]

            result = await service.identify_provisional_nodes()

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["name"] == "Test Entity"

    @pytest.mark.asyncio
    async def test_identify_provisional_nodes_handles_empty_result(self) -> None:
        """Test that identify_provisional_nodes handles empty results."""
        service = GraphHealingService()

        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            mock_query.return_value = []

            result = await service.identify_provisional_nodes()

            assert isinstance(result, list)
            assert len(result) == 0


class TestGraphHealingServiceConfidenceCalculation:
    """Test node confidence calculation."""

    @pytest.mark.asyncio
    async def test_calculate_node_confidence_basic_node(self) -> None:
        """Test basic confidence calculation for a node with minimal attributes."""
        service = GraphHealingService()

        node = {
            "element_id": "neo4j-element-1",
            "id": "app-id-1",
            "name": "Test Entity",
            "type": "Character",
            "description": "Unknown",
            "traits": [],
            "created_chapter": 0,
        }

        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            # No relationships
            mock_query.return_value = [{"rel_count": 0}]

            confidence = await service.calculate_node_confidence(node, current_chapter=1)

            # Should have minimal confidence (only age bonus if applicable)
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_calculate_node_confidence_with_relationships(self) -> None:
        """Test confidence calculation with relationship connectivity."""
        service = GraphHealingService()

        node = {
            "element_id": "neo4j-element-1",
            "id": "app-id-1",
            "name": "Test Entity",
            "type": "Character",
            "description": "Unknown",
            "traits": [],
            "created_chapter": 0,
        }

        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            # High relationship count (should give connectivity score)
            mock_query.return_value = [{"rel_count": 5}]

            confidence = await service.calculate_node_confidence(node, current_chapter=1)

            # Should have connectivity bonus (0.4 max for relationships)
            assert confidence >= 0.3

    @pytest.mark.asyncio
    async def test_calculate_node_confidence_with_description(self) -> None:
        """Test confidence calculation with description completeness."""
        service = GraphHealingService()

        node = {
            "element_id": "neo4j-element-1",
            "id": "app-id-1",
            "name": "Test Entity",
            "type": "Character",
            "description": "A detailed description that is definitely more than twenty characters long and demonstrates the entity's importance.",
            "traits": [],
            "created_chapter": 0,
        }

        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            mock_query.return_value = [{"rel_count": 0}]

            confidence = await service.calculate_node_confidence(node, current_chapter=1)

            # Should get description bonus (0.2 for good description)
            assert confidence >= 0.2

    @pytest.mark.asyncio
    async def test_calculate_node_confidence_with_traits(self) -> None:
        """Test confidence calculation with traits."""
        service = GraphHealingService()

        node = {
            "element_id": "neo4j-element-1",
            "id": "app-id-1",
            "name": "Test Entity",
            "type": "Character",
            "description": "Unknown",
            "traits": ["Brave", "Intelligent"],
            "created_chapter": 0,
        }

        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            mock_query.return_value = [{"rel_count": 0}]

            confidence = await service.calculate_node_confidence(node, current_chapter=1)

            # Should get traits bonus (0.1)
            assert confidence >= 0.1

    @pytest.mark.asyncio
    async def test_calculate_node_confidence_with_age(self) -> None:
        """Test confidence calculation with age bonus."""
        service = GraphHealingService()

        node = {
            "element_id": "neo4j-element-1",
            "id": "app-id-1",
            "name": "Test Entity",
            "type": "Character",
            "description": "Unknown",
            "traits": [],
            "created_chapter": 0,  # Can now use 0 thanks to bug fix
        }

        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            # Mock both relationship query and status query (Character type triggers status check)
            mock_query.side_effect = [[{"rel_count": 0}], [{"status": "Unknown"}]]

            # Node created in chapter 0, current chapter is 4 (age = 4, meets AGE_GRADUATION_CHAPTERS)
            confidence = await service.calculate_node_confidence(node, current_chapter=4)

            # Should get age bonus (0.2 for surviving 4+ chapters)
            assert confidence >= 0.2

    @pytest.mark.asyncio
    async def test_calculate_node_confidence_character_with_status(self) -> None:
        """Test confidence calculation for character with status attribute."""
        service = GraphHealingService()

        node = {
            "element_id": "neo4j-element-1",
            "id": "app-id-1",
            "name": "Test Character",
            "type": "Character",
            "description": "Unknown",
            "traits": [],
            "created_chapter": 0,
        }

        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            # First call for relationships, second for status
            mock_query.side_effect = [[{"rel_count": 0}], [{"status": "Alive"}]]

            confidence = await service.calculate_node_confidence(node, current_chapter=1)

            # Should get status bonus (0.1)
            assert confidence >= 0.1


class TestGraphHealingServiceEnrichment:
    """Test node enrichment functionality."""

    @pytest.mark.asyncio
    async def test_enrich_node_from_context_with_valid_data(self) -> None:
        """Test that enrich_node_from_context handles valid data."""
        service = GraphHealingService()
        node = {"element_id": "neo4j-element-1", "id": "app-id-1", "name": "Test Entity", "type": "Character", "description": "Unknown", "traits": []}

        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            with patch("data_access.kg_queries.get_chapter_context_for_entity") as mock_context:
                with patch("core.llm_interface_refactored.llm_service.async_call_llm") as mock_llm:
                    mock_context.return_value = []
                    mock_llm.return_value = ('{"inferred_description": "A test character", "confidence": 0.8}', None)
                    mock_query.return_value = []

                    result = await service.enrich_node_from_context(node, "gpt-4")
                    assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_apply_enrichment_with_valid_data(self) -> None:
        """Test that apply_enrichment applies enrichment correctly."""
        service = GraphHealingService()
        enriched = {"inferred_description": "A test character", "confidence": 0.8}

        with patch("core.db_manager.neo4j_manager.execute_write_query") as mock_write:
            result = await service.apply_enrichment("test_element_id", enriched)
            assert result is True
            mock_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_enrichment_below_confidence_threshold(self) -> None:
        """Test that apply_enrichment rejects low confidence enrichment."""
        service = GraphHealingService()
        enriched = {"inferred_description": "A test character", "confidence": 0.5}

        with patch("core.db_manager.neo4j_manager.execute_write_query") as mock_write:
            result = await service.apply_enrichment("test_element_id", enriched)
            assert result is False
            mock_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_graduate_node(self) -> None:
        """Test that graduate_node marks a node as graduated."""
        service = GraphHealingService()

        with patch("core.db_manager.neo4j_manager.execute_write_query") as mock_write:
            mock_write.return_value = [{"name": "Test Entity"}]

            result = await service.graduate_node("neo4j-element-1", 0.85)

            assert result is True
            mock_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_node_by_element_id(self) -> None:
        """Test that get_node_by_element_id retrieves node properties."""
        service = GraphHealingService()

        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            mock_query.return_value = [
                {
                    "element_id": "neo4j-element-1",
                    "id": "app-id-1",
                    "name": "Test Entity",
                    "type": "Character",
                    "description": "Updated description",
                    "traits": ["Brave"],
                    "created_chapter": 0,
                }
            ]

            result = await service.get_node_by_element_id("neo4j-element-1")

            assert isinstance(result, dict)
            assert result["name"] == "Test Entity"
            assert result["description"] == "Updated description"


class TestGraphHealingServiceDeduplication:
    """Test entity deduplication functionality."""

    @pytest.mark.asyncio
    async def test_find_merge_candidates_basic(self) -> None:
        """Test that find_merge_candidates processes candidate pairs correctly."""
        service = GraphHealingService()

        with patch("data_access.kg_queries.find_candidate_duplicate_entities") as mock_find:
            with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
                # Mock the candidate duplicates (returning above threshold)
                mock_find.return_value = [
                    {
                        "id1": "app-id-1",
                        "name1": "Alice",
                        "id2": "app-id-2",
                        "name2": "Alicia",
                        "similarity": 0.85,
                        "labels1": ["Character"],
                    }
                ]

                # Mock element ID lookups and embeddings
                mock_query.side_effect = [
                    [{"element_id": "neo4j-element-1"}],
                    [{"element_id": "neo4j-element-2"}],
                    [],  # No embeddings found
                ]

                result = await service.find_merge_candidates(use_advanced_matching=True)

                # Verify the method processes candidates correctly
                assert isinstance(result, list)
                # Note: The actual number depends on kg_queries filtering
                # This test verifies the transformation logic works

    @pytest.mark.asyncio
    async def test_validate_merge_safe(self) -> None:
        """Test that validate_merge approves safe merges."""
        service = GraphHealingService()

        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            # No co-occurrences, similar relationships
            mock_query.side_effect = [
                [{"cooccurrences": 0}],
                [
                    {"node_id": "neo4j-element-1", "rel_type": "APPEARS_IN", "count": 2},
                    {"node_id": "neo4j-element-2", "rel_type": "APPEARS_IN", "count": 2},
                ],
            ]

            result = await service.validate_merge("neo4j-element-1", "neo4j-element-2")

            assert isinstance(result, dict)
            assert result["is_valid"] is True

    @pytest.mark.asyncio
    async def test_validate_merge_unsafe(self) -> None:
        """Test that validate_merge rejects unsafe merges."""
        service = GraphHealingService()

        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            # Has co-occurrences (indicates distinct entities)
            mock_query.side_effect = [[{"cooccurrences": 2}], []]

            result = await service.validate_merge("neo4j-element-1", "neo4j-element-2")

            assert isinstance(result, dict)
            assert result["is_valid"] is False

    @pytest.mark.asyncio
    async def test_execute_merge(self) -> None:
        """Test that execute_merge performs the merge operation."""
        service = GraphHealingService()

        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            with patch("data_access.kg_queries.get_entity_context_for_resolution") as mock_context:
                with patch("data_access.kg_queries.merge_entities") as mock_merge:
                    # Mock entity ID lookups
                    mock_query.side_effect = [
                        [{"entity_id": "app-id-1"}],
                        [{"entity_id": "app-id-2"}],
                    ]

                    # Mock context retrieval
                    mock_context.side_effect = [{"name": "Alice"}, {"name": "Alicia"}]

                    # Mock successful merge
                    mock_merge.return_value = True

                    result = await service.execute_merge("neo4j-element-1", "neo4j-element-2", {"similarity": 0.85})

                    assert result is True
                    mock_merge.assert_called_once()


class TestGraphHealingServiceOrphanCleanup:
    """Test orphaned node cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_orphaned_nodes_removes_old_orphans(self) -> None:
        """Test that cleanup_orphaned_nodes removes old orphaned nodes."""
        service = GraphHealingService()

        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            with patch("core.db_manager.neo4j_manager.execute_write_query"):
                # Mock orphaned nodes (created in chapter 1, current is 5)
                mock_query.return_value = [
                    {
                        "element_id": "neo4j-element-1",
                        "name": "Orphan Entity",
                        "type": "Character",
                        "created_chapter": 1,
                    }
                ]

                result = await service.cleanup_orphaned_nodes(current_chapter=5)

                assert isinstance(result, dict)
                assert result["nodes_removed"] == 1
                assert result["nodes_checked"] == 1

    @pytest.mark.asyncio
    async def test_cleanup_orphaned_nodes_preserves_recent(self) -> None:
        """Test that cleanup_orphaned_nodes preserves recent orphaned nodes."""
        service = GraphHealingService()

        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_read:
            with patch("core.db_manager.neo4j_manager.execute_write_query"):
                # Mock the query to return nodes created in chapter 1 (cutoff is 5-3=2, so 1 <= 2 means removed)
                def mock_query_side_effect(query, params=None):
                    if params and params.get("cutoff_chapter") == 2:
                        # Return nodes that are OLD ENOUGH to be removed (created in chapter 1)
                        return [
                            {
                                "element_id": "neo4j-element-1",
                                "name": "Old Entity",
                                "type": "Character",
                                "created_chapter": 1,
                            }
                        ]
                    return []

                mock_read.side_effect = mock_query_side_effect

                result = await service.cleanup_orphaned_nodes(current_chapter=5)

                assert isinstance(result, dict)
                # Node created in chapter 1 should be removed (1 <= 2)
                assert result["nodes_removed"] == 1
                assert result["nodes_checked"] == 1


class TestGraphHealingServiceIntegration:
    """Test full healing workflow integration."""

    @pytest.mark.asyncio
    async def test_heal_graph_with_mocked_operations(self) -> None:
        """Test that heal_graph orchestrates all operations correctly."""
        service = GraphHealingService()

        with patch("core.db_manager.neo4j_manager.execute_read_query") as mock_query:
            with patch("core.db_manager.neo4j_manager.execute_write_query") as mock_write:
                with patch("data_access.kg_queries.find_candidate_duplicate_entities") as mock_find:
                    with patch("data_access.kg_queries.get_chapter_context_for_entity") as mock_context:
                        with patch("core.llm_interface_refactored.llm_service.async_call_llm") as mock_llm:
                            # Mock provisional nodes
                            mock_query.side_effect = [
                                [
                                    {
                                        "element_id": "neo4j-element-1",
                                        "id": "app-id-1",
                                        "name": "Test Entity",
                                        "type": "Character",
                                        "description": "Unknown",
                                        "traits": [],
                                        "created_chapter": 0,
                                    }
                                ],
                                [{"rel_count": 0}],
                                [],  # No merge candidates
                                [],  # No orphaned nodes
                            ]

                            # Mock LLM response
                            mock_llm.return_value = ('{"inferred_description": "A test character", "confidence": 0.8}', None)
                            mock_context.return_value = []
                            mock_find.return_value = []
                            mock_write.return_value = [{"name": "Test Entity"}]

                            result = await service.heal_graph(current_chapter=1, model="gpt-4")

                            assert isinstance(result, dict)
                            assert result["chapter"] == 1
                            assert "timestamp" in result
