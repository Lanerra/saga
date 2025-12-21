# tests/test_phase2_deduplication.py
"""
Tests for Phase 2 (relationship-based) deduplication.

Tests the two-phase deduplication approach where Phase 2 runs after
relationships are extracted to catch duplicates missed by Phase 1.
"""

from unittest.mock import AsyncMock, patch

import pytest

from processing.entity_deduplication import (
    check_relationship_pattern_similarity,
    find_relationship_based_duplicates,
    merge_duplicate_entities,
)


@pytest.mark.asyncio
class TestRelationshipPatternSimilarity:
    """Tests for relationship pattern similarity calculation."""

    async def test_identical_relationship_patterns(self):
        """Test entities with identical relationship patterns."""
        # Mock Neo4j query result with identical relationships
        mock_result = [
            {
                "rels1": [
                    {"type": "KNOWS", "target": "Bob"},
                    {"type": "LOCATED_IN", "target": "Central Lab"},
                ],
                "rels2": [
                    {"type": "KNOWS", "target": "Bob"},
                    {"type": "LOCATED_IN", "target": "Central Lab"},
                ],
            }
        ]

        with patch("processing.entity_deduplication.neo4j_manager") as mock_neo4j:
            mock_neo4j.execute_read_query = AsyncMock(return_value=mock_result)

            similarity = await check_relationship_pattern_similarity("Alice", "Alice Chen", "character")

            # Identical patterns should have similarity of 1.0
            assert similarity == 1.0

    async def test_partially_overlapping_relationships(self):
        """Test entities with partially overlapping relationships."""
        # Alice has: KNOWS Bob, LOCATED_IN Central Lab
        # Alice Chen has: KNOWS Bob, LOCATED_IN Central Lab, WORKS_WITH Carol
        # Jaccard similarity: 2 / 3 = 0.667
        mock_result = [
            {
                "rels1": [
                    {"type": "KNOWS", "target": "Bob"},
                    {"type": "LOCATED_IN", "target": "Central Lab"},
                ],
                "rels2": [
                    {"type": "KNOWS", "target": "Bob"},
                    {"type": "LOCATED_IN", "target": "Central Lab"},
                    {"type": "WORKS_WITH", "target": "Carol"},
                ],
            }
        ]

        with patch("processing.entity_deduplication.neo4j_manager") as mock_neo4j:
            mock_neo4j.execute_read_query = AsyncMock(return_value=mock_result)

            similarity = await check_relationship_pattern_similarity("Alice", "Alice Chen", "character")

            # Should be approximately 0.667
            assert 0.65 <= similarity <= 0.70

    async def test_no_relationship_overlap(self):
        """Test entities with completely different relationships."""
        mock_result = [
            {
                "rels1": [
                    {"type": "KNOWS", "target": "Bob"},
                    {"type": "LOCATED_IN", "target": "Central Lab"},
                ],
                "rels2": [
                    {"type": "KNOWS", "target": "Dave"},
                    {"type": "LOCATED_IN", "target": "East Wing"},
                ],
            }
        ]

        with patch("processing.entity_deduplication.neo4j_manager") as mock_neo4j:
            mock_neo4j.execute_read_query = AsyncMock(return_value=mock_result)

            similarity = await check_relationship_pattern_similarity("Alice", "Eve", "character")

            # No overlap should result in 0.0
            assert similarity == 0.0

    async def test_entity_with_no_relationships(self):
        """Test when one entity has no relationships."""
        mock_result = [
            {
                "rels1": [
                    {"type": "KNOWS", "target": "Bob"},
                ],
                "rels2": [],  # No relationships
            }
        ]

        with patch("processing.entity_deduplication.neo4j_manager") as mock_neo4j:
            mock_neo4j.execute_read_query = AsyncMock(return_value=mock_result)

            similarity = await check_relationship_pattern_similarity("Alice", "New Character", "character")

            # Can't determine similarity without relationships
            assert similarity == 0.0

    async def test_handles_neo4j_errors_gracefully(self):
        """Test error handling when Neo4j query fails."""
        with patch("processing.entity_deduplication.neo4j_manager") as mock_neo4j:
            mock_neo4j.execute_read_query = AsyncMock(side_effect=Exception("Database connection failed"))

            similarity = await check_relationship_pattern_similarity("Alice", "Alice Chen", "character")

            # Should return 0.0 on error to prevent blocking
            assert similarity == 0.0


@pytest.mark.asyncio
class TestFindRelationshipBasedDuplicates:
    """Tests for finding duplicates based on relationship patterns."""

    async def test_finds_duplicates_with_high_similarity(self):
        """Test finding duplicates with high name and relationship similarity."""
        # Mock query for entities with similar names
        mock_name_query = [
            {"name1": "Alice", "name2": "Alice Chen", "name_sim": 0.75},
        ]

        with patch("processing.entity_deduplication.neo4j_manager") as mock_neo4j:
            mock_neo4j.execute_read_query = AsyncMock(return_value=mock_name_query)

            with patch(
                "processing.entity_deduplication.check_relationship_pattern_similarity",
                new=AsyncMock(return_value=0.85),  # High relationship similarity
            ):
                duplicates = await find_relationship_based_duplicates(
                    entity_type="character",
                    name_similarity_threshold=0.6,
                    relationship_similarity_threshold=0.7,
                )

                # Should find the duplicate pair
                assert len(duplicates) == 1
                entity1, entity2, name_sim, rel_sim = duplicates[0]
                assert entity1 == "Alice"
                assert entity2 == "Alice Chen"
                assert name_sim == 0.75
                assert rel_sim == 0.85

    async def test_excludes_duplicates_with_low_relationship_similarity(self):
        """Test that pairs with low relationship similarity are excluded."""
        mock_name_query = [
            {"name1": "Alice", "name2": "Alice Chen", "name_sim": 0.75},
        ]

        with patch("processing.entity_deduplication.neo4j_manager") as mock_neo4j:
            mock_neo4j.execute_read_query = AsyncMock(return_value=mock_name_query)

            with patch(
                "processing.entity_deduplication.check_relationship_pattern_similarity",
                new=AsyncMock(return_value=0.3),  # Low relationship similarity
            ):
                duplicates = await find_relationship_based_duplicates(
                    entity_type="character",
                    name_similarity_threshold=0.6,
                    relationship_similarity_threshold=0.7,
                )

                # Should not find any duplicates
                assert len(duplicates) == 0

    async def test_excludes_high_name_similarity_pairs(self):
        """Test that pairs already merged in Phase 1 are excluded."""
        # Pairs with name_sim >= 0.8 should be excluded (already handled by Phase 1)
        mock_name_query = []  # Query excludes name_sim >= 0.8

        with patch("processing.entity_deduplication.neo4j_manager") as mock_neo4j:
            mock_neo4j.execute_read_query = AsyncMock(return_value=mock_name_query)

            duplicates = await find_relationship_based_duplicates(
                entity_type="character",
                name_similarity_threshold=0.6,
                relationship_similarity_threshold=0.7,
            )

            # Should not find any duplicates
            assert len(duplicates) == 0

    async def test_returns_empty_when_disabled(self):
        """Test that Phase 2 returns empty list when disabled."""
        with patch("config.ENABLE_DUPLICATE_PREVENTION", False):
            duplicates = await find_relationship_based_duplicates()
            assert duplicates == []


@pytest.mark.asyncio
class TestMergeDuplicateEntities:
    """Tests for merging duplicate entities."""

    async def test_merges_characters_successfully(self):
        """Test successful merge of duplicate characters."""
        # Mock query to determine which entity to keep
        mock_created_query = [
            {
                "name1": "Alice",
                "name2": "Alice Chen",
                "created1": 5,
                "created2": 7,
            }
        ]

        with patch("processing.entity_deduplication.neo4j_manager") as mock_neo4j:
            mock_neo4j.execute_read_query = AsyncMock(return_value=mock_created_query)
            mock_neo4j.execute_write_query = AsyncMock()

            success = await merge_duplicate_entities("Alice", "Alice Chen", entity_type="character")

            assert success is True
            assert mock_neo4j.execute_write_query.called

            call_args = mock_neo4j.execute_write_query.call_args
            query = call_args[0][0]

            # Regression: Phase 2 merge must not destroy relationship types or collapse edges.
            # The previous implementation rewrote every relationship to GENERIC_REL and used MERGE.
            assert "GENERIC_REL" not in query
            assert "apoc.create.relationship" in query
            assert "apoc.do.when" in query
            assert "DETACH DELETE duplicate" in query

    async def test_keeps_earlier_entity(self):
        """Test that the entity created earlier is kept."""
        mock_created_query = [
            {
                "name1": "Alice Chen",
                "name2": "Alice",
                "created1": 7,  # Created later
                "created2": 5,  # Created earlier
            }
        ]

        with patch("processing.entity_deduplication.neo4j_manager") as mock_neo4j:
            mock_neo4j.execute_read_query = AsyncMock(return_value=mock_created_query)
            mock_neo4j.execute_write_query = AsyncMock()

            success = await merge_duplicate_entities("Alice Chen", "Alice", entity_type="character")

            assert success is True
            # Verify the correct entity (Alice) is kept
            call_args = mock_neo4j.execute_write_query.call_args
            params = call_args[0][1]
            assert params["canonical"] == "Alice"
            assert params["duplicate"] == "Alice Chen"

    async def test_respects_keep_entity_parameter(self):
        """Test that keep_entity parameter is respected."""
        with patch("processing.entity_deduplication.neo4j_manager") as mock_neo4j:
            mock_neo4j.execute_write_query = AsyncMock()

            await merge_duplicate_entities("Alice", "Alice Chen", entity_type="character", keep_entity="Alice Chen")

            # Should use specified entity
            call_args = mock_neo4j.execute_write_query.call_args
            params = call_args[0][1]
            assert params["canonical"] == "Alice Chen"
            assert params["duplicate"] == "Alice"

    async def test_handles_missing_entities(self):
        """Test handling when entities don't exist."""
        with patch("processing.entity_deduplication.neo4j_manager") as mock_neo4j:
            mock_neo4j.execute_read_query = AsyncMock(return_value=[])  # No entities found

            success = await merge_duplicate_entities("NonExistent1", "NonExistent2", entity_type="character")

            assert success is False

    async def test_handles_merge_errors_gracefully(self):
        """Test error handling during merge operation."""
        mock_created_query = [
            {
                "name1": "Alice",
                "name2": "Alice Chen",
                "created1": 5,
                "created2": 7,
            }
        ]

        with patch("processing.entity_deduplication.neo4j_manager") as mock_neo4j:
            mock_neo4j.execute_read_query = AsyncMock(return_value=mock_created_query)
            mock_neo4j.execute_write_query = AsyncMock(side_effect=Exception("Merge failed"))

            success = await merge_duplicate_entities("Alice", "Alice Chen", entity_type="character")

            # Should return False on error
            assert success is False


@pytest.mark.asyncio
class TestPhase2Integration:
    """Integration tests for Phase 2 deduplication in commit node."""

    async def test_phase2_runs_after_commit(self):
        """Test that Phase 2 runs after relationships are committed."""
        from core.langgraph.nodes.commit_node import _run_phase2_deduplication

        with patch("config.ENABLE_PHASE2_DEDUPLICATION", True):
            with patch(
                "processing.entity_deduplication.find_relationship_based_duplicates",
                new=AsyncMock(return_value=[]),
            ):
                result = await _run_phase2_deduplication(chapter=5)

                assert result["characters"] == 0
                assert result["world_items"] == 0

    async def test_phase2_disabled_returns_zero_merges(self):
        """Test that Phase 2 returns zero merges when disabled."""
        from core.langgraph.nodes.commit_node import _run_phase2_deduplication

        with patch("config.ENABLE_PHASE2_DEDUPLICATION", False):
            result = await _run_phase2_deduplication(chapter=5)

            assert result["characters"] == 0
            assert result["world_items"] == 0

    async def test_phase2_counts_merges_correctly(self):
        """Test that Phase 2 correctly counts successful merges."""
        from core.langgraph.nodes.commit_node import _run_phase2_deduplication

        # Mock finding 2 character duplicates and 1 world item duplicate
        mock_char_duplicates = [
            ("Alice", "Alice Chen", 0.75, 0.85),
            ("Bob", "Bobby", 0.70, 0.90),
        ]
        mock_world_duplicates = [
            ("Central Lab", "The Central Laboratory", 0.72, 0.88),
        ]

        with patch("config.ENABLE_PHASE2_DEDUPLICATION", True):
            with patch(
                "processing.entity_deduplication.find_relationship_based_duplicates",
                side_effect=[mock_char_duplicates, mock_world_duplicates],
            ):
                with patch(
                    "processing.entity_deduplication.merge_duplicate_entities",
                    new=AsyncMock(return_value=True),
                ):
                    result = await _run_phase2_deduplication(chapter=5)

                    assert result["characters"] == 2
                    assert result["world_items"] == 1

    async def test_phase2_invalidates_caches_when_merges_occur(self):
        """Phase 2 must invalidate data_access caches after merges to prevent stale reads."""
        from core.langgraph.nodes.commit_node import _run_phase2_deduplication

        mock_char_duplicates = [
            ("Alice", "Alice Chen", 0.75, 0.85),
        ]

        with patch("config.ENABLE_PHASE2_DEDUPLICATION", True):
            with patch(
                "processing.entity_deduplication.find_relationship_based_duplicates",
                side_effect=[mock_char_duplicates, []],
            ):
                with patch(
                    "processing.entity_deduplication.merge_duplicate_entities",
                    new=AsyncMock(return_value=True),
                ):
                    with patch(
                        "data_access.cache_coordinator.clear_character_read_caches",
                        return_value={"get_character_profile_by_name": True, "get_character_profile_by_id": True},
                    ) as clear_character:
                        with patch(
                            "data_access.cache_coordinator.clear_world_read_caches",
                            return_value={"get_world_item_by_id": True},
                        ) as clear_world:
                            with patch(
                                "data_access.cache_coordinator.clear_kg_read_caches",
                                return_value={"query_kg_from_db": True, "get_novel_info_property_from_db": True},
                            ) as clear_kg:
                                result = await _run_phase2_deduplication(chapter=5)

                                assert result["characters"] == 1
                                assert result["world_items"] == 0

                                assert clear_character.call_count == 1
                                assert clear_world.call_count == 1
                                assert clear_kg.call_count == 1

    async def test_phase2_handles_partial_failures(self):
        """Test that Phase 2 handles when some merges fail."""
        from core.langgraph.nodes.commit_node import _run_phase2_deduplication

        mock_char_duplicates = [
            ("Alice", "Alice Chen", 0.75, 0.85),
            ("Bob", "Bobby", 0.70, 0.90),
        ]

        with patch("config.ENABLE_PHASE2_DEDUPLICATION", True):
            with patch(
                "processing.entity_deduplication.find_relationship_based_duplicates",
                side_effect=[mock_char_duplicates, []],
            ):
                # First merge succeeds, second fails
                with patch(
                    "processing.entity_deduplication.merge_duplicate_entities",
                    side_effect=[True, False],
                ):
                    result = await _run_phase2_deduplication(chapter=5)

                    # Only 1 successful merge
                    assert result["characters"] == 1
                    assert result["world_items"] == 0
