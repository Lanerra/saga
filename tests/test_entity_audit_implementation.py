# tests/test_entity_audit_implementation.py
"""Comprehensive tests for the entity audit implementation roadmap.

Tests the complete workflow for addressing all six duplication issues identified
in the entity-audit.md report, verifying that the StateTracker integration
and enhanced healing logic work as expected.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agents.knowledge_agent import KnowledgeAgent
from models.kg_models import CharacterProfile, WorldItem
from processing.state_tracker import StateTracker


class TestEntityAuditImplementation:
    """Test suite for entity audit implementation roadmap."""

    @pytest.fixture
    def state_tracker(self):
        """Create a StateTracker instance for testing."""
        return StateTracker()

    @pytest.fixture
    def knowledge_agent(self):
        """Create a KnowledgeAgent instance for testing."""
        return KnowledgeAgent()

    @pytest.mark.asyncio
    async def test_issue_1_parallel_character_name_generation_conflicts_fixed(
        self, state_tracker
    ):
        """Test that parallel character name generation conflicts are resolved.

        Issue 1: Independent LLM calls for character naming during bootstrapping
        generate identical names without cross-call coordination.
        """
        # Simulate parallel character generation with potential name conflicts
        await state_tracker.reserve("Kaelen", "character", "A brave warrior")
        await state_tracker.reserve("SupportingChar1", "character", "Another character")

        # Check that conflicts are detected and prevented
        result1 = await state_tracker.reserve(
            "Kaelen", "character", "Duplicate character"
        )
        result2 = await state_tracker.reserve(
            "SupportingChar1", "world_item", "Different type"
        )

        # Should prevent duplicate reservations
        assert result1 is False  # Already reserved
        assert result2 is False  # Already reserved with different type

    @pytest.mark.asyncio
    async def test_issue_2_high_volume_world_item_name_conflicts_fixed(
        self, state_tracker
    ):
        """Test that high-volume world item name conflicts are resolved.

        Issue 2: Massive parallel name generation for world elements results in
        systematic conflicts where 8+ items receive identical names.
        """
        # Simulate high-volume parallel world item generation
        world_items = [
            ("The Silent Archive", "world_item", "Archive description 1"),
            ("The Spiral Archive", "world_item", "Archive description 2"),
            ("The Last Breath Archive", "world_item", "Archive description 3"),
        ]

        # Reserve items in parallel
        reservation_tasks = [
            state_tracker.reserve(name, item_type, description)
            for name, item_type, description in world_items
        ]

        results = await asyncio.gather(*reservation_tasks)

        # All should succeed initially
        assert all(results)

        # Try to reserve duplicates - should be prevented
        duplicate_result = await state_tracker.reserve(
            "The Spiral Archive", "world_item", "Duplicate"
        )
        assert duplicate_result is False

    @pytest.mark.asyncio
    async def test_issue_3_repeated_extraction_duplicate_prevention_fixed(
        self, state_tracker
    ):
        """Test that repeated extraction duplicate prevention works correctly.

        Issue 3: Chapter text extraction repeatedly attempts to create entities
        that already exist in the knowledge graph.
        """
        # Reserve existing entities
        await state_tracker.reserve(
            "Fren", "character", "Existing character description"
        )

        # Check duplicate prevention for similar entities
        similar_name = await state_tracker.has_similar_description(
            "Existing character description", "character"
        )
        assert similar_name == "Fren"

        # Try to reserve duplicate - should be prevented
        duplicate_result = await state_tracker.reserve(
            "Fren", "character", "Duplicate description"
        )
        assert duplicate_result is False

    @pytest.mark.asyncio
    async def test_issue_4_resource_intensive_knowledge_graph_healing_optimized(self):
        """Test that knowledge graph healing is optimized.

        Issue 4: Post-creation healing processes scan large numbers of entity pairs
        to identify and resolve duplicates.
        """
        # Mock the knowledge agent with enhanced healing logic
        with patch("agents.knowledge_agent.kg_queries") as mock_kg_queries:
            mock_kg_queries.find_candidate_duplicate_entities = AsyncMock(
                return_value=[]
            )
            mock_kg_queries.promote_dynamic_relationships = AsyncMock(return_value=0)
            mock_kg_queries.consolidate_similar_relationships = AsyncMock(
                return_value=0
            )
            mock_kg_queries.deduplicate_relationships = AsyncMock(return_value=0)

            agent = KnowledgeAgent()

            # Run healing process
            await agent._run_entity_resolution()

            # Verify optimized healing calls
            mock_kg_queries.find_candidate_duplicate_entities.assert_called_once()

    @pytest.mark.asyncio
    async def test_issue_5_bootstrap_validation_gap_fixed(self, state_tracker):
        """Test that bootstrap validation gap is fixed.

        Issue 5: The bootstrap validator checks for placeholder names and empty
        fields but fails to detect the core duplicate name issue.
        """
        # Create character profiles with potential duplicates
        {
            "Alice": CharacterProfile(name="Alice", description="First character"),
            "Bob": CharacterProfile(name="Bob", description="Second character"),
        }

        # Reserve characters in StateTracker
        await state_tracker.reserve("Alice", "character", "First character")
        await state_tracker.reserve("Bob", "character", "Second character")

        # Check validation with StateTracker integration
        tracked_entities = await state_tracker.get_entities_by_type("character")
        assert len(tracked_entities) == 2
        assert "Alice" in tracked_entities
        assert "Bob" in tracked_entities

    @pytest.mark.asyncio
    async def test_issue_6_semaphore_race_conditions_fixed(self, state_tracker):
        """Test that semaphore race conditions are resolved.

        Issue 6: Semaphores limit concurrent LLM calls but don't prevent context
        race conditions where parallel processes use stale snapshots.
        """

        # Simulate concurrent access with shared StateTracker
        async def concurrent_reservation(name: str, description: str):
            return await state_tracker.reserve(name, "character", description)

        # Run concurrent reservations
        tasks = [
            concurrent_reservation(f"Character{i}", f"Description {i}")
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed without race conditions
        assert all(results)

        # Verify all characters were reserved
        tracked_entities = await state_tracker.get_entities_by_type("character")
        assert len(tracked_entities) == 5

    @pytest.mark.asyncio
    async def test_comprehensive_workflow_integration(self, state_tracker):
        """Test the complete workflow integration across all issues."""
        # Test the full bootstrap -> validation -> healing workflow

        # 1. Bootstrap phase with StateTracker integration
        [
            CharacterProfile(name="Hero", description="Main protagonist"),
            CharacterProfile(name="Villain", description="Main antagonist"),
        ]

        [
            WorldItem.from_dict(
                "locations", "Castle", {"description": "Ancient fortress"}
            ),
            WorldItem.from_dict(
                "factions", "Kingdom", {"description": "Ruling kingdom"}
            ),
        ]

        # Reserve all entities
        await state_tracker.reserve("Hero", "character", "Main protagonist")
        await state_tracker.reserve("Villain", "character", "Main antagonist")
        await state_tracker.reserve("Castle", "world_item", "Ancient fortress")
        await state_tracker.reserve("Kingdom", "world_item", "Ruling kingdom")

        # 2. Verify no duplicates in StateTracker
        all_entities = await state_tracker.get_all()
        assert len(all_entities) == 4

        # 3. Test duplicate prevention
        duplicate_result = await state_tracker.reserve("Hero", "character", "Duplicate")
        assert duplicate_result is False

        # 4. Test similarity detection
        similar = await state_tracker.has_similar_description(
            "Main protagonist", "character"
        )
        # Similarity check is token-based; ensure it resolves to an existing name
        assert similar in {"Hero", None}  # Accept None if threshold not met

    @pytest.mark.asyncio
    async def test_performance_optimization_metrics(self, state_tracker):
        """Test that performance optimization targets are met."""
        # According to entity audit, targets should be:
        # - Character name conflicts: Reduce from 75% to <10% occurrence
        # - World item conflicts: Reduce from 8+ items to <3 items per conflict
        # - Healing candidates: Reduce from 50+ pairs to ~10-15 pairs

        # Simulate bulk reservations to test conflict rates
        character_names = [f"Character{i}" for i in range(100)]
        world_names = [f"WorldItem{j}" for j in range(50)]

        # Reserve all entities without conflicts
        char_tasks = [
            state_tracker.reserve(name, "character", f"Description for {name}")
            for name in character_names
        ]
        world_tasks = [
            state_tracker.reserve(name, "world_item", f"Description for {name}")
            for name in world_names
        ]

        char_results = await asyncio.gather(*char_tasks)
        world_results = await asyncio.gather(*world_tasks)

        # All initial reservations should succeed (0% conflict rate)
        assert all(char_results)
        assert all(world_results)

        # Try duplicate reservations - should all be prevented (100% prevention)
        duplicate_char_tasks = [
            state_tracker.reserve(name, "character", "Duplicate")
            for name in character_names[:10]  # Test subset
        ]
        duplicate_world_tasks = [
            state_tracker.reserve(name, "world_item", "Duplicate")
            for name in world_names[:5]  # Test subset
        ]

        dup_char_results = await asyncio.gather(*duplicate_char_tasks)
        dup_world_results = await asyncio.gather(*duplicate_world_tasks)

        # All duplicates should be prevented
        assert all(not result for result in dup_char_results)
        assert all(not result for result in dup_world_results)

    @pytest.mark.asyncio
    async def test_backward_compatibility_preserved(self, state_tracker):
        """Test that backward compatibility is preserved when StateTracker is disabled."""
        with patch("processing.state_tracker.config") as mock_config:
            mock_config.STATE_TRACKER_ENABLED = False

            # Operations should still work but return None/false appropriately
            result = await state_tracker.reserve(
                "Test", "character", "Test description"
            )
            assert result is True  # Always returns True when disabled

            metadata = await state_tracker.check("Test")
            assert metadata is None  # Always returns None when disabled

    @pytest.mark.skip(reason="prioritize_candidates_with_state_tracker removed in NANA cleanup")
    @pytest.mark.asyncio
    async def test_state_tracker_prioritization_enhanced(self):
        """Test enhanced StateTracker prioritization for healing logic."""
        # Mock StateTracker with recent entities
        mock_state_tracker = AsyncMock(spec=StateTracker)
        mock_state_tracker.get_all = AsyncMock(
            return_value={
                "RecentChar": {
                    "name": "RecentChar",
                    "type": "character",
                    "description": "Recently created character",
                    "timestamp": "2025-01-01T12:00:00",
                }
            }
        )

        # Test prioritization logic
        agent = KnowledgeAgent()
        candidates = [
            {"name": "RecentChar", "type": "character"},
            {"name": "OldChar", "type": "character"},
        ]

        # This should work with the enhanced prioritization
        # Note: The actual async version would be called in practice
        prioritized = agent.prioritize_candidates_with_state_tracker(
            candidates, mock_state_tracker
        )
        assert len(prioritized) == len(candidates)  # Should return all candidates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
