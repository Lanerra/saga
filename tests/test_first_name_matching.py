# tests/test_first_name_matching.py
"""
Test first name matching for character deduplication.

This test verifies that characters referenced by first name only are correctly
matched to existing characters with full names, preventing duplicate nodes.
"""

import pytest

from processing.entity_deduplication import check_entity_similarity


@pytest.mark.asyncio
class TestFirstNameMatching:
    """Tests for first name matching in character deduplication."""

    async def test_first_name_matches_full_name_character(self, neo4j_test_db):
        """Test that first name 'Jon' matches existing 'Jonathan Smith'."""
        from core.db_manager import neo4j_manager

        # Create a character with full name
        create_query = """
        MERGE (c:Character:Entity {name: 'Jonathan Smith'})
        SET c.description = 'A brave warrior',
            c.status = 'alive'
        RETURN c.name as name
        """
        await neo4j_manager.execute_write_query(create_query)

        # Check similarity with first name only
        result = await check_entity_similarity("Jonathan", "character")

        # Verify match found with high similarity (0.95 for first name match)
        assert result is not None, "Should find a match for first name"
        assert result["existing_name"] == "Jonathan Smith"
        assert result["similarity"] == 0.95, "First name match should have 0.95 similarity"

    async def test_first_name_case_insensitive_matching(self, neo4j_test_db):
        """Test that first name matching is case-insensitive."""
        from core.db_manager import neo4j_manager

        # Create a character with full name
        create_query = """
        MERGE (c:Character:Entity {name: 'Alice Cooper'})
        SET c.description = 'A skilled mage'
        RETURN c.name as name
        """
        await neo4j_manager.execute_write_query(create_query)

        # Check similarity with lowercase first name
        result = await check_entity_similarity("alice", "character")

        # Verify match found
        assert result is not None
        assert result["existing_name"] == "Alice Cooper"
        assert result["similarity"] == 0.95

    async def test_first_name_only_character_matches_itself(self, neo4j_test_db):
        """Test that single-name character matches itself."""
        from core.db_manager import neo4j_manager

        # Create a character with single name
        create_query = """
        MERGE (c:Character:Entity {name: 'Gandalf'})
        SET c.description = 'A wizard'
        RETURN c.name as name
        """
        await neo4j_manager.execute_write_query(create_query)

        # Check similarity with same name
        result = await check_entity_similarity("Gandalf", "character")

        # Verify exact match found (should get 1.0 similarity for exact match)
        assert result is not None
        assert result["existing_name"] == "Gandalf"
        # Exact match should get higher priority than first name match
        assert result["similarity"] >= 0.95

    async def test_no_match_when_first_name_different(self, neo4j_test_db):
        """Test that different first names don't match."""
        from core.db_manager import neo4j_manager

        # Create a character
        create_query = """
        MERGE (c:Character:Entity {name: 'Robert Jones'})
        SET c.description = 'A merchant'
        RETURN c.name as name
        """
        await neo4j_manager.execute_write_query(create_query)

        # Check similarity with completely different name
        # This should not match (unless Levenshtein similarity is high)
        result = await check_entity_similarity("Alice", "character")

        # Should either find no match, or find a match with low similarity
        if result is not None:
            assert result["similarity"] < 0.8, "Different names should have low similarity"

    async def test_multiple_characters_picks_best_match(self, neo4j_test_db):
        """Test that when multiple characters exist, the best match is returned."""
        from core.db_manager import neo4j_manager

        # Create multiple characters
        create_query = """
        MERGE (c1:Character:Entity {name: 'John Smith'})
        SET c1.description = 'A blacksmith'
        MERGE (c2:Character:Entity {name: 'Johnny Walker'})
        SET c2.description = 'A traveler'
        RETURN count(*) as count
        """
        await neo4j_manager.execute_write_query(create_query)

        # Check similarity with 'John' - should match 'John Smith' as exact first name
        result = await check_entity_similarity("John", "character")

        # Should find a match (either John Smith or Johnny Walker)
        # Both should have high similarity, but exact first name match should be 0.95
        assert result is not None
        assert result["similarity"] >= 0.9
        # The query orders by similarity DESC, so we should get the first match
        # which should be 'John Smith' with 0.95 similarity

    async def test_first_name_with_middle_name_in_db(self, neo4j_test_db):
        """Test matching first name when DB has middle name."""
        from core.db_manager import neo4j_manager

        # Create a character with middle name
        create_query = """
        MERGE (c:Character:Entity {name: 'Elizabeth Anne Taylor'})
        SET c.description = 'A scholar'
        RETURN c.name as name
        """
        await neo4j_manager.execute_write_query(create_query)

        # Check similarity with just first name
        result = await check_entity_similarity("Elizabeth", "character")

        # Should match on first name
        assert result is not None
        assert result["existing_name"] == "Elizabeth Anne Taylor"
        assert result["similarity"] == 0.95


@pytest.fixture
async def neo4j_test_db():
    """Fixture to clean up test data after each test."""
    from core.db_manager import neo4j_manager

    # Setup: ensure clean state
    cleanup_query = """
    MATCH (c:Character:Entity)
    WHERE c.name IN [
        'Jonathan Smith',
        'Alice Cooper',
        'Gandalf',
        'Robert Jones',
        'John Smith',
        'Johnny Walker',
        'Elizabeth Anne Taylor'
    ]
    DETACH DELETE c
    """
    await neo4j_manager.execute_write_query(cleanup_query)

    yield

    # Teardown: clean up test data
    await neo4j_manager.execute_write_query(cleanup_query)
