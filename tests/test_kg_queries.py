"""Tests for data_access/kg_queries.py"""
from unittest.mock import AsyncMock, patch

import pytest

from core.exceptions import DatabaseError
from data_access import kg_queries


class TestTypeInference:
    """Tests for type inference functions."""

    def test_infer_specific_node_type_character(self):
        """Test inferring Character type."""
        # `classify_category_label()` maps "person"/"people"/"npc" to "Character";
        # it does not treat "characters" as a special case (defaults to Item).
        result = kg_queries._infer_specific_node_type("Alice", category="person")
        assert result == "Character"

    def test_infer_specific_node_type_location(self):
        """Test inferring Location type."""
        # The current label taxonomy uses "Location" (not legacy "Structure").
        result = kg_queries._infer_specific_node_type("The Castle", category="Locations")
        assert result == "Location"

    def test_infer_specific_node_type_event(self):
        """Test inferring Event type."""
        result = kg_queries._infer_specific_node_type("The Battle", category="Events")
        assert result == "Event"

    def test_to_pascal_case(self):
        """Test converting to PascalCase."""
        assert kg_queries._to_pascal_case("hello world") == "HelloWorld"
        assert kg_queries._to_pascal_case("test_string") == "TestString"
        assert kg_queries._to_pascal_case("AlreadyPascal") == "AlreadyPascal"


class TestRelationshipTypeValidation:
    """Tests for relationship type validation and normalization."""

    def test_validate_relationship_type_known(self):
        """Test validating a known relationship type."""
        result = kg_queries.validate_relationship_type("FRIEND_OF")
        assert result == "FRIEND_OF"

    def test_validate_relationship_type_novel(self):
        """Test validating a novel relationship type."""
        result = kg_queries.validate_relationship_type("NOVEL_TYPE")
        assert result == "NOVEL_TYPE"

    def test_validate_relationship_type_normalization(self):
        """Test lenient relationship type normalization (non-security)."""
        assert kg_queries.validate_relationship_type("friend_of") == "FRIEND_OF"
        assert kg_queries.validate_relationship_type("FriendOf") == "FRIENDOF"
        assert kg_queries.validate_relationship_type("FRIEND_OF") == "FRIEND_OF"


class TestCypherLabelGeneration:
    """Tests for Cypher label generation."""

    def test_get_cypher_labels_character(self):
        """Test generating labels for Character type."""
        result = kg_queries._get_cypher_labels("Character")
        assert ":Character" in result or "Character" in result

    def test_get_cypher_labels_location(self):
        """Test generating labels for Location type."""
        result = kg_queries._get_cypher_labels("Location")
        assert ":Location" in result or "Location" in result

    def test_get_cypher_labels_none(self):
        """Test generating labels with no type."""
        with pytest.raises(ValueError, match="Entity type must be provided"):
            kg_queries._get_cypher_labels(None)


@pytest.mark.asyncio
class TestKGBatchOperations:
    """Tests for batch KG operations."""

    async def test_add_kg_triples_batch_empty(self, monkeypatch):
        """Test adding empty batch of triples."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(
            kg_queries.neo4j_manager, "execute_cypher_batch", mock_execute
        )

        result = await kg_queries.add_kg_triples_batch_to_db(
            [], 1, is_from_flawed_draft=False
        )
        assert result is None
        mock_execute.assert_not_called()

    async def test_add_kg_triples_batch_with_entities(self, monkeypatch):
        """Test adding batch with entity objects (current structured triple format)."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(
            kg_queries.neo4j_manager, "execute_cypher_batch", mock_execute
        )

        triples = [
            {
                "subject": {"name": "Alice", "type": "Character"},
                "predicate": "FRIEND_OF",
                "object_entity": {"name": "Bob", "type": "Character"},
                "is_literal_object": False,
            }
        ]

        result = await kg_queries.add_kg_triples_batch_to_db(
            triples, 1, is_from_flawed_draft=False
        )
        assert result is None
        mock_execute.assert_called_once()


@pytest.mark.asyncio
class TestKGQueries:
    """Tests for KG query functions."""

    async def test_query_kg_from_db(self, monkeypatch):
        """Test querying KG from database."""
        kg_queries.query_kg_from_db.cache_clear()

        mock_read = AsyncMock(
            return_value=[
                {
                    "subject": "Alice",
                    "predicate": "FRIEND_OF",
                    "object": "Bob",
                }
            ]
        )
        monkeypatch.setattr(
            kg_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await kg_queries.query_kg_from_db("Alice")
        assert len(result) > 0
        mock_read.assert_called_once()

    async def test_query_kg_from_db_raises_database_error_on_db_failure(self, monkeypatch):
        """P1.9: DB failures should raise standardized DatabaseError (not return [])."""
        kg_queries.query_kg_from_db.cache_clear()

        mock_read = AsyncMock(side_effect=Exception("connection refused"))
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        with pytest.raises(DatabaseError):
            await kg_queries.query_kg_from_db("Alice")

    async def test_get_novel_info_property_from_db(self, monkeypatch):
        """Test getting novel info property."""
        kg_queries.get_novel_info_property_from_db.cache_clear()

        mock_read = AsyncMock(return_value=[{"value": "Test Novel"}])
        monkeypatch.setattr(
            kg_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await kg_queries.get_novel_info_property_from_db("title")
        assert result == "Test Novel"

    async def test_get_novel_info_property_missing(self, monkeypatch):
        """Non-allowlisted NovelInfo keys are rejected (P0.4 Cypher injection hardening)."""
        kg_queries.get_novel_info_property_from_db.cache_clear()

        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            kg_queries.neo4j_manager, "execute_read_query", mock_read
        )

        with pytest.raises(ValueError, match=r"Unsafe NovelInfo property key"):
            await kg_queries.get_novel_info_property_from_db("missing")


@pytest.mark.asyncio
class TestKGCypherInjectionHardening:
    """Security regression tests for Cypher interpolation sites (P0.4)."""

    async def test_query_kg_from_db_rejects_unsafe_relationship_type(self, monkeypatch):
        """Unsafe relationship types must be rejected before Cypher interpolation."""
        kg_queries.query_kg_from_db.cache_clear()

        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        # Lowercase should be rejected (no silent normalization to uppercase).
        with pytest.raises(ValueError, match=r"Unsafe relationship type"):
            await kg_queries.query_kg_from_db(subject="Alice", predicate="friend_of")

        mock_read.assert_not_called()

    async def test_query_kg_from_db_allows_safe_relationship_type(self, monkeypatch):
        """A safe relationship type continues to work and is interpolated verbatim."""
        kg_queries.query_kg_from_db.cache_clear()

        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        await kg_queries.query_kg_from_db(subject="Alice", predicate="FRIEND_OF")

        mock_read.assert_called_once()
        called_query = mock_read.call_args.args[0]
        assert "MATCH (s)-[r:`FRIEND_OF`]->(o)" in called_query

    async def test_get_most_recent_value_from_db_rejects_unsafe_relationship_type(
        self, monkeypatch
    ):
        """Unsafe relationship types must be rejected before Cypher interpolation."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        with pytest.raises(ValueError, match=r"Unsafe relationship type"):
            await kg_queries.get_most_recent_value_from_db("Alice", "friend_of")

        mock_read.assert_not_called()

    async def test_add_kg_triples_batch_to_db_rejects_unsafe_relationship_type(
        self, monkeypatch
    ):
        """Batch KG writes must reject unsafe relationship types before interpolation."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(
            kg_queries.neo4j_manager, "execute_cypher_batch", mock_execute
        )

        triples = [
            {
                "subject": {"name": "Alice", "type": "Character"},
                # Contains spaces and lowercase (would previously be normalized).
                "predicate": "friend of",
                "object_entity": {"name": "Bob", "type": "Character"},
            }
        ]

        with pytest.raises(ValueError, match=r"Unsafe relationship type"):
            await kg_queries.add_kg_triples_batch_to_db(
                triples, chapter_number=1, is_from_flawed_draft=False
            )

        mock_execute.assert_not_called()


@pytest.mark.asyncio
class TestQualityAssuranceQueries:
    """Tests for QA-related queries."""

    async def test_find_contradictory_trait_characters(self, monkeypatch):
        """Test finding characters with contradictory traits."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            kg_queries.neo4j_manager, "execute_read_query", mock_read
        )

        contradictory_pairs = [("brave", "cowardly"), ("kind", "cruel")]
        result = await kg_queries.find_contradictory_trait_characters(contradictory_pairs)
        assert isinstance(result, list)
        assert mock_read.call_count == 2

    async def test_find_post_mortem_activity(self, monkeypatch):
        """Test finding post-mortem activity."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            kg_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await kg_queries.find_post_mortem_activity()
        assert isinstance(result, list)
        mock_read.assert_called_once()


@pytest.mark.asyncio
class TestEntityDeduplication:
    """Tests for entity deduplication."""

    async def test_find_candidate_duplicate_entities(self, monkeypatch):
        """Test finding candidate duplicate entities."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            kg_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await kg_queries.find_candidate_duplicate_entities()
        assert isinstance(result, list)

    async def test_get_entity_context_for_resolution(self, monkeypatch):
        """Test getting entity context for resolution."""
        mock_read = AsyncMock(
            return_value=[
                {
                    "entity_name": "Alice",
                    "entity_type": "Character",
                    "relationships": [],
                }
            ]
        )
        monkeypatch.setattr(
            kg_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await kg_queries.get_entity_context_for_resolution("alice_entity_id")
        assert result is not None

@pytest.mark.asyncio
class TestRelationshipMaintenance:
    """Tests for relationship maintenance operations."""

    async def test_deduplicate_relationships(self, monkeypatch):
        """Test deduplicating relationships."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(
            kg_queries.neo4j_manager, "execute_cypher_batch", mock_execute
        )

        result = await kg_queries.deduplicate_relationships()
        assert isinstance(result, int)

    async def test_consolidate_similar_relationships(self, monkeypatch):
        """Test consolidating similar relationships."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(
            kg_queries.neo4j_manager, "execute_cypher_batch", mock_execute
        )

        result = await kg_queries.consolidate_similar_relationships()
        assert isinstance(result, int)


@pytest.mark.asyncio
class TestDynamicRelationships:
    """Tests for dynamic relationship operations."""

    async def test_promote_dynamic_relationships(self, monkeypatch):
        """Test promoting dynamic relationships."""
        mock_read = AsyncMock(return_value=[{"count": 0}])
        monkeypatch.setattr(
            kg_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await kg_queries.promote_dynamic_relationships()
        assert result == 0


@pytest.mark.asyncio
class TestPathQueries:
    """Tests for path-based queries."""

    async def test_get_shortest_path_length_between_entities(self, monkeypatch):
        """Test getting shortest path length between entities."""
        # Implementation returns `length(p) AS len`
        mock_read = AsyncMock(return_value=[{"len": 2}])
        monkeypatch.setattr(
            kg_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await kg_queries.get_shortest_path_length_between_entities("Alice", "Bob")
        assert result == 2

    async def test_get_shortest_path_no_path(self, monkeypatch):
        """Test getting shortest path when no path exists."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(
            kg_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await kg_queries.get_shortest_path_length_between_entities(
            "Alice", "Zoe"
        )
        assert result is None


@pytest.mark.asyncio
class TestChapterContext:
    """Tests for chapter context queries."""

    async def test_get_chapter_context_for_entity(self, monkeypatch):
        """Test getting chapter context for entity."""
        mock_read = AsyncMock(
            return_value=[{"chapter": 1, "description": "First appearance"}]
        )
        monkeypatch.setattr(
            kg_queries.neo4j_manager, "execute_read_query", mock_read
        )

        result = await kg_queries.get_chapter_context_for_entity("Alice", 5)
        assert isinstance(result, list)
