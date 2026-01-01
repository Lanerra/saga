# tests/test_kg_queries.py
"""Tests for data_access/kg_queries.py"""

from unittest.mock import AsyncMock

import pytest
from neo4j.exceptions import ClientError

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
        """Canonical labels return a single strict Cypher label clause."""
        assert kg_queries._get_cypher_labels("Character") == ":Character"

    def test_get_cypher_labels_location(self):
        """Canonical labels return a single strict Cypher label clause."""
        assert kg_queries._get_cypher_labels("Location") == ":Location"

    def test_get_cypher_labels_normalizes_common_variant(self):
        """Common variants are normalized via schema validator (e.g., Person -> Character)."""
        assert kg_queries._get_cypher_labels("Person") == ":Character"

    def test_get_cypher_labels_rejects_unknown_label(self):
        """Unknown labels are rejected (strict schema enforcement)."""
        with pytest.raises(ValueError, match=r"Invalid node label"):
            kg_queries._get_cypher_labels("MadeUpLabel")

    def test_get_cypher_labels_none(self):
        """Missing entity type is rejected."""
        with pytest.raises(ValueError, match="Entity type must be provided"):
            kg_queries._get_cypher_labels(None)


@pytest.mark.asyncio
class TestKGBatchOperations:
    """Tests for batch KG operations."""

    async def test_add_kg_triples_batch_empty(self, monkeypatch):
        """Test adding empty batch of triples."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_cypher_batch", mock_execute)

        result = await kg_queries.add_kg_triples_batch_to_db([], 1, is_from_flawed_draft=False)
        assert result is None
        mock_execute.assert_not_called()

    async def test_add_kg_triples_batch_with_entities(self, monkeypatch):
        """Test adding batch with entity objects (current structured triple format)."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_cypher_batch", mock_execute)

        triples = [
            {
                "subject": {"name": "Alice", "type": "Character"},
                "predicate": "FRIEND_OF",
                "object_entity": {"name": "Bob", "type": "Character"},
                "is_literal_object": False,
            }
        ]

        result = await kg_queries.add_kg_triples_batch_to_db(triples, 1, is_from_flawed_draft=False)
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
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await kg_queries.query_kg_from_db("Alice")
        assert len(result) > 0
        mock_read.assert_called_once()

    async def test_query_kg_from_db_cached_result_is_defensive_copy(self, monkeypatch):
        """Mutating a cached read result must not contaminate future cache hits.

        Regression for: cached mutable return value risk.
        """
        kg_queries.query_kg_from_db.cache_clear()

        mock_read = AsyncMock(
            return_value=[
                {
                    "subject": "Alice",
                    "predicate": "FRIEND_OF",
                    "object": "Bob",
                    "meta": {"tags": ["t1"], "nested": [{"k": "v"}]},
                }
            ]
        )
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        # First call populates cache
        r1 = await kg_queries.query_kg_from_db(subject="Alice")
        assert r1[0]["meta"]["tags"] == ["t1"]
        assert r1[0]["meta"]["nested"][0]["k"] == "v"

        # Mutate deeply (these would contaminate the cache if the cached object were returned directly)
        r1[0]["meta"]["tags"].append("MUTATED")
        r1[0]["meta"]["nested"][0]["k"] = "CHANGED"

        # Second call should be served from cache (DB hit must not occur),
        # but MUST return an uncontaminated structure.
        r2 = await kg_queries.query_kg_from_db(subject="Alice")
        mock_read.assert_called_once()

        assert r2[0]["meta"]["tags"] == ["t1"]
        assert r2[0]["meta"]["nested"][0]["k"] == "v"

    async def test_query_kg_from_db_requires_filter_by_default(self, monkeypatch):
        """Guardrail: calling with no filters must fail fast (no full-graph scan)."""
        kg_queries.query_kg_from_db.cache_clear()

        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        with pytest.raises(ValueError, match=r"requires at least one filter"):
            await kg_queries.query_kg_from_db()

        mock_read.assert_not_called()

    async def test_query_kg_from_db_allows_unbounded_scan_when_explicit(self, monkeypatch):
        """Guardrail: caller may opt in explicitly to unbounded scan behavior."""
        kg_queries.query_kg_from_db.cache_clear()

        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        await kg_queries.query_kg_from_db(allow_unbounded_scan=True, limit_results=1)

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
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await kg_queries.get_novel_info_property_from_db("title")
        assert result == "Test Novel"

    async def test_get_novel_info_property_from_db_cached_value_is_defensive_copy(self, monkeypatch):
        """If NovelInfo returns a mutable structure, caching must not leak mutations."""
        kg_queries.get_novel_info_property_from_db.cache_clear()

        mock_read = AsyncMock(return_value=[{"value": {"arr": [1, 2], "obj": {"x": 1}}}])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        v1 = await kg_queries.get_novel_info_property_from_db("title")
        assert v1 == {"arr": [1, 2], "obj": {"x": 1}}

        # Deep mutation
        v1["arr"].append(999)
        v1["obj"]["x"] = 42

        v2 = await kg_queries.get_novel_info_property_from_db("title")
        mock_read.assert_called_once()

        assert v2 == {"arr": [1, 2], "obj": {"x": 1}}

    async def test_get_novel_info_property_missing(self, monkeypatch):
        """Non-allowlisted NovelInfo keys are rejected (P0.4 Cypher injection hardening)."""
        kg_queries.get_novel_info_property_from_db.cache_clear()

        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

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

    async def test_get_most_recent_value_from_db_rejects_unsafe_relationship_type(self, monkeypatch):
        """Unsafe relationship types must be rejected before Cypher interpolation."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        with pytest.raises(ValueError, match=r"Unsafe relationship type"):
            await kg_queries.get_most_recent_value_from_db("Alice", "friend_of")

        mock_read.assert_not_called()

    async def test_add_kg_triples_batch_to_db_rejects_unsafe_relationship_type(self, monkeypatch):
        """Batch KG writes must reject unsafe relationship types before interpolation."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_cypher_batch", mock_execute)

        triples = [
            {
                "subject": {"name": "Alice", "type": "Character"},
                # Contains spaces and lowercase (would previously be normalized).
                "predicate": "friend of",
                "object_entity": {"name": "Bob", "type": "Character"},
            }
        ]

        with pytest.raises(ValueError, match=r"Unsafe relationship type"):
            await kg_queries.add_kg_triples_batch_to_db(triples, chapter_number=1, is_from_flawed_draft=False)

        mock_execute.assert_not_called()


@pytest.mark.asyncio
class TestQualityAssuranceQueries:
    """Tests for QA-related queries."""

    async def test_find_contradictory_trait_characters(self, monkeypatch):
        """Test finding characters with contradictory traits."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        contradictory_pairs = [("brave", "cowardly"), ("kind", "cruel")]
        result = await kg_queries.find_contradictory_trait_characters(contradictory_pairs)
        assert isinstance(result, list)
        assert mock_read.call_count == 2

    async def test_find_post_mortem_activity(self, monkeypatch):
        """Test finding post-mortem activity."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await kg_queries.find_post_mortem_activity()
        assert isinstance(result, list)
        mock_read.assert_called_once()


@pytest.mark.asyncio
class TestEntityDeduplication:
    """Tests for entity deduplication."""

    async def test_find_candidate_duplicate_entities(self, monkeypatch):
        """Test finding candidate duplicate entities."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

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
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await kg_queries.get_entity_context_for_resolution("alice_entity_id")
        assert result is not None

    async def test_get_entity_context_for_resolution_raises_on_database_error(self, monkeypatch):
        """get_entity_context_for_resolution should propagate DatabaseError, not return None."""
        mock_read = AsyncMock(side_effect=ClientError("Invalid query"))
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        with pytest.raises(DatabaseError):
            await kg_queries.get_entity_context_for_resolution("entity123")

    async def test_get_entity_context_for_resolution_returns_none_when_not_found(self, monkeypatch):
        """When entity doesn't exist, should return None (not an error)."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await kg_queries.get_entity_context_for_resolution("nonexistent")
        assert result is None


@pytest.mark.asyncio
class TestRelationshipMaintenance:
    """Tests for relationship maintenance operations."""

    async def test_deduplicate_relationships(self, monkeypatch):
        """Test deduplicating relationships."""
        mock_write = AsyncMock(return_value=[{"removed": 0}])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_write_query", mock_write)

        result = await kg_queries.deduplicate_relationships()
        assert isinstance(result, int)
        mock_write.assert_called_once()

    async def test_consolidate_similar_relationships(self, monkeypatch):
        """Test consolidating similar relationships."""
        # consolidate_similar_relationships() first reads the current relationship types
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        # and only writes if there is something to consolidate
        mock_write = AsyncMock(return_value=[])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_write_query", mock_write)

        result = await kg_queries.consolidate_similar_relationships()
        assert isinstance(result, int)
        assert result == 0

        mock_read.assert_called_once()
        mock_write.assert_not_called()


@pytest.mark.asyncio
class TestDynamicRelationships:
    """Tests for dynamic relationship operations."""

    async def test_promote_dynamic_relationships(self, monkeypatch):
        """Test promoting dynamic relationships."""
        # Relationship maintenance can be disabled globally via config; this test exercises
        # the "enabled" code path so we can assert the underlying DB calls deterministically.
        monkeypatch.setattr(
            kg_queries.config.settings,
            "DISABLE_RELATIONSHIP_NORMALIZATION",
            False,
            raising=False,
        )

        # promote_dynamic_relationships() is now a thin wrapper around the canonical
        # relationship maintenance pipeline. It performs:
        # - _validate_and_correct_relationship_types(): execute_read_query (+ optional write updates)
        # - promotion: execute_write_query
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        mock_write = AsyncMock(return_value=[{"promoted": 0}])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_write_query", mock_write)

        result = await kg_queries.promote_dynamic_relationships()
        assert result == 0

        mock_read.assert_called_once()
        mock_write.assert_called_once()


@pytest.mark.asyncio
class TestPathQueries:
    """Tests for path-based queries."""

    async def test_get_shortest_path_length_between_entities(self, monkeypatch):
        """Test getting shortest path length between entities."""
        # Implementation returns `length(p) AS len`
        mock_read = AsyncMock(return_value=[{"len": 2}])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await kg_queries.get_shortest_path_length_between_entities("Alice", "Bob")
        assert result == 2

    async def test_get_shortest_path_no_path(self, monkeypatch):
        """Test getting shortest path when no path exists."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await kg_queries.get_shortest_path_length_between_entities("Alice", "Zoe")
        assert result is None

    async def test_get_shortest_path_length_between_entities_raises_on_database_error(self, monkeypatch):
        """get_shortest_path_length_between_entities should propagate DatabaseError, not return None."""
        from neo4j.exceptions import TransientError

        mock_read = AsyncMock(side_effect=TransientError("Timeout"))
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        with pytest.raises(DatabaseError):
            await kg_queries.get_shortest_path_length_between_entities("Alice", "Bob")

    async def test_get_shortest_path_length_between_entities_returns_none_when_no_path(self, monkeypatch):
        """When no path exists, should return None (not an error)."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await kg_queries.get_shortest_path_length_between_entities("Alice", "Bob")
        assert result is None


@pytest.mark.asyncio
class TestChapterContext:
    """Tests for chapter context queries."""

    async def test_get_chapter_context_for_entity_is_bounded_and_uses_subqueries(self, monkeypatch):
        """Guardrail: query should avoid OPTIONAL-MATCH row explosion and remain bounded."""
        mock_read = AsyncMock(return_value=[{"chapter_number": 1, "summary": "First appearance", "text": "..."}])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await kg_queries.get_chapter_context_for_entity(
            entity_name="Alice",
            chapter_context_limit=5,
        )
        assert isinstance(result, list)

        mock_read.assert_called_once()
        called_query = mock_read.call_args.args[0]
        called_params = mock_read.call_args.args[1]

        # Ensure the new bounded/subquery-based structure is in use.
        #
        # Neo4j 5+ deprecates `CALL { ... }` without a variable scope clause and prefers
        # `CALL (e) { ... }`. We accept either form here.
        assert ("CALL {" in called_query) or ("CALL (e) {" in called_query)
        assert "LIMIT $chapter_context_limit" in called_query

        # Ensure the expected bound params are provided.
        assert called_params["chapter_context_limit"] == 5
        assert called_params["max_event_chapters"] > 0
        assert called_params["max_rel_chapters"] > 0


@pytest.mark.asyncio
class TestNovelInfoPropertyCached:
    """Tests for _get_novel_info_property_from_db_cached exception propagation."""

    async def test_get_novel_info_property_cached_raises_on_database_error(self, monkeypatch):
        """_get_novel_info_property_from_db_cached should propagate DatabaseError, not return None."""
        from neo4j.exceptions import DatabaseUnavailable

        kg_queries._get_novel_info_property_from_db_cached.cache_clear()

        mock_read = AsyncMock(side_effect=DatabaseUnavailable("Database unavailable"))
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        with pytest.raises(DatabaseError):
            await kg_queries._get_novel_info_property_from_db_cached("title")

    async def test_get_novel_info_property_cached_returns_none_when_missing(self, monkeypatch):
        """When property doesn't exist, should return None (not an error)."""
        kg_queries._get_novel_info_property_from_db_cached.cache_clear()

        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await kg_queries._get_novel_info_property_from_db_cached("title")
        assert result is None
