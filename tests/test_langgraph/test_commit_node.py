# tests/test_langgraph/test_commit_node.py
"""
Tests for LangGraph commit node (Step 1.2.1).

Tests the commit_to_graph node and its helper functions.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import get_type_hints
from unittest.mock import AsyncMock, patch

import pytest

import config
from core.langgraph.nodes.commit_node import (
    _build_chapter_node_statement,
    _build_entity_persistence_statements,
    _build_relationship_statements,
    _convert_to_character_profiles,
    _convert_to_world_items,
    commit_to_graph,
)
from core.langgraph.state import ExtractedEntity, ExtractedRelationship
from core.service_context import get_services
from models.kg_models import CharacterProfile, WorldItem
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager
from tests.fakes.service_context import configure_empty_entity_names
from tests.test_langgraph import InlineExtractionState, LegacyEmbeddingRef


@pytest.fixture(autouse=True)
def synthetic_entity_embeddings(monkeypatch: pytest.MonkeyPatch, fake_neo4j: FakeNeo4jManager) -> Iterator[None]:
    configure_empty_entity_names(fake_neo4j)
    async def embedding_batch(texts: list[str]) -> list[list[float]]:
        return [[0.25, 0.75, 0.0] for text in texts]

    monkeypatch.setattr(get_services().language_model, 'async_get_embeddings_batch', embedding_batch)
    effective = config.EffectiveSettings.model_validate({**config.snapshot_settings().model_dump(), "EXPECTED_EMBEDDING_DIM": 3, "EMBEDDING_DTYPE": "float64"})
    with config.bind_settings(effective):
        yield


def test_persistence_builder_annotations_resolve() -> None:
    annotations = get_type_hints(_build_entity_persistence_statements)
    assert annotations["characters"] == list[CharacterProfile]
    assert annotations["world_items"] == list[WorldItem]


def test_deduplication_annotations_resolve() -> None:
    from core.langgraph.nodes.commit_validation import _deduplicate_entity_list

    annotations = get_type_hints(_deduplicate_entity_list)
    assert annotations == {"entities": list[ExtractedEntity], "return": list[ExtractedEntity]}


@pytest.mark.asyncio
class TestCommitToGraph:
    """Tests for commit_to_graph node function."""

    async def test_commit_with_no_entities(
        self,
        sample_state_with_extraction: InlineExtractionState,
        fake_neo4j: FakeNeo4jManager,
    ) -> None:
        """Test commit with no extracted entities."""
        state = sample_state_with_extraction
        state["extracted_entities"] = {}
        state["extracted_relationships"] = []

        with patch("data_access.cypher_builders.native_builders.NativeCypherBuilder") as mock_builder_class:
            mock_builder = mock_builder_class.return_value
            mock_builder.character_upsert_cypher.return_value = ("query", {})
            mock_builder.world_item_upsert_cypher.return_value = ("query", {})

            with (
                patch("data_access.cache_coordinator.clear_character_read_caches") as mock_clear_chars,
                patch("data_access.cache_coordinator.clear_world_read_caches") as mock_clear_world,
                patch("data_access.cache_coordinator.clear_kg_read_caches") as mock_clear_kg,
            ):
                result = await commit_to_graph(state)

            assert result["current_node"] == "commit_to_graph"
            assert result["last_error"] is None

            assert len(fake_neo4j.batch_statements) == 1
            assert mock_clear_chars.called
            assert mock_clear_world.called
            assert mock_clear_kg.called

    async def test_commit_with_entities_and_relationships(
        self,
        sample_state_with_extraction: InlineExtractionState,
        fake_neo4j: FakeNeo4jManager,
    ) -> None:
        """Test commit with entities and relationships."""
        state = sample_state_with_extraction

        with patch("data_access.cypher_builders.native_builders.NativeCypherBuilder") as mock_builder_class:
            mock_builder = mock_builder_class.return_value
            mock_builder.character_upsert_cypher.return_value = ("query", {})
            mock_builder.world_item_upsert_cypher.return_value = ("query", {})

            with (
                patch("data_access.cache_coordinator.clear_character_read_caches") as mock_clear_chars,
                patch("data_access.cache_coordinator.clear_world_read_caches") as mock_clear_world,
                patch("data_access.cache_coordinator.clear_kg_read_caches") as mock_clear_kg,
            ):
                result = await commit_to_graph(state)

            assert result["current_node"] == "commit_to_graph"
            assert result["last_error"] is None

            assert len(fake_neo4j.batch_statements) == 1

            assert mock_clear_chars.called
            assert mock_clear_world.called
            assert mock_clear_kg.called

    async def test_extraction_normalization_commit_reads_normalized_ref(self, tmp_path: Path, fake_neo4j: FakeNeo4jManager) -> None:
        """
        End-to-end-ish unit test for remediation item 8:
        “Resolve normalization bypass (ref/version mismatch) so relationship normalization is effective”.

        This test verifies:
        - Extraction consolidation externalizes relationships and sets `extracted_relationships_ref`.
        - Relationship normalization persists normalized relationships and updates `extracted_relationships_ref`.
        - Commit reads relationships via the ref (prefers ref) and therefore uses normalized relationship types.

        Determinism:
        - We patch [`normalization_service.normalize_relationship_type()`](core/relationship_normalization_service.py:31)
          to avoid embeddings/LLM calls while still proving semantic normalization changes the predicate.
        """
        from core.langgraph.nodes.extraction_nodes import consolidate_extraction
        from core.langgraph.nodes.relationship_normalization_node import normalize_relationships
        from core.langgraph.state import ExtractedRelationship

        project_dir = str(tmp_path)

        from core.langgraph.content_manager import ContentManager

        content_manager = ContentManager(project_dir)
        draft_ref = content_manager.save_text("draft", "draft", "chapter_1", 1)

        entities_data = {
            "characters": [
                {
                    "name": "Alice",
                    "type": "Character",
                    "description": "Alice",
                    "first_appearance_chapter": 1,
                    "attributes": {},
                },
                {
                    "name": "Bob",
                    "type": "Character",
                    "description": "Bob",
                    "first_appearance_chapter": 1,
                    "attributes": {},
                },
            ],
            "world_items": [],
        }

        relationships_data = [
            {
                "source_name": "Alice",
                "target_name": "Bob",
                "relationship_type": "COLLABORATES_WITH",
                "description": "They collaborate",
                "chapter": 1,
                "confidence": 0.9,
            }
        ]

        from core.langgraph.content_manager import save_extracted_entities, save_extracted_relationships

        entities_ref = save_extracted_entities(content_manager, entities_data, 1, 1)
        relationships_ref = save_extracted_relationships(content_manager, relationships_data, 1, 1)

        state: InlineExtractionState = {
            "project_dir": project_dir,
            "current_chapter": 1,
            "draft_ref": draft_ref,
            "draft_word_count": 0,
            "relationship_vocabulary": {
                "WORKS_WITH": {
                    "canonical_type": "WORKS_WITH",
                    "usage_count": 1,
                    "first_used_chapter": 0,
                    "example_descriptions": [],
                    "synonyms": [],
                    "last_used_chapter": 0,
                }
            },
            "extracted_entities_ref": entities_ref,
            "extracted_relationships_ref": relationships_ref,
        }

        # Step 1: Consolidate extraction -> verifies externalized refs exist.
        extraction_update = await consolidate_extraction(state)
        state = {**state, **extraction_update}

        assert state.get("extracted_relationships_ref"), "consolidate_extraction must set extracted_relationships_ref"

        # Poison the in-memory field (if present) to prove normalize_relationships reads via ref only.
        state["extracted_relationships"] = [
            ExtractedRelationship(
                source_name="Alice",
                target_name="Bob",
                relationship_type="SHOULD_NOT_SEE",
                description="poison",
                chapter=1,
                confidence=0.9,
            )
        ]

        # Step 2: Normalize relationships -> must update extracted_relationships_ref to the normalized file.
        with patch("core.langgraph.nodes.relationship_normalization_node.config.ENABLE_RELATIONSHIP_NORMALIZATION", True):
            with patch(
                "core.langgraph.nodes.relationship_normalization_node.normalization_service.normalize_relationship_type",
                new=AsyncMock(return_value=("WORKS_WITH", True, 0.99)),
            ):
                normalization_update = await normalize_relationships(state)

        state = {**state, **normalization_update}

        assert state["extracted_relationships_ref"] is not None
        assert state["extracted_relationships_ref"]["version"] >= 2

        # Re-poison after normalization: commit should still ignore in-memory and read via ref.
        state["extracted_relationships"] = [
            ExtractedRelationship(
                source_name="Alice",
                target_name="Bob",
                relationship_type="SHOULD_NOT_SEE",
                description="poison-after-normalization",
                chapter=1,
                confidence=0.9,
            )
        ]

        # Step 3: Commit reads from extracted_relationships_ref (single source of truth) and uses WORKS_WITH.
        # Avoid relying on real chapter query builder in this focused test.
        with patch(
            "core.langgraph.nodes.commit_graph_ops.chapter_queries.build_chapter_upsert_statement",
            return_value=(
                "CHAPTER_UPSERT",
                {
                    "chapter_number_param": 1,
                    "chapter_id_param": "chapter_1",
                    "summary_param": None,
                    "embedding_vector_param": None,
                    "is_provisional_param": False,
                },
            ),
        ):
            with patch("data_access.cypher_builders.native_builders.NativeCypherBuilder") as mock_builder_class:
                mock_builder = mock_builder_class.return_value
                mock_builder.character_upsert_cypher.return_value = ("CHAR_UPSERT", {})
                mock_builder.world_item_upsert_cypher.return_value = ("WORLD_UPSERT", {})

                result = await commit_to_graph(state)
                assert result["last_error"] is None

                assert len(fake_neo4j.batch_statements) == 1
                statements = fake_neo4j.batch_statements[0]

                # Find relationship statements and assert predicate type is normalized.
                rel_statements = [(q, p) for (q, p) in statements if isinstance(q, str) and "CALL apoc.merge.relationship" in q]

                assert rel_statements, "Expected at least one relationship statement"

                # Contract: relationship type is passed as a parameter (not interpolated into the query string).
                assert any(p.get("predicate_clean") == "WORKS_WITH" for (_q, p) in rel_statements)
                assert all(p.get("predicate_clean") != "COLLABORATES_WITH" for (_q, p) in rel_statements)
                assert all(p.get("predicate_clean") != "SHOULD_NOT_SEE" for (_q, p) in rel_statements)

    async def test_commit_handles_errors_gracefully(
        self,
        sample_state_with_extraction: InlineExtractionState,
        fake_neo4j: FakeNeo4jManager,
    ) -> None:
        """Test that commit handles errors gracefully."""
        state = sample_state_with_extraction

        with patch("data_access.cypher_builders.native_builders.NativeCypherBuilder") as mock_builder_class:
            mock_builder = mock_builder_class.return_value
            mock_builder.character_upsert_cypher.side_effect = Exception("Database error")

            result = await commit_to_graph(state)

            assert result["current_node"] == "commit_to_graph"
            assert result["last_error"] is not None
            assert "Database error" in result["last_error"]

    async def test_commit_with_embedding_from_ref(
        self,
        sample_state_with_extraction: InlineExtractionState,
        fake_neo4j: FakeNeo4jManager,
    ) -> None:
        """Test commit with embedding loaded from content ref."""

        state = sample_state_with_extraction
        state["embedding_ref"] = LegacyEmbeddingRef(
            path="embeddings/chapter_1.npy",
            format="npy",
            content_type="embedding",
        )

        with patch("core.langgraph.nodes.commit_node.load_embedding") as mock_load:
            mock_load.return_value = [0.1, 0.2, 0.3]

            with patch("data_access.cypher_builders.native_builders.NativeCypherBuilder") as mock_builder_class:
                mock_builder = mock_builder_class.return_value
                mock_builder.character_upsert_cypher.return_value = ("query", {})
                mock_builder.world_item_upsert_cypher.return_value = ("query", {})

                result = await commit_to_graph(state)

                assert result["last_error"] is None
                assert mock_load.called

    async def test_commit_with_embedding_load_failure(
        self,
        sample_state_with_extraction: InlineExtractionState,
        fake_neo4j: FakeNeo4jManager,
    ) -> None:
        """Test commit when embedding load fails."""

        state = sample_state_with_extraction
        state["embedding_ref"] = LegacyEmbeddingRef(
            path="embeddings/chapter_1.npy",
            format="npy",
            content_type="embedding",
        )

        with patch("core.langgraph.nodes.commit_node.load_embedding") as mock_load:
            mock_load.side_effect = Exception("File not found")

            with patch("data_access.cypher_builders.native_builders.NativeCypherBuilder") as mock_builder_class:
                mock_builder = mock_builder_class.return_value
                mock_builder.character_upsert_cypher.return_value = ("query", {})
                mock_builder.world_item_upsert_cypher.return_value = ("query", {})

                result = await commit_to_graph(state)

                assert result["last_error"] is None

    async def test_commit_with_fallback_embedding(
        self,
        sample_state_with_extraction: InlineExtractionState,
        fake_neo4j: FakeNeo4jManager,
    ) -> None:
        """Test commit with an identified chapter artifact when scene vectors are absent."""
        from core.langgraph.content_manager import ContentManager, save_embedding
        state = sample_state_with_extraction
        state["embedding_ref"] = save_embedding(ContentManager(state["project_dir"]), [0.4, 0.5, 0.6], 1, embedding_model=config.EMBEDDING_MODEL)

        with patch("data_access.cypher_builders.native_builders.NativeCypherBuilder") as mock_builder_class:
            mock_builder = mock_builder_class.return_value
            mock_builder.character_upsert_cypher.return_value = ("query", {})
            mock_builder.world_item_upsert_cypher.return_value = ("query", {})

            result = await commit_to_graph(state)

            assert result["last_error"] is None

    async def test_commit_with_duplicate_world_items_in_batch(
        self,
        sample_state_with_extraction: InlineExtractionState,
        fake_neo4j: FakeNeo4jManager,
    ) -> None:
        """Test that within-batch duplicate world items are detected."""
        state = sample_state_with_extraction
        state["extracted_entities"] = {
            "world_items": [
                {
                    "name": "Castle",
                    "type": "location",
                    "description": "First castle",
                    "first_appearance_chapter": 1,
                    "attributes": {"category": "structure"},
                },
                {
                    "name": "Castle",
                    "type": "location",
                    "description": "Duplicate castle",
                    "first_appearance_chapter": 1,
                    "attributes": {"category": "structure"},
                },
            ]
        }
        state["extracted_relationships"] = []

        with patch("data_access.cypher_builders.native_builders.NativeCypherBuilder") as mock_builder_class:
            mock_builder = mock_builder_class.return_value
            mock_builder.world_item_upsert_cypher.return_value = ("query", {})

            with patch(
                "utils.text_processing.generate_entity_id",
                return_value="castle_001",
            ):
                result = await commit_to_graph(state)

                assert result["has_fatal_error"] is True
                assert result["last_error"] == "Commit to graph failed: Conflicting same-name entity inputs"
                assert fake_neo4j.batch_statements == []
                mock_builder.world_item_upsert_cypher.assert_not_called()


class TestConvertToCharacterProfiles:
    """Tests for _convert_to_character_profiles function."""

    def test_convert_character_entities(self) -> None:
        """Test converting ExtractedEntity to CharacterProfile."""
        entities = [
            ExtractedEntity(
                name="Alice",
                type="character",
                description="A brave warrior",
                first_appearance_chapter=1,
                attributes={
                    "traits": ["brave", "loyal", "determined"],
                    "status": "alive",
                    "relationships": {"Bob": {"type": "FRIEND", "description": "Close friend"}},
                },
            ),
        ]

        name_mappings = {"Alice": "Alice"}  # No deduplication
        profiles = _convert_to_character_profiles(entities, name_mappings, 1)

        assert len(profiles) == 1
        assert profiles[0].name == "Alice"
        assert profiles[0].personality_description == "A brave warrior"
        assert profiles[0].created_chapter == 1
        assert profiles[0].traits == ["brave", "loyal", "determined"]
        assert profiles[0].status == "alive"
        assert "Bob" in profiles[0].relationships

    def test_convert_with_deduplication_mapping(self) -> None:
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

    def test_convert_empty_list(self) -> None:
        """Test converting empty entity list."""
        profiles = _convert_to_character_profiles([], {}, 1)
        assert profiles == []

    def test_convert_character_with_no_traits(self) -> None:
        """Test converting character without traits attribute defaults to empty list."""
        entities = [
            ExtractedEntity(
                name="Bob",
                type="character",
                description="A mysterious figure",
                first_appearance_chapter=2,
                attributes={
                    "status": "Unknown",
                },
            ),
        ]

        name_mappings = {"Bob": "Bob"}
        profiles = _convert_to_character_profiles(entities, name_mappings, 2)

        assert len(profiles) == 1
        assert profiles[0].name == "Bob"
        assert profiles[0].traits == []  # Should default to empty list
        assert profiles[0].status == "Unknown"


class TestConvertToWorldItems:
    """Tests for _convert_to_world_items function."""

    def test_convert_world_item_entities(self) -> None:
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

    def test_convert_with_deduplication_mapping(self) -> None:
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

    def test_convert_handles_list_attributes(self) -> None:
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

    def test_convert_empty_list(self) -> None:
        """Test converting empty entity list."""
        items = _convert_to_world_items([], {}, 1)
        assert items == []


class TestDeduplicateEntityList:
    """Tests for _deduplicate_entity_list function."""

    def test_removes_duplicate_names(self) -> None:
        """Test that duplicate entity names are removed."""
        from core.langgraph.nodes.commit_node import _deduplicate_entity_list

        entities = [
            ExtractedEntity(
                name="Alice",
                type="character",
                description="First Alice",
                first_appearance_chapter=1,
                attributes={},
            ),
            ExtractedEntity(
                name="Bob",
                type="character",
                description="Bob",
                first_appearance_chapter=1,
                attributes={},
            ),
            ExtractedEntity(
                name="Alice",
                type="character",
                description="Duplicate Alice",
                first_appearance_chapter=2,
                attributes={},
            ),
        ]

        with pytest.raises(ValueError, match="Conflicting same-name entity inputs"):
            _deduplicate_entity_list(entities)

    def test_empty_list(self) -> None:
        """Test with empty list."""
        from core.langgraph.nodes.commit_node import _deduplicate_entity_list

        result = _deduplicate_entity_list([])
        assert result == []

    def test_no_duplicates(self) -> None:
        """Test with no duplicates."""
        from core.langgraph.nodes.commit_node import _deduplicate_entity_list

        entities = [
            ExtractedEntity(
                name="Alice",
                type="character",
                description="Alice",
                first_appearance_chapter=1,
                attributes={},
            ),
            ExtractedEntity(
                name="Bob",
                type="character",
                description="Bob",
                first_appearance_chapter=1,
                attributes={},
            ),
        ]

        result = _deduplicate_entity_list(entities)
        assert len(result) == 2


@pytest.mark.asyncio
class TestBuildEntityPersistenceStatements:
    """Tests for _build_entity_persistence_statements function."""

    async def test_builds_statements_for_characters_and_world_items(self) -> None:
        """Test building statements for both characters and world items."""
        from models.kg_models import CharacterProfile, WorldItem

        characters = [
            CharacterProfile(
                name="Alice",
                description="A brave warrior",
                traits=["brave", "loyal"],
                status="alive",
                relationships={},
                created_chapter=1,
                is_provisional=False,
                updates={},
            )
        ]

        world_items = [
            WorldItem(
                id="sword_001",
                category="artifact",
                name="Magic Sword",
                description="A legendary blade",
                goals=[],
                rules=[],
                key_elements=[],
                traits=[],
                created_chapter=1,
                is_provisional=False,
                additional_properties={},
            )
        ]

        with (
            patch("data_access.cypher_builders.native_builders.NativeCypherBuilder") as mock_builder_class,
            patch("core.langgraph.nodes.commit_node.config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False),
        ):
            mock_builder = mock_builder_class.return_value
            mock_builder.character_upsert_cypher.return_value = (
                "CHARACTER QUERY",
                {"name": "Alice"},
            )
            mock_builder.world_item_upsert_cypher.return_value = (
                "WORLD ITEM QUERY",
                {"id": "sword_001"},
            )

            statements = await _build_entity_persistence_statements(characters, world_items, 1)

            assert len(statements) == 2
            assert statements[0][0] == "CHARACTER QUERY"
            assert statements[1][0] == "WORLD ITEM QUERY"
            assert mock_builder.character_upsert_cypher.called
            assert mock_builder.world_item_upsert_cypher.called

    async def test_empty_entities(self) -> None:
        """Test with empty entity lists."""
        statements = await _build_entity_persistence_statements([], [], 1)
        assert statements == []


@pytest.mark.asyncio
class TestBuildRelationshipStatements:
    """Tests for _build_relationship_statements function."""

    async def test_builds_relationship_statements(self) -> None:
        """Test building relationship statements."""
        relationships = [
            ExtractedRelationship(
                source_name="Alice",
                relationship_type="KNOWS",
                target_name="Bob",
                description="They are friends",
                chapter=1,
                confidence=0.9,
            )
        ]

        char_entities = [
            ExtractedEntity(
                name="Alice",
                type="character",
                description="Alice",
                first_appearance_chapter=1,
                attributes={},
            ),
            ExtractedEntity(
                name="Bob",
                type="character",
                description="Bob",
                first_appearance_chapter=1,
                attributes={},
            ),
        ]

        with patch("core.langgraph.nodes.commit_node._get_cypher_labels") as mock_labels:
            mock_labels.return_value = ":Character"

            statements = await _build_relationship_statements(
                relationships,
                char_entities,
                [],
                {"Alice": "Alice", "Bob": "Bob"},
                {},
                1,
                False,
            )

            # Should have the DELETE and the MERGE
            assert len(statements) == 2
            query, params = statements[1]

            assert "CALL apoc.merge.relationship" in query
            assert params["predicate_clean"] == "KNOWS"

            assert params["subject_name"] == "Alice"
            assert params["object_name"] == "Bob"

    async def test_empty_relationships(self) -> None:
        """Test with empty relationships list. Must still include the DELETE for the chapter."""
        statements = await _build_relationship_statements([], [], [], {}, {}, 1, False)
        assert len(statements) == 1
        assert "DELETE r" in statements[0][0]

    async def test_applies_character_mappings(self) -> None:
        """Test that deduplication mappings are applied."""
        relationships = [
            ExtractedRelationship(
                source_name="NewAlice",
                relationship_type="KNOWS",
                target_name="Bob",
                description="They are friends",
                chapter=1,
                confidence=0.9,
            )
        ]

        char_entities = [
            ExtractedEntity(
                name="NewAlice",
                type="character",
                description="Alice",
                first_appearance_chapter=1,
                attributes={},
            ),
            ExtractedEntity(
                name="Bob",
                type="character",
                description="Bob",
                first_appearance_chapter=1,
                attributes={},
            ),
        ]

        with patch("core.langgraph.nodes.commit_node._get_cypher_labels") as mock_labels:
            mock_labels.return_value = ":Character"

            statements = await _build_relationship_statements(
                relationships,
                char_entities,
                [],
                {"NewAlice": "ExistingAlice", "Bob": "Bob"},
                {},
                1,
                False,
            )

            assert len(statements) == 2
            query, params = statements[1]
            assert params["subject_name"] == "ExistingAlice"

    @pytest.mark.usefixtures("offline_graph_reads")
    async def test_applies_world_item_mappings(self) -> None:
        """Test that world item mappings are applied."""
        relationships = [
            ExtractedRelationship(
                source_name="Alice",
                relationship_type="WIELDS",
                target_name="Magic Sword",
                description="Alice wields the sword",
                chapter=1,
                confidence=0.9,
            )
        ]

        char_entities = [
            ExtractedEntity(
                name="Alice",
                type="character",
                description="Alice",
                first_appearance_chapter=1,
                attributes={},
            )
        ]

        world_entities = [
            ExtractedEntity(
                name="Magic Sword",
                type="object",
                description="A sword",
                first_appearance_chapter=1,
                attributes={"category": "artifact"},
            )
        ]

        statements = await _build_relationship_statements(
            relationships, char_entities, world_entities, {"Alice": "Alice"}, {"Magic Sword": "sword_001"}, 1, False,
        )
        assert len(statements) == 2
        query, params = statements[1]
        assert params["object_name"] == "Magic Sword"
        assert params["object_id"] == "sword_001"
        assert params["object_label"] == "Item"

    @pytest.mark.usefixtures("offline_graph_reads")
    async def test_validates_entity_types(self) -> None:
        """Test that entity types are validated."""
        relationships = [
            ExtractedRelationship(
                source_name="Alice",
                relationship_type="CAUSED",
                target_name="Battle",
                description="Alice caused the battle",
                chapter=1,
                confidence=0.8,
            )
        ]

        char_entities = [
            ExtractedEntity(
                name="Alice",
                type="character",
                description="Alice",
                first_appearance_chapter=1,
                attributes={},
            )
        ]

        world_entities = [
            ExtractedEntity(
                name="Battle",
                type="DevelopmentEvent",
                description="A battle",
                first_appearance_chapter=1,
                attributes={},
            )
        ]

        with patch("core.langgraph.nodes.commit_node._get_cypher_labels") as mock_labels:
            # Mock is called multiple times: once in _resolve_entity_ids_from_db and twice in relationship building
            mock_labels.side_effect = [":Event", ":Character", ":DevelopmentEvent"]

            statements = await _build_relationship_statements(
                relationships,
                char_entities,
                world_entities,
                {"Alice": "Alice"},
                {"Battle": "Battle"},
                1,
                False,
            )

            assert len(statements) == 2

    async def test_canonicalizes_subtype_labels_for_persistence(self) -> None:
        """
        CORE-011: Subtype labels (e.g., "Guild") must not cross persistence boundaries.

        With the current canonical label set, "Guild" is not a valid node label and must be
        rejected at the persistence boundary.
        """
        relationships = [
            ExtractedRelationship(
                source_name="Alice",
                relationship_type="MEMBER_OF",
                target_name="The Guild",
                description="Alice is a member of the guild",
                chapter=1,
                confidence=0.9,
            )
        ]

        char_entities = [
            ExtractedEntity(
                name="Alice",
                type="Character",
                description="Alice",
                first_appearance_chapter=1,
                attributes={},
            ),
        ]

        world_entities = [
            ExtractedEntity(
                name="The Guild",
                type="Guild",
                description="A guild",
                first_appearance_chapter=1,
                attributes={"category": "Guild"},
            )
        ]

        with pytest.raises(ValueError, match=r"Invalid entity type 'Guild' at persistence boundary"):
            await _build_relationship_statements(
                relationships,
                char_entities,
                world_entities,
                {"Alice": "Alice"},
                {"The Guild": "The Guild"},
                1,
                False,
            )

    async def test_commit_rejects_unknown_entity_label_at_persistence_boundary(self, tmp_path: Path, fake_neo4j: FakeNeo4jManager) -> None:
        """
        CORE-011: Unknown/unmappable labels must be rejected at persistence boundary.

        This is an end-to-end-ish test through commit_to_graph (the write boundary).
        """
        project_dir = str(tmp_path)

        state: InlineExtractionState = {
            "project_dir": project_dir,
            "current_chapter": 1,
            "draft_word_count": 0,
            "extracted_entities": {
                "characters": [
                    ExtractedEntity(
                        name="Alice",
                        type="MadeUpLabel",  # invalid label -> must be rejected
                        description="Alice",
                        first_appearance_chapter=1,
                        attributes={},
                    ),
                    ExtractedEntity(
                        name="Bob",
                        type="Character",
                        description="Bob",
                        first_appearance_chapter=1,
                        attributes={},
                    ),
                ],
                "world_items": [],
            },
            "extracted_relationships": [
                ExtractedRelationship(
                    source_name="Alice",
                    target_name="Bob",
                    relationship_type="KNOWS",
                    description="They are friends",
                    chapter=1,
                    confidence=0.9,
                )
            ],
        }

        # Avoid relying on real chapter query builder in this focused test.
        with patch(
            "core.langgraph.nodes.commit_graph_ops.chapter_queries.build_chapter_upsert_statement",
            return_value=(
                "CHAPTER_UPSERT",
                {
                    "chapter_number_param": 1,
                    "chapter_id_param": "chapter_1",
                    "summary_param": None,
                    "embedding_vector_param": None,
                    "is_provisional_param": False,
                },
            ),
        ):
            with patch("data_access.cypher_builders.native_builders.NativeCypherBuilder") as mock_builder_class:
                mock_builder = mock_builder_class.return_value
                mock_builder.character_upsert_cypher.return_value = ("CHAR_UPSERT", {})
                mock_builder.world_item_upsert_cypher.return_value = ("WORLD_UPSERT", {})

                result = await commit_to_graph(state)

                assert result["has_fatal_error"] is True
                assert result["last_error"] is not None
                assert "Invalid entity type" in result["last_error"]
                assert "persistence boundary" in result["last_error"]


class TestBuildChapterNodeStatement:
    """Tests for _build_chapter_node_statement function."""

    def test_builds_statement_with_all_fields(self) -> None:
        """Test building statement with all fields (must include Chapter.id)."""
        query, params = _build_chapter_node_statement(
            chapter_number=1,
            summary="Chapter summary",
            embedding=[0.1, 0.2, 0.3],
        )

        # Canonical Chapter persistence query must:
        # - MERGE by number
        # - always set c.id via coalesce(...)
        assert "MERGE" in query
        assert "Chapter" in query
        assert "c.id" in query
        assert "coalesce" in query

        assert params["chapter_number_param"] == 1
        assert params["chapter_id_param"].startswith("chapter_")
        assert params["summary_param"] == "Chapter summary"
        assert params["embedding_vector_param"] == [0.1, 0.2, 0.3]
        # commit node sets is_provisional=false deterministically
        assert params["is_provisional_param"] is False

    def test_builds_statement_without_optional_fields(self) -> None:
        """Test building statement without summary and embedding (must still include Chapter.id)."""
        query, params = _build_chapter_node_statement(
            chapter_number=2,
            summary=None,
            embedding=None,
        )

        assert "MERGE" in query
        assert "c.id" in query
        assert params["chapter_number_param"] == 2
        assert params["chapter_id_param"].startswith("chapter_")
        # When summary is omitted, it should remain NULL in params (meaning "do not update")
        assert params["summary_param"] is None
        assert params["embedding_vector_param"] is None


class TestConversionEdgeCases:
    """Tests for edge cases in conversion functions."""

    def test_character_with_non_list_traits(self) -> None:
        """Test converting character when traits is not a list."""
        entities = [
            ExtractedEntity(
                name="Alice",
                type="character",
                description="Alice",
                first_appearance_chapter=1,
                attributes={
                    "traits": "brave",
                },
            )
        ]

        profiles = _convert_to_character_profiles(entities, {"Alice": "Alice"}, 1)

        assert len(profiles) == 1

    def test_character_with_invalid_traits(self) -> None:
        """Test that invalid traits are filtered out."""
        entities = [
            ExtractedEntity(
                name="Alice",
                type="character",
                description="Alice",
                first_appearance_chapter=1,
                attributes={
                    "traits": ["brave", "123", "loyal", ""],
                },
            )
        ]

        profiles = _convert_to_character_profiles(entities, {"Alice": "Alice"}, 1)

        assert len(profiles) == 1
        assert "123" not in profiles[0].traits
        assert "" not in profiles[0].traits

    def test_character_with_non_string_status(self) -> None:
        """Test that non-string status is converted to Unknown."""
        entities = [
            ExtractedEntity(
                name="Alice",
                type="character",
                description="Alice",
                first_appearance_chapter=1,
                attributes={
                    "status": 123,
                },
            )
        ]

        profiles = _convert_to_character_profiles(entities, {"Alice": "Alice"}, 1)

        assert len(profiles) == 1
        assert profiles[0].status == "Unknown"

    def test_world_item_with_non_list_attributes(self) -> None:
        """Test converting world item when list attributes are not lists."""
        entities = [
            ExtractedEntity(
                name="Sword",
                type="object",
                description="A sword",
                first_appearance_chapter=1,
                attributes={
                    "category": "weapon",
                    "goals": "destroy evil",
                    "rules": "only for good",
                    "key_elements": "magical",
                },
            )
        ]

        items = _convert_to_world_items(entities, {"Sword": "sword_001"}, 1)

        assert len(items) == 1
        assert isinstance(items[0].goals, list)
        assert isinstance(items[0].rules, list)
        assert isinstance(items[0].key_elements, list)

    def test_world_item_preserves_additional_properties(self) -> None:
        """Test that additional properties are preserved."""
        entities = [
            ExtractedEntity(
                name="Sword",
                type="object",
                description="A sword",
                first_appearance_chapter=1,
                attributes={
                    "category": "weapon",
                    "custom_field": "custom_value",
                    "another_field": 123,
                },
            )
        ]

        items = _convert_to_world_items(entities, {"Sword": "sword_001"}, 1)

        assert len(items) == 1
        assert items[0].additional_properties["custom_field"] == "custom_value"
        assert items[0].additional_properties["another_field"] == 123
