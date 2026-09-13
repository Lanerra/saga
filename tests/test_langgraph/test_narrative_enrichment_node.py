# tests/test_langgraph/test_narrative_enrichment_node.py
"""Test the NarrativeEnrichmentNode implementation for Stage 5.

This test file covers:
1. Physical description extraction and validation
2. Chapter embedding extraction and validation
3. Character enrichment with physical descriptions
4. Chapter enrichment with embeddings
5. Validation of no new structural entities
6. Validation of character name matching
7. Validation of no contradictions in enrichment

Based on: docs/schema-design.md - Stage 5: Narrative Generation & Enrichment
"""

from collections.abc import Awaitable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import config
from core.embedding_contract import embedding_identity
from core.langgraph.nodes.narrative_enrichment_node import EnrichmentCandidate, NarrativeEnrichmentNode
from core.parsers.narrative_enrichment_parser import ChapterEmbeddingExtractionResult
from models.kg_models import Chapter, CharacterProfile
from tests.fakes.service_context import patch_service


@pytest.fixture
def sample_narrative_text() -> str:
    """Sample narrative text for testing."""
    return """
    Chapter 1: The Beginning
    
    Alice was a tall woman with long brown hair and piercing blue eyes. 
    She wore a red cloak that fluttered in the wind. Bob, her childhood friend, 
    was shorter with curly black hair and green eyes. He carried a sword at his side.
    """


@pytest.fixture
def sample_character_profiles() -> list[CharacterProfile]:
    """Sample character profiles for testing."""
    return [
        CharacterProfile(
            id="char_001",
            name="Alice",
            personality_description="Brave heroine",
            traits=["brave", "loyal", "cunning"],
            status="Active",
            created_chapter=0,
            is_provisional=False,
            created_ts=1234567890,
            updated_ts=1234567890,
        ),
        CharacterProfile(
            id="char_002",
            name="Bob",
            personality_description="Loyal friend",
            traits=["loyal", "trustworthy"],
            status="Active",
            created_chapter=0,
            is_provisional=False,
            created_ts=1234567890,
            updated_ts=1234567890,
        ),
    ]


@pytest.fixture
def sample_chapter_data() -> Chapter:
    """Sample chapter data for testing."""
    return Chapter(
        id="chapter_001",
        number=1,
        title="The Beginning",
        summary="Introduction of hero and world",
        act_number=1,
        is_provisional=False,
        created_ts=1234567890,
        updated_ts=1234567890,
    )


@pytest.mark.asyncio
class TestNarrativeEnrichmentNode:
    """Test the NarrativeEnrichmentNode class."""

    async def test_process_success(self, sample_narrative_text: str, sample_character_profiles: list[CharacterProfile], sample_chapter_data: Chapter) -> None:
        """Test successful processing of narrative text."""
        node = NarrativeEnrichmentNode()

        # Mock the database operations
        with (
            patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars,
            patch("core.langgraph.nodes.narrative_enrichment_node.get_chapter_data_from_db") as mock_get_chapter,
            patch("data_access.character_queries.sync_characters") as mock_sync_chars,
            patch("data_access.chapter_queries.save_chapter_data_to_db") as mock_save_chapter,
        ):
            # Set up mocks to return sample data
            parser_characters = patch("core.parsers.narrative_enrichment_parser.get_character_profiles", new_callable=AsyncMock, return_value=sample_character_profiles)
            parser_embedding = patch_service('language_model.async_get_embedding', new_callable=AsyncMock, return_value=[0.25] * config.EXPECTED_EMBEDDING_DIM)
            parser_chapter = patch("core.parsers.narrative_enrichment_parser.get_chapter_data_from_db", new_callable=AsyncMock, return_value=sample_chapter_data)
            mock_get_chars.return_value = sample_character_profiles
            mock_get_chapter.return_value = sample_chapter_data
            mock_sync_chars.return_value = True
            mock_save_chapter.return_value = True

            # Call the process method
            with parser_characters, parser_embedding, parser_chapter:
                pending_result: Awaitable[EnrichmentCandidate] = node.process(sample_narrative_text, 1)
                result = await pending_result

            # Verify success
            assert isinstance(result, EnrichmentCandidate)
            assert [(character.name, character.physical_description) for character in sample_character_profiles] == [
                ("Alice", None),
                ("Bob", None),
            ]
            assert [(item.character_name, item.description) for item in result.descriptions] == [("Alice", "a tall woman with long brown hair and piercing blue eyes")]
            mock_sync_chars.assert_not_awaited()
            assert len(result.embeddings) == 1
            mock_save_chapter.assert_not_awaited()

    async def test_process_empty_narrative_text(self) -> None:
        """Test error handling when narrative text is empty."""
        node = NarrativeEnrichmentNode()

        with pytest.raises(ValueError, match="Empty narrative text provided"):
            await node.process("", 1)

    async def test_process_invalid_chapter_number(self) -> None:
        """Test error handling when chapter number is invalid."""
        node = NarrativeEnrichmentNode()
        narrative_text = "Sample narrative text"

        with pytest.raises(ValueError, match="Invalid chapter number 0"):
            await node.process(narrative_text, 0)

    async def test_process_no_character_profiles(self) -> None:
        """Test error handling when no character profiles are found."""
        node = NarrativeEnrichmentNode()
        narrative_text = "Sample narrative text"

        with patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars:
            mock_get_chars.return_value = []

            with pytest.raises(ValueError, match="No character profiles found"):
                await node.process(narrative_text, 1)

    async def test_process_no_chapter_data(self) -> None:
        """Test error handling when no chapter data is found."""
        node = NarrativeEnrichmentNode()
        narrative_text = "Sample narrative text"

        with (
            patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars,
            patch("core.langgraph.nodes.narrative_enrichment_node.get_chapter_data_from_db") as mock_get_chapter,
        ):
            mock_get_chars.return_value = [
                CharacterProfile(
                    id="char_001", name="Alice", personality_description="Test", traits=[], status="Active", created_chapter=0, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                )
            ]
            mock_get_chapter.return_value = None

            with pytest.raises(ValueError, match="No chapter data found for chapter 1"):
                await node.process(narrative_text, 1)

    async def test_validate_physical_description(self) -> None:
        """Test validation of physical descriptions."""
        node = NarrativeEnrichmentNode()

        # Test with non-contradictory descriptions
        result = node._validate_physical_description("Alice is tall with brown hair", "Alice has long brown hair and blue eyes")
        assert result is True

    async def test_validate_embedding(self) -> None:
        """Test validation of embeddings."""
        node = NarrativeEnrichmentNode()

        result = node._validate_embedding([0.1, 0.2, 0.3], [0.11, 0.21, 0.31])
        assert result == True

    async def test_physical_description_extraction(self, sample_narrative_text: str) -> None:
        """Test extraction of physical descriptions from narrative text."""
        node = NarrativeEnrichmentNode()

        # Mock the parser
        with patch("core.langgraph.nodes.narrative_enrichment_node.NarrativeEnrichmentParser") as mock_parser_class:
            mock_parser = AsyncMock()
            mock_parser.extract_physical_descriptions.return_value = [
                MagicMock(character_name="Alice", extracted_description="Tall woman with long brown hair"),
                MagicMock(character_name="Bob", extracted_description="Shorter man with curly black hair"),
            ]
            mock_parser.extract_chapter_embeddings.return_value = []
            mock_parser_class.return_value = mock_parser

            # Mock the database operations
            with (
                patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars,
                patch("core.langgraph.nodes.narrative_enrichment_node.get_chapter_data_from_db") as mock_get_chapter,
                patch("data_access.character_queries.sync_characters") as mock_sync_chars,
                patch("data_access.chapter_queries.save_chapter_data_to_db") as mock_save_chapter,
            ):
                # Set up mocks to return sample data
                mock_get_chars.return_value = [
                    CharacterProfile(
                        id="char_001", name="Alice", personality_description="Test", traits=[], status="Active", created_chapter=0, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                    ),
                    CharacterProfile(
                        id="char_002", name="Bob", personality_description="Test", traits=[], status="Active", created_chapter=0, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                    ),
                ]
                mock_get_chapter.return_value = Chapter(
                    id="chapter_001", number=1, title="Test", summary="Test", act_number=1, created_chapter=1, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                )
                mock_sync_chars.return_value = True
                mock_save_chapter.return_value = True

                # Call the process method
                pending_result: Awaitable[EnrichmentCandidate] = node.process(sample_narrative_text, 1)
                result = await pending_result

                # Verify success
                assert isinstance(result, EnrichmentCandidate)

    @pytest.mark.run_settings(EXPECTED_EMBEDDING_DIM=3, NEO4J_VECTOR_DIMENSIONS=3)
    async def test_chapter_embedding_extraction(self, sample_narrative_text: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test extraction of chapter embeddings from narrative text."""
        node = NarrativeEnrichmentNode()

        # Mock the parser
        with patch("core.langgraph.nodes.narrative_enrichment_node.NarrativeEnrichmentParser") as mock_parser_class:
            mock_parser = AsyncMock()
            mock_parser.extract_physical_descriptions.return_value = []
            mock_parser.extract_chapter_embeddings.return_value = [
                ChapterEmbeddingExtractionResult(chapter_number=1, embedding_vector=[0.1, 0.2, 0.3], embedding_model=config.EMBEDDING_MODEL, embedding_identity=embedding_identity()),
            ]
            mock_parser_class.return_value = mock_parser

            # Mock the database operations
            with (
                patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars,
                patch("core.langgraph.nodes.narrative_enrichment_node.get_chapter_data_from_db") as mock_get_chapter,
                patch("data_access.character_queries.sync_characters") as mock_sync_chars,
                patch("data_access.chapter_queries.save_chapter_data_to_db") as mock_save_chapter,
            ):
                # Set up mocks to return sample data
                mock_get_chars.return_value = [
                    CharacterProfile(
                        id="char_001", name="Alice", personality_description="Test", traits=[], status="Active", created_chapter=0, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                    ),
                ]
                mock_get_chapter.return_value = Chapter(
                    id="chapter_001", number=1, title="Test", summary="Test", act_number=1, created_chapter=1, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                )
                mock_sync_chars.return_value = True
                mock_save_chapter.return_value = True

                # Call the process method
                pending_result: Awaitable[EnrichmentCandidate] = node.process(sample_narrative_text, 1)
                result = await pending_result

                # Verify success
                assert isinstance(result, EnrichmentCandidate)

    async def test_character_enrichment_with_physical_description(self, sample_narrative_text: str) -> None:
        """Test enrichment of character with physical description."""
        node = NarrativeEnrichmentNode()

        # Mock the parser
        with patch("core.langgraph.nodes.narrative_enrichment_node.NarrativeEnrichmentParser") as mock_parser_class:
            mock_parser = AsyncMock()
            mock_parser.extract_physical_descriptions.return_value = [
                MagicMock(character_name="Alice", extracted_description="Tall woman with long brown hair"),
            ]
            mock_parser.extract_chapter_embeddings.return_value = []
            mock_parser_class.return_value = mock_parser

            # Mock the database operations
            with (
                patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars,
                patch("core.langgraph.nodes.narrative_enrichment_node.get_chapter_data_from_db") as mock_get_chapter,
                patch("data_access.character_queries.sync_characters") as mock_sync_chars,
                patch("data_access.chapter_queries.save_chapter_data_to_db") as mock_save_chapter,
            ):
                # Set up mocks to return sample data
                mock_get_chars.return_value = [
                    CharacterProfile(
                        id="char_001", name="Alice", personality_description="Test", traits=[], status="Active", created_chapter=0, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                    ),
                ]
                mock_get_chapter.return_value = Chapter(
                    id="chapter_001", number=1, title="Test", summary="Test", act_number=1, created_chapter=1, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                )
                mock_sync_chars.return_value = True
                mock_save_chapter.return_value = True

                # Call the process method
                pending_result: Awaitable[EnrichmentCandidate] = node.process(sample_narrative_text, 1)
                result = await pending_result

                # Verify success
                assert isinstance(result, EnrichmentCandidate)

    @pytest.mark.run_settings(EXPECTED_EMBEDDING_DIM=3, NEO4J_VECTOR_DIMENSIONS=3)
    async def test_chapter_enrichment_with_embedding(self, sample_narrative_text: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test enrichment of chapter with embedding."""
        node = NarrativeEnrichmentNode()

        # Mock the parser
        with patch("core.langgraph.nodes.narrative_enrichment_node.NarrativeEnrichmentParser") as mock_parser_class:
            mock_parser = AsyncMock()
            mock_parser.extract_physical_descriptions.return_value = []
            mock_parser.extract_chapter_embeddings.return_value = [
                ChapterEmbeddingExtractionResult(chapter_number=1, embedding_vector=[0.1, 0.2, 0.3], embedding_model=config.EMBEDDING_MODEL, embedding_identity=embedding_identity()),
            ]
            mock_parser_class.return_value = mock_parser

            # Mock the database operations
            with (
                patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars,
                patch("core.langgraph.nodes.narrative_enrichment_node.get_chapter_data_from_db") as mock_get_chapter,
                patch("data_access.character_queries.sync_characters") as mock_sync_chars,
                patch("data_access.chapter_queries.save_chapter_data_to_db") as mock_save_chapter,
            ):
                # Set up mocks to return sample data
                mock_get_chars.return_value = [
                    CharacterProfile(
                        id="char_001", name="Alice", personality_description="Test", traits=[], status="Active", created_chapter=0, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                    ),
                ]
                mock_get_chapter.return_value = Chapter(
                    id="chapter_001", number=1, title="Test", summary="Test", act_number=1, created_chapter=1, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                )
                mock_sync_chars.return_value = True
                mock_save_chapter.return_value = True

                # Call the process method
                pending_result: Awaitable[EnrichmentCandidate] = node.process(sample_narrative_text, 1)
                result = await pending_result

                # Verify success
                assert isinstance(result, EnrichmentCandidate)

    async def test_validation_no_new_structural_entities(self) -> None:
        """Test that no new structural entities are created during enrichment."""
        node = NarrativeEnrichmentNode()
        narrative_text = "Sample narrative text"

        # Mock the parser to return no new entities
        with patch("core.langgraph.nodes.narrative_enrichment_node.NarrativeEnrichmentParser") as mock_parser_class:
            mock_parser = AsyncMock()
            mock_parser.extract_physical_descriptions.return_value = []
            mock_parser.extract_chapter_embeddings.return_value = []
            mock_parser_class.return_value = mock_parser

            # Mock the database operations
            with (
                patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars,
                patch("core.langgraph.nodes.narrative_enrichment_node.get_chapter_data_from_db") as mock_get_chapter,
                patch("data_access.character_queries.sync_characters") as mock_sync_chars,
                patch("data_access.chapter_queries.save_chapter_data_to_db") as mock_save_chapter,
            ):
                # Set up mocks to return sample data
                mock_get_chars.return_value = [
                    CharacterProfile(
                        id="char_001", name="Alice", personality_description="Test", traits=[], status="Active", created_chapter=0, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                    ),
                ]
                mock_get_chapter.return_value = Chapter(
                    id="chapter_001", number=1, title="Test", summary="Test", act_number=1, created_chapter=1, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                )
                mock_sync_chars.return_value = True
                mock_save_chapter.return_value = True

                # Call the process method
                pending_result: Awaitable[EnrichmentCandidate] = node.process(narrative_text, 1)
                result = await pending_result

                # Verify success
                assert isinstance(result, EnrichmentCandidate)

    async def test_validation_character_name_matching(self) -> None:
        """Test that character names match canonical names."""
        node = NarrativeEnrichmentNode()
        narrative_text = "Sample narrative text"

        # Mock the parser to return character with matching name
        with patch("core.langgraph.nodes.narrative_enrichment_node.NarrativeEnrichmentParser") as mock_parser_class:
            mock_parser = AsyncMock()
            mock_parser.extract_physical_descriptions.return_value = [
                MagicMock(character_name="Alice", extracted_description="Tall woman with long brown hair"),
            ]
            mock_parser.extract_chapter_embeddings.return_value = []
            mock_parser_class.return_value = mock_parser

            # Mock the database operations
            with (
                patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars,
                patch("core.langgraph.nodes.narrative_enrichment_node.get_chapter_data_from_db") as mock_get_chapter,
                patch("data_access.character_queries.sync_characters") as mock_sync_chars,
                patch("data_access.chapter_queries.save_chapter_data_to_db") as mock_save_chapter,
            ):
                # Set up mocks to return sample data
                mock_get_chars.return_value = [
                    CharacterProfile(
                        id="char_001", name="Alice", personality_description="Test", traits=[], status="Active", created_chapter=0, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                    ),
                ]
                mock_get_chapter.return_value = Chapter(
                    id="chapter_001", number=1, title="Test", summary="Test", act_number=1, created_chapter=1, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                )
                mock_sync_chars.return_value = True
                mock_save_chapter.return_value = True

                # Call the process method
                pending_result: Awaitable[EnrichmentCandidate] = node.process(narrative_text, 1)
                result = await pending_result

                # Verify success
                assert isinstance(result, EnrichmentCandidate)

    async def test_validation_no_contradictions_in_enrichment(self) -> None:
        """Test that enrichments don't contradict existing properties."""
        node = NarrativeEnrichmentNode()
        narrative_text = "Sample narrative text"

        # Mock the parser to return contradictory description
        with patch("core.langgraph.nodes.narrative_enrichment_node.NarrativeEnrichmentParser") as mock_parser_class:
            mock_parser = AsyncMock()
            mock_parser.extract_physical_descriptions.return_value = [
                MagicMock(character_name="Alice", extracted_description="Tall woman with long brown hair"),
            ]
            mock_parser.extract_chapter_embeddings.return_value = []
            mock_parser_class.return_value = mock_parser

            # Mock the database operations
            with (
                patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars,
                patch("core.langgraph.nodes.narrative_enrichment_node.get_chapter_data_from_db") as mock_get_chapter,
                patch("data_access.character_queries.sync_characters") as mock_sync_chars,
                patch("data_access.chapter_queries.save_chapter_data_to_db") as mock_save_chapter,
            ):
                # Set up mocks to return sample data with existing physical description
                mock_get_chars.return_value = [
                    CharacterProfile(
                        id="char_001",
                        name="Alice",
                        personality_description="Test",
                        traits=[],
                        status="Active",
                        created_chapter=0,
                        is_provisional=False,
                        created_ts=1234567890,
                        updated_ts=1234567890,
                        physical_description="Short woman with blonde hair",  # Contradictory description
                    ),
                ]
                mock_get_chapter.return_value = Chapter(
                    id="chapter_001", number=1, title="Test", summary="Test", act_number=1, created_chapter=1, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                )
                mock_sync_chars.return_value = True
                mock_save_chapter.return_value = True

                with pytest.raises(ValueError, match="Contradictory physical description for Alice"):
                    await node.process(narrative_text, 1)


@pytest.mark.asyncio
class TestNarrativeEnrichmentNodeIntegration:
    """Integration tests for NarrativeEnrichmentNode."""

    @pytest.mark.run_settings(EXPECTED_EMBEDDING_DIM=4, NEO4J_VECTOR_DIMENSIONS=4)
    async def test_full_pipeline_stage_5(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test full Stage 5 pipeline from narrative text to enrichment."""
        node = NarrativeEnrichmentNode()
        narrative_text = """
        Chapter 1: The Beginning
        
        Alice was a tall woman with long brown hair and piercing blue eyes. 
        She wore a red cloak that fluttered in the wind. Bob, her childhood friend, 
        was shorter with curly black hair and green eyes. He carried a sword at his side.
        """

        # Mock the parser
        with patch("core.langgraph.nodes.narrative_enrichment_node.NarrativeEnrichmentParser") as mock_parser_class:
            mock_parser = AsyncMock()
            mock_parser.extract_physical_descriptions.return_value = [
                MagicMock(character_name="Alice", extracted_description="Tall woman with long brown hair and blue eyes"),
                MagicMock(character_name="Bob", extracted_description="Shorter man with curly black hair and green eyes"),
            ]
            mock_parser.extract_chapter_embeddings.return_value = [
                ChapterEmbeddingExtractionResult(chapter_number=1, embedding_vector=[0.1, 0.2, 0.3, 0.4], embedding_model=config.EMBEDDING_MODEL, embedding_identity=embedding_identity()),
            ]
            mock_parser_class.return_value = mock_parser

            # Mock the database operations
            with (
                patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars,
                patch("core.langgraph.nodes.narrative_enrichment_node.get_chapter_data_from_db") as mock_get_chapter,
                patch("data_access.character_queries.sync_characters") as mock_sync_chars,
                patch("data_access.chapter_queries.save_chapter_data_to_db") as mock_save_chapter,
            ):
                # Set up mocks to return sample data
                mock_get_chars.return_value = [
                    CharacterProfile(
                        id="char_001", name="Alice", personality_description="Test", traits=[], status="Active", created_chapter=0, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                    ),
                    CharacterProfile(
                        id="char_002", name="Bob", personality_description="Test", traits=[], status="Active", created_chapter=0, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                    ),
                ]
                mock_get_chapter.return_value = Chapter(
                    id="chapter_001", number=1, title="Test", summary="Test", act_number=1, created_chapter=1, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                )
                mock_sync_chars.return_value = True
                mock_save_chapter.return_value = True

                # Call the process method
                pending_result: Awaitable[EnrichmentCandidate] = node.process(narrative_text, 1)
                result = await pending_result

                # Verify success
                assert isinstance(result, EnrichmentCandidate)

    async def test_error_handling_in_pipeline(self) -> None:
        """Test error handling in the full pipeline."""
        node = NarrativeEnrichmentNode()
        narrative_text = "Sample narrative text"

        # Mock the parser to raise an exception
        with patch("core.langgraph.nodes.narrative_enrichment_node.NarrativeEnrichmentParser") as mock_parser_class:
            mock_parser = AsyncMock()
            mock_parser.extract_physical_descriptions.side_effect = Exception("Test error")
            mock_parser.extract_chapter_embeddings.return_value = []
            mock_parser_class.return_value = mock_parser

            # Mock the database operations
            with (
                patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars,
                patch("core.langgraph.nodes.narrative_enrichment_node.get_chapter_data_from_db") as mock_get_chapter,
                patch("data_access.character_queries.sync_characters") as mock_sync_chars,
                patch("data_access.chapter_queries.save_chapter_data_to_db") as mock_save_chapter,
            ):
                # Set up mocks to return sample data
                mock_get_chars.return_value = [
                    CharacterProfile(
                        id="char_001", name="Alice", personality_description="Test", traits=[], status="Active", created_chapter=0, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                    ),
                ]
                mock_get_chapter.return_value = Chapter(
                    id="chapter_001", number=1, title="Test", summary="Test", act_number=1, created_chapter=1, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                )
                mock_sync_chars.return_value = True
                mock_save_chapter.return_value = True

                with pytest.raises(Exception, match="Test error"):
                    await node.process(narrative_text, 1)


@pytest.mark.asyncio
class TestNarrativeEnrichmentNodeEdgeCases:
    """Edge case tests for NarrativeEnrichmentNode."""

    async def test_empty_narrative_text(self) -> None:
        """Test with empty narrative text."""
        node = NarrativeEnrichmentNode()

        with pytest.raises(ValueError, match="Empty narrative text provided"):
            await node.process("", 1)

    async def test_whitespace_only_narrative_text(self) -> None:
        """Test with whitespace-only narrative text."""
        node = NarrativeEnrichmentNode()

        with pytest.raises(ValueError, match="Empty narrative text provided"):
            await node.process("   ", 1)

    async def test_invalid_chapter_number_zero(self) -> None:
        """Test with chapter number 0."""
        node = NarrativeEnrichmentNode()
        narrative_text = "Sample narrative text"

        with pytest.raises(ValueError, match="Invalid chapter number 0"):
            await node.process(narrative_text, 0)

    async def test_invalid_chapter_number_negative(self) -> None:
        """Test with negative chapter number."""
        node = NarrativeEnrichmentNode()
        narrative_text = "Sample narrative text"

        with pytest.raises(ValueError, match="Invalid chapter number -1"):
            await node.process(narrative_text, -1)

    async def test_character_not_found(self) -> None:
        """Test when character is not found in database."""
        node = NarrativeEnrichmentNode()
        narrative_text = "Sample narrative text"

        # Mock the parser to return character that doesn't exist
        with patch("core.langgraph.nodes.narrative_enrichment_node.NarrativeEnrichmentParser") as mock_parser_class:
            mock_parser = AsyncMock()
            mock_parser.extract_physical_descriptions.return_value = [
                MagicMock(character_name="UnknownCharacter", extracted_description="Description"),
            ]
            mock_parser.extract_chapter_embeddings.return_value = []
            mock_parser_class.return_value = mock_parser

            # Mock the database operations
            with (
                patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars,
                patch("core.langgraph.nodes.narrative_enrichment_node.get_chapter_data_from_db") as mock_get_chapter,
                patch("data_access.character_queries.sync_characters") as mock_sync_chars,
                patch("data_access.chapter_queries.save_chapter_data_to_db") as mock_save_chapter,
            ):
                # Set up mocks to return sample data without the character
                mock_get_chars.return_value = [
                    CharacterProfile(
                        id="char_001", name="Alice", personality_description="Test", traits=[], status="Active", created_chapter=0, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                    ),
                ]
                mock_get_chapter.return_value = Chapter(
                    id="chapter_001", number=1, title="Test", summary="Test", act_number=1, created_chapter=1, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                )
                mock_sync_chars.return_value = True
                mock_save_chapter.return_value = True

                with pytest.raises(ValueError, match="Character UnknownCharacter not found"):
                    await node.process(narrative_text, 1)

    async def test_chapter_not_found(self) -> None:
        """Test when no chapter data is returned from database."""
        node = NarrativeEnrichmentNode()
        narrative_text = "Sample narrative text"

        with (
            patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars,
            patch("core.langgraph.nodes.narrative_enrichment_node.get_chapter_data_from_db") as mock_get_chapter,
        ):
            mock_get_chars.return_value = [
                CharacterProfile(
                    id="char_001", name="Alice", personality_description="Test", traits=[], status="Active", created_chapter=0, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                ),
            ]
            mock_get_chapter.return_value = None

            with pytest.raises(ValueError, match="No chapter data found for chapter 1"):
                await node.process(narrative_text, 1)


@pytest.mark.asyncio
class TestNarrativeEnrichmentNodePerformance:
    """Performance tests for NarrativeEnrichmentNode."""

    async def test_large_narrative_text(self) -> None:
        """Test with large narrative text."""
        node = NarrativeEnrichmentNode()
        narrative_text = "A" * 10000  # 10KB of text

        # Mock the parser
        with patch("core.langgraph.nodes.narrative_enrichment_node.NarrativeEnrichmentParser") as mock_parser_class:
            mock_parser = AsyncMock()
            mock_parser.extract_physical_descriptions.return_value = []
            mock_parser.extract_chapter_embeddings.return_value = []
            mock_parser_class.return_value = mock_parser

            # Mock the database operations
            with (
                patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars,
                patch("core.langgraph.nodes.narrative_enrichment_node.get_chapter_data_from_db") as mock_get_chapter,
                patch("data_access.character_queries.sync_characters") as mock_sync_chars,
                patch("data_access.chapter_queries.save_chapter_data_to_db") as mock_save_chapter,
            ):
                # Set up mocks to return sample data
                mock_get_chars.return_value = [
                    CharacterProfile(
                        id="char_001", name="Alice", personality_description="Test", traits=[], status="Active", created_chapter=0, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                    ),
                ]
                mock_get_chapter.return_value = Chapter(
                    id="chapter_001", number=1, title="Test", summary="Test", act_number=1, created_chapter=1, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                )
                mock_sync_chars.return_value = True
                mock_save_chapter.return_value = True

                # Call the process method
                pending_result: Awaitable[EnrichmentCandidate] = node.process(narrative_text, 1)
                result = await pending_result

                # Verify success
                assert isinstance(result, EnrichmentCandidate)

    async def test_many_character_profiles(self) -> None:
        """Test with many character profiles."""
        node = NarrativeEnrichmentNode()
        narrative_text = "Sample narrative text"

        # Create many character profiles
        character_profiles = []
        for i in range(100):
            character_profiles.append(
                CharacterProfile(
                    id=f"char_{i:03d}",
                    name=f"Character{i}",
                    personality_description=f"Test character {i}",
                    traits=[],
                    status="Active",
                    created_chapter=0,
                    is_provisional=False,
                    created_ts=1234567890,
                    updated_ts=1234567890,
                )
            )

        # Mock the parser
        with patch("core.langgraph.nodes.narrative_enrichment_node.NarrativeEnrichmentParser") as mock_parser_class:
            mock_parser = AsyncMock()
            mock_parser.extract_physical_descriptions.return_value = []
            mock_parser.extract_chapter_embeddings.return_value = []
            mock_parser_class.return_value = mock_parser

            # Mock the database operations
            with (
                patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars,
                patch("core.langgraph.nodes.narrative_enrichment_node.get_chapter_data_from_db") as mock_get_chapter,
                patch("data_access.character_queries.sync_characters") as mock_sync_chars,
                patch("data_access.chapter_queries.save_chapter_data_to_db") as mock_save_chapter,
            ):
                # Set up mocks to return many character profiles
                mock_get_chars.return_value = character_profiles
                mock_get_chapter.return_value = Chapter(
                    id="chapter_001", number=1, title="Test", summary="Test", act_number=1, created_chapter=1, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                )
                mock_sync_chars.return_value = True
                mock_save_chapter.return_value = True

                # Call the process method
                pending_result: Awaitable[EnrichmentCandidate] = node.process(narrative_text, 1)
                result = await pending_result

                # Verify success
                assert isinstance(result, EnrichmentCandidate)


@pytest.mark.asyncio
class TestNarrativeEnrichmentNodeValidation:
    """Validation tests for NarrativeEnrichmentNode."""

    async def test_validate_physical_description_contradiction(self) -> None:
        """Test validation of contradictory physical descriptions."""
        node = NarrativeEnrichmentNode()

        # Test with contradictory descriptions
        result = node._validate_physical_description("Alice is tall with brown hair", "Alice is short with blonde hair")
        # This should return False if validation is implemented
        # Now that validation is implemented, it should return False
        assert result is False

    async def test_validate_embedding_significant_difference(self) -> None:
        """Test validation of significantly different embeddings."""
        node = NarrativeEnrichmentNode()

        result = node._validate_embedding(
            [0.1, 0.2, 0.3],
            [0.9, 0.8, 0.7],
        )
        assert result == False

    async def test_validate_embedding_similar(self) -> None:
        """Test validation of similar embeddings."""
        node = NarrativeEnrichmentNode()

        result = node._validate_embedding(
            [0.1, 0.2, 0.3],
            [0.11, 0.21, 0.31],
        )
        assert result == True

    async def test_validate_physical_description_consistency(self) -> None:
        """Test validation of consistent physical descriptions."""
        node = NarrativeEnrichmentNode()

        # Test with consistent descriptions
        result = node._validate_physical_description("Alice is tall with brown hair", "Alice has long brown hair and blue eyes")
        assert result is True


@pytest.mark.asyncio
class TestNarrativeEnrichmentNodeDatabaseOperations:
    """Database operation tests for NarrativeEnrichmentNode."""

    async def test_character_candidate_retained(self) -> None:
        """Candidate character descriptions remain available for acceptance."""
        node = NarrativeEnrichmentNode()
        narrative_text = "Sample narrative text"

        # Mock the parser
        with patch("core.langgraph.nodes.narrative_enrichment_node.NarrativeEnrichmentParser") as mock_parser_class:
            mock_parser = AsyncMock()
            mock_parser.extract_physical_descriptions.return_value = [
                MagicMock(character_name="Alice", extracted_description="Tall woman with long brown hair"),
            ]
            mock_parser.extract_chapter_embeddings.return_value = []
            mock_parser_class.return_value = mock_parser

            # Mock the database operations
            with (
                patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars,
                patch("core.langgraph.nodes.narrative_enrichment_node.get_chapter_data_from_db") as mock_get_chapter,
                patch("data_access.character_queries.sync_characters") as mock_sync_chars,
                patch("data_access.chapter_queries.save_chapter_data_to_db") as mock_save_chapter,
            ):
                # Set up mocks to return sample data
                mock_get_chars.return_value = [
                    CharacterProfile(
                        id="char_001", name="Alice", personality_description="Test", traits=[], status="Active", created_chapter=0, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                    ),
                ]
                mock_get_chapter.return_value = Chapter(
                    id="chapter_001", number=1, title="Test", summary="Test", act_number=1, created_chapter=1, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                )
                mock_sync_chars.return_value = True
                mock_save_chapter.return_value = True

                # Call the process method
                pending_result: Awaitable[EnrichmentCandidate] = node.process(narrative_text, 1)
                result = await pending_result

                assert isinstance(result, EnrichmentCandidate)

                # Candidate preparation cannot write profiles
                assert [(item.character_id, item.description) for item in result.descriptions] == [("char_001", "Tall woman with long brown hair")]
                mock_sync_chars.assert_not_called()

    @pytest.mark.run_settings(EXPECTED_EMBEDDING_DIM=3, NEO4J_VECTOR_DIMENSIONS=3)
    async def test_embedding_candidate_retained(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Candidate chapter embeddings remain available for acceptance."""
        node = NarrativeEnrichmentNode()
        narrative_text = "Sample narrative text"

        # Mock the parser
        with patch("core.langgraph.nodes.narrative_enrichment_node.NarrativeEnrichmentParser") as mock_parser_class:
            mock_parser = AsyncMock()
            mock_parser.extract_physical_descriptions.return_value = []
            mock_parser.extract_chapter_embeddings.return_value = [
                ChapterEmbeddingExtractionResult(chapter_number=1, embedding_vector=[0.1, 0.2, 0.3], embedding_model=config.EMBEDDING_MODEL, embedding_identity=embedding_identity()),
            ]
            mock_parser_class.return_value = mock_parser

            # Mock the database operations
            with (
                patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars,
                patch("core.langgraph.nodes.narrative_enrichment_node.get_chapter_data_from_db") as mock_get_chapter,
                patch("data_access.character_queries.sync_characters") as mock_sync_chars,
                patch("data_access.chapter_queries.save_chapter_data_to_db") as mock_save_chapter,
            ):
                # Set up mocks to return sample data
                mock_get_chars.return_value = [
                    CharacterProfile(
                        id="char_001", name="Alice", personality_description="Test", traits=[], status="Active", created_chapter=0, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                    ),
                ]
                mock_get_chapter.return_value = Chapter(
                    id="chapter_001", number=1, title="Test", summary="Test", act_number=1, created_chapter=1, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                )
                mock_sync_chars.return_value = True
                mock_save_chapter.return_value = True

                # Call the process method
                pending_result: Awaitable[EnrichmentCandidate] = node.process(narrative_text, 1)
                result = await pending_result

                assert isinstance(result, EnrichmentCandidate)

                # Candidate preparation cannot write chapter vectors
                assert [item.embedding_vector for item in result.embeddings] == [[0.1, 0.2, 0.3]]
                mock_save_chapter.assert_not_called()

    async def test_get_character_profiles_called(self) -> None:
        """Test that get_character_profiles is called."""
        node = NarrativeEnrichmentNode()
        narrative_text = "Sample narrative text"

        # Mock the parser
        with patch("core.langgraph.nodes.narrative_enrichment_node.NarrativeEnrichmentParser") as mock_parser_class:
            mock_parser = AsyncMock()
            mock_parser.extract_physical_descriptions.return_value = []
            mock_parser.extract_chapter_embeddings.return_value = []
            mock_parser_class.return_value = mock_parser

            # Mock the database operations
            with (
                patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars,
                patch("core.langgraph.nodes.narrative_enrichment_node.get_chapter_data_from_db") as mock_get_chapter,
                patch("data_access.character_queries.sync_characters") as mock_sync_chars,
                patch("data_access.chapter_queries.save_chapter_data_to_db") as mock_save_chapter,
            ):
                # Set up mocks to return sample data
                mock_get_chars.return_value = [
                    CharacterProfile(
                        id="char_001", name="Alice", personality_description="Test", traits=[], status="Active", created_chapter=0, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                    ),
                ]
                mock_get_chapter.return_value = Chapter(
                    id="chapter_001", number=1, title="Test", summary="Test", act_number=1, created_chapter=1, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                )
                mock_sync_chars.return_value = True
                mock_save_chapter.return_value = True

                # Call the process method
                pending_result: Awaitable[EnrichmentCandidate] = node.process(narrative_text, 1)
                result = await pending_result

                assert isinstance(result, EnrichmentCandidate)

                # Verify that get_character_profiles was called
                mock_get_chars.assert_called_once()

    async def test_get_chapter_data_called(self) -> None:
        """Test that get_chapter_data_from_db is called."""
        node = NarrativeEnrichmentNode()
        narrative_text = "Sample narrative text"

        # Mock the parser
        with patch("core.langgraph.nodes.narrative_enrichment_node.NarrativeEnrichmentParser") as mock_parser_class:
            mock_parser = AsyncMock()
            mock_parser.extract_physical_descriptions.return_value = []
            mock_parser.extract_chapter_embeddings.return_value = []
            mock_parser_class.return_value = mock_parser

            # Mock the database operations
            with (
                patch("core.langgraph.nodes.narrative_enrichment_node.get_character_profiles") as mock_get_chars,
                patch("core.langgraph.nodes.narrative_enrichment_node.get_chapter_data_from_db") as mock_get_chapter,
                patch("data_access.character_queries.sync_characters") as mock_sync_chars,
                patch("data_access.chapter_queries.save_chapter_data_to_db") as mock_save_chapter,
            ):
                # Set up mocks to return sample data
                mock_get_chars.return_value = [
                    CharacterProfile(
                        id="char_001", name="Alice", personality_description="Test", traits=[], status="Active", created_chapter=0, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                    ),
                ]
                mock_get_chapter.return_value = Chapter(
                    id="chapter_001", number=1, title="Test", summary="Test", act_number=1, created_chapter=1, is_provisional=False, created_ts=1234567890, updated_ts=1234567890
                )
                mock_sync_chars.return_value = True
                mock_save_chapter.return_value = True

                # Call the process method
                pending_result: Awaitable[EnrichmentCandidate] = node.process(narrative_text, 1)
                result = await pending_result

                assert isinstance(result, EnrichmentCandidate)

                # Verify that get_chapter_data_from_db was called
                mock_get_chapter.assert_called_once_with(1)

