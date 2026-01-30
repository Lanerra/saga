# tests/test_narrative_enrichment_parser.py
"""Test the NarrativeEnrichmentParser class.

This module tests the NarrativeEnrichmentParser class for:
1. Physical description extraction from narrative text
2. Chapter embedding extraction from narrative text
3. Validation of character enrichment
4. Persistence to Neo4j

Based on: docs/schema-design.md - Stage 5: Narrative Generation & Enrichment
"""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from core.parsers.narrative_enrichment_parser import (
    ChapterEmbeddingExtractionResult,
    NarrativeEnrichmentParser,
    PhysicalDescriptionExtractionResult,
)
from models.kg_models import Chapter, CharacterProfile


class MockCharacterProfile(BaseModel):
    """Mock character profile for testing."""

    name: str = "Test Character"
    personality_description: str = "Test personality"
    traits: list[str] = ["tall", "strong"]
    status: str = "Active"
    physical_description: str | None = None
    created_chapter: int = 0
    is_provisional: bool = False


class MockChapter(BaseModel):
    """Mock chapter for testing."""

    id: str = "chapter_1"
    number: int = 1
    title: str = "Test Chapter"
    summary: str = "Test summary"
    act_number: int = 1
    embedding: list[float] | None = None
    created_chapter: int = 1
    is_provisional: bool = False


@pytest.fixture
def sample_narrative_text():
    """Sample narrative text for testing."""
    return """
    Chapter 1: The Beginning
    
    John was a tall man with dark hair. He had a strong build and piercing eyes.
    
    Physical Description: John was 6 feet tall with dark hair and piercing eyes.
    
    Mary appeared in the room. She was short and had blonde hair.
    
    Appearance: Mary was 5 feet tall with blonde hair and a friendly smile.
    """


@pytest.fixture
def sample_character_data():
    """Sample character data for testing."""
    return [
        CharacterProfile(
            name="John",
            personality_description="A brave hero",
            traits=["tall", "strong"],
            status="Active",
            physical_description=None,
            created_chapter=0,
            is_provisional=False,
        ),
        CharacterProfile(
            name="Mary",
            personality_description="A kind friend",
            traits=["short", "friendly"],
            status="Active",
            physical_description=None,
            created_chapter=0,
            is_provisional=False,
        ),
    ]


@pytest.fixture
def sample_chapter_data():
    """Sample chapter data for testing."""
    return Chapter(
        id="chapter_1",
        number=1,
        title="Test Chapter",
        summary="Test summary",
        act_number=1,
        embedding=None,
        created_chapter=1,
        is_provisional=False,
    )


@pytest.mark.asyncio
async def test_extract_physical_descriptions(sample_narrative_text, sample_character_data):
    """Test physical description extraction from narrative text."""
    parser = NarrativeEnrichmentParser(sample_narrative_text, chapter_number=1)

    # Mock the get_character_profiles function
    with patch("core.parsers.narrative_enrichment_parser.get_character_profiles") as mock_get_chars:
        mock_get_chars.return_value = sample_character_data

        # Call the extract_physical_descriptions method
        results = await parser.extract_physical_descriptions()

        # Verify that the method was called
        mock_get_chars.assert_called_once()

        # Verify that results were returned
        assert len(results) > 0

        # Verify that the results contain the expected data
        for result in results:
            assert isinstance(result, PhysicalDescriptionExtractionResult)
            assert result.character_name in ["John", "Mary"]
            assert result.extracted_description
            assert result.confidence > 0


@pytest.mark.asyncio
async def test_extract_chapter_embeddings(sample_narrative_text, sample_chapter_data):
    """Test chapter embedding extraction from narrative text."""
    parser = NarrativeEnrichmentParser(sample_narrative_text, chapter_number=1)

    # Mock the get_chapter_data_from_db function
    with patch("core.parsers.narrative_enrichment_parser.get_chapter_data_from_db") as mock_get_chapter:
        mock_get_chapter.return_value = {
            "id": "chapter_1",
            "number": 1,
            "title": "Test Chapter",
            "summary": "Test summary",
            "act_number": 1,
            "embedding": None,
            "created_chapter": 1,
            "is_provisional": False,
        }

        # Mock the _generate_embedding_vector method
        with patch.object(parser, "_generate_embedding_vector", new_callable=AsyncMock) as mock_generate_embedding:
            mock_generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]

            # Call the extract_chapter_embeddings method
            results = await parser.extract_chapter_embeddings()

            # Verify that the method was called
            mock_get_chapter.assert_called_once_with(1)
            mock_generate_embedding.assert_called_once()

            # Verify that results were returned
            assert len(results) > 0

            # Verify that the results contain the expected data
            for result in results:
                assert isinstance(result, ChapterEmbeddingExtractionResult)
                assert result.chapter_number == 1
                assert result.embedding_vector
                assert result.confidence > 0


@pytest.mark.asyncio
async def test_validate_character_enrichment(sample_character_data):
    """Test validation of character enrichment."""
    parser = NarrativeEnrichmentParser("Test narrative", chapter_number=1)

    # Mock the get_character_profile_by_name function
    with patch("core.parsers.narrative_enrichment_parser.get_character_profile_by_name") as mock_get_char:
        mock_get_char.return_value = sample_character_data[0]

        # Test valid enrichment
        is_valid = await parser.validate_character_enrichment(
            "John",
            "John was 6 feet tall with dark hair and piercing eyes.",
        )
        assert is_valid is True

        # Test invalid enrichment (contradiction)
        is_valid = await parser.validate_character_enrichment(
            "John",
            "John was short and had blonde hair.",
        )
        assert is_valid is False


@pytest.mark.asyncio
async def test_update_character_physical_descriptions(sample_character_data):
    """Test updating character physical descriptions."""
    parser = NarrativeEnrichmentParser("Test narrative", chapter_number=1)

    # Create mock extraction results
    extraction_results = [
        PhysicalDescriptionExtractionResult(
            character_name="John",
            extracted_description="John was 6 feet tall with dark hair and piercing eyes.",
            confidence=0.9,
            source_text="Test source text",
            extraction_method="regex_pattern",
        )
    ]

    # Mock the get_character_profiles function
    with patch("core.parsers.narrative_enrichment_parser.get_character_profiles") as mock_get_chars:
        mock_get_chars.return_value = sample_character_data

        # Mock the sync_characters function
        with patch("core.parsers.narrative_enrichment_parser.sync_characters") as mock_sync_chars:
            mock_sync_chars.return_value = True

            # Call the update_character_physical_descriptions method
            success = await parser.update_character_physical_descriptions(extraction_results)

            # Verify that the method was called
            mock_get_chars.assert_called_once()
            mock_sync_chars.assert_called_once()

            # Verify that the method returned True
            assert success is True


@pytest.mark.asyncio
async def test_update_chapter_embeddings(sample_chapter_data):
    """Test updating chapter embeddings."""
    parser = NarrativeEnrichmentParser("Test narrative", chapter_number=1)

    # Create mock extraction results
    extraction_results = [
        ChapterEmbeddingExtractionResult(
            chapter_number=1,
            embedding_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
            confidence=0.95,
            source_text="Test source text",
            extraction_method="embedding_service",
        )
    ]

    # Mock the save_chapter_data_to_db function
    with patch("core.parsers.narrative_enrichment_parser.save_chapter_data_to_db") as mock_save_chapter:
        mock_save_chapter.return_value = None  # save_chapter_data_to_db returns None

        # Call the update_chapter_embeddings method
        success = await parser.update_chapter_embeddings(extraction_results)

        # Verify that the method was called
        mock_save_chapter.assert_called_once()

        # Verify that the method returned True
        assert success is True


@pytest.mark.asyncio
async def test_parse_and_persist(sample_narrative_text, sample_character_data, sample_chapter_data):
    """Test the complete parse and persist workflow."""
    parser = NarrativeEnrichmentParser(sample_narrative_text, chapter_number=1)

    # Mock the extract_physical_descriptions method
    with patch.object(parser, "extract_physical_descriptions", new_callable=AsyncMock) as mock_extract_physical:
        mock_extract_physical.return_value = [
            PhysicalDescriptionExtractionResult(
                character_name="John",
                extracted_description="John was 6 feet tall with dark hair and piercing eyes.",
                confidence=0.9,
                source_text="Test source text",
                extraction_method="regex_pattern",
            )
        ]

        # Mock the extract_chapter_embeddings method
        with patch.object(parser, "extract_chapter_embeddings", new_callable=AsyncMock) as mock_extract_chapter:
            mock_extract_chapter.return_value = [
                ChapterEmbeddingExtractionResult(
                    chapter_number=1,
                    embedding_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
                    confidence=0.95,
                    source_text="Test source text",
                    extraction_method="embedding_service",
                )
            ]

            # Mock the update_character_physical_descriptions method
            with patch.object(parser, "update_character_physical_descriptions", new_callable=AsyncMock) as mock_update_physical:
                mock_update_physical.return_value = True

                # Mock the update_chapter_embeddings method
                with patch.object(parser, "update_chapter_embeddings", new_callable=AsyncMock) as mock_update_chapter:
                    mock_update_chapter.return_value = True

                    # Call the parse_and_persist method
                    success, message = await parser.parse_and_persist()

                    # Verify that the method was called
                    mock_extract_physical.assert_called_once()
                    mock_extract_chapter.assert_called_once()
                    mock_update_physical.assert_called_once()
                    mock_update_chapter.assert_called_once()

                    # Verify that the method returned True
                    assert success is True

                    # Verify that the message contains the expected data
                    assert "Successfully parsed and persisted" in message
                    assert "character physical descriptions" in message
                    assert "chapter embeddings" in message


@pytest.mark.asyncio
async def test_empty_narrative_text():
    """Test that empty narrative text returns empty results."""
    parser = NarrativeEnrichmentParser("", chapter_number=1)

    # Call the extract_physical_descriptions method
    results = await parser.extract_physical_descriptions()

    # Verify that no results were returned
    assert len(results) == 0

    # Call the extract_chapter_embeddings method
    results = await parser.extract_chapter_embeddings()

    # Verify that no results were returned
    assert len(results) == 0


@pytest.mark.asyncio
async def test_invalid_character_name():
    """Test that invalid character names are handled gracefully."""
    parser = NarrativeEnrichmentParser("Test narrative", chapter_number=1)

    # Mock the get_character_profile_by_name function to return None
    with patch("core.parsers.narrative_enrichment_parser.get_character_profile_by_name") as mock_get_char:
        mock_get_char.return_value = None

        # Test invalid character name
        is_valid = await parser.validate_character_enrichment(
            "InvalidCharacter",
            "This character does not exist.",
        )

        # Verify that the method returned False
        assert is_valid is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
