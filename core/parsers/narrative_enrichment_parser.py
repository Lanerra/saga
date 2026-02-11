# core/parsers/narrative_enrichment_parser.py
"""Parse narrative text and extract enrichment data for Stage 5.

This module provides the NarrativeEnrichmentParser class that:
1. Extracts physical descriptions from narrative text
2. Updates Character nodes with physical_description property
3. Extracts chapter embeddings from narrative text
4. Updates Chapter nodes with embedding property
5. Validates that enrichments don't contradict existing properties

Based on: docs/schema-design.md - Stage 5: Narrative Generation & Enrichment
"""

from __future__ import annotations

import re

import numpy as np
import structlog
from pydantic import BaseModel

import config
from core.entity_embedding_service import (
    compute_entity_embedding_text,
    compute_entity_embedding_text_hash,
)
from data_access.chapter_queries import (
    get_chapter_data_from_db,
    save_chapter_data_to_db,
)
from data_access.character_queries import (
    get_character_profile_by_name,
    get_character_profiles,
    sync_characters,
)
from models.kg_models import CharacterProfile

logger = structlog.get_logger(__name__)


# Import llm_service for embedding generation
try:
    from core.llm_interface_refactored import llm_service
except ImportError:
    # Fallback if llm_service is not available
    llm_service = None  # type: ignore


class PhysicalDescriptionExtractionResult(BaseModel):
    """Result of physical description extraction from narrative text."""

    character_name: str
    extracted_description: str
    confidence: float = 0.8
    source_text: str = ""
    extraction_method: str = "regex_pattern"


class ChapterEmbeddingExtractionResult(BaseModel):
    """Result of chapter embedding extraction from narrative text."""

    chapter_number: int
    embedding_vector: list[float]
    confidence: float = 0.8
    source_text: str = ""
    extraction_method: str = "embedding_service"


class NarrativeEnrichmentParser:
    """Parse narrative text and extract enrichment data for Stage 5.

    This parser handles Stage 5 of the knowledge graph construction:
    - Extract physical descriptions from narrative text
    - Update Character nodes with physical_description property
    - Extract chapter embeddings from narrative text
    - Update Chapter nodes with embedding property
    - Validate that enrichments don't contradict existing properties

    Attributes:
        narrative_text: The narrative text to parse
        chapter_number: Chapter number for provenance
        extraction_model: Model to use for extraction (default: settings.SMALL_MODEL)
    """

    def __init__(
        self,
        narrative_text: str,
        chapter_number: int = 0,
        extraction_model: str = "gpt-4",
    ):
        """Initialize the NarrativeEnrichmentParser.

        Args:
            narrative_text: The narrative text to parse
            chapter_number: Chapter number for provenance (0 for initialization)
            extraction_model: Model to use for extraction
        """
        self.narrative_text = narrative_text
        self.chapter_number = chapter_number
        self.extraction_model = extraction_model

    async def extract_physical_descriptions(self) -> list[PhysicalDescriptionExtractionResult]:
        """Extract physical descriptions from narrative text.

        This method uses regex patterns to extract physical descriptions from the narrative text.
        It looks for patterns like:
        - "Physical Description: ..."
        - "Appearance: ..."
        - Character names followed by descriptions

        Returns:
            List of PhysicalDescriptionExtractionResult objects

        Raises:
            ValueError: If narrative text cannot be parsed
            DatabaseError: If there are issues accessing the database
        """
        # Check if physical description extraction is enabled
        if not config.ENABLE_PHYSICAL_DESCRIPTION_EXTRACTION:
            logger.info("Physical description extraction is disabled by configuration")
            return []

        if not self.narrative_text:
            logger.warning("No narrative text provided for physical description extraction")
            return []

        # Get all character profiles to validate against
        characters = await get_character_profiles()
        if not characters:
            logger.warning("No characters found in database for validation")
            return []

        # Create a mapping of character names to their profiles
        character_map = {char.name: char for char in characters}

        # Extract physical descriptions using regex patterns
        results = []

        # Pattern 1: Look for "Physical Description:" or "Appearance:" followed by text
        physical_desc_pattern = r"(Physical\s*Description|Appearance):\s*(.*?)(?=\n|\.|!|\?|$)"
        matches = re.findall(physical_desc_pattern, self.narrative_text, re.IGNORECASE | re.DOTALL)

        for match in matches:
            # match[0] is the label (Physical Description or Appearance)
            # match[1] is the description text
            description_text = match[1].strip()

            # Try to extract character name from context
            # Look backwards for a character name
            context = self.narrative_text[: self.narrative_text.find(match[0])]
            last_character_name = self._extract_last_mentioned_character(context, character_map)

            if last_character_name:
                result = PhysicalDescriptionExtractionResult(
                    character_name=last_character_name,
                    extracted_description=description_text,
                    confidence=0.9,
                    source_text=match[0] + ": " + description_text,
                    extraction_method="regex_pattern",
                )
                results.append(result)

        # Pattern 2: Look for character names followed by descriptions
        # This is more complex and may require LLM assistance
        # For now, we'll use a simpler pattern

        # Extract all unique character names from the narrative
        mentioned_characters = self._extract_mentioned_characters(self.narrative_text, character_map)

        for char_name in mentioned_characters:
            # Check if this character already has a physical description
            if char_name in character_map and character_map[char_name].physical_description:
                continue

            # Extract description using a more sophisticated approach
            char_desc = self._extract_character_description(char_name, self.narrative_text)
            if char_desc:
                result = PhysicalDescriptionExtractionResult(
                    character_name=char_name,
                    extracted_description=char_desc,
                    confidence=0.7,
                    source_text=self.narrative_text,
                    extraction_method="contextual_extraction",
                )
                results.append(result)

        return results

    def _extract_last_mentioned_character(self, text: str, character_map: dict[str, CharacterProfile]) -> str | None:
        """Extract the last mentioned character from text.

        Args:
            text: The text to search
            character_map: Mapping of character names to profiles

        Returns:
            The name of the last mentioned character, or None if not found
        """
        # Simple approach: look for capitalized words that match character names
        words = re.findall(r"\b[A-Z][a-zA-Z']*\b", text)
        words.reverse()  # Search from end to beginning

        for word in words:
            if word in character_map:
                return word

        return None

    def _extract_mentioned_characters(self, text: str, character_map: dict[str, CharacterProfile]) -> list[str]:
        """Extract all mentioned character names from text.

        Args:
            text: The text to search
            character_map: Mapping of character names to profiles

        Returns:
            List of character names found in the text
        """
        mentioned = []

        # Simple approach: look for capitalized words that match character names
        words = re.findall(r"\b[A-Z][a-zA-Z']*\b", text)

        for word in words:
            if word in character_map and word not in mentioned:
                mentioned.append(word)

        return mentioned

    def _extract_character_description(self, character_name: str, text: str) -> str | None:
        """Extract description for a specific character from text.

        Args:
            character_name: Name of the character
            text: The text to search

        Returns:
            Extracted description text, or None if not found
        """
        # Look for patterns like "[character_name] was [description]"
        # or "[character_name]'s [description]"

        escaped_name = re.escape(character_name)
        patterns = [
            rf"{escaped_name}\s+was\s+([^.!?]+)",
            rf"{escaped_name}\s+'s\s+([^.!?]+)",
            rf"{escaped_name}\s+had\s+([^.!?]+)",
            rf"{escaped_name}\s+looked\s+([^.!?]+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                return matches[0].strip()

        return None

    async def extract_chapter_embeddings(self) -> list[ChapterEmbeddingExtractionResult]:
        """Extract chapter embeddings from narrative text.

        This method uses the entity embedding service to extract embeddings from the narrative text.

        Returns:
            List of ChapterEmbeddingExtractionResult objects

        Raises:
            ValueError: If narrative text cannot be parsed
            DatabaseError: If there are issues accessing the database
        """
        # Check if chapter embedding extraction is enabled
        if not config.ENABLE_CHAPTER_EMBEDDING_EXTRACTION:
            logger.info("Chapter embedding extraction is disabled by configuration")
            return []

        if not self.narrative_text:
            logger.warning("No narrative text provided for chapter embedding extraction")
            return []

        # Get the chapter to update
        chapter = await get_chapter_data_from_db(self.chapter_number)
        if not chapter:
            logger.warning(f"No chapter found with number {self.chapter_number}")
            return []

        # Extract embedding using the embedding service
        try:
            # Compute the embedding text
            embedding_text = compute_entity_embedding_text(
                name=chapter.title,
                description=chapter.summary,
                category=f"Chapter {chapter.number}",
            )

            # Compute the embedding text hash for change detection (used for validation)
            _ = compute_entity_embedding_text_hash(embedding_text)

            # Generate the embedding vector
            embedding_vector = await self._generate_embedding_vector(embedding_text)

            if embedding_vector:
                result = ChapterEmbeddingExtractionResult(
                    chapter_number=self.chapter_number,
                    embedding_vector=embedding_vector,
                    confidence=0.95,
                    source_text=embedding_text,
                    extraction_method="embedding_service",
                )
                return [result]
            else:
                logger.warning(f"No embedding vector generated for chapter {self.chapter_number}")
                return []

        except Exception as e:
            logger.error(f"Error extracting chapter embedding: {str(e)}", exc_info=True)
            return []

    async def _generate_embedding_vector(self, text: str) -> list[float] | None:
        """Generate embedding vector from text using the embedding service.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector, or None if generation fails
        """
        try:
            embedding = await llm_service.async_get_embedding(text)

            if embedding is not None and len(embedding) > 0:
                return embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
            else:
                logger.warning(f"Empty embedding vector generated for text: {text[:100]}")
                return None

        except Exception as e:
            logger.error(f"Error generating embedding vector: {str(e)}", exc_info=True)
            return None

    async def validate_character_enrichment(
        self,
        character_name: str,
        new_physical_description: str,
    ) -> bool:
        """Validate that the new physical description doesn't contradict existing properties.

        Args:
            character_name: Name of the character to validate
            new_physical_description: New physical description to validate

        Returns:
            True if validation passes, False otherwise
        """
        # Check if physical description validation is enabled
        if not config.ENABLE_PHYSICAL_DESCRIPTION_VALIDATION:
            logger.info("Physical description validation is disabled by configuration")
            return True

        try:
            # Get the character profile
            character = await get_character_profile_by_name(character_name)

            if not character:
                logger.warning(f"Character {character_name} not found in database")
                return False

            # Check if the new description contradicts existing traits
            # For example, if the character has "tall" trait but description says "short"

            # Check for contradictions in the description
            contradictions = self._check_for_contradictions(character, new_physical_description)

            if contradictions:
                logger.warning(
                    f"Contradictions found in physical description for {character_name}: {contradictions}",
                    extra={"character": character_name, "contradictions": contradictions},
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating character enrichment: {str(e)}", exc_info=True)
            return False

    @staticmethod
    def _word_in_text(word: str, text: str) -> bool:
        """Check if a whole word appears in text using word-boundary matching."""
        return bool(re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE))

    def _check_for_contradictions(
        self,
        character: CharacterProfile,
        new_description: str,
    ) -> list[str]:
        """Check for contradictions between existing character properties and new description.

        Args:
            character: Character profile to check
            new_description: New physical description to validate

        Returns:
            List of contradiction messages
        """
        contradictions = []

        _CONTRADICTORY_DESCRIPTION_PAIRS = [
            ("tall", "short"),
            ("young", "old"),
        ]

        for trait_a, trait_b in _CONTRADICTORY_DESCRIPTION_PAIRS:
            if trait_a in character.traits and self._word_in_text(trait_b, new_description):
                contradictions.append(f"Description mentions '{trait_b}' but character has '{trait_a}' trait")
            if trait_b in character.traits and self._word_in_text(trait_a, new_description):
                contradictions.append(f"Description mentions '{trait_a}' but character has '{trait_b}' trait")

        if character.status == "Dead" and self._word_in_text("alive", new_description):
            contradictions.append("Description mentions 'alive' but character status is 'Dead'")

        return contradictions

    async def update_character_physical_descriptions(
        self,
        extraction_results: list[PhysicalDescriptionExtractionResult],
    ) -> bool:
        """Update character nodes with physical descriptions.

        Args:
            extraction_results: List of PhysicalDescriptionExtractionResult objects

        Returns:
            True if successful, False otherwise
        """
        if not extraction_results:
            logger.warning("No extraction results provided for character update")
            return False

        try:
            # Get all character profiles to update
            characters = await get_character_profiles()
            if not characters:
                logger.warning("No characters found in database for update")
                return False

            # Create a mapping of character names to their profiles
            character_map = {char.name: char for char in characters}

            # Update each character with the new physical description
            for result in extraction_results:
                character_name = result.character_name

                if character_name not in character_map:
                    logger.warning(f"Character {character_name} not found in database")
                    continue

                # Validate the enrichment before applying
                is_valid = await self.validate_character_enrichment(character_name, result.extracted_description)

                if not is_valid:
                    logger.warning(
                        f"Validation failed for physical description of {character_name}. Skipping update.",
                        extra={"character": character_name},
                    )
                    continue

                # Update the character profile
                character = character_map[character_name]

                # Only update if the new description is different from the existing one
                if character.physical_description != result.extracted_description:
                    character.physical_description = result.extracted_description
                    character.updated_ts = None  # Will be set by Neo4j

                    logger.info(
                        f"Updating physical description for {character_name}",
                        extra={"character": character_name, "description": result.extracted_description[:50]},
                    )

            # Sync the updated characters to the database
            success = await sync_characters(list(character_map.values()), self.chapter_number)

            if not success:
                logger.error("Failed to sync character updates to database")
                return False

            logger.info(
                f"Successfully updated {len(extraction_results)} character physical descriptions",
                extra={"chapter": self.chapter_number},
            )

            return True

        except Exception as e:
            logger.error(f"Error updating character physical descriptions: {str(e)}", exc_info=True)
            return False

    async def update_chapter_embeddings(
        self,
        extraction_results: list[ChapterEmbeddingExtractionResult],
    ) -> bool:
        """Update chapter nodes with embeddings.

        Args:
            extraction_results: List of ChapterEmbeddingExtractionResult objects

        Returns:
            True if successful, False otherwise
        """
        if not extraction_results:
            logger.warning("No extraction results provided for chapter embedding update")
            return False

        try:
            # Update each chapter with the new embedding
            for result in extraction_results:
                chapter_number = result.chapter_number
                embedding_vector = result.embedding_vector

                # Sync the embedding to the database using save_chapter_data_to_db
                await save_chapter_data_to_db(
                    chapter_number=chapter_number,
                    embedding_array=np.array(embedding_vector, dtype=np.float32),
                )

                logger.info(
                    f"Successfully updated embedding for chapter {chapter_number}",
                    extra={"chapter": chapter_number, "embedding_dim": len(embedding_vector)},
                )

            return True

        except Exception as e:
            logger.error(f"Error updating chapter embeddings: {str(e)}", exc_info=True)
            return False

    async def parse_and_persist(self) -> tuple[bool, str]:
        """Parse narrative text and persist enrichment data to Neo4j.

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Step 1: Extract physical descriptions (only if enabled)
            if config.ENABLE_PHYSICAL_DESCRIPTION_EXTRACTION:
                logger.info("Extracting physical descriptions from narrative text")
                physical_descriptions = await self.extract_physical_descriptions()

                if not physical_descriptions:
                    logger.warning("No physical descriptions extracted from narrative text")

                logger.info(
                    f"Extracted {len(physical_descriptions)} physical descriptions from narrative text",
                    extra={"chapter": self.chapter_number},
                )
            else:
                physical_descriptions = []
                logger.info("Physical description extraction is disabled by configuration")

            # Step 2: Extract chapter embeddings (only if enabled)
            if config.ENABLE_CHAPTER_EMBEDDING_EXTRACTION:
                logger.info("Extracting chapter embeddings from narrative text")
                chapter_embeddings = await self.extract_chapter_embeddings()

                if not chapter_embeddings:
                    logger.warning("No chapter embeddings extracted from narrative text")

                logger.info(
                    f"Extracted {len(chapter_embeddings)} chapter embeddings from narrative text",
                    extra={"chapter": self.chapter_number},
                )
            else:
                chapter_embeddings = []
                logger.info("Chapter embedding extraction is disabled by configuration")

            # Step 3: Update character physical descriptions
            if physical_descriptions:
                logger.info("Updating character physical descriptions in Neo4j")
                char_success = await self.update_character_physical_descriptions(physical_descriptions)

                if not char_success:
                    return False, "Failed to update character physical descriptions"

            # Step 4: Update chapter embeddings
            if chapter_embeddings:
                logger.info("Updating chapter embeddings in Neo4j")
                chapter_success = await self.update_chapter_embeddings(chapter_embeddings)

                if not chapter_success:
                    return False, "Failed to update chapter embeddings"

            # Build success message
            message_parts = []
            if physical_descriptions:
                message_parts.append(f"{len(physical_descriptions)} character physical descriptions")
            if chapter_embeddings:
                message_parts.append(f"{len(chapter_embeddings)} chapter embeddings")

            if not message_parts:
                return True, "No enrichment data found in narrative text"

            return True, f"Successfully parsed and persisted {', '.join(message_parts)}"

        except Exception as e:
            logger.error("Error in parse_and_persist: %s", str(e), exc_info=True)
            return False, f"Error parsing and persisting narrative enrichment: {str(e)}"


__all__ = [
    "NarrativeEnrichmentParser",
    "PhysicalDescriptionExtractionResult",
    "ChapterEmbeddingExtractionResult",
]
