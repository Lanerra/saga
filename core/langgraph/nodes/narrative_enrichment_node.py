# core/langgraph/nodes/narrative_enrichment_node.py
"""Narrative enrichment node for Stage 5: Narrative Generation & Enrichment.

This module defines the narrative enrichment node used by the narrative generation
workflow. This node:
1. Extracts physical descriptions from narrative text
2. Updates Character nodes with physical_description property
3. Extracts chapter embeddings from narrative text
4. Updates Chapter nodes with embedding property
5. Validates that enrichments don't contradict existing properties

Based on: docs/schema-design.md - Stage 5: Narrative Generation & Enrichment
"""

from typing import Any

import numpy as np
import structlog

from core.langgraph.content_manager import (
    ContentManager,
    get_draft_text,
    require_project_dir,
)
from core.langgraph.state import NarrativeState
from core.parsers.narrative_enrichment_parser import (
    NarrativeEnrichmentParser,
)
from data_access.chapter_queries import (
    get_chapter_data_from_db,
    save_chapter_data_to_db,
)
from data_access.character_queries import (
    get_character_profiles,
    sync_characters,
)

logger = structlog.get_logger(__name__)


class NarrativeEnrichmentNode:
    """Narrative enrichment node for Stage 5.

    This node handles the enrichment of the knowledge graph with narrative details
    without creating new structural entities. It extracts physical descriptions and
    chapter embeddings from narrative text and updates the existing nodes.

    Attributes:
        None
    """

    def __init__(self) -> None:
        """Initialize the NarrativeEnrichmentNode."""

    async def process(
        self,
        narrative_text: str,
        chapter_number: int,
    ) -> None:
        """Process narrative text and extract enrichment data.

        Args:
            narrative_text: The narrative text to parse
            chapter_number: Chapter number for provenance

        Raises:
            ValueError: If inputs are invalid or enrichment encounters contradictions
        """
        if not narrative_text or len(narrative_text.strip()) == 0:
            raise ValueError("Empty narrative text provided")

        if chapter_number <= 0:
            raise ValueError(f"Invalid chapter number {chapter_number}")

        character_profiles = await get_character_profiles()
        if not character_profiles:
            raise ValueError("No character profiles found")

        chapter_data = await get_chapter_data_from_db(chapter_number)
        if not chapter_data:
            raise ValueError(f"No chapter data found for chapter {chapter_number}")

        parser = NarrativeEnrichmentParser(
            narrative_text=narrative_text,
            chapter_number=chapter_number,
        )

        physical_descriptions = await parser.extract_physical_descriptions()
        if not physical_descriptions:
            logger.warning("NarrativeEnrichmentNode: No physical descriptions extracted")

        chapter_embeddings = await parser.extract_chapter_embeddings()
        if not chapter_embeddings:
            logger.warning("NarrativeEnrichmentNode: No chapter embeddings extracted")

        if physical_descriptions:
            for desc in physical_descriptions:
                character_name = desc.character_name
                extracted_description = desc.extracted_description

                character = next((c for c in character_profiles if c.name == character_name), None)

                if not character:
                    raise ValueError(f"Character {character_name} not found")

                if character.physical_description:
                    if not self._validate_physical_description(
                        character.physical_description,
                        extracted_description,
                    ):
                        raise ValueError(f"Contradictory physical description for {character_name}")
                    character.physical_description = extracted_description
                    await sync_characters([character], chapter_number)
                    logger.info(
                        "NarrativeEnrichmentNode: Updated character physical description",
                        character_name=character_name,
                    )
                else:
                    character.physical_description = extracted_description
                    await sync_characters([character], chapter_number)
                    logger.info(
                        "NarrativeEnrichmentNode: Added character physical description",
                        character_name=character_name,
                    )

        if chapter_embeddings:
            for embedding in chapter_embeddings:
                embedding_vector = embedding.embedding_vector
                embedding_array = np.array(embedding_vector) if isinstance(embedding_vector, list) else embedding_vector

                if chapter_data.embedding:
                    if not self._validate_embedding(
                        chapter_data.embedding,
                        embedding_vector,
                    ):
                        raise ValueError(f"Invalid embedding for chapter {chapter_data.number}")
                    await save_chapter_data_to_db(
                        chapter_number=chapter_data.number,
                        title=chapter_data.title,
                        act_number=chapter_data.act_number,
                        summary=chapter_data.summary,
                        embedding_array=embedding_array,
                        is_provisional=chapter_data.is_provisional,
                    )
                    logger.info(
                        "NarrativeEnrichmentNode: Updated chapter embedding",
                        chapter_number=chapter_data.number,
                    )
                else:
                    await save_chapter_data_to_db(
                        chapter_number=chapter_data.number,
                        title=chapter_data.title,
                        act_number=chapter_data.act_number,
                        summary=chapter_data.summary,
                        embedding_array=embedding_array,
                        is_provisional=chapter_data.is_provisional,
                    )
                    logger.info(
                        "NarrativeEnrichmentNode: Added chapter embedding",
                        chapter_number=chapter_data.number,
                    )

    def _validate_physical_description(
        self,
        existing_description: str,
        new_description: str,
    ) -> bool:
        """Validate that new physical description doesn't contradict existing one.

        Args:
            existing_description: Existing physical description
            new_description: New physical description to validate

        Returns:
            bool: True if valid, False if contradictory
        """
        existing_lower = existing_description.lower()
        new_lower = new_description.lower()

        if self._has_contradictory_height(existing_lower, new_lower):
            logger.error(
                "NarrativeEnrichmentNode: Contradictory height in physical description",
                existing=existing_description,
                new=new_description,
            )
            return False

        if self._has_contradictory_attribute(existing_lower, new_lower, "hair"):
            logger.error(
                "NarrativeEnrichmentNode: Contradictory hair color in physical description",
                existing=existing_description,
                new=new_description,
            )
            return False

        if self._has_contradictory_attribute(existing_lower, new_lower, "eye"):
            logger.error(
                "NarrativeEnrichmentNode: Contradictory eye color in physical description",
                existing=existing_description,
                new=new_description,
            )
            return False

        return True

    @staticmethod
    def _extract_words(text: str) -> list[str]:
        """Extract words using word boundaries, handling punctuation and hyphens."""
        import re

        return re.findall(r"\b[a-z]+\b", text)

    @staticmethod
    def _has_contradictory_height(existing: str, new: str) -> bool:
        """Check for contradictory height descriptors between two descriptions."""
        height_keywords = {
            "short",
            "tall",
            "petite",
            "lanky",
            "statuesque",
            "average",
            "medium",
            "small",
            "large",
            "big",
            "little",
            "tiny",
            "giant",
            "huge",
        }
        existing_heights = {w for w in NarrativeEnrichmentNode._extract_words(existing) if w in height_keywords}
        new_heights = {w for w in NarrativeEnrichmentNode._extract_words(new) if w in height_keywords}

        if not existing_heights or not new_heights:
            return False

        opposites = [
            ({"short", "petite", "small", "little", "tiny"}, {"tall", "lanky", "statuesque", "giant", "huge", "large", "big"}),
        ]
        for group_a, group_b in opposites:
            if (existing_heights & group_a and new_heights & group_b) or (existing_heights & group_b and new_heights & group_a):
                return True
        return False

    @staticmethod
    def _extract_color_near_context(text: str, context_words: set[str], color_keywords: set[str], compound_phrases: list[str]) -> set[str]:
        """Extract color descriptors that appear near context words (e.g., 'hair', 'eyes').

        Handles compound phrases like 'salt-and-pepper' or 'dirty blonde' as single units.
        Splits text into sentences first to prevent cross-sentence false positives
        (e.g., 'brown jacket' in one sentence matching 'hair' in another).
        Within each sentence, only extracts colors within a 4-word window of a context word.
        """
        import re

        normalized = re.sub(r"[–—]", "-", text)

        found_colors: set[str] = set()

        for phrase in compound_phrases:
            if phrase in normalized:
                found_colors.add(phrase)
                normalized = normalized.replace(phrase, "")

        sentences = re.split(r"(?<=[.!?;])\s+", normalized)

        for sentence in sentences:
            words = NarrativeEnrichmentNode._extract_words(sentence)

            context_indices = [i for i, w in enumerate(words) if w in context_words]
            if not context_indices:
                continue

            for index in context_indices:
                window_start = max(0, index - 4)
                window_end = min(len(words), index + 5)
                for i in range(window_start, window_end):
                    if words[i] in color_keywords:
                        found_colors.add(words[i])

        return found_colors

    @staticmethod
    def _has_contradictory_attribute(existing: str, new: str, attribute: str) -> bool:
        """Check for contradictory color attributes between two descriptions.

        Args:
            existing: Lowercased existing description
            new: Lowercased new description
            attribute: 'hair' or 'eye'
        """
        if attribute == "hair":
            context_words = {"hair", "haired", "locks", "tresses", "mane", "curls"}
            color_keywords = {
                "blonde",
                "blond",
                "brown",
                "black",
                "red",
                "auburn",
                "gray",
                "grey",
                "white",
                "silver",
                "platinum",
                "strawberry",
                "sandy",
                "chestnut",
                "raven",
                "ebony",
                "golden",
                "copper",
                "bronze",
            }
            compound_phrases = [
                "salt-and-pepper",
                "salt and pepper",
                "dirty blonde",
                "dirty blond",
                "dishwater blonde",
                "dishwater blond",
                "strawberry blonde",
                "strawberry blond",
                "ash blonde",
                "ash blond",
                "platinum blonde",
                "platinum blond",
            ]
        else:
            context_words = {"eye", "eyes", "eyed", "gaze", "irises"}
            color_keywords = {
                "blue",
                "green",
                "brown",
                "hazel",
                "gray",
                "grey",
                "amber",
                "violet",
                "purple",
                "black",
                "gold",
                "silver",
            }
            compound_phrases = [
                "steel grey",
                "steel gray",
                "ice blue",
                "dark brown",
            ]

        existing_colors = NarrativeEnrichmentNode._extract_color_near_context(
            existing,
            context_words,
            color_keywords,
            compound_phrases,
        )
        new_colors = NarrativeEnrichmentNode._extract_color_near_context(
            new,
            context_words,
            color_keywords,
            compound_phrases,
        )

        if not existing_colors or not new_colors:
            return False

        return not existing_colors & new_colors

    def _validate_embedding(
        self,
        existing_embedding: list[float],
        new_embedding: list[float],
        tolerance: float = 0.1,
    ) -> bool:
        """Validate that new embedding is not significantly different from existing one.

        Args:
            existing_embedding: Existing embedding vector
            new_embedding: New embedding vector to validate
            tolerance: Maximum allowed cosine distance between embeddings

        Returns:
            bool: True if valid (embeddings are similar), False if significantly different

        Notes:
            Uses cosine similarity to compare embeddings. Embeddings are considered
            "significantly different" if their cosine similarity is below (1 - tolerance).
        """
        import numpy as np

        if len(existing_embedding) != len(new_embedding):
            logger.error(
                "_validate_embedding: embedding dimension mismatch",
                existing_dim=len(existing_embedding),
                new_dim=len(new_embedding),
            )
            return False

        # Convert to numpy arrays for efficient computation
        existing_array = np.array(existing_embedding, dtype=np.float64)
        new_array = np.array(new_embedding, dtype=np.float64)

        # Compute cosine similarity
        existing_norm = np.linalg.norm(existing_array)
        new_norm = np.linalg.norm(new_array)

        if existing_norm == 0 or new_norm == 0:
            logger.warning(
                "_validate_embedding: zero-norm embedding detected",
                existing_norm=existing_norm,
                new_norm=new_norm,
            )
            return False

        cosine_similarity = np.dot(existing_array, new_array) / (existing_norm * new_norm)

        # Convert to cosine distance
        cosine_distance = 1.0 - cosine_similarity

        is_similar = cosine_distance < tolerance

        if not is_similar:
            logger.warning(
                "_validate_embedding: embeddings significantly different",
                cosine_distance=cosine_distance,
                tolerance=tolerance,
            )

        return is_similar


async def enrich_narrative(state: NarrativeState) -> dict[str, Any]:
    """Run narrative enrichment for the current chapter draft.

    Args:
        state: Workflow state.

    Returns:
        Partial state update with:
        - current_node: "narrative_enrichment"
        - last_error: Optional error string when enrichment fails
    """
    chapter_number = state.get("current_chapter", 1)
    if not isinstance(chapter_number, int) or isinstance(chapter_number, bool) or chapter_number <= 0:
        raise ValueError("narrative_enrichment expected current_chapter to be a positive int")

    if not state.get("draft_ref"):
        return {
            "current_node": "narrative_enrichment",
            "last_error": "Narrative enrichment skipped: draft_ref is missing",
        }

    project_dir = require_project_dir(state)
    content_manager = ContentManager(project_dir)
    draft_text = get_draft_text(state, content_manager)

    node = NarrativeEnrichmentNode()
    try:
        await node.process(draft_text, chapter_number)
    except Exception as error:
        logger.error("enrich_narrative: enrichment failed", error=str(error))
        return {
            "current_node": "narrative_enrichment",
            "last_error": str(error),
        }

    return {
        "current_node": "narrative_enrichment",
        "last_error": None,
    }


__all__ = ["NarrativeEnrichmentNode", "enrich_narrative"]
