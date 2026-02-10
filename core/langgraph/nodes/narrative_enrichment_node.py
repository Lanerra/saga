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

    def __init__(self):
        """Initialize the NarrativeEnrichmentNode."""
        pass

    async def process(
        self,
        narrative_text: str,
        chapter_number: int,
    ) -> str:
        """Process narrative text and extract enrichment data.

        Args:
            narrative_text: The narrative text to parse
            chapter_number: Chapter number for provenance

        Returns:
            str: Success or failure message

        Raises:
            ValueError: If narrative_text is empty or chapter_number is invalid
            DatabaseError: If database operations fail
        """
        # Validate inputs
        if not narrative_text or len(narrative_text.strip()) == 0:
            logger.error("NarrativeEnrichmentNode: Empty narrative text provided")
            return "Failed to enrich narrative: Empty narrative text provided"

        if chapter_number <= 0:
            logger.error("NarrativeEnrichmentNode: Invalid chapter number", chapter_number=chapter_number)
            return f"Failed to enrich narrative: Invalid chapter number {chapter_number}"

        try:
            # Get character profiles from database
            character_profiles = await get_character_profiles()
            if not character_profiles:
                logger.error("NarrativeEnrichmentNode: No character profiles found")
                return "Failed to enrich narrative: No character profiles found"

            # Get chapter data from database
            chapter_data = await get_chapter_data_from_db(chapter_number)
            if not chapter_data:
                logger.error("NarrativeEnrichmentNode: No chapter data found", chapter_number=chapter_number)
                return f"Failed to enrich narrative: No chapter data found for chapter {chapter_number}"

            # Initialize parser
            parser = NarrativeEnrichmentParser(
                narrative_text=narrative_text,
                chapter_number=chapter_number,
            )

            # Extract physical descriptions
            physical_descriptions = await parser.extract_physical_descriptions()
            if not physical_descriptions:
                logger.warning("NarrativeEnrichmentNode: No physical descriptions extracted")

            # Extract chapter embeddings
            chapter_embeddings = await parser.extract_chapter_embeddings()
            if not chapter_embeddings:
                logger.warning("NarrativeEnrichmentNode: No chapter embeddings extracted")

            # Update character physical descriptions
            if physical_descriptions:
                for desc in physical_descriptions:
                    character_name = desc.character_name
                    extracted_description = desc.extracted_description

                    # Find character by name
                    character = next((c for c in character_profiles if c.name == character_name), None)

                    if not character:
                        logger.error("NarrativeEnrichmentNode: Character not found", character_name=character_name)
                        return f"Failed to enrich narrative: Character {character_name} not found"

                    # Check if physical description already exists
                    if character.physical_description:
                        # Validate that new description doesn't contradict existing one
                        if self._validate_physical_description(
                            character.physical_description,
                            extracted_description,
                        ):
                            # Update character with new physical description
                            character.physical_description = extracted_description
                            await sync_characters([character])
                            logger.info(
                                "NarrativeEnrichmentNode: Updated character physical description",
                                character_name=character_name,
                            )
                        else:
                            logger.error(
                                "NarrativeEnrichmentNode: Contradictory physical description",
                                character_name=character_name,
                            )
                            return f"Failed to enrich narrative: Contradictory physical description for {character_name}"
                    else:
                        # Add physical description to character
                        character.physical_description = extracted_description
                        await sync_characters([character])
                        logger.info(
                            "NarrativeEnrichmentNode: Added character physical description",
                            character_name=character_name,
                        )

            # Update chapter embeddings
            if chapter_embeddings:
                for embedding in chapter_embeddings:
                    embedding_vector = embedding.embedding_vector

                    # Check if embedding already exists
                    if chapter_data.embedding:
                        # Validate that new embedding is not significantly different
                        if self._validate_embedding(
                            chapter_data.embedding,
                            embedding_vector,
                        ):
                            # Update chapter with new embedding
                            await save_chapter_data_to_db(
                                chapter_number=chapter_data.number,
                                title=chapter_data.title,
                                act_number=chapter_data.act_number,
                                summary=chapter_data.summary,
                                embedding_array=embedding_vector,
                                is_provisional=chapter_data.is_provisional,
                            )
                            logger.info(
                                "NarrativeEnrichmentNode: Updated chapter embedding",
                                chapter_number=chapter_data.number,
                            )
                        else:
                            logger.error(
                                "NarrativeEnrichmentNode: Invalid embedding",
                                chapter_number=chapter_data.number,
                            )
                            return f"Failed to enrich narrative: Invalid embedding for chapter {chapter_data.number}"
                    else:
                        # Add embedding to chapter
                        await save_chapter_data_to_db(
                            chapter_number=chapter_data.number,
                            title=chapter_data.title,
                            act_number=chapter_data.act_number,
                            summary=chapter_data.summary,
                            embedding_array=embedding_vector,
                            is_provisional=chapter_data.is_provisional,
                        )
                        logger.info(
                            "NarrativeEnrichmentNode: Added chapter embedding",
                            chapter_number=chapter_data.number,
                        )

            return "Successfully enriched narrative"

        except Exception as e:
            logger.error("NarrativeEnrichmentNode: Error during enrichment", error=str(e))
            return f"Failed to enrich narrative: {str(e)}"

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
        # Normalize descriptions to lowercase for comparison
        existing_lower = existing_description.lower()
        new_lower = new_description.lower()

        # Check for contradictory height descriptors
        height_keywords = {"short", "tall", "petite", "lanky", "statuesque", "average", "medium", "small", "large", "big", "little", "tiny", "giant", "huge"}

        # Extract height-related words from both descriptions
        existing_heights = [word for word in existing_lower.split() if word in height_keywords]
        new_heights = [word for word in new_lower.split() if word in height_keywords]

        # If both have height descriptors, check if they contradict
        if existing_heights and new_heights:
            # Simple opposition check - this can be enhanced with more sophisticated logic
            if ("short" in existing_heights and "tall" in new_heights) or ("tall" in existing_heights and "short" in new_heights):
                logger.error(
                    "NarrativeEnrichmentNode: Contradictory height in physical description",
                    existing=existing_description,
                    new=new_description,
                )
                return False

        # Check for contradictory hair color descriptors
        hair_color_keywords = {
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
            "dirty",
            "dishwater",
            "sandy",
            "carrot",
            "chestnut",
            "raven",
            "ebony",
            "golden",
            "copper",
            "bronze",
            "ash",
            "pepper",
            "salt",
        }

        # Extract hair color-related words from both descriptions
        existing_hair_colors = [word for word in existing_lower.split() if word in hair_color_keywords]
        new_hair_colors = [word for word in new_lower.split() if word in hair_color_keywords]

        # If both have hair color descriptors, check if they contradict
        # Only check if both descriptions mention hair color
        if existing_hair_colors and new_hair_colors:
            # Check if the hair colors are different
            # We need to compare the actual color words, not just check if they're different
            # For now, we'll only flag if they're explicitly contradictory (e.g., brown vs blonde)
            # This is a simplified check that can be enhanced later
            if existing_hair_colors and new_hair_colors:
                # Only flag if they're explicitly different and not just additional info
                # For example, "brown" and "long brown" should not be flagged
                # But "brown" and "blonde" should be flagged
                if existing_hair_colors != new_hair_colors:
                    # Check if the descriptions are actually contradictory
                    # If one is a subset of the other (e.g., "brown" vs "long brown"), it's not a contradiction
                    # We'll use a simple heuristic: if the color words are different, it's a contradiction
                    # unless they're the same color described differently (e.g., "brown" vs "dark brown")
                    # For now, we'll only flag if they're completely different
                    if not any(color in new_hair_colors for color in existing_hair_colors):
                        logger.error(
                            "NarrativeEnrichmentNode: Contradictory hair color in physical description",
                            existing=existing_description,
                            new=new_description,
                        )
                        return False

        # Check for contradictory eye color descriptors
        eye_color_keywords = {"blue", "green", "brown", "hazel", "gray", "grey", "amber", "violet", "purple", "pink", "red", "black", "white", "gold", "silver"}

        # Extract eye color-related words from both descriptions
        existing_eye_colors = [word for word in existing_lower.split() if word in eye_color_keywords]
        new_eye_colors = [word for word in new_lower.split() if word in eye_color_keywords]

        # If both have eye color descriptors, check if they contradict
        # Only check if both descriptions mention eye color
        if existing_eye_colors and new_eye_colors:
            # Check if the eye colors are different
            # We need to compare the actual color words, not just check if they're different
            # For now, we'll only flag if they're explicitly contradictory (e.g., brown vs blue)
            # This is a simplified check that can be enhanced later
            if existing_eye_colors and new_eye_colors:
                # Only flag if they're explicitly different and not just additional info
                # For example, "brown" and "long brown" should not be flagged
                # But "brown" and "blonde" should be flagged
                if existing_eye_colors != new_eye_colors:
                    # Check if the descriptions are actually contradictory
                    # If one is a subset of the other (e.g., "brown" vs "dark brown"), it's not a contradiction
                    # We'll use a simple heuristic: if the color words are different, it's a contradiction
                    # unless they're the same color described differently (e.g., "brown" vs "dark brown")
                    # For now, we'll only flag if they're completely different
                    if not any(color in new_eye_colors for color in existing_eye_colors):
                        logger.error(
                            "NarrativeEnrichmentNode: Contradictory eye color in physical description",
                            existing=existing_description,
                            new=new_description,
                        )
                        return False

        # If no contradictions found, return True
        return True

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
    result = await node.process(draft_text, chapter_number)

    if result.startswith("Failed"):
        return {
            "current_node": "narrative_enrichment",
            "last_error": result,
        }

    return {
        "current_node": "narrative_enrichment",
        "last_error": None,
    }


__all__ = ["NarrativeEnrichmentNode", "enrich_narrative"]
