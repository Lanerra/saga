"""
Relationship normalization service for SAGA.

This service maintains a self-organizing vocabulary of relationship types,
normalizing semantically similar relationships while allowing genuinely
novel relationships to emerge.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import structlog

import config
from core.llm_interface_refactored import llm_service
from utils.similarity import numpy_cosine_similarity

logger = structlog.get_logger(__name__)


class RelationshipNormalizationService:
    """Service for normalizing relationship types against accumulated vocabulary."""

    def __init__(self):
        """Initialize the normalization service."""
        self.embedding_cache: dict[str, np.ndarray] = {}

    async def normalize_relationship_type(
        self,
        rel_type: str,
        rel_description: str,
        vocabulary: dict[str, Any],
        current_chapter: int,
    ) -> tuple[str, bool, float]:
        """
        Normalize a relationship type against existing vocabulary.

        Args:
            rel_type: The extracted relationship type
            rel_description: Context description of the relationship
            vocabulary: Current relationship vocabulary (canonical_type -> RelationshipUsage dict)
            current_chapter: Current chapter number

        Returns:
            Tuple of (normalized_type, was_normalized, similarity_score)
        """
        # Early exit if normalization disabled
        if not config.ENABLE_RELATIONSHIP_NORMALIZATION:
            return rel_type, False, 0.0

        # Normalize case and punctuation if configured
        canonical_form = self._canonicalize(rel_type)

        # Check for exact match (after canonicalization)
        if canonical_form in vocabulary:
            logger.debug(
                "Exact vocabulary match",
                original=rel_type,
                canonical=canonical_form,
            )
            return canonical_form, rel_type != canonical_form, 1.0

        # Check for case/punctuation-only variants
        for vocab_type in vocabulary.keys():
            if self._canonicalize(vocab_type) == canonical_form:
                logger.info(
                    "Normalized case/punctuation variant",
                    original=rel_type,
                    normalized_to=vocab_type,
                )
                return vocab_type, True, 1.0

        # No exact match - check semantic similarity
        if not vocabulary:
            # First relationship in narrative - nothing to compare against
            logger.info(
                "First relationship type in vocabulary",
                type=canonical_form,
                chapter=current_chapter,
            )
            return canonical_form, False, 0.0

        # Find most similar existing relationship
        best_match, best_similarity = await self._find_most_similar(
            canonical_form, list(vocabulary.keys())
        )

        threshold = config.REL_NORM_SIMILARITY_THRESHOLD

        if best_similarity >= threshold:
            # Normalize to existing type
            logger.info(
                "Normalizing relationship to existing type",
                original=rel_type,
                normalized_to=best_match,
                similarity=f"{best_similarity:.3f}",
                chapter=current_chapter,
            )
            return best_match, True, best_similarity

        # Check if in ambiguous range (for optional LLM disambiguation)
        ambiguous_min = config.REL_NORM_SIMILARITY_THRESHOLD_AMBIGUOUS_MIN

        if (
            ambiguous_min <= best_similarity < threshold
            and config.REL_NORM_USE_LLM_DISAMBIGUATION
        ):
            # Use LLM to disambiguate
            should_normalize = await self._llm_disambiguate(
                rel_type,
                rel_description,
                best_match,
                vocabulary[best_match],
            )

            if should_normalize:
                logger.info(
                    "LLM-approved normalization",
                    original=rel_type,
                    normalized_to=best_match,
                    similarity=f"{best_similarity:.3f}",
                )
                return best_match, True, best_similarity

        # Novel relationship - not similar enough to existing types
        logger.info(
            "Adding novel relationship type to vocabulary",
            type=canonical_form,
            closest_match=best_match,
            similarity=f"{best_similarity:.3f}",
            chapter=current_chapter,
        )
        return canonical_form, False, best_similarity

    def _canonicalize(self, rel_type: str) -> str:
        """
        Canonicalize relationship type for comparison.

        Handles case normalization and punctuation normalization based on config.
        """
        canonical = rel_type

        # Case normalization
        if config.REL_NORM_NORMALIZE_CASE_VARIANTS:
            canonical = canonical.upper()

        # Punctuation normalization
        if config.REL_NORM_NORMALIZE_PUNCTUATION_VARIANTS:
            # Replace hyphens, underscores, spaces with underscore
            canonical = re.sub(r"[-\s]+", "_", canonical)
            # Remove other punctuation
            canonical = re.sub(r"[^\w_]", "", canonical)

        return canonical

    async def _find_most_similar(
        self, rel_type: str, vocabulary_types: list[str]
    ) -> tuple[str, float]:
        """
        Find most semantically similar relationship type in vocabulary.

        Returns:
            Tuple of (most_similar_type, similarity_score)
        """
        # Get embedding for new type
        if rel_type not in self.embedding_cache:
            embedding = await self._get_embedding(rel_type)
            if embedding is not None:
                self.embedding_cache[rel_type] = embedding

        new_embedding = self.embedding_cache.get(rel_type)
        if new_embedding is None:
            logger.warning(
                "Failed to get embedding for new relationship type",
                type=rel_type,
            )
            return "", 0.0

        best_match = None
        best_similarity = 0.0

        for vocab_type in vocabulary_types:
            # Get cached or compute embedding
            if vocab_type not in self.embedding_cache:
                embedding = await self._get_embedding(vocab_type)
                if embedding is not None:
                    self.embedding_cache[vocab_type] = embedding

            vocab_embedding = self.embedding_cache.get(vocab_type)
            if vocab_embedding is None:
                logger.warning(
                    "Failed to get embedding for vocabulary type",
                    type=vocab_type,
                )
                continue

            # Compute cosine similarity
            similarity = numpy_cosine_similarity(new_embedding, vocab_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = vocab_type

        return best_match or "", best_similarity

    async def _get_embedding(self, text: str) -> np.ndarray | None:
        """Get embedding for text using embedding service."""
        try:
            embedding = await llm_service.async_get_embedding(text)
            return embedding
        except Exception as e:
            logger.error(
                "Failed to get embedding for relationship type",
                text=text,
                error=str(e),
            )
            return None

    async def _llm_disambiguate(
        self,
        new_type: str,
        new_description: str,
        existing_type: str,
        existing_usage: dict[str, Any],
    ) -> bool:
        """
        Use LLM to determine if relationships are semantically equivalent.

        Args:
            new_type: New relationship type
            new_description: Description of new relationship
            existing_type: Existing vocabulary type
            existing_usage: Usage data for existing type

        Returns:
            True if should normalize to existing type
        """
        # Get examples of existing usage
        examples = existing_usage.get("example_descriptions", [])[:3]
        examples_str = "\n".join(f"  - {ex}" for ex in examples)

        prompt = f"""You are helping maintain consistent relationship types in a narrative.

NEW RELATIONSHIP:
Type: "{new_type}"
Description: "{new_description}"

EXISTING RELATIONSHIP IN VOCABULARY:
Type: "{existing_type}"
Used {existing_usage.get('usage_count', 0)} times
Example usage:
{examples_str}

Question: Should "{new_type}" be normalized to "{existing_type}", or are they semantically distinct?

Rules:
- Normalize if they express the SAME core relationship (e.g., "WORKS_WITH" and "COLLABORATES_WITH")
- Keep distinct if they have different semantic nuances (e.g., "MENTORS" vs "TEACHES")
- Consider context from the descriptions

Answer with EXACTLY one word: NORMALIZE or DISTINCT"""

        try:
            response, _ = await llm_service.async_call_llm(
                model_name=config.SMALL_MODEL,
                prompt=prompt,
                max_tokens=10,
            )

            decision = response.strip().upper()

            logger.info(
                "LLM disambiguation result",
                new_type=new_type,
                existing_type=existing_type,
                decision=decision,
            )

            return "NORMALIZE" in decision

        except Exception as e:
            logger.error(
                "LLM disambiguation failed",
                error=str(e),
                exc_info=True,
            )
            # Conservative fallback - treat as distinct
            return False

    def update_vocabulary_usage(
        self,
        vocabulary: dict[str, Any],
        rel_type: str,
        rel_description: str,
        current_chapter: int,
        was_normalized: bool,
        original_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Update vocabulary with usage statistics.

        Args:
            vocabulary: Current vocabulary dict
            rel_type: The (possibly normalized) relationship type
            rel_description: Description of the relationship
            current_chapter: Current chapter number
            was_normalized: Whether this was normalized from another type
            original_type: Original type if normalized

        Returns:
            Updated vocabulary dict
        """
        max_examples = config.REL_NORM_MAX_EXAMPLES_PER_RELATIONSHIP

        if rel_type in vocabulary:
            # Update existing entry
            usage = vocabulary[rel_type]
            usage["usage_count"] += 1
            usage["last_used_chapter"] = current_chapter

            # Add example if under limit
            if len(usage.get("example_descriptions", [])) < max_examples:
                if rel_description not in usage["example_descriptions"]:
                    usage["example_descriptions"].append(rel_description)

            # Track synonym if normalized
            if was_normalized and original_type:
                if original_type not in usage.get("synonyms", []):
                    usage["synonyms"].append(original_type)

        else:
            # New vocabulary entry
            vocabulary[rel_type] = {
                "canonical_type": rel_type,
                "first_used_chapter": current_chapter,
                "usage_count": 1,
                "example_descriptions": [rel_description] if rel_description else [],
                "embedding": None,  # Will be computed on next comparison
                "synonyms": [original_type]
                if (was_normalized and original_type)
                else [],
                "last_used_chapter": current_chapter,
            }

        return vocabulary

    def prune_vocabulary(
        self, vocabulary: dict[str, Any], current_chapter: int
    ) -> dict[str, Any]:
        """
        Prune rarely-used relationships from vocabulary.

        Removes single-use relationships that haven't been used recently.
        """
        prune_after = config.REL_NORM_PRUNE_SINGLE_USE_AFTER_CHAPTERS
        max_size = config.REL_NORM_MAX_VOCABULARY_SIZE

        pruned = {}
        pruned_count = 0

        for rel_type, usage in vocabulary.items():
            # Keep if used multiple times
            if usage["usage_count"] > 1:
                pruned[rel_type] = usage
                continue

            # Keep if used recently
            last_used = usage.get("last_used_chapter", 0)
            if current_chapter - last_used < prune_after:
                pruned[rel_type] = usage
                continue

            # Otherwise prune
            pruned_count += 1
            logger.debug(
                "Pruning rarely-used relationship",
                type=rel_type,
                usage_count=usage["usage_count"],
                last_used_chapter=last_used,
            )

        # Additional size-based pruning if still over limit
        if len(pruned) > max_size:
            # Sort by usage count and keep top N
            sorted_items = sorted(
                pruned.items(),
                key=lambda x: (x[1]["usage_count"], x[1]["last_used_chapter"]),
                reverse=True,
            )
            pruned = dict(sorted_items[:max_size])
            logger.warning(
                "Vocabulary size limit exceeded, pruned additional entries",
                original_size=len(vocabulary),
                pruned_size=len(pruned),
            )

        if pruned_count > 0:
            logger.info(
                "Vocabulary pruning complete",
                removed=pruned_count,
                remaining=len(pruned),
                chapter=current_chapter,
            )

        return pruned


# Singleton instance
normalization_service = RelationshipNormalizationService()


__all__ = ["RelationshipNormalizationService", "normalization_service"]
