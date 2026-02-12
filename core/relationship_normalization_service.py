# core/relationship_normalization_service.py
"""Normalize extracted relationship types against an evolving vocabulary.

This module maintains a vocabulary of relationship types and provides helpers to:
- Canonicalize case/punctuation variants.
- Normalize semantically similar relationship types when similarity is high.
- Optionally use an LLM to disambiguate borderline similarity matches.
- Record usage statistics and prune rarely-used types.

Notes:
    Normalization is best-effort. When embeddings or LLM calls fail, the implementation
    prefers returning the input relationship type unchanged rather than raising.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Any

import numpy as np
import structlog

import config
from core.llm_interface_refactored import llm_service
from prompts.prompt_renderer import render_prompt
from utils.similarity import numpy_cosine_similarity

logger = structlog.get_logger(__name__)


class RelationshipNormalizationService:
    """Normalize relationship types against an accumulated vocabulary.

    The primary entrypoint is [`core.relationship_normalization_service.RelationshipNormalizationService.normalize_relationship_type()`](core/relationship_normalization_service.py:33).
    """

    EMBEDDING_CACHE_MAX_SIZE = 1024

    def __init__(self) -> None:
        """Initialize the service and in-memory embedding cache."""
        self.embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.canonical_embeddings: dict[str, np.ndarray] = {}
        self.rejected_cache: set[str] = set()

    def _cache_embedding(self, key: str, value: np.ndarray) -> None:
        """Store an embedding in the LRU-bounded cache."""
        if key in self.embedding_cache:
            self.embedding_cache.move_to_end(key)
        else:
            if len(self.embedding_cache) >= self.EMBEDDING_CACHE_MAX_SIZE:
                self.embedding_cache.popitem(last=False)
            self.embedding_cache[key] = value

    async def map_to_canonical(self, rel_type: str, category_hint: str = "DEFAULT") -> tuple[str | None, bool, float, bool]:
        """Map a relationship type to its canonical form using strict enforcement.

        Args:
            rel_type: Extracted relationship type.
            category_hint: Category hint for threshold selection (e.g., "CHARACTER_CHARACTER").

        Returns:
            Tuple of `(canonical_type, was_normalized, similarity_score, is_property)`.

            - `canonical_type`: The canonical relationship type, or None if rejected.
            - `was_normalized`: True if the input was normalized to a different type.
            - `similarity_score`: Cosine similarity when semantic matching was used, 1.0 for exact matches.
            - `is_property`: True if the relationship should be a node property instead.

        Notes:
            This method enforces strict canonicalization and rejects unknown types
            when STRICT_CANONICAL_MODE is enabled.
        """
        from models.kg_constants import PROPERTY_RELATIONSHIPS, RELATIONSHIP_TYPES, STATIC_RELATIONSHIP_MAP

        # 0. Check rejection cache
        if rel_type in self.rejected_cache:
            return None, False, 0.0, False

        # 1. Canonicalize input (uppercase, underscores)
        canonical_input = self._canonicalize(rel_type)

        # 2. Check Property List
        if canonical_input in PROPERTY_RELATIONSHIPS:
            return None, False, 0.0, True  # is_property=True

        # 3. Exact Match
        if canonical_input in RELATIONSHIP_TYPES:
            return canonical_input, (canonical_input != rel_type), 1.0, False

        # 4. Static Map
        if config.REL_NORM_STATIC_OVERRIDES_ENABLED and canonical_input in STATIC_RELATIONSHIP_MAP:
            mapped = STATIC_RELATIONSHIP_MAP[canonical_input]
            return mapped, True, 1.0, False

        # 5. Semantic Match (only if not in strict mode)
        if config.REL_NORM_STRICT_CANONICAL_MODE:
            self.rejected_cache.add(rel_type)
            return None, False, 0.0, False

        await self._ensure_canonical_embeddings()
        incoming_embedding = await self._get_embedding(canonical_input)

        if incoming_embedding is None:
            self.rejected_cache.add(rel_type)
            return None, False, 0.0, False

        best_match = None
        best_similarity = 0.0

        for canonical_type, canonical_emb in self.canonical_embeddings.items():
            sim = numpy_cosine_similarity(incoming_embedding, canonical_emb)
            if sim > best_similarity:
                best_similarity = sim
                best_match = canonical_type

        threshold = self._get_threshold(category_hint)

        if best_similarity > threshold:
            return best_match, True, best_similarity, False
        else:
            self.rejected_cache.add(rel_type)
            return None, False, best_similarity, False

    async def normalize_relationship_type(
        self,
        rel_type: str,
        rel_description: str,
        vocabulary: dict[str, Any],
        current_chapter: int,
    ) -> tuple[str, bool, float]:
        """Normalize a relationship type against an existing vocabulary.

        Args:
            rel_type: Extracted relationship type.
            rel_description: Brief natural-language description for diagnostics and optional
                LLM disambiguation.
            vocabulary: Relationship vocabulary keyed by canonical type. Values are usage
                dicts with counters and example descriptions.
            current_chapter: Chapter number used for usage tracking and pruning windows.

        Returns:
            Tuple of `(normalized_type, was_normalized, similarity_score)`.

            `similarity_score` is the best-match cosine similarity when embeddings are
            available, or 0.0 when no comparison could be performed.

        Notes:
            - If normalization is disabled, this returns `(rel_type, False, 0.0)`.
            - When STRICT_CANONICAL_MODE is enabled, uses `map_to_canonical` for strict enforcement.
            - Case/punctuation variants can be normalized even when semantic similarity is
              not computed.
            - If the best similarity falls in an ambiguous range and LLM disambiguation is
              enabled, an LLM decision can override the threshold outcome.
        """
        # Early exit if normalization disabled
        if not config.ENABLE_RELATIONSHIP_NORMALIZATION:
            return rel_type, False, 0.0

        # Use strict canonical mode if enabled
        if config.REL_NORM_STRICT_CANONICAL_MODE:
            canonical_type, was_normalized, similarity, is_property = await self.map_to_canonical(rel_type)

            if canonical_type is None:
                if is_property:
                    logger.info(
                        "Relationship should be a node property, not an edge",
                        original=rel_type,
                    )
                else:
                    logger.warning(
                        "Rejected relationship type in strict canonical mode",
                        original=rel_type,
                        similarity=f"{similarity:.3f}" if similarity > 0 else "N/A",
                    )
                # Return original type but mark as normalized to False
                return rel_type, False, 0.0

            logger.info(
                "Strict canonical normalization",
                original=rel_type,
                normalized_to=canonical_type,
                was_normalized=was_normalized,
                similarity=f"{similarity:.3f}" if similarity < 1.0 else "exact",
            )
            return canonical_type, was_normalized, similarity

        # Legacy behavior for backward compatibility
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
        best_match, best_similarity = await self._find_most_similar(canonical_form, list(vocabulary.keys()))

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

        if ambiguous_min <= best_similarity < threshold and config.REL_NORM_USE_LLM_DISAMBIGUATION:
            # Use LLM to disambiguate
            should_normalize = await self._llm_disambiguate(
                rel_type,
                rel_description,
                best_match,
                vocabulary[best_match],
                use_json_mode=config.REL_NORM_LLM_DISAMBIGUATION_JSON_MODE,
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
        """Canonicalize a relationship type for comparison.

        Notes:
            Canonicalization is controlled by configuration flags and may:
            - Uppercase the type.
            - Replace hyphens/spaces with underscores.
            - Strip other punctuation.
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

    async def _ensure_canonical_embeddings(self) -> None:
        """Lazily compute and cache embeddings for all canonical relationship types."""
        if self.canonical_embeddings:
            return

        from models.kg_constants import RELATIONSHIP_TYPES

        for rel_type in RELATIONSHIP_TYPES:
            if rel_type not in self.canonical_embeddings:
                embedding = await self._get_embedding(rel_type)
                if embedding is not None:
                    self.canonical_embeddings[rel_type] = embedding

    def _get_threshold(self, category: str) -> float:
        """Get the similarity threshold for a given category.

        Args:
            category: Category name (e.g., "CHARACTER_CHARACTER", "DEFAULT").

        Returns:
            Similarity threshold for the category, or default if not found.
        """
        thresholds = config.REL_NORM_SIMILARITY_THRESHOLDS
        return thresholds.get(category, thresholds.get("DEFAULT", 0.75))

    async def _find_most_similar(self, rel_type: str, vocabulary_types: list[str]) -> tuple[str, float]:
        """Find the most semantically similar vocabulary type for an input type.

        Args:
            rel_type: Canonicalized relationship type to compare.
            vocabulary_types: Candidate vocabulary keys to compare against.

        Returns:
            Tuple of `(most_similar_type, similarity_score)`.

            If embeddings cannot be computed, returns `("", 0.0)`.
        """
        # Get embedding for new type
        if rel_type not in self.embedding_cache:
            embedding = await self._get_embedding(rel_type)
            if embedding is not None:
                self._cache_embedding(rel_type, embedding)

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
                    self._cache_embedding(vocab_type, embedding)

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
        """Fetch an embedding for a relationship type string.

        Notes:
            This is best-effort. Failures return None and are logged.
        """
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
        *,
        use_json_mode: bool = False,
    ) -> bool:
        """Disambiguate borderline similarity matches using an LLM.

        Args:
            new_type: Extracted relationship type.
            new_description: Description for the extracted relationship instance.
            existing_type: Candidate canonical vocabulary type.
            existing_usage: Usage dict for `existing_type`, including example descriptions.
            use_json_mode: If True, require a strict JSON object decision payload.

        Returns:
            True when the LLM indicates the new type should normalize to `existing_type`.

        Raises:
            ValueError: When `use_json_mode=True` and the LLM returns a structurally invalid
                decision payload.
        """
        # Get examples of existing usage
        examples = existing_usage.get("example_descriptions", [])[:3]
        examples_str = "\n".join(f"  - {ex}" for ex in examples)
        existing_usage_count = existing_usage.get("usage_count", 0)

        prompt = render_prompt(
            "knowledge_agent/relationship_disambiguate_normalize_or_distinct.j2",
            {
                "new_type": new_type,
                "new_description": new_description,
                "existing_type": existing_type,
                "existing_usage_count": existing_usage_count,
                "examples_str": examples_str,
            },
        )

        if use_json_mode:
            data, _ = await llm_service.async_call_llm_json_object(
                model_name=config.SMALL_MODEL,
                prompt=prompt,
                max_tokens=config.MAX_GENERATION_TOKENS,
            )

            allowed_keys = {"decision"}
            actual_keys = set(data.keys())
            if actual_keys != allowed_keys:
                raise ValueError("LLM JSON decision must have exactly one key: 'decision'. " f"actual_keys={sorted(actual_keys)}")

            decision_value = data["decision"]
            if not isinstance(decision_value, str):
                raise ValueError("LLM JSON decision value must be a string")

            if decision_value == "NORMALIZE":
                decision_should_normalize = True
            elif decision_value == "DISTINCT":
                decision_should_normalize = False
            else:
                raise ValueError('LLM JSON decision must be "NORMALIZE" or "DISTINCT"')

            logger.info(
                "LLM disambiguation JSON decision",
                new_type=new_type,
                existing_type=existing_type,
                decision=decision_value,
            )
            return decision_should_normalize

        try:
            response, _ = await llm_service.async_call_llm(
                model_name=config.SMALL_MODEL,
                prompt=prompt,
                max_tokens=config.MAX_GENERATION_TOKENS,
            )

            response_sha1 = hashlib.sha1(response.encode("utf-8")).hexdigest()[:12]
            logger.info(
                "LLM disambiguation legacy response",
                new_type=new_type,
                existing_type=existing_type,
                response_sha1=response_sha1,
                response_len=len(response),
            )

            decision = response.strip().upper()
            return "NORMALIZE" in decision

        except Exception as e:
            logger.error(
                "LLM disambiguation failed",
                error=str(e),
                exc_info=True,
            )
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
        """Update vocabulary usage counters and examples for a relationship type.

        Args:
            vocabulary: Vocabulary dict to update.
            rel_type: Canonical (possibly normalized) relationship type.
            rel_description: Example description to store for this type.
            current_chapter: Chapter number used for `first_used_chapter` and `last_used_chapter`.
            was_normalized: Whether this instance was normalized from another type.
            original_type: Original type string when `was_normalized=True`.

        Returns:
            Updated vocabulary dict (the same object, mutated in place).
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
                "synonyms": [original_type] if (was_normalized and original_type) else [],
                "last_used_chapter": current_chapter,
            }

        return vocabulary

    def prune_vocabulary(self, vocabulary: dict[str, Any], current_chapter: int) -> dict[str, Any]:
        """Prune rarely-used relationships from the vocabulary.

        Args:
            vocabulary: Vocabulary dict to prune.
            current_chapter: Current chapter number used for staleness calculations.

        Returns:
            Pruned vocabulary dict.

        Notes:
            This removes single-use relationships that have not been used recently and
            enforces an overall max vocabulary size by keeping the most-used entries.
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
