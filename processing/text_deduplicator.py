# processing/text_deduplicator.py
"""Utilities for detecting and removing duplicate text segments."""

from __future__ import annotations

import asyncio
import hashlib
import re
from typing import cast

import numpy as np
import structlog

import config
import utils
from core.llm_interface_refactored import llm_service

logger = structlog.get_logger(__name__)


class TextDeduplicator:
    """Detects duplicate text segments using hashing and embeddings."""

    def __init__(
        self,
        similarity_threshold: float = config.DEDUPLICATION_SEMANTIC_THRESHOLD,
        use_semantic_comparison: bool = config.DEDUPLICATION_USE_SEMANTIC,
        min_segment_length_chars: int = config.DEDUPLICATION_MIN_SEGMENT_LENGTH,
        prefer_newer: bool = False,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.use_semantic_comparison = use_semantic_comparison
        self.min_segment_length_chars = min_segment_length_chars
        self.prefer_newer = prefer_newer

    async def deduplicate(self, original_text: str, segment_level: str = "paragraph") -> tuple[str, int]:
        if not original_text.strip():
            return original_text, 0

        segments = utils.get_text_segments(original_text, segment_level)
        if not segments:
            return original_text, 0

        normalized_cache: list[str] = [utils._normalize_text_for_matching(seg[0]) for seg in segments]
        indices_to_remove: set[int] = set()
        fingerprint_map: dict[str, int] = {}
        iteration_range = range(len(segments) - 1, -1, -1) if self.prefer_newer else range(len(segments))

        for idx in iteration_range:
            if idx in indices_to_remove:
                continue
            seg_text, _, _ = segments[idx]
            if len(seg_text) < self.min_segment_length_chars:
                continue
            norm = normalized_cache[idx]
            fingerprint = hashlib.md5(norm.encode()).hexdigest()
            if fingerprint in fingerprint_map:
                other_idx = fingerprint_map[fingerprint]
                remove_idx = idx if not self.prefer_newer else other_idx
                indices_to_remove.add(remove_idx)
                if self.prefer_newer:
                    fingerprint_map[fingerprint] = idx
                continue
            fingerprint_map[fingerprint] = idx

        embeddings: list[np.ndarray | None] = [None] * len(segments)
        if self.use_semantic_comparison:
            unique_indices = [i for i in iteration_range if i not in indices_to_remove]

            # Batch embedding generation with configurable concurrency
            batch_size = min(config.MAX_CONCURRENT_LLM_CALLS, len(unique_indices))

            # Process embeddings in batches to control memory usage and API load
            for i in range(0, len(unique_indices), batch_size):
                batch_indices = unique_indices[i : i + batch_size]
                batch_tasks = [llm_service.async_get_embedding(segments[idx][0]) for idx in batch_indices]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for batch_idx, result in zip(batch_indices, batch_results, strict=False):
                    if not isinstance(result, Exception):
                        embeddings[batch_idx] = cast(np.ndarray | None, result)

            keepers: list[int] = []
            for idx in iteration_range:
                if idx in indices_to_remove:
                    continue
                if embeddings[idx] is None:
                    keepers.append(idx)
                    continue
                is_dup = False
                for kept_idx in keepers:
                    emb_j = embeddings[kept_idx]
                    if emb_j is None:
                        continue
                    try:
                        similarity = utils.numpy_cosine_similarity(embeddings[idx], emb_j)
                    except ValueError:
                        logger.warning("Cosine similarity shape mismatch handled: setting to 0.0 for deduplication compatibility.")
                        similarity = 0.0
                    if similarity > self.similarity_threshold:
                        remove_idx = idx if not self.prefer_newer else kept_idx
                        indices_to_remove.add(remove_idx)
                        if self.prefer_newer and remove_idx == kept_idx:
                            keepers.remove(kept_idx)
                            keepers.append(idx)
                        is_dup = True
                        break
                if not is_dup:
                    keepers.append(idx)

        if not indices_to_remove:
            return original_text, 0

        # Optimize text reconstruction using sorted spans
        spans_to_remove = [segments[i][1:] for i in sorted(indices_to_remove)]
        spans_to_remove.sort(key=lambda x: x[0])

        # Pre-allocate buffer for better performance on large texts
        text_parts = []
        last_pos = 0

        for start, end in spans_to_remove:
            if start > last_pos:
                text_parts.append(original_text[last_pos:start])
            last_pos = max(last_pos, end)

        if last_pos < len(original_text):
            text_parts.append(original_text[last_pos:])

        # Single join operation instead of multiple string operations
        dedup_text = "".join(text_parts)

        # Optimize regex operations - combine patterns
        dedup_text = re.sub(r"\n\s*\n(?:\s*\n)+", "\n\n", dedup_text).strip()

        removed_char_count = len(original_text) - len(dedup_text)
        return dedup_text, removed_char_count
