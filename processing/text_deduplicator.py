# processing/text_deduplicator.py
"""Utilities for detecting and removing duplicate text segments."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re

import numpy as np

import config
import utils
from core.llm_interface_refactored import llm_service

logger = logging.getLogger(__name__)


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

    async def deduplicate(
        self, original_text: str, segment_level: str = "paragraph"
    ) -> tuple[str, int]:
        if not original_text.strip():
            return original_text, 0

        segments = utils.get_text_segments(original_text, segment_level)
        if not segments:
            return original_text, 0

        normalized_cache: list[str] = [
            utils._normalize_text_for_matching(seg[0]) for seg in segments
        ]
        indices_to_remove: set[int] = set()
        fingerprint_map: dict[str, int] = {}
        iteration_range = (
            range(len(segments) - 1, -1, -1)
            if self.prefer_newer
            else range(len(segments))
        )

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
            tasks = [
                llm_service.async_get_embedding(segments[i][0]) for i in unique_indices
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, result in zip(unique_indices, results, strict=False):
                if not isinstance(result, Exception):
                    embeddings[idx] = result

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
                    similarity = utils.numpy_cosine_similarity(embeddings[idx], emb_j)
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

        spans_to_remove = [segments[i][1:] for i in sorted(indices_to_remove)]
        spans_to_remove.sort(key=lambda x: x[0])
        new_parts: list[str] = []
        last_pos = 0
        for start, end in spans_to_remove:
            if start > last_pos:
                new_parts.append(original_text[last_pos:start])
            last_pos = max(last_pos, end)
        if last_pos < len(original_text):
            new_parts.append(original_text[last_pos:])
        dedup_text = "".join(new_parts)
        dedup_text = re.sub(r"\n\s*\n(\s*\n)+", "\n\n", dedup_text)
        dedup_text = re.sub(r"\n{3,}", "\n\n", dedup_text).strip()
        removed_count = len(original_text) - len(dedup_text)
        return dedup_text, removed_count
