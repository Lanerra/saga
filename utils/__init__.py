"""General utility functions for the Saga Novel Generation system."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

import config
from core.llm_interface import count_tokens, llm_service

from .helpers import _is_fill_in
from .similarity import find_semantically_closest_segment, numpy_cosine_similarity
from .text_processing import (
    SpaCyModelManager,
    _normalize_for_id,
    _normalize_text_for_matching,
    find_quote_and_sentence_offsets_with_spacy,
    get_text_segments,
    load_spacy_model_if_needed,
    normalize_trait_name,
    spacy_manager,
)

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from models import SceneDetail

logger = logging.getLogger(__name__)


def format_scene_plan_for_prompt(
    chapter_plan: List["SceneDetail"],
    model_name_for_tokens: str,
    max_tokens_budget: int,
) -> str:
    """Format a chapter plan into plain text for LLM prompts respecting token limits."""
    if not chapter_plan:
        return "No detailed scene plan available."

    plan_lines = ["**Detailed Scene Plan (MUST BE FOLLOWED CLOSELY):**"]
    current_plan_parts = [plan_lines[0]]

    for scene_idx, scene in enumerate(chapter_plan):
        scene_lines = [
            f"Scene Number: {scene.get('scene_number', 'N/A')}",
            f"  Summary: {scene.get('summary', 'N/A')}",
            f"  Characters Involved: {', '.join(scene.get('characters_involved', [])) if scene.get('characters_involved') else 'None'}",
            "  Key Dialogue Points:",
        ]
        for point in scene.get("key_dialogue_points", []):
            scene_lines.append(f"    - {point}")
        scene_lines.append(f"  Setting Details: {scene.get('setting_details', 'N/A')}")
        scene_lines.append("  Scene Focus Elements:")
        for focus_el in scene.get("scene_focus_elements", []):
            scene_lines.append(f"    - {focus_el}")
        scene_lines.append(f"  Contribution: {scene.get('contribution', 'N/A')}")

        if scene_idx < len(chapter_plan) - 1:
            scene_lines.append("-" * 20)

        scene_segment = "\n".join(scene_lines)
        prospective_plan = "\n".join(current_plan_parts + [scene_segment])

        if count_tokens(prospective_plan, model_name_for_tokens) > max_tokens_budget:
            current_plan_parts.append(
                "... (plan truncated in prompt due to token limit)"
            )
            logger.warning(
                "Chapter plan was token-truncated for the prompt. Max tokens for plan: %d. Stopped before scene %s.",
                max_tokens_budget,
                scene.get("scene_number", "N/A"),
            )
            break

        current_plan_parts.append(scene_segment)

    if len(current_plan_parts) <= 1:
        return "No detailed scene plan available or plan was too long to include any scenes."

    return "\n".join(current_plan_parts)


async def deduplicate_text_segments(
    original_text: str,
    segment_level: str = "paragraph",
    similarity_threshold: float = config.DEDUPLICATION_SEMANTIC_THRESHOLD,
    use_semantic_comparison: bool = config.DEDUPLICATION_USE_SEMANTIC,
    min_segment_length_chars: int = config.DEDUPLICATION_MIN_SEGMENT_LENGTH,
    prefer_newer: bool = False,
) -> Tuple[str, int]:
    """Remove near-duplicate segments from text."""
    if not original_text.strip():
        return original_text, 0

    segments_with_offsets = get_text_segments(original_text, segment_level)
    if not segments_with_offsets:
        return original_text, 0

    num_segments = len(segments_with_offsets)
    indices_to_remove = set()

    embeddings: List[Optional[np.ndarray]] = []
    normalized_texts: List[str] = []

    if use_semantic_comparison:
        tasks = [
            llm_service.async_get_embedding(seg_text)
            for seg_text, _, _ in segments_with_offsets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        embeddings = [
            res if not isinstance(res, Exception) else None for res in results
        ]
    else:
        normalized_texts = [
            _normalize_text_for_matching(seg[0]) for seg in segments_with_offsets
        ]

    iteration_range = (
        range(num_segments - 1, -1, -1) if prefer_newer else range(num_segments)
    )

    for i in iteration_range:
        if i in indices_to_remove:
            continue

        text_i, start_i, end_i = segments_with_offsets[i]
        if len(text_i) < min_segment_length_chars:
            continue

        inner_range = (
            range(i - 1, -1, -1) if prefer_newer else range(i + 1, num_segments)
        )

        for j in inner_range:
            if j in indices_to_remove:
                continue

            text_j, start_j, end_j = segments_with_offsets[j]
            if len(text_j) < min_segment_length_chars:
                continue

            is_duplicate = False
            if use_semantic_comparison:
                emb_i = embeddings[i] if embeddings else None
                emb_j = embeddings[j] if embeddings else None
                if emb_i is not None and emb_j is not None:
                    similarity = numpy_cosine_similarity(emb_i, emb_j)
                    if similarity > similarity_threshold:
                        is_duplicate = True
                else:
                    is_duplicate = _normalize_text_for_matching(
                        text_i
                    ) == _normalize_text_for_matching(text_j)
            else:
                is_duplicate = normalized_texts[i] == normalized_texts[j]

            if is_duplicate:
                indices_to_remove.add(j)
                method_used = (
                    "semantic" if use_semantic_comparison else "normalized string"
                )
                logger.info(
                    "De-duplication: Marking segment (idx %d, chars %d-%d) for removal as duplicate of (idx %d, chars %d-%d). Method: %s.",
                    j,
                    start_j,
                    end_j,
                    i,
                    start_i,
                    end_i,
                    method_used,
                )

    if not indices_to_remove:
        return original_text, 0

    spans_to_remove_offsets = [
        segments_with_offsets[i][1:] for i in sorted(indices_to_remove)
    ]
    spans_to_remove_offsets.sort(key=lambda x: x[0])

    new_text_parts: List[str] = []
    last_pos = 0
    for start, end in spans_to_remove_offsets:
        if start > last_pos:
            new_text_parts.append(original_text[last_pos:start])
        last_pos = max(last_pos, end)

    if last_pos < len(original_text):
        new_text_parts.append(original_text[last_pos:])

    deduplicated_text = "".join(new_text_parts)
    deduplicated_text = re.sub(r"\n\s*\n(\s*\n)+", "\n\n", deduplicated_text)
    deduplicated_text = re.sub(r"\n{3,}", "\n\n", deduplicated_text).strip()

    characters_removed_count = len(original_text) - len(deduplicated_text)
    return deduplicated_text, characters_removed_count


def remove_spans_from_text(text: str, spans: List[Tuple[int, int]]) -> str:
    """Remove character spans from ``text``."""
    if not spans:
        return text

    spans_sorted = sorted(spans, key=lambda x: x[0])
    result_parts: List[str] = []
    last_end = 0
    for start, end in spans_sorted:
        if start > last_end:
            result_parts.append(text[last_end:start])
        last_end = max(last_end, end)
    result_parts.append(text[last_end:])
    return "".join(result_parts)


__all__ = [
    "_normalize_for_id",
    "normalize_trait_name",
    "SpaCyModelManager",
    "spacy_manager",
    "_is_fill_in",
    "load_spacy_model_if_needed",
    "_normalize_text_for_matching",
    "find_quote_and_sentence_offsets_with_spacy",
    "find_semantically_closest_segment",
    "numpy_cosine_similarity",
    "get_text_segments",
    "format_scene_plan_for_prompt",
    "deduplicate_text_segments",
    "remove_spans_from_text",
]
