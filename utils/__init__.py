"""General utility functions for the Saga Novel Generation system."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

import config
from core.llm_interface import count_tokens, llm_service
from processing.text_deduplicator import TextDeduplicator

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
    validate_world_item_fields,
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


# Deprecated: Use TextDeduplicator class directly instead
async def deduplicate_text_segments(
    original_text: str,
    segment_level: str = "paragraph",
    similarity_threshold: float = config.DEDUPLICATION_SEMANTIC_THRESHOLD,
    use_semantic_comparison: bool = config.DEDUPLICATION_USE_SEMANTIC,
    min_segment_length_chars: int = config.DEDUPLICATION_MIN_SEGMENT_LENGTH,
    prefer_newer: bool = False,
) -> Tuple[str, int]:
    """Remove near-duplicate segments from text.
    
    Deprecated: This function is maintained for backward compatibility but delegates
    to the TextDeduplicator class for actual implementation.
    """
    if not original_text.strip():
        return original_text, 0
        
    deduplicator = TextDeduplicator(
        similarity_threshold=similarity_threshold,
        use_semantic_comparison=use_semantic_comparison,
        min_segment_length_chars=min_segment_length_chars,
        prefer_newer=prefer_newer,
    )
    return await deduplicator.deduplicate(original_text, segment_level)


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
    "deduplicate_text_segments",  # Kept for backward compatibility
    "remove_spans_from_text",
    "validate_world_item_fields",
]
