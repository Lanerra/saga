# utils/__init__.py
"""General utility functions for the Saga Novel Generation system."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from core.llm_interface_refactored import llm_service

from .common import (
    _is_fill_in,
    extract_json_from_text,
    load_yaml_file,
    normalize_keys_recursive,
    safe_json_loads,
    split_text_into_chapters,
    truncate_for_log,
)
from .similarity import find_semantically_closest_segment, numpy_cosine_similarity
from .text_processing import (
    SpaCyModelManager,
    _normalize_for_id,
    _normalize_text_for_matching,
    find_quote_and_sentence_offsets_with_spacy,
    get_context_snippet_for_patch,
    get_text_segments,
    load_spacy_model_if_needed,
    normalize_trait_name,
    spacy_manager,
    validate_world_item_fields,
)

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from models import SceneDetail

logger = structlog.get_logger(__name__)


def format_scene_plan_for_prompt(
    chapter_plan: list[SceneDetail],
    model_name_for_tokens: str,
    max_tokens_budget: int,
) -> str:
    """Format a chapter plan into plain text for LLM prompts respecting token limits."""
    if not chapter_plan:
        return "No detailed scene plan available."

    header = "**Detailed Scene Plan (MUST BE FOLLOWED CLOSELY):**"
    current_plan_parts = [header]

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

        if (
            llm_service.count_tokens(prospective_plan, model_name_for_tokens)
            > max_tokens_budget
        ):
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


def remove_spans_from_text(text: str, spans: list[tuple[int, int]]) -> str:
    """Remove character spans from ``text``."""
    if not spans:
        return text

    spans_sorted = sorted(spans, key=lambda x: x[0])
    result_parts: list[str] = []
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
    "extract_json_from_text",
    "safe_json_loads",
    "truncate_for_log",
    "load_yaml_file",
    "normalize_keys_recursive",
    "split_text_into_chapters",
    "load_spacy_model_if_needed",
    "_normalize_text_for_matching",
    "get_context_snippet_for_patch",
    "find_quote_and_sentence_offsets_with_spacy",
    "find_semantically_closest_segment",
    "numpy_cosine_similarity",
    "get_text_segments",
    "format_scene_plan_for_prompt",
    "remove_spans_from_text",
    "validate_world_item_fields",
]
