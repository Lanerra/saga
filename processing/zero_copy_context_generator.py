# processing/zero_copy_context_generator.py
"""
Zero-copy context generation (simplified, deterministic).

This module now builds a bounded hybrid context using:
- Recent chapters (sequential, no vector search)
- Salient KG facts for the current chapter

All streaming stats, intricate token budgeting, and fallback flows
have been removed for determinism and maintainability.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

import config
from data_access.chapter_queries import get_chapter_content_batch_native
from models import SceneDetail
from prompts.prompt_data_getters import get_reliable_kg_facts_for_drafting_prompt

logger = structlog.get_logger(__name__)


class ZeroCopyContextGenerator:
    """Deterministic, bounded hybrid context builder."""

    @staticmethod
    async def generate_hybrid_context_native(
        plot_outline: dict[str, Any],
        current_chapter_number: int,
        chapter_plan: list[SceneDetail] | None = None,
        semantic_limit: int | None = None,
        kg_limit: int | None = None,  # kept for signature compatibility
    ) -> str:
        """
        Build a simple hybrid context consisting of:
        - Plot point for the target chapter (if available)
        - Recent chapter summaries/snippets (bounded by count)
        - Key reliable KG facts
        """
        if current_chapter_number <= 0:
            return ""

        recent_count = semantic_limit or config.CONTEXT_CHAPTER_COUNT
        start_ch = max(1, current_chapter_number - recent_count)
        recent_numbers = [n for n in range(start_ch, current_chapter_number)]

        # Fetch recent chapter content and KG facts concurrently
        chapters_task = get_chapter_content_batch_native(recent_numbers)
        kg_task = get_reliable_kg_facts_for_drafting_prompt(
            plot_outline, current_chapter_number, chapter_plan
        )
        chapters_map, kg_facts_str = await asyncio.gather(chapters_task, kg_task)

        # Assemble context
        parts: list[str] = []

        # Plot point (no directive heuristics; keep it minimal and deterministic)
        plot_point = ZeroCopyContextGenerator._get_plot_point_for_chapter(
            plot_outline, current_chapter_number
        )
        parts.append(f"--- PLOT POINT FOR CHAPTER {current_chapter_number} ---")
        parts.append(
            f"**Plot Point:** {plot_point}"
            if (plot_point and plot_point.strip())
            else "(None provided)"
        )
        parts.append("--- END PLOT POINT ---")

        # Recent semantic context (sequential chapters only)
        parts.append("")
        parts.append("--- RECENT CHAPTER CONTEXT (SEQUENTIAL) ---")
        if not recent_numbers or not chapters_map:
            parts.append("No prior chapters available.")
        else:
            for n in recent_numbers:
                data = chapters_map.get(n)
                if not data:
                    continue
                summary = (data.get("summary") or "").strip()
                text = (data.get("text") or "").strip()

                if summary:
                    snippet = summary[: config.NARRATIVE_CONTEXT_SUMMARY_MAX_CHARS]
                elif text:
                    tail = text[-config.NARRATIVE_CONTEXT_TEXT_TAIL_CHARS :]
                    snippet = f"...{tail}" if tail else ""
                else:
                    snippet = ""

                if snippet:
                    parts.append(f"[Chapter {n}]:\n{snippet}\n---")

        parts.append("--- END RECENT CHAPTER CONTEXT ---")

        # KG facts
        parts.append("")
        parts.append("--- KEY RELIABLE KG FACTS ---")
        parts.append(
            kg_facts_str.strip()
            if (kg_facts_str and kg_facts_str.strip())
            else "No reliable KG facts available."
        )
        parts.append("--- END KEY RELIABLE KG FACTS ---")

        return "\n".join(parts)

    @staticmethod
    def _get_plot_point_for_chapter(
        plot_outline: dict[str, Any], current_chapter_number: int
    ) -> str | None:
        """
        Extract the plot point description for the specified chapter.

        Args:
            plot_outline: Plot outline data
            current_chapter_number: Chapter number (1-based)

        Returns:
            Plot point description string, or None if not found
        """
        plot_points = plot_outline.get("plot_points", [])
        if not plot_points or current_chapter_number <= 0:
            return None

        plot_point_index = current_chapter_number - 1  # Convert to 0-based index

        if 0 <= plot_point_index < len(plot_points):
            plot_point_item = plot_points[plot_point_index]
            # Handle both string and dict formats
            if isinstance(plot_point_item, dict):
                return plot_point_item.get("description", "").strip()
            elif isinstance(plot_point_item, str):
                return plot_point_item.strip()

        return None

    @staticmethod
    # Note: Previously this module included a complex "narrative focus directive"
    # heuristic. It has been removed to keep this builder deterministic.

    @staticmethod
    # All previous semantic context generation (embedding + vector search + fallbacks)
    # has been removed. We now use simple sequential recent chapters only.

    @staticmethod
    # Complex token-budgeted chapter aggregation removed.

    @staticmethod
    def _extract_narrative_continuation(
        chap_data: dict[str, Any], chap_num: int
    ) -> str:
        """
        Extract narrative continuation information from chapter data.
        Prioritizes concrete narrative endpoints over thematic summaries.
        """
        summary = chap_data.get("summary", "").strip()
        text = chap_data.get("text", "").strip()

        # If we have both summary and text, create enhanced context
        if summary and text:
            # Extract the final paragraphs from the chapter text for narrative continuation
            final_section = ZeroCopyContextGenerator._extract_chapter_ending(text)

            # Create enhanced context that includes both thematic summary and concrete continuation
            enhanced_parts = [
                f"**Thematic Summary of Chapter {chap_num}:**\n{summary}",
            ]

            if final_section:
                enhanced_parts.append(
                    f"**Chapter {chap_num} Ending (Narrative Continuation Context):**\n{final_section}"
                )

            # Add next chapter guidance if this is the immediate previous chapter
            next_chapter_guidance = (
                ZeroCopyContextGenerator._generate_next_chapter_guidance(
                    text, summary, chap_num
                )
            )
            if next_chapter_guidance:
                enhanced_parts.append(
                    f"**Guidance for Chapter {chap_num + 1}:**\n{next_chapter_guidance}"
                )

            return "\n\n".join(enhanced_parts)

        # Fallback to existing behavior if we only have summary or text
        return summary or text

    @staticmethod
    def _extract_chapter_ending(text: str, max_chars: int = 800) -> str:
        """
        Extract the ending portion of a chapter for narrative continuation.
        Focuses on final paragraphs that contain concrete narrative state.
        """
        if not text:
            return ""

        # Split into paragraphs and get the last few meaningful ones
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not paragraphs:
            return ""

        # Take final 2-3 paragraphs up to character limit
        ending_parts = []
        char_count = 0

        for paragraph in reversed(paragraphs[-5:]):  # Look at last 5 paragraphs
            if char_count + len(paragraph) + 4 <= max_chars:  # +4 for spacing
                ending_parts.append(paragraph)
                char_count += len(paragraph) + 4
            else:
                break

        if not ending_parts:
            # If none fit, truncate the last paragraph
            last_para = paragraphs[-1]
            ending_parts = [last_para[: max_chars - 3] + "..."]

        return "\n\n".join(reversed(ending_parts))

    @staticmethod
    def _generate_next_chapter_guidance(text: str, summary: str, chap_num: int) -> str:
        """
        Generate guidance for the next chapter based on current chapter's ending.
        Extracts concrete narrative states like character locations, plot developments.
        """
        guidance_parts = []

        # Extract character locations and states from the ending
        ending_section = text[-1500:] if text else ""  # Last ~1500 chars

        # Look for location indicators
        location_indicators = [
            "stood at",
            "stood in",
            "stood before",
            "stood near",
            "sat in",
            "sat at",
            "sat before",
            "sat near",
            "walked to",
            "walked toward",
            "walked into",
            "entered",
            "arrived at",
            "reached",
            "found himself",
            "found herself",
            "remained in",
            "stayed in",
        ]

        character_locations = []
        for indicator in location_indicators:
            if indicator in ending_section.lower():
                # Extract sentence containing location
                sentences = ending_section.split(".")
                for sentence in sentences:
                    if indicator in sentence.lower():
                        character_locations.append(sentence.strip())
                        break

        if character_locations:
            guidance_parts.append(f"Character Positions: {character_locations[-1]}")

        # Look for unresolved conflicts or pending actions
        conflict_indicators = [
            "but",
            "however",
            "yet",
            "still",
            "then",
            "suddenly",
            "before he could",
            "before she could",
            "interrupted by",
            "heard",
            "saw",
            "felt",
            "sensed",
            "realized",
        ]

        pending_actions = []
        sentences = ending_section.split(".")
        for sentence in sentences[-3:]:  # Last 3 sentences
            for indicator in conflict_indicators:
                if indicator in sentence.lower():
                    pending_actions.append(sentence.strip())
                    break

        if pending_actions:
            guidance_parts.append(f"Unresolved Elements: {pending_actions[-1]}")

        # Add plot progression note
        guidance_parts.append(
            f"Chapter {chap_num + 1} should continue from this point, advancing the plot to the next stage rather than repeating previous discoveries."
        )

        return " | ".join(guidance_parts) if guidance_parts else ""

    # Legacy fallback path removed; recent-chapters path is now primary.


## Note: Legacy wrapper generate_hybrid_chapter_context_native removed.
## Call ZeroCopyContextGenerator.generate_hybrid_context_native directly.
