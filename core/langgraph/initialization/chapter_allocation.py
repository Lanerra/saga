# core/langgraph/initialization/chapter_allocation.py
"""Allocate chapter ranges to acts for initialization workflows.

This module centralizes act/chapter allocation logic used by initialization nodes.

Notes:
    These utilities operate on plain dicts because the workflow state stores outlines
    as JSON-serializable dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeGuard


@dataclass(frozen=True)
class ActRange:
    """Represent an inclusive chapter range assigned to a single act."""

    act_number: int
    chapters_start: int
    chapters_end: int

    @property
    def chapters_in_act(self) -> int:
        """Return the number of chapters covered by this range."""
        if self.chapters_end < self.chapters_start:
            return 0
        return self.chapters_end - self.chapters_start + 1

    def contains(self, chapter_number: int) -> bool:
        """Return True when `chapter_number` lies within the inclusive range."""
        return self.chapters_start <= chapter_number <= self.chapters_end


def _is_int(value: object) -> TypeGuard[int]:
    """Return True when `value` is an int (excluding bool)."""
    # bool is a subclass of int; explicitly exclude it.
    return isinstance(value, int) and not isinstance(value, bool)


def extract_explicit_act_ranges(global_outline: dict) -> dict[int, ActRange]:
    """Extract per-act chapter ranges from a global outline dict.

    Args:
        global_outline: Global outline dict that may contain an `acts` list with
            `act_number`, `chapters_start`, and `chapters_end`.

    Returns:
        Mapping of act number to `ActRange`. Returns an empty dict when no usable
        explicit ranges are present.

    Notes:
        This function does not validate coverage or ordering; it only extracts ranges
        that are present and well-typed.
    """
    acts = global_outline.get("acts")
    if not isinstance(acts, list):
        return {}

    ranges: dict[int, ActRange] = {}
    for act in acts:
        if not isinstance(act, dict):
            continue

        act_number = act.get("act_number")
        chapters_start = act.get("chapters_start")
        chapters_end = act.get("chapters_end")

        if not (_is_int(act_number) and _is_int(chapters_start) and _is_int(chapters_end)):
            continue

        ranges[int(act_number)] = ActRange(
            act_number=int(act_number),
            chapters_start=int(chapters_start),
            chapters_end=int(chapters_end),
        )

    return ranges


def compute_balanced_act_ranges(total_chapters: int, act_count: int) -> dict[int, ActRange]:
    """Compute balanced chapter ranges for `act_count` acts.

    Args:
        total_chapters: Total number of chapters in the project.
        act_count: Number of acts to allocate.

    Returns:
        Mapping of act number to `ActRange`.

        When `act_count > total_chapters`, later acts receive empty ranges
        (`chapters_end < chapters_start`). When `act_count <= 0`, returns an empty dict.

    Notes:
        Remainder chapters are distributed to earlier acts so act sizes differ by at most 1.
    """
    if not _is_int(total_chapters) or total_chapters < 0:
        total_chapters = 0
    if not _is_int(act_count) or act_count <= 0:
        return {}

    base = total_chapters // act_count
    remainder = total_chapters % act_count

    ranges: dict[int, ActRange] = {}
    cursor = 1  # chapters are 1-indexed

    for act_number in range(1, act_count + 1):
        size = base + (1 if act_number <= remainder else 0)
        start = cursor
        end = cursor + size - 1
        if size == 0:
            end = start - 1  # empty range

        ranges[act_number] = ActRange(
            act_number=act_number,
            chapters_start=start,
            chapters_end=end,
        )
        cursor += size

    return ranges


def choose_act_ranges(global_outline: dict, total_chapters: int) -> dict[int, ActRange]:
    """Choose act ranges for downstream initialization nodes.

    Args:
        global_outline: Global outline dict that may contain explicit act ranges.
        total_chapters: Total number of chapters in the project.

    Returns:
        Mapping of act number to `ActRange`.

    Notes:
        Preference order:
        1) Use explicit act ranges when all acts 1..act_count provide `chapters_start` and
           `chapters_end`.
        2) Otherwise compute balanced ranges.

        When `act_count` is missing or invalid, this defaults to 3.
    """
    act_count_raw = global_outline.get("act_count", 3)
    act_count = act_count_raw if _is_int(act_count_raw) else 3
    if act_count <= 0:
        act_count = 3

    explicit = extract_explicit_act_ranges(global_outline)
    if explicit and all(i in explicit for i in range(1, act_count + 1)):
        return explicit

    return compute_balanced_act_ranges(total_chapters=total_chapters, act_count=act_count)


def determine_act_for_chapter(
    global_outline: dict,
    total_chapters: int,
    chapter_number: int,
) -> int:
    """Return the act number for a given chapter number.

    Args:
        global_outline: Global outline dict that may contain explicit act ranges.
        total_chapters: Total number of chapters in the project.
        chapter_number: 1-indexed chapter number.

    Returns:
        Act number, clamped into `[1, act_count]` when possible.
    """
    act_count_raw = global_outline.get("act_count", 3)
    act_count = act_count_raw if _is_int(act_count_raw) else 3
    if act_count <= 0:
        act_count = 3

    ranges = choose_act_ranges(global_outline=global_outline, total_chapters=total_chapters)

    for act_number in range(1, act_count + 1):
        act_range = ranges.get(act_number)
        if act_range and act_range.contains(chapter_number):
            return act_number

    # Fallback: if out-of-range, clamp to last act; if <=0, clamp to 1
    if chapter_number <= 0:
        return 1
    return act_count
