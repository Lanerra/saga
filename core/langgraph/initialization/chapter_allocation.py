# core/langgraph/initialization/chapter_allocation.py
"""
Shared helpers for mapping chapters to acts and allocating chapter ranges.

This module centralizes the logic used by initialization nodes so that:
- Explicit chapter ranges from the global outline (chapters_start/chapters_end)
  are respected when present.
- Fallback allocation covers all chapters exactly once, distributing remainder
  chapters across early acts.
- Edge cases like act_count > total_chapters (empty act ranges) do not crash.

These utilities intentionally operate on plain dicts because the graph state
stores outlines as JSON-serializable dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActRange:
    act_number: int
    chapters_start: int
    chapters_end: int

    @property
    def chapters_in_act(self) -> int:
        if self.chapters_end < self.chapters_start:
            return 0
        return self.chapters_end - self.chapters_start + 1

    def contains(self, chapter_number: int) -> bool:
        return self.chapters_start <= chapter_number <= self.chapters_end


def _is_int(value: object) -> bool:
    # bool is a subclass of int; explicitly exclude it.
    return isinstance(value, int) and not isinstance(value, bool)


def extract_explicit_act_ranges(global_outline: dict) -> dict[int, ActRange]:
    """
    Extract explicit per-act chapter ranges from the global outline (if present).

    Returns an empty dict if:
    - global_outline has no "acts" list, or
    - required fields are missing / not integers.

    Note: This does not attempt to validate coverage or ordering; the global
    outline node already records validation issues in state for visibility.
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
    """
    Compute balanced act chapter ranges as a safe fallback.

    Properties:
    - Covers chapters 1..total_chapters exactly once (if total_chapters >= 1)
    - Distributes remainder chapters to early acts: sizes differ by at most 1
    - If act_count > total_chapters, later acts receive empty ranges
      (chapters_start = next chapter, chapters_end = chapters_start - 1)

    Returns empty dict when act_count <= 0.
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
    """
    Choose act ranges for downstream nodes.

    Preference order:
    1) Use explicit act ranges from the global outline when *all* acts 1..act_count
       have chapters_start/chapters_end present.
    2) Otherwise compute balanced ranges.

    If act_count is missing/invalid, defaults to 3.
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
    """
    Determine which act a chapter belongs to, using explicit ranges when available.

    Returns an act number in the range [1, act_count] when possible.
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
