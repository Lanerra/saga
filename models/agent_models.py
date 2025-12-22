# models/agent_models.py
"""Define inter-agent payload shapes.

These `TypedDict` definitions describe the structured messages exchanged between
SAGA agents (planner/evaluator/patcher).

Notes:
    All payloads use `total=False`, so keys are optional and may be omitted when not
    applicable or unknown.
"""

from __future__ import annotations

from typing import TypedDict


class SceneDetail(TypedDict, total=False):
    """Describe a detailed plan for a single scene."""

    # Canonical scene-planning output keys (used by scene planning + prompt rendering).
    title: str
    pov_character: str
    setting: str
    characters: list[str]
    plot_point: str
    conflict: str
    outcome: str

    # Optional enriched/normalized fields used by downstream prompt composition.
    scene_number: int
    summary: str
    characters_involved: list[str]
    key_dialogue_points: list[str]
    setting_details: str
    scene_focus_elements: list[str]
    contribution: str
    scene_type: str
    pacing: str
    character_arc_focus: str | None
    relationship_development: str | None


class ProblemDetail(TypedDict, total=False):
    """Describe a single issue found during evaluation.

    Notes:
        Character offsets refer to positions in the evaluated text, when available.
    """

    issue_category: str
    problem_description: str
    quote_from_original_text: str
    quote_char_start: int | None
    quote_char_end: int | None
    sentence_char_start: int | None
    sentence_char_end: int | None
    suggested_fix_focus: str


class EvaluationResult(TypedDict, total=False):
    """Represent the evaluator agent's structured result."""

    needs_revision: bool
    reasons: list[str]
    problems_found: list[ProblemDetail]


class PatchInstruction(TypedDict, total=False):
    """Describe a single patch to apply to chapter text.

    Notes:
        When present, `target_char_start` and `target_char_end` define the character-span
        in the original text to be replaced by `replace_with`.
    """

    original_problem_quote_text: str
    target_char_start: int | None
    target_char_end: int | None
    replace_with: str
    reason_for_change: str
