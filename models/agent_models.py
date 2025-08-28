# models/agent_models.py
"""TypedDict structures used for inter-agent communication."""

from __future__ import annotations

from typing import TypedDict


class SceneDetail(TypedDict, total=False):
    """A detailed plan for a single scene."""

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
    """Information about a problem found during evaluation."""

    issue_category: str
    problem_description: str
    quote_from_original_text: str
    quote_char_start: int | None
    quote_char_end: int | None
    sentence_char_start: int | None
    sentence_char_end: int | None
    suggested_fix_focus: str


class EvaluationResult(TypedDict, total=False):
    """Structured result from the evaluator agent."""

    needs_revision: bool
    reasons: list[str]
    problems_found: list[ProblemDetail]


class PatchInstruction(TypedDict, total=False):
    """Instruction for applying a single patch to chapter text."""

    original_problem_quote_text: str
    target_char_start: int | None
    target_char_end: int | None
    replace_with: str
    reason_for_change: str
