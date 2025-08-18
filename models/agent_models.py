"""TypedDict structures used for inter-agent communication."""

from __future__ import annotations

from typing import List, Optional, TypedDict


class SceneDetail(TypedDict, total=False):
    """A detailed plan for a single scene."""

    scene_number: int
    summary: str
    characters_involved: List[str]
    key_dialogue_points: List[str]
    setting_details: str
    scene_focus_elements: List[str]
    contribution: str
    scene_type: str
    pacing: str
    character_arc_focus: Optional[str]
    relationship_development: Optional[str]


class ProblemDetail(TypedDict, total=False):
    """Information about a problem found during evaluation."""

    issue_category: str
    problem_description: str
    quote_from_original_text: str
    quote_char_start: Optional[int]
    quote_char_end: Optional[int]
    sentence_char_start: Optional[int]
    sentence_char_end: Optional[int]
    suggested_fix_focus: str


class EvaluationResult(TypedDict, total=False):
    """Structured result from the evaluator agent."""

    needs_revision: bool
    reasons: List[str]
    problems_found: List[ProblemDetail]


class PatchInstruction(TypedDict, total=False):
    """Instruction for applying a single patch to chapter text."""

    original_problem_quote_text: str
    target_char_start: Optional[int]
    target_char_end: Optional[int]
    replace_with: str
    reason_for_change: str
