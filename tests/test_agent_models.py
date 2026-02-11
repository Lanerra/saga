# tests/test_agent_models.py
"""Tests for models/agent_models.py - inter-agent payload TypedDicts."""

import json

from models.agent_models import (
    EvaluationResult,
    PatchInstruction,
    ProblemDetail,
    SceneDetail,
)


class TestSceneDetail:
    """Tests for SceneDetail TypedDict construction and serialization."""

    def test_full_construction(self) -> None:
        """All fields populated."""
        scene: SceneDetail = {
            "title": "The Crossing",
            "pov_character": "Aria",
            "setting": "River bridge at dawn",
            "characters": ["Aria", "Bren"],
            "plot_point": "Aria confronts Bren about the betrayal",
            "conflict": "Trust vs. self-preservation",
            "outcome": "Aria chooses to forgive",
            "beats": ["Aria approaches", "Dialogue escalates", "Resolution"],
            "scene_number": 3,
            "summary": "Aria and Bren reconcile on the bridge",
            "characters_involved": ["Aria", "Bren"],
            "key_dialogue_points": ["accusation", "confession"],
            "setting_details": "Mist rising from the river",
            "scene_focus_elements": ["tension", "forgiveness"],
            "contribution": "Advances trust arc",
            "scene_type": "confrontation",
            "pacing": "slow build to climax",
            "character_arc_focus": "Aria's growth",
            "relationship_development": "Aria-Bren reconciliation",
        }

        assert scene["title"] == "The Crossing"
        assert scene["pov_character"] == "Aria"
        assert scene["setting"] == "River bridge at dawn"
        assert scene["characters"] == ["Aria", "Bren"]
        assert scene["plot_point"] == "Aria confronts Bren about the betrayal"
        assert scene["conflict"] == "Trust vs. self-preservation"
        assert scene["outcome"] == "Aria chooses to forgive"
        assert scene["beats"] == ["Aria approaches", "Dialogue escalates", "Resolution"]
        assert scene["scene_number"] == 3
        assert scene["summary"] == "Aria and Bren reconcile on the bridge"
        assert scene["characters_involved"] == ["Aria", "Bren"]
        assert scene["key_dialogue_points"] == ["accusation", "confession"]
        assert scene["setting_details"] == "Mist rising from the river"
        assert scene["scene_focus_elements"] == ["tension", "forgiveness"]
        assert scene["contribution"] == "Advances trust arc"
        assert scene["scene_type"] == "confrontation"
        assert scene["pacing"] == "slow build to climax"
        assert scene["character_arc_focus"] == "Aria's growth"
        assert scene["relationship_development"] == "Aria-Bren reconciliation"

    def test_partial_construction(self) -> None:
        """Only a subset of fields populated."""
        scene: SceneDetail = {
            "title": "Ambush",
            "pov_character": "Kael",
            "scene_number": 7,
        }

        assert scene["title"] == "Ambush"
        assert scene["pov_character"] == "Kael"
        assert scene["scene_number"] == 7
        assert len(scene) == 3

    def test_empty_construction(self) -> None:
        """No fields populated."""
        scene: SceneDetail = {}

        assert dict(scene) == {}
        assert len(scene) == 0

    def test_json_roundtrip(self) -> None:
        """json.dumps followed by json.loads preserves structure."""
        scene: SceneDetail = {
            "title": "The Escape",
            "characters": ["Mira", "Dex"],
            "beats": ["chase begins", "narrow miss", "freedom"],
            "scene_number": 12,
            "character_arc_focus": None,
            "relationship_development": None,
        }

        serialized = json.dumps(scene)
        deserialized = json.loads(serialized)

        assert deserialized == scene


class TestProblemDetail:
    """Tests for ProblemDetail TypedDict construction and serialization."""

    def test_full_construction(self) -> None:
        """All fields populated with concrete values."""
        problem: ProblemDetail = {
            "issue_category": "continuity",
            "problem_description": "Character eye colour changes mid-scene",
            "quote_from_original_text": "Her green eyes flashed blue",
            "quote_char_start": 140,
            "quote_char_end": 170,
            "sentence_char_start": 130,
            "sentence_char_end": 185,
            "suggested_fix_focus": "Standardise eye colour to green",
        }

        assert problem["issue_category"] == "continuity"
        assert problem["problem_description"] == "Character eye colour changes mid-scene"
        assert problem["quote_from_original_text"] == "Her green eyes flashed blue"
        assert problem["quote_char_start"] == 140
        assert problem["quote_char_end"] == 170
        assert problem["sentence_char_start"] == 130
        assert problem["sentence_char_end"] == 185
        assert problem["suggested_fix_focus"] == "Standardise eye colour to green"

    def test_nullable_offset_fields(self) -> None:
        """Offset fields accept None."""
        problem: ProblemDetail = {
            "issue_category": "pacing",
            "problem_description": "Scene drags without purpose",
            "quote_from_original_text": "They sat in silence for hours",
            "quote_char_start": None,
            "quote_char_end": None,
            "sentence_char_start": None,
            "sentence_char_end": None,
            "suggested_fix_focus": "Trim or add tension",
        }

        assert problem["quote_char_start"] is None
        assert problem["quote_char_end"] is None
        assert problem["sentence_char_start"] is None
        assert problem["sentence_char_end"] is None

    def test_json_roundtrip(self) -> None:
        """json.dumps followed by json.loads preserves structure including None."""
        problem: ProblemDetail = {
            "issue_category": "dialogue",
            "problem_description": "Stilted exchange",
            "quote_from_original_text": "Hello. Yes. Goodbye.",
            "quote_char_start": 50,
            "quote_char_end": 70,
            "sentence_char_start": None,
            "sentence_char_end": None,
            "suggested_fix_focus": "Make dialogue natural",
        }

        serialized = json.dumps(problem)
        deserialized = json.loads(serialized)

        assert deserialized == problem


class TestEvaluationResult:
    """Tests for EvaluationResult TypedDict construction and serialization."""

    def test_full_construction_with_nested_problems(self) -> None:
        """Full construction with a nested ProblemDetail list."""
        problem_one: ProblemDetail = {
            "issue_category": "continuity",
            "problem_description": "Name inconsistency",
            "quote_from_original_text": "Jon said",
            "quote_char_start": 10,
            "quote_char_end": 18,
            "sentence_char_start": 0,
            "sentence_char_end": 30,
            "suggested_fix_focus": "Use consistent name John",
        }
        problem_two: ProblemDetail = {
            "issue_category": "pacing",
            "problem_description": "Abrupt scene transition",
            "quote_from_original_text": "Suddenly they were elsewhere",
            "quote_char_start": None,
            "quote_char_end": None,
            "sentence_char_start": None,
            "sentence_char_end": None,
            "suggested_fix_focus": "Add transitional passage",
        }

        result: EvaluationResult = {
            "needs_revision": True,
            "reasons": ["continuity error", "pacing issue"],
            "problems_found": [problem_one, problem_two],
        }

        assert result["needs_revision"] is True
        assert result["reasons"] == ["continuity error", "pacing issue"]
        assert len(result["problems_found"]) == 2
        assert result["problems_found"][0]["issue_category"] == "continuity"
        assert result["problems_found"][1]["issue_category"] == "pacing"

    def test_empty_construction(self) -> None:
        """No fields populated."""
        result: EvaluationResult = {}

        assert dict(result) == {}
        assert len(result) == 0


class TestPatchInstruction:
    """Tests for PatchInstruction TypedDict construction and serialization."""

    def test_full_construction(self) -> None:
        """All fields populated with concrete values."""
        patch: PatchInstruction = {
            "original_problem_quote_text": "Her green eyes flashed blue",
            "target_char_start": 140,
            "target_char_end": 170,
            "replace_with": "Her green eyes flashed with fury",
            "reason_for_change": "Fix continuity: eye colour must stay green",
        }

        assert patch["original_problem_quote_text"] == "Her green eyes flashed blue"
        assert patch["target_char_start"] == 140
        assert patch["target_char_end"] == 170
        assert patch["replace_with"] == "Her green eyes flashed with fury"
        assert patch["reason_for_change"] == "Fix continuity: eye colour must stay green"

    def test_nullable_offset_fields(self) -> None:
        """Offset fields accept None when position is unknown."""
        patch: PatchInstruction = {
            "original_problem_quote_text": "vague passage",
            "target_char_start": None,
            "target_char_end": None,
            "replace_with": "precise passage",
            "reason_for_change": "Improve clarity",
        }

        assert patch["target_char_start"] is None
        assert patch["target_char_end"] is None

    def test_json_roundtrip(self) -> None:
        """json.dumps followed by json.loads preserves structure including None."""
        patch: PatchInstruction = {
            "original_problem_quote_text": "They walked slow",
            "target_char_start": 200,
            "target_char_end": 216,
            "replace_with": "They walked slowly",
            "reason_for_change": "Grammar correction",
        }

        serialized = json.dumps(patch)
        deserialized = json.loads(serialized)

        assert deserialized == patch
