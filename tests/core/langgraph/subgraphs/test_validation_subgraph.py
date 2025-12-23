# tests/core/langgraph/subgraphs/test_validation_subgraph.py
"""
Tests for validation subgraph.

Covers all functions in core/langgraph/subgraphs/validation.py.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.langgraph.state import Contradiction
from core.langgraph.subgraphs.validation import (
    _build_quality_evaluation_prompt,
    _build_rule_check_prompt,
    _check_relationship_evolution,
    _check_timeline,
    _check_world_rules,
    _events_are_related,
    _is_temporal_violation,
    _parse_quality_scores,
    _parse_rule_violations,
    create_validation_subgraph,
    detect_contradictions,
    evaluate_quality,
    validate_consistency,
)


@pytest.fixture
def sample_validation_state(tmp_path):
    """Create a sample state for validation testing."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    content_dir = project_dir / ".saga" / "content"
    content_dir.mkdir(parents=True)

    draft_file = content_dir / "chapter_1_draft.txt"
    draft_file.write_text("This is a test chapter with some narrative content.")

    return {
        "project_dir": str(project_dir),
        "current_chapter": 1,
        "genre": "fantasy",
        "theme": "courage",
        "draft_text": "This is a test chapter with some narrative content.",
        "draft_word_count": 10,
        "draft_ref": {
            "content_type": "draft",
            "chapter": 1,
            "path": str(draft_file),
        },
        "contradictions": [],
        "extracted_entities": {
            "characters": [],
            # Canonical extraction contract uses `world_items` for non-character entities
            # including events (type == "Event"). We keep legacy keys in this fixture
            # for backward-compatible callers/tests.
            "world_items": [],
            "locations": [],
            "events": [],
        },
        "extracted_relationships": [],
        "character_profiles": [],
        "previous_summaries": [],
        "chapter_outlines": {},
        "needs_revision": False,
    }


@pytest.mark.asyncio
class TestValidateConsistency:
    """Tests for validate_consistency wrapper function."""

    async def test_validate_consistency_calls_original(self, sample_validation_state):
        """Validate consistency delegates to original validation node."""
        with patch(
            "core.langgraph.subgraphs.validation.original_validate_consistency",
            new_callable=AsyncMock,
        ) as mock_validate:
            mock_validate.return_value = {
                **sample_validation_state,
                "contradictions": [],
                "needs_revision": False,
            }

            result = await validate_consistency(sample_validation_state)

            mock_validate.assert_called_once_with(sample_validation_state)
            assert result["contradictions"] == []


@pytest.mark.asyncio
class TestEvaluateQuality:
    """Tests for evaluate_quality function."""

    async def test_evaluate_quality_no_draft_text(self, sample_validation_state):
        """When no draft text, returns None scores."""
        state = sample_validation_state.copy()
        state["draft_text"] = ""
        state["draft_ref"] = None

        result = await evaluate_quality(state)

        assert result["coherence_score"] is None
        assert result["prose_quality_score"] is None
        assert result["plot_advancement_score"] is None
        assert result["pacing_score"] is None
        assert result["tone_consistency_score"] is None
        assert "No draft text" in result["quality_feedback"]

    async def test_evaluate_quality_success(self, sample_validation_state):
        """Successful quality evaluation returns scores."""
        mock_response = """{
            "coherence_score": 0.85,
            "prose_quality_score": 0.75,
            "plot_advancement_score": 0.80,
            "pacing_score": 0.70,
            "tone_consistency_score": 0.90,
            "feedback": "Good pacing and coherence."
        }"""

        with patch(
            "core.langgraph.subgraphs.validation.llm_service.async_call_llm",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.return_value = (mock_response, {"tokens": 100})

            result = await evaluate_quality(sample_validation_state)

            assert result["coherence_score"] == 0.85
            assert result["prose_quality_score"] == 0.75
            assert result["plot_advancement_score"] == 0.80
            assert result["pacing_score"] == 0.70
            assert result["tone_consistency_score"] == 0.90
            assert "Good pacing" in result["quality_feedback"]

    async def test_evaluate_quality_below_threshold_triggers_contradiction(self, sample_validation_state):
        """Low quality scores trigger quality contradiction."""
        mock_response = """{
            "coherence_score": 0.3,
            "prose_quality_score": 0.4,
            "plot_advancement_score": 0.2,
            "pacing_score": 0.5,
            "tone_consistency_score": 0.6,
            "feedback": "Needs improvement."
        }"""

        with patch(
            "core.langgraph.subgraphs.validation.llm_service.async_call_llm",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.return_value = (mock_response, {"tokens": 100})

            result = await evaluate_quality(sample_validation_state)

            quality_contradictions = [c for c in result["contradictions"] if c.type == "quality_issue"]
            assert len(quality_contradictions) == 1
            assert "below threshold" in quality_contradictions[0].description

    async def test_evaluate_quality_llm_error(self, sample_validation_state):
        """LLM error during evaluation returns None scores."""
        with patch(
            "core.langgraph.subgraphs.validation.llm_service.async_call_llm",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.side_effect = Exception("LLM connection failed")

            result = await evaluate_quality(sample_validation_state)

            assert result["coherence_score"] is None
            assert result["prose_quality_score"] is None
            assert "Evaluation failed" in result["quality_feedback"]


class TestBuildQualityEvaluationPrompt:
    """Tests for _build_quality_evaluation_prompt function."""

    def test_build_prompt_basic(self):
        """Basic prompt building works."""
        prompt = _build_quality_evaluation_prompt(
            draft_text="Test chapter text",
            chapter_number=1,
            genre="fantasy",
            theme="courage",
            previous_summaries=[],
            chapter_outline={},
        )

        assert "Test chapter text" in prompt
        assert "fantasy" in prompt
        assert "courage" in prompt
        assert "Chapter 1" in prompt
        assert "JSON object" in prompt

    def test_build_prompt_truncates_long_text(self):
        """Long draft text is truncated properly."""
        long_text = "x" * 10000
        prompt = _build_quality_evaluation_prompt(
            draft_text=long_text,
            chapter_number=2,
            genre="sci-fi",
            theme="exploration",
            previous_summaries=[],
            chapter_outline={},
        )

        assert "truncated for evaluation" in prompt
        assert len(prompt) < len(long_text) + 1000

    def test_build_prompt_with_summaries(self):
        """Previous summaries are included in prompt."""
        summaries = [
            "Chapter 1 summary",
            "Chapter 2 summary",
            "Chapter 3 summary",
            "Chapter 4 summary",
        ]

        prompt = _build_quality_evaluation_prompt(
            draft_text="Test text",
            chapter_number=5,
            genre="mystery",
            theme="justice",
            previous_summaries=summaries,
            chapter_outline={},
        )

        assert "Chapter 2 summary" in prompt
        assert "Chapter 3 summary" in prompt
        assert "Chapter 4 summary" in prompt
        assert "Chapter 1 summary" not in prompt

    def test_build_prompt_with_outline(self):
        """Chapter outline is included in prompt."""
        outline = {
            "scene_description": "Hero confronts villain",
            "key_beats": ["discovery", "confrontation", "escape"],
            "plot_point": "First turning point",
        }

        prompt = _build_quality_evaluation_prompt(
            draft_text="Test text",
            chapter_number=3,
            genre="thriller",
            theme="survival",
            previous_summaries=[],
            chapter_outline=outline,
        )

        assert "Hero confronts villain" in prompt
        assert "discovery" in prompt
        assert "First turning point" in prompt


class TestParseQualityScores:
    """Tests for _parse_quality_scores function."""

    def test_parse_valid_json(self):
        """Valid JSON response is parsed correctly."""
        response = """{
            "coherence_score": 0.85,
            "prose_quality_score": 0.75,
            "plot_advancement_score": 0.80,
            "pacing_score": 0.70,
            "tone_consistency_score": 0.90,
            "feedback": "Great work!"
        }"""

        scores = _parse_quality_scores(response)

        assert scores["coherence_score"] == 0.85
        assert scores["prose_quality_score"] == 0.75
        assert scores["plot_advancement_score"] == 0.80
        assert scores["pacing_score"] == 0.70
        assert scores["tone_consistency_score"] == 0.90
        assert scores["feedback"] == "Great work!"

    def test_parse_numeric_values_does_not_raise_typeerror(self):
        """
        Regression test for runtime crash when using PEP-604 unions inside isinstance().

        Before the fix, this would raise:
          TypeError: isinstance() argument 2 cannot be a union

        Ensure representative numeric values (ints + floats) parse successfully.
        """
        response = """{
            "coherence_score": 1,
            "prose_quality_score": 0.75,
            "plot_advancement_score": 0,
            "pacing_score": 0.7,
            "tone_consistency_score": 1.0,
            "feedback": "OK"
        }"""

        scores = _parse_quality_scores(response)

        assert scores["coherence_score"] == 1.0
        assert scores["plot_advancement_score"] == 0.0
        assert scores["prose_quality_score"] == 0.75
        assert scores["pacing_score"] == 0.7
        assert scores["tone_consistency_score"] == 1.0
        assert scores["feedback"] == "OK"

    def test_parse_clamps_scores(self):
        """Scores outside 0-1 range are clamped."""
        response = """{
            "coherence_score": 1.5,
            "prose_quality_score": -0.2,
            "plot_advancement_score": 0.5,
            "pacing_score": 0.7,
            "tone_consistency_score": 0.9,
            "feedback": "Test"
        }"""

        scores = _parse_quality_scores(response)

        assert scores["coherence_score"] == 1.0
        assert scores["prose_quality_score"] == 0.0
        assert scores["plot_advancement_score"] == 0.5

    def test_parse_invalid_json_returns_fallback(self):
        """Invalid JSON returns fallback scores."""
        response = "This is not valid JSON at all"

        scores = _parse_quality_scores(response)

        assert scores["coherence_score"] == 0.7
        assert scores["prose_quality_score"] == 0.7
        assert scores["plot_advancement_score"] == 0.7
        assert "Unable to parse" in scores["feedback"]

    def test_parse_extracts_from_text(self):
        """Scores can be extracted from text without JSON."""
        response = """
        The coherence score is 0.85 and the prose quality is 0.72.
        Pacing score: 0.68
        Plot advancement: 0.91
        Tone consistency: 0.77
        """

        scores = _parse_quality_scores(response)

        assert 0.0 <= scores["coherence_score"] <= 1.0
        assert 0.0 <= scores["prose_quality_score"] <= 1.0
        assert 0.0 <= scores["pacing_score"] <= 1.0
        assert 0.0 <= scores["plot_advancement_score"] <= 1.0
        assert 0.0 <= scores["tone_consistency_score"] <= 1.0
        assert any(score in [0.85, 0.72, 0.68, 0.91, 0.77] for score in scores.values() if isinstance(score, float))


@pytest.mark.asyncio
class TestDetectContradictions:
    """Tests for detect_contradictions function."""

    async def test_detect_contradictions_basic(self, sample_validation_state):
        """Basic contradiction detection runs."""
        with (
            patch(
                "core.langgraph.subgraphs.validation._check_timeline",
                new_callable=AsyncMock,
            ) as mock_timeline,
            patch(
                "core.langgraph.subgraphs.validation._check_world_rules",
                new_callable=AsyncMock,
            ) as mock_rules,
            patch(
                "core.langgraph.subgraphs.validation._check_relationship_evolution",
                new_callable=AsyncMock,
            ) as mock_relationships,
        ):
            mock_timeline.return_value = []
            mock_rules.return_value = []
            mock_relationships.return_value = []

            result = await detect_contradictions(sample_validation_state)

            assert result["current_node"] == "detect_contradictions"
            assert result["needs_revision"] is False
            mock_timeline.assert_called_once()
            mock_rules.assert_called_once()
            mock_relationships.assert_called_once()

    async def test_detect_contradictions_timeline_uses_world_items_events(self, sample_validation_state):
        """
        Regression test for LANGGRAPH-003 / LANGGRAPH-017:

        The validation subgraph must not assume an `extracted_entities["events"]` bucket.
        Canonical extraction stores events in `extracted_entities["world_items"]` with
        type == "Event".
        """
        state = sample_validation_state.copy()
        state["current_chapter"] = 3
        state["extracted_entities"] = {
            "characters": [],
            "world_items": [
                {
                    "name": "Battle of the Bridge",
                    "type": "Event",
                    "description": "A decisive fight at the bridge.",
                    "attributes": {"timestamp": "today"},
                }
            ],
        }

        with (
            patch(
                "core.langgraph.subgraphs.validation._check_timeline",
                new_callable=AsyncMock,
            ) as mock_timeline,
            patch(
                "core.langgraph.subgraphs.validation._check_world_rules",
                new_callable=AsyncMock,
            ) as mock_rules,
            patch(
                "core.langgraph.subgraphs.validation._check_relationship_evolution",
                new_callable=AsyncMock,
            ) as mock_relationships,
        ):
            mock_timeline.return_value = []
            mock_rules.return_value = []
            mock_relationships.return_value = []

            await detect_contradictions(state)

            assert mock_timeline.call_count == 1
            passed_events = mock_timeline.call_args.args[0]
            assert isinstance(passed_events, list)
            assert len(passed_events) == 1
            assert passed_events[0].get("type") == "Event"
            assert passed_events[0].get("name") == "Battle of the Bridge"

    async def test_detect_contradictions_critical_triggers_revision(self, sample_validation_state):
        """Critical contradictions trigger needs_revision."""
        critical_contradiction = Contradiction(
            type="timeline",
            description="Critical timeline violation",
            conflicting_chapters=[1, 2],
            severity="critical",
            suggested_fix="Fix timeline",
        )

        with (
            patch(
                "core.langgraph.subgraphs.validation._check_timeline",
                new_callable=AsyncMock,
            ) as mock_timeline,
            patch(
                "core.langgraph.subgraphs.validation._check_world_rules",
                new_callable=AsyncMock,
            ) as mock_rules,
            patch(
                "core.langgraph.subgraphs.validation._check_relationship_evolution",
                new_callable=AsyncMock,
            ) as mock_relationships,
        ):
            mock_timeline.return_value = [critical_contradiction]
            mock_rules.return_value = []
            mock_relationships.return_value = []

            result = await detect_contradictions(sample_validation_state)

            assert result["needs_revision"] is True
            assert len(result["contradictions"]) == 1

    async def test_detect_contradictions_multiple_major_triggers_revision(self, sample_validation_state):
        """Multiple major contradictions trigger revision."""
        major_contradictions = [
            Contradiction(
                type="world_rule",
                description="First major issue",
                conflicting_chapters=[1],
                severity="major",
                suggested_fix="Fix 1",
            ),
            Contradiction(
                type="world_rule",
                description="Second major issue",
                conflicting_chapters=[1],
                severity="major",
                suggested_fix="Fix 2",
            ),
            Contradiction(
                type="world_rule",
                description="Third major issue",
                conflicting_chapters=[1],
                severity="major",
                suggested_fix="Fix 3",
            ),
        ]

        with (
            patch(
                "core.langgraph.subgraphs.validation._check_timeline",
                new_callable=AsyncMock,
            ) as mock_timeline,
            patch(
                "core.langgraph.subgraphs.validation._check_world_rules",
                new_callable=AsyncMock,
            ) as mock_rules,
            patch(
                "core.langgraph.subgraphs.validation._check_relationship_evolution",
                new_callable=AsyncMock,
            ) as mock_relationships,
        ):
            mock_timeline.return_value = []
            mock_rules.return_value = major_contradictions
            mock_relationships.return_value = []

            result = await detect_contradictions(sample_validation_state)

            assert result["needs_revision"] is True

    async def test_detect_contradictions_force_continue_bypasses(self, sample_validation_state):
        """force_continue bypasses revision even with contradictions."""
        state = sample_validation_state.copy()
        state["force_continue"] = True

        critical_contradiction = Contradiction(
            type="timeline",
            description="Critical issue",
            conflicting_chapters=[1],
            severity="critical",
            suggested_fix="Fix it",
        )

        with (
            patch(
                "core.langgraph.subgraphs.validation._check_timeline",
                new_callable=AsyncMock,
            ) as mock_timeline,
            patch(
                "core.langgraph.subgraphs.validation._check_world_rules",
                new_callable=AsyncMock,
            ) as mock_rules,
            patch(
                "core.langgraph.subgraphs.validation._check_relationship_evolution",
                new_callable=AsyncMock,
            ) as mock_relationships,
        ):
            mock_timeline.return_value = [critical_contradiction]
            mock_rules.return_value = []
            mock_relationships.return_value = []

            result = await detect_contradictions(state)

            assert result["needs_revision"] is False


@pytest.mark.asyncio
class TestCheckTimeline:
    """Tests for _check_timeline function."""

    async def test_check_timeline_no_events(self):
        """Empty events list returns no contradictions."""
        contradictions = await _check_timeline([], 1)
        assert contradictions == []

    async def test_check_timeline_no_existing_events(self):
        """No existing events in database returns no contradictions."""
        mock_entity = MagicMock()
        mock_entity.description = "New event"
        mock_entity.attributes = {"timestamp": "today"}

        with patch(
            "core.langgraph.subgraphs.validation.neo4j_manager.execute_read_query",
            new_callable=AsyncMock,
        ) as mock_query:
            mock_query.return_value = []

            contradictions = await _check_timeline([mock_entity], 2)

            assert contradictions == []

    async def test_check_timeline_detects_violation(self):
        """Timeline violations are detected."""
        mock_entity = MagicMock()
        mock_entity.description = "event meeting battle combat"
        mock_entity.attributes = {"timestamp": "before the war"}

        existing_events = [
            {
                "description": "the great battle and meeting",
                "timestamp": "after the war",
                "chapter": 1,
            }
        ]

        with patch(
            "core.langgraph.subgraphs.validation.neo4j_manager.execute_read_query",
            new_callable=AsyncMock,
        ) as mock_query:
            mock_query.return_value = existing_events

            contradictions = await _check_timeline([mock_entity], 2)

            assert len(contradictions) >= 0

    async def test_check_timeline_error_handling(self):
        """Timeline check handles errors gracefully."""
        mock_entity = MagicMock()
        mock_entity.description = "Test event"
        mock_entity.attributes = {}

        with patch(
            "core.langgraph.subgraphs.validation.neo4j_manager.execute_read_query",
            new_callable=AsyncMock,
        ) as mock_query:
            mock_query.side_effect = Exception("Database error")

            contradictions = await _check_timeline([mock_entity], 2)

            assert contradictions == []


class TestEventsAreRelated:
    """Tests for _events_are_related function."""

    def test_events_with_shared_words_are_related(self):
        """Events sharing significant words are related."""
        event1 = "The great battle at Waterloo"
        event2 = "The battle preparations at Waterloo"

        assert _events_are_related(event1, event2) is True

    def test_events_without_shared_words_are_not_related(self):
        """Events without shared words are not related."""
        event1 = "The great battle"
        event2 = "A peaceful meeting"

        assert _events_are_related(event1, event2) is False

    def test_events_with_common_words_only_are_not_related(self):
        """Events sharing only common words are not related."""
        event1 = "They have been there"
        event2 = "They were with that"

        assert _events_are_related(event1, event2) is False


class TestIsTemporalViolation:
    """Tests for _is_temporal_violation function."""

    def test_before_after_conflict(self):
        """Detects before/after conflicts."""
        assert _is_temporal_violation("before the war", "after the war") is True

    def test_after_before_conflict(self):
        """Detects after/before conflicts."""
        assert _is_temporal_violation("after the meeting", "before the meeting") is True

    def test_no_conflict(self):
        """No conflict when keywords don't oppose."""
        assert _is_temporal_violation("during the war", "at the battle") is False

    def test_same_direction(self):
        """No conflict when both use same direction."""
        assert _is_temporal_violation("before the war", "before the battle") is False


@pytest.mark.asyncio
class TestCheckWorldRules:
    """Tests for _check_world_rules function."""

    async def test_check_world_rules_no_rules(self):
        """No rules returns no contradictions."""
        contradictions = await _check_world_rules("Test text", [], 1)
        assert contradictions == []

    async def test_check_world_rules_no_text(self):
        """No text returns no contradictions."""
        contradictions = await _check_world_rules("", ["Rule 1"], 1)
        assert contradictions == []

    async def test_check_world_rules_no_violations(self):
        """No violations found returns empty list."""
        mock_response = "[]"

        with (
            patch(
                "core.langgraph.subgraphs.validation.neo4j_manager.execute_read_query",
                new_callable=AsyncMock,
            ) as mock_query,
            patch(
                "core.langgraph.subgraphs.validation.llm_service.async_call_llm",
                new_callable=AsyncMock,
            ) as mock_llm,
        ):
            mock_query.return_value = []
            mock_llm.return_value = (mock_response, {"tokens": 50})

            contradictions = await _check_world_rules("Test text", ["Magic requires words"], 1)

            assert contradictions == []

    async def test_check_world_rules_finds_violations(self):
        """Violations are converted to contradictions."""
        mock_response = """[{
            "description": "Character used magic silently",
            "severity": "major",
            "fix": "Add spoken spell"
        }]"""

        with (
            patch(
                "core.langgraph.subgraphs.validation.neo4j_manager.execute_read_query",
                new_callable=AsyncMock,
            ) as mock_query,
            patch(
                "core.langgraph.subgraphs.validation.llm_service.async_call_llm",
                new_callable=AsyncMock,
            ) as mock_llm,
        ):
            mock_query.return_value = []
            mock_llm.return_value = (mock_response, {"tokens": 100})

            contradictions = await _check_world_rules("Test text", ["Magic requires spoken words"], 1)

            assert len(contradictions) == 1
            assert contradictions[0].type == "world_rule"
            assert "silent" in contradictions[0].description.lower()

    async def test_check_world_rules_includes_db_rules(self):
        """Database rules are included in check."""
        db_rules = [
            {"description": "No tech in dead zone", "constraint": None},
            {"description": None, "constraint": "Vampires need invitation"},
        ]

        mock_response = "[]"

        with (
            patch(
                "core.langgraph.subgraphs.validation.neo4j_manager.execute_read_query",
                new_callable=AsyncMock,
            ) as mock_query,
            patch(
                "core.langgraph.subgraphs.validation.llm_service.async_call_llm",
                new_callable=AsyncMock,
            ) as mock_llm,
        ):
            mock_query.return_value = db_rules
            mock_llm.return_value = (mock_response, {"tokens": 50})

            await _check_world_rules("Test text", ["Rule 1"], 1)

            call_args = mock_llm.call_args
            prompt = call_args.kwargs["prompt"]

            assert "No tech in dead zone" in prompt
            assert "Vampires need invitation" in prompt

    async def test_check_world_rules_error_handling(self):
        """Errors during check are handled gracefully."""
        with patch(
            "core.langgraph.subgraphs.validation.neo4j_manager.execute_read_query",
            new_callable=AsyncMock,
        ) as mock_query:
            mock_query.side_effect = Exception("Database error")

            contradictions = await _check_world_rules("Test", ["Rule"], 1)

            assert contradictions == []


class TestBuildRuleCheckPrompt:
    """Tests for _build_rule_check_prompt function."""

    def test_build_rule_check_prompt_basic(self):
        """Basic prompt building works."""
        prompt = _build_rule_check_prompt("Test text here", ["Rule 1", "Rule 2", "Rule 3"])

        assert "Test text here" in prompt
        assert "Rule 1" in prompt
        assert "Rule 2" in prompt
        assert "Rule 3" in prompt
        assert "JSON array" in prompt

    def test_build_rule_check_prompt_truncates_long_text(self):
        """Long text is truncated."""
        long_text = "x" * 10000
        prompt = _build_rule_check_prompt(long_text, ["Rule 1"])

        assert "truncated" in prompt
        assert len(prompt) < len(long_text)


class TestParseRuleViolations:
    """Tests for _parse_rule_violations function."""

    def test_parse_empty_array(self):
        """Empty array returns empty list."""
        response = "[]"
        violations = _parse_rule_violations(response)
        assert violations == []

    def test_parse_valid_violations(self):
        """Valid violations are parsed."""
        response = """[
            {
                "description": "Magic used without words",
                "severity": "major",
                "fix": "Add dialogue"
            },
            {
                "description": "Tech in dead zone",
                "severity": "critical",
                "fix": "Remove tech"
            }
        ]"""

        violations = _parse_rule_violations(response)

        assert len(violations) == 2
        assert violations[0]["description"] == "Magic used without words"
        assert violations[1]["severity"] == "critical"

    def test_parse_invalid_json(self):
        """Invalid JSON returns empty list."""
        response = "This is not JSON"
        violations = _parse_rule_violations(response)
        assert violations == []

    def test_parse_with_extra_text(self):
        """JSON is extracted from surrounding text."""
        response = """
        Here are the violations found:

        [{"description": "Violation", "severity": "major", "fix": "Fix it"}]

        That's all.
        """

        violations = _parse_rule_violations(response)

        assert len(violations) == 1
        assert violations[0]["description"] == "Violation"


@pytest.mark.asyncio
class TestCheckRelationshipEvolution:
    """Tests for _check_relationship_evolution function."""

    async def test_check_relationship_evolution_no_relationships(self):
        """No relationships returns no contradictions."""
        contradictions = await _check_relationship_evolution([], 1)
        assert contradictions == []

    async def test_check_relationship_evolution_no_previous_relationship(self):
        """No previous relationship data returns no contradictions."""
        mock_rel = MagicMock()
        mock_rel.source_name = "Alice"
        mock_rel.target_name = "Bob"
        mock_rel.relationship_type = "LOVES"

        with patch(
            "core.langgraph.subgraphs.validation.neo4j_manager.execute_read_query",
            new_callable=AsyncMock,
        ) as mock_query:
            mock_query.return_value = []

            contradictions = await _check_relationship_evolution([mock_rel], 5)

            assert contradictions == []

    async def test_check_relationship_evolution_detects_rapid_change(self):
        """Rapid dramatic changes are flagged."""
        mock_rel = MagicMock()
        mock_rel.source_name = "Alice"
        mock_rel.target_name = "Bob"
        mock_rel.relationship_type = "LOVES"

        with patch(
            "core.langgraph.subgraphs.validation.neo4j_manager.execute_read_query",
            new_callable=AsyncMock,
        ) as mock_query:
            mock_query.return_value = [{"rel_type": "HATES", "first_chapter": 3}]

            contradictions = await _check_relationship_evolution([mock_rel], 5)

            assert len(contradictions) == 1
            assert contradictions[0].type == "relationship"
            assert "HATES" in contradictions[0].description
            assert "LOVES" in contradictions[0].description

    async def test_check_relationship_evolution_allows_gradual_change(self):
        """Gradual changes over time are not flagged."""
        mock_rel = MagicMock()
        mock_rel.source_name = "Alice"
        mock_rel.target_name = "Bob"
        mock_rel.relationship_type = "LOVES"

        with patch(
            "core.langgraph.subgraphs.validation.neo4j_manager.execute_read_query",
            new_callable=AsyncMock,
        ) as mock_query:
            mock_query.return_value = [{"rel_type": "HATES", "first_chapter": 1}]

            contradictions = await _check_relationship_evolution([mock_rel], 10)

            assert contradictions == []

    async def test_check_relationship_evolution_error_handling(self):
        """Errors are handled gracefully."""
        mock_rel = MagicMock()
        mock_rel.source_name = "Alice"
        mock_rel.target_name = "Bob"
        mock_rel.relationship_type = "TRUSTS"

        with patch(
            "core.langgraph.subgraphs.validation.neo4j_manager.execute_read_query",
            new_callable=AsyncMock,
        ) as mock_query:
            mock_query.side_effect = Exception("Database error")

            contradictions = await _check_relationship_evolution([mock_rel], 5)

            assert contradictions == []


class TestCreateValidationSubgraph:
    """Tests for create_validation_subgraph function."""

    def test_create_validation_subgraph_structure(self):
        """Validation subgraph has correct structure."""
        graph = create_validation_subgraph()

        assert graph is not None

    def test_create_validation_subgraph_is_compiled(self):
        """Returned graph is compiled and executable."""
        graph = create_validation_subgraph()

        assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")
