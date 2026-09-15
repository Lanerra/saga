# tests/core/langgraph/subgraphs/test_validation_subgraph.py
"""
Tests for validation subgraph.

Covers all functions in core/langgraph/subgraphs/validation.py.
"""

from pathlib import Path
from typing import Any, Never
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.langgraph.state import Contradiction
from core.langgraph.subgraphs.validation import (
    _build_quality_evaluation_prompt,
    _check_relationship_evolution,
    _parse_quality_scores,
    create_validation_subgraph,
    detect_contradictions,
    evaluate_quality,
    validate_consistency,
)
from tests.fakes.service_context import patch_service
from tests.test_langgraph import InlineExtractionState


class ValidationFixtureState(InlineExtractionState, total=False):
    """Keep the fixture's empty legacy metadata outside the production state contract."""

    character_profiles: list[Never]
    previous_summaries: list[Never]
    chapter_outlines: dict[Never, Never]


@pytest.fixture
def sample_validation_state(tmp_path: Path) -> ValidationFixtureState:
    """Create a sample state for validation testing."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    from core.langgraph.content_manager import ContentManager

    content_manager = ContentManager(str(project_dir))
    draft_text = "This is a test chapter with some narrative content."
    draft_ref = content_manager.save_text(draft_text, "draft", "chapter_1", 1)

    return {
        "project_dir": str(project_dir),
        "current_chapter": 1,
        "genre": "fantasy",
        "theme": "courage",
        "draft_word_count": 10,
        "draft_ref": draft_ref,
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

    async def test_validate_consistency_calls_original(self, sample_validation_state: InlineExtractionState) -> None:
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

    async def test_evaluate_quality_no_draft_text(self, sample_validation_state: InlineExtractionState) -> None:
        """When no draft text, returns None scores."""
        state = sample_validation_state.copy()
        state["draft_ref"] = None

        result = await evaluate_quality(state)

        assert result["coherence_score"] is None
        assert result["prose_quality_score"] is None
        assert result["plot_advancement_score"] is None
        assert result["pacing_score"] is None
        assert result["tone_consistency_score"] is None
        assert result["quality_feedback"] is not None
        assert "No draft text" in result["quality_feedback"]

    async def test_evaluate_quality_success(self, sample_validation_state: InlineExtractionState) -> None:
        """Successful quality evaluation returns scores."""
        mock_response = """{
            "coherence_score": 0.85,
            "prose_quality_score": 0.75,
            "plot_advancement_score": 0.80,
            "pacing_score": 0.70,
            "tone_consistency_score": 0.90,
            "feedback": "Good pacing and coherence."
        }"""

        with patch_service(
            'language_model.async_call_llm',
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.return_value = (mock_response, {"tokens": 100})

            result = await evaluate_quality(sample_validation_state)

            assert result["coherence_score"] == 0.85
            assert result["prose_quality_score"] == 0.75
            assert result["plot_advancement_score"] == 0.80
            assert result["pacing_score"] == 0.70
            assert result["tone_consistency_score"] == 0.90
            assert result["quality_feedback"] is not None
            assert "Good pacing" in result["quality_feedback"]

    async def test_evaluate_quality_below_threshold_triggers_contradiction(self, sample_validation_state: InlineExtractionState) -> None:
        """Low quality scores trigger quality contradiction."""
        mock_response = """{
            "coherence_score": 0.3,
            "prose_quality_score": 0.4,
            "plot_advancement_score": 0.2,
            "pacing_score": 0.5,
            "tone_consistency_score": 0.6,
            "feedback": "Needs improvement."
        }"""

        with patch_service(
            'language_model.async_call_llm',
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.return_value = (mock_response, {"tokens": 100})

            result = await evaluate_quality(sample_validation_state)

            quality_contradictions = [c for c in result["contradictions"] if c.type == "quality_issue"]
            assert len(quality_contradictions) == 1
            assert "below threshold" in quality_contradictions[0].description

    async def test_evaluate_quality_llm_error(self, sample_validation_state: InlineExtractionState) -> None:
        """LLM error during evaluation returns None scores."""
        with patch_service(
            'language_model.async_call_llm',
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.side_effect = Exception("LLM connection failed")

            result = await evaluate_quality(sample_validation_state)

            assert result["coherence_score"] is None
            assert result["prose_quality_score"] is None
            assert result["quality_feedback"] is not None
            assert "Evaluation failed" in result["quality_feedback"]


class TestBuildQualityEvaluationPrompt:
    """Tests for _build_quality_evaluation_prompt function."""

    def test_build_prompt_basic(self) -> None:
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

    def test_build_prompt_truncates_long_text(self) -> None:
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

    def test_build_prompt_with_summaries(self) -> None:
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

    def test_build_prompt_with_outline(self) -> None:
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

    def test_parse_valid_json(self) -> None:
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

    def test_parse_numeric_values_does_not_raise_typeerror(self) -> None:
        """
        Regression test for runtime crash when using PEP-604 unions inside isinstance().
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

    def test_parse_rejects_out_of_range_scores(self) -> None:
        response = """{
            "coherence_score": 1.5,
            "prose_quality_score": -0.2,
            "plot_advancement_score": 0.5,
            "pacing_score": 0.7,
            "tone_consistency_score": 0.9,
            "feedback": "Test"
        }"""

        with pytest.raises(ValueError, match="finite numbers"):
            _parse_quality_scores(response)

    def test_parse_invalid_json_rejects(self) -> None:
        response = "This is not valid JSON at all"

        with pytest.raises(ValueError, match="all five scores"):
            _parse_quality_scores(response)

    def test_parse_rejects_text_without_score_schema(self) -> None:
        response = """
        The coherence score is 0.85 and the prose quality is 0.72.
        Pacing score: 0.68
        Plot advancement: 0.91
        Tone consistency: 0.77
        """

        with pytest.raises(ValueError, match="all five scores"):
            _parse_quality_scores(response)


@pytest.mark.asyncio
class TestDetectContradictions:
    """Tests for detect_contradictions function."""

    async def test_detect_contradictions_basic(self, sample_validation_state: InlineExtractionState) -> None:
        """Basic contradiction detection runs."""
        with (
            patch(
                "core.langgraph.subgraphs.validation._check_relationship_evolution",
                new_callable=AsyncMock,
            ) as mock_relationships,
            patch("core.langgraph.subgraphs.validation._fetch_validation_data", new_callable=AsyncMock) as mock_fetch,
        ):
            mock_relationships.return_value = []
            mock_fetch.return_value = {"relationships": {}}

            result = await detect_contradictions(sample_validation_state)

            assert result["current_node"] == "detect_contradictions"
            assert result["needs_revision"] is False
            mock_relationships.assert_called_once()

    async def test_detect_contradictions_critical_triggers_revision(self, sample_validation_state: InlineExtractionState) -> None:
        """Critical contradictions trigger needs_revision."""
        critical_contradiction = Contradiction(
            type="relationship",
            description="Critical relationship violation",
            conflicting_chapters=[1, 2],
            severity="critical",
            suggested_fix="Fix relationship",
        )

        with (
            patch(
                "core.langgraph.subgraphs.validation._check_relationship_evolution",
                new_callable=AsyncMock,
            ) as mock_relationships,
            patch("core.langgraph.subgraphs.validation._fetch_validation_data", new_callable=AsyncMock) as mock_fetch,
        ):
            mock_relationships.return_value = [critical_contradiction]
            mock_fetch.return_value = {"relationships": {}}

            result = await detect_contradictions(sample_validation_state)

            assert result["needs_revision"] is True
            assert len(result["contradictions"]) == 1

    async def test_detect_contradictions_multiple_major_triggers_revision(self, sample_validation_state: InlineExtractionState) -> None:
        """Multiple major contradictions trigger revision."""
        major_contradictions = [
            Contradiction(
                type="relationship",
                description="First major issue",
                conflicting_chapters=[1],
                severity="major",
                suggested_fix="Fix 1",
            ),
            Contradiction(
                type="relationship",
                description="Second major issue",
                conflicting_chapters=[1],
                severity="major",
                suggested_fix="Fix 2",
            ),
            Contradiction(
                type="relationship",
                description="Third major issue",
                conflicting_chapters=[1],
                severity="major",
                suggested_fix="Fix 3",
            ),
        ]

        with (
            patch(
                "core.langgraph.subgraphs.validation._check_relationship_evolution",
                new_callable=AsyncMock,
            ) as mock_relationships,
            patch("core.langgraph.subgraphs.validation._fetch_validation_data", new_callable=AsyncMock) as mock_fetch,
        ):
            mock_relationships.return_value = major_contradictions
            mock_fetch.return_value = {"relationships": {}}

            result = await detect_contradictions(sample_validation_state)

            assert result["needs_revision"] is True

    async def test_detect_contradictions_force_continue_bypasses(self, sample_validation_state: InlineExtractionState) -> None:
        """force_continue bypasses revision even with contradictions."""
        state = sample_validation_state.copy()
        state["force_continue"] = True

        critical_contradiction = Contradiction(
            type="relationship",
            description="Critical issue",
            conflicting_chapters=[1],
            severity="critical",
            suggested_fix="Fix it",
        )

        with (
            patch(
                "core.langgraph.subgraphs.validation._check_relationship_evolution",
                new_callable=AsyncMock,
            ) as mock_relationships,
            patch("core.langgraph.subgraphs.validation._fetch_validation_data", new_callable=AsyncMock) as mock_fetch,
        ):
            mock_relationships.return_value = [critical_contradiction]
            mock_fetch.return_value = {"relationships": {}}

            result = await detect_contradictions(state)

            assert result["needs_revision"] is False

    async def test_detect_contradictions_at_max_iterations_accepts_best_effort(self, sample_validation_state: InlineExtractionState) -> None:
        """At max iterations with remaining issues, accept best-effort draft instead of fatal error."""
        state = sample_validation_state.copy()
        state["iteration_count"] = 3
        state["max_iterations"] = 3

        critical_contradiction = Contradiction(
            type="relationship",
            description="Critical relationship violation on final iteration",
            conflicting_chapters=[1, 2],
            severity="critical",
            suggested_fix="Fix relationship",
        )

        with (
            patch(
                "core.langgraph.subgraphs.validation._check_relationship_evolution",
                new_callable=AsyncMock,
            ) as mock_relationships,
            patch("core.langgraph.subgraphs.validation._fetch_validation_data", new_callable=AsyncMock) as mock_fetch,
        ):
            mock_relationships.return_value = [critical_contradiction]
            mock_fetch.return_value = {"relationships": {}}

            result = await detect_contradictions(state)

            assert "has_fatal_error" not in result
            assert result["needs_revision"] is False
            assert result["contradictions"] == [critical_contradiction]


@pytest.mark.asyncio
class TestCheckRelationshipEvolution:
    """Tests for _check_relationship_evolution function."""

    async def test_check_relationship_evolution_no_relationships(self) -> None:
        """No relationships returns no contradictions."""
        contradictions = await _check_relationship_evolution([], 1, {})
        assert contradictions == []

    async def test_check_relationship_evolution_no_previous_relationship(self) -> None:
        """No previous relationship data returns no contradictions."""
        mock_rel = MagicMock()
        mock_rel.source_name = "Alice"
        mock_rel.target_name = "Bob"
        mock_rel.relationship_type = "LOVES"

        contradictions = await _check_relationship_evolution([mock_rel], 5, {})
        assert contradictions == []

    async def test_check_relationship_evolution_detects_rapid_change(self) -> None:
        """Rapid dramatic changes are flagged."""
        mock_rel = MagicMock()
        mock_rel.source_name = "Alice"
        mock_rel.target_name = "Bob"
        mock_rel.relationship_type = "LOVES"

        contradictions = await _check_relationship_evolution([mock_rel], 5, {("Alice", "Bob"): [{"rel_type": "HATES", "first_chapter": 3}]})
        assert len(contradictions) == 1
        assert contradictions[0].type == "relationship"
        assert "HATES" in contradictions[0].description
        assert "LOVES" in contradictions[0].description

    async def test_check_relationship_evolution_allows_gradual_change(self) -> None:
        """Gradual changes over time are not flagged."""
        mock_rel = MagicMock()
        mock_rel.source_name = "Alice"
        mock_rel.target_name = "Bob"
        mock_rel.relationship_type = "LOVES"

        contradictions = await _check_relationship_evolution([mock_rel], 10, {("Alice", "Bob"): [{"rel_type": "HATES", "first_chapter": 1}]})
        assert contradictions == []

    async def test_check_relationship_evolution_error_handling(self) -> None:
        """A missing history argument cannot trigger an unfiltered fallback read."""
        mock_rel = MagicMock()
        mock_rel.source_name = "Alice"
        mock_rel.target_name = "Bob"
        mock_rel.relationship_type = "TRUSTS"

        with pytest.raises(TypeError, match="existing_relationships"):
            check: Any = _check_relationship_evolution
            await check([mock_rel], 5)


class TestCreateValidationSubgraph:
    """Tests for create_validation_subgraph function."""

    def test_create_validation_subgraph_structure(self) -> None:
        """Validation subgraph has correct structure."""
        graph = create_validation_subgraph()

        assert graph is not None

    def test_create_validation_subgraph_is_compiled(self) -> None:
        """Returned graph is compiled and executable."""
        graph = create_validation_subgraph()

        assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")
