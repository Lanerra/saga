# tests/test_langgraph/test_revision_node.py
"""
Tests for LangGraph revision node (Phase 2, Step 2.2).

Tests the revise_chapter node and its helper functions.

Migration Reference: docs/phase2_migration_plan.md - Step 2.2
"""

from unittest.mock import AsyncMock, patch

import pytest

from core.langgraph.content_manager import ContentManager, get_draft_text
from core.langgraph.nodes.revision_node import (
    _construct_revision_prompt,
    _format_contradictions_for_prompt,
    revise_chapter,
)
from core.langgraph.state import Contradiction, create_initial_state


@pytest.fixture
def sample_revision_state(tmp_path):
    """Sample state ready for revision."""
    project_dir = str(tmp_path / "test-project")
    state = create_initial_state(
        project_id="test-project",
        title="Test Novel",
        genre="Fantasy",
        theme="Adventure",
        setting="Medieval world",
        target_word_count=80000,
        total_chapters=20,
        project_dir=project_dir,
        protagonist_name="Hero",
        generation_model="test-model",
        medium_model="test-model",
        revision_model="test-revision-model",
    )

    # Add draft text that needs revision via ContentManager
    draft_text = """
    The hero walked through the dark forest. The trees were tall.
    Suddenly, a dragon appeared. The hero fought bravely.
    The dragon was defeated. The hero continued walking.
    """
    content_manager = ContentManager(project_dir)
    draft_ref = content_manager.save_text(draft_text, "draft", "chapter_1", 1)

    state["draft_ref"] = draft_ref

    # Add chapter outlines via content manager
    chapter_outlines = {
        1: {
            "plot_point": "The hero begins their journey",
            "chapter_summary": "Introduction to the protagonist",
        }
    }
    chapter_outlines_ref = content_manager.save_json(chapter_outlines, "chapter_outlines", "all", 1)
    state["chapter_outlines_ref"] = chapter_outlines_ref

    # Add contradictions
    state["contradictions"] = [
        Contradiction(
            type="character_trait",
            description="Hero acts cowardly, contradicting established brave trait",
            conflicting_chapters=[1],
            severity="major",
            suggested_fix="Revise to show Hero's courage",
        ),
        Contradiction(
            type="plot_consistency",
            description="Dragon appears without foreshadowing",
            conflicting_chapters=[1],
            severity="minor",
            suggested_fix="Add earlier hints of dragon presence",
        ),
    ]

    # Set iteration tracking
    state["iteration_count"] = 0
    state["max_iterations"] = 3
    state["needs_revision"] = True

    # Add hybrid context
    state["hybrid_context"] = "**Previous context:** Hero is established as brave and honorable."

    return state


@pytest.fixture
def mock_llm_revision():
    """Mock LLM service for revision testing."""
    with patch("core.langgraph.nodes.revision_node.llm_service") as mock:
        mock.async_call_llm = AsyncMock(
            return_value=(
                "The hero strode confidently through the dark forest, hand on sword hilt. "
                "Ancient trees towered overhead, their branches whispering warnings of danger ahead. "
                "When the dragon emerged from its lair, breathing fire, the hero stood firm. "
                "Drawing upon years of training and natural courage, the hero engaged the beast "
                "in fierce combat, ultimately prevailing through skill and bravery. "
                "With the threat vanquished, the hero continued deeper into the forest.",
                {"prompt_tokens": 800, "completion_tokens": 150, "total_tokens": 950},
            )
        )
        mock.count_tokens = lambda text, model: 800
        yield mock


@pytest.fixture
def mock_prompt_data_getters():
    """Mock prompt data getter functions."""
    with patch("core.langgraph.nodes.revision_node.get_reliable_kg_facts_for_drafting_prompt") as mock_kg:
        mock_kg.return_value = "**Knowledge Graph Facts:**\n- Hero is brave\n- Forest is dangerous"
        yield {"get_kg_facts": mock_kg}


class TestFormatContradictionsForPrompt:
    """Tests for _format_contradictions_for_prompt helper function."""

    def test_format_empty_contradictions(self):
        """Test formatting with no contradictions."""
        result = _format_contradictions_for_prompt([])
        assert result == "General quality improvements needed."

    def test_format_single_contradiction(self):
        """Test formatting with a single contradiction."""
        contradictions = [
            Contradiction(
                type="test",
                description="Test issue",
                conflicting_chapters=[1],
                severity="major",
            )
        ]
        result = _format_contradictions_for_prompt(contradictions)
        assert "MAJOR ISSUES:" in result
        assert "Test issue" in result

    def test_format_multiple_contradictions_by_severity(self):
        """Test formatting with multiple contradictions of different severities."""
        contradictions = [
            Contradiction(
                type="critical",
                description="Critical problem",
                conflicting_chapters=[1],
                severity="critical",
            ),
            Contradiction(
                type="major",
                description="Major problem",
                conflicting_chapters=[1],
                severity="major",
            ),
            Contradiction(
                type="minor",
                description="Minor problem",
                conflicting_chapters=[1],
                severity="minor",
            ),
        ]
        result = _format_contradictions_for_prompt(contradictions)

        # Check all severity sections are present
        assert "CRITICAL ISSUES:" in result
        assert "MAJOR ISSUES:" in result
        assert "MINOR ISSUES:" in result

        # Check all descriptions are present
        assert "Critical problem" in result
        assert "Major problem" in result
        assert "Minor problem" in result

    def test_format_preserves_order_by_severity(self):
        """Test that contradictions are grouped by severity (critical first)."""
        contradictions = [
            Contradiction(
                type="minor1",
                description="Minor1",
                conflicting_chapters=[1],
                severity="minor",
            ),
            Contradiction(
                type="critical1",
                description="Critical1",
                conflicting_chapters=[1],
                severity="critical",
            ),
            Contradiction(
                type="major1",
                description="Major1",
                conflicting_chapters=[1],
                severity="major",
            ),
        ]
        result = _format_contradictions_for_prompt(contradictions)

        # Find positions
        critical_pos = result.find("CRITICAL ISSUES:")
        major_pos = result.find("MAJOR ISSUES:")
        minor_pos = result.find("MINOR ISSUES:")

        # Verify order
        assert critical_pos < major_pos < minor_pos


@pytest.mark.asyncio
class TestConstructRevisionPrompt:
    """Tests for _construct_revision_prompt function."""

    async def test_construct_basic_revision_prompt(self, mock_prompt_data_getters):
        """Test constructing a basic revision prompt."""
        contradictions = [
            Contradiction(
                type="test",
                description="Test contradiction",
                conflicting_chapters=[1],
                severity="major",
                suggested_fix="Fix it",
            )
        ]

        prompt = await _construct_revision_prompt(
            draft_text="Original text here",
            contradictions=contradictions,
            chapter_number=1,
            chapter_outlines={1: {"plot_point": "Test point"}},
            hybrid_context="Test context",
            novel_title="Test Novel",
            novel_genre="Fantasy",
            protagonist_name="Hero",
        )

        assert isinstance(prompt, str)
        # Novel title not included in template, but other elements should be
        assert "Fantasy" in prompt
        assert "Test contradiction" in prompt
        assert "Original text here" in prompt
        assert "Test context" in prompt
        assert "Hero" in prompt  # Protagonist name is in template

    async def test_construct_prompt_with_no_hybrid_context(self, mock_prompt_data_getters):
        """Test prompt construction when hybrid context is missing."""
        prompt = await _construct_revision_prompt(
            draft_text="Text",
            contradictions=[],
            chapter_number=1,
            chapter_outlines={},
            hybrid_context=None,
            novel_title="Test",
            novel_genre="Sci-Fi",
            protagonist_name="Protagonist",
        )

        assert isinstance(prompt, str)
        # Should fetch KG facts
        mock_prompt_data_getters["get_kg_facts"].assert_called_once()

    async def test_construct_prompt_does_not_include_length_requirements(self, mock_prompt_data_getters):
        """Prompt construction does not enforce removed length requirements."""
        short_text = "Too short"

        prompt = await _construct_revision_prompt(
            draft_text=short_text,
            contradictions=[],
            chapter_number=1,
            chapter_outlines={},
            hybrid_context="Context",
            novel_title="Test",
            novel_genre="Fantasy",
            protagonist_name="Hero",
        )

        assert isinstance(prompt, str)
        assert short_text in prompt
        assert "LENGTH REQUIREMENT" not in prompt

    async def test_construct_prompt_with_plot_point(self, mock_prompt_data_getters):
        """Test prompt includes plot point focus."""
        prompt = await _construct_revision_prompt(
            draft_text="Text",
            contradictions=[],
            chapter_number=1,
            chapter_outlines={1: {"plot_point": "Hero discovers magic"}},
            hybrid_context="Context",
            novel_title="Test",
            novel_genre="Fantasy",
            protagonist_name="Hero",
        )

        assert "Hero discovers magic" in prompt
        assert "Original Chapter Focus" in prompt


@pytest.mark.asyncio
class TestReviseChapter:
    """Tests for revise_chapter node function."""

    async def test_revise_chapter_success(self, sample_revision_state, mock_llm_revision, mock_prompt_data_getters):
        """Test successful chapter revision."""
        result = await revise_chapter(sample_revision_state)

        # Check that draft text was revised
        cm = ContentManager(sample_revision_state["project_dir"])
        revised_text = get_draft_text(result, cm)
        original_text = get_draft_text(sample_revision_state, cm)

        assert revised_text != original_text
        assert len(revised_text) > 0

        # Check word count was recalculated
        assert result["draft_word_count"] > 0

        # Check iteration count was incremented
        assert result["iteration_count"] == 1

        # Check contradictions were cleared for re-validation
        assert result["contradictions"] == []

        # Check needs_revision flag was reset
        assert result["needs_revision"] is False

        # Check current node was updated
        assert result["current_node"] == "revise"

        # Check no errors
        assert result["last_error"] is None

        # Verify LLM was called
        mock_llm_revision.async_call_llm.assert_called_once()

    async def test_revise_chapter_max_iterations_reached(self, sample_revision_state, mock_llm_revision, mock_prompt_data_getters):
        """Test revision stops when max iterations reached."""
        state = {**sample_revision_state}
        state["iteration_count"] = 3
        state["max_iterations"] = 3

        result = await revise_chapter(state)

        # Should not revise, just return error state
        assert result["needs_revision"] is False
        assert result["last_error"] is not None
        assert "Max revision attempts" in result["last_error"]
        assert result["current_node"] == "revise_failed"

        # LLM should not be called
        mock_llm_revision.async_call_llm.assert_not_called()

    async def test_revise_chapter_no_draft_text(self, sample_revision_state, mock_llm_revision, mock_prompt_data_getters):
        """Test revision fails gracefully without draft text."""
        state = {**sample_revision_state}
        state["draft_ref"] = None

        result = await revise_chapter(state)

        # Should return error state
        assert result["last_error"] is not None
        assert "No draft text" in result["last_error"]
        assert result["current_node"] == "revise"

        # LLM should not be called
        mock_llm_revision.async_call_llm.assert_not_called()

    async def test_revise_chapter_empty_llm_response(self, sample_revision_state, mock_prompt_data_getters):
        """Test handling of empty LLM response."""
        with patch("core.langgraph.nodes.revision_node.llm_service") as mock_llm:
            mock_llm.async_call_llm = AsyncMock(return_value=("", {}))
            mock_llm.count_tokens = lambda text, model: 800

            result = await revise_chapter(sample_revision_state)

            # Should return error and increment iteration
            assert result["last_error"] is not None
            assert "empty" in result["last_error"].lower()
            assert result["has_fatal_error"] is True
            assert result["iteration_count"] == 1

    async def test_revise_chapter_llm_exception(self, sample_revision_state, mock_prompt_data_getters):
        """Test handling of LLM exceptions."""
        with patch("core.langgraph.nodes.revision_node.llm_service") as mock_llm:
            mock_llm.async_call_llm = AsyncMock(side_effect=Exception("LLM service unavailable"))
            mock_llm.count_tokens = lambda text, model: 800

            result = await revise_chapter(sample_revision_state)

            # Should return error state
            assert result["last_error"] is not None
            assert "LLM service unavailable" in result["last_error"]
            assert result["current_node"] == "revise"

    async def test_revise_chapter_token_budget_exceeded(self, sample_revision_state, mock_prompt_data_getters):
        """Test handling when token budget is exceeded."""
        with patch("core.langgraph.nodes.revision_node.llm_service") as mock_llm:
            # Set very high token count to exceed budget
            mock_llm.count_tokens = lambda text, model: 150000

            result = await revise_chapter(sample_revision_state)

            # Should return error
            assert result["last_error"] is not None
            assert "token" in result["last_error"].lower()

            # LLM should not be called
            mock_llm.async_call_llm.assert_not_called()

    async def test_revise_chapter_uses_revision_model(self, sample_revision_state, mock_llm_revision, mock_prompt_data_getters):
        """Test that revision uses the revision_model from state."""
        await revise_chapter(sample_revision_state)

        # Verify correct model was used
        call_args = mock_llm_revision.async_call_llm.call_args
        assert call_args.kwargs["model_name"] == "test-revision-model"

    async def test_revise_chapter_uses_lower_temperature(self, sample_revision_state, mock_llm_revision, mock_prompt_data_getters):
        """Test that revision uses lower temperature for consistency."""
        await revise_chapter(sample_revision_state)

        # Verify temperature is lower than generation default
        call_args = mock_llm_revision.async_call_llm.call_args
        # Should use REVISION temperature (0.5) which is lower than CHAPTER_GENERATION (0.7)
        assert call_args.kwargs["temperature"] <= 0.7

    async def test_revise_chapter_increments_iteration(self, sample_revision_state, mock_llm_revision, mock_prompt_data_getters):
        """Test that iteration count is properly incremented."""
        # Start at iteration 1
        state = {**sample_revision_state}
        state["iteration_count"] = 1

        result = await revise_chapter(state)

        # Should increment to 2
        assert result["iteration_count"] == 2

    async def test_revise_chapter_clears_contradictions(self, sample_revision_state, mock_llm_revision, mock_prompt_data_getters):
        """Test that contradictions are cleared after revision."""
        # Start with contradictions
        assert len(sample_revision_state["contradictions"]) > 0

        result = await revise_chapter(sample_revision_state)

        # Should clear contradictions for re-validation
        assert result["contradictions"] == []

    async def test_revise_chapter_word_count_recalculation(self, sample_revision_state, mock_prompt_data_getters):
        """Test that word count is recalculated for revised text."""
        test_text = "One two three four five six seven eight nine ten."
        with patch("core.langgraph.nodes.revision_node.llm_service") as mock_llm:
            mock_llm.async_call_llm = AsyncMock(return_value=(test_text, {}))
            mock_llm.count_tokens = lambda text, model: 800

            result = await revise_chapter(sample_revision_state)

            # Should calculate correct word count
            assert result["draft_word_count"] == len(test_text.split())

    async def test_revise_chapter_prompt_construction_error(self, sample_revision_state, mock_llm_revision):
        """Test handling of prompt construction errors.

        Note: KG query failures are handled gracefully with a warning,
        so revision continues with "Context unavailable."
        """
        with patch("core.langgraph.nodes.revision_node.get_reliable_kg_facts_for_drafting_prompt") as mock_kg:
            mock_kg.side_effect = Exception("KG query failed")

            # Remove hybrid_context to force KG facts fetch
            state = {**sample_revision_state}
            state["hybrid_context"] = None

            result = await revise_chapter(state)

            # Should handle error gracefully and continue (no last_error)
            # Revision should succeed despite KG fetch failure
            assert result["draft_ref"] is not None
            assert result["last_error"] is None  # Graceful degradation, not failure


@pytest.mark.asyncio
class TestRevisionIntegration:
    """Integration tests for revision node."""

    async def test_full_revision_workflow(self, sample_revision_state, mock_llm_revision, mock_prompt_data_getters):
        """Test complete revision workflow."""
        # Revise chapter
        result = await revise_chapter(sample_revision_state)
        merged = {**sample_revision_state, **result}

        # Verify all expected fields are updated
        assert merged["draft_ref"] is not None
        assert merged["draft_word_count"] > 0
        assert merged["iteration_count"] == 1
        assert merged["contradictions"] == []
        assert merged["needs_revision"] is False
        assert merged["current_node"] == "revise"
        assert merged["last_error"] is None

        # Verify state carries forward from input
        assert merged["current_chapter"] == sample_revision_state["current_chapter"]
        assert merged["title"] == sample_revision_state["title"]
        assert merged["genre"] == sample_revision_state["genre"]

    async def test_multiple_revision_iterations(self, sample_revision_state, mock_llm_revision, mock_prompt_data_getters):
        """Test multiple revision iterations."""
        state = sample_revision_state

        # First revision
        result1 = await revise_chapter(state)
        merged1 = {**state, **result1}
        assert merged1["iteration_count"] == 1

        # Second revision (simulate validation finding more issues)
        state2 = {**merged1}
        state2["contradictions"] = [
            Contradiction(
                type="new_issue",
                description="New problem found",
                conflicting_chapters=[1],
                severity="minor",
            )
        ]
        state2["needs_revision"] = True

        result2 = await revise_chapter(state2)
        merged2 = {**state2, **result2}
        assert merged2["iteration_count"] == 2

        # Third revision
        state3 = {**merged2}
        state3["contradictions"] = [
            Contradiction(
                type="another_issue",
                description="Another problem",
                conflicting_chapters=[1],
                severity="minor",
            )
        ]
        state3["needs_revision"] = True

        result3 = await revise_chapter(state3)
        merged3 = {**state3, **result3}
        assert merged3["iteration_count"] == 3

        # Fourth attempt should hit max_iterations (default 3)
        state4 = {**merged3}
        state4["contradictions"] = [
            Contradiction(
                type="yet_another",
                description="Yet another",
                conflicting_chapters=[1],
                severity="minor",
            )
        ]
        state4["needs_revision"] = True

        result4 = await revise_chapter(state4)
        merged4 = {**state4, **result4}
        assert merged4["current_node"] == "revise_failed"
        assert "Max revision attempts" in merged4["last_error"]

    async def test_revision_with_minimal_contradictions(self, sample_revision_state, mock_llm_revision, mock_prompt_data_getters):
        """Test revision with minimal contradiction information."""
        state = {**sample_revision_state}
        state["contradictions"] = []  # No specific contradictions

        result = await revise_chapter(state)
        merged = {**state, **result}

        # Should still succeed with general quality improvement
        assert merged["draft_ref"] is not None
        assert merged["last_error"] is None


@pytest.mark.asyncio
class TestRevisionErrorHandling:
    """Tests for error handling in revision node (P1.1 & P1.3)."""

    async def test_revise_chapter_max_iterations_fatal_error(self, sample_revision_state, mock_llm_revision, mock_prompt_data_getters):
        """Test revision with max iterations exceeded triggers fatal error."""
        state = {**sample_revision_state}
        state["iteration_count"] = 3
        state["max_iterations"] = 3

        result = await revise_chapter(state)

        assert result["last_error"] is not None
        assert "Max revision attempts" in result["last_error"]
        assert result["has_fatal_error"] is True
        assert result["error_node"] == "revise"
        assert result["needs_revision"] is False
        assert result["current_node"] == "revise_failed"

        mock_llm_revision.async_call_llm.assert_not_called()

    async def test_revise_chapter_missing_draft_text_fatal_error(self, sample_revision_state, mock_llm_revision, mock_prompt_data_getters):
        """Test revision with missing draft_text triggers fatal error."""
        state = {**sample_revision_state}
        state["draft_ref"] = None

        result = await revise_chapter(state)

        assert result["last_error"] is not None
        assert "No draft text available" in result["last_error"]
        assert result["has_fatal_error"] is True
        assert result["error_node"] == "revise"
        assert result["current_node"] == "revise"

        mock_llm_revision.async_call_llm.assert_not_called()

    async def test_revise_chapter_token_budget_exceeded_fatal_error(self, sample_revision_state, mock_prompt_data_getters):
        """Test revision with token budget exceeded triggers fatal error."""
        with patch("core.langgraph.nodes.revision_node.llm_service") as mock_llm:
            mock_llm.count_tokens = lambda text, model: 150000

            result = await revise_chapter(sample_revision_state)

            assert result["last_error"] is not None
            assert "token" in result["last_error"].lower()
            assert result["has_fatal_error"] is True
            assert result["error_node"] == "revise"

            mock_llm.async_call_llm.assert_not_called()

    async def test_revise_chapter_prompt_construction_failure_fatal_error(self, sample_revision_state, mock_llm_revision):
        """Test revision with prompt construction failure triggers fatal error."""
        with patch("core.langgraph.nodes.revision_node._construct_revision_prompt") as mock_prompt:
            mock_prompt.side_effect = Exception("Prompt construction failed")

            state = {**sample_revision_state}
            state["hybrid_context"] = None

            result = await revise_chapter(state)

            assert result["last_error"] is not None
            assert "Prompt construction failed" in result["last_error"]
            assert result["has_fatal_error"] is True
            assert result["error_node"] == "revise"

    async def test_revise_chapter_llm_failure_fatal_error(self, sample_revision_state, mock_prompt_data_getters):
        """Test revision with LLM failure triggers fatal error."""
        with patch("core.langgraph.nodes.revision_node.llm_service") as mock_llm:
            mock_llm.async_call_llm = AsyncMock(side_effect=Exception("LLM service error"))
            mock_llm.count_tokens = lambda text, model: 800

            result = await revise_chapter(sample_revision_state)

            assert result["last_error"] is not None
            assert "LLM service error" in result["last_error"]
            assert result["has_fatal_error"] is True
            assert result["error_node"] == "revise"

    async def test_revise_chapter_empty_llm_response_returns_error(self, sample_revision_state, mock_prompt_data_getters):
        """Test revision with empty LLM response returns error but not fatal."""
        with patch("core.langgraph.nodes.revision_node.llm_service") as mock_llm:
            mock_llm.async_call_llm = AsyncMock(return_value=("", {}))
            mock_llm.count_tokens = lambda text, model: 800

            result = await revise_chapter(sample_revision_state)

            assert result["last_error"] is not None
            assert "empty" in result["last_error"].lower()
            assert result["has_fatal_error"] is True
            assert result["current_node"] == "revise"
            assert result["iteration_count"] == 1
