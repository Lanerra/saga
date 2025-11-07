"""
Tests for LangGraph summary node (Phase 2, Step 2.3).

Tests the summarize_chapter node and its helper functions.

Migration Reference: docs/phase2_migration_plan.md - Step 2.3
"""

from unittest.mock import AsyncMock, patch

import pytest

from core.langgraph.nodes.summary_node import (
    _parse_summary_response,
    summarize_chapter,
)
from core.langgraph.state import create_initial_state


@pytest.fixture
def sample_summary_state():
    """Sample state ready for summarization."""
    state = create_initial_state(
        project_id="test-project",
        title="Test Novel",
        genre="Fantasy",
        theme="Adventure",
        setting="Medieval world",
        target_word_count=80000,
        total_chapters=20,
        project_dir="/tmp/test-project",
        protagonist_name="Hero",
        generation_model="test-model",
        extraction_model="test-extraction-model",
        revision_model="test-model",
    )

    # Add finalized chapter text
    state["draft_text"] = """
    The hero embarked on their quest, leaving the village behind.
    Along the forest path, they encountered a wise hermit who warned
    of dangers ahead. The hero accepted a magical amulet for protection
    and pressed onward into the unknown, determined to fulfill their destiny.
    """

    # Set current chapter
    state["current_chapter"] = 1

    # Add some existing summaries
    state["previous_chapter_summaries"] = []

    return state


@pytest.fixture
def mock_llm_summary():
    """Mock LLM service for summary testing."""
    with patch("core.langgraph.nodes.summary_node.llm_service") as mock:
        mock.async_call_llm = AsyncMock(
            return_value=(
                '{"summary": "The hero left the village, met a wise hermit who gave them a protective amulet, and continued their quest into the forest."}',
                {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
            )
        )
        yield mock


@pytest.fixture
def mock_neo4j():
    """Mock Neo4j manager for testing."""
    with patch("core.langgraph.nodes.summary_node.neo4j_manager") as mock:
        mock.execute_write_query = AsyncMock(return_value=None)
        yield mock


class TestParseSummaryResponse:
    """Tests for _parse_summary_response helper function."""

    def test_parse_valid_json_summary(self):
        """Test parsing valid JSON with summary key."""
        response = '{"summary": "This is a test summary."}'
        result = _parse_summary_response(response)
        assert result == "This is a test summary."

    def test_parse_json_with_markdown_wrapper(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        response = '```json\n{"summary": "Markdown wrapped summary."}\n```'
        result = _parse_summary_response(response)
        assert result == "Markdown wrapped summary."

    def test_parse_json_string_value(self):
        """Test parsing when LLM returns just a string."""
        response = '"Just a plain string summary."'
        result = _parse_summary_response(response)
        assert result == "Just a plain string summary."

    def test_parse_plain_text_fallback(self):
        """Test fallback to plain text when JSON parsing fails."""
        response = "This is not JSON, just plain text summary."
        result = _parse_summary_response(response)
        assert result == "This is not JSON, just plain text summary."

    def test_parse_empty_response(self):
        """Test handling of empty response."""
        result = _parse_summary_response("")
        assert result is None

    def test_parse_very_short_response(self):
        """Test handling of very short response."""
        result = _parse_summary_response("short")
        # Too short to be a reasonable summary
        assert result is None

    def test_parse_json_without_summary_key(self):
        """Test handling of JSON without summary key."""
        response = '{"text": "Some other key", "data": "value"}'
        result = _parse_summary_response(response)
        # Should fall back to treating as plain text
        assert result is not None

    def test_parse_whitespace_handling(self):
        """Test that whitespace is properly stripped."""
        response = '  {"summary": "  Summary with spaces  "}  '
        result = _parse_summary_response(response)
        assert result == "Summary with spaces"


@pytest.mark.asyncio
class TestSummarizeChapter:
    """Tests for summarize_chapter node function."""

    async def test_summarize_chapter_success(
        self, sample_summary_state, mock_llm_summary, mock_neo4j
    ):
        """Test successful chapter summarization."""
        result = await summarize_chapter(sample_summary_state)

        # Check that summary was added to state
        assert len(result["previous_chapter_summaries"]) == 1
        assert "hero" in result["previous_chapter_summaries"][0].lower()

        # Check current node was updated
        assert result["current_node"] == "summarize"

        # Verify LLM was called with extraction model
        mock_llm_summary.async_call_llm.assert_called_once()
        call_args = mock_llm_summary.async_call_llm.call_args
        assert call_args.kwargs["model_name"] == "test-extraction-model"

        # Verify Neo4j was called to save summary
        mock_neo4j.execute_write_query.assert_called_once()

    async def test_summarize_chapter_no_draft_text(
        self, sample_summary_state, mock_llm_summary, mock_neo4j
    ):
        """Test summarization skips gracefully without draft text."""
        state = {**sample_summary_state}
        state["draft_text"] = None

        result = await summarize_chapter(state)

        # Should complete without adding summary
        assert len(result.get("previous_chapter_summaries", [])) == 0
        assert result["current_node"] == "summarize"

        # LLM should not be called
        mock_llm_summary.async_call_llm.assert_not_called()

    async def test_summarize_chapter_empty_llm_response(
        self, sample_summary_state, mock_neo4j
    ):
        """Test handling of empty LLM response."""
        with patch("core.langgraph.nodes.summary_node.llm_service") as mock_llm:
            mock_llm.async_call_llm = AsyncMock(return_value=("", {}))

            result = await summarize_chapter(sample_summary_state)

            # Should complete without adding summary
            assert len(result.get("previous_chapter_summaries", [])) == 0
            assert result["current_node"] == "summarize"

    async def test_summarize_chapter_llm_exception(
        self, sample_summary_state, mock_neo4j
    ):
        """Test handling of LLM exceptions."""
        with patch("core.langgraph.nodes.summary_node.llm_service") as mock_llm:
            mock_llm.async_call_llm = AsyncMock(
                side_effect=Exception("LLM service unavailable")
            )

            result = await summarize_chapter(sample_summary_state)

            # Should complete gracefully without summary
            assert len(result.get("previous_chapter_summaries", [])) == 0
            assert result["current_node"] == "summarize"

    async def test_summarize_chapter_neo4j_failure(
        self, sample_summary_state, mock_llm_summary
    ):
        """Test that Neo4j failures don't block workflow."""
        with patch("core.langgraph.nodes.summary_node.neo4j_manager") as mock_neo4j:
            mock_neo4j.execute_write_query = AsyncMock(
                side_effect=Exception("Neo4j unavailable")
            )

            result = await summarize_chapter(sample_summary_state)

            # Should still add summary to state even if Neo4j fails
            assert len(result["previous_chapter_summaries"]) == 1
            assert result["current_node"] == "summarize"

    async def test_summarize_chapter_rolling_window(
        self, sample_summary_state, mock_llm_summary, mock_neo4j
    ):
        """Test that summary rolling window keeps only last 5."""
        # Add 4 existing summaries
        state = {**sample_summary_state}
        state["previous_chapter_summaries"] = [
            "Summary 1",
            "Summary 2",
            "Summary 3",
            "Summary 4",
        ]
        state["current_chapter"] = 5

        result = await summarize_chapter(state)

        # Should have 5 summaries (kept last 4, added new one)
        assert len(result["previous_chapter_summaries"]) == 5

    async def test_summarize_chapter_rolling_window_exceeds_limit(
        self, sample_summary_state, mock_llm_summary, mock_neo4j
    ):
        """Test that rolling window drops oldest when exceeding 5."""
        # Add 5 existing summaries
        state = {**sample_summary_state}
        state["previous_chapter_summaries"] = [
            "Summary 1",
            "Summary 2",
            "Summary 3",
            "Summary 4",
            "Summary 5",
        ]
        state["current_chapter"] = 6

        result = await summarize_chapter(state)

        # Should still have only 5 summaries (dropped oldest)
        assert len(result["previous_chapter_summaries"]) == 5
        # First summary should be dropped
        assert "Summary 1" not in result["previous_chapter_summaries"]
        # New summary should be last
        assert "hero" in result["previous_chapter_summaries"][-1].lower()

    async def test_summarize_chapter_uses_low_temperature(
        self, sample_summary_state, mock_llm_summary, mock_neo4j
    ):
        """Test that summarization uses low temperature for consistency."""
        result = await summarize_chapter(sample_summary_state)

        # Verify temperature is low
        call_args = mock_llm_summary.async_call_llm.call_args
        assert call_args.kwargs["temperature"] == 0.3

    async def test_summarize_chapter_uses_extraction_model(
        self, sample_summary_state, mock_llm_summary, mock_neo4j
    ):
        """Test that summarization uses extraction model (fast model)."""
        result = await summarize_chapter(sample_summary_state)

        # Verify correct model was used
        call_args = mock_llm_summary.async_call_llm.call_args
        assert call_args.kwargs["model_name"] == "test-extraction-model"

    async def test_summarize_chapter_short_max_tokens(
        self, sample_summary_state, mock_llm_summary, mock_neo4j
    ):
        """Test that summarization uses short max_tokens."""
        result = await summarize_chapter(sample_summary_state)

        # Verify short max_tokens for concise summary
        call_args = mock_llm_summary.async_call_llm.call_args
        assert call_args.kwargs["max_tokens"] == 200

    async def test_summarize_chapter_plain_text_fallback(
        self, sample_summary_state, mock_neo4j
    ):
        """Test fallback to plain text when JSON parsing fails."""
        with patch("core.langgraph.nodes.summary_node.llm_service") as mock_llm:
            # Return plain text instead of JSON
            mock_llm.async_call_llm = AsyncMock(
                return_value=(
                    "The hero embarked on a quest and received a magical amulet.",
                    {},
                )
            )

            result = await summarize_chapter(sample_summary_state)

            # Should still work with plain text
            assert len(result["previous_chapter_summaries"]) == 1
            assert "quest" in result["previous_chapter_summaries"][0].lower()

    async def test_summarize_chapter_preserves_other_state(
        self, sample_summary_state, mock_llm_summary, mock_neo4j
    ):
        """Test that summarization preserves other state fields."""
        result = await summarize_chapter(sample_summary_state)

        # Verify all other state is preserved
        assert result["current_chapter"] == sample_summary_state["current_chapter"]
        assert result["title"] == sample_summary_state["title"]
        assert result["genre"] == sample_summary_state["genre"]
        assert result["draft_text"] == sample_summary_state["draft_text"]


@pytest.mark.asyncio
class TestSummaryIntegration:
    """Integration tests for summary node."""

    async def test_full_summarization_workflow(
        self, sample_summary_state, mock_llm_summary, mock_neo4j
    ):
        """Test complete summarization workflow."""
        # Summarize chapter
        result = await summarize_chapter(sample_summary_state)

        # Verify all expected operations occurred
        assert len(result["previous_chapter_summaries"]) == 1
        assert result["current_node"] == "summarize"

        # Verify LLM was called
        mock_llm_summary.async_call_llm.assert_called_once()

        # Verify Neo4j was updated
        mock_neo4j.execute_write_query.assert_called_once()
        call_args = mock_neo4j.execute_write_query.call_args
        assert "Chapter" in call_args.args[0]  # Query contains Chapter
        assert "summary" in call_args.args[0].lower()  # Query sets summary

    async def test_multiple_chapter_summaries(
        self, sample_summary_state, mock_llm_summary, mock_neo4j
    ):
        """Test summarizing multiple chapters builds summary history."""
        state = sample_summary_state

        # Summarize chapter 1
        result1 = await summarize_chapter(state)
        assert len(result1["previous_chapter_summaries"]) == 1

        # Summarize chapter 2
        state2 = {**result1}
        state2["current_chapter"] = 2
        state2["draft_text"] = "Chapter 2 text with new events."
        result2 = await summarize_chapter(state2)
        assert len(result2["previous_chapter_summaries"]) == 2

        # Summarize chapter 3
        state3 = {**result2}
        state3["current_chapter"] = 3
        state3["draft_text"] = "Chapter 3 continues the story."
        result3 = await summarize_chapter(state3)
        assert len(result3["previous_chapter_summaries"]) == 3

    async def test_summary_persistence_to_neo4j(
        self, sample_summary_state, mock_llm_summary, mock_neo4j
    ):
        """Test that summary is correctly persisted to Neo4j."""
        result = await summarize_chapter(sample_summary_state)

        # Verify Neo4j write query was called
        mock_neo4j.execute_write_query.assert_called_once()

        # Check the query parameters
        call_args = mock_neo4j.execute_write_query.call_args
        query = call_args.args[0]
        params = call_args.args[1]

        # Verify query structure
        assert "MERGE (c:Chapter" in query
        assert "c.summary = $summary" in query

        # Verify parameters
        assert params["chapter_number"] == 1
        assert "summary" in params
        assert len(params["summary"]) > 0
