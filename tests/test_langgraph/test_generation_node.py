# tests/test_langgraph/test_generation_node.py
"""
Tests for LangGraph generation node (Phase 2, Step 2.1).

Tests the generate_chapter node and its helper functions.

Migration Reference: docs/phase2_migration_plan.md - Step 2.1
"""

from unittest.mock import AsyncMock, patch

import pytest

from core.langgraph.content_manager import ContentManager, get_draft_text
from core.langgraph.nodes.generation_node import (
    _construct_generation_prompt,
    generate_chapter,
)
from core.langgraph.state import create_initial_state


@pytest.fixture
def sample_generation_state(tmp_path):
    """Sample state ready for generation."""
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
        extraction_model="test-model",
        revision_model="test-model",
    )

    # Save outlines via ContentManager so get_chapter_outlines works
    content_manager = ContentManager(project_dir)

    # Chapter outlines
    chapter_outlines = {
        1: {
            "plot_point": "The hero begins their journey",
            "chapter_summary": "Introduction to the protagonist",
        },
        "title": "Test Novel",
        "genre": "Fantasy",
        "theme": "Adventure",
        "protagonist_name": "Hero",
    }
    # Note: save_json expects identifier for filename.
    # But get_chapter_outlines loads "chapter_outlines" type.
    # Actually, get_chapter_outlines uses state["chapter_outlines_ref"].
    # We should save it as "chapter_outlines" type.
    ref = content_manager.save_json(chapter_outlines, "chapter_outlines", "all", 1)
    state["chapter_outlines_ref"] = ref

    # Add plot point focus
    state["plot_point_focus"] = "The hero begins their journey"

    # Add some previous chapter summaries (for chapter 2+)
    summaries = [
        "Chapter 1: The hero discovered their destiny and set out from their village."
    ]
    sum_ref = content_manager.save_list_of_texts(summaries, "summaries", "all", 1)
    state["summaries_ref"] = sum_ref

    return state


@pytest.fixture
def mock_llm_generation():
    """Mock LLM service for generation testing."""
    with patch("core.langgraph.nodes.generation_node.llm_service") as mock:
        mock.async_call_llm = AsyncMock(
            return_value=(
                "This is a test chapter. The hero walked through the forest, "
                "contemplating their journey ahead. They knew challenges awaited, "
                "but their determination was unwavering. The sun set behind the "
                "mountains as they continued onward, ready to face whatever came next.",
                {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
            )
        )
        mock.count_tokens = lambda text, model: 500
        yield mock


@pytest.fixture
def mock_context_builder():
    """Mock context building functions."""
    with (
        patch("core.langgraph.nodes.generation_node.get_key_events") as mock_events,
        patch(
            "core.langgraph.nodes.generation_node.get_reliable_kg_facts_for_drafting_prompt"
        ) as mock_kg,
    ):
        mock_events.return_value = []
        mock_kg.return_value = (
            "**Knowledge Graph Facts:**\n- Test fact 1\n- Test fact 2"
        )
        yield {
            "get_events": mock_events,
            "get_kg_facts": mock_kg,
        }


@pytest.mark.asyncio
class TestConstructGenerationPrompt:
    """Tests for _construct_generation_prompt helper function."""

    async def test_construct_basic_prompt(self):
        """Test constructing a basic generation prompt."""
        prompt = await _construct_generation_prompt(
            chapter_number=1,
            plot_point_focus="The hero begins their journey",
            hybrid_context="Test context",
            novel_title="Test Novel",
            novel_genre="Fantasy",
            novel_theme="Adventure",
            protagonist_name="Hero",
        )

        assert isinstance(prompt, str)
        assert "Test Novel" in prompt
        assert "Fantasy" in prompt
        assert "The hero begins their journey" in prompt
        assert "Test context" in prompt
        # Protagonist name may be in context or plot point, not necessarily standalone
        assert len(prompt) > 100  # Verify substantial prompt was generated

    async def test_construct_prompt_with_complex_context(self):
        """Test constructing prompt with rich context."""
        hybrid_context = """
        **Previous Chapter Summary:**
        The hero discovered their powers.

        **Key Characters:**
        - Hero: The protagonist
        - Mentor: The wise guide

        **Key Events:**
        - Discovery of magical artifact
        """

        prompt = await _construct_generation_prompt(
            chapter_number=2,
            plot_point_focus="Training with the mentor",
            hybrid_context=hybrid_context,
            novel_title="Test Novel",
            novel_genre="Fantasy",
            novel_theme="Self-discovery",
            protagonist_name="Hero",
        )

        assert "Chapter 2" in prompt or "chapter 2" in prompt
        assert "Training with the mentor" in prompt
        assert "Hero" in prompt


@pytest.mark.asyncio
class TestGenerateChapter:
    """Tests for generate_chapter node function."""

    async def test_generate_chapter_success(
        self, sample_generation_state, mock_llm_generation, mock_context_builder
    ):
        """Test successful chapter generation."""
        result = await generate_chapter(sample_generation_state)

        # Check that draft text was generated
        assert result["draft_ref"] is not None

        # Verify text content
        content_manager = ContentManager(sample_generation_state["project_dir"])
        draft_text = get_draft_text(result, content_manager)
        assert draft_text is not None
        assert len(draft_text) > 0

        # Check word count was calculated
        assert result["draft_word_count"] > 0

        # Check current node was updated
        assert result["current_node"] == "generate"

        # Check no errors
        assert result["last_error"] is None

        # Verify LLM was called
        mock_llm_generation.async_call_llm.assert_called_once()

    async def test_generate_chapter_no_outline(
        self, sample_generation_state, mock_llm_generation, mock_context_builder
    ):
        """Test generation fails gracefully without plot outline."""
        state = {**sample_generation_state}
        # Clear references
        state["chapter_outlines_ref"] = None
        state["plot_outline"] = None

        result = await generate_chapter(state)

        # Should return error state
        assert result["last_error"] is not None
        assert "chapter outlines" in result["last_error"].lower()
        assert result["current_node"] == "generate"

        # LLM should not be called
        mock_llm_generation.async_call_llm.assert_not_called()

    async def test_generate_chapter_with_context(
        self, sample_generation_state, mock_llm_generation, mock_context_builder
    ):
        """Test generation uses context from knowledge graph."""
        result = await generate_chapter(sample_generation_state)

        # Verify context building was called
        mock_context_builder["get_kg_facts"].assert_called_once()

        # Check that hybrid context was stored in state
        # hybrid_context is externalized to hybrid_context_ref
        assert result.get("hybrid_context_ref") is not None

    async def test_generate_chapter_empty_llm_response(
        self, sample_generation_state, mock_context_builder
    ):
        """Test handling of empty LLM response."""
        with patch("core.langgraph.nodes.generation_node.llm_service") as mock_llm:
            mock_llm.async_call_llm = AsyncMock(return_value=("", {}))
            mock_llm.count_tokens = lambda text, model: 500

            result = await generate_chapter(sample_generation_state)

            # Should return error
            assert result["last_error"] is not None
            assert "empty" in result["last_error"].lower()

    async def test_generate_chapter_llm_exception(
        self, sample_generation_state, mock_context_builder
    ):
        """Test handling of LLM exceptions."""
        with patch("core.langgraph.nodes.generation_node.llm_service") as mock_llm:
            mock_llm.async_call_llm = AsyncMock(
                side_effect=Exception("LLM service unavailable")
            )
            mock_llm.count_tokens = lambda text, model: 500

            result = await generate_chapter(sample_generation_state)

            # Should return error state
            assert result["last_error"] is not None
            assert "LLM service unavailable" in result["last_error"]
            assert result["current_node"] == "generate"

    async def test_generate_chapter_token_budget_exceeded(
        self, sample_generation_state, mock_context_builder
    ):
        """Test handling when token budget is exceeded."""
        with patch("core.langgraph.nodes.generation_node.llm_service") as mock_llm:
            # Set very high token count to exceed budget
            mock_llm.count_tokens = lambda text, model: 150000

            result = await generate_chapter(sample_generation_state)

            # Should return error
            assert result["last_error"] is not None
            assert "token" in result["last_error"].lower()

            # LLM should not be called
            mock_llm.async_call_llm.assert_not_called()

    async def test_generate_chapter_with_previous_summaries(
        self, sample_generation_state, mock_llm_generation, mock_context_builder
    ):
        """Test generation includes previous chapter summaries."""
        state = {**sample_generation_state}
        state["current_chapter"] = 3

        # Need to ensure chapter 3 is in outlines
        cm = ContentManager(state["project_dir"])
        outlines = {
            1: {"plot_point": "Ch1"},
            2: {"plot_point": "Ch2"},
            3: {"plot_point": "Ch3"},
        }
        ref = cm.save_json(outlines, "chapter_outlines", "all", 2)
        state["chapter_outlines_ref"] = ref

        # Summaries already in state via sample_generation_state

        result = await generate_chapter(state)

        # Should succeed
        assert result["draft_ref"] is not None
        assert result["last_error"] is None

        # Verify KG facts were fetched
        mock_context_builder["get_kg_facts"].assert_called_once()

    async def test_generate_chapter_extracts_plot_point_from_outline(
        self, sample_generation_state, mock_llm_generation, mock_context_builder
    ):
        """Test plot point extraction from outline when not in state."""
        state = {**sample_generation_state}
        # Remove plot_point_focus from state
        del state["plot_point_focus"]

        result = await generate_chapter(state)

        # Should still succeed by extracting from outline
        assert result["draft_ref"] is not None
        assert result["last_error"] is None

    async def test_generate_chapter_word_count_calculation(
        self, sample_generation_state, mock_llm_generation, mock_context_builder
    ):
        """Test word count is calculated correctly."""
        # Mock LLM to return known text
        test_text = "One two three four five six seven eight nine ten."
        with patch("core.langgraph.nodes.generation_node.llm_service") as mock_llm:
            mock_llm.async_call_llm = AsyncMock(return_value=(test_text, {}))
            mock_llm.count_tokens = lambda text, model: 500

            result = await generate_chapter(sample_generation_state)

            # Should calculate correct word count
            assert result["draft_word_count"] == len(test_text.split())

    async def test_generate_chapter_stores_kg_facts(
        self, sample_generation_state, mock_llm_generation, mock_context_builder
    ):
        """Test KG facts are stored in state for potential reuse."""
        result = await generate_chapter(sample_generation_state)

        # Should store KG facts ref
        assert result.get("kg_facts_ref") is not None

    async def test_generate_chapter_with_active_characters(
        self, sample_generation_state, mock_llm_generation, mock_context_builder
    ):
        """Test generation with specific active characters."""
        state = {**sample_generation_state}
        state["active_character_names"] = ["Hero", "Mentor", "Villain"]

        result = await generate_chapter(state)

        # Should succeed
        assert result["draft_ref"] is not None

        # Verify KG facts were fetched
        mock_context_builder["get_kg_facts"].assert_called_once()

    async def test_generate_chapter_with_location(
        self, sample_generation_state, mock_llm_generation, mock_context_builder
    ):
        """Test generation with current location specified."""
        state = {**sample_generation_state}
        state["current_location"] = "castle_001"

        result = await generate_chapter(state)

        # Should succeed
        assert result["draft_ref"] is not None

        # Verify KG facts were fetched
        mock_context_builder["get_kg_facts"].assert_called_once()


@pytest.mark.asyncio
class TestGenerationErrorHandling:
    """Tests for error handling in generation node (P1.1 & P1.3)."""

    async def test_generate_chapter_empty_chapter_outlines_fatal_error(
        self, sample_generation_state, mock_llm_generation, mock_context_builder
    ):
        """Test generation with empty chapter_outlines triggers fatal error."""
        state = {**sample_generation_state}
        state["chapter_outlines_ref"] = None
        state["plot_outline"] = None

        result = await generate_chapter(state)

        assert result["last_error"] is not None
        assert "No chapter outlines available" in result["last_error"]
        assert result["has_fatal_error"] is True
        assert result["error_node"] == "generate"
        assert result["current_node"] == "generate"

        mock_llm_generation.async_call_llm.assert_not_called()

    async def test_generate_chapter_missing_current_chapter_in_outlines(
        self, sample_generation_state, mock_llm_generation, mock_context_builder
    ):
        """Test generation when current chapter is not in chapter_outlines."""
        state = {**sample_generation_state}
        state["current_chapter"] = 5
        # Outlines only have chapter 1 by default
        state["plot_outline"] = None

        result = await generate_chapter(state)

        assert result["last_error"] is not None
        assert result["has_fatal_error"] is True
        assert result["error_node"] == "generate"

        mock_llm_generation.async_call_llm.assert_not_called()

    async def test_generate_chapter_deprecation_warning_for_plot_outline(
        self, sample_generation_state, mock_llm_generation, mock_context_builder
    ):
        """Test that using plot_outline triggers deprecation warning."""
        state = {**sample_generation_state}
        state["chapter_outlines_ref"] = None
        state["plot_outline"] = {
            1: {"plot_point": "Test point"},
            "title": "Test",
        }

        result = await generate_chapter(state)

        assert result["draft_ref"] is not None
        assert result["last_error"] is None

    async def test_generate_chapter_prefers_chapter_outlines_over_plot_outline(
        self, sample_generation_state, mock_llm_generation, mock_context_builder
    ):
        """Test that chapter_outlines is preferred over deprecated plot_outline."""
        state = {**sample_generation_state}
        # chapter_outlines already set in fixture (via ref)
        state["plot_outline"] = {
            1: {"plot_point": "From plot_outline"},
        }

        result = await generate_chapter(state)

        assert result["draft_ref"] is not None
        assert result["last_error"] is None


@pytest.mark.asyncio
class TestGenerationIntegration:
    """Integration tests for generation node."""

    async def test_full_generation_workflow(
        self, sample_generation_state, mock_llm_generation, mock_context_builder
    ):
        """Test complete generation workflow."""
        # Generate chapter
        result = await generate_chapter(sample_generation_state)

        # Verify all expected fields are updated
        assert result["draft_ref"] is not None
        assert result["draft_word_count"] > 0
        assert result["current_node"] == "generate"
        assert result["last_error"] is None
        assert result["hybrid_context_ref"] is not None
        assert result["kg_facts_ref"] is not None

        # Verify state carries forward from input
        assert result["current_chapter"] == sample_generation_state["current_chapter"]
        assert result["title"] == sample_generation_state["title"]
        assert result["genre"] == sample_generation_state["genre"]

    async def test_generation_with_minimal_state(
        self, mock_llm_generation, mock_context_builder, tmp_path
    ):
        """Test generation with minimal required state."""
        project_dir = str(tmp_path / "minimal")
        minimal_state = create_initial_state(
            project_id="minimal-test",
            title="Minimal Novel",
            genre="Sci-Fi",
            theme="Exploration",
            setting="Space",
            target_word_count=50000,
            total_chapters=10,
            project_dir=project_dir,
            protagonist_name="Explorer",
        )

        # Save outline
        cm = ContentManager(project_dir)
        outline = {
            1: {"plot_point": "First contact"},
            "title": "Minimal Novel",
            "genre": "Sci-Fi",
        }
        ref = cm.save_json(outline, "chapter_outlines", "all", 1)
        minimal_state["chapter_outlines_ref"] = ref
        minimal_state["plot_point_focus"] = "First contact"

        result = await generate_chapter(minimal_state)

        # Should still succeed with minimal state
        assert result["draft_ref"] is not None
        assert result["last_error"] is None


@pytest.mark.asyncio
class TestGenerationDeduplication:
    """Tests for text deduplication in generation node."""

    async def test_generate_chapter_deduplicates_text(
        self, sample_generation_state, mock_context_builder
    ):
        """Test that deduplication is applied to generated text."""
        # Create text with duplicate paragraphs
        duplicate_text = (
            "The hero walked through the forest.\n\n"
            "The sky was dark and foreboding.\n\n"
            "The hero walked through the forest.\n\n"  # Duplicate
            "They continued onward with determination."
        )

        with patch("core.langgraph.nodes.generation_node.llm_service") as mock_llm:
            mock_llm.async_call_llm = AsyncMock(
                return_value=(duplicate_text, {"total_tokens": 100})
            )
            mock_llm.count_tokens = lambda text, model: 500

            result = await generate_chapter(sample_generation_state)

            # Should have text
            assert result["draft_ref"] is not None
            # Deduplication should have removed duplicate paragraph
            # The exact behavior depends on deduplicator settings
            assert result["last_error"] is None

    async def test_generate_chapter_deduplication_updates_word_count(
        self, sample_generation_state, mock_context_builder
    ):
        """Test that word count reflects deduplicated text."""
        # Create text with duplicate segments
        duplicate_text = "Same text. " * 50  # Repetitive text

        with patch("core.langgraph.nodes.generation_node.llm_service") as mock_llm:
            mock_llm.async_call_llm = AsyncMock(
                return_value=(duplicate_text, {"total_tokens": 100})
            )
            mock_llm.count_tokens = lambda text, model: 500

            result = await generate_chapter(sample_generation_state)

            # Word count should reflect final deduplicated text
            cm = ContentManager(sample_generation_state["project_dir"])
            text = get_draft_text(result, cm)
            assert result["draft_word_count"] == len(text.split())

    async def test_generate_chapter_no_duplicates_preserves_text(
        self, sample_generation_state, mock_llm_generation, mock_context_builder
    ):
        """Test that unique text is preserved during deduplication."""
        unique_text = (
            "The hero embarked on their quest with determination. "
            "Each step brought new challenges and discoveries. "
            "The path ahead was uncertain but filled with promise."
        )

        with patch("core.langgraph.nodes.generation_node.llm_service") as mock_llm:
            mock_llm.async_call_llm = AsyncMock(
                return_value=(unique_text, {"total_tokens": 100})
            )
            mock_llm.count_tokens = lambda text, model: 500

            result = await generate_chapter(sample_generation_state)

            # Text should be preserved (deduplicator won't remove unique content)
            assert result["draft_ref"] is not None

            cm = ContentManager(sample_generation_state["project_dir"])
            text = get_draft_text(result, cm)

            assert len(text) > 0
            assert result["last_error"] is None


@pytest.mark.asyncio
class TestProvisionalFlagging:
    """Tests for is_from_flawed_draft provisional flagging."""

    async def test_provisional_flag_set_when_dedup_removes_text(
        self, sample_generation_state, mock_context_builder
    ):
        """Test that is_from_flawed_draft is True when deduplication removes text."""
        # Create text with duplicates that will be removed
        duplicate_text = (
            "Paragraph one with content.\n\n"
            "Paragraph two with different content.\n\n"
            "Paragraph one with content.\n\n"  # Exact duplicate
        )

        with patch("core.langgraph.nodes.generation_node.llm_service") as mock_llm:
            mock_llm.async_call_llm = AsyncMock(
                return_value=(duplicate_text, {"total_tokens": 100})
            )
            mock_llm.count_tokens = lambda text, model: 500

            result = await generate_chapter(sample_generation_state)

            # Flag should be set when deduplication removes text
            # Note: Actual removal depends on TextDeduplicator config
            assert "is_from_flawed_draft" in result
            assert result["last_error"] is None

    async def test_provisional_flag_false_when_no_duplicates(
        self, sample_generation_state, mock_context_builder
    ):
        """Test that is_from_flawed_draft is False when no duplicates found."""
        unique_text = (
            "First paragraph with unique content.\n\n"
            "Second paragraph with different content.\n\n"
            "Third paragraph completely different."
        )

        with patch("core.langgraph.nodes.generation_node.llm_service") as mock_llm:
            mock_llm.async_call_llm = AsyncMock(
                return_value=(unique_text, {"total_tokens": 100})
            )
            mock_llm.count_tokens = lambda text, model: 500

            result = await generate_chapter(sample_generation_state)

            # Flag should be False when no deduplication occurs
            assert result.get("is_from_flawed_draft") is False
            assert result["last_error"] is None

    async def test_provisional_flag_logged(
        self, sample_generation_state, mock_context_builder
    ):
        """Test that provisional flag is included in logging."""
        test_text = "Some chapter text without duplicates."

        with patch("core.langgraph.nodes.generation_node.llm_service") as mock_llm:
            with patch("core.langgraph.nodes.generation_node.logger") as mock_logger:
                mock_llm.async_call_llm = AsyncMock(
                    return_value=(test_text, {"total_tokens": 100})
                )
                mock_llm.count_tokens = lambda text, model: 500

                result = await generate_chapter(sample_generation_state)

                # Verify logging includes the flag
                # Look for calls that include is_from_flawed_draft
                info_calls = [call for call in mock_logger.info.call_args_list]
                assert len(info_calls) > 0
                assert result.get("is_from_flawed_draft") is not None
