# tests/test_langgraph/test_finalize_node.py
"""
Tests for LangGraph finalize node (Phase 2, Step 2.4).

Tests the finalize_chapter node and its helper functions.

Migration Reference: docs/phase2_migration_plan.md - Step 2.4
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from core.langgraph.content_manager import ContentManager, get_draft_text
from core.langgraph.nodes.finalize_node import finalize_chapter
from core.langgraph.state import create_initial_state


@pytest.fixture
def sample_finalize_state(tmp_path):
    """Sample state ready for finalization."""
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

    # Add finalized chapter text via ContentManager
    draft_text = """
    The hero completed their quest, returning to the village victorious.
    The villagers celebrated their bravery and the kingdom was saved.
    Peace was restored to the land.
    """
    content_manager = ContentManager(project_dir)
    draft_ref = content_manager.save_text(draft_text, "draft", "chapter_1", 1)

    state["draft_ref"] = draft_ref
    state["draft_word_count"] = 25

    # Set current chapter
    state["current_chapter"] = 1

    # Add extraction results (to be cleaned up)
    state["extracted_entities"] = {
        "characters": [{"name": "Hero", "type": "character"}],
        "locations": [{"name": "Village", "type": "location"}],
    }
    state["extracted_relationships"] = [{"source": "Hero", "target": "Village", "type": "RETURNED_TO"}]

    # Add contradictions (to be cleared)
    state["contradictions"] = []

    # Add iteration tracking
    state["iteration_count"] = 2
    state["needs_revision"] = False

    # Add summary via ContentManager
    summaries = ["The hero completed their quest and returned home victorious."]
    summaries_ref = content_manager.save_list_of_texts(summaries, "summaries", "all", 1)
    state["summaries_ref"] = summaries_ref

    return state


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for embedding generation."""
    with patch("core.langgraph.nodes.finalize_node.llm_service") as mock:
        # Return a mock embedding vector
        mock.async_get_embedding = AsyncMock(return_value=np.random.rand(1024).astype(np.float32))
        yield mock


@pytest.fixture
def mock_save_chapter_data():
    """Mock chapter data saving."""
    with patch("core.langgraph.nodes.finalize_node.save_chapter_data_to_db") as mock:
        mock.return_value = AsyncMock(return_value=None)
        yield mock


@pytest.mark.asyncio
class TestFinalizeChapter:
    """Tests for finalize_chapter node function."""

    async def test_finalize_chapter_success(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test successful chapter finalization."""
        result = await finalize_chapter(sample_finalize_state)

        merged = {**sample_finalize_state, **result}

        # Check that state was cleaned up
        assert merged["extracted_entities"] == {}
        assert merged["extracted_relationships"] == []
        assert merged["contradictions"] == []
        assert merged["iteration_count"] == 0
        assert merged["needs_revision"] is False

        # Check current node was updated
        assert merged["current_node"] == "finalize"

        # Check no errors
        assert merged["last_error"] is None

        # Verify embedding was generated
        mock_llm_service.async_get_embedding.assert_called_once()

        # Verify Neo4j save was called
        mock_save_chapter_data.assert_called_once()
        call_args = mock_save_chapter_data.call_args

        assert call_args.kwargs["chapter_number"] == 1
        # P2.10: save_chapter_data_to_db no longer accepts `text` / `raw_llm_output`;
        # finalization persists summary/embedding/provisional only (text is saved to filesystem + used for embedding).
        assert call_args.kwargs["summary"] is not None
        assert call_args.kwargs["embedding_array"] is not None
        assert call_args.kwargs["is_provisional"] is False

    async def test_finalize_chapter_no_draft_text(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test finalization fails gracefully without draft text."""
        state = {**sample_finalize_state}
        state["draft_ref"] = None

        result = await finalize_chapter(state)
        merged = {**state, **result}

        # Should return error state
        assert merged["last_error"] is not None
        assert "No draft text" in merged["last_error"]
        assert merged["current_node"] == "finalize"

        # Embedding generation should not be called
        mock_llm_service.async_get_embedding.assert_not_called()

        # Neo4j save should not be called
        mock_save_chapter_data.assert_not_called()

    async def test_finalize_chapter_filesystem_save_markdown_and_text(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test that chapter is saved as .md with YAML front matter and .txt mirror."""
        await finalize_chapter(sample_finalize_state)

        project_dir = Path(sample_finalize_state["project_dir"])
        chapters_dir = project_dir / "chapters"

        # .md file assertions
        md_path = chapters_dir / "chapter_001.md"
        assert md_path.exists()

        md_content = md_path.read_text(encoding="utf-8")
        # Expect front matter delimiters
        parts = md_content.split("---")
        assert len(parts) >= 3  # "", front_matter, body... or similar
        front_matter_raw = parts[1].strip()
        body = "---".join(parts[2:]).lstrip("\n")

        import yaml

        cm = ContentManager(sample_finalize_state["project_dir"])
        expected_text = get_draft_text(sample_finalize_state, cm)

        meta = yaml.safe_load(front_matter_raw)
        assert isinstance(meta, dict)
        assert meta["chapter"] == 1
        assert meta["title"] == "Chapter 1"
        assert isinstance(meta["word_count"], int)
        assert meta["word_count"] == len(expected_text.split())
        # generated_at can be str or datetime depending on yaml loader
        import datetime

        assert isinstance(meta["generated_at"], str | datetime.datetime)
        assert meta["generated_at"]
        assert meta["version"] == 1

        # Body must equal original draft text (ignoring leading/trailing whitespace difference)
        assert body.strip() == expected_text.strip()

        # .txt legacy mirror assertions (plain text only)
        txt_path = chapters_dir / "chapter_001.txt"
        assert txt_path.exists()
        assert txt_path.read_text(encoding="utf-8") == expected_text

    async def test_finalize_chapter_embedding_generation(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test that embedding is generated for the chapter."""
        await finalize_chapter(sample_finalize_state)

        cm = ContentManager(sample_finalize_state["project_dir"])
        expected_text = get_draft_text(sample_finalize_state, cm)

        # Verify embedding was requested
        mock_llm_service.async_get_embedding.assert_called_once()
        call_args = mock_llm_service.async_get_embedding.call_args
        assert call_args.args[0] == expected_text

        # Verify embedding was passed to Neo4j
        neo4j_call_args = mock_save_chapter_data.call_args
        embedding = neo4j_call_args.kwargs["embedding_array"]
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)

    async def test_finalize_chapter_embedding_failure(self, sample_finalize_state, mock_save_chapter_data):
        """Test that embedding failures don't block finalization."""
        with patch("core.langgraph.nodes.finalize_node.llm_service") as mock_llm:
            mock_llm.async_get_embedding = AsyncMock(side_effect=Exception("Embedding service unavailable"))

            result = await finalize_chapter(sample_finalize_state)

            # Should still succeed without embedding
            assert result["last_error"] is None
            assert result["current_node"] == "finalize"

            # Neo4j should still be called (with None embedding)
            mock_save_chapter_data.assert_called_once()
            call_args = mock_save_chapter_data.call_args
            assert call_args.kwargs["embedding_array"] is None

    async def test_finalize_chapter_neo4j_failure(self, sample_finalize_state, mock_llm_service):
        """Test that Neo4j failures are reported as errors."""
        with patch("core.langgraph.nodes.finalize_node.save_chapter_data_to_db") as mock_save:
            mock_save.side_effect = Exception("Neo4j unavailable")

            result = await finalize_chapter(sample_finalize_state)

            # Should return error state
            assert result["last_error"] is not None
            assert "Neo4j" in result["last_error"]
            assert result["current_node"] == "finalize"

    async def test_finalize_chapter_filesystem_failure_continues(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test that filesystem failures don't block Neo4j save."""

        with patch("core.langgraph.nodes.finalize_node._save_chapter_to_filesystem") as mock_save_fs:
            mock_save_fs.side_effect = Exception("Filesystem error")

            result = await finalize_chapter(sample_finalize_state)

            # Should still succeed (Neo4j is source of truth)
            assert result["last_error"] is None

            # Neo4j save should still be called
            mock_save_chapter_data.assert_called_once()

    async def test_finalize_chapter_clears_extracted_entities(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test that extracted entities are cleared after finalization."""
        # Add extracted entities
        state = {**sample_finalize_state}
        state["extracted_entities"] = {
            "characters": [{"name": "Hero"}],
            "locations": [{"name": "Castle"}],
        }

        result = await finalize_chapter(state)
        merged = {**state, **result}

        # Should be cleared
        assert result["extracted_entities"] == {}

    async def test_finalize_chapter_clears_relationships(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test that extracted relationships are cleared."""
        result = await finalize_chapter(sample_finalize_state)

        # Should be cleared
        assert result["extracted_relationships"] == []

    async def test_finalize_chapter_clears_contradictions(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test that contradictions are cleared."""
        # Add contradictions
        state = {**sample_finalize_state}
        state["contradictions"] = [{"type": "character", "description": "Test contradiction"}]

        result = await finalize_chapter(state)

        # Should be cleared
        assert result["contradictions"] == []

    async def test_finalize_chapter_resets_iteration_count(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test that iteration count is reset."""
        # Set iteration count
        state = {**sample_finalize_state}
        state["iteration_count"] = 3

        result = await finalize_chapter(state)
        merged = {**state, **result}

        # Should be reset
        assert merged["iteration_count"] == 0

    async def test_finalize_chapter_resets_needs_revision(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test that needs_revision flag is reset."""
        # Set needs_revision
        state = {**sample_finalize_state}
        state["needs_revision"] = True

        result = await finalize_chapter(state)
        merged = {**state, **result}

        # Should be reset
        assert merged["needs_revision"] is False

    async def test_finalize_chapter_preserves_draft_text(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test that draft_text is preserved in state."""
        result = await finalize_chapter(sample_finalize_state)
        merged = {**sample_finalize_state, **result}

        # Draft text should still be available
        assert merged["draft_ref"] == sample_finalize_state["draft_ref"]
        assert merged["draft_word_count"] == sample_finalize_state["draft_word_count"]

    async def test_finalize_chapter_includes_summary(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test that summary is included in Neo4j save."""
        await finalize_chapter(sample_finalize_state)

        # Verify summary was passed to Neo4j
        call_args = mock_save_chapter_data.call_args
        summary = call_args.kwargs["summary"]
        assert summary is not None

        cm = ContentManager(sample_finalize_state["project_dir"])
        # We need to manually load summaries to verify
        # But get_previous_summaries is used in node
        # Let's rely on internal implementation correctness or verify logic:
        # summary is last element of summaries_ref list
        from core.langgraph.content_manager import get_previous_summaries

        expected_summary = get_previous_summaries(sample_finalize_state, cm)[-1]
        assert summary == expected_summary

    async def test_finalize_chapter_no_summary(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test finalization works without summary."""
        state = {**sample_finalize_state}
        state["summaries_ref"] = None

        result = await finalize_chapter(state)
        merged = {**state, **result}

        # Should succeed
        assert merged["last_error"] is None

        # Neo4j should be called with None summary
        call_args = mock_save_chapter_data.call_args
        assert call_args.kwargs["summary"] is None

    async def test_finalize_chapter_creates_chapters_directory(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test that chapters directory is created if it doesn't exist."""
        # Use a new temporary directory
        project_dir = Path(sample_finalize_state["project_dir"])
        chapters_dir = project_dir / "chapters"

        # Ensure directory doesn't exist
        if chapters_dir.exists():
            import shutil

            shutil.rmtree(chapters_dir)

        await finalize_chapter(sample_finalize_state)

        # Directory should be created
        assert chapters_dir.exists()
        assert chapters_dir.is_dir()

    async def test_finalize_chapter_correct_filename_format(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test that chapter filenames use zero-padded format for .md and .txt."""
        for chapter_num in [1, 10, 99]:
            state = {**sample_finalize_state}
            state["current_chapter"] = chapter_num

            await finalize_chapter(state)

            project_dir = Path(state["project_dir"])
            chapters_dir = project_dir / "chapters"

            md_expected = chapters_dir / f"chapter_{chapter_num:03d}.md"
            txt_expected = chapters_dir / f"chapter_{chapter_num:03d}.txt"

            assert md_expected.exists()
            assert txt_expected.exists()

    async def test_finalize_chapter_preserves_other_state(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test that other state fields are preserved."""
        result = await finalize_chapter(sample_finalize_state)
        merged = {**sample_finalize_state, **result}

        # Verify important fields are preserved
        assert merged["current_chapter"] == sample_finalize_state["current_chapter"]
        assert merged["title"] == sample_finalize_state["title"]
        assert merged["genre"] == sample_finalize_state["genre"]
        assert merged["project_id"] == sample_finalize_state["project_id"]


@pytest.mark.asyncio
class TestFinalizeErrorHandling:
    """Tests for error handling in finalize node (P1.1 & P1.3)."""

    async def test_finalize_chapter_missing_draft_text_fatal_error(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test finalization with missing draft_text triggers fatal error."""
        state = {**sample_finalize_state}
        state["draft_ref"] = None

        result = await finalize_chapter(state)

        assert result["last_error"] is not None
        assert "No draft text available" in result["last_error"]
        assert result["has_fatal_error"] is True
        assert result["error_node"] == "finalize"
        assert result["current_node"] == "finalize"

        mock_llm_service.async_get_embedding.assert_not_called()
        mock_save_chapter_data.assert_not_called()

    async def test_finalize_chapter_empty_draft_text_fatal_error(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test finalization with empty draft_text triggers fatal error."""
        # Create empty draft file
        cm = ContentManager(sample_finalize_state["project_dir"])
        ref = cm.save_text("", "draft", "chapter_empty", 1)

        state = {**sample_finalize_state}
        state["draft_ref"] = ref

        result = await finalize_chapter(state)

        assert result["last_error"] is not None
        assert "No draft text available" in result["last_error"]
        assert result["has_fatal_error"] is True
        assert result["error_node"] == "finalize"

        mock_llm_service.async_get_embedding.assert_not_called()
        mock_save_chapter_data.assert_not_called()

    async def test_finalize_chapter_neo4j_failure_fatal_error(self, sample_finalize_state, mock_llm_service):
        """Test finalization with Neo4j failure triggers fatal error."""
        with patch("core.langgraph.nodes.finalize_node.save_chapter_data_to_db") as mock_save:
            mock_save.side_effect = Exception("Neo4j connection failed")

            result = await finalize_chapter(sample_finalize_state)

            assert result["last_error"] is not None
            assert "Neo4j" in result["last_error"]
            assert result["has_fatal_error"] is True
            assert result["error_node"] == "finalize"
            assert result["current_node"] == "finalize"

    async def test_finalize_chapter_embedding_failure_continues_gracefully(self, sample_finalize_state, mock_save_chapter_data):
        """Test that embedding failures don't trigger fatal error."""
        with patch("core.langgraph.nodes.finalize_node.llm_service") as mock_llm:
            mock_llm.async_get_embedding = AsyncMock(side_effect=Exception("Embedding service down"))

            result = await finalize_chapter(sample_finalize_state)

            assert result["last_error"] is None
            assert result.get("has_fatal_error", False) is False
            assert result["current_node"] == "finalize"

            mock_save_chapter_data.assert_called_once()
            call_args = mock_save_chapter_data.call_args
            assert call_args.kwargs["embedding_array"] is None


@pytest.mark.asyncio
class TestFinalizeIntegration:
    """Integration tests for finalize node."""

    async def test_full_finalization_workflow(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test complete finalization workflow."""
        # Finalize chapter
        result = await finalize_chapter(sample_finalize_state)

        # Verify all expected operations occurred
        assert result["current_node"] == "finalize"
        assert result["last_error"] is None

        # Verify cleanup
        assert result["extracted_entities"] == {}
        assert result["extracted_relationships"] == []
        assert result["contradictions"] == []
        assert result["iteration_count"] == 0
        assert result["needs_revision"] is False

        # Verify embedding was generated
        mock_llm_service.async_get_embedding.assert_called_once()

        # Verify Neo4j was updated
        mock_save_chapter_data.assert_called_once()

        # Verify filesystem save of canonical .md and legacy .txt
        project_dir = Path(sample_finalize_state["project_dir"])
        chapters_dir = project_dir / "chapters"
        assert (chapters_dir / "chapter_001.md").exists()
        assert (chapters_dir / "chapter_001.txt").exists()

    async def test_finalization_ready_for_next_chapter(self, sample_finalize_state, mock_llm_service, mock_save_chapter_data):
        """Test that state is ready for next chapter after finalization."""
        result = await finalize_chapter(sample_finalize_state)
        merged = {**sample_finalize_state, **result}

        # State should be clean for next chapter
        assert merged["iteration_count"] == 0
        assert merged["needs_revision"] is False
        assert merged["extracted_entities"] == {}
        assert merged["extracted_relationships"] == []
        assert merged["contradictions"] == []

        # Summary ref should be preserved
        assert merged["summaries_ref"] is not None
        # We can verify content if needed, but presence is enough for now
