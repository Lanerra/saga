# tests/test_langgraph/test_chapter_loop.py
"""Tests for the internal chapter loop in the LangGraph workflow."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.langgraph.state import NarrativeState, create_initial_state
from core.langgraph.workflow import (
    advance_chapter,
    create_full_workflow_graph,
    should_continue_to_next_chapter,
)


@pytest.fixture
def sample_state(tmp_path: Any) -> NarrativeState:
    """Sample state for testing the chapter loop."""
    state = create_initial_state(
        project_id="test-project",
        title="Test Novel",
        genre="Fantasy",
        theme="Adventure",
        setting="Medieval world",
        target_word_count=80000,
        total_chapters=2,
        project_dir=str(tmp_path / "test-project"),
        protagonist_name="Hero",
    )
    state["initialization_complete"] = True
    return state


def test_advance_chapter_increments_and_resets(sample_state: NarrativeState) -> None:
    """Verify that advance_chapter increments the chapter and resets flags."""
    # Set some flags that should be reset
    sample_state["current_chapter"] = 1
    sample_state["iteration_count"] = 2
    sample_state["needs_revision"] = True
    sample_state["draft_text"] = "Some draft text"
    sample_state["has_fatal_error"] = True

    updated_state = advance_chapter(sample_state)

    assert updated_state["current_chapter"] == 2
    assert updated_state["iteration_count"] == 0
    assert updated_state["needs_revision"] is False
    assert updated_state["draft_text"] is None
    assert updated_state["has_fatal_error"] is False


def test_should_continue_to_next_chapter_logic(sample_state: NarrativeState) -> None:
    """Verify the routing logic for should_continue_to_next_chapter."""
    # Case 1: More chapters remain
    sample_state["current_chapter"] = 1
    sample_state["total_chapters"] = 2
    assert should_continue_to_next_chapter(sample_state) == "continue"

    # Case 2: All chapters complete
    sample_state["current_chapter"] = 2
    sample_state["total_chapters"] = 2
    assert should_continue_to_next_chapter(sample_state) == "end"

    # Case 3: Fatal error
    sample_state["has_fatal_error"] = True
    assert should_continue_to_next_chapter(sample_state) == "error"


@pytest.mark.asyncio
async def test_workflow_loops_to_next_chapter(sample_state: NarrativeState) -> None:
    """Verify that the full workflow graph loops back to chapter_outline when continuing."""
    sample_state["current_chapter"] = 1
    sample_state["total_chapters"] = 2

    # Mock all nodes
    mock_chapter_outline = MagicMock(side_effect=lambda s: {**s, "current_node": "chapter_outline"})
    mock_generate = MagicMock(side_effect=lambda s: {**s, "current_node": "generate"})
    mock_gen_embedding = MagicMock(side_effect=lambda s: {**s, "current_node": "gen_embedding"})
    mock_extract = MagicMock(side_effect=lambda s: {**s, "current_node": "extract"})
    mock_gen_scene_embeddings = MagicMock(side_effect=lambda s: {**s, "current_node": "gen_scene_embeddings"})
    mock_assemble_chapter = MagicMock(side_effect=lambda s: {**s, "draft_ref": {"path": "mock_draft"}, "draft_word_count": 1, "current_node": "assemble_chapter"})
    mock_normalize = MagicMock(side_effect=lambda s: {**s, "current_node": "normalize_relationships"})
    mock_commit = MagicMock(side_effect=lambda s: {**s, "current_node": "commit"})
    mock_validate = MagicMock(side_effect=lambda s: {**s, "current_node": "validate", "needs_revision": False})
    mock_summarize = MagicMock(side_effect=lambda s: {**s, "current_node": "summarize"})
    mock_finalize = MagicMock(side_effect=lambda s: {**s, "current_node": "finalize"})
    mock_heal = MagicMock(side_effect=lambda s: {**s, "current_node": "heal_graph"})
    mock_quality = MagicMock(side_effect=lambda s: {**s, "current_node": "check_quality"})

    with (
        patch("core.langgraph.initialization.generate_chapter_outline", mock_chapter_outline),
        patch("core.langgraph.subgraphs.generation.create_generation_subgraph", return_value=mock_generate),
        patch("core.langgraph.workflow.generate_embedding", mock_gen_embedding),
        patch("core.langgraph.subgraphs.scene_extraction.create_scene_extraction_subgraph", return_value=mock_extract),
        patch("core.langgraph.workflow.generate_scene_embeddings", mock_gen_scene_embeddings),
        patch("core.langgraph.workflow.assemble_chapter", mock_assemble_chapter),
        patch("core.langgraph.workflow.normalize_relationships", mock_normalize),
        patch("core.langgraph.workflow.commit_to_graph", mock_commit),
        patch("core.langgraph.subgraphs.validation.create_validation_subgraph", return_value=mock_validate),
        patch("core.langgraph.workflow.summarize_chapter", mock_summarize),
        patch("core.langgraph.workflow.finalize_chapter", mock_finalize),
        patch("core.langgraph.workflow.heal_graph", mock_heal),
        patch("core.langgraph.workflow.check_quality", mock_quality),
    ):
        graph = create_full_workflow_graph()

        # To avoid an infinite loop in the test (since we want to verify it loops ONCE),
        # we can use recursion limit or just inspect calls.
        # But LangGraph will keep going.

        # We'll mock mock_chapter_outline to stop after the second call (second chapter)
        # to avoid infinite loop if total_chapters was larger.
        # Since total_chapters is 2, it should go:
        # Chapter 1: ... -> check_quality -> advance_chapter -> chapter_outline (call 2)
        # Chapter 2: ... -> check_quality -> END

        result = await graph.ainvoke(sample_state, config={"recursion_limit": 300})

        assert result["current_chapter"] == 2
        assert mock_chapter_outline.call_count == 2
        assert mock_quality.call_count == 2
        assert mock_generate.call_count == 2
