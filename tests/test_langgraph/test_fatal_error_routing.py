# tests/test_langgraph/test_fatal_error_routing.py
"""Tests for fatal error routing in Phase 2 LangGraph workflow."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.langgraph.state import NarrativeState, create_initial_state
from core.langgraph.workflow import create_full_workflow_graph


@pytest.fixture
def sample_generation_state(tmp_path: Any) -> NarrativeState:
    """Sample state ready for the chapter-generation portion of the workflow."""
    state = create_initial_state(
        project_id="test-project",
        title="Test Novel",
        genre="Fantasy",
        theme="Adventure",
        setting="Medieval world",
        target_word_count=80000,
        total_chapters=1,
        project_dir=str(tmp_path / "test-project"),
        protagonist_name="Hero",
        generation_model="test-model",
        extraction_model="test-model",
        revision_model="test-model",
    )
    state["current_chapter"] = 1
    state["initialization_complete"] = True

    # Add mock chapter outlines ref
    state["chapter_outlines_ref"] = {
        "path": "mock_outlines.json",
        "size_bytes": 100,
        "checksum": "abc",
    }

    return state


@pytest.mark.asyncio
async def test_workflow_routes_to_error_handler_on_generation_failure(
    sample_generation_state: NarrativeState,
) -> None:
    """Verify that a fatal error in 'generate' routes to 'error_handler' and halts execution."""

    # Mock all Phase 2 nodes
    mock_chapter_outline = MagicMock(side_effect=lambda s: {**s, "current_node": "chapter_outline"})

    # This node will fail fatally
    mock_generate = MagicMock(
        side_effect=lambda s: {
            **s,
            "has_fatal_error": True,
            "last_error": "Simulated generation failure",
            "error_node": "generate",
            "current_node": "generate",
        }
    )

    # These nodes should NOT be called
    mock_gen_embedding = MagicMock(side_effect=lambda s: {**s, "current_node": "gen_embedding"})
    mock_extract = MagicMock(side_effect=lambda s: {**s, "current_node": "extract"})
    mock_gen_scene_embeddings = MagicMock(side_effect=lambda s: {**s, "current_node": "gen_scene_embeddings"})
    mock_assemble_chapter = MagicMock(side_effect=lambda s: {**s, "current_node": "assemble_chapter"})

    with (
        patch(
            "core.langgraph.initialization.generate_chapter_outline",
            mock_chapter_outline,
        ),
        patch(
            "core.langgraph.subgraphs.generation.create_generation_subgraph",
            return_value=mock_generate,
        ),
        patch("core.langgraph.workflow.generate_embedding", mock_gen_embedding),
        patch(
            "core.langgraph.subgraphs.scene_extraction.create_scene_extraction_subgraph",
            return_value=mock_extract,
        ),
        patch("core.langgraph.workflow.generate_scene_embeddings", mock_gen_scene_embeddings),
        patch("core.langgraph.workflow.assemble_chapter", mock_assemble_chapter),
        patch("core.langgraph.workflow.normalize_relationships", MagicMock()),
        patch("core.langgraph.workflow.commit_to_graph", MagicMock()),
        patch("core.langgraph.subgraphs.validation.create_validation_subgraph", MagicMock()),
        patch("core.langgraph.workflow.revise_chapter", MagicMock()),
        patch("core.langgraph.workflow.summarize_chapter", MagicMock()),
        patch("core.langgraph.workflow.finalize_chapter", MagicMock()),
        patch("core.langgraph.workflow.heal_graph", MagicMock()),
        patch("core.langgraph.workflow.check_quality", MagicMock()),
    ):
        # Create workflow
        graph = create_full_workflow_graph()

        # Execute workflow
        result = await graph.ainvoke(sample_generation_state)

        # Verify routing
        assert result["has_fatal_error"] is True
        assert result["current_node"] == "error_handler"
        assert result["error_node"] == "generate"

        # Verify node calls
        mock_chapter_outline.assert_called_once()
        mock_generate.assert_called_once()

        # Verify execution halted
        mock_gen_embedding.assert_not_called()
        mock_extract.assert_not_called()


@pytest.mark.asyncio
async def test_workflow_routes_to_error_handler_on_validation_fatal_error(
    sample_generation_state: NarrativeState,
) -> None:
    """Verify that a fatal error in 'validate' routes to 'error_handler' via should_revise_or_handle_error."""

    # Mock nodes up to validate
    mock_chapter_outline = MagicMock(side_effect=lambda s: {**s, "current_node": "chapter_outline"})
    mock_generate = MagicMock(side_effect=lambda s: {**s, "current_node": "generate"})
    mock_gen_embedding = MagicMock(side_effect=lambda s: {**s, "current_node": "gen_embedding"})
    mock_extract = MagicMock(side_effect=lambda s: {**s, "current_node": "extract"})
    mock_gen_scene_embeddings = MagicMock(side_effect=lambda s: {**s, "current_node": "gen_scene_embeddings"})
    mock_assemble_chapter = MagicMock(side_effect=lambda s: {**s, "current_node": "assemble_chapter"})
    mock_normalize = MagicMock(side_effect=lambda s: {**s, "current_node": "normalize_relationships"})
    mock_commit = MagicMock(side_effect=lambda s: {**s, "current_node": "commit"})

    # Validation node fails fatally
    mock_validate = MagicMock(
        side_effect=lambda s: {
            **s,
            "has_fatal_error": True,
            "last_error": "Simulated validation failure",
            "error_node": "validate",
            "current_node": "validate",
        }
    )

    # Subsequent nodes should NOT be called
    mock_summarize = MagicMock(side_effect=lambda s: {**s, "current_node": "summarize"})
    mock_revise = MagicMock(side_effect=lambda s: {**s, "current_node": "revise"})

    with (
        patch(
            "core.langgraph.initialization.generate_chapter_outline",
            mock_chapter_outline,
        ),
        patch(
            "core.langgraph.subgraphs.generation.create_generation_subgraph",
            return_value=mock_generate,
        ),
        patch("core.langgraph.workflow.generate_embedding", mock_gen_embedding),
        patch(
            "core.langgraph.subgraphs.scene_extraction.create_scene_extraction_subgraph",
            return_value=mock_extract,
        ),
        patch("core.langgraph.workflow.generate_scene_embeddings", mock_gen_scene_embeddings),
        patch("core.langgraph.workflow.assemble_chapter", mock_assemble_chapter),
        patch("core.langgraph.workflow.normalize_relationships", mock_normalize),
        patch("core.langgraph.workflow.commit_to_graph", mock_commit),
        patch(
            "core.langgraph.subgraphs.validation.create_validation_subgraph",
            return_value=mock_validate,
        ),
        patch("core.langgraph.workflow.revise_chapter", mock_revise),
        patch("core.langgraph.workflow.summarize_chapter", mock_summarize),
        patch("core.langgraph.workflow.finalize_chapter", MagicMock()),
        patch("core.langgraph.workflow.heal_graph", MagicMock()),
        patch("core.langgraph.workflow.check_quality", MagicMock()),
    ):
        # Create workflow
        graph = create_full_workflow_graph()

        # Execute workflow
        result = await graph.ainvoke(sample_generation_state)

        # Verify routing
        assert result["has_fatal_error"] is True
        assert result["current_node"] == "error_handler"
        assert result["error_node"] == "validate"

        # Verify execution halted
        mock_summarize.assert_not_called()
        mock_revise.assert_not_called()


@pytest.mark.asyncio
async def test_workflow_continues_when_no_fatal_error(
    sample_generation_state: NarrativeState,
) -> None:
    """Verify that workflow proceeds normally when no fatal error is set."""

    # Mock all Phase 2 nodes with successful returns
    mock_chapter_outline = MagicMock(side_effect=lambda s: {**s, "current_node": "chapter_outline"})
    mock_generate = MagicMock(side_effect=lambda s: {**s, "current_node": "generate"})
    mock_gen_embedding = MagicMock(side_effect=lambda s: {**s, "current_node": "gen_embedding"})
    mock_extract = MagicMock(side_effect=lambda s: {**s, "current_node": "extract"})
    mock_gen_scene_embeddings = MagicMock(side_effect=lambda s: {**s, "current_node": "gen_scene_embeddings"})
    mock_assemble_chapter = MagicMock(side_effect=lambda s: {**s, "current_node": "assemble_chapter"})
    mock_normalize = MagicMock(side_effect=lambda s: {**s, "current_node": "normalize_relationships"})
    mock_commit = MagicMock(side_effect=lambda s: {**s, "current_node": "commit"})
    mock_validate = MagicMock(side_effect=lambda s: {**s, "current_node": "validate", "needs_revision": False})
    mock_summarize = MagicMock(side_effect=lambda s: {**s, "current_node": "summarize"})
    mock_finalize = MagicMock(side_effect=lambda s: {**s, "current_node": "finalize"})
    mock_heal = MagicMock(side_effect=lambda s: {**s, "current_node": "heal_graph"})
    mock_quality = MagicMock(side_effect=lambda s: {**s, "current_node": "check_quality"})

    with (
        patch(
            "core.langgraph.initialization.generate_chapter_outline",
            mock_chapter_outline,
        ),
        patch(
            "core.langgraph.subgraphs.generation.create_generation_subgraph",
            return_value=mock_generate,
        ),
        patch("core.langgraph.workflow.generate_embedding", mock_gen_embedding),
        patch(
            "core.langgraph.subgraphs.scene_extraction.create_scene_extraction_subgraph",
            return_value=mock_extract,
        ),
        patch("core.langgraph.workflow.generate_scene_embeddings", mock_gen_scene_embeddings),
        patch("core.langgraph.workflow.assemble_chapter", mock_assemble_chapter),
        patch("core.langgraph.workflow.normalize_relationships", mock_normalize),
        patch("core.langgraph.workflow.commit_to_graph", mock_commit),
        patch(
            "core.langgraph.subgraphs.validation.create_validation_subgraph",
            return_value=mock_validate,
        ),
        patch("core.langgraph.workflow.summarize_chapter", mock_summarize),
        patch("core.langgraph.workflow.finalize_chapter", mock_finalize),
        patch("core.langgraph.workflow.heal_graph", mock_heal),
        patch("core.langgraph.workflow.check_quality", mock_quality),
    ):
        # Create workflow
        graph = create_full_workflow_graph()

        # Execute workflow
        result = await graph.ainvoke(sample_generation_state)

        # Verify success
        assert result.get("has_fatal_error", False) is False
        assert result["current_node"] == "check_quality"

        # Verify all nodes called
        mock_chapter_outline.assert_called_once()
        mock_generate.assert_called_once()
        mock_gen_embedding.assert_not_called()
        mock_extract.assert_called_once()
        mock_gen_scene_embeddings.assert_called_once()
        mock_assemble_chapter.assert_called_once()
        mock_normalize.assert_called_once()
        mock_commit.assert_called_once()
        mock_validate.assert_called_once()
        mock_summarize.assert_called_once()
        mock_finalize.assert_called_once()
        mock_heal.assert_called_once()
        mock_quality.assert_called_once()
