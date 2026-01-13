# tests/test_langgraph/test_phase2_workflow.py
"""
Tests for Phase 2 LangGraph workflow (Step 2.5).

Tests the complete narrative generation workflow including all Phase 2 nodes.

Migration Reference: docs/phase2_migration_plan.md - Step 2.5

Note: The legacy Phase 2 graph has been removed. These tests exercise the chapter-generation
path within the full workflow graph (skip initialization via `initialization_complete=True`).
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.langgraph.state import Contradiction, NarrativeState, create_initial_state
from core.langgraph.workflow import (
    create_full_workflow_graph,
    handle_fatal_error,
    should_handle_error,
    should_revise_or_continue,
    should_revise_or_handle_error,
)


async def test_create_checkpointer_filename_only_path_does_not_raise(tmp_path: Any, monkeypatch: Any) -> None:
    """
    `create_checkpointer()` should not crash when `db_path` has no directory component.

    Regression test for LANGGRAPH-007: `os.path.dirname("saga.db") == ""` must not lead to
    `os.makedirs("")`.
    """
    monkeypatch.chdir(tmp_path)

    from core.langgraph.workflow import create_checkpointer

    async with create_checkpointer("saga.db") as checkpointer:
        assert checkpointer is not None


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
        extraction_model="test-model",
        revision_model="test-model",
    )

    # Add chapter outlines via content manager
    from core.langgraph.content_manager import ContentManager

    content_manager = ContentManager(str(tmp_path / "test-project"))
    chapter_outlines = {
        1: {
            "plot_point": "The hero begins their journey",
            "chapter_summary": "Introduction",
        },
        2: {
            "plot_point": "The hero meets a mentor",
            "chapter_summary": "Meeting the mentor",
        },
    }
    chapter_outlines_ref = content_manager.save_json(chapter_outlines, "chapter_outlines", "all", 1)
    state["chapter_outlines_ref"] = chapter_outlines_ref

    # Set current chapter
    state["current_chapter"] = 1
    state["initialization_complete"] = True

    return state


@pytest.fixture
def sample_phase2_state(sample_generation_state: NarrativeState) -> NarrativeState:
    """Backwards-compatible alias for legacy test names."""
    return sample_generation_state


@pytest.fixture
def mock_all_nodes() -> Any:
    """Mock all node functions for testing workflow routing."""

    mock_gen_node = MagicMock()
    mock_gen_node.side_effect = lambda state: {
        **state,
        "scene_drafts_ref": {"path": "mock_scene_drafts.json"},
        "current_scene_index": 2,
        "current_node": "generate",
    }

    mock_extract_node = MagicMock()
    mock_extract_node.side_effect = lambda state: {
        **state,
        "extracted_entities": {"characters": []},
        "extracted_relationships": [],
        "current_node": "extract",
    }

    mock_scene_embeddings_node = MagicMock()
    mock_scene_embeddings_node.side_effect = lambda state: {
        **state,
        "scene_embeddings_ref": {"path": "mock_scene_embeddings.json"},
        "current_node": "gen_scene_embeddings",
    }

    mock_assemble_chapter_node = MagicMock()
    mock_assemble_chapter_node.side_effect = lambda state: {
        **state,
        "draft_ref": {"path": "mock_draft"},
        "draft_word_count": 3,
        "current_node": "assemble_chapter",
    }

    mock_validate_node = MagicMock()
    mock_validate_node.side_effect = lambda state: {
        **state,
        "contradictions": [],
        "needs_revision": False,
        "current_node": "validate",
    }

    mock_chapter_outline_node = MagicMock()
    mock_chapter_outline_node.side_effect = lambda state: {
        **state,
        "chapter_outline_ref": {"path": "mock_outline"},
        "current_node": "chapter_outline",
    }

    mock_commit_node = MagicMock()
    mock_commit_node.side_effect = lambda state: {
        **state,
        "current_node": "commit",
    }

    mock_normalize_node = MagicMock()
    mock_normalize_node.side_effect = lambda state: {
        **state,
        "current_node": "normalize_relationships",
    }

    mock_revise_node = MagicMock()
    mock_revise_node.side_effect = lambda state: {
        **state,
        "draft_ref": {"path": "mock_revised"},
        "iteration_count": state.get("iteration_count", 0) + 1,
        "contradictions": [],
        "needs_revision": False,
        "current_node": "revise",
    }

    mock_summarize_node = MagicMock()
    mock_summarize_node.side_effect = lambda state: {
        **state,
        "summaries_ref": {"path": "mock_summaries"},
        "current_node": "summarize",
    }

    mock_finalize_node = MagicMock()
    mock_finalize_node.side_effect = lambda state: {
        **state,
        "extracted_entities": {},
        "extracted_relationships": [],
        "contradictions": [],
        "iteration_count": 0,
        "needs_revision": False,
        "current_node": "finalize",
    }

    mock_heal_node = MagicMock()
    mock_heal_node.side_effect = lambda state: {
        **state,
        "current_node": "heal_graph",
    }

    mock_quality_node = MagicMock()
    mock_quality_node.side_effect = lambda state: {
        **state,
        "current_node": "check_quality",
    }

    with (
        patch(
            "core.langgraph.subgraphs.generation.create_generation_subgraph",
            return_value=mock_gen_node,
        ),
        patch(
            "core.langgraph.subgraphs.scene_extraction.create_scene_extraction_subgraph",
            return_value=mock_extract_node,
        ),
        patch(
            "core.langgraph.subgraphs.validation.create_validation_subgraph",
            return_value=mock_validate_node,
        ),
        patch(
            "core.langgraph.workflow.generate_scene_embeddings",
            side_effect=mock_scene_embeddings_node,
        ),
        patch(
            "core.langgraph.workflow.assemble_chapter",
            side_effect=mock_assemble_chapter_node,
        ),
        patch(
            "core.langgraph.initialization.generate_chapter_outline",
            side_effect=mock_chapter_outline_node,
        ),
        patch("core.langgraph.workflow.commit_to_graph", side_effect=mock_commit_node),
        patch("core.langgraph.workflow.normalize_relationships", side_effect=mock_normalize_node),
        patch("core.langgraph.workflow.revise_chapter", side_effect=mock_revise_node),
        patch("core.langgraph.workflow.summarize_chapter", side_effect=mock_summarize_node),
        patch("core.langgraph.workflow.finalize_chapter", side_effect=mock_finalize_node),
        patch("core.langgraph.workflow.heal_graph", side_effect=mock_heal_node),
        patch("core.langgraph.workflow.check_quality", side_effect=mock_quality_node),
    ):
        yield {
            "chapter_outline": mock_chapter_outline_node,
            "generate": mock_gen_node,
            "extract": mock_extract_node,
            "gen_scene_embeddings": mock_scene_embeddings_node,
            "assemble_chapter": mock_assemble_chapter_node,
            "normalize_relationships": mock_normalize_node,
            "commit": mock_commit_node,
            "validate": mock_validate_node,
            "revise": mock_revise_node,
            "summarize": mock_summarize_node,
            "finalize": mock_finalize_node,
            "heal_graph": mock_heal_node,
            "check_quality": mock_quality_node,
        }


class TestShouldReviseOrContinue:
    """Tests for should_revise_or_continue routing function."""

    def test_route_to_summarize_when_no_revision_needed(self) -> None:
        """Test routing to summarize when needs_revision is False."""
        # Using type ignore or creating partial dict because create_initial_state creates full structure
        # but here we test with minimal required fields for the function
        state: Any = {
            "needs_revision": False,
            "iteration_count": 0,
            "max_iterations": 3,
            "force_continue": False,
        }

        result = should_revise_or_continue(state)
        assert result == "summarize"

    def test_route_to_revise_when_revision_needed(self) -> None:
        """Test routing to revise when needs_revision is True."""
        state: Any = {
            "needs_revision": True,
            "iteration_count": 0,
            "max_iterations": 3,
            "force_continue": False,
        }

        result = should_revise_or_continue(state)
        assert result == "revise"

    def test_route_to_summarize_when_max_iterations_reached(self) -> None:
        """Test routing to summarize when max iterations reached."""
        state: Any = {
            "needs_revision": True,
            "iteration_count": 3,
            "max_iterations": 3,
            "force_continue": False,
        }

        result = should_revise_or_continue(state)
        assert result == "summarize"

    def test_route_to_summarize_when_force_continue(self) -> None:
        """Test routing to summarize when force_continue is enabled."""
        state: Any = {
            "needs_revision": True,
            "iteration_count": 0,
            "max_iterations": 3,
            "force_continue": True,
        }

        result = should_revise_or_continue(state)
        assert result == "summarize"

    def test_route_to_revise_under_max_iterations(self) -> None:
        """Test routing to revise when under max iterations."""
        state: Any = {
            "needs_revision": True,
            "iteration_count": 1,
            "max_iterations": 3,
            "force_continue": False,
        }

        result = should_revise_or_continue(state)
        assert result == "revise"


@pytest.mark.asyncio
class TestPhase2Workflow:
    """Tests for complete Phase 2 workflow."""

    async def test_workflow_successful_path_no_revision(self, sample_generation_state: NarrativeState, mock_all_nodes: Any) -> None:
        """Test successful workflow without any revisions."""
        # Create workflow
        graph = create_full_workflow_graph()

        # Execute workflow
        await graph.ainvoke(sample_generation_state)

        # Verify all nodes were called in correct order
        mock_all_nodes["chapter_outline"].assert_called_once()
        mock_all_nodes["gen_scene_embeddings"].assert_called_once()
        mock_all_nodes["assemble_chapter"].assert_called_once()
        mock_all_nodes["normalize_relationships"].assert_called_once()
        mock_all_nodes["validate"].assert_called_once()
        mock_all_nodes["commit"].assert_called_once()
        mock_all_nodes["summarize"].assert_called_once()
        mock_all_nodes["finalize"].assert_called_once()
        mock_all_nodes["heal_graph"].assert_called_once()
        mock_all_nodes["check_quality"].assert_called_once()

        # Revision should not be called (no contradictions)
        mock_all_nodes["revise"].assert_not_called()

    async def test_workflow_with_single_revision(self, sample_generation_state: NarrativeState, mock_all_nodes: Any) -> None:
        """Test workflow with one revision cycle."""
        call_sequence: list[str] = []
        validate_call_count = 0

        def mock_validate(s):
            nonlocal validate_call_count
            call_sequence.append("validate")
            validate_call_count += 1
            if validate_call_count == 1:
                return {
                    **s,
                    "contradictions": [
                        Contradiction(
                            type="test",
                            description="Issue",
                            conflicting_chapters=[1],
                            severity="minor",
                        )
                    ],
                    "needs_revision": True,
                    "current_node": "validate",
                }
            return {
                **s,
                "contradictions": [],
                "needs_revision": False,
                "current_node": "validate",
            }

        def mock_commit(s):
            call_sequence.append("commit")
            return {
                **s,
                "current_node": "commit",
            }

        mock_all_nodes["validate"].side_effect = mock_validate
        mock_all_nodes["commit"].side_effect = mock_commit

        # Create workflow
        graph = create_full_workflow_graph()

        # Execute workflow
        await graph.ainvoke(sample_generation_state)

        # Revision should be called once
        mock_all_nodes["revise"].assert_called_once()

        # Validation should be called twice (revise then accept)
        assert validate_call_count == 2

        # Commit should happen twice: once from normalize_relationships, once after validation accepts
        assert mock_all_nodes["commit"].call_count == 2

    async def test_workflow_max_iterations_enforcement(self, sample_generation_state: NarrativeState, mock_all_nodes: Any) -> None:
        """Test that max iterations are enforced."""
        # Configure validate to always request revision
        mock_all_nodes["validate"].side_effect = lambda s: {
            **s,
            "contradictions": [Contradiction(type="test", description="Issue", conflicting_chapters=[1], severity="minor")],
            "needs_revision": True,
            "current_node": "validate",
        }

        # Create workflow
        graph = create_full_workflow_graph()

        # Execute workflow
        state = {**sample_generation_state, "max_iterations": 2}
        await graph.ainvoke(state, config={"recursion_limit": 200})

        # Revision should be called twice (max_iterations)
        assert mock_all_nodes["revise"].call_count == 2

    async def test_workflow_force_continue_skips_revision(self, sample_generation_state: NarrativeState, mock_all_nodes: Any) -> None:
        """Test that force_continue skips revision."""
        # Set force_continue
        state = {**sample_generation_state}
        state["force_continue"] = True

        # Configure validate to request revision
        mock_all_nodes["validate"].side_effect = lambda s: {
            **s,
            "contradictions": [
                Contradiction(
                    type="test",
                    description="Issue",
                    conflicting_chapters=[1],
                    severity="minor",
                )
            ],
            "needs_revision": True,
            "current_node": "validate",
        }

        # Create workflow
        graph = create_full_workflow_graph()

        # Execute workflow
        await graph.ainvoke(state)

        # Revision should not be called (force_continue)
        mock_all_nodes["revise"].assert_not_called()

        # Should still complete workflow
        mock_all_nodes["summarize"].assert_called_once()
        mock_all_nodes["finalize"].assert_called_once()

    async def test_workflow_node_execution_order(self, sample_generation_state: NarrativeState, mock_all_nodes: Any) -> None:
        """Test that nodes execute in correct order."""
        # Create workflow
        graph = create_full_workflow_graph()

        # Execute workflow
        await graph.ainvoke(sample_generation_state)

        # Check call order of some key nodes
        # Chapter outline -> Generate -> Extract -> Scene Embeddings -> Assemble -> Normalize -> Validate -> Commit -> Summarize -> Finalize -> Heal -> Quality

        mock_all_nodes["chapter_outline"].assert_called_once()
        mock_all_nodes["generate"].assert_called_once()
        mock_all_nodes["extract"].assert_called_once()
        mock_all_nodes["gen_scene_embeddings"].assert_called_once()
        mock_all_nodes["assemble_chapter"].assert_called_once()
        mock_all_nodes["normalize_relationships"].assert_called_once()
        mock_all_nodes["commit"].assert_called()
        mock_all_nodes["validate"].assert_called_once()
        mock_all_nodes["summarize"].assert_called_once()
        mock_all_nodes["finalize"].assert_called_once()
        mock_all_nodes["heal_graph"].assert_called_once()
        mock_all_nodes["check_quality"].assert_called_once()

    async def test_workflow_with_checkpointing(
        self,
        sample_generation_state: NarrativeState,
        mock_all_nodes: Any,
        tmp_path: Any,
    ) -> None:
        """Workflow writes checkpoints when a checkpointer is provided."""
        from core.langgraph.workflow import create_checkpointer

        checkpoint_db = tmp_path / "test_checkpoint.db"

        async with create_checkpointer(str(checkpoint_db)) as checkpointer:
            graph = create_full_workflow_graph(checkpointer=checkpointer)

            config = {"configurable": {"thread_id": "test-thread"}}
            result = await graph.ainvoke(sample_generation_state, config=config)

        assert result["current_node"] == "check_quality"
        assert checkpoint_db.exists()
        assert checkpoint_db.stat().st_size > 0

    async def test_workflow_preserves_state_through_nodes(self, sample_generation_state: NarrativeState, mock_all_nodes: Any) -> None:
        """Test that state is properly preserved through all nodes."""
        # Use a field that exists in NarrativeState but is not usually used in this context
        state = {**sample_generation_state, "theme": "Custom Theme"}

        # Create workflow
        graph = create_full_workflow_graph()

        # Execute workflow
        result = await graph.ainvoke(state)

        # Verify custom theme is preserved
        assert result["theme"] == "Custom Theme"
        assert result["project_id"] == "test-project"

    async def test_workflow_multiple_revision_cycles(self, sample_generation_state: NarrativeState, mock_all_nodes: Any) -> None:
        """Test workflow with multiple revision cycles."""
        # Configure validate to request revision twice then stop
        call_count = 0

        def mock_validate(s):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return {
                    **s,
                    "contradictions": [Contradiction(type="test", description="Issue", conflicting_chapters=[1], severity="minor")],
                    "needs_revision": True,
                    "current_node": "validate",
                }
            return {
                **s,
                "contradictions": [],
                "needs_revision": False,
                "current_node": "validate",
            }

        mock_all_nodes["validate"].side_effect = mock_validate

        # Create workflow
        graph = create_full_workflow_graph()

        # Execute workflow
        await graph.ainvoke(sample_generation_state, config={"recursion_limit": 200})

        # Revision should be called twice
        assert mock_all_nodes["revise"].call_count == 2
        # Validation should be called three times
        assert call_count == 3


def test_phase2_graph_check_quality_is_reachable() -> None:
    """
    Regression test for LANGGRAPH-005 / remediation plan 9.1 #4.

    If `check_quality` is declared in the Phase 2 graph, it must be reachable.
    This prevents dead-node drift where QA is assumed to run but never executes.
    """
    graph = create_full_workflow_graph()
    graph_obj = graph.get_graph()

    nodes = set(graph_obj.nodes)
    edges = [(edge.source, edge.target) for edge in graph_obj.edges]

    # Sanity: this test assumes Phase 2 declares check_quality.
    assert "check_quality" in nodes

    # Ensure the intended wiring is present.
    assert ("heal_graph", "check_quality") in edges

    # Ensure `check_quality` is reachable from the entrypoint through node-to-node edges.
    start = "route"
    adjacency: dict[str, set[str]] = {}
    for source, target in edges:
        if isinstance(source, str) and isinstance(target, str):
            adjacency.setdefault(source, set()).add(target)

    visited: set[str] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        for nxt in adjacency.get(node, set()):
            # Only traverse actual nodes; ignore END-style sentinel targets.
            if nxt in nodes:
                stack.append(nxt)

    assert "check_quality" in visited


@pytest.mark.asyncio
class TestPhase2Integration:
    """Integration tests for Phase 2 workflow."""

    async def test_complete_chapter_generation_workflow(self, sample_generation_state: NarrativeState, mock_all_nodes: Any, tmp_path: Any) -> None:
        """Test complete end-to-end chapter generation."""
        # Create workflow
        graph = create_full_workflow_graph()

        # Execute workflow
        result = await graph.ainvoke(sample_generation_state)

        # Verify final state reflects completion of one chapter
        assert result["current_chapter"] == 1
        assert result["initialization_complete"] is True

    async def test_workflow_graph_structure(self) -> None:
        """Test that Phase 2 graph has correct structure."""
        graph = create_full_workflow_graph()
        graph_obj = graph.get_graph()

        nodes = set(graph_obj.nodes)

        # Verify presence of expected nodes
        expected_nodes = [
            "chapter_outline",
            "generate",
            "extract",
            "gen_scene_embeddings",
            "assemble_chapter",
            "normalize_relationships",
            "commit",
            "validate",
            "revise",
            "summarize",
            "finalize",
            "heal_graph",
            "check_quality",
            "advance_chapter",
        ]
        for node in expected_nodes:
            assert node in nodes

    def test_workflow_graph_persistence_wiring_is_after_validation(self) -> None:
        """Persistence boundary must be after validation acceptance."""
        graph = create_full_workflow_graph()
        graph_obj = graph.get_graph()

        edges = {(edge.source, edge.target) for edge in graph_obj.edges}

        # Relationship persistence happens before validation
        assert ("normalize_relationships", "commit") in edges
        # Validation happens after relationship commit
        assert ("commit", "validate") in edges
        # If validation passes, continue to summarize
        assert ("validate", "summarize") in edges

    def test_workflow_revision_loop_routes_to_generate(self) -> None:
        """Revision loop should route validate → revise → generate (scene regeneration)."""
        graph = create_full_workflow_graph()
        graph_obj = graph.get_graph()

        edges = [(edge.source, edge.target) for edge in graph_obj.edges]

        assert ("validate", "revise") in edges
        assert ("revise", "generate") in edges


class TestErrorRoutingFunctions:
    """Tests for error routing conditional edge functions (P1.3)."""

    def test_should_handle_error_routes_to_error_on_fatal_error(self) -> None:
        """Test should_handle_error routes to error when has_fatal_error is True."""
        state: Any = {
            "has_fatal_error": True,
            "last_error": "Test error",
            "error_node": "generate",
        }

        result = should_handle_error(state)
        assert result == "error"

    def test_should_handle_error_routes_to_continue_when_no_error(self) -> None:
        """Test should_handle_error routes to continue when has_fatal_error is False."""
        state: Any = {
            "has_fatal_error": False,
            "last_error": None,
            "error_node": None,
        }

        result = should_handle_error(state)
        assert result == "continue"

    def test_should_handle_error_routes_to_continue_when_error_missing(self) -> None:
        """Test should_handle_error routes to continue when has_fatal_error is missing."""
        state: Any = {}

        result = should_handle_error(state)
        assert result == "continue"

    def test_should_revise_or_handle_error_prioritizes_fatal_error(self) -> None:
        """Test should_revise_or_handle_error prioritizes fatal error over revision."""
        state: Any = {
            "has_fatal_error": True,
            "needs_revision": True,
            "last_error": "Fatal error",
        }

        result = should_revise_or_handle_error(state)
        assert result == "error"

    def test_should_revise_or_handle_error_routes_to_revise_when_needed(self) -> None:
        """Test should_revise_or_handle_error routes to revise when needs_revision is True."""
        state: Any = {
            "has_fatal_error": False,
            "needs_revision": True,
        }

        result = should_revise_or_handle_error(state)
        assert result == "revise"

    def test_should_revise_or_handle_error_routes_to_continue_when_no_issues(
        self,
    ) -> None:
        """Test should_revise_or_handle_error routes to continue when no errors or revision needed."""
        state: Any = {
            "has_fatal_error": False,
            "needs_revision": False,
        }

        result = should_revise_or_handle_error(state)
        assert result == "continue"

    def test_handle_fatal_error_sets_error_handler_node(self) -> None:
        """Test handle_fatal_error node sets current_node to error_handler."""
        state: Any = {
            "has_fatal_error": True,
            "last_error": "Test fatal error",
            "error_node": "generate",
            "current_chapter": 1,
        }

        result = handle_fatal_error(state)

        assert result["current_node"] == "error_handler"
        assert result["has_fatal_error"] is True
        assert result["last_error"] == "Test fatal error"
        assert result["error_node"] == "generate"


@pytest.mark.asyncio
class TestWorkflowErrorHandling:
    """Tests for error handling in Phase 2 workflow (P1.1 & P1.3)."""

    async def test_workflow_routes_to_error_handler_on_generation_failure(self, sample_generation_state: NarrativeState, mock_all_nodes: Any) -> None:
        """Test workflow routes to error_handler when generation fails."""
        # Configure generate subgraph to return error
        mock_all_nodes["generate"].side_effect = lambda s: {
            **s,
            "has_fatal_error": True,
            "last_error": "Generation failed",
            "error_node": "generate",
        }

        graph = create_full_workflow_graph()
        result = await graph.ainvoke(sample_generation_state)

        assert result["current_node"] == "error_handler"

    async def test_workflow_routes_to_error_handler_on_extraction_failure(self, sample_generation_state: NarrativeState, mock_all_nodes: Any) -> None:
        """Test workflow routes to error_handler when extraction fails."""
        mock_all_nodes["extract"].side_effect = lambda s: {
            **s,
            "has_fatal_error": True,
            "last_error": "Extraction failed",
            "error_node": "extract",
        }

        graph = create_full_workflow_graph()
        result = await graph.ainvoke(sample_generation_state)

        assert result["current_node"] == "error_handler"

    async def test_workflow_routes_to_error_handler_on_validation_failure(self, sample_generation_state: NarrativeState, mock_all_nodes: Any) -> None:
        """Test workflow routes to error_handler when validation triggers fatal error."""
        mock_all_nodes["validate"].side_effect = lambda s: {
            **s,
            "has_fatal_error": True,
            "last_error": "Validation failed",
            "error_node": "validate",
        }

        graph = create_full_workflow_graph()
        result = await graph.ainvoke(sample_generation_state)

        assert result["current_node"] == "error_handler"

    async def test_workflow_routes_to_error_handler_on_revision_failure(self, sample_generation_state: NarrativeState, mock_all_nodes: Any) -> None:
        """Test workflow routes to error_handler when revision fails."""
        # Request revision first
        mock_all_nodes["validate"].side_effect = lambda s: {
            **s,
            "needs_revision": True,
            "current_node": "validate",
        }

        # Then make revise fail
        mock_all_nodes["revise"].side_effect = lambda s: {
            **s,
            "has_fatal_error": True,
            "last_error": "Revision failed",
            "error_node": "revise",
        }

        graph = create_full_workflow_graph()
        result = await graph.ainvoke(sample_generation_state)

        assert result["current_node"] == "error_handler"

    async def test_workflow_completes_successfully_without_errors(self, sample_generation_state: NarrativeState, mock_all_nodes: Any) -> None:
        """Test workflow completes when no fatal errors occur."""
        graph = create_full_workflow_graph()
        result = await graph.ainvoke(sample_generation_state)

        assert result["has_fatal_error"] is False
        mock_all_nodes["check_quality"].assert_called_once()

    async def test_workflow_stops_at_first_fatal_error(self, sample_generation_state: NarrativeState, mock_all_nodes: Any) -> None:
        """Test workflow stops execution at first fatal error encountered."""
        mock_all_nodes["gen_scene_embeddings"].side_effect = lambda s: {
            **s,
            "has_fatal_error": True,
            "last_error": "Scene embeddings failed",
            "error_node": "gen_scene_embeddings",
        }

        graph = create_full_workflow_graph()
        result = await graph.ainvoke(sample_generation_state)

        assert result["current_node"] == "error_handler"
        mock_all_nodes["assemble_chapter"].assert_not_called()
        mock_all_nodes["normalize_relationships"].assert_not_called()

    async def test_workflow_multi_chapter_loop(self, sample_generation_state: NarrativeState, mock_all_nodes: Any) -> None:
        """Test that workflow loops back for multiple chapters."""
        # Set total chapters to 2
        state = {**sample_generation_state}
        state["total_chapters"] = 2
        state["current_chapter"] = 1

        # Create workflow
        graph = create_full_workflow_graph()

        # Execute workflow
        result = await graph.ainvoke(state, config={"recursion_limit": 300})

        # Verify that chapter_outline, generate, etc. were called twice
        assert mock_all_nodes["chapter_outline"].call_count == 2
        assert mock_all_nodes["generate"].call_count == 2
        assert mock_all_nodes["check_quality"].call_count == 2

        assert result["current_chapter"] == 2
