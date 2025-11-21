# tests/test_langgraph/test_phase2_workflow.py
"""
Tests for Phase 2 LangGraph workflow (Step 2.5).

Tests the complete narrative generation workflow including all Phase 2 nodes.

Migration Reference: docs/phase2_migration_plan.md - Step 2.5
"""

from unittest.mock import patch

import pytest

from core.langgraph.state import Contradiction, create_initial_state
from core.langgraph.workflow import (
    create_phase2_graph,
    handle_fatal_error,
    should_handle_error,
    should_revise_or_continue,
    should_revise_or_handle_error,
)


@pytest.fixture
def sample_phase2_state(tmp_path):
    """Sample state ready for Phase 2 workflow."""
    state = create_initial_state(
        project_id="test-project",
        title="Test Novel",
        genre="Fantasy",
        theme="Adventure",
        setting="Medieval world",
        target_word_count=80000,
        total_chapters=20,
        project_dir=str(tmp_path / "test-project"),
        protagonist_name="Hero",
        generation_model="test-model",
        extraction_model="test-model",
        revision_model="test-model",
    )

    # Add plot outline
    state["plot_outline"] = {
        1: {
            "plot_point": "The hero begins their journey",
            "chapter_summary": "Introduction",
        },
        "title": "Test Novel",
        "genre": "Fantasy",
    }

    # Set current chapter
    state["current_chapter"] = 1
    state["plot_point_focus"] = "The hero begins their journey"

    return state


@pytest.fixture
def mock_all_nodes():
    """Mock all node functions for testing workflow routing."""
    with (
        patch("core.langgraph.workflow.generate_chapter") as mock_generate,
        patch("core.langgraph.workflow.extract_entities") as mock_extract,
        patch("core.langgraph.workflow.commit_to_graph") as mock_commit,
        patch("core.langgraph.workflow.validate_consistency") as mock_validate,
        patch("core.langgraph.workflow.revise_chapter") as mock_revise,
        patch("core.langgraph.workflow.summarize_chapter") as mock_summarize,
        patch("core.langgraph.workflow.finalize_chapter") as mock_finalize,
    ):
        # Configure generate node
        mock_generate.side_effect = lambda state: {
            **state,
            "draft_text": "Generated chapter text.",
            "draft_word_count": 3,
            "current_node": "generate",
        }

        # Configure extract node
        mock_extract.side_effect = lambda state: {
            **state,
            "extracted_entities": {"characters": []},
            "extracted_relationships": [],
            "current_node": "extract",
        }

        # Configure commit node
        mock_commit.side_effect = lambda state: {
            **state,
            "current_node": "commit",
        }

        # Configure validate node (no contradictions)
        mock_validate.side_effect = lambda state: {
            **state,
            "contradictions": [],
            "needs_revision": False,
            "current_node": "validate",
        }

        # Configure revise node
        mock_revise.side_effect = lambda state: {
            **state,
            "draft_text": "Revised chapter text.",
            "iteration_count": state.get("iteration_count", 0) + 1,
            "contradictions": [],
            "needs_revision": False,
            "current_node": "revise",
        }

        # Configure summarize node
        mock_summarize.side_effect = lambda state: {
            **state,
            "previous_chapter_summaries": ["Chapter summary"],
            "current_node": "summarize",
        }

        # Configure finalize node
        mock_finalize.side_effect = lambda state: {
            **state,
            "extracted_entities": {},
            "extracted_relationships": [],
            "contradictions": [],
            "iteration_count": 0,
            "needs_revision": False,
            "current_node": "finalize",
        }

        yield {
            "generate": mock_generate,
            "extract": mock_extract,
            "commit": mock_commit,
            "validate": mock_validate,
            "revise": mock_revise,
            "summarize": mock_summarize,
            "finalize": mock_finalize,
        }


class TestShouldReviseOrContinue:
    """Tests for should_revise_or_continue routing function."""

    def test_route_to_summarize_when_no_revision_needed(self):
        """Test routing to summarize when needs_revision is False."""
        state = {
            "needs_revision": False,
            "iteration_count": 0,
            "max_iterations": 3,
            "force_continue": False,
        }

        result = should_revise_or_continue(state)
        assert result == "summarize"

    def test_route_to_revise_when_revision_needed(self):
        """Test routing to revise when needs_revision is True."""
        state = {
            "needs_revision": True,
            "iteration_count": 0,
            "max_iterations": 3,
            "force_continue": False,
        }

        result = should_revise_or_continue(state)
        assert result == "revise"

    def test_route_to_summarize_when_max_iterations_reached(self):
        """Test routing to summarize when max iterations reached."""
        state = {
            "needs_revision": True,
            "iteration_count": 3,
            "max_iterations": 3,
            "force_continue": False,
        }

        result = should_revise_or_continue(state)
        assert result == "summarize"

    def test_route_to_summarize_when_force_continue(self):
        """Test routing to summarize when force_continue is enabled."""
        state = {
            "needs_revision": True,
            "iteration_count": 0,
            "max_iterations": 3,
            "force_continue": True,
        }

        result = should_revise_or_continue(state)
        assert result == "summarize"

    def test_route_to_revise_under_max_iterations(self):
        """Test routing to revise when under max iterations."""
        state = {
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

    async def test_workflow_successful_path_no_revision(
        self, sample_phase2_state, mock_all_nodes
    ):
        """Test successful workflow without any revisions."""
        # Create workflow
        graph = create_phase2_graph()

        # Execute workflow
        result = await graph.ainvoke(sample_phase2_state)

        # Verify all nodes were called in correct order
        mock_all_nodes["generate"].assert_called_once()
        mock_all_nodes["extract"].assert_called()
        mock_all_nodes["commit"].assert_called()
        mock_all_nodes["validate"].assert_called()
        mock_all_nodes["summarize"].assert_called_once()
        mock_all_nodes["finalize"].assert_called_once()

        # Revision should not be called (no contradictions)
        mock_all_nodes["revise"].assert_not_called()

        # Verify final state
        assert result["current_node"] == "finalize"
        assert result["draft_text"] is not None

    async def test_workflow_with_single_revision(
        self, sample_phase2_state, mock_all_nodes
    ):
        """Test workflow with one revision cycle."""
        # Configure validate to trigger revision on first call only
        call_count = {"validate": 0}

        def validate_with_revision(state):
            call_count["validate"] += 1
            if call_count["validate"] == 1:
                # First validation: needs revision
                return {
                    **state,
                    "contradictions": [
                        Contradiction(
                            type="test",
                            description="Test issue",
                            conflicting_chapters=[1],
                            severity="minor",
                        )
                    ],
                    "needs_revision": True,
                    "current_node": "validate",
                }
            else:
                # Second validation: all good
                return {
                    **state,
                    "contradictions": [],
                    "needs_revision": False,
                    "current_node": "validate",
                }

        mock_all_nodes["validate"].side_effect = validate_with_revision

        # Create workflow
        graph = create_phase2_graph()

        # Execute workflow
        result = await graph.ainvoke(sample_phase2_state)

        # Verify revision was called
        mock_all_nodes["revise"].assert_called_once()

        # Verify extract was called twice (initial + after revision)
        assert mock_all_nodes["extract"].call_count == 2

        # Verify commit was called twice
        assert mock_all_nodes["commit"].call_count == 2

        # Verify validate was called twice
        assert mock_all_nodes["validate"].call_count == 2

        # Verify finalization happened
        mock_all_nodes["summarize"].assert_called_once()
        mock_all_nodes["finalize"].assert_called_once()

        # Verify final state
        assert result["current_node"] == "finalize"

    async def test_workflow_max_iterations_enforcement(
        self, sample_phase2_state, mock_all_nodes
    ):
        """Test that max iterations are enforced."""
        # Set low max iterations
        state = {**sample_phase2_state}
        state["max_iterations"] = 2

        # Configure validate to always request revision
        mock_all_nodes["validate"].side_effect = lambda s: {
            **s,
            "contradictions": [
                Contradiction(
                    type="test",
                    description="Persistent issue",
                    conflicting_chapters=[1],
                    severity="minor",
                )
            ],
            "needs_revision": True,
            "current_node": "validate",
        }

        # Create workflow
        graph = create_phase2_graph()

        # Execute workflow
        await graph.ainvoke(state)

        # Should revise exactly max_iterations times (2)
        assert mock_all_nodes["revise"].call_count == 2

        # Should still complete workflow
        mock_all_nodes["summarize"].assert_called_once()
        mock_all_nodes["finalize"].assert_called_once()

    async def test_workflow_force_continue_skips_revision(
        self, sample_phase2_state, mock_all_nodes
    ):
        """Test that force_continue skips revision."""
        # Set force_continue
        state = {**sample_phase2_state}
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
        graph = create_phase2_graph()

        # Execute workflow
        await graph.ainvoke(state)

        # Revision should not be called (force_continue)
        mock_all_nodes["revise"].assert_not_called()

        # Should still complete workflow
        mock_all_nodes["summarize"].assert_called_once()
        mock_all_nodes["finalize"].assert_called_once()

    async def test_workflow_node_execution_order(
        self, sample_phase2_state, mock_all_nodes
    ):
        """Test that nodes execute in correct order."""
        execution_order = []

        # Track execution order
        for node_name, mock_node in mock_all_nodes.items():
            original_side_effect = mock_node.side_effect

            def make_tracker(name, original):
                def tracker(state):
                    execution_order.append(name)
                    return original(state)

                return tracker

            mock_node.side_effect = make_tracker(node_name, original_side_effect)

        # Create workflow
        graph = create_phase2_graph()

        # Execute workflow
        await graph.ainvoke(sample_phase2_state)

        # Verify execution order
        expected_order = [
            "generate",
            "extract",
            "commit",
            "validate",
            "summarize",
            "finalize",
        ]
        assert execution_order == expected_order

    @pytest.mark.skip(
        reason="AsyncSqliteSaver.from_conn_string() returns async context manager. "
        "Checkpointing functionality is verified in Phase 1 tests. "
        "This test needs refactoring to use async context manager properly."
    )
    async def test_workflow_with_checkpointing(
        self, sample_phase2_state, mock_all_nodes, tmp_path
    ):
        """Test workflow with checkpointing enabled."""
        from core.langgraph.workflow import create_checkpointer

        # Create checkpointer
        checkpoint_db = tmp_path / "test_checkpoint.db"
        checkpointer = create_checkpointer(str(checkpoint_db))

        # Create workflow with checkpointing
        graph = create_phase2_graph(checkpointer=checkpointer)

        # Execute workflow
        config = {"configurable": {"thread_id": "test-thread"}}
        result = await graph.ainvoke(sample_phase2_state, config=config)

        # Verify completion
        assert result["current_node"] == "finalize"

        # Verify checkpoint file was created
        assert checkpoint_db.exists()

    async def test_workflow_preserves_state_through_nodes(
        self, sample_phase2_state, mock_all_nodes
    ):
        """Test that state is properly preserved through all nodes."""
        # Create workflow
        graph = create_phase2_graph()

        # Execute workflow
        result = await graph.ainvoke(sample_phase2_state)

        # Verify important state fields are preserved
        assert result["title"] == sample_phase2_state["title"]
        assert result["genre"] == sample_phase2_state["genre"]
        assert result["current_chapter"] == sample_phase2_state["current_chapter"]
        assert result["project_id"] == sample_phase2_state["project_id"]

    async def test_workflow_multiple_revision_cycles(
        self, sample_phase2_state, mock_all_nodes
    ):
        """Test workflow with multiple revision cycles."""
        # Configure validate to request revision twice, then succeed
        call_count = {"validate": 0}

        def validate_with_multiple_revisions(state):
            call_count["validate"] += 1
            if call_count["validate"] <= 2:
                return {
                    **state,
                    "contradictions": [
                        Contradiction(
                            type="test",
                            description=f"Issue {call_count['validate']}",
                            conflicting_chapters=[1],
                            severity="minor",
                        )
                    ],
                    "needs_revision": True,
                    "current_node": "validate",
                }
            else:
                return {
                    **state,
                    "contradictions": [],
                    "needs_revision": False,
                    "current_node": "validate",
                }

        mock_all_nodes["validate"].side_effect = validate_with_multiple_revisions

        # Create workflow
        graph = create_phase2_graph()

        # Execute workflow
        result = await graph.ainvoke(sample_phase2_state)

        # Should revise twice
        assert mock_all_nodes["revise"].call_count == 2

        # Should complete successfully
        assert result["current_node"] == "finalize"


@pytest.mark.asyncio
class TestPhase2Integration:
    """Integration tests for Phase 2 workflow."""

    async def test_complete_chapter_generation_workflow(
        self, sample_phase2_state, mock_all_nodes, tmp_path
    ):
        """Test complete end-to-end chapter generation."""
        # Create workflow
        graph = create_phase2_graph()

        # Execute workflow
        result = await graph.ainvoke(sample_phase2_state)

        # Verify all stages completed
        assert result["draft_text"] is not None  # Generation
        assert "extracted_entities" in result  # Extraction (cleaned by finalize)
        assert result["current_node"] == "finalize"  # Finalization

        # Verify cleanup happened
        assert result["iteration_count"] == 0
        assert result["needs_revision"] is False

    async def test_workflow_graph_structure(self):
        """Test that Phase 2 graph has correct structure."""
        graph = create_phase2_graph()

        # Verify graph was compiled successfully
        assert graph is not None

        # Graph should have nodes and edges
        # (LangGraph doesn't expose direct inspection, but compilation validates structure)


class TestErrorRoutingFunctions:
    """Tests for error routing conditional edge functions (P1.3)."""

    def test_should_handle_error_routes_to_error_on_fatal_error(self):
        """Test should_handle_error routes to error when has_fatal_error is True."""
        state = {
            "has_fatal_error": True,
            "last_error": "Test error",
            "error_node": "generate",
        }

        result = should_handle_error(state)
        assert result == "error"

    def test_should_handle_error_routes_to_continue_when_no_error(self):
        """Test should_handle_error routes to continue when has_fatal_error is False."""
        state = {
            "has_fatal_error": False,
            "last_error": None,
            "error_node": None,
        }

        result = should_handle_error(state)
        assert result == "continue"

    def test_should_handle_error_routes_to_continue_when_error_missing(self):
        """Test should_handle_error routes to continue when has_fatal_error is missing."""
        state = {}

        result = should_handle_error(state)
        assert result == "continue"

    def test_should_revise_or_handle_error_prioritizes_fatal_error(self):
        """Test should_revise_or_handle_error prioritizes fatal error over revision."""
        state = {
            "has_fatal_error": True,
            "needs_revision": True,
            "last_error": "Fatal error",
        }

        result = should_revise_or_handle_error(state)
        assert result == "error"

    def test_should_revise_or_handle_error_routes_to_revise_when_needed(self):
        """Test should_revise_or_handle_error routes to revise when needs_revision is True."""
        state = {
            "has_fatal_error": False,
            "needs_revision": True,
        }

        result = should_revise_or_handle_error(state)
        assert result == "revise"

    def test_should_revise_or_handle_error_routes_to_continue_when_no_issues(self):
        """Test should_revise_or_handle_error routes to continue when no errors or revision needed."""
        state = {
            "has_fatal_error": False,
            "needs_revision": False,
        }

        result = should_revise_or_handle_error(state)
        assert result == "continue"

    def test_handle_fatal_error_sets_error_handler_node(self):
        """Test handle_fatal_error node sets current_node to error_handler."""
        state = {
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

    async def test_workflow_routes_to_error_handler_on_generation_failure(
        self, sample_phase2_state, mock_all_nodes
    ):
        """Test workflow routes to error_handler when generation fails."""
        mock_all_nodes["generate"].side_effect = lambda state: {
            **state,
            "last_error": "Generation failed",
            "has_fatal_error": True,
            "error_node": "generate",
            "current_node": "generate",
        }

        graph = create_phase2_graph()

        result = await graph.ainvoke(sample_phase2_state)

        assert result.get("has_fatal_error") is True
        assert result.get("error_node") == "generate"
        mock_all_nodes["generate"].assert_called_once()
        mock_all_nodes["extract"].assert_not_called()

    async def test_workflow_routes_to_error_handler_on_extraction_failure(
        self, sample_phase2_state, mock_all_nodes
    ):
        """Test workflow routes to error_handler when extraction fails."""
        mock_all_nodes["extract"].side_effect = lambda state: {
            **state,
            "last_error": "Extraction failed",
            "has_fatal_error": True,
            "error_node": "extract",
            "current_node": "extract_entities",
        }

        graph = create_phase2_graph()

        result = await graph.ainvoke(sample_phase2_state)

        assert result.get("has_fatal_error") is True
        assert result.get("error_node") == "extract"
        mock_all_nodes["generate"].assert_called_once()
        mock_all_nodes["extract"].assert_called()
        mock_all_nodes["commit"].assert_not_called()

    async def test_workflow_routes_to_error_handler_on_validation_failure(
        self, sample_phase2_state, mock_all_nodes
    ):
        """Test workflow routes to error_handler when validation triggers fatal error."""
        mock_all_nodes["validate"].side_effect = lambda state: {
            **state,
            "last_error": "Validation failed",
            "has_fatal_error": True,
            "error_node": "validate",
            "current_node": "validate",
        }

        graph = create_phase2_graph()

        result = await graph.ainvoke(sample_phase2_state)

        assert result.get("has_fatal_error") is True
        assert result.get("error_node") == "validate"
        mock_all_nodes["summarize"].assert_not_called()
        mock_all_nodes["finalize"].assert_not_called()

    async def test_workflow_routes_to_error_handler_on_revision_failure(
        self, sample_phase2_state, mock_all_nodes
    ):
        """Test workflow routes to error_handler when revision fails."""
        call_count = {"validate": 0}

        def validate_with_revision_then_error(state):
            call_count["validate"] += 1
            if call_count["validate"] == 1:
                return {
                    **state,
                    "needs_revision": True,
                    "contradictions": [
                        Contradiction(
                            type="test",
                            description="Test issue",
                            conflicting_chapters=[1],
                            severity="minor",
                        )
                    ],
                    "current_node": "validate",
                }
            else:
                return {
                    **state,
                    "has_fatal_error": True,
                    "last_error": "Revision max iterations",
                    "error_node": "revise",
                    "current_node": "validate",
                }

        mock_all_nodes["validate"].side_effect = validate_with_revision_then_error
        mock_all_nodes["revise"].side_effect = lambda state: {
            **state,
            "last_error": "Max revisions reached",
            "has_fatal_error": True,
            "error_node": "revise",
            "needs_revision": False,
            "current_node": "revise",
        }

        graph = create_phase2_graph()

        result = await graph.ainvoke(sample_phase2_state)

        assert result.get("has_fatal_error") is True
        mock_all_nodes["revise"].assert_called_once()
        mock_all_nodes["summarize"].assert_not_called()

    async def test_workflow_completes_successfully_without_errors(
        self, sample_phase2_state, mock_all_nodes
    ):
        """Test workflow completes when no fatal errors occur."""
        graph = create_phase2_graph()

        result = await graph.ainvoke(sample_phase2_state)

        assert result.get("has_fatal_error", False) is False
        assert result["current_node"] == "finalize"
        mock_all_nodes["generate"].assert_called_once()
        mock_all_nodes["extract"].assert_called()
        mock_all_nodes["commit"].assert_called()
        mock_all_nodes["validate"].assert_called()
        mock_all_nodes["summarize"].assert_called_once()
        mock_all_nodes["finalize"].assert_called_once()

    async def test_workflow_stops_at_first_fatal_error(
        self, sample_phase2_state, mock_all_nodes
    ):
        """Test workflow stops execution at first fatal error encountered."""
        mock_all_nodes["generate"].side_effect = lambda state: {
            **state,
            "draft_text": "Generated text",
            "current_node": "generate",
        }

        mock_all_nodes["extract"].side_effect = lambda state: {
            **state,
            "last_error": "First fatal error",
            "has_fatal_error": True,
            "error_node": "extract",
            "current_node": "extract_entities",
        }

        graph = create_phase2_graph()

        result = await graph.ainvoke(sample_phase2_state)

        assert result.get("has_fatal_error") is True
        assert result.get("error_node") == "extract"

        mock_all_nodes["generate"].assert_called_once()
        mock_all_nodes["extract"].assert_called()
        mock_all_nodes["commit"].assert_not_called()
        mock_all_nodes["validate"].assert_not_called()
        mock_all_nodes["summarize"].assert_not_called()
        mock_all_nodes["finalize"].assert_not_called()
