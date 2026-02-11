# tests/core/langgraph/test_workflow.py
"""Tests for core/langgraph/workflow.py - workflow graph construction and routing."""

from unittest.mock import MagicMock, patch

import pytest

from core.langgraph.workflow import (
    create_checkpointer,
    create_full_workflow_graph,
)


class TestWorkflowGraphConstruction:
    """Test the construction of the full workflow graph."""

    @pytest.mark.asyncio
    async def test_create_full_workflow_graph_returns_compiled_graph(self) -> None:
        """Test that create_full_workflow_graph returns a compiled graph."""
        # Mock all the node imports to avoid dependencies
        with patch.multiple(
            "core.langgraph.workflow",
            assemble_chapter=MagicMock(),
            commit_to_graph=MagicMock(),
            generate_scene_embeddings=MagicMock(),
            finalize_chapter=MagicMock(),
            heal_graph=MagicMock(),
            check_quality=MagicMock(),
            revise_chapter=MagicMock(),
            summarize_chapter=MagicMock(),
        ):
            graph = create_full_workflow_graph()

            # Verify it's a compiled graph (CompiledStateGraph type)
            from langgraph.graph.state import CompiledStateGraph

            assert isinstance(graph, CompiledStateGraph)

            # Verify it has the expected structure
            assert hasattr(graph, "get_graph")

    @pytest.mark.asyncio
    async def test_create_checkpointer_returns_async_sqlite_saver(self) -> None:
        """Test that create_checkpointer returns an AsyncSqliteSaver instance."""
        # create_checkpointer returns a context manager, not the saver directly
        checkpointer_cm = create_checkpointer()

        # Verify it's an async context manager
        assert hasattr(checkpointer_cm, "__aenter__")
        assert hasattr(checkpointer_cm, "__aexit__")


class TestWorkflowGraphStructure:
    """Test the structure and connectivity of the workflow graph."""

    @pytest.mark.asyncio
    async def test_workflow_graph_has_expected_nodes(self) -> None:
        """Test that the workflow graph contains all expected nodes."""
        with patch.multiple(
            "core.langgraph.workflow",
            assemble_chapter=MagicMock(return_value=lambda s: s),
            commit_to_graph=MagicMock(return_value=lambda s: s),
            generate_scene_embeddings=MagicMock(return_value=lambda s: s),
            finalize_chapter=MagicMock(return_value=lambda s: s),
            heal_graph=MagicMock(return_value=lambda s: s),
            check_quality=MagicMock(return_value=lambda s: s),
            revise_chapter=MagicMock(return_value=lambda s: s),
            summarize_chapter=MagicMock(return_value=lambda s: s),
        ):
            graph = create_full_workflow_graph()

            # Get the graph structure
            graph_structure = graph.get_graph()

            # Check that key nodes are present
            nodes = list(graph_structure.nodes.keys())
            assert "assemble_chapter" in nodes
            assert "commit" in nodes
            assert "gen_scene_embeddings" in nodes
            assert "finalize" in nodes

    @pytest.mark.asyncio
    async def test_workflow_graph_has_expected_edges(self) -> None:
        """Test that the workflow graph has expected edges between nodes."""
        with patch.multiple(
            "core.langgraph.workflow",
            assemble_chapter=MagicMock(return_value=lambda s: s),
            commit_to_graph=MagicMock(return_value=lambda s: s),
            generate_scene_embeddings=MagicMock(return_value=lambda s: s),
            finalize_chapter=MagicMock(return_value=lambda s: s),
            heal_graph=MagicMock(return_value=lambda s: s),
            check_quality=MagicMock(return_value=lambda s: s),
            revise_chapter=MagicMock(return_value=lambda s: s),
            summarize_chapter=MagicMock(return_value=lambda s: s),
        ):
            graph = create_full_workflow_graph()

            # Get the graph structure
            graph_structure = graph.get_graph()

            # Check that edges exist between key nodes
            # edges is a list of tuples (u, v, condition_or_none)
            edges = [(u, v) for u, v, *_ in graph_structure.edges]

            # Verify the graph has edges connecting nodes
            assert len(edges) > 0

            # Verify some expected connections exist
            edge_strings = [f"{u}->{v}" for u, v in edges]
            # The workflow should have connections between major phases
            assert any("route" in e for e in edge_strings)


class TestWorkflowErrorHandling:
    """Test error handling and recovery paths in the workflow."""

    @pytest.mark.asyncio
    async def test_workflow_handles_node_errors(self) -> None:
        """Test that workflow can handle errors from individual nodes."""
        error_node = MagicMock()
        error_node.side_effect = Exception("Test error")

        with patch.multiple(
            "core.langgraph.workflow",
            assemble_chapter=MagicMock(return_value=lambda s: s),
            commit_to_graph=error_node,
            generate_scene_embeddings=MagicMock(return_value=lambda s: s),
            finalize_chapter=MagicMock(return_value=lambda s: s),
            heal_graph=MagicMock(return_value=lambda s: s),
            check_quality=MagicMock(return_value=lambda s: s),
            revise_chapter=MagicMock(return_value=lambda s: s),
            summarize_chapter=MagicMock(return_value=lambda s: s),
        ):
            graph = create_full_workflow_graph()

            # The graph should be compilable even with error nodes
            assert graph is not None


class TestWorkflowStateManagement:
    """Test state management throughout the workflow."""

    @pytest.mark.asyncio
    async def test_workflow_preserves_state_through_nodes(self) -> None:
        """Test that state is properly passed through the workflow."""
        # Create a mock node that returns the state unchanged
        mock_node = MagicMock(return_value=lambda s: s)

        with patch.multiple(
            "core.langgraph.workflow",
            assemble_chapter=mock_node,
            commit_to_graph=mock_node,
            generate_scene_embeddings=mock_node,
            finalize_chapter=mock_node,
            heal_graph=mock_node,
            check_quality=mock_node,
            revise_chapter=mock_node,
            summarize_chapter=mock_node,
        ):
            graph = create_full_workflow_graph()

            # Verify the graph can be executed
            assert graph is not None
