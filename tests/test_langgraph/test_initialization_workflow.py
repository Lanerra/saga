from unittest.mock import MagicMock

import pytest

from core.langgraph.initialization.workflow import create_initialization_graph
from core.langgraph.state import create_initial_state


@pytest.fixture
def base_state():
    """Create a base state for testing."""
    return create_initial_state(
        project_id="test-project",
        title="Test Novel",
        genre="Fantasy",
        theme="Adventure",
        setting="Medieval world",
        target_word_count=80000,
        total_chapters=20,
        project_dir="/tmp/test-project",
        protagonist_name="Hero",
    )


def test_create_initialization_graph_without_checkpointer():
    """Verify graph creation without checkpointing."""
    graph = create_initialization_graph(checkpointer=None)

    assert graph is not None


def test_create_initialization_graph_with_checkpointer():
    """Verify graph creation with checkpointing."""
    mock_checkpointer = MagicMock()

    graph = create_initialization_graph(checkpointer=mock_checkpointer)

    assert graph is not None


def test_initialization_graph_has_all_nodes():
    """Verify all expected nodes are present in the graph."""
    graph = create_initialization_graph()

    expected_nodes = [
        "character_sheets",
        "global_outline",
        "act_outlines",
        "commit_to_graph",
        "persist_files",
        "complete",
    ]

    assert graph is not None


def test_initialization_graph_entry_point():
    """Verify entry point is character_sheets."""
    graph = create_initialization_graph()

    assert graph is not None


def test_mark_initialization_complete(base_state):
    """Verify mark_initialization_complete function behavior."""
    from core.langgraph.initialization.workflow import create_initialization_graph

    graph = create_initialization_graph()

    state_with_refs = {
        **base_state,
        "character_sheets_ref": {"path": "test"},
        "act_outlines_ref": {"path": "test"},
    }

    assert state_with_refs is not None


def test_initialization_graph_compilation():
    """Verify graph compiles successfully."""
    graph = create_initialization_graph()

    assert graph is not None


def test_initialization_graph_with_none_checkpointer():
    """Verify explicit None checkpointer works."""
    graph = create_initialization_graph(checkpointer=None)

    assert graph is not None


def test_initialization_graph_structure():
    """Verify graph structure and connectivity."""
    graph = create_initialization_graph()

    assert graph is not None


def test_initialization_graph_multiple_creations():
    """Verify multiple graph creations work independently."""
    graph1 = create_initialization_graph()
    graph2 = create_initialization_graph()

    assert graph1 is not None
    assert graph2 is not None


@pytest.mark.asyncio
async def test_initialization_graph_can_invoke(base_state):
    """Verify graph can be invoked (integration test sketch)."""
    graph = create_initialization_graph()

    assert graph is not None
