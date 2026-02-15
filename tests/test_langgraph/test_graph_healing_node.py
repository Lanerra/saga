# tests/test_langgraph/test_graph_healing_node.py
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.langgraph.nodes.graph_healing_node import heal_graph
from core.langgraph.state import create_initial_state


@pytest.mark.asyncio
async def test_heal_graph_clamps_provisional_count_non_negative(tmp_path) -> None:
    """
    Regression test for LANGGRAPH-029 / remediation 9.1 #5.

    `heal_graph()` must never return a negative `provisional_count`, even if the
    healing service returns a provisional_count value that is smaller than the
    number of nodes graduated in the same run.
    """
    state = create_initial_state(
        project_id="test_project",
        title="Test Novel",
        genre="Fantasy",
        theme="Heroism",
        setting="A magical world",
        target_word_count=50000,
        total_chapters=10,
        project_dir=str(tmp_path / "test_project"),
        protagonist_name="Test Hero",
    )
    state["current_chapter"] = 3

    results = {
        "nodes_graduated": 7,
        "nodes_merged": 0,
        "nodes_enriched": 0,
        "nodes_removed": 0,
        # Smaller than nodes_graduated -> would underflow without clamping.
        "provisional_count": 2,
        "actions": [],
    }

    with patch(
        "core.langgraph.nodes.graph_healing_node.graph_healing_service.heal_graph",
        new=AsyncMock(return_value=results),
    ):
        out = await heal_graph(state)

    assert out["current_node"] == "heal_graph"
    assert out["last_healing_chapter"] == 3
    assert out["last_error"] is None

    # Clamped
    assert out["provisional_count"] == 0

    # New observability fields (LANGGRAPH-025 hardening): should exist and be empty when no warnings.
    assert out["last_healing_warnings"] == []
    assert out["last_apoc_available"] is None


@pytest.mark.asyncio
async def test_heal_graph_accumulates_running_totals(tmp_path) -> None:
    """Running totals for graduated/merged/enriched/removed should accumulate."""
    state = create_initial_state(
        project_id="test_project",
        title="Test Novel",
        genre="Fantasy",
        theme="Heroism",
        setting="A magical world",
        target_word_count=50000,
        total_chapters=10,
        project_dir=str(tmp_path / "test_project"),
        protagonist_name="Test Hero",
    )
    state["current_chapter"] = 4

    state["nodes_graduated"] = 10
    state["nodes_merged"] = 5
    state["nodes_enriched"] = 2
    state["nodes_removed"] = 1

    results = {
        "nodes_graduated": 1,
        "nodes_merged": 2,
        "nodes_enriched": 3,
        "nodes_removed": 4,
        "provisional_count": 1,
        "actions": [],
    }

    with patch(
        "core.langgraph.nodes.graph_healing_node.graph_healing_service.heal_graph",
        new=AsyncMock(return_value=results),
    ):
        out = await heal_graph(state)

    assert out["provisional_count"] == 0

    assert out["last_healing_warnings"] == []
    assert out["last_apoc_available"] is None

    assert out["nodes_graduated"] == 11
    assert out["nodes_merged"] == 7
    assert out["nodes_enriched"] == 5
    assert out["nodes_removed"] == 5


@pytest.mark.asyncio
async def test_heal_graph_surfaces_healing_warnings_in_state_and_logs_once(tmp_path) -> None:
    """
    Regression test for LANGGRAPH-025 / remediation plan item 11.

    If graph healing returns warnings, the node must not "silently degrade"—it must
    surface warnings into state and log them.
    """
    state = create_initial_state(
        project_id="test_project",
        title="Test Novel",
        genre="Fantasy",
        theme="Heroism",
        setting="A magical world",
        target_word_count=50000,
        total_chapters=10,
        project_dir=str(tmp_path / "test_project"),
        protagonist_name="Test Hero",
    )
    state["current_chapter"] = 5

    results = {
        "nodes_graduated": 0,
        "nodes_merged": 0,
        "nodes_enriched": 0,
        "nodes_removed": 0,
        "provisional_count": 0,
        "actions": [],
        "apoc_available": True,
        "warnings": [{"type": "healing_warning", "message": "Example healing warning"}],
    }

    with (
        patch(
            "core.langgraph.nodes.graph_healing_node.graph_healing_service.heal_graph",
            new=AsyncMock(return_value=results),
        ),
        patch("core.langgraph.nodes.graph_healing_node.logger.warning", new=MagicMock()) as warn,
    ):
        out = await heal_graph(state)

    assert out["current_node"] == "heal_graph"
    assert out["last_error"] is None
    assert out["last_healing_warnings"] == results["warnings"]
    assert out["last_apoc_available"] is True

    # The node should log warnings (at least once for this call).
    assert warn.call_count == 1
