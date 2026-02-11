# core/langgraph/subgraphs/scene_extraction.py
"""Build the scene-level extraction subgraph for SAGA.

This subgraph replaces chapter-level extraction to keep prompts small.
"""

from typing import Literal

import structlog
from langgraph.graph import END, StateGraph  # type: ignore[import-not-found, attr-defined]

from core.langgraph.nodes.extraction_nodes import consolidate_extraction
from core.langgraph.nodes.scene_extraction import extract_from_scenes
from core.langgraph.state import NarrativeState

logger = structlog.get_logger(__name__)


def _should_continue_or_error(state: NarrativeState) -> Literal["continue", "error"]:
    """Gate on has_fatal_error before proceeding to the next node."""
    if state.get("has_fatal_error", False):
        return "error"
    return "continue"


def create_scene_extraction_subgraph() -> StateGraph:
    """Create and compile the scene-level extraction subgraph.

    Order of operations:
        1. `extract_from_scenes` - Process each scene individually, consolidate
        2. `consolidate` - Externalize results to disk (uses existing consolidate_extraction)

    Returns:
        A compiled StateGraph implementing scene-level extraction.
    """
    workflow = StateGraph(NarrativeState)

    workflow.add_node("extract_from_scenes", extract_from_scenes)
    workflow.add_node("consolidate", consolidate_extraction)

    workflow.set_entry_point("extract_from_scenes")
    workflow.add_conditional_edges(
        "extract_from_scenes",
        _should_continue_or_error,
        {"continue": "consolidate", "error": END},
    )
    workflow.add_edge("consolidate", END)

    return workflow.compile()
