# core/langgraph/subgraphs/generation.py
"""Build the scene-based generation subgraph for SAGA.

This subgraph generates scene drafts for a chapter:
plan scenes → retrieve context → draft scenes.

Chapter assembly is intentionally performed at the top-level workflow after scene
embeddings are generated.
"""

from typing import Literal

import structlog
from langgraph.graph import END, StateGraph  # type: ignore[import-not-found, attr-defined]

from core.langgraph.nodes.context_retrieval_node import retrieve_context
from core.langgraph.nodes.scene_generation_node import draft_scene
from core.langgraph.nodes.scene_planning_node import plan_scenes
from core.langgraph.state import NarrativeState

logger = structlog.get_logger(__name__)


def _should_continue_or_error(state: NarrativeState) -> Literal["continue", "error"]:
    """Gate on has_fatal_error before proceeding to the next node."""
    if state.get("has_fatal_error", False):
        return "error"
    return "continue"


def should_continue_scenes(state: NarrativeState) -> Literal["continue", "end", "error"]:
    """Route within the generation subgraph based on scene progress.

    Args:
        state: Workflow state. This function reads:
            - has_fatal_error: Whether a fatal error has occurred.
            - current_scene_index: Index of the next scene to draft.
            - chapter_plan_scene_count: Total number of scenes in the chapter plan.

    Returns:
        "error" if a fatal error occurred, "continue" to draft another scene,
        or "end" to end the subgraph.
    """
    if state.get("has_fatal_error", False):
        return "error"

    scene_count = state.get("chapter_plan_scene_count", 0)
    if isinstance(scene_count, bool) or not isinstance(scene_count, int):
        raise TypeError("chapter_plan_scene_count must be an int")

    if scene_count <= 0:
        return "end"

    current_index = state.get("current_scene_index", 0)

    if current_index < scene_count:
        return "continue"

    return "end"


def create_generation_subgraph() -> StateGraph:
    """Create and compile the scene-based generation subgraph.

    Returns:
        A compiled `StateGraph` implementing the scene-based generation phase.
    """
    workflow = StateGraph(NarrativeState)

    workflow.add_node("plan_scenes", plan_scenes)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("draft_scene", draft_scene)

    workflow.set_entry_point("plan_scenes")

    workflow.add_conditional_edges(
        "plan_scenes",
        _should_continue_or_error,
        {"continue": "retrieve_context", "error": END},
    )
    workflow.add_conditional_edges(
        "retrieve_context",
        _should_continue_or_error,
        {"continue": "draft_scene", "error": END},
    )

    workflow.add_conditional_edges(
        "draft_scene",
        should_continue_scenes,
        {"continue": "retrieve_context", "end": END, "error": END},
    )

    return workflow.compile()


__all__ = [
    "create_generation_subgraph",
    "should_continue_scenes",
]
