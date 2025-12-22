# core/langgraph/subgraphs/generation.py
"""Build the scene-based generation subgraph for SAGA.

This subgraph is the canonical chapter generation implementation:
plan scenes → retrieve context → draft scenes → assemble chapter.
"""

from typing import Literal

import structlog
from langgraph.graph import END, StateGraph  # type: ignore

from core.langgraph.nodes.assemble_chapter_node import assemble_chapter
from core.langgraph.nodes.context_retrieval_node import retrieve_context
from core.langgraph.nodes.scene_generation_node import draft_scene
from core.langgraph.nodes.scene_planning_node import plan_scenes
from core.langgraph.state import NarrativeState

logger = structlog.get_logger(__name__)


def should_continue_scenes(state: NarrativeState) -> Literal["continue", "end"]:
    """Route within the generation subgraph based on scene progress.

    Args:
        state: Workflow state. This function reads:
            - current_scene_index: Index of the next scene to draft.
            - chapter_plan: Scene plan used to bound generation.

    Returns:
        "continue" to draft another scene, or "end" to assemble the chapter.
    """
    current_index = state.get("current_scene_index", 0)
    chapter_plan = state.get("chapter_plan", [])

    if not chapter_plan:
        return "end"

    if current_index < len(chapter_plan):
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
    workflow.add_node("assemble_chapter", assemble_chapter)

    workflow.set_entry_point("plan_scenes")

    workflow.add_edge("plan_scenes", "retrieve_context")
    workflow.add_edge("retrieve_context", "draft_scene")

    workflow.add_conditional_edges(
        "draft_scene",
        should_continue_scenes,
        {"continue": "retrieve_context", "end": "assemble_chapter"},
    )

    workflow.add_edge("assemble_chapter", END)

    return workflow.compile()


def generate_chapter() -> StateGraph:
    """Return the compiled generation subgraph used by workflows."""
    return create_generation_subgraph()


__all__ = [
    "create_generation_subgraph",
    "generate_chapter",
    "should_continue_scenes",
]
