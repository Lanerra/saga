# core/langgraph/subgraphs/generation.py
from typing import Literal

import structlog
from langgraph.graph import END, StateGraph  # type: ignore

from core.langgraph.nodes.assemble_chapter_node import assemble_chapter
from core.langgraph.nodes.context_retrieval_node import retrieve_context
from core.langgraph.nodes.scene_generation_node import draft_scene

# Import new nodes
from core.langgraph.nodes.scene_planning_node import plan_scenes
from core.langgraph.state import NarrativeState

logger = structlog.get_logger(__name__)


def should_continue_scenes(state: NarrativeState) -> Literal["continue", "end"]:
    """
    Determine if there are more scenes to generate.
    """
    current_index = state.get("current_scene_index", 0)
    chapter_plan = state.get("chapter_plan", [])

    if not chapter_plan:
        # If no plan, we can't continue. This shouldn't happen if plan_scenes works.
        return "end"

    if current_index < len(chapter_plan):
        return "continue"

    return "end"


def create_generation_subgraph() -> StateGraph:
    """
    Create the generation subgraph.
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
