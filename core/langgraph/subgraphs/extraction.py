# core/langgraph/subgraphs/extraction.py
import structlog
from langgraph.graph import END, StateGraph

from core.langgraph.nodes.extraction_nodes import (
    extract_characters,
    extract_locations,
    extract_events,
    extract_relationships,
    consolidate_extraction,
)
from core.langgraph.state import NarrativeState

logger = structlog.get_logger(__name__)


def extract_router(state: NarrativeState) -> NarrativeState:
    """
    Prepare inputs for parallel extraction.
    """
    logger.info("extract_router: preparing for parallel extraction")
    return state


def create_extraction_subgraph() -> StateGraph:
    """
    Create the extraction subgraph.
    """
    workflow = StateGraph(NarrativeState)

    workflow.add_node("extract_router", extract_router)
    workflow.add_node("extract_characters", extract_characters)
    workflow.add_node("extract_locations", extract_locations)
    workflow.add_node("extract_events", extract_events)
    workflow.add_node("extract_relationships", extract_relationships)
    workflow.add_node("consolidate", consolidate_extraction)

    workflow.set_entry_point("extract_router")

    # Parallel execution
    workflow.add_edge("extract_router", "extract_characters")
    workflow.add_edge("extract_router", "extract_locations")
    workflow.add_edge("extract_router", "extract_events")
    workflow.add_edge("extract_router", "extract_relationships")

    workflow.add_edge("extract_characters", "consolidate")
    workflow.add_edge("extract_locations", "consolidate")
    workflow.add_edge("extract_events", "consolidate")
    workflow.add_edge("extract_relationships", "consolidate")

    workflow.add_edge("consolidate", END)

    return workflow.compile()
