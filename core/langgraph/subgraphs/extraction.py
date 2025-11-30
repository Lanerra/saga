# core/langgraph/subgraphs/extraction.py
import structlog
from langgraph.graph import END, StateGraph

from core.langgraph.nodes.extraction_nodes import (
    consolidate_extraction,
    extract_characters,
    extract_events,
    extract_locations,
    extract_relationships,
)
from core.langgraph.state import NarrativeState

logger = structlog.get_logger(__name__)


def create_extraction_subgraph() -> StateGraph:
    """
    Create the extraction subgraph with SEQUENTIAL extraction.

    Extraction runs sequentially to avoid reducer-based accumulation issues:
    1. extract_characters: Clears state, extracts characters
    2. extract_locations: Appends locations to world_items
    3. extract_events: Appends events to world_items
    4. extract_relationships: Extracts relationships
    5. consolidate: Logs completion

    Sequential execution means no reducers needed, so state replacement
    works correctly and prevents cross-chapter accumulation.
    """
    workflow = StateGraph(NarrativeState)

    workflow.add_node("extract_characters", extract_characters)
    workflow.add_node("extract_locations", extract_locations)
    workflow.add_node("extract_events", extract_events)
    workflow.add_node("extract_relationships", extract_relationships)
    workflow.add_node("consolidate", consolidate_extraction)

    # Sequential execution (no parallel branches)
    workflow.set_entry_point("extract_characters")
    workflow.add_edge("extract_characters", "extract_locations")
    workflow.add_edge("extract_locations", "extract_events")
    workflow.add_edge("extract_events", "extract_relationships")
    workflow.add_edge("extract_relationships", "consolidate")
    workflow.add_edge("consolidate", END)

    return workflow.compile()
