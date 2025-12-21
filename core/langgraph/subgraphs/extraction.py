# core/langgraph/subgraphs/extraction.py
"""Build the extraction subgraph for SAGA.

Extraction is intentionally sequential to avoid cross-chapter accumulation from
reducer-based merges.
"""

import structlog
from langgraph.graph import END, StateGraph  # type: ignore

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
    """Create and compile the extraction subgraph.

    Order of operations:
        1. `extract_characters` resets extraction buckets for the chapter.
        2. `extract_locations` appends to `world_items`.
        3. `extract_events` appends to `world_items`.
        4. `extract_relationships` sets `extracted_relationships` and may add
           entities mentioned only in relationships.
        5. `consolidate_extraction` externalizes and logs extraction completion.

    Returns:
        A compiled `StateGraph` implementing the extraction phase.
    """
    workflow = StateGraph(NarrativeState)

    workflow.add_node("extract_characters", extract_characters)
    workflow.add_node("extract_locations", extract_locations)
    workflow.add_node("extract_events", extract_events)
    workflow.add_node("extract_relationships", extract_relationships)
    workflow.add_node("consolidate", consolidate_extraction)

    workflow.set_entry_point("extract_characters")
    workflow.add_edge("extract_characters", "extract_locations")
    workflow.add_edge("extract_locations", "extract_events")
    workflow.add_edge("extract_events", "extract_relationships")
    workflow.add_edge("extract_relationships", "consolidate")
    workflow.add_edge("consolidate", END)

    return workflow.compile()
