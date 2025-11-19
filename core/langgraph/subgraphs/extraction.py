# core/langgraph/subgraphs/extraction.py
import structlog
from langgraph.graph import END, StateGraph

from core.langgraph.nodes.extraction_node import (
    extract_entities as original_extract_entities,
)
from core.langgraph.state import NarrativeState

logger = structlog.get_logger(__name__)


def extract_router(state: NarrativeState) -> NarrativeState:
    """
    Prepare inputs for parallel extraction.
    """
    logger.info("extract_router: preparing for parallel extraction")
    return state


def extract_characters(state: NarrativeState) -> NarrativeState:
    """
    Extract character details.
    """
    logger.info("extract_characters: extracting characters")
    # Placeholder: In full implementation, this would call a specific LLM prompt
    return state


def extract_locations(state: NarrativeState) -> NarrativeState:
    """
    Extract location details.
    """
    logger.info("extract_locations: extracting locations")
    return state


def extract_events(state: NarrativeState) -> NarrativeState:
    """
    Extract events.
    """
    logger.info("extract_events: extracting events")
    return state


def extract_relationships(state: NarrativeState) -> NarrativeState:
    """
    Extract relationships.
    """
    logger.info("extract_relationships: extracting relationships")
    return state


def consolidate_extraction(state: NarrativeState) -> NarrativeState:
    """
    Merge results from parallel extractions.
    """
    logger.info("consolidate_extraction: merging results")
    # For this initial refactor, we call the original monolithic extraction here
    # to ensure we still get valid results while the parallel nodes are placeholders.
    return original_extract_entities(state)


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

    # In a real parallel execution, we would use map/reduce or parallel branches.
    # LangGraph supports parallel execution by adding multiple edges from one node.
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
