# core/langgraph/subgraphs/validation.py
import structlog
from langgraph.graph import END, StateGraph

from core.langgraph.nodes.validation_node import (
    validate_consistency as original_validate_consistency,
)
from core.langgraph.state import NarrativeState

logger = structlog.get_logger(__name__)


def validate_consistency(state: NarrativeState) -> NarrativeState:
    """
    Check against graph constraints.
    """
    logger.info("validate_consistency: checking graph constraints")
    return original_validate_consistency(state)


def evaluate_quality(state: NarrativeState) -> NarrativeState:
    """
    Analyze prose quality, pacing, and tone.
    """
    logger.info("evaluate_quality: checking prose quality")
    # Placeholder for future quality checks
    return state


def detect_contradictions(state: NarrativeState) -> NarrativeState:
    """
    Specific logic for narrative contradictions.
    """
    logger.info("detect_contradictions: checking for narrative contradictions")
    # Placeholder
    return state


def create_validation_subgraph() -> StateGraph:
    """
    Create the validation subgraph.
    """
    workflow = StateGraph(NarrativeState)

    workflow.add_node("validate_consistency", validate_consistency)
    workflow.add_node("evaluate_quality", evaluate_quality)
    workflow.add_node("detect_contradictions", detect_contradictions)

    workflow.set_entry_point("validate_consistency")

    workflow.add_edge("validate_consistency", "evaluate_quality")
    workflow.add_edge("evaluate_quality", "detect_contradictions")
    workflow.add_edge("detect_contradictions", END)

    return workflow.compile()
