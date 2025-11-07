"""
LangGraph workflow for SAGA narrative generation.

This module wires together all Phase 1 nodes into a complete workflow graph
with conditional edges, revision loops, and checkpointing.

Workflow Structure:
    START
      ↓
    [generate_chapter]  (placeholder - will be added in Phase 2)
      ↓
    [extract_entities]
      ↓
    [commit_to_graph]
      ↓
    [validate_consistency]
      ↓
    {needs_revision?}
      ├─ Yes → [revise_chapter] → (loop back)
      └─ No → END

Phase 1 Implementation:
    - extract_entities: Extract entities/relationships from draft text
    - commit_to_graph: Deduplicate and commit to Neo4j
    - validate_consistency: Check for contradictions
    - Conditional routing based on validation results
"""

import structlog
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from core.langgraph.state import NarrativeState
from core.langgraph.nodes.extraction_node import extract_entities
from core.langgraph.nodes.commit_node import commit_to_graph
from core.langgraph.nodes.validation_node import validate_consistency

logger = structlog.get_logger(__name__)


def should_revise(state: NarrativeState) -> Literal["revise", "end"]:
    """
    Conditional edge function: Determine if chapter needs revision.

    Routes to:
    - "revise": If needs_revision=True and iterations < max_iterations
    - "end": If needs_revision=False or max iterations reached

    Args:
        state: Current narrative state

    Returns:
        Next node name ("revise" or "end")
    """
    needs_revision = state.get("needs_revision", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    force_continue = state.get("force_continue", False)

    logger.info(
        "should_revise: routing decision",
        needs_revision=needs_revision,
        iteration=iteration_count,
        max_iterations=max_iterations,
        force_continue=force_continue,
    )

    # If force_continue is set, skip revision
    if force_continue:
        logger.info("should_revise: force_continue enabled, skipping revision")
        return "end"

    # If we've hit max iterations, end workflow
    if iteration_count >= max_iterations:
        logger.warning(
            "should_revise: max iterations reached, ending workflow",
            iteration_count=iteration_count,
            max_iterations=max_iterations,
        )
        return "end"

    # If needs revision and under max iterations, revise
    if needs_revision:
        logger.info("should_revise: revision needed, routing to revise node")
        return "revise"

    # Otherwise, we're done
    logger.info("should_revise: no revision needed, ending workflow")
    return "end"


def create_phase1_graph(checkpointer=None) -> StateGraph:
    """
    Create Phase 1 LangGraph workflow.

    This is a minimal workflow focusing on extraction, commit, and validation.
    Generation and revision nodes will be added in Phase 2.

    Args:
        checkpointer: Optional checkpoint saver (SqliteSaver, PostgresSaver, etc.)
                     If None, no checkpointing is enabled.

    Returns:
        Compiled LangGraph StateGraph ready for execution
    """
    logger.info("create_phase1_graph: building workflow graph")

    # Create graph
    workflow = StateGraph(NarrativeState)

    # Add nodes
    # Note: For Phase 1, we assume draft_text is already in state
    # Generation node will be added in Phase 2
    workflow.add_node("extract", extract_entities)
    workflow.add_node("commit", commit_to_graph)
    workflow.add_node("validate", validate_consistency)

    # Placeholder for revision node (Phase 2)
    # For now, we'll just loop back to extract if revision is needed
    def placeholder_revise(state: NarrativeState) -> NarrativeState:
        """Placeholder revision node - increments iteration counter."""
        logger.warning(
            "placeholder_revise: revision requested but full revision "
            "node not yet implemented (Phase 2)"
        )
        return {
            **state,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "current_node": "revise_placeholder",
            # In Phase 2, this will trigger actual revision
            # For now, we just increment and continue
            "needs_revision": False,  # Prevent infinite loop
        }

    workflow.add_node("revise", placeholder_revise)

    # Define edges
    # Linear flow: extract → commit → validate
    workflow.add_edge("extract", "commit")
    workflow.add_edge("commit", "validate")

    # Conditional edge: validate → (revise or end)
    workflow.add_conditional_edges(
        "validate",
        should_revise,
        {
            "revise": "revise",
            "end": END,
        },
    )

    # After revision, loop back to extract
    # (In Phase 2, this will go to regenerate instead)
    workflow.add_edge("revise", "extract")

    # Set entry point
    workflow.set_entry_point("extract")

    logger.info(
        "create_phase1_graph: graph built successfully",
        nodes=["extract", "commit", "validate", "revise"],
        entry_point="extract",
    )

    # Compile graph
    if checkpointer:
        logger.info("create_phase1_graph: compiling with checkpointing enabled")
        compiled_graph = workflow.compile(checkpointer=checkpointer)
    else:
        logger.info("create_phase1_graph: compiling without checkpointing")
        compiled_graph = workflow.compile()

    return compiled_graph


def create_checkpointer(db_path: str = "./checkpoints/saga.db") -> SqliteSaver:
    """
    Create a SQLite checkpoint saver for workflow persistence.

    This enables:
    - Resuming workflows after crashes
    - Time-travel debugging
    - Replay from any checkpoint

    Args:
        db_path: Path to SQLite database file

    Returns:
        SqliteSaver instance
    """
    import os

    # Ensure checkpoints directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    logger.info(
        "create_checkpointer: creating SQLite checkpointer",
        db_path=db_path,
    )

    # Create checkpointer
    # Note: SqliteSaver handles connection lifecycle automatically
    checkpointer = SqliteSaver.from_conn_string(db_path)

    return checkpointer


__all__ = [
    "create_phase1_graph",
    "create_checkpointer",
    "should_revise",
]
