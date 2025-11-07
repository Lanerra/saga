"""
LangGraph workflow for SAGA narrative generation.

This module wires together all nodes into complete workflow graphs
with conditional edges, revision loops, and checkpointing.

Phase 1 Workflow Structure:
    START
      ↓
    [extract_entities]
      ↓
    [commit_to_graph]
      ↓
    [validate_consistency]
      ↓
    {needs_revision?}
      ├─ Yes → [revise_placeholder] → (loop back)
      └─ No → END

Phase 2 Workflow Structure (COMPLETE):
    START
      ↓
    [generate_chapter]
      ↓
    [extract_entities]
      ↓
    [commit_to_graph]
      ↓
    [validate_consistency]
      ↓
    {needs_revision?}
      ├─ Yes → [revise_chapter] → (loop back to extract)
      └─ No → [summarize_chapter]
              ↓
            [finalize_chapter]
              ↓
            END

Migration Reference: docs/phase2_migration_plan.md - Step 2.5
"""

from typing import Literal

import structlog

from core.langgraph.nodes.commit_node import commit_to_graph
from core.langgraph.nodes.extraction_node import extract_entities
from core.langgraph.nodes.finalize_node import finalize_chapter
from core.langgraph.nodes.generation_node import generate_chapter
from core.langgraph.nodes.revision_node import revise_chapter
from core.langgraph.nodes.summary_node import summarize_chapter
from core.langgraph.nodes.validation_node import validate_consistency
from core.langgraph.state import NarrativeState
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

logger = structlog.get_logger(__name__)


def should_revise(state: NarrativeState) -> Literal["revise", "end"]:
    """
    Conditional edge function: Determine if chapter needs revision (Phase 1).

    Routes to:
    - "revise": If needs_revision=True and iterations < max_iterations
    - "end": If needs_revision=False or max iterations reached

    Note: Phase 1 workflow only. Phase 2 uses should_revise_or_continue().

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


def should_revise_or_continue(
    state: NarrativeState,
) -> Literal["revise", "summarize"]:
    """
    Conditional edge function: Determine if chapter needs revision (Phase 2).

    Routes to:
    - "revise": If needs_revision=True and iterations < max_iterations
    - "summarize": If needs_revision=False or max iterations reached

    Note: This is the Phase 2 version that routes to summarize instead of end.

    Args:
        state: Current narrative state

    Returns:
        Next node name ("revise" or "summarize")
    """
    needs_revision = state.get("needs_revision", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    force_continue = state.get("force_continue", False)

    logger.info(
        "should_revise_or_continue: routing decision",
        needs_revision=needs_revision,
        iteration=iteration_count,
        max_iterations=max_iterations,
        force_continue=force_continue,
    )

    # If force_continue is set, skip revision and go to summary
    if force_continue:
        logger.info(
            "should_revise_or_continue: force_continue enabled, routing to summarize"
        )
        return "summarize"

    # If we've hit max iterations, go to summary
    if iteration_count >= max_iterations:
        logger.warning(
            "should_revise_or_continue: max iterations reached, routing to summarize",
            iteration_count=iteration_count,
            max_iterations=max_iterations,
        )
        return "summarize"

    # If needs revision and under max iterations, revise
    if needs_revision:
        logger.info(
            "should_revise_or_continue: revision needed, routing to revise node"
        )
        return "revise"

    # Otherwise, proceed to summary
    logger.info("should_revise_or_continue: no revision needed, routing to summarize")
    return "summarize"


def create_phase2_graph(checkpointer=None) -> StateGraph:
    """
    Create Phase 2 LangGraph workflow (COMPLETE).

    This is the full narrative generation workflow including:
    - Generation: Create chapter from outline and context
    - Extraction: Extract entities and relationships
    - Commitment: Deduplicate and commit to Neo4j
    - Validation: Check for contradictions and quality issues
    - Revision: Fix issues (conditional, with iteration limit)
    - Summarization: Create concise chapter summary
    - Finalization: Persist to filesystem and Neo4j

    Workflow:
        generate → extract → commit → validate → {revise? revise → extract : summarize} → finalize → END

    Migration Reference: docs/phase2_migration_plan.md - Step 2.5

    Args:
        checkpointer: Optional checkpoint saver (SqliteSaver, PostgresSaver, etc.)
                     If None, no checkpointing is enabled.

    Returns:
        Compiled LangGraph StateGraph ready for execution
    """
    logger.info("create_phase2_graph: building complete workflow graph")

    # Create graph
    workflow = StateGraph(NarrativeState)

    # Add all nodes
    workflow.add_node("generate", generate_chapter)
    workflow.add_node("extract", extract_entities)
    workflow.add_node("commit", commit_to_graph)
    workflow.add_node("validate", validate_consistency)
    workflow.add_node("revise", revise_chapter)
    workflow.add_node("summarize", summarize_chapter)
    workflow.add_node("finalize", finalize_chapter)

    # Define linear edges
    workflow.add_edge("generate", "extract")
    workflow.add_edge("extract", "commit")
    workflow.add_edge("commit", "validate")

    # Conditional edge: validate → (revise or summarize)
    workflow.add_conditional_edges(
        "validate",
        should_revise_or_continue,
        {
            "revise": "revise",
            "summarize": "summarize",
        },
    )

    # After revision, loop back to extract
    # (revised text needs to be re-extracted and re-validated)
    workflow.add_edge("revise", "extract")

    # Linear flow to completion
    workflow.add_edge("summarize", "finalize")
    workflow.add_edge("finalize", END)

    # Set entry point
    workflow.set_entry_point("generate")

    logger.info(
        "create_phase2_graph: graph built successfully",
        nodes=[
            "generate",
            "extract",
            "commit",
            "validate",
            "revise",
            "summarize",
            "finalize",
        ],
        entry_point="generate",
    )

    # Compile graph
    if checkpointer:
        logger.info("create_phase2_graph: compiling with checkpointing enabled")
        compiled_graph = workflow.compile(checkpointer=checkpointer)
    else:
        logger.info("create_phase2_graph: compiling without checkpointing")
        compiled_graph = workflow.compile()

    return compiled_graph


def create_checkpointer(db_path: str = "./checkpoints/saga.db") -> AsyncSqliteSaver:
    """
    Create an async SQLite checkpoint saver for workflow persistence.

    This enables:
    - Resuming workflows after crashes
    - Time-travel debugging
    - Replay from any checkpoint

    Args:
        db_path: Path to SQLite database file

    Returns:
        AsyncSqliteSaver instance
    """
    import os

    # Ensure checkpoints directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    logger.info(
        "create_checkpointer: creating async SQLite checkpointer",
        db_path=db_path,
    )

    # Create async checkpointer
    # AsyncSqliteSaver.from_conn_string handles the async connection
    checkpointer = AsyncSqliteSaver.from_conn_string(db_path)

    return checkpointer


__all__ = [
    "create_phase1_graph",
    "create_phase2_graph",
    "create_checkpointer",
    "should_revise",
    "should_revise_or_continue",
]
