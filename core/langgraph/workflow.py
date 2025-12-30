# core/langgraph/workflow.py
"""
Build LangGraph workflows for SAGA narrative generation.

This module wires node callables and subgraphs into executable workflow graphs,
including checkpointing and revision/error routing.
"""

from typing import Any, Literal

import structlog
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # type: ignore
from langgraph.graph import END, StateGraph  # type: ignore[import-not-found, attr-defined]

from core.langgraph.nodes.assemble_chapter_node import assemble_chapter
from core.langgraph.nodes.commit_node import commit_to_graph
from core.langgraph.nodes.embedding_node import generate_scene_embeddings
from core.langgraph.nodes.finalize_node import finalize_chapter
from core.langgraph.nodes.graph_healing_node import heal_graph
from core.langgraph.nodes.quality_assurance_node import check_quality
from core.langgraph.nodes.relationship_normalization_node import (
    normalize_relationships,
)
from core.langgraph.nodes.revision_node import revise_chapter
from core.langgraph.nodes.summary_node import summarize_chapter
from core.langgraph.state import NarrativeState
from core.langgraph.state_helpers import (
    clear_error_state,
    clear_extraction_state,
    clear_generation_artifacts,
    clear_validation_state,
)

logger = structlog.get_logger(__name__)


def should_revise_or_continue(
    state: NarrativeState,
) -> Literal["revise", "summarize"]:
    """Route to revision or summarization for the Phase 2 graph.

    Args:
        state: Workflow state. Uses the following keys:
            - needs_revision: Whether the validation phase requested revision.
            - iteration_count: Number of revision cycles already attempted.
            - max_iterations: Upper bound on revision cycles.
            - force_continue: If true, skip revision regardless of validation.

    Returns:
        "revise" or "summarize".
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
        logger.info("should_revise_or_continue: force_continue enabled, routing to summarize")
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
        logger.info("should_revise_or_continue: revision needed, routing to revise node")
        return "revise"

    # Otherwise, proceed to summary
    logger.info("should_revise_or_continue: no revision needed, routing to summarize")
    return "summarize"


def should_handle_error(state: NarrativeState) -> Literal["error", "continue"]:
    """Route to the error handler when the workflow is in a fatal error state.

    Args:
        state: Workflow state. Uses `has_fatal_error` plus optional diagnostics
            (`last_error`, `error_node`).

    Returns:
        "error" when `has_fatal_error` is true, otherwise "continue".
    """
    if state.get("has_fatal_error", False):
        logger.error(
            "should_handle_error: fatal error detected",
            error=state.get("last_error"),
            node=state.get("error_node"),
        )
        return "error"

    return "continue"


def should_revise_or_handle_error(
    state: NarrativeState,
) -> Literal["error", "revise", "continue"]:
    """Route after validation, prioritizing fatal errors over revision.

    Args:
        state: Workflow state. Uses `has_fatal_error`/`last_error`/`error_node`,
            plus revision controls (`needs_revision`, `iteration_count`,
            `max_iterations`, `force_continue`).

    Returns:
        "error" when `has_fatal_error` is true, otherwise "revise" when revision
        is requested and allowed, otherwise "continue".
    """
    # Check for fatal errors first
    if state.get("has_fatal_error", False):
        logger.error(
            "should_revise_or_handle_error: fatal error detected",
            error=state.get("last_error"),
            node=state.get("error_node"),
        )
        return "error"

    # Check if revision needed
    needs_revision = state.get("needs_revision", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    force_continue = state.get("force_continue", False)

    if force_continue:
        logger.info("should_revise_or_handle_error: force_continue enabled, routing to continue")
        return "continue"

    if iteration_count >= max_iterations:
        logger.warning(
            "should_revise_or_handle_error: max iterations reached, routing to continue",
            iteration_count=iteration_count,
            max_iterations=max_iterations,
        )
        return "continue"

    if needs_revision:
        logger.info("should_revise_or_handle_error: revision needed, routing to revise")
        return "revise"

    return "continue"


def handle_fatal_error(state: NarrativeState) -> NarrativeState:
    """Finalize state for a clean exit after a fatal workflow error.

    Args:
        state: Workflow state with `has_fatal_error` set and optional diagnostics
            (`last_error`, `error_node`).

    Returns:
        Updated state with `current_node="error_handler"`.
    """
    logger.error(
        "handle_fatal_error: workflow terminated due to fatal error",
        error=state.get("last_error"),
        failed_node=state.get("error_node"),
        chapter=state.get("current_chapter"),
    )

    # Could write error to file for debugging in the future
    # error_file = Path(state["project_dir"]) / ".saga" / "last_error.json"

    return {
        **state,
        "current_node": "error_handler",
    }


def create_checkpointer(db_path: str = "./checkpoints/saga.db") -> AsyncSqliteSaver:
    """Create an async SQLite checkpointer for LangGraph state persistence.

    Args:
        db_path: Path to the SQLite checkpoint database.

    Returns:
        An `AsyncSqliteSaver` instance.
    """
    import os

    # Ensure checkpoints directory exists.
    #
    # Note: `os.path.dirname("saga.db") == ""`, and `os.makedirs("")` can raise.
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    logger.info(
        "create_checkpointer: creating async SQLite checkpointer",
        db_path=db_path,
    )

    # Create async checkpointer
    # AsyncSqliteSaver.from_conn_string handles the async connection
    checkpointer = AsyncSqliteSaver.from_conn_string(db_path)

    return checkpointer


def should_continue_init(state: NarrativeState) -> Literal["continue", "error"]:
    """Route within initialization based on whether the previous step failed.

    Args:
        state: Workflow state. Uses `last_error` plus the human-readable
            `initialization_step` marker.

    Returns:
        "error" when a failure indicator is present, otherwise "continue".
    """
    last_error = state.get("last_error")
    init_step = state.get("initialization_step", "")

    # Check for failure indicators
    if last_error or (init_step and "failed" in init_step.lower()):
        logger.error(
            "should_continue_init: initialization failed, halting workflow",
            error=last_error,
            step=init_step,
        )
        return "error"

    logger.info(
        "should_continue_init: initialization step succeeded, continuing",
        step=init_step,
    )
    return "continue"


def should_initialize(state: NarrativeState) -> Literal["initialize", "generate"]:
    """Route into initialization when `initialization_complete` is false.

    Args:
        state: Workflow state.

    Returns:
        "initialize" when initialization is required, otherwise "generate".
    """
    initialization_complete = state.get("initialization_complete", False)

    logger.info(
        "should_initialize: routing decision",
        initialization_complete=initialization_complete,
    )

    if not initialization_complete:
        logger.info("should_initialize: initialization needed, routing to init workflow")
        return "initialize"
    else:
        logger.info("should_initialize: initialization complete, routing to chapter generation")
        return "generate"


def advance_chapter(state: NarrativeState) -> NarrativeState:
    """Increment the chapter counter and reset per-chapter flags.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with `current_chapter` incremented and flags reset.
    """
    next_chapter = state.get("current_chapter", 1) + 1

    logger.info(
        "advance_chapter: moving to next chapter",
        previous_chapter=state.get("current_chapter"),
        next_chapter=next_chapter,
    )

    return {
        **state,
        "current_chapter": next_chapter,
        "iteration_count": 0,
        "force_continue": False,
        "chapter_plan_ref": None,
        **clear_generation_artifacts(),
        **clear_validation_state(),
        **clear_error_state(),
        **clear_extraction_state(),
    }


def should_continue_to_next_chapter(
    state: NarrativeState,
) -> Literal["continue", "end", "error"]:
    """Determine if the workflow should proceed to the next chapter.

    Args:
        state: Current workflow state.

    Returns:
        "continue" to advance, "end" if finished, or "error" if fatal error.
    """
    if state.get("has_fatal_error", False):
        return "error"

    current = state.get("current_chapter", 1)
    total = state.get("total_chapters", 1)

    logger.info(
        "should_continue_to_next_chapter: checking progress",
        current=current,
        total=total,
    )

    if current < total:
        return "continue"

    logger.info("should_continue_to_next_chapter: all chapters complete")
    return "end"


def create_full_workflow_graph(checkpointer: Any | None = None) -> StateGraph:
    """Create the end-to-end workflow graph (initialization + chapter loop).

    Args:
        checkpointer: Optional LangGraph checkpointer instance.

    Returns:
        Compiled LangGraph StateGraph ready for execution
    """
    logger.info("create_full_workflow_graph: building complete workflow")

    from core.langgraph.initialization import (
        commit_initialization_to_graph,
        generate_act_outlines,
        generate_chapter_outline,
        generate_character_sheets,
        generate_global_outline,
        persist_initialization_files,
    )

    # Create graph
    workflow = StateGraph(NarrativeState)

    # Add routing node for conditional entry
    def route_entry(state: NarrativeState) -> NarrativeState:
        """Pass-through routing node for conditional entry."""
        logger.info("route_entry: determining workflow path")
        return state

    workflow.add_node("route", route_entry)

    # Add error handler node for initialization failures
    def handle_init_error(state: NarrativeState) -> NarrativeState:
        """Handle initialization errors and terminate workflow gracefully."""
        error = state.get("last_error", "Unknown initialization error")
        step = state.get("initialization_step", "unknown")
        logger.error(
            "handle_init_error: initialization failed, terminating workflow",
            error=error,
            step=step,
        )
        return {
            **state,
            "has_fatal_error": True,
            "last_error": f"Initialization failed at step {step}: {error}",
        }

    workflow.add_node("init_error", handle_init_error)

    # Add initialization nodes (grouped into sub-workflow)
    workflow.add_node("init_character_sheets", generate_character_sheets)
    workflow.add_node("init_global_outline", generate_global_outline)
    workflow.add_node("init_act_outlines", generate_act_outlines)
    workflow.add_node("init_commit_to_graph", commit_initialization_to_graph)
    workflow.add_node("init_persist_files", persist_initialization_files)

    # Mark initialization complete
    def mark_initialization_complete(state: NarrativeState) -> NarrativeState:
        """Mark the initialization phase as complete."""
        logger.info(
            "mark_initialization_complete: initialization phase finished",
            title=state.get("title", ""),
            has_character_sheets_ref=bool(state.get("character_sheets_ref")),
            has_act_outlines_ref=bool(state.get("act_outlines_ref")),
        )
        return {
            **state,
            "initialization_complete": True,
            "initialization_step": "complete",
        }

    workflow.add_node("init_complete", mark_initialization_complete)

    # Add chapter outline generation (always run before drafting)
    workflow.add_node("chapter_outline", generate_chapter_outline)

    from core.langgraph.subgraphs.generation import create_generation_subgraph
    from core.langgraph.subgraphs.scene_extraction import create_scene_extraction_subgraph
    from core.langgraph.subgraphs.validation import create_validation_subgraph

    # Add all generation nodes (using subgraphs where applicable)
    workflow.add_node("generate", create_generation_subgraph())
    workflow.add_node("extract", create_scene_extraction_subgraph())
    workflow.add_node("gen_scene_embeddings", generate_scene_embeddings)
    workflow.add_node("assemble_chapter", assemble_chapter)
    workflow.add_node("normalize_relationships", normalize_relationships)
    workflow.add_node("commit", commit_to_graph)
    workflow.add_node("validate", create_validation_subgraph())
    workflow.add_node("revise", revise_chapter)
    workflow.add_node("summarize", summarize_chapter)
    workflow.add_node("finalize", finalize_chapter)
    workflow.add_node("heal_graph", heal_graph)
    workflow.add_node("check_quality", check_quality)
    workflow.add_node("advance_chapter", advance_chapter)

    # Phase 2 fatal error handler
    workflow.add_node("error_handler", handle_fatal_error)
    workflow.add_edge("error_handler", END)

    # Conditional entry: route → (init or chapter_outline)
    workflow.add_conditional_edges(
        "route",
        should_initialize,
        {
            "initialize": "init_character_sheets",
            "generate": "chapter_outline",
        },
    )

    # Initialization flow with error handling
    # Gate every init node on should_continue_init so failures halt deterministically.
    workflow.add_conditional_edges(
        "init_character_sheets",
        should_continue_init,
        {
            "continue": "init_global_outline",
            "error": "init_error",
        },
    )
    workflow.add_conditional_edges(
        "init_global_outline",
        should_continue_init,
        {
            "continue": "init_act_outlines",
            "error": "init_error",
        },
    )
    workflow.add_conditional_edges(
        "init_act_outlines",
        should_continue_init,
        {
            "continue": "init_commit_to_graph",
            "error": "init_error",
        },
    )
    workflow.add_conditional_edges(
        "init_commit_to_graph",
        should_continue_init,
        {
            "continue": "init_persist_files",
            "error": "init_error",
        },
    )
    workflow.add_conditional_edges(
        "init_persist_files",
        should_continue_init,
        {
            "continue": "init_complete",
            "error": "init_error",
        },
    )

    # Add error → END edge to terminate workflow gracefully
    workflow.add_edge("init_error", END)

    workflow.add_edge("init_complete", "chapter_outline")

    # Chapter outline → generate
    # Gate on should_handle_error to catch outline failures before drafting.
    workflow.add_conditional_edges(
        "chapter_outline",
        should_handle_error,
        {
            "continue": "generate",
            "error": "error_handler",
        },
    )

    # Generation flow
    # Mainline Phase 2 ordering:
    # generate → extract → gen_scene_embeddings → assemble_chapter → normalize_relationships → validate → commit → summarize …
    workflow.add_conditional_edges(
        "generate",
        should_handle_error,
        {
            "continue": "extract",
            "error": "error_handler",
        },
    )

    def should_continue_after_extract(
        state: NarrativeState,
    ) -> Literal["scene_embeddings", "error"]:
        if state.get("has_fatal_error", False):
            return "error"

        return "scene_embeddings"

    workflow.add_conditional_edges(
        "extract",
        should_continue_after_extract,
        {
            "scene_embeddings": "gen_scene_embeddings",
            "error": "error_handler",
        },
    )
    workflow.add_conditional_edges(
        "gen_scene_embeddings",
        should_handle_error,
        {
            "continue": "assemble_chapter",
            "error": "error_handler",
        },
    )
    workflow.add_conditional_edges(
        "assemble_chapter",
        should_handle_error,
        {
            "continue": "normalize_relationships",
            "error": "error_handler",
        },
    )
    workflow.add_conditional_edges(
        "normalize_relationships",
        should_handle_error,
        {
            "continue": "validate",
            "error": "error_handler",
        },
    )

    # Conditional edge: validate → (revise, commit, or error)
    # Uses should_revise_or_handle_error to prioritize fatal errors over revision.
    workflow.add_conditional_edges(
        "validate",
        should_revise_or_handle_error,
        {
            "revise": "revise",
            "continue": "commit",
            "error": "error_handler",
        },
    )

    workflow.add_conditional_edges(
        "commit",
        should_handle_error,
        {
            "continue": "summarize",
            "error": "error_handler",
        },
    )

    # Revision loop
    workflow.add_conditional_edges(
        "revise",
        should_handle_error,
        {
            "continue": "generate",
            "error": "error_handler",
        },
    )

    # Finalization, graph healing, and quality assurance
    workflow.add_conditional_edges(
        "summarize",
        should_handle_error,
        {
            "continue": "finalize",
            "error": "error_handler",
        },
    )
    workflow.add_conditional_edges(
        "finalize",
        should_handle_error,
        {
            "continue": "heal_graph",
            "error": "error_handler",
        },
    )
    workflow.add_conditional_edges(
        "heal_graph",
        should_handle_error,
        {
            "continue": "check_quality",
            "error": "error_handler",
        },
    )
    workflow.add_conditional_edges(
        "check_quality",
        should_continue_to_next_chapter,
        {
            "continue": "advance_chapter",
            "end": END,
            "error": "error_handler",
        },
    )

    workflow.add_edge("advance_chapter", "chapter_outline")

    # Set entry point to routing node
    workflow.set_entry_point("route")

    logger.info(
        "create_full_workflow_graph: graph built successfully",
        total_nodes=24,  # route + init_error + 6 init + 15 generation/QA nodes + error_handler
        entry_point="route",
    )

    # Compile graph
    if checkpointer:
        logger.info("create_full_workflow_graph: compiling with checkpointing enabled")
        compiled_graph = workflow.compile(checkpointer=checkpointer)
    else:
        logger.info("create_full_workflow_graph: compiling without checkpointing")
        compiled_graph = workflow.compile()

    return compiled_graph


__all__ = [
    "create_checkpointer",
    "create_full_workflow_graph",
    "should_revise_or_continue",
    "should_initialize",
    "should_handle_error",
    "should_revise_or_handle_error",
    "handle_fatal_error",
    "advance_chapter",
    "should_continue_to_next_chapter",
]
