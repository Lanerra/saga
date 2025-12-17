# core/langgraph/workflow.py
"""
LangGraph workflow for SAGA narrative generation.

This module wires together all nodes into complete workflow graphs
with conditional edges, revision loops, and checkpointing.
"""

from typing import Any, Literal

import structlog
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # type: ignore
from langgraph.graph import END, StateGraph  # type: ignore

from core.langgraph.content_manager import ContentManager, get_chapter_outlines
from core.langgraph.nodes.commit_node import commit_to_graph
from core.langgraph.nodes.embedding_node import generate_embedding
from core.langgraph.nodes.finalize_node import finalize_chapter
from core.langgraph.nodes.graph_healing_node import heal_graph
from core.langgraph.nodes.quality_assurance_node import check_quality
from core.langgraph.nodes.relationship_normalization_node import (
    normalize_relationships,
)
from core.langgraph.nodes.revision_node import revise_chapter
from core.langgraph.nodes.summary_node import summarize_chapter
from core.langgraph.state import NarrativeState

logger = structlog.get_logger(__name__)


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
    """
    Check if a fatal error occurred and needs handling.

    Routes to:
    - "error": If has_fatal_error=True (stop workflow gracefully)
    - "continue": If no fatal error (proceed normally)

    Args:
        state: Current narrative state

    Returns:
        "error" or "continue"
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
    """
    Combined check for fatal errors and revision needs.

    This function is used after validation to determine the next step.
    It prioritizes fatal errors over revision needs.

    Routes to:
    - "error": If has_fatal_error=True (stop workflow gracefully)
    - "revise": If needs_revision=True and no fatal error
    - "continue": If no errors and no revision needed (proceed to summarize)

    Priority:
    1. Fatal error → "error"
    2. Needs revision → "revise"
    3. Otherwise → "continue"

    Args:
        state: Current narrative state

    Returns:
        "error", "revise", or "continue"
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
    """
    Handle fatal errors gracefully.

    Logs detailed error information and prepares state for clean exit.

    Args:
        state: Current narrative state with error information

    Returns:
        Updated state with error_handler as current_node
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


def create_phase2_graph(checkpointer: Any | None = None) -> StateGraph:
    """
    Create Phase 2 LangGraph workflow (COMPLETE).

    This is the full narrative generation workflow including:
    - Generation: Create chapter from outline and context (SUBGRAPH)
    - Extraction: Extract entities and relationships (SUBGRAPH)
    - Normalization: Normalize relationship types against vocabulary
    - Commitment: Deduplicate and commit to Neo4j
    - Validation: Check for contradictions and quality issues (SUBGRAPH)
    - Revision: Fix issues (conditional, with iteration limit)
    - Summarization: Create concise chapter summary
    - Finalization: Persist to filesystem and Neo4j

    Workflow:
        generate_subgraph → extract_subgraph → normalize_relationships → commit → validate_subgraph
        → {revise? revise → extract : summarize} → finalize → heal_graph → check_quality → END

    Migration Reference: docs/phase2_migration_plan.md - Step 2.5

    Args:
        checkpointer: Optional checkpoint saver (SqliteSaver, PostgresSaver, etc.)
                     If None, no checkpointing is enabled.

    Returns:
        Compiled LangGraph StateGraph ready for execution
    """
    logger.info("create_phase2_graph: building complete workflow graph")

    from core.langgraph.subgraphs.extraction import create_extraction_subgraph
    from core.langgraph.subgraphs.generation import create_generation_subgraph
    from core.langgraph.subgraphs.validation import create_validation_subgraph

    # Create graph
    workflow = StateGraph(NarrativeState)

    # Add nodes (using subgraphs where applicable)
    workflow.add_node("generate", create_generation_subgraph())
    workflow.add_node("extract", create_extraction_subgraph())
    workflow.add_node("normalize_relationships", normalize_relationships)
    workflow.add_node("commit", commit_to_graph)
    workflow.add_node("validate", create_validation_subgraph())
    workflow.add_node("revise", revise_chapter)
    workflow.add_node("summarize", summarize_chapter)
    workflow.add_node("finalize", finalize_chapter)
    workflow.add_node("heal_graph", heal_graph)
    workflow.add_node("check_quality", check_quality)
    workflow.add_node("error_handler", handle_fatal_error)

    # Add error checking after generate
    workflow.add_conditional_edges(
        "generate",
        should_handle_error,
        {
            "error": "error_handler",
            "continue": "extract",
        },
    )

    # Add error checking after extract
    workflow.add_conditional_edges(
        "extract",
        should_handle_error,
        {
            "error": "error_handler",
            "continue": "normalize_relationships",
        },
    )

    # Add error checking after normalize_relationships
    workflow.add_conditional_edges(
        "normalize_relationships",
        should_handle_error,
        {
            "error": "error_handler",
            "continue": "commit",
        },
    )

    # Add error checking after commit
    workflow.add_conditional_edges(
        "commit",
        should_handle_error,
        {
            "error": "error_handler",
            "continue": "validate",
        },
    )

    # Conditional edge: validate → (error, revise, or summarize)
    workflow.add_conditional_edges(
        "validate",
        should_revise_or_handle_error,
        {
            "error": "error_handler",
            "revise": "revise",
            "continue": "summarize",
        },
    )

    # After revision, check for errors then loop back to extract
    workflow.add_conditional_edges(
        "revise",
        should_handle_error,
        {
            "error": "error_handler",
            "continue": "extract",
        },
    )

    # Summarize failures are non-critical (continue anyway)
    workflow.add_edge("summarize", "finalize")

    # Check for errors after finalize
    workflow.add_conditional_edges(
        "finalize",
        should_handle_error,
        {
            "error": "error_handler",
            "continue": "heal_graph",
        },
    )

    # Graph healing runs after successful finalization
    workflow.add_edge("heal_graph", "check_quality")
    workflow.add_edge("check_quality", END)

    # Error handler terminates workflow
    workflow.add_edge("error_handler", END)

    # Set entry point
    workflow.set_entry_point("generate")

    logger.info(
        "create_phase2_graph: graph built successfully",
        nodes=[
            "generate (subgraph)",
            "extract (subgraph)",
            "normalize_relationships",
            "commit",
            "validate (subgraph)",
            "revise",
            "summarize",
            "finalize",
            "heal_graph",
            "check_quality",
            "error_handler",
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


def should_generate_chapter_outline(
    state: NarrativeState,
) -> Literal["chapter_outline", "generate"]:
    """
    Conditional edge function: Determine if chapter outline generation is needed.

    Routes to:
    - "chapter_outline": If chapter outline doesn't exist for current chapter
    - "generate": If chapter outline exists

    Args:
        state: Current narrative state

    Returns:
        Next node name ("chapter_outline" or "generate")
    """
    current_chapter = state.get("current_chapter", 1)
    content_manager = ContentManager(state.get("project_dir", ""))
    chapter_outlines = get_chapter_outlines(state, content_manager)

    if current_chapter not in chapter_outlines:
        logger.info(
            "should_generate_chapter_outline: outline needed",
            chapter=current_chapter,
        )
        return "chapter_outline"
    else:
        logger.info(
            "should_generate_chapter_outline: outline exists",
            chapter=current_chapter,
        )
        return "generate"


def should_continue_init(state: NarrativeState) -> Literal["continue", "error"]:
    """
    Conditional routing: Check if initialization step succeeded.

    Routes to:
    - "continue": If initialization step succeeded
    - "error": If initialization step failed

    Args:
        state: Current narrative state

    Returns:
        Next step ("continue" or "error")
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
    """
    Conditional entry routing: Determine if initialization is needed.

    Routes to:
    - "initialize": If initialization_complete=False (run init workflow)
    - "generate": If initialization_complete=True (skip to chapter generation)

    Args:
        state: Current narrative state

    Returns:
        Next node name ("initialize" or "generate")
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


def create_full_workflow_graph(checkpointer: Any | None = None) -> StateGraph:
    """
    Create complete workflow with initialization and generation phases.

    This is the full end-to-end workflow that:
    1. Runs initialization (character sheets, outlines) if needed
    2. For each chapter:
       a. Generates chapter outline (on-demand)
       b. Generates chapter text
       c. Extracts entities
       d. Normalizes relationship types
       e. Commits to graph
       f. Validates
       g. Optionally revises
       h. Summarizes and finalizes

    Workflow:
        START → {init?}
                ├─ initialize → [character_sheets → global_outline → act_outlines] → chapter_loop
                └─ chapter_loop → {outline?}
                                  ├─ chapter_outline → generate
                                  └─ generate → extract → normalize_relationships → commit → validate → {revise?}
                                                                                                           ├─ revise → extract
                                                                                                           └─ summarize → finalize → END

    Args:
        checkpointer: Optional checkpoint saver (SqliteSaver, PostgresSaver, etc.)

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
            "workflow_failed": True,
            "failure_reason": f"Initialization failed at step {step}: {error}",
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

    # Add chapter outline generation (on-demand)
    workflow.add_node("chapter_outline", generate_chapter_outline)

    from core.langgraph.subgraphs.extraction import create_extraction_subgraph
    from core.langgraph.subgraphs.generation import create_generation_subgraph
    from core.langgraph.subgraphs.validation import create_validation_subgraph

    # Add all generation nodes (using subgraphs where applicable)
    workflow.add_node("generate", create_generation_subgraph())
    workflow.add_node("gen_embedding", generate_embedding)
    workflow.add_node("extract", create_extraction_subgraph())
    workflow.add_node("normalize_relationships", normalize_relationships)
    workflow.add_node("commit", commit_to_graph)
    workflow.add_node("validate", create_validation_subgraph())
    workflow.add_node("revise", revise_chapter)
    workflow.add_node("summarize", summarize_chapter)
    workflow.add_node("finalize", finalize_chapter)
    workflow.add_node("heal_graph", heal_graph)
    workflow.add_node("check_quality", check_quality)

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
    workflow.add_edge("chapter_outline", "generate")

    # Generation flow
    # Run embedding generation and extraction after text generation
    # Ideally these could be parallel, but for now we sequence them
    # generate -> gen_embedding -> extract -> normalize_relationships -> commit
    workflow.add_edge("generate", "gen_embedding")
    workflow.add_edge("gen_embedding", "extract")
    workflow.add_edge("extract", "normalize_relationships")
    workflow.add_edge("normalize_relationships", "commit")
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

    # Revision loop
    workflow.add_edge("revise", "extract")

    # Finalization, graph healing, and quality assurance
    workflow.add_edge("summarize", "finalize")
    workflow.add_edge("finalize", "heal_graph")
    workflow.add_edge("heal_graph", "check_quality")
    workflow.add_edge("check_quality", END)

    # Set entry point to routing node
    workflow.set_entry_point("route")

    logger.info(
        "create_full_workflow_graph: graph built successfully",
        total_nodes=16,  # route + init_error + 6 init + 8 generation (added normalize_relationships)
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
    "create_phase2_graph",
    "create_checkpointer",
    "create_full_workflow_graph",
    "should_revise_or_continue",
    "should_initialize",
    "should_generate_chapter_outline",
    "should_handle_error",
    "should_revise_or_handle_error",
    "handle_fatal_error",
]
