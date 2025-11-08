"""
Initialization Workflow for LangGraph-based narrative generation.

This module defines the initialization phase workflow that runs before
chapter generation, establishing character sheets, outlines, and other
foundational elements.

Initialization Workflow Structure:
    START
      ↓
    [generate_character_sheets]
      ↓
    [generate_global_outline]
      ↓
    [generate_act_outlines]
      ↓
    END (initialization complete)

The chapter outline generation is done on-demand during the generation loop,
not as part of the initialization workflow.
"""

import structlog

from core.langgraph.initialization.act_outlines_node import generate_act_outlines
from core.langgraph.initialization.character_sheets_node import (
    generate_character_sheets,
)
from core.langgraph.initialization.commit_init_node import (
    commit_initialization_to_graph,
)
from core.langgraph.initialization.global_outline_node import generate_global_outline
from core.langgraph.initialization.persist_files_node import (
    persist_initialization_files,
)
from core.langgraph.state import NarrativeState
from langgraph.graph import END, StateGraph

logger = structlog.get_logger(__name__)


def create_initialization_graph(checkpointer=None) -> StateGraph:
    """
    Create the initialization workflow graph.

    This graph runs the initialization phase that establishes:
    1. Character sheets for main characters
    2. Global story outline
    3. Act-level outlines
    4. Commits all initialization data to Neo4j knowledge graph
    5. Persists human-readable files to disk (YAML/Markdown)

    After initialization, the state is ready for the main generation loop.

    Workflow:
        START → character_sheets → global_outline → act_outlines →
                commit_to_graph → persist_files → END

    Args:
        checkpointer: Optional checkpoint saver (SqliteSaver, PostgresSaver, etc.)
                     If None, no checkpointing is enabled.

    Returns:
        Compiled LangGraph StateGraph ready for execution
    """
    logger.info("create_initialization_graph: building initialization workflow")

    # Create graph
    workflow = StateGraph(NarrativeState)

    # Add initialization nodes
    workflow.add_node("character_sheets", generate_character_sheets)
    workflow.add_node("global_outline", generate_global_outline)
    workflow.add_node("act_outlines", generate_act_outlines)
    workflow.add_node("commit_to_graph", commit_initialization_to_graph)
    workflow.add_node("persist_files", persist_initialization_files)

    # Add finalization node to mark initialization complete
    def mark_initialization_complete(state: NarrativeState) -> NarrativeState:
        """Mark the initialization phase as complete."""
        logger.info(
            "mark_initialization_complete: initialization phase finished",
            title=state["title"],
            characters=len(state.get("character_sheets", {})),
            acts=len(state.get("act_outlines", {})),
        )
        return {
            **state,
            "initialization_complete": True,
            "initialization_step": "complete",
            "current_node": "init_complete",
        }

    workflow.add_node("complete", mark_initialization_complete)

    # Define linear flow
    workflow.add_edge("character_sheets", "global_outline")
    workflow.add_edge("global_outline", "act_outlines")
    workflow.add_edge("act_outlines", "commit_to_graph")
    workflow.add_edge("commit_to_graph", "persist_files")
    workflow.add_edge("persist_files", "complete")
    workflow.add_edge("complete", END)

    # Set entry point
    workflow.set_entry_point("character_sheets")

    logger.info(
        "create_initialization_graph: graph built successfully",
        nodes=[
            "character_sheets",
            "global_outline",
            "act_outlines",
            "commit_to_graph",
            "persist_files",
            "complete",
        ],
        entry_point="character_sheets",
    )

    # Compile graph
    if checkpointer:
        logger.info("create_initialization_graph: compiling with checkpointing enabled")
        compiled_graph = workflow.compile(checkpointer=checkpointer)
    else:
        logger.info("create_initialization_graph: compiling without checkpointing")
        compiled_graph = workflow.compile()

    return compiled_graph


__all__ = ["create_initialization_graph"]
