# core/langgraph/__init__.py
"""
Integrate SAGA with LangGraph workflows.

Migration Reference: docs/langgraph_migration_plan.md

This package re-exports the public workflow API (state schema, node entrypoints,
graph builders, and visualization utilities) for use by orchestrators and tests.
"""

from core.langgraph.graph_context import build_context_from_graph, get_key_events
from core.langgraph.nodes import (
    commit_to_graph,
    finalize_chapter,
    generate_chapter_single_shot,
    revise_chapter,
    summarize_chapter,
    validate_consistency,
)
from core.langgraph.state import (
    Contradiction,
    ExtractedEntity,
    ExtractedRelationship,
    NarrativeState,
    State,
    create_initial_state,
)
from core.langgraph.subgraphs.generation import generate_chapter
from core.langgraph.visualization import (
    print_workflow_summary,
    visualize_workflow,
)
from core.langgraph.workflow import (
    create_checkpointer,
    create_full_workflow_graph,
    should_revise_or_continue,
)

__all__ = [
    # State
    "NarrativeState",
    "State",
    "ExtractedEntity",
    "ExtractedRelationship",
    "Contradiction",
    "create_initial_state",
    # Nodes
    "commit_to_graph",
    "validate_consistency",
    # Canonical generation API (scene-based subgraph)
    "generate_chapter",
    # Backcompat: explicit single-shot node (async)
    "generate_chapter_single_shot",
    "revise_chapter",
    "summarize_chapter",
    "finalize_chapter",
    # Context
    "build_context_from_graph",
    "get_key_events",
    # Workflow
    "create_checkpointer",
    "create_full_workflow_graph",
    "should_revise_or_continue",
    # Visualization
    "visualize_workflow",
    "print_workflow_summary",
]

__version__ = "0.1.0"
