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
from core.langgraph.subgraphs.generation import create_generation_subgraph
from core.langgraph.visualization import (
    print_workflow_summary,
    visualize_workflow,
)
from core.langgraph.workflow import (
    create_checkpointer,
    create_full_workflow_graph,
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
    "create_generation_subgraph",
    "revise_chapter",
    "summarize_chapter",
    "finalize_chapter",
    # Context
    "build_context_from_graph",
    "get_key_events",
    # Workflow
    "create_checkpointer",
    "create_full_workflow_graph",
    # Visualization
    "visualize_workflow",
    "print_workflow_summary",
]

__version__ = "0.1.0"
