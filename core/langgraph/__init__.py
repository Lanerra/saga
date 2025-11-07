"""
LangGraph integration module for SAGA.

This module provides the LangGraph-based workflow implementation for SAGA,
enabling state-based narrative generation with automatic persistence,
parallel execution, and structured quality control.

Migration Reference: docs/langgraph_migration_plan.md

Components:
    state: Core state schema for LangGraph workflow
    nodes: Individual processing nodes (extraction, validation, generation, etc.)
    graph_context: Neo4j context construction (wraps existing data_access queries)
    workflow: Graph definition and workflow orchestration

Usage:
    from core.langgraph import (
        NarrativeState,
        create_initial_state,
        create_phase1_graph,
        create_checkpointer
    )

    # Create initial state
    state = create_initial_state(
        project_id="my-novel",
        title="My Novel",
        genre="Fantasy",
        theme="Adventure",
        setting="Medieval world",
        target_word_count=80000,
        total_chapters=20,
        project_dir="./output/my-novel",
        protagonist_name="Hero"
    )

    # Create and run workflow
    checkpointer = create_checkpointer("./checkpoints/my-novel.db")
    graph = create_phase1_graph(checkpointer=checkpointer)

    # Execute workflow
    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "my-novel-ch1"}})
"""

from core.langgraph.graph_context import build_context_from_graph, get_key_events
from core.langgraph.nodes import (
    commit_to_graph,
    extract_entities,
    generate_chapter,
    revise_chapter,
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
from core.langgraph.workflow import (
    create_checkpointer,
    create_phase1_graph,
    should_revise,
)

__all__ = [
    # State
    "NarrativeState",
    "State",
    "ExtractedEntity",
    "ExtractedRelationship",
    "Contradiction",
    "create_initial_state",
    # Nodes (Phase 1)
    "extract_entities",
    "commit_to_graph",
    "validate_consistency",
    # Nodes (Phase 2)
    "generate_chapter",
    "revise_chapter",
    # Context
    "build_context_from_graph",
    "get_key_events",
    # Workflow
    "create_phase1_graph",
    "create_checkpointer",
    "should_revise",
]

__version__ = "0.1.0"
