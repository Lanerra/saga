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
    graph: Graph definition and workflow orchestration

Usage:
    from core.langgraph import (
        NarrativeState,
        create_initial_state,
        extract_entities,
        commit_to_graph,
        build_context_from_graph
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

    # Build context from knowledge graph
    context = await build_context_from_graph(
        current_chapter=5,
        lookback_chapters=3
    )

    # Use nodes in sequence
    state = await extract_entities(state)
    state = await commit_to_graph(state)
"""

from core.langgraph.graph_context import build_context_from_graph, get_key_events
from core.langgraph.nodes import commit_to_graph, extract_entities
from core.langgraph.state import (
    Contradiction,
    ExtractedEntity,
    ExtractedRelationship,
    NarrativeState,
    State,
    create_initial_state,
)

__all__ = [
    "NarrativeState",
    "State",
    "ExtractedEntity",
    "ExtractedRelationship",
    "Contradiction",
    "create_initial_state",
    "extract_entities",
    "commit_to_graph",
    "build_context_from_graph",
    "get_key_events",
]

__version__ = "0.1.0"
