"""
LangGraph nodes for SAGA narrative generation workflow.

This module contains individual processing nodes that make up the
LangGraph-based narrative generation workflow.

Implemented Nodes:
    - extraction_node: Entity extraction from generated text ✓
    - commit_node: Deduplication and Neo4j commitment ✓

Planned Nodes (from migration plan):
    - validation_node: Consistency and quality validation
    - generation_node: Chapter text generation
    - revision_node: Content revision based on feedback
    - context_node: Context construction from knowledge graph

Migration Reference: docs/langgraph_migration_plan.md - Phase 1

Each node follows the LangGraph signature:
    async def node_function(state: NarrativeState) -> NarrativeState:
        # Process state
        return updated_state
"""

from core.langgraph.nodes.commit_node import commit_to_graph
from core.langgraph.nodes.extraction_node import extract_entities

__all__ = ["extract_entities", "commit_to_graph"]

__version__ = "0.1.0"
