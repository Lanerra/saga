"""
LangGraph nodes for SAGA narrative generation workflow.

This module contains individual processing nodes that make up the
LangGraph-based narrative generation workflow.

Implemented Nodes:
    - extraction_node: Entity extraction from generated text ✓
    - commit_node: Deduplication and Neo4j commitment ✓
    - validation_node: Consistency and quality validation ✓
    - generation_node: Chapter text generation ✓ (Phase 2)

Planned Nodes (from migration plan):
    - revision_node: Content revision based on feedback
    - summary_node: Chapter summarization
    - finalize_node: Chapter persistence and cleanup

Migration Reference: docs/langgraph_migration_plan.md - Phase 1

Each node follows the LangGraph signature:
    async def node_function(state: NarrativeState) -> NarrativeState:
        # Process state
        return updated_state
"""

from core.langgraph.nodes.commit_node import commit_to_graph
from core.langgraph.nodes.extraction_node import extract_entities
from core.langgraph.nodes.generation_node import generate_chapter
from core.langgraph.nodes.validation_node import validate_consistency

__all__ = [
    "extract_entities",
    "commit_to_graph",
    "validate_consistency",
    "generate_chapter",
]

__version__ = "0.1.0"
