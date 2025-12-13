# core/langgraph/nodes/__init__.py
"""
LangGraph nodes for SAGA narrative generation workflow.

This module contains individual processing nodes that make up the
LangGraph-based narrative generation workflow.

Implemented Nodes:
    - extraction_node: Entity extraction from generated text ✓
    - commit_node: Deduplication and Neo4j commitment ✓
    - validation_node: Consistency and quality validation ✓
    - generation_node: Chapter text generation ✓ (Phase 2)
    - revision_node: Content revision based on feedback ✓ (Phase 2)
    - summary_node: Chapter summarization ✓ (Phase 2)
    - finalize_node: Chapter persistence and cleanup ✓ (Phase 2)
    - graph_healing_node: Provisional node enrichment and merge ✓

All Phase 2 nodes complete! Ready for workflow integration.

Migration Reference: docs/langgraph_migration_plan.md - Phase 1

Each node follows the LangGraph signature:
    async def node_function(state: NarrativeState) -> NarrativeState:
        # Process state
        return updated_state
"""

from core.langgraph.nodes.commit_node import commit_to_graph
from core.langgraph.nodes.finalize_node import finalize_chapter
from core.langgraph.nodes.generation_node import generate_chapter_single_shot
from core.langgraph.nodes.graph_healing_node import heal_graph
from core.langgraph.nodes.relationship_normalization_node import (
    normalize_relationships,
)
from core.langgraph.nodes.revision_node import revise_chapter
from core.langgraph.nodes.summary_node import summarize_chapter
from core.langgraph.nodes.validation_node import validate_consistency

__all__ = [
    "commit_to_graph",
    "validate_consistency",
    "generate_chapter_single_shot",
    "revise_chapter",
    "summarize_chapter",
    "finalize_chapter",
    "heal_graph",
    "normalize_relationships",
]

__version__ = "0.1.0"
