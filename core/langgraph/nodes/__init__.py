# core/langgraph/nodes/__init__.py
"""
Provide LangGraph node entrypoints for the SAGA workflow.

Migration Reference: docs/langgraph_migration_plan.md

This package re-exports node callables used by graph builders. Node functions
generally accept and return the workflow state (a `NarrativeState` mapping).

Node Return Convention:
Nodes return only the fields they modify and `current_node`. LangGraph merges
partial updates into the existing state to preserve immutability.

Example:
    return {
        "extracted_entities_ref": entities_ref,
        "current_node": "extract",
    }
"""

from core.langgraph.nodes.commit_node import commit_to_graph
from core.langgraph.nodes.finalize_node import finalize_chapter
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
    "revise_chapter",
    "summarize_chapter",
    "finalize_chapter",
    "heal_graph",
    "normalize_relationships",
]

__version__ = "0.1.0"
