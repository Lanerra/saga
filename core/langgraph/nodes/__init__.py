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

from core.langgraph.nodes.assemble_chapter_node import assemble_chapter
from core.langgraph.nodes.commit_node import commit_to_graph
from core.langgraph.nodes.context_retrieval_node import retrieve_context
from core.langgraph.nodes.embedding_node import generate_scene_embeddings
from core.langgraph.nodes.extraction_nodes import consolidate_extraction
from core.langgraph.nodes.finalize_node import finalize_chapter
from core.langgraph.nodes.graph_healing_node import heal_graph
from core.langgraph.nodes.narrative_enrichment_node import enrich_narrative
from core.langgraph.nodes.quality_assurance_node import check_quality
from core.langgraph.nodes.relationship_normalization_node import (
    normalize_relationships,
)
from core.langgraph.nodes.revision_node import revise_chapter
from core.langgraph.nodes.scene_extraction import extract_from_scenes
from core.langgraph.nodes.scene_generation_node import draft_scene
from core.langgraph.nodes.scene_planning_node import plan_scenes
from core.langgraph.nodes.summary_node import summarize_chapter
from core.langgraph.nodes.validation_node import validate_consistency

__all__ = [
    "assemble_chapter",
    "check_quality",
    "commit_to_graph",
    "consolidate_extraction",
    "draft_scene",
    "enrich_narrative",
    "extract_from_scenes",
    "finalize_chapter",
    "generate_scene_embeddings",
    "heal_graph",
    "normalize_relationships",
    "plan_scenes",
    "retrieve_context",
    "revise_chapter",
    "summarize_chapter",
    "validate_consistency",
]
