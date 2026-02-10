# core/langgraph/state_helpers.py
"""Helper functions for managing workflow state field clearing.

This module provides centralized functions for clearing different categories of
state fields, ensuring consistency across the workflow and avoiding duplication.
"""

from typing import Any


def _safe_list_update(existing: list[Any], new_items: list[Any]) -> list[Any]:
    """Create a new list by appending new items to existing list.

    Returns a new list instead of mutating the existing one, ensuring LangGraph
    state immutability requirements are met.

    Args:
        existing: Original list (will not be mutated).
        new_items: Items to append.

    Returns:
        New list containing all items from existing plus new_items.
    """
    return list(existing) + new_items


def clear_generation_artifacts() -> dict:
    """Clear all generation artifacts (drafts, embeddings, scenes).

    Returns:
        Dictionary with generation fields cleared for merging into state.
    """
    return {
        "draft_ref": None,
        "embedding_ref": None,
        "scene_embeddings_ref": None,
        "generated_embedding": None,
        "scene_drafts_ref": None,
        "current_scene_index": 0,
    }


def clear_validation_state() -> dict:
    """Clear validation and quality state.

    Returns:
        Dictionary with validation fields cleared for merging into state.
    """
    return {
        "contradictions": [],
        "needs_revision": False,
        "revision_guidance_ref": None,
    }


def clear_error_state() -> dict:
    """Clear error tracking state.

    Returns:
        Dictionary with error fields cleared for merging into state.
    """
    return {
        "last_error": None,
        "has_fatal_error": False,
        "error_node": None,
    }


def clear_extraction_state() -> dict:
    """Clear entity extraction state.

    Returns:
        Dictionary with extraction fields cleared for merging into state.
    """
    return {
        "extracted_entities_ref": None,
        "extracted_relationships_ref": None,
    }
