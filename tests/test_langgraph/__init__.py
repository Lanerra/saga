# tests/test_langgraph/__init__.py
"""
LangGraph Test Suite.

This package contains comprehensive tests for the LangGraph-based workflow components.

Test Modules:
    - test_state: Tests for state schema
    - test_commit_node: Tests for commit node
    - test_graph_context: Tests for graph context
    - test_validation_node: Tests for validation node
    - test_phase2_workflow: Tests for the chapter-generation workflow path
    - test_workflow: Tests for workflow orchestration

Run tests with:
    pytest tests/test_langgraph/ -v
"""

from core.langgraph.state import ContentRef, ExtractedEntity, ExtractedRelationship, NarrativeState


class InlineExtractionState(NarrativeState, total=False):
    """Historical inline payloads used to exercise reference precedence and rejection."""

    extracted_entities: dict[str, list[ExtractedEntity | dict[str, object]]]
    extracted_relationships: list[ExtractedRelationship | dict[str, object]]
    relationships_rejected_this_chapter: int
    relationships_property_converted_this_chapter: int
    relationship_rejection_rate: float


class LegacyEmbeddingRef(ContentRef, total=False):
    """Legacy embedding metadata supplied to patched storage readers."""

    format: str
