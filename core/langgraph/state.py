# core/langgraph/state.py
"""
LangGraph State Schema for SAGA Narrative Generation.

This module defines the state schema for the LangGraph-based workflow,
designed to minimize disruption to existing SAGA code while enabling
the migration to LangGraph architecture.

Migration Reference: docs/langgraph_migration_plan.md - Step 1.1.1
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

# Import existing SAGA models for compatibility
from models.kg_models import CharacterProfile, WorldItem

# Import settings for model configuration
from config.settings import settings


class ExtractedEntity(BaseModel):
    """
    Entity extracted from generated text (before Neo4j commit).

    This model represents entities identified during text generation
    that will be validated, deduplicated, and committed to the knowledge graph.

    The type field now accepts any valid node type from the ontology
    (e.g., "Character", "DevelopmentEvent", "PlotPoint", "Artifact", etc.)
    instead of being limited to just "character", "location", "event", "object".
    """

    name: str
    type: str  # Changed from Literal to str to accept all node types from ontology
    description: str
    first_appearance_chapter: int
    attributes: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        frozen = False
        validate_assignment = True


class ExtractedRelationship(BaseModel):
    """
    Relationship extracted from generated text.

    Represents connections between entities discovered during generation,
    pending validation and commitment to the knowledge graph.
    """

    source_name: str
    target_name: str
    relationship_type: str
    description: str
    chapter: int
    confidence: float = 0.8

    class Config:
        """Pydantic configuration."""

        frozen = False
        validate_assignment = True


class Contradiction(BaseModel):
    """
    Detected inconsistency in narrative or knowledge graph.

    Used by validation nodes to flag potential issues that may
    require revision or human review.
    """

    type: str
    description: str
    conflicting_chapters: list[int]
    severity: Literal["minor", "major", "critical"]
    suggested_fix: str | None = None

    class Config:
        """Pydantic configuration."""

        frozen = False
        validate_assignment = True


class NarrativeState(TypedDict, total=False):
    """
    LangGraph state for SAGA narrative generation workflow.

    All fields are automatically persisted by LangGraph's checkpointer.
    This schema is designed to align with existing SAGA data structures
    to minimize migration disruption.

    Migration Strategy:
    - Preserves existing field names where possible (e.g., plot_outline)
    - Maintains compatibility with CharacterProfile and WorldItem models
    - Adds new fields for entity extraction and validation workflows
    - Uses Optional types for flexibility during gradual migration
    """

    # =========================================================================
    # Project Metadata
    # =========================================================================
    project_id: str
    title: str
    genre: str
    theme: str
    setting: str
    target_word_count: int

    # =========================================================================
    # Neo4j Connection (reconstructed on load, not persisted)
    # =========================================================================
    neo4j_conn: Any | None  # Will be reconstructed from project_id

    # =========================================================================
    # Current Position in Story
    # =========================================================================
    current_chapter: int
    total_chapters: int
    current_act: int

    # =========================================================================
    # Plot Outline - DEPRECATED
    # =========================================================================
    # DEPRECATED: Use chapter_outlines instead (see Initialization Phase State below)
    # This field is kept for backward compatibility and will be removed in v3.0
    plot_outline: dict[int, dict[str, Any]]

    # =========================================================================
    # Active Context (for prompt construction)
    # =========================================================================
    active_characters: list[CharacterProfile]  # Reuses existing model
    current_location: dict[str, Any] | None
    previous_chapter_summaries: list[str]
    key_events: list[dict[str, Any]]

    # =========================================================================
    # Generated Content (current chapter)
    # =========================================================================
    draft_text: str | None
    draft_word_count: int
    generated_embedding: list[float] | None

    # =========================================================================
    # Entity Extraction Results (NEW: centralized extraction state)
    # =========================================================================
    extracted_entities: dict[str, list[ExtractedEntity]]
    extracted_relationships: list[ExtractedRelationship]
    
    # Temporary keys for parallel extraction
    character_updates: list[ExtractedEntity]
    location_updates: list[ExtractedEntity]
    event_updates: list[ExtractedEntity]
    relationship_updates: list[ExtractedRelationship]

    # =========================================================================
    # Validation and Quality Control (NEW: formalized validation state)
    # =========================================================================
    contradictions: list[Contradiction]
    needs_revision: bool
    revision_feedback: str | None
    is_from_flawed_draft: (
        bool  # True if deduplication removed text or other quality issues detected
    )

    # =========================================================================
    # Quality Metrics (NEW: LLM-evaluated quality scores)
    # =========================================================================
    coherence_score: float | None  # 0.0-1.0 score for narrative coherence
    prose_quality_score: float | None  # 0.0-1.0 score for prose quality
    plot_advancement_score: float | None  # 0.0-1.0 score for plot advancement
    pacing_score: float | None  # 0.0-1.0 score for narrative pacing
    tone_consistency_score: float | None  # 0.0-1.0 score for tone consistency
    quality_feedback: str | None  # Detailed feedback from quality evaluation

    # =========================================================================
    # Model Configuration
    # =========================================================================
    generation_model: str
    extraction_model: str
    revision_model: str
    # New tiered model configuration
    large_model: str
    medium_model: str
    small_model: str
    narrative_model: str

    # =========================================================================
    # Workflow Control
    # =========================================================================
    current_node: str  # Tracks which node last updated state
    iteration_count: int
    max_iterations: int
    force_continue: bool  # Override validation failures

    # =========================================================================
    # Error Handling
    # =========================================================================
    last_error: str | None
    has_fatal_error: bool  # True if workflow should stop due to unrecoverable error
    error_node: str | None  # Which node encountered the fatal error
    retry_count: int

    # =========================================================================
    # Filesystem Paths
    # =========================================================================
    project_dir: str
    chapters_dir: str
    summaries_dir: str

    # =========================================================================
    # Context Management (maintains compatibility with existing context system)
    # =========================================================================
    context_epoch: int  # Compatible with NarrativeState.context_epoch
    hybrid_context: str | None  # Compatible with ContextSnapshot
    kg_facts_block: str | None  # Compatible with ContextSnapshot

    # =========================================================================
    # Chapter Planning (compatible with existing SceneDetail structure)
    # =========================================================================
    chapter_plan: list[dict[str, Any]] | None  # List of SceneDetail dicts
    plot_point_focus: str | None
    current_scene_index: int  # Index of the scene currently being processed
    scene_drafts: list[str]  # List of generated text for each scene

    # =========================================================================
    # Revision State (compatible with existing evaluation workflow)
    # =========================================================================
    evaluation_result: dict[str, Any] | None  # EvaluationResult structure
    patch_instructions: list[dict[str, Any]] | None  # PatchInstruction list

    # =========================================================================
    # World Building Context
    # =========================================================================
    world_items: list[WorldItem]  # Reuses existing model
    current_world_rules: list[str]

    # =========================================================================
    # Protagonist and Key Characters
    # =========================================================================
    protagonist_name: str
    protagonist_profile: CharacterProfile | None

    # =========================================================================
    # Initialization Phase State (for initialization workflow)
    # =========================================================================
    # Character sheets generated during initialization
    character_sheets: dict[str, dict[str, Any]]  # character_name -> character_sheet

    # Global outline generated during initialization
    global_outline: dict[str, Any] | None

    # Act outlines generated during initialization
    act_outlines: dict[int, dict[str, Any]]  # act_number -> act_outline

    # Chapter outlines (generated on-demand or pre-generated) - CANONICAL SOURCE
    # This is the primary source of truth for chapter outlines.
    # Schema per chapter: {chapter, act, scene_description, key_beats, plot_point, ...}
    chapter_outlines: dict[int, dict[str, Any]]  # chapter_number -> chapter_outline

    # Initialization state tracking
    initialization_complete: bool
    initialization_step: str | None  # Current initialization step

    # =========================================================================
    # Graph Healing State (for provisional node enrichment and merging)
    # =========================================================================
    provisional_count: int  # Number of provisional nodes in the graph
    last_healing_chapter: int  # Last chapter where healing was run
    merge_candidates: list[dict[str, Any]]  # Potential merge pairs with scores
    pending_merges: list[dict[str, Any]]  # Merges awaiting user approval
    auto_approved_merges: list[dict[str, Any]]  # High-confidence auto-approved merges
    healing_history: list[dict[str, Any]]  # Log of healing actions taken
    nodes_graduated: int  # Count of nodes graduated from provisional status
    nodes_merged: int  # Count of nodes merged in this session
    nodes_enriched: int  # Count of nodes enriched in this session


# Type alias for improved readability in node signatures
State = NarrativeState


def create_initial_state(
    *,
    project_id: str,
    title: str,
    genre: str,
    theme: str,
    setting: str,
    target_word_count: int,
    total_chapters: int,
    project_dir: str,
    protagonist_name: str,
    generation_model: str = settings.NARRATIVE_MODEL,
    extraction_model: str = settings.SMALL_MODEL,
    revision_model: str = settings.MEDIUM_MODEL,
    # New model params with defaults
    large_model: str = settings.LARGE_MODEL,
    medium_model: str = settings.MEDIUM_MODEL,
    small_model: str = settings.SMALL_MODEL,
    narrative_model: str = settings.NARRATIVE_MODEL,
    max_iterations: int = 3,
) -> NarrativeState:
    """
    Create initial state for a new narrative generation workflow.

    This factory function provides sensible defaults for all required fields,
    ensuring the state is properly initialized before entering the LangGraph workflow.

    Args:
        project_id: Unique identifier for the project
        title: Novel title
        genre: Novel genre
        theme: Central theme
        setting: Primary setting description
        target_word_count: Target word count for the complete novel
        total_chapters: Total number of chapters planned
        project_dir: Base directory for project files
        protagonist_name: Name of the protagonist
        generation_model: Model for text generation
        extraction_model: Model for entity extraction
        revision_model: Model for revision
        max_iterations: Maximum revision iterations per chapter

    Returns:
        Initialized NarrativeState ready for LangGraph workflow
    """
    import os

    state: NarrativeState = {
        # Project metadata
        "project_id": project_id,
        "title": title,
        "genre": genre,
        "theme": theme,
        "setting": setting,
        "target_word_count": target_word_count,
        # Position
        "current_chapter": 1,
        "total_chapters": total_chapters,
        "current_act": 1,
        # Neo4j connection (will be set by workflow)
        "neo4j_conn": None,
        # Outline (will be populated by planning node)
        "plot_outline": {},
        # Active context (initially empty)
        "active_characters": [],
        "current_location": None,
        "previous_chapter_summaries": [],
        "key_events": [],
        # Generated content
        "draft_text": None,
        "draft_word_count": 0,
        "generated_embedding": None,
        # Entity extraction
        "extracted_entities": {},
        "extracted_relationships": [],
        # Validation
        "contradictions": [],
        "needs_revision": False,
        "revision_feedback": None,
        "is_from_flawed_draft": False,
        # Quality metrics
        "coherence_score": None,
        "prose_quality_score": None,
        "plot_advancement_score": None,
        "pacing_score": None,
        "tone_consistency_score": None,
        "quality_feedback": None,
        # Model configuration
        "generation_model": generation_model,
        "extraction_model": extraction_model,
        "revision_model": revision_model,
        "large_model": large_model,
        "medium_model": medium_model,
        "small_model": small_model,
        "narrative_model": narrative_model,
        # Workflow control
        "current_node": "init",
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "force_continue": False,
        # Error handling
        "last_error": None,
        "has_fatal_error": False,
        "error_node": None,
        "retry_count": 0,
        # Filesystem paths
        "project_dir": project_dir,
        "chapters_dir": os.path.join(project_dir, "chapters"),
        "summaries_dir": os.path.join(project_dir, "summaries"),
        # Context management
        "context_epoch": 0,
        "hybrid_context": None,
        "kg_facts_block": None,
        # Chapter planning
        "chapter_plan": None,
        "plot_point_focus": None,
        "current_scene_index": 0,
        "scene_drafts": [],
        # Revision state
        "evaluation_result": None,
        "patch_instructions": None,
        # World building
        "world_items": [],
        "current_world_rules": [],
        # Protagonist
        "protagonist_name": protagonist_name,
        "protagonist_profile": None,
        # Initialization phase
        "character_sheets": {},
        "global_outline": None,
        "act_outlines": {},
        "chapter_outlines": {},
        "initialization_complete": False,
        "initialization_step": None,
        # Graph healing
        "provisional_count": 0,
        "last_healing_chapter": 0,
        "merge_candidates": [],
        "pending_merges": [],
        "auto_approved_merges": [],
        "healing_history": [],
        "nodes_graduated": 0,
        "nodes_merged": 0,
        "nodes_enriched": 0,
    }

    return state


__all__ = [
    "NarrativeState",
    "State",
    "ExtractedEntity",
    "ExtractedRelationship",
    "Contradiction",
    "create_initial_state",
]
