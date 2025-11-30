# core/langgraph/state.py
"""
LangGraph State Schema for SAGA Narrative Generation.

This module defines the state schema for the LangGraph-based workflow,
designed to minimize disruption to existing SAGA code while enabling
the migration to LangGraph architecture.

Migration Reference: docs/langgraph_migration_plan.md - Step 1.1.1

State Field Organization:
- Metadata: Immutable project configuration
- Progress: Current position in the narrative
- Content: Generated text and drafts (mostly externalized via ContentRef)
- Extraction: Entities and relationships extracted from text
- Validation: Quality scores and contradiction detection
- Models: LLM model configuration
- Workflow: Control flow and iteration tracking
- Error Handling: Error state and recovery
- Filesystem: Directory paths
- Context: Dynamic context for generation
- Planning: Chapter and scene planning
- Revision: Revision state and feedback
- World Building: World items and rules
- Characters: Protagonist and character profiles
- Initialization: Initialization workflow state
- Graph Healing: Provisional node enrichment and merging
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

# Import settings for model configuration
from config.settings import settings

# Import TypedDict structures for proper type annotations
from models.agent_models import (
    EvaluationResult,
    PatchInstruction,
    SceneDetail,
)

# Import existing SAGA models for compatibility
from models.kg_models import CharacterProfile, WorldItem

# Import ContentRef for externalized content
try:
    from core.langgraph.content_manager import ContentRef
except ImportError:
    # Fallback for when content_manager is not available
    ContentRef = dict  # type: ignore


class ExtractedEntity(BaseModel):
    """
    Entity extracted from generated text (before Neo4j commit).

    This model represents entities identified during text generation
    that will be validated, deduplicated, and committed to the knowledge graph.
    """

    name: str
    type: str  # Allows specific types from ontology (e.g. "Person", "Place")
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

    ## State Update Patterns

    ### Sequential Extraction
    Extraction runs sequentially without reducers:
    1. `extract_characters`: CLEARS extraction state, extracts characters
    2. `extract_locations`: APPENDS locations to world_items
    3. `extract_events`: APPENDS events to world_items
    4. `extract_relationships`: Sets relationships

    This sequential approach prevents cross-chapter accumulation that occurred
    with reducer-based parallel extraction.

    ### Content Externalization
    Large content fields are externalized via ContentRef to avoid bloating the
    SQLite checkpoint database:
    - `draft_ref`: Reference to externalized draft text
    - `embedding_ref`: Reference to externalized embeddings
    - `scene_drafts_ref`: Reference to externalized scene drafts
    - And other `*_ref` fields

    ### Field Categories
    Fields are organized into logical categories (see module docstring):
    - Metadata: Immutable project configuration (project_id, genre, etc.)
    - Progress: Current position (current_chapter, current_act)
    - Content: Generated text (externalized via ContentRef)
    - Extraction: Entities and relationships from text
    - Validation: Quality scores and contradiction detection
    - Models: LLM model configuration
    - Workflow: Control flow and iteration tracking
    - Error Handling: Error state and recovery
    - And more (see individual sections below)

    ## Type Safety Notes
    This TypedDict uses `total=False`, making all fields technically optional.
    However, the `create_initial_state` factory function initializes all required
    fields with sensible defaults. Some fields are truly optional (e.g., error fields),
    while others should always be present after initialization.
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
    key_events: list[dict[str, Any]]

    # Externalized context references
    summaries_ref: ContentRef | None  # Reference to externalized summaries
    active_characters_ref: (
        ContentRef | None
    )  # Reference to externalized active characters

    # =========================================================================
    # Generated Content (current chapter)
    # =========================================================================
    draft_word_count: int

    # Externalized content references
    draft_ref: ContentRef | None  # Reference to externalized draft text
    embedding_ref: ContentRef | None  # Reference to externalized embedding

    # =========================================================================
    # Entity Extraction Results
    # =========================================================================
    # Sequential extraction (no reducers needed):
    # - extract_characters: Clears and populates extracted_entities["characters"]
    # - extract_locations: Appends to extracted_entities["world_items"]
    # - extract_events: Appends to extracted_entities["world_items"]
    # - extract_relationships: Populates extracted_relationships
    #
    # Each extraction cycle starts fresh by clearing these fields in the first node.
    extracted_entities: dict[str, list[ExtractedEntity]]
    extracted_relationships: list[ExtractedRelationship]

    # Externalized extraction references (to reduce state bloat)
    extracted_entities_ref: (
        ContentRef | None
    )  # Reference to externalized extracted entities
    extracted_relationships_ref: (
        ContentRef | None
    )  # Reference to externalized extracted relationships

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
    # Quality Metrics (LLM-evaluated quality scores)
    # =========================================================================
    coherence_score: float | None  # 0.0-1.0 score for narrative coherence
    prose_quality_score: float | None  # 0.0-1.0 score for prose quality
    plot_advancement_score: float | None  # 0.0-1.0 score for plot advancement
    pacing_score: float | None  # 0.0-1.0 score for narrative pacing
    tone_consistency_score: float | None  # 0.0-1.0 score for tone consistency
    quality_feedback: str | None  # Free-form feedback summarizing strengths/weaknesses

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

    # Externalized context references
    hybrid_context_ref: ContentRef | None  # Reference to externalized hybrid context
    kg_facts_ref: ContentRef | None  # Reference to externalized KG facts

    # =========================================================================
    # Chapter Planning (properly typed with SceneDetail TypedDict)
    # =========================================================================
    chapter_plan: list[SceneDetail] | None  # List of SceneDetail TypedDicts
    plot_point_focus: str | None
    current_scene_index: int  # Index of the scene currently being processed

    # Externalized scene drafts reference
    scene_drafts_ref: ContentRef | None  # Reference to externalized scene drafts
    chapter_plan_ref: ContentRef | None  # Reference to externalized chapter plan

    # =========================================================================
    # Revision State (properly typed with TypedDict structures)
    # =========================================================================
    evaluation_result: EvaluationResult | None  # EvaluationResult TypedDict
    patch_instructions: (
        list[PatchInstruction] | None
    )  # List of PatchInstruction TypedDicts

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
    # Externalized initialization content references
    character_sheets_ref: (
        ContentRef | None
    )  # Reference to externalized character sheets
    global_outline_ref: ContentRef | None  # Reference to externalized global outline
    act_outlines_ref: ContentRef | None  # Reference to externalized act outlines
    chapter_outlines_ref: (
        ContentRef | None
    )  # Reference to externalized chapter outlines

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
        "key_events": [],
        # Generated content
        "draft_word_count": 0,
        # Externalized content references
        "draft_ref": None,
        "embedding_ref": None,
        "summaries_ref": None,
        "scene_drafts_ref": None,
        "hybrid_context_ref": None,
        "kg_facts_ref": None,
        "character_sheets_ref": None,
        "global_outline_ref": None,
        "act_outlines_ref": None,
        "chapter_outlines_ref": None,
        "extracted_entities_ref": None,
        "extracted_relationships_ref": None,
        "active_characters_ref": None,
        "chapter_plan_ref": None,
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
        # Chapter planning
        "chapter_plan": None,
        "plot_point_focus": None,
        "current_scene_index": 0,
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
