# core/langgraph/state.py
"""
Define the LangGraph state schema for SAGA workflows.

Migration Reference: docs/langgraph_migration_plan.md - Step 1.1.1

This module defines:
- Typed state used by LangGraph nodes (`NarrativeState`).
- Pydantic payload models for extracted entities/relationships and contradictions.
- A factory (`create_initial_state`) that initializes required fields with defaults.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator

# Import settings for model configuration
from config.settings import settings

# Import ContentRef for externalized content
from core.langgraph.content_manager import ContentRef
from core.schema_validator import schema_validator

# Import TypedDict structures for proper type annotations
# Import existing SAGA models for compatibility
from models.kg_models import CharacterProfile, WorldItem


class ExtractedEntity(BaseModel):
    """Represent an entity extracted from draft text prior to graph commit.

    Notes:
        Entity `type` is normalized via [`schema_validator`](core/schema_validator.py:124).
        When normalization occurs, the original type string may be preserved in
        `attributes["original_type"]` for downstream use (e.g., category hints).
    """

    name: str
    type: str  # Allows specific types from ontology (e.g. "Person", "Place")
    description: str
    first_appearance_chapter: int
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_entity_type(self) -> ExtractedEntity:
        """Normalize the extracted entity type using the schema validator."""
        current_type = self.type
        is_valid, normalized_type, _ = schema_validator.validate_entity_type(current_type)

        # Preserve the original specific type in attributes before normalization
        if is_valid and normalized_type != current_type:
            if "original_type" not in self.attributes:
                self.attributes["original_type"] = current_type
            if "category" not in self.attributes:
                self.attributes["category"] = current_type.lower()
            self.type = normalized_type

        return self

    class Config:
        """Configure Pydantic validation behavior."""

        frozen = False
        validate_assignment = True


class ExtractedRelationship(BaseModel):
    """Represent a relationship extracted from draft text prior to graph commit."""

    source_name: str
    target_name: str
    relationship_type: str
    description: str
    chapter: int
    confidence: float = 0.8
    source_type: str | None = None
    target_type: str | None = None

    class Config:
        """Configure Pydantic validation behavior."""

        frozen = False
        validate_assignment = True


class Contradiction(BaseModel):
    """Describe a detected inconsistency requiring revision or review."""

    type: str
    description: str
    conflicting_chapters: list[int]
    severity: Literal["minor", "major", "critical"]
    suggested_fix: str | None = None

    class Config:
        """Configure Pydantic validation behavior."""

        frozen = False
        validate_assignment = True


class NarrativeState(TypedDict, total=False):
    """Represent LangGraph workflow state for narrative generation.

    Notes:
        - This TypedDict uses `total=False`, but callers should treat the state as
          fully initialized via [`create_initial_state()`](core/langgraph/state.py:383).
        - Large payloads are typically externalized to disk via `*_ref` fields
          (see [`ContentManager`](core/langgraph/content_manager.py:42)).
        - Extraction is designed to be sequential per chapter:
          `extract_characters` resets extraction buckets, and subsequent extraction
          nodes append/replace within that same chapter cycle.
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
    narrative_style: str

    # =========================================================================
    # Current Position in Story
    # =========================================================================
    current_chapter: int
    total_chapters: int
    run_start_chapter: int

    # =========================================================================
    # Active Context (for prompt construction)
    # =========================================================================
    active_characters: list[CharacterProfile]  # Reuses existing model
    key_events: list[dict[str, Any]]

    # Externalized context references
    summaries_ref: ContentRef | None  # Reference to externalized summaries

    # =========================================================================
    # Generated Content (current chapter)
    # =========================================================================
    draft_word_count: int

    # Externalized content references
    draft_ref: ContentRef | None  # Reference to externalized draft text
    embedding_ref: ContentRef | None  # Reference to externalized embedding
    scene_embeddings_ref: ContentRef | None  # Reference to externalized scene embeddings (per chapter)

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
    # NOTE: These fields have been removed in favor of ContentRef-based externalization
    # extracted_entities: dict[str, list[dict[str, Any]]]
    # extracted_relationships: list[dict[str, Any]]

    # Externalized extraction references (to reduce state bloat)
    extracted_entities_ref: ContentRef | None  # Reference to externalized extracted entities
    extracted_relationships_ref: ContentRef | None  # Reference to externalized extracted relationships

    # =========================================================================
    # Validation and Quality Control (NEW: formalized validation state)
    # =========================================================================
    contradictions: list[Contradiction]
    needs_revision: bool
    revision_guidance_ref: ContentRef | None

    # Used by finalize/persistence to store the latest summary string without re-loading.
    current_summary: str | None

    # Phase 2 deduplication metadata (produced during commit).
    phase2_deduplication_merges: dict[str, int]

    # Quality assurance (periodic KG checks).
    last_qa_chapter: int
    qa_results: dict[str, Any]
    qa_history: list[dict[str, Any]]
    total_qa_issues: int
    total_qa_fixes: int

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

    # =========================================================================
    # Filesystem Paths
    # =========================================================================
    project_dir: str

    # =========================================================================
    # Context Management (maintains compatibility with existing context system)
    # =========================================================================
    # Externalized context references
    hybrid_context_ref: ContentRef | None  # Reference to externalized hybrid context

    # =========================================================================
    # Chapter Planning (properly typed with SceneDetail TypedDict)
    # =========================================================================
    # NOTE: chapter_plan field has been removed in favor of ContentRef-based externalization
    # chapter_plan: list[SceneDetail] | None  # List of SceneDetail TypedDicts
    current_scene_index: int  # Index of the scene currently being processed
    chapter_plan_scene_count: int  # Total number of scenes in the current chapter plan

    # Externalized scene drafts reference
    scene_drafts_ref: ContentRef | None  # Reference to externalized scene drafts
    chapter_plan_ref: ContentRef | None  # Reference to externalized chapter plan

    # =========================================================================
    # World Building Context
    # =========================================================================
    world_items: list[WorldItem]  # Reuses existing model
    current_world_rules: list[str]

    # =========================================================================
    # Protagonist and Key Characters
    # =========================================================================
    protagonist_name: str

    # =========================================================================
    # Initialization Phase State (for initialization workflow)
    # =========================================================================
    # Externalized initialization content references
    character_sheets_ref: ContentRef | None  # Reference to externalized character sheets
    global_outline_ref: ContentRef | None  # Reference to externalized global outline
    act_outlines_ref: ContentRef | None  # Reference to externalized act outlines
    chapter_outlines_ref: ContentRef | None  # Reference to externalized chapter outlines

    # Initialization state tracking
    initialization_complete: bool
    initialization_step: str | None  # Current initialization step

    # =========================================================================
    # Relationship Vocabulary (for normalization)
    # =========================================================================
    relationship_vocabulary: dict[str, Any]  # Maps canonical_type -> RelationshipUsage dict
    relationship_vocabulary_size: int  # Track vocabulary growth
    relationships_normalized_this_chapter: int  # Monitoring metric
    relationships_novel_this_chapter: int  # Monitoring metric
    last_pruned_chapter: int  # Track last chapter where vocabulary pruning ran

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
    nodes_removed: int  # Count of nodes removed during healing

    # Graph healing diagnostics (cached from the last run).
    last_healing_warnings: list[str]
    last_apoc_available: bool | None


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
    narrative_style: str = settings.DEFAULT_NARRATIVE_STYLE,
    extraction_model: str = settings.SMALL_MODEL,
    revision_model: str = settings.MEDIUM_MODEL,
    # New model params with defaults
    large_model: str = settings.LARGE_MODEL,
    medium_model: str = settings.MEDIUM_MODEL,
    small_model: str = settings.SMALL_MODEL,
    narrative_model: str = settings.NARRATIVE_MODEL,
    max_iterations: int = 2,
) -> NarrativeState:
    """Create an initial, ready-to-run LangGraph workflow state.

    Args:
        project_id: Unique identifier for the project.
        title: Novel title.
        genre: Novel genre.
        theme: Central theme.
        setting: Primary setting description.
        target_word_count: Target word count for the complete novel.
        total_chapters: Total number of chapters planned.
        project_dir: Base directory for project files.
        protagonist_name: Protagonist name used for prompts and initialization.
        extraction_model: Default model for entity/relationship extraction.
        revision_model: Default model for revision passes.
        large_model: Large model tier identifier (used by some nodes/subgraphs).
        medium_model: Medium model tier identifier (used by some nodes/subgraphs).
        small_model: Small model tier identifier (used by some nodes/subgraphs).
        narrative_model: Model identifier used by narrative generation nodes.
        max_iterations: Maximum number of revision cycles per chapter.

    Returns:
        A fully initialized state mapping suitable for `graph.invoke()` / `graph.ainvoke()`.
    """

    state: NarrativeState = {
        # Project metadata
        "project_id": project_id,
        "title": title,
        "genre": genre,
        "theme": theme,
        "setting": setting,
        "target_word_count": target_word_count,
        "narrative_style": narrative_style,
        # Position
        "current_chapter": 1,
        "total_chapters": total_chapters,
        "run_start_chapter": 1,
        # Active context (initially empty)
        "active_characters": [],
        "key_events": [],
        # Generated content
        "draft_word_count": 0,
        # Externalized content references
        "draft_ref": None,
        "embedding_ref": None,
        "scene_embeddings_ref": None,
        "summaries_ref": None,
        "scene_drafts_ref": None,
        "hybrid_context_ref": None,
        "character_sheets_ref": None,
        "global_outline_ref": None,
        "act_outlines_ref": None,
        "chapter_outlines_ref": None,
        "extracted_entities_ref": None,
        "extracted_relationships_ref": None,
        "chapter_plan_ref": None,
        "revision_guidance_ref": None,
        # Entity extraction
        # NOTE: These fields have been removed in favor of ContentRef-based externalization
        # "extracted_entities": {},
        # "extracted_relationships": [],
        # Validation
        "contradictions": [],
        "needs_revision": False,
        "current_summary": None,
        "phase2_deduplication_merges": {},
        "last_qa_chapter": 0,
        "qa_results": {},
        "qa_history": [],
        "total_qa_issues": 0,
        "total_qa_fixes": 0,
        # Quality metrics
        "coherence_score": None,
        "prose_quality_score": None,
        "plot_advancement_score": None,
        "pacing_score": None,
        "tone_consistency_score": None,
        "quality_feedback": None,
        # Model configuration
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
        # Filesystem paths
        "project_dir": project_dir,
        # Context management
        # Chapter planning
        # NOTE: chapter_plan field has been removed in favor of ContentRef-based externalization
        # "chapter_plan": None,
        "current_scene_index": 0,
        "chapter_plan_scene_count": 0,
        # World building
        "world_items": [],
        "current_world_rules": [],
        # Protagonist
        "protagonist_name": protagonist_name,
        # Initialization phase
        "initialization_complete": False,
        "initialization_step": None,
        # Relationship normalization
        "relationship_vocabulary": {},
        "relationship_vocabulary_size": 0,
        "relationships_normalized_this_chapter": 0,
        "relationships_novel_this_chapter": 0,
        "last_pruned_chapter": 0,
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
        "nodes_removed": 0,
        "last_healing_warnings": [],
        "last_apoc_available": None,
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
