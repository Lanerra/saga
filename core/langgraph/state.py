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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Import settings for model configuration
import config

# Import ContentRef for externalized content
from core.langgraph.content_manager import ContentRef
from core.schema_validator import schema_validator


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

    model_config = ConfigDict(frozen=False, validate_assignment=True)


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
    source_id: str | None = Field(default=None, strict=True)
    target_id: str | None = Field(default=None, strict=True)
    scene_index: int | None = Field(default=None, ge=0, strict=True)
    scene_assertions: list[dict[str, Any]] | None = None

    @field_validator("source_id", "target_id")
    @classmethod
    def exact_identity(cls, value: str | None) -> str | None:
        if value is not None and (not value or value != value.strip()):
            raise ValueError("Explicit relationship ID must be an exact nonblank string")
        return value

    @model_validator(mode="after")
    def validate_scene_assertions(self) -> ExtractedRelationship:
        if self.scene_assertions is None:
            return self
        if not self.scene_assertions or self.scene_index is None:
            raise ValueError("Scene assertions require a nonempty history and selected scene_index")
        parent = self.model_dump(exclude={"scene_assertions"})
        previous_index = -1
        for assertion in self.scene_assertions:
            if "scene_index" not in assertion or "description" not in assertion or "scene_assertions" in assertion:
                raise ValueError("Scene assertion requires scene_index and description without nested history")
            for field in ("source_id", "target_id", "source_name", "target_name", "source_type", "target_type", "relationship_type", "chapter"):
                if field in assertion and assertion[field] != parent[field]:
                    raise ValueError("Scene assertion conflicts with relationship identity or chapter")
            validated = ExtractedRelationship(**{**parent, **assertion})
            if validated.scene_index is None or not previous_index <= validated.scene_index <= self.scene_index:
                raise ValueError("Scene assertions must be chronological and not later than the selected scene")
            previous_index = validated.scene_index
        return self

    model_config = ConfigDict(frozen=False, validate_assignment=True, extra="forbid")


class Contradiction(BaseModel):
    """Describe a detected inconsistency requiring revision or review."""

    type: str
    description: str
    conflicting_chapters: list[int]
    severity: Literal["minor", "major", "critical"]
    suggested_fix: str | None = None

    model_config = ConfigDict(frozen=False, validate_assignment=True)


SceneExtractionType = Literal["characters", "locations", "events", "relationships"]


class SceneExtractionOutcome(TypedDict):
    """Record completion of one scene/type slot, independently of deduplication."""

    chapter_number: int
    scene_index: int
    extraction_type: SceneExtractionType
    status: Literal["succeeded", "failed"]
    item_count: int
    error_type: str
    error: str


class RevisionRollbackFailure(TypedDict):
    """Retain the rejected attempt and diagnostics until lifecycle reconciliation."""

    chapter_number: int
    iteration_count: int
    error: str
    previous_error: str | None
    previous_error_node: str | None


class NarrativeState(TypedDict, total=False):
    """Represent LangGraph workflow state for narrative generation.

    Notes:
        - This is the checkpoint channel superset and the partial node-update type.
          Admission validates required fields through phase-specific projections.
        - Optional telemetry is absent until produced; absence is not a completed check.
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
    graph_project_id: str
    lifecycle_version: Literal[1]
    quality_policy: dict[str, Any]
    quality_checks: dict[str, Any]
    graph_quality_check: dict[str, Any]
    attempt_id: str | None
    lifecycle_phase: str
    extraction_source: dict[str, Any] | None
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

    # Externalized context references
    summaries_ref: ContentRef | None  # Reference to externalized summaries

    # =========================================================================
    # Generated Content (current chapter)
    # =========================================================================
    draft_word_count: int

    # Externalized content references
    draft_ref: ContentRef | None  # Reference to externalized draft text
    embedding_ref: ContentRef | None  # Reference to externalized embedding
    # Input-only compatibility channel: reject non-null legacy payloads at admission.
    generated_embedding: object
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

    extraction_policy: Literal["fail_closed"]
    extraction_status: Literal["complete", "failed"]
    extraction_outcomes: list[SceneExtractionOutcome]

    # =========================================================================
    # Validation and Quality Control (NEW: formalized validation state)
    # =========================================================================
    contradictions: list[Contradiction]
    needs_revision: bool
    revision_guidance_ref: ContentRef | None

    # Used by finalize/persistence to store the latest summary string without re-loading.
    current_summary: str | None

    # Quality assurance (periodic KG checks).
    last_qa_chapter: int
    qa_results: dict[str, Any]
    qa_history: list[dict[str, Any]]
    total_qa_issues: int
    total_qa_fixes: int

    # =========================================================================
    # Quality metrics retained in acceptance evidence and checkpoints.
    # =========================================================================
    coherence_score: float | None
    prose_quality_score: float | None
    plot_advancement_score: float | None
    pacing_score: float | None
    tone_consistency_score: float | None
    quality_feedback: str | None

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
    revision_rollback_failure: RevisionRollbackFailure | None
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
    # Protagonist and Key Characters
    # =========================================================================
    protagonist_name: str

    # =========================================================================
    # Initialization Phase State (for initialization workflow)
    # =========================================================================
    # Externalized initialization content references
    character_sheets_ref: ContentRef | None  # Reference to externalized character sheets
    initialization_id: str
    initialization_catalog_ref: ContentRef | None
    global_outline_ref: ContentRef | None  # Reference to externalized global outline
    act_outlines_ref: ContentRef | None  # Reference to externalized act outlines
    outline_relationships_ref: ContentRef | None  # Reference to externalized outline relationships
    chapter_outlines_ref: ContentRef | None  # Reference to externalized chapter outlines

    # Initialization state tracking
    initialization_complete: bool
    initialization_step: str | None  # Current initialization step

    # =========================================================================
    # Relationship Vocabulary (for normalization)
    # =========================================================================
    # Vocabulary and normalization telemetry persist across checkpoints.
    relationship_vocabulary: dict[str, Any]  # Maps canonical_type -> RelationshipUsage dict
    relationship_vocabulary_size: int
    relationships_normalized_this_chapter: int
    relationships_novel_this_chapter: int
    last_pruned_chapter: int  # Track last chapter where vocabulary pruning ran

    # =========================================================================
    # Graph Healing State (for provisional node enrichment and merging)
    # =========================================================================
    provisional_count: int  # Number of provisional nodes in the graph
    last_healing_chapter: int
    healing_history: list[dict[str, Any]]
    nodes_graduated: int
    nodes_merged: int
    nodes_enriched: int
    nodes_removed: int

    # Graph healing diagnostics (cached from the last run).
    last_healing_warnings: list[str]
    last_apoc_available: bool | None


# Type alias for improved readability in node signatures
State = NarrativeState


class AuthoringState(BaseModel):
    """Required admission projection, never a replacement for checkpoint values."""

    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)

    project_id: str = Field(min_length=1)
    project_dir: str = Field(min_length=1)
    title: str = Field(min_length=1)
    genre: str
    theme: str
    setting: str
    protagonist_name: str = Field(min_length=1)
    narrative_style: str = Field(min_length=1)
    target_word_count: int = Field(gt=0)
    total_chapters: int = Field(gt=0)
    current_chapter: int = Field(gt=0)
    run_start_chapter: int = Field(gt=0)
    initialization_complete: bool
    large_model: str = Field(min_length=1)
    medium_model: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_word_allocation(self) -> AuthoringState:
        if self.target_word_count < self.total_chapters:
            raise ValueError("target_word_count must allocate at least one word per chapter")
        return self


class InitializationState(AuthoringState):
    """Inputs required while producing the retained initialization artifacts."""

    initialization_complete: Literal[False]


class GenerationState(AuthoringState):
    """Inputs required for scene generation and subsequent chapter attempts."""

    initialization_complete: Literal[True]
    narrative_model: str = Field(min_length=1)
    extraction_model: str = Field(min_length=1)
    revision_model: str = Field(min_length=1)
    small_model: str = Field(min_length=1)
    iteration_count: int = Field(ge=0)
    max_iterations: int = Field(ge=0)
    current_scene_index: int = Field(ge=0)
    chapter_plan_scene_count: int = Field(ge=0)
    force_continue: bool
    quality_policy: dict[str, Any]


def validate_state_contract(state: NarrativeState) -> None:
    """Admit retained values without filling policy from current configuration.

    Historical null/absent generated_embedding channels are compatible. Every
    other value requires explicit offline recovery from identified artifacts;
    neither raw vectors nor references in that ambiguous channel are reinterpreted.
    Keep the channel so native loading cannot silently discard recovery evidence.
    """
    if state.get("generated_embedding") is not None:
        raise ValueError(
            "Legacy generated_embedding is not supported; preserve the checkpoint and artifacts, "
            "then explicitly recover an identified embedding_ref or scene_embeddings_ref. Producer identity cannot be inferred."
        )
    if type(state.get("initialization_complete")) is not bool:
        raise ValueError("initialization_complete must be an explicit boolean")
    contract = GenerationState if state["initialization_complete"] else InitializationState
    contract.model_validate(state)


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
    narrative_style: str | None = None,
    extraction_model: str | None = None,
    revision_model: str | None = None,
    # New model params with defaults
    large_model: str | None = None,
    medium_model: str | None = None,
    small_model: str | None = None,
    narrative_model: str | None = None,
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
        "narrative_style": config.settings.DEFAULT_NARRATIVE_STYLE if narrative_style is None else narrative_style,
        # Position
        "current_chapter": 1,
        "total_chapters": total_chapters,
        "run_start_chapter": 1,
        # Active context (initially empty)
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
        "outline_relationships_ref": None,
        "initialization_catalog_ref": None,
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
        "quality_checks": {},
        "graph_quality_check": {},
        "current_summary": None,
        # Model configuration
        "extraction_model": config.settings.SMALL_MODEL if extraction_model is None else extraction_model,
        "revision_model": config.settings.MEDIUM_MODEL if revision_model is None else revision_model,
        "large_model": config.settings.LARGE_MODEL if large_model is None else large_model,
        "medium_model": config.settings.MEDIUM_MODEL if medium_model is None else medium_model,
        "small_model": config.settings.SMALL_MODEL if small_model is None else small_model,
        "narrative_model": config.settings.NARRATIVE_MODEL if narrative_model is None else narrative_model,
        # Workflow control
        "current_node": "init",
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "force_continue": False,
        # Error handling
        "last_error": None,
        "revision_rollback_failure": None,
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
        # Protagonist
        "protagonist_name": protagonist_name,
        # Initialization phase
        "initialization_complete": False,
        "initialization_step": None,
        # Relationship normalization
        "relationship_vocabulary": {},
    }

    from core.langgraph.quality_policy import configured_policy

    state["quality_policy"] = configured_policy()
    validate_state_contract(state)
    return state


__all__ = [
    "NarrativeState",
    "State",
    "ExtractedEntity",
    "ExtractedRelationship",
    "Contradiction",
    "create_initial_state",
]
