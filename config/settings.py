# config/settings.py
"""
Define runtime configuration for SAGA.

Settings are loaded at import time by constructing a [`SagaSettings`](config/settings.py:153)
instance. Values come from the process environment and may be sourced from a `.env` file.

Import-time side effects:
- Read `.env` via `dotenv.load_dotenv()` (non-overriding) and via Pydantic's configured
  `env_file=".env"`.
- Create output directories under `BASE_OUTPUT_DIR`.
- Configure structlog and attach a handler to the root logger.

Notes:
    Environment variables already present in the process take precedence over values loaded
    from `.env` during import. Reloading via [`config.loader.reload_settings()`](config/loader.py:35)
    uses `load_dotenv(override=True)` which can overwrite existing environment variables.
"""

from __future__ import annotations

import logging as stdlib_logging
import os
from collections.abc import MutableMapping
from typing import Any

import structlog
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

logger = structlog.get_logger()


class SchemaEnforcementSettings(BaseSettings):
    """Configure node-label schema enforcement.

    This settings group controls how strictly SAGA validates entity types and whether it
    normalizes common label variants into canonical values.

    Notes:
        Environment variables use the `SAGA_SCHEMA_` prefix.
    """

    ENFORCE_SCHEMA_VALIDATION: bool = Field(default=True, description="Master toggle for schema validation")

    REJECT_INVALID_ENTITIES: bool = Field(
        default=True,
        description="If True, entities with invalid types are rejected. If False, soft validation (warnings).",
    )

    NORMALIZE_COMMON_VARIANTS: bool = Field(
        default=True,
        description="Automatically map common variants (Person->Character) to canonical labels",
    )

    LOG_SCHEMA_VIOLATIONS: bool = Field(default=True, description="Log detailed warnings when schema violations occur")

    class Config:
        env_prefix = "SAGA_SCHEMA_"


class ValidationSettings(BaseSettings):
    """Configure validation behavior.

    This settings group controls whether validation checks are performed
    during chapter generation.

    Notes:
        Environment variables use the `SAGA_VALIDATION_` prefix.
    """

    ENABLE_VALIDATION: bool = Field(default=True, description="Enable validation checks in the validation node")

    class Config:
        env_prefix = "DEBUG_"


class RelationshipNormalizationSettings(BaseSettings):
    """Configure relationship type normalization.

    This settings group controls whether and how SAGA normalizes relationship type names
    into a stable vocabulary. When enabled, normalization affects how relationships are
    written to and queried from the knowledge graph.

    Notes:
        Environment variables use the `SAGA_REL_NORM_` prefix.
    """

    # Master toggle
    ENABLE_RELATIONSHIP_NORMALIZATION: bool = Field(default=True, description="Enable relationship normalization system")

    # Strict canonical mode
    STRICT_CANONICAL_MODE: bool = Field(
        default=True,
        description="If True, disables dynamic vocabulary expansion and rejects unknown types",
    )

    STATIC_OVERRIDES_ENABLED: bool = Field(
        default=True,
        description="Enable static dictionary lookups for exact synonyms",
    )

    # Category-specific similarity thresholds
    SIMILARITY_THRESHOLDS: dict[str, float] = Field(
        default={
            "CHARACTER_CHARACTER": 0.75,
            "CHARACTER_WORLD": 0.70,
            "PLOT_STRUCTURE": 0.80,
            "DEFAULT": 0.75,
        },
        description="Category-specific similarity thresholds for relationship canonicalization",
    )

    # Legacy similarity thresholds (for backward compatibility)
    SIMILARITY_THRESHOLD: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for normalizing relationships",
    )

    SIMILARITY_THRESHOLD_AMBIGUOUS_MIN: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum similarity for ambiguous cases requiring LLM review",
    )

    # Vocabulary management
    MIN_USAGE_FOR_AUTHORITY: int = Field(
        default=5,
        ge=1,
        description="Relationship must be used this many times before it's authoritative",
    )

    PRUNE_SINGLE_USE_AFTER_CHAPTERS: int = Field(
        default=5,
        ge=1,
        description="Remove single-use relationships after this many chapters",
    )

    MAX_VOCABULARY_SIZE: int = Field(
        default=50,
        ge=10,
        description="Maximum number of relationship types to maintain",
    )

    # Example retention
    MAX_EXAMPLES_PER_RELATIONSHIP: int = Field(
        default=3,
        ge=1,
        description="Maximum example descriptions to keep per relationship type",
    )

    # Advanced features
    USE_LLM_DISAMBIGUATION: bool = Field(default=True, description="Use LLM to disambiguate ambiguous similarity cases")

    LLM_DISAMBIGUATION_JSON_MODE: bool = Field(
        default=True,
        description="If True, require strict JSON output for relationship normalization disambiguation",
    )

    NORMALIZE_CASE_VARIANTS: bool = Field(
        default=True,
        description="Treat case variations as identical (WORKS_WITH == works_with)",
    )

    NORMALIZE_PUNCTUATION_VARIANTS: bool = Field(
        default=True,
        description="Treat punctuation variations as identical (WORKS_WITH == WORKS-WITH)",
    )

    class Config:
        env_prefix = "SAGA_REL_NORM_"


class SagaSettings(BaseSettings):
    """Define the SAGA settings model.

    Instantiating this model reads configuration from environment variables and `.env`
    (per `model_config`). Pydantic performs type coercion and validation at construction
    time; invalid values typically represent user configuration errors and raise
    `pydantic.ValidationError`.

    Notes:
        This module constructs a singleton [`settings`](config/settings.py:356) instance at
        import time. Any validation failure therefore fails fast during import.
    """

    # API and Model Configuration
    EMBEDDING_API_BASE: str = "http://127.0.0.1:11434"
    EMBEDDING_API_KEY: str = ""
    OPENAI_API_BASE: str = "http://127.0.0.1:8080/v1"
    OPENAI_API_KEY: str = "nope"

    EMBEDDING_MODEL: str = "nomic-embed-text:latest"
    EMBEDDING_MAX_INPUT_TOKENS: int = 8192
    EXPECTED_EMBEDDING_DIM: int = 768
    EMBEDDING_DTYPE: str = "float16"

    # Neo4j Connection Settings
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "saga_password"
    NEO4J_DATABASE: str | None = "neo4j"

    # Neo4j Vector Index Configuration (Chapters)
    NEO4J_VECTOR_INDEX_NAME: str = "chapterEmbeddings"
    NEO4J_VECTOR_DIMENSIONS: int = 768
    NEO4J_VECTOR_SIMILARITY_FUNCTION: str = "cosine"

    # Neo4j Vector Index Configuration (Entities)
    #
    # Entity embeddings are stored on domain entities (Character/Location/Item/Event)
    # under a separate property to avoid confusion with Chapter embeddings.
    ENTITY_EMBEDDING_VECTOR_PROPERTY: str = "entity_embedding_vector"
    ENTITY_EMBEDDING_TEXT_HASH_PROPERTY: str = "entity_embedding_text_hash"
    ENTITY_EMBEDDING_MODEL_PROPERTY: str = "entity_embedding_model"

    NEO4J_CHARACTER_ENTITY_VECTOR_INDEX_NAME: str = "characterEntityEmbeddings"
    NEO4J_LOCATION_ENTITY_VECTOR_INDEX_NAME: str = "locationEntityEmbeddings"
    NEO4J_ITEM_ENTITY_VECTOR_INDEX_NAME: str = "itemEntityEmbeddings"
    NEO4J_EVENT_ENTITY_VECTOR_INDEX_NAME: str = "eventEntityEmbeddings"

    # Entity embeddings feature flags
    #
    # Default off to keep unit tests deterministic and to avoid introducing new
    # embedding-service dependencies into unrelated workflows. Enable explicitly
    # when you want entity-level semantic deduplication and merge scoring.
    ENABLE_ENTITY_EMBEDDING_PERSISTENCE: bool = False
    ENABLE_ENTITY_EMBEDDING_DEDUPLICATION: bool = True
    ENABLE_ENTITY_EMBEDDING_GRAPH_HEALING: bool = True

    # Entity embedding similarity configuration
    ENTITY_EMBEDDING_DEDUPLICATION_TOP_K: int = 15
    ENTITY_EMBEDDING_DEDUPLICATION_SIMILARITY_THRESHOLD: float = 0.85

    # Base Model Definitions
    LARGE_MODEL: str = "qwen3-a3b"
    MEDIUM_MODEL: str = "qwen3-a3b"
    SMALL_MODEL: str = "qwen3-a3b"
    NARRATIVE_MODEL: str = "qwen3-a3b"

    # Temperature Settings
    TEMPERATURE_INITIAL_SETUP: float = 0.7
    TEMPERATURE_DRAFTING: float = 0.7
    TEMPERATURE_REVISION: float = 0.65
    TEMPERATURE_PLANNING: float = 0.6
    TEMPERATURE_EVALUATION: float = 0.3
    TEMPERATURE_CONSISTENCY_CHECK: float = 0.2
    TEMPERATURE_KG_EXTRACTION: float = 0.1
    TEMPERATURE_SUMMARY: float = 0.3
    TEMPERATURE_PATCH: float = 0.7

    # Global Temperature Override
    TEMPERATURE_OVERRIDE: float | None = None

    FILL_IN: str = ""

    # LLM Call Settings & Fallbacks
    LLM_RETRY_ATTEMPTS: int = 3
    LLM_RETRY_DELAY_SECONDS: float = 3.0
    HTTPX_TIMEOUT: float = 120.0
    ENABLE_LLM_NO_THINK_DIRECTIVE: bool = False
    TIKTOKEN_DEFAULT_ENCODING: str = "cl100k_base"
    FALLBACK_CHARS_PER_TOKEN: float = 4.0

    # Concurrency and Rate Limiting
    MAX_CONCURRENT_LLM_CALLS: int = 1
    LLM_TOP_P: float = 0.95

    # LLM Frequency and Presence Penalties
    FREQUENCY_PENALTY_DRAFTING: float = 0.0
    PRESENCE_PENALTY_DRAFTING: float = 0.0

    # Output and File Paths
    PROJECTS_ROOT: str = "projects"
    BASE_OUTPUT_DIR: str = "output"
    PLOT_OUTLINE_FILE: str = "plot_outline.json"
    CHARACTER_PROFILES_FILE: str = "character_profiles.json"
    WORLD_BUILDER_FILE: str = "world_building.json"
    CHAPTERS_DIR: str = "chapters"
    CHAPTER_LOGS_DIR: str = "chapter_logs"

    USER_STORY_ELEMENTS_FILE_PATH: str = "user_story_elements.yaml"

    # Generation Parameters
    # Token budgets (defaults are generous)
    MAX_CONTEXT_TOKENS: int = 32768
    MAX_GENERATION_TOKENS: int = 16384
    CONTEXT_CHAPTER_COUNT: int = 2
    CHAPTERS_PER_RUN: int = 3
    TOTAL_CHAPTERS: int = 15
    TARGET_PLOT_POINTS_INITIAL_GENERATION: int = 12
    MAX_CONCURRENT_CHAPTERS: int = 1

    # Caching
    EMBEDDING_CACHE_SIZE: int = 128
    SUMMARY_CACHE_SIZE: int = 32
    KG_TRIPLE_EXTRACTION_CACHE_SIZE: int = 16
    TOKENIZER_CACHE_SIZE: int = 10

    # Agentic Planning & Prompt Context Snippets
    MAX_PLANNING_TOKENS: int = 16384
    TARGET_SCENES_MIN: int = 4
    TARGET_SCENES_MAX: int = 6

    # Revision and Validation
    REVISION_EVALUATION_THRESHOLD: float = 0.85
    MIN_QUALITY_THRESHOLD: float = 0.7
    PLOT_STAGNATION_MIN_WORD_COUNT: int = 1500
    TARGET_WORD_COUNT: int = 80000
    MAX_REVISION_CYCLES_PER_CHAPTER: int = 2
    MAX_SUMMARY_TOKENS: int = 16384
    MAX_KG_TRIPLE_TOKENS: int = 16384
    MAX_PREPOP_KG_TOKENS: int = 16384

    # Quality Assurance Configuration
    ENABLE_QA_CHECKS: bool = False
    QA_CHECK_FREQUENCY: int = 3
    QA_CHECK_CONTRADICTORY_TRAITS: bool = True
    QA_CHECK_POST_MORTEM_ACTIVITY: bool = True
    QA_DEDUPLICATE_RELATIONSHIPS: bool = True
    QA_CONSOLIDATE_RELATIONSHIPS: bool = True

    # Knowledge Graph Entity Filtering (Proper Noun Preference)
    ENTITY_MENTION_THRESHOLD_PROPER_NOUN: int = 1
    ENTITY_MENTION_THRESHOLD_COMMON_NOUN: int = 3
    RELATIONSHIP_LOWERCASE_TARGET_ALLOWLIST: list[str] = []

    # Narrative Agent Configuration
    KG_PREPOPULATION_CHAPTER_NUM: int = 0

    # De-duplication Configuration
    # DEPRECATED: Deduplication is no longer needed per Phase 4 requirements
    # Entities are canonical from Stage 1, so deduplication is disabled
    DEDUPLICATION_USE_SEMANTIC: bool = False
    DEDUPLICATION_SEMANTIC_THRESHOLD: float = 0.55
    DEDUPLICATION_MIN_SEGMENT_LENGTH: int = 150

    # DEPRECATED: Duplicate prevention is disabled per Phase 4 requirements
    # Entities are canonical from Stage 1
    ENABLE_DUPLICATE_PREVENTION: bool = False
    DUPLICATE_PREVENTION_SIMILARITY_THRESHOLD: float = 0.6
    DUPLICATE_PREVENTION_CHARACTER_ENABLED: bool = False
    DUPLICATE_PREVENTION_WORLD_ITEM_ENABLED: bool = False

    # DEPRECATED: Phase 2 deduplication is disabled per Phase 4 requirements
    # Relationships are canonical from Stage 1
    ENABLE_PHASE2_DEDUPLICATION: bool = False
    PHASE2_NAME_SIMILARITY_THRESHOLD: float = 0.5
    PHASE2_RELATIONSHIP_SIMILARITY_THRESHOLD: float = 0.6

    # Chapter Generation Configuration
    MIN_CHAPTER_LENGTH_CHARS: int = 12000  # Approximately 2500-3000 words
    GENERATE_ALL_CHAPTER_OUTLINES_AT_INIT: bool = True

    # Narrative Style Defaults
    DEFAULT_NARRATIVE_STYLE: str = "Third-Person, personal with internal monologue"

    # Logging & UI
    LOG_LEVEL_STR: str = Field("INFO", alias="LOG_LEVEL")
    LOG_FORMAT: str = "%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_FILE: str | None = "saga_run.log"
    ENABLE_RICH_PROGRESS: bool = True
    # Minimal logging mode for single-user setups: console only, no rotation/Rich
    SIMPLE_LOGGING_MODE: bool = False

    # NLP / spaCy configuration
    SPACY_MODEL: str | None = None  # default None => utils.text_processing uses en_core_web_lg
    ENABLE_ENTITY_VALIDATION: bool = True  # Enable spaCy-based entity validation during extraction

    # Stage 5: Narrative Generation & Enrichment Configuration
    # These settings control what can be extracted from narrative text in Stage 5
    # According to schema design, Stage 5 should only extract physical descriptions and embeddings
    # It should NOT create new structural entities (Characters, Events, Locations, Items)

    # Physical description extraction settings
    ENABLE_PHYSICAL_DESCRIPTION_EXTRACTION: bool = True  # Extract physical descriptions from narrative
    ENABLE_PHYSICAL_DESCRIPTION_VALIDATION: bool = True  # Validate extracted descriptions against existing properties

    # Chapter embedding extraction settings
    ENABLE_CHAPTER_EMBEDDING_EXTRACTION: bool = True  # Extract chapter embeddings from narrative

    # DEPRECATED: These settings are permanently disabled per Phase 4 requirements
    # Stage 5 should NOT extract or create new structural entities
    # These are kept for backward compatibility but should never be enabled
    ENABLE_CHARACTER_EXTRACTION_FROM_NARRATIVE: bool = False  # DEPRECATED - characters created in Stage 1 only
    ENABLE_LOCATION_EXTRACTION_FROM_NARRATIVE: bool = False  # DEPRECATED - locations created in Stage 2/3 only
    ENABLE_EVENT_EXTRACTION_FROM_NARRATIVE: bool = False  # DEPRECATED - events created in Stage 2/3/4 only
    ENABLE_ITEM_EXTRACTION_FROM_NARRATIVE: bool = False  # DEPRECATED - items created in Stage 2 only

    # DEPRECATED: Relationship extraction is permanently disabled
    # Relationships are canonical from Stage 1 and should not be extracted from narrative
    ENABLE_RELATIONSHIP_EXTRACTION_FROM_NARRATIVE: bool = False  # DEPRECATED - relationships created in Stage 1 only

    # Novel Configuration (Defaults / Placeholders)
    CONFIGURED_GENRE: str = "grimdark science fiction"
    CONFIGURED_THEME: str = "the hubris of humanity"
    CONFIGURED_SETTING_DESCRIPTION: str = "a remote outpost on the surface of Jupiter's moon, Callisto"
    DEFAULT_PROTAGONIST_NAME: str = "Ilya Lakatos"
    DEFAULT_PLOT_OUTLINE_TITLE: str = "Untitled Narrative"

    MAIN_NOVEL_INFO_NODE_ID: str = "main_novel_info"

    # Identifier for the root World Container node in the Neo4j graph.
    # This constant is used throughout the codebase for bootstrapping and
    # querying world‑level structures.  It was previously defined in the
    # legacy ``config.py`` file; adding it here restores compatibility.
    MAIN_WORLD_CONTAINER_NODE_ID: str = "world_container"

    # Enhanced character bootstrap settings
    BOOTSTRAP_MIN_TRAITS_PROTAGONIST: int = 6
    BOOTSTRAP_MIN_TRAITS_ANTAGONIST: int = 5
    BOOTSTRAP_MIN_TRAITS_SUPPORTING: int = 4

    # DEPRECATED: Relationship normalization is disabled per Phase 4 requirements
    # Relationships are canonical from Stage 1 and should not be normalized
    relationship_normalization: RelationshipNormalizationSettings = Field(
        default_factory=lambda: RelationshipNormalizationSettings(
            ENABLE_RELATIONSHIP_NORMALIZATION=True,
            STRICT_CANONICAL_MODE=True,
            STATIC_OVERRIDES_ENABLED=False,
        )
    )

    # Schema Enforcement
    schema_enforcement: SchemaEnforcementSettings = Field(default_factory=SchemaEnforcementSettings)

    # Validation Settings
    validation: ValidationSettings = Field(default_factory=lambda: ValidationSettings())

    # Legacy Degradation Flags
    ENABLE_STATUS_IS_ALIAS: bool = False

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")


settings = SagaSettings()


# --- Reconstruct objects for backward compatibility ---
class ModelsCompat:
    LARGE: str
    MEDIUM: str
    SMALL: str
    NARRATOR: str


class TempsCompat:
    INITIAL_SETUP: float
    DRAFTING: float
    REVISION: float
    PLANNING: float
    EVALUATION: float
    CONSISTENCY_CHECK: float
    KG_EXTRACTION: float
    SUMMARY: float
    PATCH: float
    DEFAULT: float
    OVERRIDE: float | None


Models = ModelsCompat()
Models.LARGE = settings.LARGE_MODEL
Models.MEDIUM = settings.MEDIUM_MODEL
Models.SMALL = settings.SMALL_MODEL
Models.NARRATOR = settings.NARRATIVE_MODEL

Temperatures = TempsCompat()
Temperatures.INITIAL_SETUP = settings.TEMPERATURE_INITIAL_SETUP
Temperatures.DRAFTING = settings.TEMPERATURE_DRAFTING
Temperatures.REVISION = settings.TEMPERATURE_REVISION
Temperatures.PLANNING = settings.TEMPERATURE_PLANNING
Temperatures.EVALUATION = settings.TEMPERATURE_EVALUATION
Temperatures.CONSISTENCY_CHECK = settings.TEMPERATURE_CONSISTENCY_CHECK
Temperatures.KG_EXTRACTION = settings.TEMPERATURE_KG_EXTRACTION
Temperatures.SUMMARY = settings.TEMPERATURE_SUMMARY
Temperatures.PATCH = settings.TEMPERATURE_PATCH
Temperatures.DEFAULT = 0.7  # Set default explicitly
Temperatures.OVERRIDE = settings.TEMPERATURE_OVERRIDE


# Update module level variables for backward compatibility
for _field in settings.model_fields:
    globals()[_field] = getattr(settings, _field)


PLOT_OUTLINE_FILE = os.path.join(settings.BASE_OUTPUT_DIR, settings.PLOT_OUTLINE_FILE)
CHARACTER_PROFILES_FILE = os.path.join(settings.BASE_OUTPUT_DIR, settings.CHARACTER_PROFILES_FILE)
WORLD_BUILDER_FILE = os.path.join(settings.BASE_OUTPUT_DIR, settings.WORLD_BUILDER_FILE)
CHAPTERS_DIR = os.path.join(settings.BASE_OUTPUT_DIR, settings.CHAPTERS_DIR)
CHAPTER_LOGS_DIR = os.path.join(settings.BASE_OUTPUT_DIR, settings.CHAPTER_LOGS_DIR)

# Ensure output directories exist
os.makedirs(settings.BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(CHAPTERS_DIR, exist_ok=True)
os.makedirs(CHAPTER_LOGS_DIR, exist_ok=True)

# Configure structlog to integrate with standard logging and output human‑readable messages
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%m/%d/%Y, %H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


# Filter internal structlog fields
def filter_internal_keys(logger: Any, name: str, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Remove internal structlog fields from the event dictionary.

    Args:
        logger: Structlog logger instance (unused by this processor).
        name: Logger name (unused by this processor).
        event_dict: Mutable structlog event payload.

    Returns:
        The same mapping instance, with any keys starting with `_` removed.

    Notes:
        This function mutates `event_dict` in place.
    """
    keys_to_remove = [k for k in event_dict.keys() if k.startswith("_")]
    for key in keys_to_remove:
        event_dict.pop(key, None)
    return event_dict


# Simple human-readable formatter for structlog (with Rich markup for console)
def simple_log_format_rich(logger: Any, name: str, event_dict: MutableMapping[str, Any]) -> str:
    """Format a structlog event as a Rich-marked-up single line.

    Args:
        logger: Structlog logger instance (unused by this processor).
        name: Logger name (unused by this processor).
        event_dict: Mutable structlog event payload.

    Returns:
        A formatted log line string.

    Notes:
        This function mutates `event_dict` by popping structlog-internal keys and the
        normalized fields it renders.
    """
    # Remove internal structlog metadata
    event_dict.pop("_record", None)
    event_dict.pop("_from_structlog", None)

    level = event_dict.pop("level", "INFO")
    timestamp = event_dict.pop("timestamp", "")
    logger_name = event_dict.pop("logger", "")
    event = event_dict.pop("event", "")

    # Format as simple human-readable line
    parts = []
    if timestamp:
        parts.append(f"{timestamp}")
    if logger_name:
        # Shorten logger names for readability
        short_name = logger_name.split(".")[-1] if "." in logger_name else logger_name
        parts.append(f"[cyan]{short_name}[/cyan]")

    # Add level with color coding for Rich handler
    level_upper = level.upper()
    if level_upper == "ERROR" or level_upper == "CRITICAL":
        parts.append(f"[red]{level_upper}[/red]")
    elif level_upper == "WARNING":
        parts.append(f"[yellow]{level_upper}[/yellow]")
    elif level_upper == "INFO":
        parts.append(f"[green]{level_upper}[/green]")
    else:
        parts.append(level_upper)

    # Add the main event message
    parts.append(f"[bold]{event}[/bold]" if event else "")

    # Add any remaining key-value pairs as context (more compact format)
    if event_dict:
        context_parts = []
        for key, value in event_dict.items():
            # Skip internal keys
            if key.startswith("_"):
                continue
            # Format value nicely
            if isinstance(value, str) and len(value) > 200:
                value_str = f"{value[:197]}..."
            else:
                value_str = str(value)
            context_parts.append(f"[dim]{key}[/dim]={value_str}")
        if context_parts:
            parts.append(f"({', '.join(context_parts)})")

    return " ".join(parts)


# Simple human-readable formatter for structlog (plain text for files)
def simple_log_format_plain(logger: Any, name: str, event_dict: MutableMapping[str, Any]) -> str:
    """Format a structlog event as a plain-text single line.

    Args:
        logger: Structlog logger instance (unused by this processor).
        name: Logger name (unused by this processor).
        event_dict: Mutable structlog event payload.

    Returns:
        A formatted log line string.

    Notes:
        This function mutates `event_dict` by popping structlog-internal keys and the
        normalized fields it renders.
    """
    # Remove internal structlog metadata
    event_dict.pop("_record", None)
    event_dict.pop("_from_structlog", None)

    level = event_dict.pop("level", "INFO")
    timestamp = event_dict.pop("timestamp", "")
    logger_name = event_dict.pop("logger", "")
    event = event_dict.pop("event", "")

    # Format as simple human-readable line
    parts = []
    if timestamp:
        parts.append(f"{timestamp}")
    if logger_name:
        # Shorten logger names for readability
        short_name = logger_name.split(".")[-1] if "." in logger_name else logger_name
        parts.append(f"[{short_name}]")

    # Add level without color coding
    parts.append(level.upper())

    # Add the main event message
    parts.append(event if event else "")

    # Add any remaining key-value pairs as context (more compact format)
    if event_dict:
        context_parts = []
        for key, value in event_dict.items():
            # Skip internal keys
            if key.startswith("_"):
                continue
            # Format value nicely
            if isinstance(value, str) and len(value) > 200:
                value_str = f"{value[:197]}..."
            else:
                value_str = str(value)
            context_parts.append(f"{key}={value_str}")
        if context_parts:
            parts.append(f"({', '.join(context_parts)})")

    return " ".join(parts)


formatter = structlog.stdlib.ProcessorFormatter(
    foreign_pre_chain=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
    ],
    processors=[
        structlog.dev.ConsoleRenderer(
            colors=False,
            exception_formatter=structlog.dev.plain_traceback,
            sort_keys=False,
        )
    ],
)

# Formatter for file output (plain text, no Rich markup)
simple_formatter = structlog.stdlib.ProcessorFormatter(
    foreign_pre_chain=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%m/%d/%Y, %H:%M:%S"),
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
    ],
    processors=[
        filter_internal_keys,  # Remove internal fields first
        simple_log_format_plain,  # Then format for display (no markup)
    ],
)

# Formatter for Rich console output (with color markup)
rich_formatter = structlog.stdlib.ProcessorFormatter(
    foreign_pre_chain=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%m/%d/%Y, %H:%M:%S"),
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
    ],
    processors=[
        filter_internal_keys,  # Remove internal fields first
        simple_log_format_rich,  # Then format with Rich markup
    ],
)

handler: stdlib_logging.Handler = stdlib_logging.StreamHandler()
if settings.LOG_FILE:
    handler = stdlib_logging.FileHandler(os.path.join(settings.BASE_OUTPUT_DIR, settings.LOG_FILE))


handler.setFormatter(simple_formatter)
root_logger = stdlib_logging.getLogger()
root_logger.addHandler(handler)
root_logger.setLevel(settings.LOG_LEVEL_STR)
