# config.py
"""Configuration settings for the Saga Novel Generation system.
Uses Pydantic BaseSettings for automatic environment variable loading.
"""

from __future__ import annotations

import json
import logging as stdlib_logging
import os

import structlog
from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

logger = structlog.get_logger()


async def _load_list_from_json_async(
    file_path: str, default_if_missing: list[str] | None = None
) -> list[str]:
    """Load a list of strings from a JSON file asynchronously."""
    if default_if_missing is None:
        default_if_missing = []
    try:
        if os.path.exists(file_path):
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and all(
                    isinstance(item, str) for item in data
                ):
                    return data
                logger.warning(
                    "Content of file is not a list of strings. Using default.",
                    file_path=file_path,
                )
                return default_if_missing
        logger.warning(
            "Configuration file not found. Using default.", file_path=file_path
        )
        return default_if_missing
    except json.JSONDecodeError:
        logger.error(
            "Error decoding JSON from file. Using default.",
            file_path=file_path,
            exc_info=True,
        )
        return default_if_missing
    except Exception:
        logger.error(
            "Unexpected error loading file. Using default.",
            file_path=file_path,
            exc_info=True,
        )
        return default_if_missing


class SagaSettings(BaseSettings):
    """Full configuration for the Saga system."""

    # API and Model Configuration
    OLLAMA_EMBED_URL: str = "http://127.0.0.1:11434"
    OPENAI_API_BASE: str = "http://127.0.0.1:8080/v1"
    OPENAI_API_KEY: str = "nope"

    EMBEDDING_MODEL: str = "nomic-embed-text:latest"
    # Reranker model needs to be loaded in Ollama and support the /api/rerank endpoint.
    # E.g., bge-reranker-base, mxbai-rerank-large-v1, etc.
    RERANKER_MODEL: str = "mxbai-rerank-large-v1:latest"
    EXPECTED_EMBEDDING_DIM: int = 768
    EMBEDDING_DTYPE: str = "float32"

    # Neo4j Connection Settings
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "saga_password"
    NEO4J_DATABASE: str | None = "neo4j"

    # Neo4j Vector Index Configuration
    NEO4J_VECTOR_INDEX_NAME: str = "chapterEmbeddings"
    NEO4J_VECTOR_NODE_LABEL: str = "Chapter"
    NEO4J_VECTOR_PROPERTY_NAME: str = "embedding_vector"
    NEO4J_VECTOR_DIMENSIONS: int = 768
    NEO4J_VECTOR_SIMILARITY_FUNCTION: str = "cosine"

    # Base Model Definitions
    LARGE_MODEL: str = "qwen3-a3b"
    MEDIUM_MODEL: str = "qwen3-a3b"
    SMALL_MODEL: str = "qwen3-a3b"
    NARRATIVE_MODEL: str = "qwen3-a3b"

    # Temperature Settings
    TEMPERATURE_INITIAL_SETUP: float = 0.8
    TEMPERATURE_DRAFTING: float = 0.8
    TEMPERATURE_REVISION: float = 0.65
    TEMPERATURE_PLANNING: float = 0.6
    TEMPERATURE_EVALUATION: float = 0.3
    TEMPERATURE_CONSISTENCY_CHECK: float = 0.2
    TEMPERATURE_KG_EXTRACTION: float = 0.4
    TEMPERATURE_SUMMARY: float = 0.5
    TEMPERATURE_PATCH: float = 0.7

    # Placeholder fill-in
    FILL_IN: str = ""

    # LLM Call Settings & Fallbacks
    LLM_RETRY_ATTEMPTS: int = 3
    LLM_RETRY_DELAY_SECONDS: float = 3.0
    HTTPX_TIMEOUT: float = 600.0
    ENABLE_LLM_NO_THINK_DIRECTIVE: bool = True
    TIKTOKEN_DEFAULT_ENCODING: str = "cl100k_base"
    FALLBACK_CHARS_PER_TOKEN: float = 4.0
    # Concurrency and Rate Limiting
    MAX_CONCURRENT_LLM_CALLS: int = 4

    # Dynamic Model Assignments (set from base models if not specified in env)
    FALLBACK_GENERATION_MODEL: str | None = None
    MAIN_GENERATION_MODEL: str | None = None
    KNOWLEDGE_UPDATE_MODEL: str | None = None
    INITIAL_SETUP_MODEL: str | None = None
    PLANNING_MODEL: str | None = None
    DRAFTING_MODEL: str | None = None
    NARRATIVE_MODEL: str | None = None
    REVISION_MODEL: str | None = None
    EVALUATION_MODEL: str | None = None
    PATCH_GENERATION_MODEL: str | None = None

    LLM_TOP_P: float = 0.8

    # LLM Frequency and Presence Penalties
    FREQUENCY_PENALTY_DRAFTING: float = 0.3
    PRESENCE_PENALTY_DRAFTING: float = 1.5
    FREQUENCY_PENALTY_REVISION: float = 0.2
    PRESENCE_PENALTY_REVISION: float = 1.5
    FREQUENCY_PENALTY_PATCH: float = 0.2
    PRESENCE_PENALTY_PATCH: float = 1.5
    FREQUENCY_PENALTY_PLANNING: float = 0.0
    PRESENCE_PENALTY_PLANNING: float = 1.5
    FREQUENCY_PENALTY_INITIAL_SETUP: float = 0.1
    PRESENCE_PENALTY_INITIAL_SETUP: float = 1.5
    FREQUENCY_PENALTY_EVALUATION: float = 0.0
    PRESENCE_PENALTY_EVALUATION: float = 1.5
    FREQUENCY_PENALTY_KG_EXTRACTION: float = 0.0
    PRESENCE_PENALTY_KG_EXTRACTION: float = 1.5
    FREQUENCY_PENALTY_SUMMARY: float = 0.0
    PRESENCE_PENALTY_SUMMARY: float = 1.5
    FREQUENCY_PENALTY_CONSISTENCY_CHECK: float = 0.0
    PRESENCE_PENALTY_CONSISTENCY_CHECK: float = 1.5

    # Output and File Paths
    BASE_OUTPUT_DIR: str = "novel_output"
    PLOT_OUTLINE_FILE: str = "plot_outline.json"
    CHARACTER_PROFILES_FILE: str = "character_profiles.json"
    WORLD_BUILDER_FILE: str = "world_building.json"
    CHAPTERS_DIR: str = "chapters"
    CHAPTER_LOGS_DIR: str = "chapter_logs"
    DEBUG_OUTPUTS_DIR: str = "debug_outputs"

    USER_STORY_ELEMENTS_FILE_PATH: str = "user_story_elements.yaml"


    # Generation Parameters
    MAX_CONTEXT_TOKENS: int = 40960
    MAX_GENERATION_TOKENS: int = 16384
    CONTEXT_CHAPTER_COUNT: int = 3
    CHAPTERS_PER_RUN: int = 4
    KG_HEALING_INTERVAL: int = 2
    TARGET_PLOT_POINTS_INITIAL_GENERATION: int = 20
    # Concurrency limiting for chapter processing to prevent resource exhaustion
    MAX_CONCURRENT_CHAPTERS: int = 4

    # Caching
    EMBEDDING_CACHE_SIZE: int = 128
    SUMMARY_CACHE_SIZE: int = 32
    KG_TRIPLE_EXTRACTION_CACHE_SIZE: int = 16
    TOKENIZER_CACHE_SIZE: int = 10

    # Reranking Configuration
    ENABLE_RERANKING: bool = False
    RERANKER_CANDIDATE_COUNT: int = 15

    # Agentic Planning & Prompt Context Snippets
    ENABLE_AGENTIC_PLANNING: bool = True
    MAX_PLANNING_TOKENS: int = 16384
    TARGET_SCENES_MIN: int = 4
    TARGET_SCENES_MAX: int = 6
    PLANNING_CONTEXT_MAX_CHARS_PER_PROFILE_DESC: int = 80
    PLANNING_CONTEXT_MAX_RECENT_DEV_PER_PROFILE: int = 120
    PLANNING_CONTEXT_MAX_CHARACTERS_IN_SNIPPET: int = 5
    PLANNING_CONTEXT_MAX_LOCATIONS_IN_SNIPPET: int = 3
    PLANNING_CONTEXT_MAX_FACTIONS_IN_SNIPPET: int = 2
    PLANNING_CONTEXT_MAX_SYSTEMS_IN_SNIPPET: int = 2

    # Revision and Validation
    ENABLE_COMPREHENSIVE_EVALUATION: bool = True
    ENABLE_WORLD_CONTINUITY_CHECK: bool = True
    ENABLE_SCENE_PLAN_VALIDATION: bool = True
    ENABLE_PATCH_BASED_REVISION: bool = True
    AGENT_ENABLE_PATCH_VALIDATION: bool = True
    MAX_PATCH_INSTRUCTIONS_TO_GENERATE: int = 5
    PATCH_GENERATION_ATTEMPTS: int = 1
    MAX_CHARS_FOR_PATCH_CONTEXT_WINDOW: int = 16384
    PATCH_VALIDATION_THRESHOLD: int = 70
    REVISION_COHERENCE_THRESHOLD: float = 0.60
    REVISION_SIMILARITY_ACCEPTANCE: float = 0.995
    POST_PATCH_PROBLEM_THRESHOLD: int = 0
    MAX_REVISION_CYCLES_PER_CHAPTER: int = 2
    MAX_SUMMARY_TOKENS: int = 4096
    MAX_KG_TRIPLE_TOKENS: int = 8192
    MAX_PREPOP_KG_TOKENS: int = 16384

    MIN_ACCEPTABLE_DRAFT_LENGTH: int = 12000

    ENABLE_DYNAMIC_STATE_ADAPTATION: bool = True
    KG_PREPOPULATION_CHAPTER_NUM: int = 0

    # De-duplication Configuration
    DEDUPLICATION_USE_SEMANTIC: bool = False
    DEDUPLICATION_SEMANTIC_THRESHOLD: float = 0.85
    DEDUPLICATION_MIN_SEGMENT_LENGTH: int = 150

    # Relationship Constraint Configuration
    ENABLE_RELATIONSHIP_CONSTRAINTS: bool = True
    RELATIONSHIP_CONSTRAINT_MIN_CONFIDENCE: float = (
        0.3  # Lower threshold to accept more corrections
    )
    RELATIONSHIP_CONSTRAINT_STRICT_MODE: bool = (
        False  # If True, rejects invalid relationships; if False, uses fallbacks
    )
    RELATIONSHIP_CONSTRAINT_LOG_VIOLATIONS: bool = True
    RELATIONSHIP_CONSTRAINT_AUTO_CORRECT: bool = (
        False  # Allow automatic corrections of relationship types
    )
    DISABLE_RELATIONSHIP_SEMANTIC_FLATTENING: bool = (
        True  # If True, preserves original relationship types without fallbacks
    )

    # Enhanced Node Type Configuration
    ENABLE_ENHANCED_NODE_TYPES: bool = (
        True  # Use enhanced specific node types instead of generic ones
    )

    # Logging & UI
    LOG_LEVEL_STR: str = Field("INFO", alias="LOG_LEVEL")
    LOG_FORMAT: str = (
        "%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
    )
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_FILE: str | None = "saga_run.log"
    ENABLE_RICH_PROGRESS: bool = True

    # Novel Configuration (Defaults / Placeholders)
    CONFIGURED_GENRE: str = "grimdark science fiction"
    CONFIGURED_THEME: str = "the hubris of humanity"
    CONFIGURED_SETTING_DESCRIPTION: str = (
        "a remote outpost on the surface of Jupiter's moon, Callisto"
    )
    DEFAULT_PROTAGONIST_NAME: str = "Ilya Lakatos"
    DEFAULT_PLOT_OUTLINE_TITLE: str = "Untitled Narrative"

    MAIN_NOVEL_INFO_NODE_ID: str = "main_novel_info"
    MAIN_CHARACTERS_CONTAINER_NODE_ID: str = "main_characters_container"
    MAIN_WORLD_CONTAINER_NODE_ID: str = "main_world_container"

    DISABLE_RELATIONSHIP_NORMALIZATION: bool = (
        True  # Toggle relationship normalization for testing
    )

    # Bootstrap Enhancement Configuration
    BOOTSTRAP_CREATE_RELATIONSHIPS: bool = True
    BOOTSTRAP_USE_ENHANCED_NODE_TYPES: bool = True
    BOOTSTRAP_MIN_CHARACTERS: int = 3
    BOOTSTRAP_MIN_WORLD_ELEMENTS: int = 4
    BOOTSTRAP_RELATIONSHIP_COUNT_TARGET: int = 8
    BOOTSTRAP_USE_VALIDATION: bool = True
    
    # Enhanced character bootstrap settings
    BOOTSTRAP_MIN_TRAITS_PROTAGONIST: int = 6
    BOOTSTRAP_MIN_TRAITS_ANTAGONIST: int = 5
    BOOTSTRAP_MIN_TRAITS_SUPPORTING: int = 4
    
    # Dynamic Schema System Configuration
    ENABLE_DYNAMIC_SCHEMA: bool = True                   # Master switch for dynamic schema system
    DYNAMIC_SCHEMA_AUTO_REFRESH: bool = True             # Auto-refresh schema data when stale  
    DYNAMIC_SCHEMA_CACHE_TTL_MINUTES: int = 2            # Cache time-to-live for schema data
    DYNAMIC_SCHEMA_LEARNING_ENABLED: bool = True         # Enable learning from existing data
    DYNAMIC_SCHEMA_FALLBACK_ENABLED: bool = False         # Fall back to static methods on failure
    
    # Type Inference Configuration
    DYNAMIC_TYPE_INFERENCE_CONFIDENCE_THRESHOLD: float = 0.3   # Min confidence for dynamic inference
    DYNAMIC_TYPE_PATTERN_MIN_FREQUENCY: int = 3                # Min frequency for patterns to be retained
    
    # Constraint System Configuration  
    DYNAMIC_CONSTRAINT_CONFIDENCE_THRESHOLD: float = 0.3       # Min confidence for constraint validation
    DYNAMIC_CONSTRAINT_MIN_SAMPLES: int = 3                    # Min samples to learn a constraint
    
    # Schema Discovery Configuration
    SCHEMA_INTROSPECTION_CACHE_TTL_MINUTES: int = 2           # Cache TTL for introspection queries

    @model_validator(mode="after")
    def set_dynamic_model_defaults(self) -> SagaSettings:
        if self.FALLBACK_GENERATION_MODEL is None:
            self.FALLBACK_GENERATION_MODEL = self.MEDIUM_MODEL
        if self.MAIN_GENERATION_MODEL is None:
            self.MAIN_GENERATION_MODEL = self.NARRATIVE_MODEL
        if self.KNOWLEDGE_UPDATE_MODEL is None:
            self.KNOWLEDGE_UPDATE_MODEL = self.MEDIUM_MODEL
        if self.INITIAL_SETUP_MODEL is None:
            self.INITIAL_SETUP_MODEL = self.MEDIUM_MODEL
        if self.PLANNING_MODEL is None:
            self.PLANNING_MODEL = self.LARGE_MODEL
        if self.DRAFTING_MODEL is None:
            self.DRAFTING_MODEL = self.NARRATIVE_MODEL
        if self.REVISION_MODEL is None:
            self.REVISION_MODEL = self.NARRATIVE_MODEL
        if self.EVALUATION_MODEL is None:
            self.EVALUATION_MODEL = self.LARGE_MODEL
        if self.PATCH_GENERATION_MODEL is None:
            self.PATCH_GENERATION_MODEL = self.MEDIUM_MODEL
        return self

    model_config = SettingsConfigDict(env_prefix="", env_file=".env")


settings = SagaSettings()


# --- Reconstruct objects for backward compatibility ---
class ModelsCompat:
    pass


class TempsCompat:
    pass


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
Temperatures.DEFAULT = 0.6  # Set default explicitly


# Update module level variables for backward compatibility
for _field in settings.model_fields:
    globals()[_field] = getattr(settings, _field)


PLOT_OUTLINE_FILE = os.path.join(settings.BASE_OUTPUT_DIR, settings.PLOT_OUTLINE_FILE)
CHARACTER_PROFILES_FILE = os.path.join(
    settings.BASE_OUTPUT_DIR, settings.CHARACTER_PROFILES_FILE
)
WORLD_BUILDER_FILE = os.path.join(settings.BASE_OUTPUT_DIR, settings.WORLD_BUILDER_FILE)
CHAPTERS_DIR = os.path.join(settings.BASE_OUTPUT_DIR, settings.CHAPTERS_DIR)
CHAPTER_LOGS_DIR = os.path.join(settings.BASE_OUTPUT_DIR, settings.CHAPTER_LOGS_DIR)
DEBUG_OUTPUTS_DIR = os.path.join(settings.BASE_OUTPUT_DIR, settings.DEBUG_OUTPUTS_DIR)

# Ensure output directories exist
os.makedirs(settings.BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(CHAPTERS_DIR, exist_ok=True)
os.makedirs(CHAPTER_LOGS_DIR, exist_ok=True)
os.makedirs(DEBUG_OUTPUTS_DIR, exist_ok=True)

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

formatter = structlog.stdlib.ProcessorFormatter(
    foreign_pre_chain=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
    ],
    processors=[structlog.dev.ConsoleRenderer()],
)

handler = stdlib_logging.StreamHandler()
if settings.LOG_FILE:
    handler = stdlib_logging.FileHandler(
        os.path.join(settings.BASE_OUTPUT_DIR, settings.LOG_FILE)
    )
handler.setFormatter(formatter)
root_logger = stdlib_logging.getLogger()
root_logger.addHandler(handler)
root_logger.setLevel(settings.LOG_LEVEL_STR)

REVISION_EVALUATION_THRESHOLD = 0.85

# Bootstrap Integration Settings (Phase 1: Knowledge Graph Integration Strategy)
BOOTSTRAP_INTEGRATION_ENABLED: bool = False
BOOTSTRAP_INTEGRATION_CHAPTERS: int = 0
MAX_BOOTSTRAP_ELEMENTS_PER_CONTEXT: int = 0  # Limit to prevent prompt bloat
BOOTSTRAP_HEALING_LIMIT: int = 0

# Context Selection Settings (Phase 1.1: Balanced Context Selection)
EARLY_CHAPTER_BALANCED_SELECTION: bool = False  # Use balanced char selection
PROTAGONIST_PRIORITY_START_CHAPTER: int = (
    3  # When to start protagonist-priority selection
)

# Duplicate Prevention Settings
ENABLE_DUPLICATE_PREVENTION: bool = True  # Enable proactive duplicate prevention
DUPLICATE_PREVENTION_SIMILARITY_THRESHOLD: float = (
    0.65  # Similarity threshold for merging entities
)
DUPLICATE_PREVENTION_CHARACTER_ENABLED: bool = (
    True  # Enable character duplicate prevention
)
DUPLICATE_PREVENTION_WORLD_ITEM_ENABLED: bool = (
    True  # Enable world item duplicate prevention
)
