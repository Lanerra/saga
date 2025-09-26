# config/settings.py
"""
Configuration settings for the Saga Novel Generation system.
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
            "Unexpected error loading file.", file_path=file_path, exc_info=True
        )
        return default_if_missing


class SagaSettings(BaseSettings):
    """Full configuration for the Saga system."""

    # API and Model Configuration
    EMBEDDING_API_BASE: str = "http://127.0.0.1:11434"
    EMBEDDING_API_KEY: str = ""
    OPENAI_API_BASE: str = "http://127.0.0.1:8080/v1"
    OPENAI_API_KEY: str = "nope"

    EMBEDDING_MODEL: str = "mxbai-embed-large:latest"
    EXPECTED_EMBEDDING_DIM: int = 1024
    EMBEDDING_DTYPE: str = "float16"

    # Neo4j Connection Settings
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "saga_password"
    NEO4J_DATABASE: str | None = "neo4j"

    # Neo4j Vector Index Configuration
    NEO4J_VECTOR_INDEX_NAME: str = "chapterEmbeddings"
    NEO4J_VECTOR_NODE_LABEL: str = "Chapter"
    NEO4J_VECTOR_PROPERTY_NAME: str = "embedding_vector"
    NEO4J_VECTOR_DIMENSIONS: int = 1024
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
    LLM_TOP_P: float = 0.8

    # LLM Frequency and Presence Penalties
    FREQUENCY_PENALTY_DRAFTING: float = 0.3
    PRESENCE_PENALTY_DRAFTING: float = 1.0
    FREQUENCY_PENALTY_REVISION: float = 0.2
    PRESENCE_PENALTY_REVISION: float = 1.0
    FREQUENCY_PENALTY_PATCH: float = 0.2
    PRESENCE_PENALTY_PATCH: float = 1.0
    FREQUENCY_PENALTY_PLANNING: float = 0.0
    PRESENCE_PENALTY_PLANNING: float = 1.0
    FREQUENCY_PENALTY_INITIAL_SETUP: float = 0.1
    PRESENCE_PENALTY_INITIAL_SETUP: float = 1.0
    FREQUENCY_PENALTY_EVALUATION: float = 0.0
    PRESENCE_PENALTY_EVALUATION: float = 1.0
    FREQUENCY_PENALTY_KG_EXTRACTION: float = 0.0
    PRESENCE_PENALTY_KG_EXTRACTION: float = 1.0
    FREQUENCY_PENALTY_SUMMARY: float = 0.0
    PRESENCE_PENALTY_SUMMARY: float = 1.0
    FREQUENCY_PENALTY_CONSISTENCY_CHECK: float = 0.0
    PRESENCE_PENALTY_CONSISTENCY_CHECK: float = 1.0

    # Output and File Paths
    BASE_OUTPUT_DIR: str = "output"
    PLOT_OUTLINE_FILE: str = "plot_outline.json"
    CHARACTER_PROFILES_FILE: str = "character_profiles.json"
    WORLD_BUILDER_FILE: str = "world_building.json"
    CHAPTERS_DIR: str = "chapters"
    CHAPTER_LOGS_DIR: str = "chapter_logs"
    DEBUG_OUTPUTS_DIR: str = "debug_outputs"

    USER_STORY_ELEMENTS_FILE_PATH: str = "user_story_elements.yaml"

    # Generation Parameters
    # Token budgets (defaults are generous; override via FAST_PROFILE for laptops)
    MAX_CONTEXT_TOKENS: int = 40960
    MAX_GENERATION_TOKENS: int = 16384
    CONTEXT_CHAPTER_COUNT: int = 3
    CHAPTERS_PER_RUN: int = 2
    KG_HEALING_INTERVAL: int = 2
    TARGET_PLOT_POINTS_INITIAL_GENERATION: int = 20
    MAX_CONCURRENT_CHAPTERS: int = 1

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
    PLANNING_CONTEXT_MAX_WORLD_ITEMS_PER_CATEGORY: int = 3
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
    MAX_REVISION_CYCLES_PER_CHAPTER: int = 0
    MAX_SUMMARY_TOKENS: int = 8192
    MAX_KG_TRIPLE_TOKENS: int = 8192
    MAX_PREPOP_KG_TOKENS: int = 16384

    MIN_ACCEPTABLE_DRAFT_LENGTH: int = 12000

    # Narrative Agent Configuration
    NARRATIVE_CONTEXT_SUMMARY_MAX_CHARS: int = 1000
    NARRATIVE_CONTEXT_TEXT_TAIL_CHARS: int = 1000
    NARRATIVE_TOKEN_BUFFER: int = 200
    NARRATIVE_JSON_DEBUG_SAVE: bool = True

    ENABLE_DYNAMIC_STATE_ADAPTATION: bool = True
    KG_PREPOPULATION_CHAPTER_NUM: int = 0

    # De-duplication Configuration
    DEDUPLICATION_USE_SEMANTIC: bool = False
    DEDUPLICATION_SEMANTIC_THRESHOLD: float = 0.45
    DEDUPLICATION_MIN_SEGMENT_LENGTH: int = 150

    # Relationship Constraint Configuration
    ENABLE_RELATIONSHIP_CONSTRAINTS: bool = True
    RELATIONSHIP_CONSTRAINT_MIN_CONFIDENCE: float = 0.3
    RELATIONSHIP_CONSTRAINT_STRICT_MODE: bool = False
    RELATIONSHIP_CONSTRAINT_LOG_VIOLATIONS: bool = True
    RELATIONSHIP_CONSTRAINT_AUTO_CORRECT: bool = False
    DISABLE_RELATIONSHIP_SEMANTIC_FLATTENING: bool = True

    # Enhanced Node Type Configuration
    ENABLE_ENHANCED_NODE_TYPES: bool = True

    # Logging & UI
    LOG_LEVEL_STR: str = Field("INFO", alias="LOG_LEVEL")
    LOG_FORMAT: str = (
        "%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
    )
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_FILE: str | None = "saga_run.log"
    ENABLE_RICH_PROGRESS: bool = True
    # Minimal logging mode for single-user setups: console only, no rotation/Rich
    SIMPLE_LOGGING_MODE: bool = False

    # NLP / spaCy configuration
    SPACY_MODEL: str | None = (
        None  # default None => utils.text_processing uses en_core_web_sm
    )

    # Novel Configuration (Defaults / Placeholders)
    CONFIGURED_GENRE: str = "grimdark science fiction"
    CONFIGURED_THEME: str = "the hubris of humanity"
    CONFIGURED_SETTING_DESCRIPTION: str = (
        "a remote outpost on the surface of Jupiter's moon, Callisto"
    )
    DEFAULT_PROTAGONIST_NAME: str = "Ilya Lakatos"
    DEFAULT_PLOT_OUTLINE_TITLE: str = "Untitled Narrative"

    MAIN_NOVEL_INFO_NODE_ID: str = "main_novel_info"

    # Identifier for the root World Container node in the Neo4j graph.
    # This constant is used throughout the codebase for bootstrapping and
    # querying world‑level structures.  It was previously defined in the
    # legacy ``config.py`` file; adding it here restores compatibility.
    MAIN_WORLD_CONTAINER_NODE_ID: str = "world_container"

    DISABLE_RELATIONSHIP_NORMALIZATION: bool = True

    # Bootstrap Enhancement Configuration
    BOOTSTRAP_CREATE_RELATIONSHIPS: bool = True
    BOOTSTRAP_USE_ENHANCED_NODE_TYPES: bool = True
    BOOTSTRAP_MIN_CHARACTERS: int = 3
    BOOTSTRAP_MIN_WORLD_ELEMENTS: int = 4
    BOOTSTRAP_RELATIONSHIP_COUNT_TARGET: int = 8
    BOOTSTRAP_USE_VALIDATION: bool = True
    # Super-charged bootstrap toggles
    BOOTSTRAP_ENABLED_DEFAULT: bool = False
    BOOTSTRAP_HIGHER_SETTING: str = "enhanced"  # basic|enhanced|max
    BOOTSTRAP_VALIDATE_EACH_PHASE: bool = True
    BOOTSTRAP_PUSH_TO_KG_EACH_PHASE: bool = True
    BOOTSTRAP_RUN_KG_HEAL: bool = True
    BOOTSTRAP_FAIL_FAST: bool = True

    # Enhanced character bootstrap settings
    BOOTSTRAP_MIN_TRAITS_PROTAGONIST: int = 6
    BOOTSTRAP_MIN_TRAITS_ANTAGONIST: int = 5
    BOOTSTRAP_MIN_TRAITS_SUPPORTING: int = 4

    # Dynamic Schema System Configuration (Disabled for single-user deployment)
    ENABLE_DYNAMIC_SCHEMA: bool = False
    DYNAMIC_SCHEMA_AUTO_REFRESH: bool = False
    DYNAMIC_SCHEMA_CACHE_TTL_MINUTES: int = 2
    DYNAMIC_SCHEMA_LEARNING_ENABLED: bool = False
    DYNAMIC_SCHEMA_FALLBACK_ENABLED: bool = False

    # Type Inference Configuration
    DYNAMIC_TYPE_INFERENCE_CONFIDENCE_THRESHOLD: float = 0.6
    DYNAMIC_TYPE_PATTERN_MIN_FREQUENCY: int = 3

    # Constraint System Configuration
    DYNAMIC_CONSTRAINT_CONFIDENCE_THRESHOLD: float = 0.6
    DYNAMIC_CONSTRAINT_MIN_SAMPLES: int = 3

    # Schema Discovery Configuration
    SCHEMA_INTROSPECTION_CACHE_TTL_MINUTES: int = 2

    @model_validator(mode="after")
    def set_dynamic_model_defaults(self) -> SagaSettings:
        # Optional FAST profile for consumer laptops: lower budgets to avoid timeouts.
        # Activate with FAST_PROFILE=true (case-insensitive).
        fast = os.getenv("FAST_PROFILE", "false").lower() in {"1", "true", "yes", "on"}
        if fast:
            object.__setattr__(
                self, "MAX_CONTEXT_TOKENS", min(self.MAX_CONTEXT_TOKENS, 8192)
            )
            object.__setattr__(
                self, "MAX_GENERATION_TOKENS", min(self.MAX_GENERATION_TOKENS, 2048)
            )
            object.__setattr__(
                self,
                "MIN_ACCEPTABLE_DRAFT_LENGTH",
                min(self.MIN_ACCEPTABLE_DRAFT_LENGTH, 3500),
            )
            # Slightly reduce planning and patch windows to match
            object.__setattr__(
                self,
                "MAX_PLANNING_TOKENS",
                min(getattr(self, "MAX_PLANNING_TOKENS", 16384), 8192),
            )
            object.__setattr__(
                self,
                "MAX_KG_TRIPLE_TOKENS",
                min(getattr(self, "MAX_KG_TRIPLE_TOKENS", 8192), 4096),
            )
            object.__setattr__(
                self,
                "MAX_PREPOP_KG_TOKENS",
                min(getattr(self, "MAX_PREPOP_KG_TOKENS", 16384), 8192),
            )
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

# Configure structlog to integrate with standard logging and output human‑readable messages
structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%m/%d/%Y, %H:%M"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Simple human‑readable formatter for structlog
formatter = structlog.stdlib.ProcessorFormatter(
    foreign_pre_chain=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
    ],
    processors=[structlog.dev.ConsoleRenderer(colors=False)],
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


# Duplicate Prevention Settings
ENABLE_DUPLICATE_PREVENTION: bool = True  # Enable proactive duplicate prevention
DUPLICATE_PREVENTION_SIMILARITY_THRESHOLD: float = (
    0.3  # Similarity threshold for merging entities
)
DUPLICATE_PREVENTION_CHARACTER_ENABLED: bool = (
    True  # Enable character duplicate prevention
)
DUPLICATE_PREVENTION_WORLD_ITEM_ENABLED: bool = (
    True  # Enable world item duplicate prevention
)

# State Tracker Configuration
STATE_TRACKER_ENABLED: bool = True  # Enable StateTracker for bootstrap generation
STATE_TRACKER_SIMILARITY_THRESHOLD: float = (
    0.75  # Threshold for description similarity checks
)

# Legacy Degradation Flags (non-breaking defaults)
# Legacy WorldElement toggle removed; single typed-entity model is standard
ENABLE_LEGACY_WORLDELEMENT: bool = False  # Deprecated/no-op
ENABLE_STATUS_IS_ALIAS: bool = True
