# config/__init__.py
"""Expose SAGA configuration as stable module-level constants.

This package provides a backwards-compatible facade over the underlying Pydantic settings
model defined in [`config.settings`](config/settings.py:1). The primary API is the
[`settings`](config/settings.py:356) singleton plus a set of module-level constants
mirroring its fields.

Configuration precedence and lifecycle:
- On initial import, configuration is loaded by importing [`config.settings`](config/settings.py:1),
  which constructs the `settings` singleton.
- Values come from the process environment and may be sourced from a `.env` file (see
  [`config.settings`](config/settings.py:1) for import-time side effects).
- [`reload()`](config/__init__.py:170) triggers a refresh that re-reads `.env` with override enabled,
  then replaces this module's exported values (see [`config.loader.reload_settings()`](config/loader.py:35)).

Notes:
    This module intentionally duplicates values into module globals for legacy callers
    (e.g., `config.OPENAI_API_KEY`). New code should prefer the `settings` object.
"""

# Explicit exports for MyPy compatibility
from typing import Any

from . import settings as settings_mod
from .settings import (
    Models as Models,
)
from .settings import (
    Temperatures as Temperatures,
)
from .settings import (
    rich_formatter as rich_formatter,
)
from .settings import (
    settings as settings,
)
from .settings import (
    simple_formatter as simple_formatter,
)

BASE_OUTPUT_DIR = settings.BASE_OUTPUT_DIR
BOOTSTRAP_MIN_TRAITS_ANTAGONIST = settings.BOOTSTRAP_MIN_TRAITS_ANTAGONIST
BOOTSTRAP_MIN_TRAITS_PROTAGONIST = settings.BOOTSTRAP_MIN_TRAITS_PROTAGONIST
BOOTSTRAP_MIN_TRAITS_SUPPORTING = settings.BOOTSTRAP_MIN_TRAITS_SUPPORTING
CHAPTERS_DIR = settings_mod.CHAPTERS_DIR
CHAPTERS_PER_RUN = settings.CHAPTERS_PER_RUN
TOTAL_CHAPTERS = settings.TOTAL_CHAPTERS
CHAPTER_LOGS_DIR = settings_mod.CHAPTER_LOGS_DIR
CHARACTER_PROFILES_FILE = settings_mod.CHARACTER_PROFILES_FILE
CONFIGURED_GENRE = settings.CONFIGURED_GENRE
CONFIGURED_SETTING_DESCRIPTION = settings.CONFIGURED_SETTING_DESCRIPTION
CONFIGURED_THEME = settings.CONFIGURED_THEME
CONTEXT_CHAPTER_COUNT = settings.CONTEXT_CHAPTER_COUNT
DEDUPLICATION_MIN_SEGMENT_LENGTH = settings.DEDUPLICATION_MIN_SEGMENT_LENGTH
DEDUPLICATION_SEMANTIC_THRESHOLD = settings.DEDUPLICATION_SEMANTIC_THRESHOLD
DEDUPLICATION_USE_SEMANTIC = settings.DEDUPLICATION_USE_SEMANTIC
DEFAULT_PLOT_OUTLINE_TITLE = settings.DEFAULT_PLOT_OUTLINE_TITLE
DEFAULT_PROTAGONIST_NAME = settings.DEFAULT_PROTAGONIST_NAME
DISABLE_RELATIONSHIP_NORMALIZATION = settings.DISABLE_RELATIONSHIP_NORMALIZATION
EMBEDDING_API_BASE = settings.EMBEDDING_API_BASE
EMBEDDING_API_KEY = settings.EMBEDDING_API_KEY
EMBEDDING_CACHE_SIZE = settings.EMBEDDING_CACHE_SIZE
EMBEDDING_DTYPE = settings.EMBEDDING_DTYPE
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
EMBEDDING_MAX_INPUT_TOKENS = settings.EMBEDDING_MAX_INPUT_TOKENS
ENABLE_LLM_NO_THINK_DIRECTIVE = settings.ENABLE_LLM_NO_THINK_DIRECTIVE
ENABLE_PHASE2_DEDUPLICATION = settings.ENABLE_PHASE2_DEDUPLICATION
ENABLE_RICH_PROGRESS = settings.ENABLE_RICH_PROGRESS
ENTITY_MENTION_THRESHOLD_COMMON_NOUN = settings.ENTITY_MENTION_THRESHOLD_COMMON_NOUN
ENTITY_MENTION_THRESHOLD_PROPER_NOUN = settings.ENTITY_MENTION_THRESHOLD_PROPER_NOUN
EXPECTED_EMBEDDING_DIM = settings.EXPECTED_EMBEDDING_DIM
FALLBACK_CHARS_PER_TOKEN = settings.FALLBACK_CHARS_PER_TOKEN
FILL_IN = settings.FILL_IN
FREQUENCY_PENALTY_DRAFTING = settings.FREQUENCY_PENALTY_DRAFTING
HTTPX_TIMEOUT = settings.HTTPX_TIMEOUT
KG_PREPOPULATION_CHAPTER_NUM = settings.KG_PREPOPULATION_CHAPTER_NUM
KG_TRIPLE_EXTRACTION_CACHE_SIZE = settings.KG_TRIPLE_EXTRACTION_CACHE_SIZE
LARGE_MODEL = settings.LARGE_MODEL
LLM_RETRY_ATTEMPTS = settings.LLM_RETRY_ATTEMPTS
LLM_RETRY_DELAY_SECONDS = settings.LLM_RETRY_DELAY_SECONDS
LLM_TOP_P = settings.LLM_TOP_P
LOG_DATE_FORMAT = settings.LOG_DATE_FORMAT
LOG_FILE = settings.LOG_FILE
LOG_FORMAT = settings.LOG_FORMAT
LOG_LEVEL_STR = settings.LOG_LEVEL_STR
MAIN_NOVEL_INFO_NODE_ID = settings.MAIN_NOVEL_INFO_NODE_ID
MAIN_WORLD_CONTAINER_NODE_ID = settings.MAIN_WORLD_CONTAINER_NODE_ID
MAX_CONCURRENT_CHAPTERS = settings.MAX_CONCURRENT_CHAPTERS
MAX_CONCURRENT_LLM_CALLS = settings.MAX_CONCURRENT_LLM_CALLS
MAX_CONTEXT_TOKENS = settings.MAX_CONTEXT_TOKENS
MAX_GENERATION_TOKENS = settings.MAX_GENERATION_TOKENS
MAX_KG_TRIPLE_TOKENS = settings.MAX_KG_TRIPLE_TOKENS
MAX_PLANNING_TOKENS = settings.MAX_PLANNING_TOKENS
MAX_PREPOP_KG_TOKENS = settings.MAX_PREPOP_KG_TOKENS
MAX_REVISION_CYCLES_PER_CHAPTER = settings.MAX_REVISION_CYCLES_PER_CHAPTER
MAX_SUMMARY_TOKENS = settings.MAX_SUMMARY_TOKENS
MEDIUM_MODEL = settings.MEDIUM_MODEL
MIN_CHAPTER_LENGTH_CHARS = settings.MIN_CHAPTER_LENGTH_CHARS
NARRATIVE_MODEL = settings.NARRATIVE_MODEL
NEO4J_DATABASE = settings.NEO4J_DATABASE
NEO4J_PASSWORD = settings.NEO4J_PASSWORD
NEO4J_URI = settings.NEO4J_URI
NEO4J_USER = settings.NEO4J_USER
NEO4J_VECTOR_DIMENSIONS = settings.NEO4J_VECTOR_DIMENSIONS
NEO4J_VECTOR_INDEX_NAME = settings.NEO4J_VECTOR_INDEX_NAME
NEO4J_VECTOR_SIMILARITY_FUNCTION = settings.NEO4J_VECTOR_SIMILARITY_FUNCTION
OPENAI_API_BASE = settings.OPENAI_API_BASE
OPENAI_API_KEY = settings.OPENAI_API_KEY
PHASE2_NAME_SIMILARITY_THRESHOLD = settings.PHASE2_NAME_SIMILARITY_THRESHOLD
PHASE2_RELATIONSHIP_SIMILARITY_THRESHOLD = settings.PHASE2_RELATIONSHIP_SIMILARITY_THRESHOLD
PLOT_OUTLINE_FILE = settings_mod.PLOT_OUTLINE_FILE

# Entity embedding configuration (Neo4j)
ENTITY_EMBEDDING_VECTOR_PROPERTY = settings.ENTITY_EMBEDDING_VECTOR_PROPERTY
ENTITY_EMBEDDING_TEXT_HASH_PROPERTY = settings.ENTITY_EMBEDDING_TEXT_HASH_PROPERTY
ENTITY_EMBEDDING_MODEL_PROPERTY = settings.ENTITY_EMBEDDING_MODEL_PROPERTY

NEO4J_CHARACTER_ENTITY_VECTOR_INDEX_NAME = settings.NEO4J_CHARACTER_ENTITY_VECTOR_INDEX_NAME
NEO4J_LOCATION_ENTITY_VECTOR_INDEX_NAME = settings.NEO4J_LOCATION_ENTITY_VECTOR_INDEX_NAME
NEO4J_ITEM_ENTITY_VECTOR_INDEX_NAME = settings.NEO4J_ITEM_ENTITY_VECTOR_INDEX_NAME
NEO4J_EVENT_ENTITY_VECTOR_INDEX_NAME = settings.NEO4J_EVENT_ENTITY_VECTOR_INDEX_NAME

# Entity embedding feature flags
ENABLE_ENTITY_EMBEDDING_PERSISTENCE = settings.ENABLE_ENTITY_EMBEDDING_PERSISTENCE
ENABLE_ENTITY_EMBEDDING_DEDUPLICATION = settings.ENABLE_ENTITY_EMBEDDING_DEDUPLICATION
ENABLE_ENTITY_EMBEDDING_GRAPH_HEALING = settings.ENABLE_ENTITY_EMBEDDING_GRAPH_HEALING

# Entity embedding similarity configuration
ENTITY_EMBEDDING_DEDUPLICATION_TOP_K = settings.ENTITY_EMBEDDING_DEDUPLICATION_TOP_K
ENTITY_EMBEDDING_DEDUPLICATION_SIMILARITY_THRESHOLD = settings.ENTITY_EMBEDDING_DEDUPLICATION_SIMILARITY_THRESHOLD
PRESENCE_PENALTY_DRAFTING = settings.PRESENCE_PENALTY_DRAFTING
SIMPLE_LOGGING_MODE = settings.SIMPLE_LOGGING_MODE
SMALL_MODEL = settings.SMALL_MODEL
SPACY_MODEL = settings.SPACY_MODEL
SUMMARY_CACHE_SIZE = settings.SUMMARY_CACHE_SIZE
TARGET_PLOT_POINTS_INITIAL_GENERATION = settings.TARGET_PLOT_POINTS_INITIAL_GENERATION
TARGET_SCENES_MAX = settings.TARGET_SCENES_MAX
TARGET_SCENES_MIN = settings.TARGET_SCENES_MIN
TEMPERATURE_CONSISTENCY_CHECK = settings.TEMPERATURE_CONSISTENCY_CHECK
TEMPERATURE_DRAFTING = settings.TEMPERATURE_DRAFTING
TEMPERATURE_EVALUATION = settings.TEMPERATURE_EVALUATION
TEMPERATURE_INITIAL_SETUP = settings.TEMPERATURE_INITIAL_SETUP
TEMPERATURE_KG_EXTRACTION = settings.TEMPERATURE_KG_EXTRACTION
TEMPERATURE_PATCH = settings.TEMPERATURE_PATCH
TEMPERATURE_PLANNING = settings.TEMPERATURE_PLANNING
TEMPERATURE_REVISION = settings.TEMPERATURE_REVISION
TEMPERATURE_SUMMARY = settings.TEMPERATURE_SUMMARY
TIKTOKEN_DEFAULT_ENCODING = settings.TIKTOKEN_DEFAULT_ENCODING
TOKENIZER_CACHE_SIZE = settings.TOKENIZER_CACHE_SIZE
USER_STORY_ELEMENTS_FILE_PATH = settings.USER_STORY_ELEMENTS_FILE_PATH
WORLD_BUILDER_FILE = settings_mod.WORLD_BUILDER_FILE

ENABLE_DUPLICATE_PREVENTION = settings.ENABLE_DUPLICATE_PREVENTION
DUPLICATE_PREVENTION_SIMILARITY_THRESHOLD = settings.DUPLICATE_PREVENTION_SIMILARITY_THRESHOLD
DUPLICATE_PREVENTION_CHARACTER_ENABLED = settings.DUPLICATE_PREVENTION_CHARACTER_ENABLED
DUPLICATE_PREVENTION_WORLD_ITEM_ENABLED = settings.DUPLICATE_PREVENTION_WORLD_ITEM_ENABLED
ENABLE_PHASE2_DEDUPLICATION = settings.ENABLE_PHASE2_DEDUPLICATION
PHASE2_NAME_SIMILARITY_THRESHOLD = settings.PHASE2_NAME_SIMILARITY_THRESHOLD
PHASE2_RELATIONSHIP_SIMILARITY_THRESHOLD = settings.PHASE2_RELATIONSHIP_SIMILARITY_THRESHOLD
ENABLE_STATUS_IS_ALIAS = settings.ENABLE_STATUS_IS_ALIAS
# REVISION_EVALUATION_THRESHOLD is defined directly in settings.py as a module-level variable
# and is already available via the automatic globals() loop in settings.py
REVISION_EVALUATION_THRESHOLD = 0.85

# Relationship Normalization Settings
ENABLE_RELATIONSHIP_NORMALIZATION = settings.relationship_normalization.ENABLE_RELATIONSHIP_NORMALIZATION
REL_NORM_SIMILARITY_THRESHOLD = settings.relationship_normalization.SIMILARITY_THRESHOLD
REL_NORM_SIMILARITY_THRESHOLD_AMBIGUOUS_MIN = settings.relationship_normalization.SIMILARITY_THRESHOLD_AMBIGUOUS_MIN
REL_NORM_MIN_USAGE_FOR_AUTHORITY = settings.relationship_normalization.MIN_USAGE_FOR_AUTHORITY
REL_NORM_PRUNE_SINGLE_USE_AFTER_CHAPTERS = settings.relationship_normalization.PRUNE_SINGLE_USE_AFTER_CHAPTERS
REL_NORM_MAX_VOCABULARY_SIZE = settings.relationship_normalization.MAX_VOCABULARY_SIZE
REL_NORM_MAX_EXAMPLES_PER_RELATIONSHIP = settings.relationship_normalization.MAX_EXAMPLES_PER_RELATIONSHIP
REL_NORM_USE_LLM_DISAMBIGUATION = settings.relationship_normalization.USE_LLM_DISAMBIGUATION
REL_NORM_LLM_DISAMBIGUATION_JSON_MODE = settings.relationship_normalization.LLM_DISAMBIGUATION_JSON_MODE
REL_NORM_NORMALIZE_CASE_VARIANTS = settings.relationship_normalization.NORMALIZE_CASE_VARIANTS
REL_NORM_NORMALIZE_PUNCTUATION_VARIANTS = settings.relationship_normalization.NORMALIZE_PUNCTUATION_VARIANTS

# Schema Enforcement Settings
ENFORCE_SCHEMA_VALIDATION = settings.schema_enforcement.ENFORCE_SCHEMA_VALIDATION
REJECT_INVALID_ENTITIES = settings.schema_enforcement.REJECT_INVALID_ENTITIES
NORMALIZE_COMMON_VARIANTS = settings.schema_enforcement.NORMALIZE_COMMON_VARIANTS
LOG_SCHEMA_VIOLATIONS = settings.schema_enforcement.LOG_SCHEMA_VIOLATIONS


# Legacy compatibility functions
def get(key: str) -> Any:
    """Return the value of a configuration attribute from `settings`.

    Args:
        key: Attribute name on the `settings` singleton.

    Returns:
        The current value of the named attribute.

    Raises:
        AttributeError: If `key` is not a valid attribute on `settings`.
    """
    return getattr(settings, key)


def set(key: str, value: Any) -> None:
    """Set a configuration attribute on `settings` at runtime.

    This mutates the in-memory settings instance and does not persist to `.env`.

    Args:
        key: Attribute name on the `settings` singleton.
        value: Value to assign.

    Raises:
        AttributeError: If `key` is not a valid attribute on `settings`.
    """
    setattr(settings, key, value)


def reload() -> None:
    """Reload configuration and refresh this package's exported constants.

    This delegates to [`config.loader.reload_settings()`](config/loader.py:35), which may
    overwrite process environment variables by re-reading `.env` with override enabled.

    Raises:
        Exception: Any exception raised by the loader propagates if the loader's internal
            error handling changes. Currently, the loader returns a boolean status and
            suppresses exceptions.
    """
    from .loader import reload_settings

    reload_settings()
