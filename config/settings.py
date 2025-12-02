# config/settings.py
"""
Configuration settings for the Saga Novel Generation system.
Uses Pydantic BaseSettings for automatic environment variable loading.
"""

from __future__ import annotations

import json
import logging as stdlib_logging
import os
from collections.abc import MutableMapping
from typing import Any

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
    TEMPERATURE_KG_EXTRACTION: float = 0.1
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
    PRESENCE_PENALTY_DRAFTING: float = 0.5
    FREQUENCY_PENALTY_KG_EXTRACTION: float = 0.0
    PRESENCE_PENALTY_KG_EXTRACTION: float = 0.5

    # Output and File Paths
    BASE_OUTPUT_DIR: str = "output"
    PLOT_OUTLINE_FILE: str = "plot_outline.json"
    CHARACTER_PROFILES_FILE: str = "character_profiles.json"
    WORLD_BUILDER_FILE: str = "world_building.json"
    CHAPTERS_DIR: str = "chapters"
    CHAPTER_LOGS_DIR: str = "chapter_logs"

    USER_STORY_ELEMENTS_FILE_PATH: str = "user_story_elements.yaml"

    # Generation Parameters
    # Token budgets (defaults are generous; override via FAST_PROFILE for laptops)
    MAX_CONTEXT_TOKENS: int = 40960
    MAX_GENERATION_TOKENS: int = 16384
    CONTEXT_CHAPTER_COUNT: int = 2
    CHAPTERS_PER_RUN: int = 2
    TARGET_PLOT_POINTS_INITIAL_GENERATION: int = 20
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
    MAX_REVISION_CYCLES_PER_CHAPTER: int = 0
    MAX_SUMMARY_TOKENS: int = 8192
    MAX_KG_TRIPLE_TOKENS: int = 16384
    MAX_PREPOP_KG_TOKENS: int = 16384

    # Knowledge Graph Entity Filtering (Proper Noun Preference)
    ENTITY_MENTION_THRESHOLD_PROPER_NOUN: int = 1
    ENTITY_MENTION_THRESHOLD_COMMON_NOUN: int = 3

    # Narrative Agent Configuration
    KG_PREPOPULATION_CHAPTER_NUM: int = 0

    # De-duplication Configuration
    DEDUPLICATION_USE_SEMANTIC: bool = False
    DEDUPLICATION_SEMANTIC_THRESHOLD: float = 0.45
    DEDUPLICATION_MIN_SEGMENT_LENGTH: int = 150

    # Chapter Generation Configuration
    MIN_CHAPTER_LENGTH_CHARS: int = 12000  # Approximately 2500-3000 words

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

    # Enhanced character bootstrap settings
    BOOTSTRAP_MIN_TRAITS_PROTAGONIST: int = 6
    BOOTSTRAP_MIN_TRAITS_ANTAGONIST: int = 5
    BOOTSTRAP_MIN_TRAITS_SUPPORTING: int = 4

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
def filter_internal_keys(
    logger: Any, name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Remove internal structlog fields from event dict."""
    keys_to_remove = [k for k in event_dict.keys() if k.startswith("_")]
    for key in keys_to_remove:
        event_dict.pop(key, None)
    return event_dict


# Simple human-readable formatter for structlog (with Rich markup for console)
def simple_log_format_rich(
    logger: Any, name: str, event_dict: MutableMapping[str, Any]
) -> str:
    """Simple human-readable log formatter with Rich markup for console output."""
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
            if isinstance(value, str) and len(value) > 50:
                value_str = f"{value[:47]}..."
            else:
                value_str = str(value)
            context_parts.append(f"[dim]{key}[/dim]={value_str}")
        if context_parts:
            parts.append(f"({', '.join(context_parts)})")

    return " ".join(parts)


# Simple human-readable formatter for structlog (plain text for files)
def simple_log_format_plain(
    logger: Any, name: str, event_dict: MutableMapping[str, Any]
) -> str:
    """Simple human-readable log formatter without markup for file output."""
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
            if isinstance(value, str) and len(value) > 50:
                value_str = f"{value[:47]}..."
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
    handler = stdlib_logging.FileHandler(
        os.path.join(settings.BASE_OUTPUT_DIR, settings.LOG_FILE)
    )


handler.setFormatter(simple_formatter)
root_logger = stdlib_logging.getLogger()
root_logger.addHandler(handler)
root_logger.setLevel(settings.LOG_LEVEL_STR)

REVISION_EVALUATION_THRESHOLD = 0.85


# Duplicate Prevention Settings
ENABLE_DUPLICATE_PREVENTION: bool = True  # Enable proactive duplicate prevention
DUPLICATE_PREVENTION_SIMILARITY_THRESHOLD: float = (
    0.6  # Similarity threshold for merging entities
)
DUPLICATE_PREVENTION_CHARACTER_ENABLED: bool = (
    True  # Enable character duplicate prevention
)
DUPLICATE_PREVENTION_WORLD_ITEM_ENABLED: bool = (
    True  # Enable world item duplicate prevention
)

# Legacy Degradation Flags (non-breaking defaults)
# Legacy WorldElement toggle removed; single typed-entity model is standard
ENABLE_STATUS_IS_ALIAS: bool = True
