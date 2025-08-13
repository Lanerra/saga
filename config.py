# config.py
"""Configuration settings for the Saga Novel Generation system.
Uses Pydantic BaseSettings for automatic environment variable loading.
"""

from __future__ import annotations

import json
import logging as stdlib_logging
import os
from typing import List, Optional

import structlog
from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

logger = structlog.get_logger()


async def _load_list_from_json_async(
    file_path: str, default_if_missing: Optional[List[str]] = None
) -> List[str]:
    """Load a list of strings from a JSON file asynchronously."""
    if default_if_missing is None:
        default_if_missing = []
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
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
    NEO4J_DATABASE: Optional[str] = "neo4j"

    # Neo4j Vector Index Configuration
    NEO4J_VECTOR_INDEX_NAME: str = "chapterEmbeddings"
    NEO4J_VECTOR_NODE_LABEL: str = "Chapter"
    NEO4J_VECTOR_PROPERTY_NAME: str = "embedding_vector"
    NEO4J_VECTOR_DIMENSIONS: int = 768
    NEO4J_VECTOR_SIMILARITY_FUNCTION: str = "cosine"

    # Base Model Definitions
    LARGE_MODEL: str = "Qwen3-14B-Q4"
    MEDIUM_MODEL: str = "Qwen3-8B-Q4"
    SMALL_MODEL: str = "Qwen3-4B-Q4"
    NARRATIVE_MODEL: str = "Qwen3-14B-Q4"

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
    FILL_IN: str = "[Fill-in]"

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
    FALLBACK_GENERATION_MODEL: Optional[str] = None
    MAIN_GENERATION_MODEL: Optional[str] = None
    KNOWLEDGE_UPDATE_MODEL: Optional[str] = None
    INITIAL_SETUP_MODEL: Optional[str] = None
    PLANNING_MODEL: Optional[str] = None
    DRAFTING_MODEL: Optional[str] = None
    NARRATIVE_MODEL: Optional[str] = None
    REVISION_MODEL: Optional[str] = None
    EVALUATION_MODEL: Optional[str] = None
    PATCH_GENERATION_MODEL: Optional[str] = None

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

    UNHINGED_DATA_DIR: str = "unhinged_data"
    UNHINGED_GENRES_FILE: str = "unhinged_genres.json"
    UNHINGED_THEMES_FILE: str = "unhinged_themes.json"
    UNHINGED_SETTINGS_FILE: str = "unhinged_settings_archetypes.json"
    UNHINGED_PROTAGONISTS_FILE: str = "unhinged_protagonist_archetypes.json"
    UNHINGED_CONFLICTS_FILE: str = "unhinged_conflict_types.json"

    # Generation Parameters
    MAX_CONTEXT_TOKENS: int = 40960
    MAX_GENERATION_TOKENS: int = 16384
    CONTEXT_CHAPTER_COUNT: int = 5
    CHAPTERS_PER_RUN: int = 3
    KG_HEALING_INTERVAL: int = 2
    TARGET_PLOT_POINTS_INITIAL_GENERATION: int = 18

    # Caching
    EMBEDDING_CACHE_SIZE: int = 128
    SUMMARY_CACHE_SIZE: int = 32
    KG_TRIPLE_EXTRACTION_CACHE_SIZE: int = 16
    TOKENIZER_CACHE_SIZE: int = 10

    # Reranking Configuration
    ENABLE_RERANKING: bool = False
    RERANKER_CANDIDATE_COUNT: int = 15

    # Agentic Planning & Prompt Context Snippets
    ENABLE_AGENTIC_PLANNING: bool = False
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

    MIN_ACCEPTABLE_DRAFT_LENGTH_DEFAULT: int = 12000
    MIN_ACCEPTABLE_DRAFT_LENGTH: int = 12000

    ENABLE_DYNAMIC_STATE_ADAPTATION: bool = True
    KG_PREPOPULATION_CHAPTER_NUM: int = 0

    # De-duplication Configuration
    DEDUPLICATION_USE_SEMANTIC: bool = True
    DEDUPLICATION_SEMANTIC_THRESHOLD: float = 0.85
    DEDUPLICATION_MIN_SEGMENT_LENGTH: int = 150

    # Logging & UI
    LOG_LEVEL_STR: str = Field("INFO", alias="LOG_LEVEL")
    LOG_FORMAT: str = (
        "%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
    )
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_FILE: Optional[str] = "saga_run.log"
    ENABLE_RICH_PROGRESS: bool = True

    # Novel Configuration (Defaults / Placeholders)
    UNHINGED_PLOT_MODE: bool = False
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

    @model_validator(mode="after")
    def set_dynamic_model_defaults(self) -> "SagaSettings":
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


# --- Unhinged Mode Data (Loaded from JSON files) ---
_DEFAULT_GENRE_LIST = ["science fiction", "fantasy", "horror"]
_DEFAULT_THEMES_LIST = ["the nature of reality", "the cost of power"]
_DEFAULT_SETTINGS_LIST = ["a floating city", "a derelict starship"]
_DEFAULT_PROTAGONISTS_LIST = ["a reluctant hero", "a cynical detective"]
_DEFAULT_CONFLICTS_LIST = ["man vs self", "man vs society"]

# These will be populated by load_unhinged_data_async
UNHINGED_GENRES: List[str] = []
UNHINGED_THEMES: List[str] = []
UNHINGED_SETTINGS_ARCHETYPES: List[str] = []
UNHINGED_PROTAGONIST_ARCHETYPES: List[str] = []
UNHINGED_CONFLICT_TYPES: List[str] = []


async def load_unhinged_data_async() -> None:
    """Asynchronously load all unhinged data files."""
    global UNHINGED_GENRES, UNHINGED_THEMES, UNHINGED_SETTINGS_ARCHETYPES
    global UNHINGED_PROTAGONIST_ARCHETYPES, UNHINGED_CONFLICT_TYPES

    UNHINGED_GENRES = await _load_list_from_json_async(
        os.path.join(settings.UNHINGED_DATA_DIR, settings.UNHINGED_GENRES_FILE),
        _DEFAULT_GENRE_LIST,
    )
    UNHINGED_THEMES = await _load_list_from_json_async(
        os.path.join(settings.UNHINGED_DATA_DIR, settings.UNHINGED_THEMES_FILE),
        _DEFAULT_THEMES_LIST,
    )
    UNHINGED_SETTINGS_ARCHETYPES = await _load_list_from_json_async(
        os.path.join(settings.UNHINGED_DATA_DIR, settings.UNHINGED_SETTINGS_FILE),
        _DEFAULT_SETTINGS_LIST,
    )
    UNHINGED_PROTAGONIST_ARCHETYPES = await _load_list_from_json_async(
        os.path.join(settings.UNHINGED_DATA_DIR, settings.UNHINGED_PROTAGONISTS_FILE),
        _DEFAULT_PROTAGONISTS_LIST,
    )
    UNHINGED_CONFLICT_TYPES = await _load_list_from_json_async(
        os.path.join(settings.UNHINGED_DATA_DIR, settings.UNHINGED_CONFLICTS_FILE),
        _DEFAULT_CONFLICTS_LIST,
    )

    if not UNHINGED_GENRES or UNHINGED_GENRES == _DEFAULT_GENRE_LIST:
        logger.warning(
            "UNHINGED_GENRES might be using default values. Check unhinged_genres.json."
        )


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
UNHINGED_GENRES_FILE = os.path.join(
    settings.UNHINGED_DATA_DIR, settings.UNHINGED_GENRES_FILE
)
UNHINGED_THEMES_FILE = os.path.join(
    settings.UNHINGED_DATA_DIR, settings.UNHINGED_THEMES_FILE
)
UNHINGED_SETTINGS_FILE = os.path.join(
    settings.UNHINGED_DATA_DIR, settings.UNHINGED_SETTINGS_FILE
)
UNHINGED_PROTAGONISTS_FILE = os.path.join(
    settings.UNHINGED_DATA_DIR, settings.UNHINGED_PROTAGONISTS_FILE
)
UNHINGED_CONFLICTS_FILE = os.path.join(
    settings.UNHINGED_DATA_DIR, settings.UNHINGED_CONFLICTS_FILE
)

# Ensure output directories exist
os.makedirs(settings.UNHINGED_DATA_DIR, exist_ok=True)
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

NARRATIVE_MODEL = "qwen3-a3b"
KNOWLEDGE_GRAPH_URL = "bolt://localhost:7687"
REVISION_EVALUATION_THRESHOLD = 0.85
