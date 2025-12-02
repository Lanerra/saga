# tests/test_langgraph/conftest.py
"""
Shared fixtures for LangGraph Phase 1 tests.

This module provides common fixtures, mocks, and test data for testing
the LangGraph migration Phase 1 components.
"""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.langgraph.state import (
    Contradiction,
    ExtractedEntity,
    ExtractedRelationship,
    NarrativeState,
    create_initial_state,
)
from models.kg_models import CharacterProfile, WorldItem


@pytest.fixture
def sample_extracted_entity() -> ExtractedEntity:
    """Sample ExtractedEntity for testing."""
    return ExtractedEntity(
        name="Test Character",
        type="character",
        description="A brave test character",
        first_appearance_chapter=1,
        attributes={
            "brave": "",
            "loyal": "",
            "status": "alive",
        },
    )


@pytest.fixture
def sample_extracted_relationship() -> ExtractedRelationship:
    """Sample ExtractedRelationship for testing."""
    return ExtractedRelationship(
        source_name="Alice",
        target_name="Bob",
        relationship_type="FRIEND_OF",
        description="They are close friends",
        chapter=1,
        confidence=0.9,
    )


@pytest.fixture
def sample_contradiction() -> Contradiction:
    """Sample Contradiction for testing."""
    return Contradiction(
        type="character_trait",
        description="Conflicting traits detected",
        conflicting_chapters=[1, 2],
        severity="major",
        suggested_fix="Review character development",
    )


@pytest.fixture
def sample_character_profile() -> CharacterProfile:
    """Sample CharacterProfile for testing."""
    return CharacterProfile(
        name="Test Hero",
        description="A brave hero on a quest",
        traits=["brave", "loyal"],
        status="alive",
        relationships={"Mentor": "student of"},
        created_chapter=1,
        is_provisional=False,
        updates={},
    )


@pytest.fixture
def sample_world_item() -> WorldItem:
    """Sample WorldItem for testing."""
    return WorldItem(
        id="location_001",
        category="location",
        name="Ancient Castle",
        description="A grand medieval castle",
        goals=[],
        rules=["No magic allowed inside"],
        key_elements=["throne room", "dungeon"],
        traits=[],
        created_chapter=1,
        is_provisional=False,
        additional_properties={},
    )


@pytest.fixture
def sample_initial_state() -> NarrativeState:
    """Sample initial NarrativeState for testing."""
    return create_initial_state(
        project_id="test-project",
        title="Test Novel",
        genre="Fantasy",
        theme="Adventure",
        setting="Medieval world",
        target_word_count=80000,
        total_chapters=20,
        project_dir="/tmp/test-project",
        protagonist_name="Hero",
        generation_model="test-model",
        extraction_model="test-model",
        revision_model="test-model",
    )


@pytest.fixture
def sample_state_with_extraction() -> NarrativeState:
    """Sample state with extracted entities and relationships."""
    state = create_initial_state(
        project_id="test-project",
        title="Test Novel",
        genre="Fantasy",
        theme="Adventure",
        setting="Medieval world",
        target_word_count=80000,
        total_chapters=20,
        project_dir="/tmp/test-project",
        protagonist_name="Hero",
    )

    # Add extracted entities
    state["extracted_entities"] = {
        "characters": [
            ExtractedEntity(
                name="Alice",
                type="character",
                description="A brave warrior",
                first_appearance_chapter=1,
                attributes={"brave": "", "status": "alive"},
            ),
            ExtractedEntity(
                name="Bob",
                type="character",
                description="A wise wizard",
                first_appearance_chapter=1,
                attributes={"wise": "", "status": "alive"},
            ),
        ],
        "world_items": [
            ExtractedEntity(
                name="Magic Sword",
                type="object",
                description="A legendary sword",
                first_appearance_chapter=1,
                attributes={"category": "artifact"},
            ),
        ],
    }

    # Add relationships
    state["extracted_relationships"] = [
        ExtractedRelationship(
            source_name="Alice",
            target_name="Bob",
            relationship_type="FRIEND_OF",
            description="They travel together",
            chapter=1,
            confidence=0.9,
        ),
    ]

    # Add draft text
    state["draft_text"] = "Alice and Bob traveled through the forest..."
    state["draft_word_count"] = 2000

    return state


@pytest.fixture
def mock_neo4j_manager() -> Generator[MagicMock, None, None]:
    """Mock Neo4j manager for testing."""
    with patch("core.db_manager.neo4j_manager") as mock:
        mock.execute_read_query = AsyncMock(return_value=[])
        mock.execute_write_query = AsyncMock(return_value=[])
        mock.execute_cypher_batch = AsyncMock(return_value=None)
        yield mock


@pytest.fixture
def mock_llm_service() -> Generator[MagicMock, None, None]:
    """Mock LLM service for testing."""
    with patch("core.llm_interface_refactored.llm_service") as mock:
        mock.async_call_llm = AsyncMock(
            return_value=(
                '{"character_updates": {}, "world_updates": {}, "kg_triples": []}',
                {"prompt_tokens": 100, "completion_tokens": 50},
            )
        )
        yield mock


@pytest.fixture
def mock_knowledge_graph_service() -> Generator[MagicMock, None, None]:
    """Mock knowledge graph service for testing."""
    with patch("core.knowledge_graph_service.knowledge_graph_service") as mock:
        mock.persist_entities = AsyncMock(return_value=True)
        yield mock


@pytest.fixture
def mock_character_queries() -> Generator[MagicMock, None, None]:
    """Mock character queries for testing."""
    with patch("data_access.character_queries") as mock:
        mock.get_character_profile_by_name = AsyncMock(return_value=None)
        mock.get_characters_for_chapter_context_native = AsyncMock(return_value=[])
        yield mock


@pytest.fixture
def mock_world_queries() -> Generator[MagicMock, None, None]:
    """Mock world queries for testing."""
    with patch("data_access.world_queries") as mock:
        mock.get_world_items_for_chapter_context_native = AsyncMock(return_value=[])
        yield mock


@pytest.fixture
def mock_chapter_queries() -> Generator[MagicMock, None, None]:
    """Mock chapter queries for testing."""
    with patch("data_access.chapter_queries") as mock:
        mock.get_chapter_content_batch_native = AsyncMock(return_value={})
        mock.save_chapter_data_to_db = AsyncMock(return_value=None)
        yield mock


@pytest.fixture
def mock_kg_queries() -> Generator[MagicMock, None, None]:
    """Mock kg queries for testing."""
    with patch("data_access.kg_queries") as mock:
        mock.add_kg_triples_batch_to_db = AsyncMock(return_value=None)
        yield mock


@pytest.fixture
def mock_entity_deduplication() -> Generator[MagicMock, None, None]:
    """Mock entity deduplication functions."""
    with patch("processing.entity_deduplication.check_entity_similarity") as mock_check:
        mock_check.return_value = AsyncMock(return_value=None)
        yield mock_check


@pytest.fixture
def sample_llm_extraction_response() -> str:
    """Sample LLM extraction response in JSON format."""
    return """{
        "character_updates": {
            "Alice": {
                "description": "A brave warrior with a strong sense of justice",
                "traits": ["brave", "loyal", "determined"],
                "status": "alive",
                "relationships": {
                    "Bob": "friend and traveling companion"
                }
            },
            "Bob": {
                "description": "A wise wizard with extensive magical knowledge",
                "traits": ["wise", "patient", "powerful"],
                "status": "alive",
                "relationships": {
                    "Alice": "friend and protector"
                }
            }
        },
        "world_updates": {
            "locations": {
                "Dark Forest": {
                    "description": "A mysterious forest filled with ancient magic",
                    "rules": ["No fire magic allowed"],
                    "key_elements": ["ancient trees", "hidden paths"]
                }
            },
            "artifacts": {
                "Magic Sword": {
                    "description": "A legendary sword that glows with power",
                    "goals": ["Defeat the dark lord"],
                    "rules": ["Only the worthy can wield it"]
                }
            }
        },
        "kg_triples": [
            "Character:Alice | WIELDS | Artifact:Magic Sword",
            "Character:Alice | LOCATED_IN | Location:Dark Forest",
            "Character:Bob | TRAVELS_WITH | Character:Alice"
        ]
    }"""
