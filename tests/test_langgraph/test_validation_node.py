# tests/test_langgraph/test_validation_node.py
"""
Tests for LangGraph validation node (Step 1.4.1).

Tests the validate_consistency node and its helper functions.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import config
from core.graph_ownership import load_graph_project_id
from core.langgraph.nodes.validation_node import (
    _check_character_traits,
    _is_plot_stagnant,
    _validate_relationships,
    validate_consistency,
)
from core.langgraph.state import ExtractedEntity, ExtractedRelationship, NarrativeState
from tests.fakes.service_context import patch_service
from tests.test_langgraph import InlineExtractionState


def admit_empty_history(state: NarrativeState, manager: MagicMock) -> None:
    state["lifecycle_version"] = 1
    state["extraction_status"] = "complete"
    state["extraction_policy"] = "fail_closed"
    state["graph_project_id"] = load_graph_project_id(Path(state["project_dir"]))
    manager.require_project_binding.return_value = state["graph_project_id"]
    manager.execute_read_query.return_value = []


# Mock ValidationResult for tests (constraint system removed)
class ValidationResult:
    """Mock ValidationResult for backward compatibility with tests."""

    def __init__(
        self,
        is_valid: bool,
        original_relationship: str,
        validated_relationship: str,
        errors: list[str] | None = None,
        suggestions: list[tuple[str, str]] | None = None,
    ) -> None:
        self.is_valid = is_valid
        self.original_relationship = original_relationship
        self.validated_relationship = validated_relationship
        self.errors = errors or []
        self.suggestions = suggestions or []


@pytest.mark.asyncio
class TestValidateConsistency:
    """Tests for validate_consistency node function."""

    async def test_validate_with_no_contradictions(self, sample_state_with_extraction: InlineExtractionState, mock_neo4j_manager: MagicMock) -> None:
        """Test validation with no contradictions found."""
        state = sample_state_with_extraction
        state["draft_word_count"] = 2000

        admit_empty_history(state, mock_neo4j_manager)
        with patch_service('database', mock_neo4j_manager):
            # Mock character trait check to return no contradictions
            with patch(
                "core.langgraph.nodes.validation_node._check_character_traits",
                return_value=[],
            ):
                result = await validate_consistency(state)

                assert result["current_node"] == "validate_consistency"
                assert result["needs_revision"] is False
                assert len(result["contradictions"]) == 0

    async def test_validate_plot_stagnation_detected(self, sample_initial_state: InlineExtractionState, mock_neo4j_manager: MagicMock) -> None:
        """Test that plot stagnation is detected."""
        state = sample_initial_state

        from core.langgraph.content_manager import ContentManager

        content_manager = ContentManager(state["project_dir"])
        draft_ref = content_manager.save_text("Short text", "draft", "chapter_1", 1)

        state["draft_ref"] = draft_ref
        state["draft_word_count"] = 500  # Below threshold
        state["extracted_entities"] = {}
        state["extracted_relationships"] = []

        admit_empty_history(state, mock_neo4j_manager)
        with patch_service('database', mock_neo4j_manager):
            with patch(
                "core.langgraph.nodes.validation_node._check_character_traits",
                return_value=[],
            ):
                with config.bind_settings(config.snapshot_settings().model_copy(update={"validation": config.settings.validation.model_copy(update={"ENABLE_VALIDATION": True})})):
                    result = await validate_consistency(state)

                    stagnation_contradictions = [c for c in result["contradictions"] if c.type == "plot_stagnation"]
                    assert len(stagnation_contradictions) == 1

    async def test_validate_force_continue_bypasses_revision(self, sample_state_with_extraction: InlineExtractionState, mock_neo4j_manager: MagicMock) -> None:
        """Test that force_continue bypasses revision."""
        state = sample_state_with_extraction
        state["draft_word_count"] = 500
        state["extracted_entities"] = {}
        state["extracted_relationships"] = []
        state["force_continue"] = True

        admit_empty_history(state, mock_neo4j_manager)
        with patch_service('database', mock_neo4j_manager):
            with patch(
                "core.langgraph.nodes.validation_node._check_character_traits",
                return_value=[],
            ):
                with config.bind_settings(config.snapshot_settings().model_copy(update={"validation": config.settings.validation.model_copy(update={"ENABLE_VALIDATION": True})})):
                    result = await validate_consistency(state)

                    assert len(result["contradictions"]) == 1
                    assert result["needs_revision"] is False


@pytest.mark.asyncio
class TestValidateRelationships:
    """Tests for _validate_relationships helper function."""

    async def test_validate_valid_relationships(self) -> None:
        """Test validating valid relationships."""
        relationships = [
            ExtractedRelationship(
                source_name="Alice",
                target_name="Bob",
                relationship_type="FRIEND_OF",
                description="Friends",
                chapter=1,
            )
        ]

        # Mock extracted entities to provide type information
        extracted_entities = {
            "characters": [
                ExtractedEntity(
                    name="Alice",
                    type="Character",
                    description="A protagonist",
                    first_appearance_chapter=1,
                ),
                ExtractedEntity(
                    name="Bob",
                    type="Character",
                    description="A supporting character",
                    first_appearance_chapter=1,
                ),
            ],
            "world_items": [],
        }

        # Valid social relationship between characters should pass
        contradictions = await _validate_relationships(relationships, 1, extracted_entities)
        assert len(contradictions) == 0

    async def test_validate_empty_relationships(self) -> None:
        """Test validating empty relationship list."""
        contradictions = await _validate_relationships([], 1)
        assert contradictions == []


@pytest.mark.asyncio
class TestCheckCharacterTraits:
    """Tests for _check_character_traits helper function."""

    async def test_check_no_contradictions(self, mock_neo4j_manager: MagicMock) -> None:
        """Test checking traits with no contradictions."""
        characters = [
            ExtractedEntity(
                name="Alice",
                type="character",
                description="A brave warrior",
                first_appearance_chapter=1,
                # Extraction contract: traits are stored under attributes["traits"] as values.
                attributes={"traits": ["brave", "loyal"]},
            )
        ]

        history = {"Alice": [{"traits": ["Brave", "LOYAL"], "first_chapter": 1}]}
        contradictions = await _check_character_traits(characters, 2, history)
        assert contradictions == []

    async def test_check_contradictory_traits(self, mock_neo4j_manager: MagicMock) -> None:
        """Test checking traits with contradictions (trait VALUES, normalized)."""
        characters = [
            ExtractedEntity(
                name="Alice",
                type="character",
                description="A cowardly warrior",
                first_appearance_chapter=5,
                # Contradicts established "brave" (case/whitespace should not matter)
                attributes={"traits": ["  Cowardly  "]},
            )
        ]

        history = {"Alice": [{"traits": ["Brave"], "first_chapter": 1}]}
        contradictions = await _check_character_traits(characters, 5, history)
        assert len(contradictions) == 1
        assert contradictions[0].type == "character_trait"

    async def test_check_empty_characters(self) -> None:
        """Test checking empty character list."""
        contradictions = await _check_character_traits([], 1, {})
        assert contradictions == []

    async def test_check_new_character(self, mock_neo4j_manager: MagicMock) -> None:
        """Test checking new character with no established traits."""
        characters = [
            ExtractedEntity(
                name="NewCharacter",
                type="character",
                description="New",
                first_appearance_chapter=1,
                attributes={},
            )
        ]

        contradictions = await _check_character_traits(characters, 1, {})
        assert contradictions == []


class TestIsPlotStagnant:
    """Tests for _is_plot_stagnant helper function."""

    def test_stagnant_low_word_count(self, sample_initial_state: InlineExtractionState) -> None:
        """Test that low word count is flagged as stagnant."""
        state = sample_initial_state
        state["draft_word_count"] = 1000  # Below 1500 threshold
        state["extracted_entities"] = {}
        state["extracted_relationships"] = []

        assert _is_plot_stagnant(state) is True

    def test_sufficient_word_count_with_entities_is_not_stagnant(self, sample_initial_state: InlineExtractionState) -> None:
        """Sufficient word count with entities present is not stagnant."""
        state = sample_initial_state
        state["draft_word_count"] = 2000
        state["extracted_entities"] = {
            "characters": [{"name": "Hero"}],
        }
        state["extracted_relationships"] = [
            {"source_name": "Hero", "target_name": "Villain", "relationship_type": "FIGHTS"},
        ]

        assert _is_plot_stagnant(state) is False

    def test_sufficient_word_count_with_no_entities_is_stagnant(self, sample_initial_state: InlineExtractionState) -> None:
        """Sufficient word count but zero entities/relationships is stagnant."""
        state = sample_initial_state
        state["draft_word_count"] = 2000
        state["extracted_entities"] = {}
        state["extracted_relationships"] = []

        assert _is_plot_stagnant(state) is True
