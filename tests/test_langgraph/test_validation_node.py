# tests/test_langgraph/test_validation_node.py
"""
Tests for LangGraph validation node (Step 1.4.1).

Tests the validate_consistency node and its helper functions.
"""

from unittest.mock import patch

import pytest

from core.langgraph.nodes.validation_node import (
    _check_character_traits,
    _is_plot_stagnant,
    _validate_relationships,
    validate_consistency,
)
from core.langgraph.state import ExtractedEntity, ExtractedRelationship


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
    ):
        self.is_valid = is_valid
        self.original_relationship = original_relationship
        self.validated_relationship = validated_relationship
        self.errors = errors or []
        self.suggestions = suggestions or []


@pytest.mark.asyncio
class TestValidateConsistency:
    """Tests for validate_consistency node function."""

    async def test_validate_with_no_contradictions(self, sample_state_with_extraction, mock_neo4j_manager):
        """Test validation with no contradictions found."""
        state = sample_state_with_extraction
        state["draft_word_count"] = 2000

        with patch("core.langgraph.nodes.validation_node.neo4j_manager", mock_neo4j_manager):
            # Mock character trait check to return no contradictions
            with patch(
                "core.langgraph.nodes.validation_node._check_character_traits",
                return_value=[],
            ):
                result = await validate_consistency(state)

                assert result["current_node"] == "validate_consistency"
                assert result["needs_revision"] is False
                assert len(result["contradictions"]) == 0

    async def test_validate_plot_stagnation_detected(self, sample_initial_state, mock_neo4j_manager):
        """Test that plot stagnation is detected."""
        state = sample_initial_state
        state["draft_text"] = "Short text"
        state["draft_word_count"] = 500  # Below threshold
        state["extracted_entities"] = {}
        state["extracted_relationships"] = []

        with patch("core.langgraph.nodes.validation_node.neo4j_manager", mock_neo4j_manager):
            with patch(
                "core.langgraph.nodes.validation_node._check_character_traits",
                return_value=[],
            ):
                result = await validate_consistency(state)

                # Should detect plot stagnation
                stagnation_contradictions = [c for c in result["contradictions"] if c.type == "plot_stagnation"]
                assert len(stagnation_contradictions) > 0

    async def test_validate_force_continue_bypasses_revision(self, sample_state_with_extraction, mock_neo4j_manager):
        """Test that force_continue bypasses revision."""
        state = sample_state_with_extraction
        # Create a condition that would normally trigger revision (e.g. stagnation)
        state["draft_word_count"] = 500
        state["extracted_entities"] = {}
        state["extracted_relationships"] = []
        state["force_continue"] = True

        with patch("core.langgraph.nodes.validation_node.neo4j_manager", mock_neo4j_manager):
            with patch(
                "core.langgraph.nodes.validation_node._check_character_traits",
                return_value=[],
            ):
                result = await validate_consistency(state)

                # Should have contradictions (stagnation)
                assert len(result["contradictions"]) > 0
                # But should not need revision due to force_continue
                assert result["needs_revision"] is False


@pytest.mark.asyncio
class TestValidateRelationships:
    """Tests for _validate_relationships helper function."""

    async def test_validate_valid_relationships(self):
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

    async def test_validate_empty_relationships(self):
        """Test validating empty relationship list."""
        contradictions = await _validate_relationships([], 1)
        assert contradictions == []


@pytest.mark.asyncio
class TestCheckCharacterTraits:
    """Tests for _check_character_traits helper function."""

    async def test_check_no_contradictions(self, mock_neo4j_manager):
        """Test checking traits with no contradictions."""
        characters = [
            ExtractedEntity(
                name="Alice",
                type="character",
                description="A brave warrior",
                first_appearance_chapter=1,
                attributes={"brave": "", "loyal": ""},
            )
        ]

        # Mock Neo4j to return existing traits that don't conflict
        mock_neo4j_manager.execute_read_query.return_value = [{"traits": ["brave", "loyal"], "first_chapter": 1, "description": "..."}]

        with patch("core.langgraph.nodes.validation_node.neo4j_manager", mock_neo4j_manager):
            contradictions = await _check_character_traits(characters, 1)
            assert len(contradictions) == 0

    async def test_check_contradictory_traits(self, mock_neo4j_manager):
        """Test checking traits with contradictions."""
        characters = [
            ExtractedEntity(
                name="Alice",
                type="character",
                description="A cowardly warrior",
                first_appearance_chapter=5,
                attributes={"cowardly": ""},  # Contradicts "brave"
            )
        ]

        # Mock Neo4j to return "brave" as established trait
        mock_neo4j_manager.execute_read_query.return_value = [{"traits": ["brave"], "first_chapter": 1, "description": "A brave warrior"}]

        with patch("core.langgraph.nodes.validation_node.neo4j_manager", mock_neo4j_manager):
            contradictions = await _check_character_traits(characters, 5)

            # Should detect brave/cowardly contradiction
            assert len(contradictions) > 0
            assert contradictions[0].type == "character_trait"

    async def test_check_empty_characters(self):
        """Test checking empty character list."""
        contradictions = await _check_character_traits([], 1)
        assert contradictions == []

    async def test_check_new_character(self, mock_neo4j_manager):
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

        # Mock Neo4j to return empty result (no existing character)
        mock_neo4j_manager.execute_read_query.return_value = []

        with patch("core.langgraph.nodes.validation_node.neo4j_manager", mock_neo4j_manager):
            contradictions = await _check_character_traits(characters, 1)
            assert len(contradictions) == 0


class TestIsPlotStagnant:
    """Tests for _is_plot_stagnant helper function."""

    def test_stagnant_low_word_count(self, sample_initial_state):
        """Test that low word count is flagged as stagnant."""
        state = sample_initial_state
        state["draft_word_count"] = 1000  # Below 1500 threshold
        state["extracted_entities"] = {}
        state["extracted_relationships"] = []

        assert _is_plot_stagnant(state) is True

    def test_stagnant_no_elements(self, sample_initial_state):
        """Test that no new elements or relationships is stagnant."""
        state = sample_initial_state
        state["draft_word_count"] = 2000  # Above threshold
        state["extracted_entities"] = {}  # No entities
        state["extracted_relationships"] = []  # No relationships

        assert _is_plot_stagnant(state) is True

    def test_not_stagnant_with_content(self, sample_state_with_extraction):
        """Test that chapter with content is not stagnant."""
        state = sample_state_with_extraction
        state["draft_word_count"] = 2000

        assert _is_plot_stagnant(state) is False

    def test_not_stagnant_with_relationships(self, sample_initial_state):
        """Test that chapter with relationships is not stagnant."""
        state = sample_initial_state
        state["draft_word_count"] = 2000
        state["extracted_entities"] = {}
        state["extracted_relationships"] = [
            ExtractedRelationship(
                source_name="Alice",
                target_name="Bob",
                relationship_type="FRIEND_OF",
                description="Friends",
                chapter=1,
            )
        ]

        assert _is_plot_stagnant(state) is False

    def test_not_stagnant_with_new_characters(self, sample_initial_state):
        """Test that chapter with new characters is not stagnant."""
        state = sample_initial_state
        state["draft_word_count"] = 2000
        state["extracted_entities"] = {
            "characters": [
                ExtractedEntity(
                    name="NewChar",
                    type="character",
                    description="New",
                    first_appearance_chapter=1,
                    attributes={},
                )
            ],
            "world_items": [],
        }
        state["extracted_relationships"] = []

        assert _is_plot_stagnant(state) is False
