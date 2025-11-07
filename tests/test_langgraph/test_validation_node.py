"""
Tests for LangGraph validation node (Step 1.4.1).

Tests the validate_consistency node and its helper functions.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.langgraph.nodes.validation_node import (
    validate_consistency,
    _validate_relationships,
    _check_character_traits,
    _is_plot_stagnant,
)
from core.langgraph.state import ExtractedEntity, ExtractedRelationship
from core.relationship_validator import ValidationResult


@pytest.mark.asyncio
class TestValidateConsistency:
    """Tests for validate_consistency node function."""

    async def test_validate_with_no_contradictions(
        self, sample_state_with_extraction, mock_neo4j_manager
    ):
        """Test validation with no contradictions found."""
        state = sample_state_with_extraction
        state["draft_word_count"] = 2000

        # Mock validation to return valid results
        with patch("core.langgraph.nodes.validation_node.validate_batch_constraints") as mock_validate:
            mock_validate.return_value = [
                ValidationResult(
                    is_valid=True,
                    original_relationship="FRIEND_OF",
                    validated_relationship="FRIEND_OF",
                )
            ]

            with patch("core.langgraph.nodes.validation_node.neo4j_manager", mock_neo4j_manager):
                result = await validate_consistency(state)

                assert result["current_node"] == "validate_consistency"
                assert result["needs_revision"] is False
                assert len(result["contradictions"]) == 0

    async def test_validate_with_invalid_relationships(
        self, sample_state_with_extraction, mock_neo4j_manager
    ):
        """Test validation with invalid relationships."""
        state = sample_state_with_extraction
        state["draft_word_count"] = 2000

        # Mock validation to return invalid results
        with patch("core.langgraph.nodes.validation_node.validate_batch_constraints") as mock_validate:
            mock_validate.return_value = [
                ValidationResult(
                    is_valid=False,
                    original_relationship="INVALID_REL",
                    validated_relationship="INVALID_REL",
                    errors=["Invalid semantic match"],
                    suggestions=[("FRIEND_OF", "Use FRIEND_OF instead")],
                )
            ]

            with patch("core.langgraph.nodes.validation_node.neo4j_manager", mock_neo4j_manager):
                result = await validate_consistency(state)

                assert len(result["contradictions"]) > 0
                assert result["contradictions"][0].type == "relationship"

    async def test_validate_plot_stagnation_detected(
        self, sample_initial_state, mock_neo4j_manager
    ):
        """Test that plot stagnation is detected."""
        state = sample_initial_state
        state["draft_text"] = "Short text"
        state["draft_word_count"] = 500  # Below threshold
        state["extracted_entities"] = {}
        state["extracted_relationships"] = []

        with patch("core.langgraph.nodes.validation_node.validate_batch_constraints") as mock_validate:
            mock_validate.return_value = []

            with patch("core.langgraph.nodes.validation_node.neo4j_manager", mock_neo4j_manager):
                result = await validate_consistency(state)

                # Should detect plot stagnation
                stagnation_contradictions = [
                    c for c in result["contradictions"] if c.type == "plot_stagnation"
                ]
                assert len(stagnation_contradictions) > 0

    async def test_validate_revision_logic_major_issues(
        self, sample_state_with_extraction, mock_neo4j_manager
    ):
        """Test that multiple major issues trigger revision."""
        state = sample_state_with_extraction
        state["draft_word_count"] = 2000

        # Add 3 relationships to match the 3 ValidationResults
        state["extracted_relationships"] = [
            ExtractedRelationship(
                source_name="Alice", target_name="Bob",
                relationship_type="REL1", description="Rel 1", chapter=1
            ),
            ExtractedRelationship(
                source_name="Bob", target_name="Charlie",
                relationship_type="REL2", description="Rel 2", chapter=1
            ),
            ExtractedRelationship(
                source_name="Alice", target_name="Charlie",
                relationship_type="REL3", description="Rel 3", chapter=1
            ),
        ]

        # Mock validation to return multiple invalid results
        with patch("core.langgraph.nodes.validation_node.validate_batch_constraints") as mock_validate:
            mock_validate.return_value = [
                ValidationResult(
                    is_valid=False,
                    original_relationship="REL1",
                    errors=["Error 1"],
                ),
                ValidationResult(
                    is_valid=False,
                    original_relationship="REL2",
                    errors=["Error 2"],
                ),
                ValidationResult(
                    is_valid=False,
                    original_relationship="REL3",
                    errors=["Error 3"],
                ),
            ]

            with patch("core.langgraph.nodes.validation_node.neo4j_manager", mock_neo4j_manager):
                result = await validate_consistency(state)

                # More than 2 major issues should trigger revision
                assert result["needs_revision"] is True

    async def test_validate_force_continue_bypasses_revision(
        self, sample_state_with_extraction, mock_neo4j_manager
    ):
        """Test that force_continue bypasses revision."""
        state = sample_state_with_extraction
        state["draft_word_count"] = 2000
        state["force_continue"] = True

        # Mock validation to return invalid results
        with patch("core.langgraph.nodes.validation_node.validate_batch_constraints") as mock_validate:
            mock_validate.return_value = [
                ValidationResult(
                    is_valid=False,
                    original_relationship="INVALID",
                    errors=["Error"],
                )
            ]

            with patch("core.langgraph.nodes.validation_node.neo4j_manager", mock_neo4j_manager):
                result = await validate_consistency(state)

                # Should not need revision due to force_continue
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

        with patch("core.langgraph.nodes.validation_node.validate_batch_constraints") as mock_validate:
            mock_validate.return_value = [
                ValidationResult(
                    is_valid=True,
                    original_relationship="FRIEND_OF",
                    validated_relationship="FRIEND_OF",
                )
            ]

            contradictions = await _validate_relationships(relationships, 1)
            assert len(contradictions) == 0

    async def test_validate_empty_relationships(self):
        """Test validating empty relationship list."""
        contradictions = await _validate_relationships([], 1)
        assert contradictions == []

    async def test_validate_handles_errors(self):
        """Test that validation handles errors gracefully."""
        relationships = [
            ExtractedRelationship(
                source_name="Alice",
                target_name="Bob",
                relationship_type="FRIEND_OF",
                description="Friends",
                chapter=1,
            )
        ]

        with patch("core.langgraph.nodes.validation_node.validate_batch_constraints") as mock_validate:
            mock_validate.side_effect = Exception("Validation error")

            contradictions = await _validate_relationships(relationships, 1)
            # Should return empty list on error
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
        mock_neo4j_manager.execute_read_query.return_value = [
            {"traits": ["brave", "loyal"], "first_chapter": 1, "description": "..."}
        ]

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
        mock_neo4j_manager.execute_read_query.return_value = [
            {"traits": ["brave"], "first_chapter": 1, "description": "A brave warrior"}
        ]

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
