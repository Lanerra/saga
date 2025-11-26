# tests/test_langgraph/test_state.py
"""
Tests for LangGraph state schema (Step 1.1.1).

Tests the NarrativeState, ExtractedEntity, ExtractedRelationship,
Contradiction models, and create_initial_state function.
"""

import pytest
from pydantic import ValidationError

from core.langgraph.state import (
    Contradiction,
    ExtractedEntity,
    ExtractedRelationship,
    create_initial_state,
)


class TestExtractedEntity:
    """Tests for ExtractedEntity model."""

    def test_create_character_entity(self, sample_extracted_entity):
        """Test creating a character entity."""
        entity = sample_extracted_entity
        assert entity.name == "Test Character"
        assert entity.type == "character"
        assert entity.description == "A brave test character"
        assert entity.first_appearance_chapter == 1
        assert "brave" in entity.attributes

    def test_create_location_entity(self):
        """Test creating a location entity."""
        entity = ExtractedEntity(
            name="Ancient Castle",
            type="location",
            description="A grand medieval castle",
            first_appearance_chapter=1,
            attributes={"category": "settlement"},
        )
        assert entity.type == "location"
        assert entity.name == "Ancient Castle"

    def test_create_object_entity(self):
        """Test creating an object entity."""
        entity = ExtractedEntity(
            name="Magic Sword",
            type="object",
            description="A legendary blade",
            first_appearance_chapter=1,
            attributes={"category": "artifact"},
        )
        assert entity.type == "object"

    def test_create_event_entity(self):
        """Test creating an event entity."""
        entity = ExtractedEntity(
            name="Battle of the Fields",
            type="event",
            description="A decisive battle",
            first_appearance_chapter=5,
            attributes={"importance": 0.9},
        )
        assert entity.type == "event"

    def test_entity_with_empty_attributes(self):
        """Test entity with empty attributes dict."""
        entity = ExtractedEntity(
            name="Test",
            type="character",
            description="Test character",
            first_appearance_chapter=1,
            attributes={},
        )
        assert entity.attributes == {}

    def test_entity_attributes_are_mutable(self):
        """Test that entity attributes can be modified."""
        entity = ExtractedEntity(
            name="Test",
            type="character",
            description="Test",
            first_appearance_chapter=1,
            attributes={"trait1": "value1"},
        )
        entity.attributes["trait2"] = "value2"
        assert "trait2" in entity.attributes


class TestExtractedRelationship:
    """Tests for ExtractedRelationship model."""

    def test_create_relationship(self, sample_extracted_relationship):
        """Test creating a relationship."""
        rel = sample_extracted_relationship
        assert rel.source_name == "Alice"
        assert rel.target_name == "Bob"
        assert rel.relationship_type == "FRIEND_OF"
        assert rel.description == "They are close friends"
        assert rel.chapter == 1
        assert rel.confidence == 0.9

    def test_relationship_with_default_confidence(self):
        """Test relationship uses default confidence."""
        rel = ExtractedRelationship(
            source_name="Alice",
            target_name="Bob",
            relationship_type="KNOWS",
            description="They know each other",
            chapter=1,
        )
        assert rel.confidence == 0.8  # Default value

    def test_relationship_with_custom_confidence(self):
        """Test relationship with custom confidence."""
        rel = ExtractedRelationship(
            source_name="Alice",
            target_name="Bob",
            relationship_type="LOVES",
            description="Deep love",
            chapter=5,
            confidence=0.95,
        )
        assert rel.confidence == 0.95

    def test_relationship_validation(self):
        """Test that relationship validates required fields."""
        with pytest.raises(ValidationError):
            ExtractedRelationship(
                source_name="Alice",
                # Missing target_name
                relationship_type="FRIEND_OF",
                description="Test",
                chapter=1,
            )


class TestContradiction:
    """Tests for Contradiction model."""

    def test_create_contradiction(self, sample_contradiction):
        """Test creating a contradiction."""
        contradiction = sample_contradiction
        assert contradiction.type == "character_trait"
        assert contradiction.description == "Conflicting traits detected"
        assert contradiction.conflicting_chapters == [1, 2]
        assert contradiction.severity == "major"
        assert contradiction.suggested_fix == "Review character development"

    def test_contradiction_severity_validation(self):
        """Test contradiction severity must be valid literal."""
        # Valid severity
        contradiction = Contradiction(
            type="test",
            description="Test",
            conflicting_chapters=[1],
            severity="minor",
        )
        assert contradiction.severity == "minor"

        # Invalid severity should raise error
        with pytest.raises(ValidationError):
            Contradiction(
                type="test",
                description="Test",
                conflicting_chapters=[1],
                severity="invalid",  # Not in Literal types
            )

    def test_contradiction_without_suggested_fix(self):
        """Test contradiction can be created without suggested fix."""
        contradiction = Contradiction(
            type="plot_stagnation",
            description="Plot not advancing",
            conflicting_chapters=[3],
            severity="major",
        )
        assert contradiction.suggested_fix is None

    def test_contradiction_with_multiple_chapters(self):
        """Test contradiction with multiple conflicting chapters."""
        contradiction = Contradiction(
            type="timeline",
            description="Timeline inconsistency",
            conflicting_chapters=[1, 3, 5, 7],
            severity="critical",
        )
        assert len(contradiction.conflicting_chapters) == 4


class TestCreateInitialState:
    """Tests for create_initial_state function."""

    def test_create_minimal_state(self):
        """Test creating state with minimal required parameters."""
        state = create_initial_state(
            project_id="test",
            title="Test Novel",
            genre="Fantasy",
            theme="Adventure",
            setting="Medieval",
            target_word_count=80000,
            total_chapters=20,
            project_dir="/tmp/test",
            protagonist_name="Hero",
        )

        # Check required fields
        assert state["project_id"] == "test"
        assert state["title"] == "Test Novel"
        assert state["genre"] == "Fantasy"
        assert state["theme"] == "Adventure"
        assert state["setting"] == "Medieval"
        assert state["target_word_count"] == 80000
        assert state["total_chapters"] == 20
        assert state["protagonist_name"] == "Hero"

        # Check defaults
        assert state["current_chapter"] == 1
        assert state["current_act"] == 1
        assert state["plot_outline"] == {}
        assert state["active_characters"] == []
        assert state["draft_text"] is None
        assert state["draft_word_count"] == 0
        assert state["extracted_entities"] == {}
        assert state["extracted_relationships"] == []
        assert state["contradictions"] == []
        assert state["needs_revision"] is False
        assert state["iteration_count"] == 0
        assert state["max_iterations"] == 3

    def test_create_state_with_custom_models(self):
        """Test creating state with custom model names."""
        state = create_initial_state(
            project_id="test",
            title="Test",
            genre="Fantasy",
            theme="Adventure",
            setting="Medieval",
            target_word_count=80000,
            total_chapters=20,
            project_dir="/tmp/test",
            protagonist_name="Hero",
            generation_model="custom-gen",
            extraction_model="custom-extract",
            revision_model="custom-revise",
        )

        assert state["generation_model"] == "custom-gen"
        assert state["extraction_model"] == "custom-extract"
        assert state["revision_model"] == "custom-revise"

    def test_create_state_with_custom_max_iterations(self):
        """Test creating state with custom max iterations."""
        state = create_initial_state(
            project_id="test",
            title="Test",
            genre="Fantasy",
            theme="Adventure",
            setting="Medieval",
            target_word_count=80000,
            total_chapters=20,
            project_dir="/tmp/test",
            protagonist_name="Hero",
            max_iterations=5,
        )

        assert state["max_iterations"] == 5

    def test_state_filesystem_paths(self):
        """Test that filesystem paths are correctly constructed."""
        state = create_initial_state(
            project_id="test",
            title="Test",
            genre="Fantasy",
            theme="Adventure",
            setting="Medieval",
            target_word_count=80000,
            total_chapters=20,
            project_dir="/tmp/my-novel",
            protagonist_name="Hero",
        )

        assert state["project_dir"] == "/tmp/my-novel"
        assert state["chapters_dir"] == "/tmp/my-novel/chapters"
        assert state["summaries_dir"] == "/tmp/my-novel/summaries"

    def test_state_has_all_required_fields(self, sample_initial_state):
        """Test that initial state has all required fields from schema."""
        state = sample_initial_state

        # Project metadata
        assert "project_id" in state
        assert "title" in state
        assert "genre" in state
        assert "theme" in state
        assert "setting" in state
        assert "target_word_count" in state

        # Position
        assert "current_chapter" in state
        assert "total_chapters" in state
        assert "current_act" in state

        # Neo4j connection
        assert "neo4j_conn" in state

        # Outline
        assert "plot_outline" in state

        # Active context
        assert "active_characters" in state
        assert "current_location" in state
        assert "previous_chapter_summaries" in state
        assert "key_events" in state

        # Generated content
        assert "draft_text" in state
        assert "draft_word_count" in state

        # Entity extraction
        assert "extracted_entities" in state
        assert "extracted_relationships" in state

        # Validation
        assert "contradictions" in state
        assert "needs_revision" in state
        assert "revision_feedback" in state

        # Model configuration
        assert "generation_model" in state
        assert "extraction_model" in state
        assert "revision_model" in state

        # Workflow control
        assert "current_node" in state
        assert "iteration_count" in state
        assert "max_iterations" in state
        assert "force_continue" in state

        # Error handling
        assert "last_error" in state
        assert "retry_count" in state

        # Filesystem paths
        assert "project_dir" in state
        assert "chapters_dir" in state
        assert "summaries_dir" in state

    def test_state_is_mutable(self, sample_initial_state):
        """Test that state dict is mutable and can be updated."""
        state = sample_initial_state

        # Update values
        state["current_chapter"] = 5
        state["draft_text"] = "Chapter 5 text..."
        state["contradictions"] = [
            Contradiction(
                type="test",
                description="Test contradiction",
                conflicting_chapters=[5],
                severity="minor",
            )
        ]

        assert state["current_chapter"] == 5
        assert state["draft_text"] == "Chapter 5 text..."
        assert len(state["contradictions"]) == 1
