"""Tests for LOCATED_AT relationship semantic validation and type inference."""

import pytest

from core.langgraph.state import ExtractedEntity, ExtractedRelationship
from core.relationship_validation import (
    infer_entity_type_from_relationship,
    validate_relationship_semantics_strict,
)


class TestLocatedAtSemanticValidation:
    """Ensure LOCATED_AT only links Characters to Locations."""

    def test_validates_character_to_location(self):
        """Valid: Character → LOCATED_AT → Location should be accepted."""
        is_valid, error = validate_relationship_semantics_strict(
            relationship_type="LOCATED_AT",
            source_type="Character",
            target_type="Location",
        )
        assert is_valid is True
        assert error is None

    def test_rejects_character_to_character(self):
        """Invalid: Character → LOCATED_AT → Character should be rejected."""
        is_valid, error = validate_relationship_semantics_strict(
            relationship_type="LOCATED_AT",
            source_type="Character",
            target_type="Character",
        )
        assert is_valid is False
        assert error is not None
        assert "Character" in error
        assert "LOCATED_AT" in error

    def test_rejects_character_to_item(self):
        """Invalid: Character → LOCATED_AT → Item should be rejected."""
        is_valid, error = validate_relationship_semantics_strict(
            relationship_type="LOCATED_AT",
            source_type="Character",
            target_type="Item",
        )
        assert is_valid is False
        assert error is not None

    def test_validates_item_to_location(self):
        """Valid: Item → LOCATED_AT → Location should be accepted."""
        is_valid, error = validate_relationship_semantics_strict(
            relationship_type="LOCATED_AT",
            source_type="Item",
            target_type="Location",
        )
        assert is_valid is True
        assert error is None


class TestTypeInferenceFromRelationship:
    """Test entity type inference from relationship semantics."""

    def test_infers_location_from_located_at_target(self):
        """When target type is unknown, infer Location from LOCATED_AT."""
        inferred = infer_entity_type_from_relationship(
            entity_name="Castle",
            relationship_type="LOCATED_AT",
            role="target",
        )
        assert inferred == "Location"

    def test_infers_item_from_owns_target(self):
        """When target type is unknown, infer Item from OWNS."""
        inferred = infer_entity_type_from_relationship(
            entity_name="Sword",
            relationship_type="OWNS",
            role="target",
        )
        assert inferred == "Item"

    def test_infers_character_from_ally_of_target(self):
        """When target type is unknown, infer Character from ALLY_OF."""
        inferred = infer_entity_type_from_relationship(
            entity_name="Marcus",
            relationship_type="ALLY_OF",
            role="target",
        )
        assert inferred == "Character"

    def test_infers_character_from_ally_of_source(self):
        """When source type is unknown, infer Character from ALLY_OF."""
        inferred = infer_entity_type_from_relationship(
            entity_name="Sarah",
            relationship_type="ALLY_OF",
            role="source",
        )
        assert inferred == "Character"

    def test_returns_none_when_no_inference_possible(self):
        """When inference is not possible, return None."""
        inferred = infer_entity_type_from_relationship(
            entity_name="Something",
            relationship_type="UNKNOWN_REL",
            role="target",
        )
        assert inferred is None

    def test_infers_event_from_happens_before_target(self):
        """When target type is unknown, infer Event from HAPPENS_BEFORE."""
        inferred = infer_entity_type_from_relationship(
            entity_name="Battle",
            relationship_type="HAPPENS_BEFORE",
            role="target",
        )
        assert inferred == "Event"


class TestOtherSemanticRules:
    """Test other strict semantic validation rules."""

    def test_validates_owns_character_to_item(self):
        """Valid: Character → OWNS → Item should be accepted."""
        is_valid, error = validate_relationship_semantics_strict(
            relationship_type="OWNS",
            source_type="Character",
            target_type="Item",
        )
        assert is_valid is True
        assert error is None

    def test_rejects_owns_character_to_character(self):
        """Invalid: Character → OWNS → Character should be rejected."""
        is_valid, error = validate_relationship_semantics_strict(
            relationship_type="OWNS",
            source_type="Character",
            target_type="Character",
        )
        assert is_valid is False
        assert error is not None
        assert "OWNS" in error

    def test_validates_ally_of_character_to_character(self):
        """Valid: Character → ALLY_OF → Character should be accepted."""
        is_valid, error = validate_relationship_semantics_strict(
            relationship_type="ALLY_OF",
            source_type="Character",
            target_type="Character",
        )
        assert is_valid is True
        assert error is None

    def test_rejects_ally_of_character_to_location(self):
        """Invalid: Character → ALLY_OF → Location should be rejected."""
        is_valid, error = validate_relationship_semantics_strict(
            relationship_type="ALLY_OF",
            source_type="Character",
            target_type="Location",
        )
        assert is_valid is False
        assert error is not None

    def test_validates_loves_character_to_anything(self):
        """Valid: Character → LOVES → (anything) should be accepted."""
        is_valid_char, error_char = validate_relationship_semantics_strict(
            relationship_type="LOVES",
            source_type="Character",
            target_type="Character",
        )
        assert is_valid_char is True

        is_valid_item, error_item = validate_relationship_semantics_strict(
            relationship_type="LOVES",
            source_type="Character",
            target_type="Item",
        )
        assert is_valid_item is True

        is_valid_loc, error_loc = validate_relationship_semantics_strict(
            relationship_type="LOVES",
            source_type="Character",
            target_type="Location",
        )
        assert is_valid_loc is True


class TestIntegrationWithExtractedRelationships:
    """Test validation with actual ExtractedRelationship objects."""

    def test_creates_valid_character_location_relationship(self):
        """A valid Character → LOCATED_AT → Location relationship."""
        rel = ExtractedRelationship(
            source_name="Alice",
            source_type="Character",
            target_name="Castle",
            target_type="Location",
            relationship_type="LOCATED_AT",
            description="Alice is at the Castle",
            confidence=1.0,
            chapter=1,
        )

        is_valid, error = validate_relationship_semantics_strict(
            rel.relationship_type,
            rel.source_type,
            rel.target_type,
        )
        assert is_valid is True
        assert error is None

    def test_rejects_invalid_character_character_relationship(self):
        """An invalid Character → LOCATED_AT → Character relationship."""
        rel = ExtractedRelationship(
            source_name="Alice",
            source_type="Character",
            target_name="Bob",
            target_type="Character",
            relationship_type="LOCATED_AT",
            description="Alice is at Bob",
            confidence=1.0,
            chapter=1,
        )

        is_valid, error = validate_relationship_semantics_strict(
            rel.relationship_type,
            rel.source_type,
            rel.target_type,
        )
        assert is_valid is False
        assert error is not None
        assert "Character" in error or "LOCATED_AT" in error
