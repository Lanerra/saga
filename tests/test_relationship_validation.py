# tests/test_relationship_validation.py
"""
Tests for the relationship validation system.

This module tests the semantic validation of relationships to ensure
that relationships make sense given the entity types involved.
"""

from core.relationship_validation import (
    RelationshipValidationRule,
    RelationshipValidator,
    get_relationship_validator,
)


class TestRelationshipValidationRule:
    """Tests for RelationshipValidationRule class."""

    def test_rule_validation_pass(self) -> None:
        """Test that a rule validates correctly when constraints are met."""
        rule = RelationshipValidationRule(
            relationship_types={"FRIEND_OF"},
            valid_source_types={"Character", "Person"},
            valid_target_types={"Character", "Person"},
            rule_name="test_friendship",
            rationale="Testing friendship validation",
        )

        is_valid, error = rule.validate("FRIEND_OF", "Character", "Character")
        assert is_valid is True
        assert error is None

    def test_rule_validation_fail_source(self) -> None:
        """Test that a rule fails when source type is invalid."""
        rule = RelationshipValidationRule(
            relationship_types={"FRIEND_OF"},
            valid_source_types={"Character", "Person"},
            valid_target_types={"Character", "Person"},
            rule_name="test_friendship",
            rationale="Testing friendship validation",
        )

        is_valid, error = rule.validate("FRIEND_OF", "Location", "Character")
        assert is_valid is False
        assert error is not None
        assert "Location" in error
        assert "source" in error.lower()

    def test_rule_validation_fail_target(self) -> None:
        """Test that a rule fails when target type is invalid."""
        rule = RelationshipValidationRule(
            relationship_types={"LOCATED_IN"},
            valid_source_types="ANY",
            valid_target_types={"Location", "Structure", "Region"},
            rule_name="test_location",
            rationale="Testing location validation",
        )

        is_valid, error = rule.validate("LOCATED_IN", "Character", "Character")
        assert is_valid is False
        assert error is not None
        assert "Character" in error
        assert "target" in error.lower()

    def test_rule_validation_any_source(self) -> None:
        """Test that a rule with ANY source accepts all source types."""
        rule = RelationshipValidationRule(
            relationship_types={"LOCATED_IN"},
            valid_source_types="ANY",
            valid_target_types={"Location"},
            rule_name="test_location",
            rationale="Testing location validation",
        )

        # Should accept any source type
        is_valid, _ = rule.validate("LOCATED_IN", "Character", "Location")
        assert is_valid is True

        is_valid, _ = rule.validate("LOCATED_IN", "Object", "Location")
        assert is_valid is True

    def test_rule_validation_any_target(self) -> None:
        """Test that a rule with ANY target accepts all target types."""
        rule = RelationshipValidationRule(
            relationship_types={"LOVES"},
            valid_source_types={"Character"},
            valid_target_types="ANY",
            rule_name="test_emotion",
            rationale="Testing emotional validation",
        )

        # Should accept any target type
        is_valid, _ = rule.validate("LOVES", "Character", "Character")
        assert is_valid is True

        is_valid, _ = rule.validate("LOVES", "Character", "Location")
        assert is_valid is True

    def test_rule_ignores_unrelated_relationship(self) -> None:
        """Test that a rule ignores relationship types it doesn't handle."""
        rule = RelationshipValidationRule(
            relationship_types={"FRIEND_OF"},
            valid_source_types={"Character"},
            valid_target_types={"Character"},
            rule_name="test_friendship",
            rationale="Testing friendship validation",
        )

        # Should ignore LOVES relationship
        is_valid, error = rule.validate("LOVES", "Location", "Location")
        assert is_valid is True
        assert error is None


class TestRelationshipValidator:
    """Tests for RelationshipValidator class."""

    def test_validate_known_relationship_type(self) -> None:
        """Test validation of a known relationship type."""
        validator = RelationshipValidator(enable_strict_mode=True)
        is_valid, warning = validator.validate_relationship_type("FRIEND_OF")
        assert is_valid is True
        assert warning is None

    def test_validate_unknown_relationship_type(self) -> None:
        """Test validation of an unknown relationship type."""
        validator = RelationshipValidator(enable_strict_mode=True)
        is_valid, warning = validator.validate_relationship_type("UNKNOWN_TYPE")
        assert is_valid is False
        assert warning is not None
        assert "Unknown relationship type" in warning

    def test_validate_known_entity_types(self) -> None:
        """Test validation of known entity types."""
        validator = RelationshipValidator(enable_strict_mode=True)
        all_valid, warnings = validator.validate_entity_types("Character", "Location")
        assert all_valid is True
        assert len(warnings) == 0

    def test_validate_unknown_source_type(self) -> None:
        """Test validation with unknown source entity type."""
        validator = RelationshipValidator(enable_strict_mode=True)
        all_valid, warnings = validator.validate_entity_types("UnknownType", "Location")
        assert all_valid is False
        assert len(warnings) == 1
        assert "UnknownType" in warnings[0]

    def test_validate_unknown_target_type(self) -> None:
        """Test validation with unknown target entity type."""
        validator = RelationshipValidator(enable_strict_mode=True)
        all_valid, warnings = validator.validate_entity_types("Character", "UnknownType")
        assert all_valid is False
        assert len(warnings) == 1
        assert "UnknownType" in warnings[0]

    def test_validate_social_relationship_valid(self) -> None:
        """Test validation of a valid social relationship."""
        validator = RelationshipValidator(enable_strict_mode=True)
        is_valid, errors, warnings = validator.validate(
            relationship_type="FRIEND_OF",
            source_name="Alice",
            source_type="Character",
            target_name="Bob",
            target_type="Character",
            severity_mode="flexible",
        )
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_social_relationship_invalid(self) -> None:
        """Test validation of an invalid social relationship."""
        validator = RelationshipValidator(enable_strict_mode=True)
        is_valid, errors, warnings = validator.validate(
            relationship_type="FRIEND_OF",
            source_name="Alice",
            source_type="Character",
            target_name="The Tower",
            target_type="Structure",
            severity_mode="flexible",
        )
        assert is_valid is False
        assert len(errors) > 0
        assert "Structure" in errors[0]

    def test_validate_emotional_relationship_valid(self) -> None:
        """Test validation of a valid emotional relationship."""
        validator = RelationshipValidator(enable_strict_mode=True)
        is_valid, errors, warnings = validator.validate(
            relationship_type="LOVES",
            source_name="Alice",
            source_type="Character",
            target_name="Bob",
            target_type="Character",
            severity_mode="flexible",
        )
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_emotional_relationship_valid_any_target(self) -> None:
        """Test that emotional relationships can target any entity type."""
        validator = RelationshipValidator(enable_strict_mode=True)
        # Characters can love objects, places, etc.
        is_valid, errors, warnings = validator.validate(
            relationship_type="LOVES",
            source_name="Alice",
            source_type="Character",
            target_name="The Sword",
            target_type="Artifact",
            severity_mode="flexible",
        )
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_spatial_relationship_valid(self) -> None:
        """Test validation of a valid spatial relationship."""
        validator = RelationshipValidator(enable_strict_mode=True)
        is_valid, errors, warnings = validator.validate(
            relationship_type="LOCATED_IN",
            source_name="Alice",
            source_type="Character",
            target_name="The Castle",
            target_type="Structure",
            severity_mode="flexible",
        )
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_organizational_membership_valid(self) -> None:
        """Subtype entity types are warnings (not errors) in strict+flexible mode."""
        validator = RelationshipValidator(enable_strict_mode=True)
        is_valid, errors, warnings = validator.validate(
            relationship_type="MEMBER_OF",
            source_name="Alice",
            source_type="Character",
            target_name="The Guild",
            target_type="Guild",
            severity_mode="flexible",
        )
        assert is_valid is True
        assert errors == []
        assert len(warnings) > 0
        assert any("Unknown relationship type 'MEMBER_OF'" in warning for warning in warnings)
        assert any("Unknown entity type 'Guild'" in warning for warning in warnings)

    def test_validate_organizational_membership_invalid(self) -> None:
        """Non-canonical entity types still warn (not fail) in strict+flexible mode."""
        validator = RelationshipValidator(enable_strict_mode=True)
        is_valid, errors, warnings = validator.validate(
            relationship_type="MEMBER_OF",
            source_name="The Sword",
            source_type="Artifact",
            target_name="The Guild",
            target_type="Guild",
            severity_mode="flexible",
        )
        assert is_valid is True
        assert errors == []
        assert len(warnings) > 0
        assert any("Unknown relationship type 'MEMBER_OF'" in warning for warning in warnings)
        assert any("Unknown entity type 'Artifact'" in warning for warning in warnings)
        assert any("Unknown entity type 'Guild'" in warning for warning in warnings)

    def test_validate_ownership_valid(self) -> None:
        """Test validation of a valid ownership relationship."""
        validator = RelationshipValidator(enable_strict_mode=True)
        is_valid, errors, warnings = validator.validate(
            relationship_type="OWNS",
            source_name="Alice",
            source_type="Character",
            target_name="The Sword",
            target_type="Artifact",
            severity_mode="flexible",
        )
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_ownership_invalid(self) -> None:
        """Test validation of an invalid ownership relationship."""
        validator = RelationshipValidator(enable_strict_mode=True)
        # Abstract concepts can't own physical objects
        is_valid, errors, warnings = validator.validate(
            relationship_type="OWNS",
            source_name="Justice",
            source_type="Concept",
            target_name="The Sword",
            target_type="Artifact",
            severity_mode="flexible",
        )
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_temporal_relationship_valid(self) -> None:
        """Test validation of a valid temporal relationship."""
        validator = RelationshipValidator(enable_strict_mode=True)
        is_valid, errors, warnings = validator.validate(
            relationship_type="HAPPENS_BEFORE",
            source_name="The Battle",
            source_type="Event",
            target_name="The Treaty",
            target_type="Event",
            severity_mode="flexible",
        )
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_temporal_relationship_invalid(self) -> None:
        """Test validation of an invalid temporal relationship."""
        validator = RelationshipValidator(enable_strict_mode=True)
        # Characters can't happen before/after things
        is_valid, errors, warnings = validator.validate(
            relationship_type="HAPPENS_BEFORE",
            source_name="Alice",
            source_type="Character",
            target_name="Bob",
            target_type="Character",
            severity_mode="flexible",
        )
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_ability_trait_valid(self) -> None:
        """Test validation of a valid ability/trait relationship."""
        validator = RelationshipValidator(enable_strict_mode=True)
        is_valid, errors, warnings = validator.validate(
            relationship_type="HAS_TRAIT",
            source_name="Alice",
            source_type="Character",
            target_name="Courage",
            target_type="Trait",
            severity_mode="flexible",
        )
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_ability_trait_invalid(self) -> None:
        """Test validation of an invalid ability/trait relationship."""
        validator = RelationshipValidator(enable_strict_mode=True)
        is_valid, errors, warnings = validator.validate(
            relationship_type="HAS_TRAIT",
            source_name="The Tower",
            source_type="Structure",
            target_name="Courage",
            target_type="Trait",
            severity_mode="strict",
        )
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_flexible_mode_unknown_type(self) -> None:
        """Test that flexible mode allows unknown types with warnings."""
        validator = RelationshipValidator(enable_strict_mode=True)
        is_valid, errors, warnings = validator.validate(
            relationship_type="CUSTOM_REL",
            source_name="Alice",
            source_type="Character",
            target_name="Bob",
            target_type="Character",
            severity_mode="flexible",
        )
        # Should be valid in flexible mode, but with warnings
        assert is_valid is True
        assert len(errors) == 0
        assert len(warnings) > 0

    def test_validate_strict_mode_unknown_type(self) -> None:
        """Test that strict mode rejects unknown types."""
        validator = RelationshipValidator(enable_strict_mode=True)
        is_valid, errors, warnings = validator.validate(
            relationship_type="CUSTOM_REL",
            source_name="Alice",
            source_type="Character",
            target_name="Bob",
            target_type="Character",
            severity_mode="strict",
        )
        # Should be invalid in strict mode
        assert is_valid is False
        assert len(errors) > 0


class TestGetRelationshipValidator:
    """Tests for get_relationship_validator function."""

    def test_get_validator_singleton(self) -> None:
        """Test that get_relationship_validator returns the same instance."""
        validator1 = get_relationship_validator()
        validator2 = get_relationship_validator()
        assert validator1 is validator2

    def test_get_validator_returns_validator(self) -> None:
        """Test that get_relationship_validator returns a RelationshipValidator."""
        validator = get_relationship_validator()
        assert isinstance(validator, RelationshipValidator)


class TestComplexValidationScenarios:
    """Tests for complex real-world validation scenarios."""

    def test_character_in_multiple_organizations(self) -> None:
        """Test that a character can be in multiple organizations."""
        validator = RelationshipValidator(enable_strict_mode=True)

        # Alice is a member of The Guild
        is_valid1, _, _ = validator.validate("MEMBER_OF", "Alice", "Character", "The Guild", "Guild", "flexible")
        # Alice is also a member of The Council
        is_valid2, _, _ = validator.validate("MEMBER_OF", "Alice", "Character", "The Council", "Council", "flexible")

        assert is_valid1 is True
        assert is_valid2 is True

    def test_character_owns_artifact_located_in_structure(self) -> None:
        """Test a chain of valid relationships."""
        validator = RelationshipValidator(enable_strict_mode=True)

        # Alice owns The Sword
        is_valid1, _, _ = validator.validate("OWNS", "Alice", "Character", "The Sword", "Artifact", "flexible")
        # The Sword is located in The Castle
        is_valid2, _, _ = validator.validate("LOCATED_IN", "The Sword", "Artifact", "The Castle", "Structure", "flexible")

        assert is_valid1 is True
        assert is_valid2 is True

    def test_deity_worshipped_by_culture(self) -> None:
        """Test that deities can have organizational/cultural relationships."""
        validator = RelationshipValidator(enable_strict_mode=True)

        # The Sun God is worshipped (using represents/symbolizes)
        is_valid, _, _ = validator.validate("SYMBOLIZES", "The Sun God", "Deity", "Power", "Concept", "flexible")

        assert is_valid is True

    def test_event_chain(self) -> None:
        """Test a sequence of temporal events."""
        validator = RelationshipValidator(enable_strict_mode=True)

        # The Battle happens before The Treaty
        is_valid1, _, _ = validator.validate("HAPPENS_BEFORE", "The Battle", "Event", "The Treaty", "Event", "flexible")
        # The Treaty happens before The Coronation
        is_valid2, _, _ = validator.validate(
            "HAPPENS_BEFORE",
            "The Treaty",
            "Event",
            "The Coronation",
            "Event",
            "flexible",
        )

        assert is_valid1 is True
        assert is_valid2 is True

    def test_creature_as_character(self) -> None:
        """Test that sentient creatures can have social relationships."""
        validator = RelationshipValidator(enable_strict_mode=True)

        # Dragon (Creature) can be friends with Alice (Character)
        is_valid, _, _ = validator.validate("FRIEND_OF", "Dragon", "Creature", "Alice", "Character", "flexible")

        assert is_valid is True

    def test_spirit_emotional_relationships(self) -> None:
        """Test that spirits can have emotional relationships."""
        validator = RelationshipValidator(enable_strict_mode=True)

        # Ghost (Spirit) can fear Alice (Character)
        is_valid, _, _ = validator.validate("FEARS", "Ghost", "Spirit", "Alice", "Character", "flexible")

        assert is_valid is True
