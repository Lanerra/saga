# tests/test_relationship_constraints.py
"""
Comprehensive test suite for the relationship constraint validation system.

Tests cover node type compatibility, semantic validation rules, and integration
with the knowledge graph pipeline.
"""

import pytest
from typing import Dict, List, Any

from core.relationship_constraints import (
    validate_relationship_semantics,
    get_relationship_suggestions,
    get_all_valid_relationships_for_node_pair,
    is_relationship_valid,
    get_node_classifications,
    NodeClassifications,
    RELATIONSHIP_CONSTRAINTS,
)
from core.relationship_validator import (
    RelationshipConstraintValidator,
    ValidationResult,
    validate_relationship_constraint,
    validate_triple_constraint,
    enhance_triple_with_validation,
    should_accept_relationship,
)
import config


class TestNodeClassifications:
    """Test node type classification system."""
    
    def test_character_classifications(self):
        """Characters should have sentient, conscious, and social traits."""
        classifications = get_node_classifications("Character")
        assert "SENTIENT" in classifications
        assert "CONSCIOUS" in classifications
        assert "SOCIAL" in classifications
        assert "PHYSICAL_PRESENCE" in classifications
        assert "LOCATABLE" in classifications
    
    def test_worldelement_classifications(self):
        """WorldElements should be inanimate and ownable."""
        classifications = get_node_classifications("WorldElement")
        assert "INANIMATE" in classifications
        assert "OWNABLE" in classifications
        assert "PHYSICAL_PRESENCE" in classifications
        assert "LOCATABLE" in classifications
        # Should not be conscious or social
        assert "CONSCIOUS" not in classifications
        assert "SOCIAL" not in classifications
    
    def test_location_classifications(self):
        """Locations should be spatial and have physical presence."""
        classifications = get_node_classifications("Location")
        assert "SPATIAL" in classifications
        assert "PHYSICAL_PRESENCE" in classifications
        # Should not be sentient or ownable
        assert "SENTIENT" not in classifications
        assert "OWNABLE" not in classifications
    
    def test_trait_classifications(self):
        """Traits should be abstract."""
        classifications = get_node_classifications("Trait")
        assert "ABSTRACT" in classifications
        # Should not have physical presence
        assert "PHYSICAL_PRESENCE" not in classifications


class TestRelationshipSemantics:
    """Test semantic validation of relationships."""
    
    def test_valid_emotional_relationships(self):
        """Test valid emotional relationships between characters."""
        is_valid, errors = validate_relationship_semantics("Character", "LOVES", "Character")
        assert is_valid
        assert len(errors) == 0
        
        is_valid, errors = validate_relationship_semantics("Character", "FEARS", "Character")
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_emotional_relationships(self):
        """Test invalid emotional relationships from inanimate objects."""
        is_valid, errors = validate_relationship_semantics("WorldElement", "LOVES", "Character")
        assert not is_valid
        assert any("Invalid subject type" in error for error in errors)
        
        is_valid, errors = validate_relationship_semantics("Location", "HATES", "Character")
        assert not is_valid
        assert any("Invalid subject type" in error for error in errors)
    
    def test_valid_spatial_relationships(self):
        """Test valid spatial relationships."""
        is_valid, errors = validate_relationship_semantics("Character", "LOCATED_IN", "Location")
        assert is_valid
        assert len(errors) == 0
        
        is_valid, errors = validate_relationship_semantics("WorldElement", "LOCATED_AT", "Location")
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_spatial_relationships(self):
        """Test invalid spatial relationships."""
        is_valid, errors = validate_relationship_semantics("Trait", "LOCATED_IN", "Location")
        assert not is_valid
        assert any("Invalid subject type" in error for error in errors)
        
        is_valid, errors = validate_relationship_semantics("Character", "LOCATED_IN", "Character")
        assert not is_valid
        assert any("Invalid object type" in error for error in errors)
    
    def test_ownership_constraints(self):
        """Test ownership relationship constraints."""
        # Valid ownership
        is_valid, errors = validate_relationship_semantics("Character", "OWNS", "WorldElement")
        assert is_valid
        
        # Invalid: characters can't own other characters (anti-slavery)
        is_valid, errors = validate_relationship_semantics("Character", "OWNS", "Character")
        assert not is_valid
        assert any("explicitly forbidden" in error for error in errors)
        
        # Invalid: objects can't own things
        is_valid, errors = validate_relationship_semantics("WorldElement", "OWNS", "Character")
        assert not is_valid
    
    def test_social_relationship_constraints(self):
        """Test social relationship constraints."""
        # Valid social relationships
        is_valid, errors = validate_relationship_semantics("Character", "FAMILY_OF", "Character")
        assert is_valid
        
        is_valid, errors = validate_relationship_semantics("Character", "FRIEND_OF", "Character")
        assert is_valid
        
        # Invalid: objects can't have family relationships
        is_valid, errors = validate_relationship_semantics("WorldElement", "FAMILY_OF", "Character")
        assert not is_valid
        
        # Invalid: characters can't be family with objects
        is_valid, errors = validate_relationship_semantics("Character", "FAMILY_OF", "WorldElement")
        assert not is_valid
    
    def test_cognitive_relationship_constraints(self):
        """Test cognitive relationship constraints."""
        # Valid cognitive relationships
        is_valid, errors = validate_relationship_semantics("Character", "BELIEVES", "Trait")
        assert is_valid
        
        is_valid, errors = validate_relationship_semantics("Character", "REMEMBERS", "Location")
        assert is_valid
        
        # Invalid: non-conscious entities can't have cognitive relationships
        is_valid, errors = validate_relationship_semantics("WorldElement", "BELIEVES", "Trait")
        assert not is_valid
        
        is_valid, errors = validate_relationship_semantics("Location", "REMEMBERS", "Character")
        assert not is_valid


class TestRelationshipSuggestions:
    """Test relationship suggestion system."""
    
    def test_character_to_character_suggestions(self):
        """Test suggestions for character-to-character relationships."""
        suggestions = get_relationship_suggestions("Character", "Character")
        suggestion_types = [rel_type for rel_type, _ in suggestions]
        
        # Should include social and emotional relationships
        assert "FAMILY_OF" in suggestion_types
        assert "FRIEND_OF" in suggestion_types
        assert "LOVES" in suggestion_types
        assert "ENEMY_OF" in suggestion_types
        
        # Should not include spatial relationships
        assert "LOCATED_IN" not in suggestion_types
    
    def test_character_to_location_suggestions(self):
        """Test suggestions for character-to-location relationships."""
        suggestions = get_relationship_suggestions("Character", "Location")
        suggestion_types = [rel_type for rel_type, _ in suggestions]
        
        # Should include spatial relationships
        assert "LOCATED_IN" in suggestion_types
        assert "LOCATED_AT" in suggestion_types
        assert "NEAR" in suggestion_types
        
        # Should not include social relationships
        assert "FAMILY_OF" not in suggestion_types
        assert "FRIEND_OF" not in suggestion_types
    
    def test_character_to_worldelement_suggestions(self):
        """Test suggestions for character-to-object relationships."""
        suggestions = get_relationship_suggestions("Character", "WorldElement")
        suggestion_types = [rel_type for rel_type, _ in suggestions]
        
        # Should include possession relationships
        assert "OWNS" in suggestion_types
        assert "POSSESSES" in suggestion_types
        
        # Should include emotional relationships (can fear/love objects)
        assert "FEARS" in suggestion_types
        
        # Should not include family relationships
        assert "FAMILY_OF" not in suggestion_types


class TestValidationResult:
    """Test ValidationResult class behavior."""
    
    def test_valid_result_creation(self):
        """Test creation of valid validation result."""
        result = ValidationResult(
            is_valid=True,
            original_relationship="LOVE",
            validated_relationship="LOVES",
            confidence=0.9
        )
        assert result.is_valid
        assert result.original_relationship == "LOVE"
        assert result.validated_relationship == "LOVES"
        assert result.confidence == 0.9
        assert len(result.errors) == 0
    
    def test_invalid_result_creation(self):
        """Test creation of invalid validation result."""
        errors = ["Invalid subject type", "Invalid object type"]
        result = ValidationResult(
            is_valid=False,
            original_relationship="MOUNTAIN_LOVES",
            errors=errors
        )
        assert not result.is_valid
        assert result.errors == errors
        assert result.confidence == 1.0  # Default confidence
    
    def test_should_accept_relationship_logic(self):
        """Test relationship acceptance logic based on confidence."""
        # High confidence - should accept
        high_conf_result = ValidationResult(True, "LOVES", confidence=0.9)
        assert should_accept_relationship(high_conf_result, min_confidence=0.3)
        
        # Low confidence but above threshold - should accept
        medium_conf_result = ValidationResult(True, "LOVES", confidence=0.4)
        assert should_accept_relationship(medium_conf_result, min_confidence=0.3)
        
        # Very low confidence below threshold - should reject
        low_conf_result = ValidationResult(True, "RELATES_TO", confidence=0.2)
        assert not should_accept_relationship(low_conf_result, min_confidence=0.3)
        
        # Invalid result - should reject regardless of confidence
        invalid_result = ValidationResult(False, "INVALID", confidence=1.0)
        assert not should_accept_relationship(invalid_result, min_confidence=0.3)


class TestRelationshipValidator:
    """Test the main relationship validator class."""
    
    def test_validator_initialization(self):
        """Test validator initializes with clean stats."""
        validator = RelationshipConstraintValidator()
        stats = validator.get_validation_statistics()
        assert stats["total_validations"] == 0
        assert stats["valid_relationships"] == 0
        assert stats["invalid_relationships"] == 0
    
    def test_valid_relationship_validation(self):
        """Test validation of a valid relationship."""
        validator = RelationshipConstraintValidator()
        result = validator.validate_relationship("Character", "LOVES", "Character")
        
        assert result.is_valid
        assert result.validated_relationship == "LOVES"
        assert result.confidence > 0.8
        
        stats = validator.get_validation_statistics()
        assert stats["total_validations"] == 1
        assert stats["valid_relationships"] == 1
    
    def test_invalid_relationship_correction(self):
        """Test correction of invalid relationships."""
        validator = RelationshipConstraintValidator()
        result = validator.validate_relationship("Character", "FRIEND", "Character")
        
        # Should be corrected to FRIEND_OF
        assert result.is_valid
        assert result.validated_relationship == "FRIEND_OF"
        assert result.original_relationship == "FRIEND"
        assert result.confidence < 1.0  # Lower confidence for corrections
    
    def test_completely_invalid_relationship(self):
        """Test handling of completely invalid relationships."""
        validator = RelationshipConstraintValidator()
        result = validator.validate_relationship("WorldElement", "FAMILY_OF", "Character")
        
        # Should either be invalid or fall back to RELATES_TO
        if not result.is_valid:
            assert len(result.errors) > 0
        else:
            # If valid, should be a fallback
            assert result.validated_relationship in ["RELATES_TO", "ASSOCIATED_WITH"]
            assert result.confidence < 0.5
    
    def test_triple_validation(self):
        """Test validation of complete triples."""
        validator = RelationshipConstraintValidator()
        
        # Valid triple
        valid_triple = {
            "subject": {"name": "Alice", "type": "Character"},
            "predicate": "LOVES",
            "object_entity": {"name": "Bob", "type": "Character"},
            "is_literal_object": False
        }
        
        result = validator.validate_triple(valid_triple)
        assert result.is_valid
        assert result.validated_relationship == "LOVES"
        
        # Invalid triple
        invalid_triple = {
            "subject": {"name": "Sword", "type": "WorldElement"},
            "predicate": "FAMILY_OF",
            "object_entity": {"name": "Alice", "type": "Character"},
            "is_literal_object": False
        }
        
        result = validator.validate_triple(invalid_triple)
        # Should either be invalid or corrected to fallback
        if result.is_valid:
            assert result.confidence < 0.5  # Low confidence fallback
    
    def test_batch_validation(self):
        """Test batch validation of multiple triples."""
        validator = RelationshipConstraintValidator()
        
        triples = [
            {
                "subject": {"name": "Alice", "type": "Character"},
                "predicate": "LOVES",
                "object_entity": {"name": "Bob", "type": "Character"},
                "is_literal_object": False
            },
            {
                "subject": {"name": "Alice", "type": "Character"},
                "predicate": "LOCATED_IN",
                "object_entity": {"name": "Castle", "type": "Location"},
                "is_literal_object": False
            },
            {
                "subject": {"name": "Sword", "type": "WorldElement"},
                "predicate": "FAMILY_OF",  # Invalid
                "object_entity": {"name": "Shield", "type": "WorldElement"},
                "is_literal_object": False
            }
        ]
        
        results = validator.validate_batch(triples)
        assert len(results) == 3
        
        # First two should be valid
        assert results[0].is_valid
        assert results[1].is_valid
        
        # Third should be invalid or low-confidence fallback
        if results[2].is_valid:
            assert results[2].confidence < 0.5


class TestTripleEnhancement:
    """Test enhancement of triples with validation metadata."""
    
    def test_enhance_valid_triple(self):
        """Test enhancement of a valid triple."""
        triple = {
            "subject": {"name": "Alice", "type": "Character"},
            "predicate": "LOVES",
            "object_entity": {"name": "Bob", "type": "Character"},
            "is_literal_object": False
        }
        
        enhanced = enhance_triple_with_validation(triple)
        
        assert "validation" in enhanced
        assert enhanced["validation"]["is_valid"]
        assert enhanced["validation"]["validated_predicate"] == "LOVES"
        assert enhanced["predicate"] == "LOVES"  # Unchanged
    
    def test_enhance_corrected_triple(self):
        """Test enhancement of a triple that gets corrected."""
        triple = {
            "subject": {"name": "Alice", "type": "Character"},
            "predicate": "FRIEND",  # Should be corrected to FRIEND_OF
            "object_entity": {"name": "Bob", "type": "Character"},
            "is_literal_object": False
        }
        
        enhanced = enhance_triple_with_validation(triple)
        
        assert "validation" in enhanced
        if enhanced["validation"]["is_valid"]:
            assert enhanced["validation"]["original_predicate"] == "FRIEND"
            assert enhanced["predicate"] == "FRIEND_OF"  # Should be corrected
    
    def test_enhance_invalid_triple(self):
        """Test enhancement of an invalid triple."""
        triple = {
            "subject": {"name": "Mountain", "type": "Location"},
            "predicate": "LOVES",
            "object_entity": {"name": "River", "type": "Location"},
            "is_literal_object": False
        }
        
        enhanced = enhance_triple_with_validation(triple)
        
        assert "validation" in enhanced
        # Should either be invalid or fallback with low confidence
        if not enhanced["validation"]["is_valid"]:
            assert len(enhanced["validation"]["errors"]) > 0
        else:
            # If accepted as fallback, should have low confidence
            assert enhanced["validation"]["confidence"] < 0.5


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_common_valid_relationships(self):
        """Test validation of commonly used valid relationships."""
        test_cases = [
            ("Character", "LIVES_IN", "Location"),  # Should normalize to LOCATED_IN
            ("Character", "HAS", "WorldElement"),   # Should normalize to OWNS
            ("Character", "KNOWS", "Character"),
            ("WorldElement", "LOCATED_AT", "Location"),
            ("Character", "MEMBER_OF", "Faction"),
        ]
        
        for subject_type, predicate, object_type in test_cases:
            result = validate_relationship_constraint(subject_type, predicate, object_type)
            assert result.is_valid, f"Expected {subject_type} {predicate} {object_type} to be valid"
    
    def test_common_invalid_relationships(self):
        """Test validation catches commonly problematic relationships."""
        test_cases = [
            ("WorldElement", "LOVES", "Character"),    # Objects can't love
            ("Location", "FAMILY_OF", "Character"),    # Places can't have family
            ("Character", "OWNS", "Character"),        # No slavery
            ("Trait", "LOCATED_IN", "Location"),       # Abstracts can't be located
        ]
        
        for subject_type, predicate, object_type in test_cases:
            result = validate_relationship_constraint(subject_type, predicate, object_type)
            # Should either be invalid or very low confidence fallback
            if result.is_valid:
                assert result.confidence < 0.5, f"Expected {subject_type} {predicate} {object_type} to be rejected or low confidence"
    
    def test_narrative_edge_cases(self):
        """Test edge cases that might arise in narrative generation."""
        # Character fearing abstract concepts (should be valid)
        result = validate_relationship_constraint("Character", "FEARS", "Trait")
        assert result.is_valid
        
        # Character believing in lore/mythology (should be valid)
        result = validate_relationship_constraint("Character", "BELIEVES", "Lore")
        assert result.is_valid
        
        # Objects symbolizing traits (should be valid)
        result = validate_relationship_constraint("WorldElement", "SYMBOLIZES", "Trait")
        assert result.is_valid
        
        # Locations containing other locations (should be valid)
        result = validate_relationship_constraint("Location", "CONTAINS", "Location")
        assert result.is_valid


@pytest.fixture
def sample_triples():
    """Fixture providing sample triples for testing."""
    return [
        {
            "subject": {"name": "Hero", "type": "Character"},
            "predicate": "LOVES",
            "object_entity": {"name": "Princess", "type": "Character"},
            "is_literal_object": False
        },
        {
            "subject": {"name": "Dragon", "type": "Character"},
            "predicate": "LOCATED_IN",
            "object_entity": {"name": "Cave", "type": "Location"},
            "is_literal_object": False
        },
        {
            "subject": {"name": "Sword", "type": "WorldElement"},
            "predicate": "OWNED_BY",
            "object_entity": {"name": "Knight", "type": "Character"},
            "is_literal_object": False
        }
    ]


class TestConstraintPerformance:
    """Test performance aspects of constraint validation."""
    
    def test_batch_validation_performance(self, sample_triples):
        """Test that batch validation is reasonably efficient."""
        validator = RelationshipConstraintValidator()
        
        # Create larger batch
        large_batch = sample_triples * 100  # 300 triples
        
        results = validator.validate_batch(large_batch)
        assert len(results) == len(large_batch)
        
        # Verify statistics are tracked correctly
        stats = validator.get_validation_statistics()
        assert stats["total_validations"] == len(large_batch)
    
    def test_constraint_caching_behavior(self):
        """Test that constraint lookups don't degrade performance."""
        validator = RelationshipConstraintValidator()
        
        # Validate the same relationship multiple times
        for _ in range(100):
            result = validator.validate_relationship("Character", "LOVES", "Character")
            assert result.is_valid
        
        stats = validator.get_validation_statistics()
        assert stats["total_validations"] == 100
        assert stats["valid_relationships"] == 100


class TestConfigurationDrivenValidation:
    """Test configuration-driven validation behavior."""
    
    def test_auto_correct_disabled(self):
        """Test that auto-correction can be disabled via config."""
        # Save original config
        original_auto_correct = config.settings.RELATIONSHIP_CONSTRAINT_AUTO_CORRECT
        
        try:
            # Disable auto-correction
            config.settings.RELATIONSHIP_CONSTRAINT_AUTO_CORRECT = False
            validator = RelationshipConstraintValidator()
            
            # Test invalid relationship that would normally be auto-corrected
            # Use a completely invalid relationship for node types (not just normalization)
            result = validator.validate_relationship("Object", "LOVES", "Character")
            
            # Should fall back to RELATES_TO/ASSOCIATED_WITH instead of being auto-corrected
            assert result.validated_relationship in ["RELATES_TO", "ASSOCIATED_WITH"] or not result.is_valid
            
        finally:
            # Restore original config
            config.settings.RELATIONSHIP_CONSTRAINT_AUTO_CORRECT = original_auto_correct
    
    def test_strict_mode_rejection(self):
        """Test that strict mode rejects invalid relationships."""
        # Save original config
        original_strict_mode = config.settings.RELATIONSHIP_CONSTRAINT_STRICT_MODE
        
        try:
            # Enable strict mode
            config.settings.RELATIONSHIP_CONSTRAINT_STRICT_MODE = True
            validator = RelationshipConstraintValidator()
            
            # Test completely invalid relationship
            result = validator.validate_relationship("Object", "LOVES", "Character")
            
            # Should be rejected in strict mode
            assert not result.is_valid
            
        finally:
            # Restore original config
            config.settings.RELATIONSHIP_CONSTRAINT_STRICT_MODE = original_strict_mode
    
    def test_permissive_mode_fallbacks(self):
        """Test that permissive mode uses fallbacks."""
        # Save original config
        original_strict_mode = config.settings.RELATIONSHIP_CONSTRAINT_STRICT_MODE
        
        try:
            # Disable strict mode (permissive)
            config.settings.RELATIONSHIP_CONSTRAINT_STRICT_MODE = False
            validator = RelationshipConstraintValidator()
            
            # Test invalid relationship
            result = validator.validate_relationship("Object", "LOVES", "Character")
            
            # Should use fallback in permissive mode
            assert result.is_valid
            assert result.validated_relationship in ["RELATES_TO", "ASSOCIATED_WITH"]
            assert result.confidence < 0.5
            
        finally:
            # Restore original config
            config.settings.RELATIONSHIP_CONSTRAINT_STRICT_MODE = original_strict_mode
    
    def test_confidence_threshold_acceptance(self):
        """Test that confidence threshold determines acceptance."""
        # Save original config
        original_min_confidence = config.settings.RELATIONSHIP_CONSTRAINT_MIN_CONFIDENCE
        
        try:
            # Set high confidence threshold
            config.settings.RELATIONSHIP_CONSTRAINT_MIN_CONFIDENCE = 0.8
            
            # Create a low-confidence result (like a fallback)
            low_confidence_result = ValidationResult(
                is_valid=True,
                original_relationship="invalid_rel",
                validated_relationship="RELATES_TO",
                confidence=0.3
            )
            
            # Should not be accepted with high threshold
            assert not should_accept_relationship(low_confidence_result)
            
            # Set low confidence threshold
            config.settings.RELATIONSHIP_CONSTRAINT_MIN_CONFIDENCE = 0.1
            
            # Should be accepted with low threshold
            assert should_accept_relationship(low_confidence_result)
            
        finally:
            # Restore original config
            config.settings.RELATIONSHIP_CONSTRAINT_MIN_CONFIDENCE = original_min_confidence


class TestIntegrationValidation:
    """Test validation integration with existing pipeline."""
    
    def test_triple_validation_integration(self):
        """Test that triple validation works with real data structures."""
        # Test triple from knowledge extraction
        triple_dict = {
            "subject": {"name": "Alice", "type": "Character"},
            "predicate": "LOVES",
            "object_entity": {"name": "Bob", "type": "Character"},
            "is_literal_object": False
        }
        
        result = validate_triple_constraint(triple_dict)
        assert result.is_valid
        assert result.validated_relationship == "LOVES"
    
    def test_enhanced_triple_processing(self):
        """Test that triple enhancement adds validation metadata."""
        triple_dict = {
            "subject": {"name": "Sword", "type": "Object"},
            "predicate": "LOVES",  # Invalid - objects can't love
            "object_entity": {"name": "Hero", "type": "Character"},
            "is_literal_object": False
        }
        
        enhanced_triple = enhance_triple_with_validation(triple_dict)
        
        # Should have validation metadata
        assert "validation" in enhanced_triple
        validation = enhanced_triple["validation"]
        
        # Should not be valid or should be corrected
        assert not validation["is_valid"] or validation["validated_predicate"] != "LOVES"
        assert "errors" in validation