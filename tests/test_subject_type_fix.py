"""
Test to verify the fix for subject_type being None in validate_triple method.
"""

import pytest
from core.relationship_validator import RelationshipConstraintValidator


def test_validate_triple_with_none_subject_type():
    """Test that validate_triple handles None subject_type gracefully."""
    validator = RelationshipConstraintValidator()
    
    # Test triple with None subject type (simulating LLM parsing issue)
    triple_dict = {
        "subject": {"name": "John", "type": None},  # This would cause the issue
        "predicate": "LOVES",
        "object_entity": {"name": "Mary", "type": "Character"},
        "is_literal_object": False
    }
    
    result = validator.validate_triple(triple_dict)
    
    # Should not crash and should use Entity as fallback
    assert result.is_valid is True  # Should be valid since we're just testing the type handling
    assert result.original_relationship == "LOVES"


def test_validate_triple_with_empty_subject_type():
    """Test that validate_triple handles empty subject_type gracefully."""
    validator = RelationshipConstraintValidator()
    
    # Test triple with empty subject type
    triple_dict = {
        "subject": {"name": "John", "type": ""},  # Empty type
        "predicate": "LOVES",
        "object_entity": {"name": "Mary", "type": "Character"},
        "is_literal_object": False
    }
    
    result = validator.validate_triple(triple_dict)
    
    # Should not crash and should use Entity as fallback
    assert result.is_valid is True


def test_validate_triple_with_missing_subject_type():
    """Test that validate_triple handles missing subject type gracefully."""
    validator = RelationshipConstraintValidator()
    
    # Test triple with missing type key
    triple_dict = {
        "subject": {"name": "John"},  # No type key
        "predicate": "LOVES",
        "object_entity": {"name": "Mary", "type": "Character"},
        "is_literal_object": False
    }
    
    result = validator.validate_triple(triple_dict)
    
    # Should not crash and should use Entity as fallback
    assert result.is_valid is True


if __name__ == "__main__":
    pytest.main([__file__])
