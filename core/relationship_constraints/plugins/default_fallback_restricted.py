"""Relationship constraints for category: default_fallback_restricted."""

RELATIONSHIP_CONSTRAINTS = {
    "RELATES_TO": {
        "valid_subject_types": {
            "Entity"
        },  # Only allow generic Entity types - everything else should use specific relationships
        "valid_object_types": {"Entity"},  # Only allow generic Entity types
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Generic relationship fallback - RESTRICTED to truly ambiguous Entity relationships only",
        "examples_valid": ["Entity:Unknown | RELATES_TO | Entity:Mysterious"],
        "examples_invalid": [
            "Character:Hero | RELATES_TO | PlotPoint:Quest"
        ],  # Should use specific relationships instead
    },
}
