"""Relationship constraints for category: association_relationships."""
from models.kg_constants import NODE_LABELS

RELATIONSHIP_CONSTRAINTS = {
    "ASSOCIATED_WITH": {
        "valid_subject_types": NODE_LABELS,
        "valid_object_types": NODE_LABELS,
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "General association relationship",
        "examples_valid": [
            "Character:Hero | ASSOCIATED_WITH | Faction:Guild",
            "Symbol:Crown | ASSOCIATED_WITH | Concept:Royalty",
        ],
        "examples_invalid": [],
    },
}
