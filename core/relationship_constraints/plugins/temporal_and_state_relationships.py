"""Relationship constraints for category: temporal_and_state_relationships."""
from ..classifications import NodeClassifications
from models.kg_constants import NODE_LABELS

RELATIONSHIP_CONSTRAINTS = {
    "REPLACED_BY": {
        "valid_subject_types": NODE_LABELS,
        "valid_object_types": NODE_LABELS,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Replacement or succession relationship",
        "examples_valid": [
            "System:OldMagic | REPLACED_BY | System:NewMagic",
            "Character:OldKing | REPLACED_BY | Character:NewKing",
        ],
        "examples_invalid": [],
    },
    "LINKED_TO": {
        "valid_subject_types": NODE_LABELS,
        "valid_object_types": NODE_LABELS,
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Connection or linkage relationship",
        "examples_valid": [
            "System:Network | LINKED_TO | System:Database",
            "Location:Portal | LINKED_TO | Location:Destination",
        ],
        "examples_invalid": [],
    },
}
