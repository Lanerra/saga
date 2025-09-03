"""Relationship constraints for category: status_and_state_change_relationships."""
from ..classifications import NodeClassifications
from models.kg_constants import NODE_LABELS

RELATIONSHIP_CONSTRAINTS = {
    "WAS_REPLACED_BY": {
        "valid_subject_types": NODE_LABELS,
        "valid_object_types": NODE_LABELS,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Past replacement relationship (inverse form)",
        "examples_valid": ["System:OldMagic | WAS_REPLACED_BY | System:NewMagic"],
        "examples_invalid": [],
    },
    "CHARACTERIZED_BY": {
        "valid_subject_types": NodeClassifications.TEMPORAL
        | NodeClassifications.ABSTRACT
        | NodeClassifications.SPATIAL,
        "valid_object_types": NodeClassifications.ABSTRACT
        | NodeClassifications.INFORMATIONAL
        | {"Concept", "Attribute"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Characterized or defined by certain traits",
        "examples_valid": ["Era:Medieval | CHARACTERIZED_BY | Concept:Feudalism"],
        "examples_invalid": [],
    },
    "IS_NOW": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.ORGANIZATIONAL | {"Role", "Status"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Current role or status",
        "examples_valid": ["Character:Hero | IS_NOW | Role:Leader"],
        "examples_invalid": [],
    },
    "IS_NO_LONGER": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.ORGANIZATIONAL | {"Role", "Status"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Former role or status",
        "examples_valid": ["Character:Hero | IS_NO_LONGER | Status:Apprentice"],
        "examples_invalid": [],
    },
    "DIFFERS_FROM": {
        "valid_subject_types": NODE_LABELS,
        "valid_object_types": NODE_LABELS,
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Difference or distinction relationship",
        "examples_valid": ["Document:V1 | DIFFERS_FROM | Document:V2"],
        "examples_invalid": [],
    },
}
