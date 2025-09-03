"""Relationship constraints for category: accessibility_and_usage_relationships."""
from ..classifications import NodeClassifications
from models.kg_constants import NODE_LABELS

RELATIONSHIP_CONSTRAINTS = {
    "ACCESSIBLE_BY": {
        "valid_subject_types": NodeClassifications.SPATIAL
        | NodeClassifications.INFORMATIONAL
        | NodeClassifications.INANIMATE,
        "valid_object_types": NodeClassifications.SENTIENT
        | NodeClassifications.SPATIAL
        | {"Path", "Role", "Entity"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Accessibility relationship",
        "examples_valid": [
            "Location:Vault | ACCESSIBLE_BY | Character:Keyholder",
            "Document:File | ACCESSIBLE_BY | Role:Admin",
        ],
        "examples_invalid": [],
    },
    "USED_IN": {
        "valid_subject_types": NodeClassifications.INANIMATE
        | NodeClassifications.ABSTRACT
        | NodeClassifications.SYSTEM_ENTITIES,
        "valid_object_types": NodeClassifications.TEMPORAL
        | NodeClassifications.ABSTRACT
        | {"Entity"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Usage in events or contexts",
        "examples_valid": [
            "Artifact:Sword | USED_IN | Event:Battle",
            "System:Magic | USED_IN | Event:Ritual",
        ],
        "examples_invalid": [],
    },
    "TARGETS": {
        "valid_subject_types": NodeClassifications.INFORMATIONAL
        | NodeClassifications.INANIMATE
        | NodeClassifications.ABSTRACT,
        "valid_object_types": NODE_LABELS - {"ValueNode"},  # Can target almost anything
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Targeting or directing toward something",
        "examples_valid": [
            "Document:Report | TARGETS | Location:Area",
            "System:Weapon | TARGETS | Character:Enemy",
        ],
        "examples_invalid": [],
    },
}
