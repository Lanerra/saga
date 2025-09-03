"""Relationship constraints for category: information_and_recording_relationships."""

from ..classifications import NodeClassifications

RELATIONSHIP_CONSTRAINTS = {
    "RECORDS": {
        "valid_subject_types": NodeClassifications.SENTIENT
        | NodeClassifications.SYSTEM_ENTITIES
        | NodeClassifications.INFORMATIONAL,
        "valid_object_types": NodeClassifications.INFORMATIONAL
        | NodeClassifications.ABSTRACT
        | {"ValueNode", "Entity"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Recording or documenting information",
        "examples_valid": [
            "Character:Scribe | RECORDS | Message:Event",
            "System:Database | RECORDS | Knowledge:Data",
        ],
        "examples_invalid": [],
    },
    "PRESERVES": {
        "valid_subject_types": NodeClassifications.CONTAINERS
        | NodeClassifications.SPATIAL
        | NodeClassifications.ORGANIZATIONAL,
        "valid_object_types": NodeClassifications.INFORMATIONAL
        | NodeClassifications.INANIMATE
        | NodeClassifications.ABSTRACT,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Preservation or archival relationship",
        "examples_valid": [
            "Archive:Library | PRESERVES | Document:Scroll",
            "Organization:Museum | PRESERVES | Artifact:Relic",
        ],
        "examples_invalid": [],
    },
    "HAS_METADATA": {
        "valid_subject_types": NodeClassifications.INFORMATIONAL
        | NodeClassifications.INANIMATE
        | NodeClassifications.SYSTEM_ENTITIES,
        "valid_object_types": NodeClassifications.INFORMATIONAL
        | NodeClassifications.ABSTRACT
        | {"ValueNode", "Attribute"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Contains metadata or descriptive information",
        "examples_valid": [
            "Document:File | HAS_METADATA | Attribute:CreationDate",
            "System:Database | HAS_METADATA | ValueNode:Schema",
        ],
        "examples_invalid": [],
    },
}
