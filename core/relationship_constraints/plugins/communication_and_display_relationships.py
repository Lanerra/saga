"""Relationship constraints for category: communication_and_display_relationships."""

from ..classifications import NodeClassifications

RELATIONSHIP_CONSTRAINTS = {
    "DISPLAYS": {
        "valid_subject_types": NodeClassifications.INANIMATE
        | NodeClassifications.SYSTEM_ENTITIES
        | NodeClassifications.SPATIAL
        | {"WorldElement", "Entity"},
        "valid_object_types": NodeClassifications.INFORMATIONAL
        | NodeClassifications.ABSTRACT
        | {"ValueNode", "Entity"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Display or presentation of information",
        "examples_valid": [
            "System:Screen | DISPLAYS | Message:Alert",
            "Document:Map | DISPLAYS | Location:Territory",
        ],
        "examples_invalid": [],
    },
    "SPOKEN_BY": {
        "valid_subject_types": NodeClassifications.INFORMATIONAL
        | {"Message", "ValueNode"},
        "valid_object_types": NodeClassifications.SENTIENT,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Communication originating from sentient beings",
        "examples_valid": [
            "Message:Warning | SPOKEN_BY | Character:Guard",
            "ValueNode:Word | SPOKEN_BY | Character:Mage",
        ],
        "examples_invalid": [],
    },
    "EMITS": {
        "valid_subject_types": NodeClassifications.INANIMATE
        | NodeClassifications.SYSTEM_ENTITIES
        | {"WorldElement", "Entity"},
        "valid_object_types": NodeClassifications.ABSTRACT
        | NodeClassifications.INFORMATIONAL
        | {"ValueNode", "Entity"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Emission of energy, sound, or information",
        "examples_valid": [
            "WorldElement:Crystal | EMITS | ValueNode:Light",
            "System:Radio | EMITS | Message:Signal",
        ],
        "examples_invalid": [],
    },
}
