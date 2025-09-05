"""Relationship constraints for category: cognitive_mental_relationships."""

from models.kg_constants import NODE_LABELS

from ..classifications import NodeClassifications

RELATIONSHIP_CONSTRAINTS = {
    "BELIEVES": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.ABSTRACT
        | NodeClassifications.SENTIENT,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Belief relationship from sentient beings",
        "examples_valid": [
            "Character:Priest | BELIEVES | Lore:Prophecy",
            "Character:Student | BELIEVES | Character:Teacher",
        ],
        "examples_invalid": ["WorldElement:Book | BELIEVES | Lore:Truth"],
    },
    "REALIZES": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.ABSTRACT | {"PlotPoint"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Realization or understanding",
        "examples_valid": [
            "Character:Detective | REALIZES | PlotPoint:Solution",
            "Character:Hero | REALIZES | Trait:Truth",
        ],
        "examples_invalid": ["Location:Library | REALIZES | Lore:Knowledge"],
    },
    "REMEMBERS": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NODE_LABELS - {"Entity"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Memory relationship from sentient beings",
        "examples_valid": [
            "Character:Veteran | REMEMBERS | Character:Friend",
            "Character:Elder | REMEMBERS | Location:Hometown",
        ],
        "examples_invalid": ["WorldElement:Diary | REMEMBERS | Character:Writer"],
    },
    "UNDERSTANDS": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.ABSTRACT
        | NodeClassifications.INANIMATE,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Understanding relationship",
        "examples_valid": [
            "Character:Scholar | UNDERSTANDS | System:Magic",
            "Character:Craftsman | UNDERSTANDS | WorldElement:Tool",
        ],
        "examples_invalid": ["System:Magic | UNDERSTANDS | Character:Wizard"],
    },
}
