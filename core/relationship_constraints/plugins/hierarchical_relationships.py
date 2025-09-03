"""Relationship constraints for category: hierarchical_relationships."""

from ..classifications import NodeClassifications

RELATIONSHIP_CONSTRAINTS = {
    "MENTOR_TO": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.SENTIENT,
        "invalid_combinations": [],
        "bidirectional": False,  # Mentorship has clear direction
        "description": "Teaching/guidance relationship between sentient beings",
        "examples_valid": ["Character:Master | MENTOR_TO | Character:Apprentice"],
        "examples_invalid": ["WorldElement:Book | MENTOR_TO | Character:Student"],
    },
    "STUDENT_OF": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.SENTIENT,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Learning relationship between sentient beings",
        "examples_valid": ["Character:Apprentice | STUDENT_OF | Character:Master"],
        "examples_invalid": ["Character:Student | STUDENT_OF | WorldElement:TextBook"],
    },
    "LEADS": {
        "valid_subject_types": NodeClassifications.SENTIENT
        | NodeClassifications.ORGANIZATIONAL,
        "valid_object_types": NodeClassifications.SENTIENT
        | NodeClassifications.ORGANIZATIONAL,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Leadership relationship",
        "examples_valid": [
            "Character:Captain | LEADS | Character:Soldier",
            "Character:King | LEADS | Faction:Army",
        ],
        "examples_invalid": ["WorldElement:Crown | LEADS | Character:King"],
    },
    "WORKS_FOR": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.SENTIENT
        | NodeClassifications.ORGANIZATIONAL,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Employment or service relationship",
        "examples_valid": [
            "Character:Guard | WORKS_FOR | Character:Lord",
            "Character:Merchant | WORKS_FOR | Faction:Guild",
        ],
        "examples_invalid": ["WorldElement:Tool | WORKS_FOR | Character:Craftsman"],
    },
    "SERVES": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.SENTIENT
        | NodeClassifications.ORGANIZATIONAL
        | NodeClassifications.ABSTRACT,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Service or allegiance relationship",
        "examples_valid": [
            "Character:Knight | SERVES | Character:King",
            "Character:Priest | SERVES | Lore:God",
        ],
        "examples_invalid": ["Location:Temple | SERVES | Lore:God"],
    },
}
