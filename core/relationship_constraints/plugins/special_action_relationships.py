"""Relationship constraints for category: special_action_relationships."""
from models.kg_constants import NODE_LABELS

from ..classifications import NodeClassifications

RELATIONSHIP_CONSTRAINTS = {
    "WHISPERS": {
        "valid_subject_types": NodeClassifications.SENTIENT
        | NodeClassifications.SYSTEM_ENTITIES,
        "valid_object_types": NodeClassifications.INFORMATIONAL
        | {"Message", "ValueNode"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Quiet communication or subtle emission",
        "examples_valid": [
            "Character:Ghost | WHISPERS | Message:Secret",
            "System:Wind | WHISPERS | Message:Warning",
        ],
        "examples_invalid": [],
    },
    "WORE": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.INANIMATE | {"Item", "Object"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Past wearing or carrying relationship",
        "examples_valid": ["Character:King | WORE | Item:Crown"],
        "examples_invalid": [],
    },
    "DEPRECATED": {
        "valid_subject_types": NodeClassifications.SYSTEM_ENTITIES
        | NodeClassifications.ORGANIZATIONAL,
        "valid_object_types": NODE_LABELS - {"Entity"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Marked as obsolete or no longer recommended",
        "examples_valid": ["System:Database | DEPRECATED | Document:OldSchema"],
        "examples_invalid": [],
    },
}
