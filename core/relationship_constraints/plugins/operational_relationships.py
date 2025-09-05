"""Relationship constraints for category: operational_relationships."""

from models.kg_constants import NODE_LABELS

from ..classifications import NodeClassifications

RELATIONSHIP_CONSTRAINTS = {
    "EMPLOYS": {
        "valid_subject_types": NodeClassifications.SENTIENT
        | NodeClassifications.ORGANIZATIONAL,
        "valid_object_types": NodeClassifications.SENTIENT,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Employment or hiring relationship",
        "examples_valid": [
            "Organization:Company | EMPLOYS | Character:Worker",
            "Character:Lord | EMPLOYS | Character:Servant",
        ],
        "examples_invalid": [],
    },
    "CONTROLS": {
        "valid_subject_types": NodeClassifications.SENTIENT
        | NodeClassifications.ORGANIZATIONAL
        | NodeClassifications.SYSTEM_ENTITIES,
        "valid_object_types": NodeClassifications.SYSTEM_ENTITIES
        | NodeClassifications.INANIMATE
        | {"WorldElement", "Entity"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Control or management relationship",
        "examples_valid": [
            "Character:Mage | CONTROLS | System:Magic",
            "Organization:Guild | CONTROLS | System:Trade",
        ],
        "examples_invalid": [],
    },
    "REQUIRES": {
        "valid_subject_types": NODE_LABELS,  # Almost anything can have requirements
        "valid_object_types": NODE_LABELS,  # Almost anything can be required
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Dependency or requirement relationship",
        "examples_valid": [
            "System:Magic | REQUIRES | Resource:Mana",
            "Character:Hero | REQUIRES | Item:Sword",
        ],
        "examples_invalid": [],
    },
}
