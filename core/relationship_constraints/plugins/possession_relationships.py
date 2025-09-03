"""Relationship constraints for category: possession_relationships."""

from ..classifications import NodeClassifications

RELATIONSHIP_CONSTRAINTS = {
    "OWNS": {
        "valid_subject_types": NodeClassifications.SENTIENT
        | NodeClassifications.ORGANIZATIONAL,
        "valid_object_types": NodeClassifications.OWNABLE | NodeClassifications.SPATIAL,
        "invalid_combinations": [
            # Characters can't own other characters (slavery check)
            ("Character", "Character"),
        ],
        "bidirectional": False,
        "description": "Ownership relationship",
        "examples_valid": [
            "Character:Lord | OWNS | Location:Castle",
            "Character:Wizard | OWNS | WorldElement:Staff",
        ],
        "examples_invalid": [
            "WorldElement:Sword | OWNS | Character:Warrior",
            "Character:Master | OWNS | Character:Servant",
        ],
    },
    "POSSESSES": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.OWNABLE
        | NodeClassifications.ABSTRACT,
        "invalid_combinations": [
            ("Character", "Character"),  # Can't possess people
        ],
        "bidirectional": False,
        "description": "Physical or metaphorical possession",
        "examples_valid": [
            "Character:Mage | POSSESSES | WorldElement:Crystal",
            "Character:Hero | POSSESSES | Trait:Courage",
        ],
        "examples_invalid": ["WorldElement:Ring | POSSESSES | Character:Wearer"],
    },
    "CREATED_BY": {
        "valid_subject_types": NodeClassifications.INANIMATE
        | NodeClassifications.SPATIAL
        | NodeClassifications.ABSTRACT
        | NodeClassifications.INFORMATIONAL
        | NodeClassifications.SYSTEM_ENTITIES,
        "valid_object_types": NodeClassifications.SENTIENT
        | NodeClassifications.ORGANIZATIONAL
        | NodeClassifications.INANIMATE
        | NodeClassifications.SYSTEM_ENTITIES,  # Things can be created by other things, including systems
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Creation relationship - expanded to include documents, systems, and information",
        "examples_valid": [
            "WorldElement:Sword | CREATED_BY | Character:Smith",
            "Document:Report | CREATED_BY | Character:Analyst",
            "System:Database | CREATED_BY | Organization:TechGuild",
            "Artifact:Relic | CREATED_BY | Character:Mage",
        ],
        "examples_invalid": ["Character:Hero | CREATED_BY | WorldElement:Potion"],
    },
    "CREATES": {
        "valid_subject_types": NodeClassifications.SENTIENT
        | NodeClassifications.ORGANIZATIONAL
        | NodeClassifications.INANIMATE
        | NodeClassifications.SYSTEM_ENTITIES,  # Inverse of CREATED_BY
        "valid_object_types": NodeClassifications.INANIMATE
        | NodeClassifications.SPATIAL
        | NodeClassifications.ABSTRACT
        | NodeClassifications.INFORMATIONAL
        | NodeClassifications.SYSTEM_ENTITIES,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Active creation relationship - inverse of CREATED_BY",
        "examples_valid": [
            "Character:Smith | CREATES | WorldElement:Sword",
            "Character:Analyst | CREATES | Document:Report",
            "Organization:TechGuild | CREATES | System:Database",
        ],
        "examples_invalid": ["WorldElement:Potion | CREATES | Character:Hero"],
    },
}
