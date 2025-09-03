"""Relationship constraints for category: abstract_trait_relationships."""

from ..classifications import NodeClassifications

RELATIONSHIP_CONSTRAINTS = {
    "HAS_TRAIT": {
        "valid_subject_types": NodeClassifications.SENTIENT
        | NodeClassifications.INANIMATE,
        "valid_object_types": {"Trait"} | NodeClassifications.ABSTRACT,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Trait or characteristic relationship",
        "examples_valid": [
            "Character:Hero | HAS_TRAIT | Trait:Brave",
            "WorldElement:Sword | HAS_TRAIT | Trait:Sharp",
        ],
        "examples_invalid": ["Trait:Courage | HAS_TRAIT | Character:Hero"],
    },
    "HAS_VOICE": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.INANIMATE
        | NodeClassifications.ABSTRACT,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Voice or communication channel relationship",
        "examples_valid": [
            "Character:Oracle | HAS_VOICE | WorldElement:Crystal",
            "Character:AI | HAS_VOICE | System:Network",
        ],
        "examples_invalid": ["WorldElement:Stone | HAS_VOICE | Character:Hero"],
    },
    "SYMBOLIZES": {
        "valid_subject_types": NodeClassifications.INANIMATE
        | NodeClassifications.SPATIAL,
        "valid_object_types": NodeClassifications.ABSTRACT | {"Trait"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Symbolic representation",
        "examples_valid": [
            "WorldElement:Crown | SYMBOLIZES | Trait:Authority",
            "Location:Temple | SYMBOLIZES | Lore:Faith",
        ],
        "examples_invalid": ["Character:King | SYMBOLIZES | WorldElement:Crown"],
    },
}
