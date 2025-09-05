"""Relationship constraints for category: physical_relationships."""

from models.kg_constants import NODE_LABELS

from ..classifications import NodeClassifications

RELATIONSHIP_CONSTRAINTS = {
    "PART_OF": {
        "valid_subject_types": NodeClassifications.PHYSICAL_PRESENCE,
        "valid_object_types": NodeClassifications.PHYSICAL_PRESENCE,
        "invalid_combinations": [
            ("Character", "WorldElement"),  # Characters aren't parts of objects
            ("Character", "Location"),  # Characters aren't parts of places
        ],
        "bidirectional": False,
        "description": "Component relationship",
        "examples_valid": [
            "WorldElement:Blade | PART_OF | WorldElement:Sword",
            "Location:Tower | PART_OF | Location:Castle",
        ],
        "examples_invalid": ["Character:Guard | PART_OF | Location:Castle"],
    },
    "CONNECTED_TO": {
        "valid_subject_types": NodeClassifications.PHYSICAL_PRESENCE,
        "valid_object_types": NodeClassifications.PHYSICAL_PRESENCE,
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Physical connection",
        "examples_valid": [
            "Location:Bridge | CONNECTED_TO | Location:Shore",
            "WorldElement:Chain | CONNECTED_TO | WorldElement:Anchor",
        ],
        "examples_invalid": ["Trait:Loyalty | CONNECTED_TO | Character:Knight"],
    },
    "CONNECTS_TO": {
        "valid_subject_types": NodeClassifications.PHYSICAL_PRESENCE,
        "valid_object_types": NodeClassifications.PHYSICAL_PRESENCE,
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Physical or functional connection (alias for CONNECTED_TO)",
        "examples_valid": [
            "WorldElement:USB_Drive | CONNECTS_TO | Character:User",
            "Location:Tunnel | CONNECTS_TO | Location:Cave",
        ],
        "examples_invalid": ["Trait:Courage | CONNECTS_TO | Character:Hero"],
    },
    "PULSES_WITH": {
        "valid_subject_types": NodeClassifications.PHYSICAL_PRESENCE,
        "valid_object_types": NodeClassifications.ABSTRACT
        | {"ValueNode"},  # Can pulse with emotions, energies, etc.
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Rhythmic emanation or resonance",
        "examples_valid": [
            "WorldElement:Crystal | PULSES_WITH | ValueNode:Energy",
            "WorldElement:Heart | PULSES_WITH | Trait:Life",
        ],
        "examples_invalid": ["Character:Hero | PULSES_WITH | Character:Villain"],
    },
    "RESPONDS_TO": {
        "valid_subject_types": NodeClassifications.PHYSICAL_PRESENCE
        | NodeClassifications.ABSTRACT,
        "valid_object_types": NODE_LABELS - {"Entity"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Reactive relationship - responds to stimuli or inputs",
        "examples_valid": [
            "WorldElement:Sensor | RESPONDS_TO | Character:Movement",
            "System:Security | RESPONDS_TO | WorldElement:Trigger",
        ],
        "examples_invalid": [],
    },
}
