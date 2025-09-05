"""Relationship constraints for category: temporal_causal_relationships."""

from models.kg_constants import NODE_LABELS

from ..classifications import NodeClassifications

RELATIONSHIP_CONSTRAINTS = {
    "CAUSES": {
        "valid_subject_types": NODE_LABELS,  # Allow Entity type - almost anything can cause something
        "valid_object_types": NODE_LABELS,  # Allow Entity type for objects too
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Causal relationship between entities or events",
        "examples_valid": [
            "Character:Hero | CAUSES | PlotPoint:Victory",
            "WorldElement:Potion | CAUSES | Trait:Healing",
            "Entity:Unknown | CAUSES | WorldElement:Effect",
        ],
        "examples_invalid": [],  # Very permissive for narrative flexibility
    },
    "OCCURRED_IN": {
        "valid_subject_types": NodeClassifications.TEMPORAL
        | NodeClassifications.ABSTRACT,
        "valid_object_types": NodeClassifications.TEMPORAL
        | NodeClassifications.SPATIAL
        | {"ValueNode"},  # Can occur at times or places
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Temporal occurrence relationship",
        "examples_valid": [
            "DevelopmentEvent:Battle | OCCURRED_IN | Chapter:Five",
            "WorldElaborationEvent:Ceremony | OCCURRED_IN | Location:Temple",
        ],
        "examples_invalid": [],
    },
    "PREVENTS": {
        "valid_subject_types": NodeClassifications.SENTIENT
        | NodeClassifications.INANIMATE,
        "valid_object_types": NODE_LABELS - {"Entity"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Prevention relationship",
        "examples_valid": [
            "Character:Guard | PREVENTS | PlotPoint:Theft",
            "WorldElement:Ward | PREVENTS | Character:Enemy",
        ],
        "examples_invalid": ["PlotPoint:Quest | PREVENTS | Character:Hero"],
    },
    "ENABLES": {
        "valid_subject_types": NODE_LABELS - {"Entity"},
        "valid_object_types": NODE_LABELS - {"Entity"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Enablement relationship",
        "examples_valid": [
            "WorldElement:Key | ENABLES | PlotPoint:Escape",
            "Character:Mentor | ENABLES | Character:Student",
        ],
        "examples_invalid": [],  # Very permissive
    },
    "TRIGGERS": {
        "valid_subject_types": NODE_LABELS - {"Entity"},
        "valid_object_types": NodeClassifications.TEMPORAL
        | NodeClassifications.ABSTRACT,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Triggering of events or states",
        "examples_valid": [
            "Character:Hero | TRIGGERS | PlotPoint:Quest",
            "WorldElement:Trap | TRIGGERS | DevelopmentEvent:Alarm",
        ],
        "examples_invalid": [],
    },
}
