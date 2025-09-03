"""Relationship constraints for category: spatial_relationships."""
from ..classifications import NodeClassifications
from models.kg_constants import NODE_LABELS

RELATIONSHIP_CONSTRAINTS = {
    "LOCATED_IN": {
        "valid_subject_types": NodeClassifications.LOCATABLE,
        "valid_object_types": NodeClassifications.SPATIAL,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Containment within a spatial location",
        "examples_valid": [
            "Character:Hero | LOCATED_IN | Location:Castle",
            "WorldElement:Treasure | LOCATED_IN | Location:Cave",
        ],
        "examples_invalid": [
            "Location:Forest | LOCATED_IN | Character:Ranger",
            "PlotPoint:Quest | LOCATED_IN | Location:Town",
        ],
    },
    "LOCATED_AT": {
        "valid_subject_types": NodeClassifications.LOCATABLE,
        "valid_object_types": NodeClassifications.SPATIAL,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Presence at a specific location",
        "examples_valid": [
            "Character:Merchant | LOCATED_AT | Location:Market",
            "WorldElement:Statue | LOCATED_AT | Location:Plaza",
        ],
        "examples_invalid": ["Trait:Brave | LOCATED_AT | Location:Battlefield"],
    },
    "CONTAINS": {
        "valid_subject_types": NodeClassifications.SPATIAL
        | NodeClassifications.CONTAINERS
        | NodeClassifications.INFORMATIONAL
        | NodeClassifications.SYSTEM_ENTITIES
        | {
            "WorldElement",
            "Entity",
            "Artifact",
        },  # Expanded to include information containers
        "valid_object_types": NodeClassifications.LOCATABLE
        | NodeClassifications.INFORMATIONAL
        | NodeClassifications.ABSTRACT
        | {
            "ValueNode",
            "Entity",
        },  # Expanded to include abstract concepts and information
        "invalid_combinations": [
            ("Character", "Character"),  # Characters don't contain other characters
        ],
        "bidirectional": False,
        "description": "Containment relationship - spatial, informational, or conceptual",
        "examples_valid": [
            "Location:Chest | CONTAINS | WorldElement:Gold",
            "Document:Book | CONTAINS | Knowledge:Secrets",
            "System:Database | CONTAINS | Message:Data",
            "Artifact:Memory | CONTAINS | Memory:Experience",
        ],
        "examples_invalid": ["Character:Hero | CONTAINS | Character:Friend"],
    },
    "NEAR": {
        "valid_subject_types": NodeClassifications.LOCATABLE
        | NodeClassifications.SPATIAL,
        "valid_object_types": NodeClassifications.LOCATABLE
        | NodeClassifications.SPATIAL,
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Proximity relationship",
        "examples_valid": [
            "Location:Village | NEAR | Location:Forest",
            "Character:Guard | NEAR | Location:Gate",
        ],
        "examples_invalid": ["Trait:Courage | NEAR | Location:Battlefield"],
    },
    "ADJACENT_TO": {
        "valid_subject_types": NodeClassifications.SPATIAL,
        "valid_object_types": NodeClassifications.SPATIAL,
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Direct spatial adjacency",
        "examples_valid": ["Location:Kitchen | ADJACENT_TO | Location:Dining_Room"],
        "examples_invalid": ["Character:Cook | ADJACENT_TO | Location:Kitchen"],
    },
}
