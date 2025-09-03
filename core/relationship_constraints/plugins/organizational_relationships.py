"""Relationship constraints for category: organizational_relationships."""
from ..classifications import NodeClassifications
from models.kg_constants import NODE_LABELS

RELATIONSHIP_CONSTRAINTS = {
    "MEMBER_OF": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.ORGANIZATIONAL,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Membership in organizations",
        "examples_valid": [
            "Character:Knight | MEMBER_OF | Faction:Order",
            "Character:Merchant | MEMBER_OF | Faction:Guild",
        ],
        "examples_invalid": ["WorldElement:Banner | MEMBER_OF | Faction:Army"],
    },
    "LEADER_OF": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.ORGANIZATIONAL
        | NodeClassifications.SENTIENT,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Leadership role",
        "examples_valid": [
            "Character:General | LEADER_OF | Faction:Army",
            "Character:Captain | LEADER_OF | Character:Squad",
        ],
        "examples_invalid": ["Faction:Council | LEADER_OF | Character:Mayor"],
    },
    "FOUNDED": {
        "valid_subject_types": NodeClassifications.SENTIENT
        | NodeClassifications.ORGANIZATIONAL,
        "valid_object_types": NodeClassifications.ORGANIZATIONAL
        | NodeClassifications.SPATIAL,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Founding relationship",
        "examples_valid": [
            "Character:King | FOUNDED | Faction:Kingdom",
            "Faction:Settlers | FOUNDED | Location:Village",
        ],
        "examples_invalid": ["Location:City | FOUNDED | Character:Mayor"],
    },
}
