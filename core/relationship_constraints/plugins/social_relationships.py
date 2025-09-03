"""Relationship constraints for category: social_relationships."""

from ..classifications import NodeClassifications

RELATIONSHIP_CONSTRAINTS = {
    "FAMILY_OF": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.SENTIENT,
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Family relationships between sentient beings",
        "examples_valid": ["Character:Father | FAMILY_OF | Character:Daughter"],
        "examples_invalid": [
            "Character:Hero | FAMILY_OF | WorldElement:Sword",
            "Location:House | FAMILY_OF | Character:Owner",
        ],
    },
    "FRIEND_OF": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.SENTIENT,
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Friendship between sentient beings",
        "examples_valid": ["Character:Alice | FRIEND_OF | Character:Bob"],
        "examples_invalid": ["Character:Hero | FRIEND_OF | WorldElement:Sword"],
    },
    "ENEMY_OF": {
        "valid_subject_types": NodeClassifications.SENTIENT
        | NodeClassifications.ORGANIZATIONAL,
        "valid_object_types": NodeClassifications.SENTIENT
        | NodeClassifications.ORGANIZATIONAL,
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Enmity between sentient beings or organizations",
        "examples_valid": [
            "Character:Hero | ENEMY_OF | Character:Villain",
            "Faction:Kingdom | ENEMY_OF | Faction:Empire",
        ],
        "examples_invalid": ["WorldElement:Sword | ENEMY_OF | Character:Hero"],
    },
    "ALLY_OF": {
        "valid_subject_types": NodeClassifications.SENTIENT
        | NodeClassifications.ORGANIZATIONAL,
        "valid_object_types": NodeClassifications.SENTIENT
        | NodeClassifications.ORGANIZATIONAL,
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Alliance between sentient beings or organizations",
        "examples_valid": [
            "Character:Hero | ALLY_OF | Character:Wizard",
            "Faction:Guild | ALLY_OF | Faction:Order",
        ],
        "examples_invalid": ["Location:Castle | ALLY_OF | Character:King"],
    },
    "ROMANTIC_WITH": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.SENTIENT,
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Romantic relationships between sentient beings",
        "examples_valid": ["Character:Romeo | ROMANTIC_WITH | Character:Juliet"],
        "examples_invalid": ["Character:Hero | ROMANTIC_WITH | WorldElement:Artifact"],
    },
}
