"""Relationship constraints for category: emotional_relationships."""

from models.kg_constants import NODE_LABELS

from ..classifications import NodeClassifications

RELATIONSHIP_CONSTRAINTS = {
    "LOVES": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.SENTIENT
        | NodeClassifications.SPATIAL
        | NodeClassifications.ABSTRACT,
        "invalid_combinations": [
            # No self-love validation needed - that's philosophically valid
        ],
        "bidirectional": True,
        "description": "Emotional affection between sentient beings, or toward places/concepts",
        "examples_valid": [
            "Character:Alice | LOVES | Character:Bob",
            "Character:Hero | LOVES | Location:Hometown",
        ],
        "examples_invalid": [
            "WorldElement:Sword | LOVES | Character:Hero",
            "Location:Forest | LOVES | Character:Ranger",
        ],
    },
    "HATES": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.SENTIENT
        | NodeClassifications.SPATIAL
        | NodeClassifications.ABSTRACT,
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Emotional animosity from sentient beings toward any entity",
        "examples_valid": [
            "Character:Villain | HATES | Character:Hero",
            "Character:Explorer | HATES | Location:Dungeon",
        ],
        "examples_invalid": ["WorldElement:Rock | HATES | Character:Hero"],
    },
    "FEARS": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NODE_LABELS
        - {"Entity"},  # Can fear anything except the base Entity type
        "invalid_combinations": [],
        "bidirectional": False,  # Fear is typically directional
        "description": "Fear response from sentient beings toward any entity or concept",
        "examples_valid": [
            "Character:Child | FEARS | Character:Monster",
            "Character:Sailor | FEARS | Location:Storm",
        ],
        "examples_invalid": ["Location:Cave | FEARS | Character:Hero"],
    },
    "RESPECTS": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.SENTIENT
        | NodeClassifications.ABSTRACT
        | NodeClassifications.ORGANIZATIONAL,
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Respect between sentient beings or toward abstract concepts/institutions",
        "examples_valid": [
            "Character:Student | RESPECTS | Character:Teacher",
            "Character:Citizen | RESPECTS | Faction:Council",
        ],
        "examples_invalid": ["WorldElement:Hammer | RESPECTS | Character:Smith"],
    },
}
