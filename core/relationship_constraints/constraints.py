# core/relationship_constraints/constraints.py
"""Unified relationship constraint definitions."""

from __future__ import annotations

from core.enhanced_node_taxonomy import NodeClassification as NodeClassifications
from models.kg_constants import NODE_LABELS

ABSTRACT_TRAIT_RELATIONSHIPS: dict[str, dict[str, object]] = {
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


ACCESSIBILITY_AND_USAGE_RELATIONSHIPS: dict[str, dict[str, object]] = {
    "ACCESSIBLE_BY": {
        "valid_subject_types": NodeClassifications.SPATIAL
        | NodeClassifications.INFORMATIONAL
        | NodeClassifications.INANIMATE,
        "valid_object_types": NodeClassifications.SENTIENT
        | NodeClassifications.SPATIAL
        | {"Path", "Role", "Entity"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Accessibility relationship",
        "examples_valid": [
            "Location:Vault | ACCESSIBLE_BY | Character:Keyholder",
            "Document:File | ACCESSIBLE_BY | Role:Admin",
        ],
        "examples_invalid": [],
    },
    "USED_IN": {
        "valid_subject_types": NodeClassifications.INANIMATE
        | NodeClassifications.ABSTRACT
        | NodeClassifications.SYSTEM_ENTITIES,
        "valid_object_types": NodeClassifications.TEMPORAL
        | NodeClassifications.ABSTRACT
        | {"Entity"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Usage in events or contexts",
        "examples_valid": [
            "Artifact:Sword | USED_IN | Event:Battle",
            "System:Magic | USED_IN | Event:Ritual",
        ],
        "examples_invalid": [],
    },
    "TARGETS": {
        "valid_subject_types": NodeClassifications.INFORMATIONAL
        | NodeClassifications.INANIMATE
        | NodeClassifications.ABSTRACT,
        "valid_object_types": NODE_LABELS - {"ValueNode"},  # Can target almost anything
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Targeting or directing toward something",
        "examples_valid": [
            "Document:Report | TARGETS | Location:Area",
            "System:Weapon | TARGETS | Character:Enemy",
        ],
        "examples_invalid": [],
    },
}


ASSOCIATION_RELATIONSHIPS: dict[str, dict[str, object]] = {
    "ASSOCIATED_WITH": {
        "valid_subject_types": NODE_LABELS,
        "valid_object_types": NODE_LABELS,
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "General association relationship",
        "examples_valid": [
            "Character:Hero | ASSOCIATED_WITH | Faction:Guild",
            "Symbol:Crown | ASSOCIATED_WITH | Concept:Royalty",
        ],
        "examples_invalid": [],
    },
}


COGNITIVE_MENTAL_RELATIONSHIPS: dict[str, dict[str, object]] = {
    "BELIEVES": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.ABSTRACT
        | NodeClassifications.SENTIENT,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Belief relationship from sentient beings",
        "examples_valid": [
            "Character:Priest | BELIEVES | Lore:Prophecy",
            "Character:Student | BELIEVES | Character:Teacher",
        ],
        "examples_invalid": ["WorldElement:Book | BELIEVES | Lore:Truth"],
    },
    "REALIZES": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.ABSTRACT | {"PlotPoint"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Realization or understanding",
        "examples_valid": [
            "Character:Detective | REALIZES | PlotPoint:Solution",
            "Character:Hero | REALIZES | Trait:Truth",
        ],
        "examples_invalid": ["Location:Library | REALIZES | Lore:Knowledge"],
    },
    "REMEMBERS": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NODE_LABELS - {"Entity"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Memory relationship from sentient beings",
        "examples_valid": [
            "Character:Veteran | REMEMBERS | Character:Friend",
            "Character:Elder | REMEMBERS | Location:Hometown",
        ],
        "examples_invalid": ["WorldElement:Diary | REMEMBERS | Character:Writer"],
    },
    "UNDERSTANDS": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.ABSTRACT
        | NodeClassifications.INANIMATE,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Understanding relationship",
        "examples_valid": [
            "Character:Scholar | UNDERSTANDS | System:Magic",
            "Character:Craftsman | UNDERSTANDS | WorldElement:Tool",
        ],
        "examples_invalid": ["System:Magic | UNDERSTANDS | Character:Wizard"],
    },
}


COMMUNICATION_AND_DISPLAY_RELATIONSHIPS: dict[str, dict[str, object]] = {
    "DISPLAYS": {
        "valid_subject_types": NodeClassifications.INANIMATE
        | NodeClassifications.SYSTEM_ENTITIES
        | NodeClassifications.SPATIAL
        | {"WorldElement", "Entity"},
        "valid_object_types": NodeClassifications.INFORMATIONAL
        | NodeClassifications.ABSTRACT
        | {"ValueNode", "Entity"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Display or presentation of information",
        "examples_valid": [
            "System:Screen | DISPLAYS | Message:Alert",
            "Document:Map | DISPLAYS | Location:Territory",
        ],
        "examples_invalid": [],
    },
    "SPOKEN_BY": {
        "valid_subject_types": NodeClassifications.INFORMATIONAL
        | {"Message", "ValueNode"},
        "valid_object_types": NodeClassifications.SENTIENT,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Communication originating from sentient beings",
        "examples_valid": [
            "Message:Warning | SPOKEN_BY | Character:Guard",
            "ValueNode:Word | SPOKEN_BY | Character:Mage",
        ],
        "examples_invalid": [],
    },
    "EMITS": {
        "valid_subject_types": NodeClassifications.INANIMATE
        | NodeClassifications.SYSTEM_ENTITIES
        | {"WorldElement", "Entity"},
        "valid_object_types": NodeClassifications.ABSTRACT
        | NodeClassifications.INFORMATIONAL
        | {"ValueNode", "Entity"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Emission of energy, sound, or information",
        "examples_valid": [
            "WorldElement:Crystal | EMITS | ValueNode:Light",
            "System:Radio | EMITS | Message:Signal",
        ],
        "examples_invalid": [],
    },
}


DEFAULT_FALLBACK_RESTRICTED: dict[str, dict[str, object]] = {
    "RELATES_TO": {
        "valid_subject_types": {
            "Entity"
        },  # Only allow generic Entity types - everything else should use specific relationships
        "valid_object_types": {"Entity"},  # Only allow generic Entity types
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Generic relationship fallback - RESTRICTED to truly ambiguous Entity relationships only",
        "examples_valid": ["Entity:Unknown | RELATES_TO | Entity:Mysterious"],
        "examples_invalid": [
            "Character:Hero | RELATES_TO | PlotPoint:Quest"
        ],  # Should use specific relationships instead
    },
}


EMOTIONAL_RELATIONSHIPS: dict[str, dict[str, object]] = {
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


HIERARCHICAL_RELATIONSHIPS: dict[str, dict[str, object]] = {
    "MENTOR_TO": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.SENTIENT,
        "invalid_combinations": [],
        "bidirectional": False,  # Mentorship has clear direction
        "description": "Teaching/guidance relationship between sentient beings",
        "examples_valid": ["Character:Master | MENTOR_TO | Character:Apprentice"],
        "examples_invalid": ["WorldElement:Book | MENTOR_TO | Character:Student"],
    },
    "STUDENT_OF": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.SENTIENT,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Learning relationship between sentient beings",
        "examples_valid": ["Character:Apprentice | STUDENT_OF | Character:Master"],
        "examples_invalid": ["Character:Student | STUDENT_OF | WorldElement:TextBook"],
    },
    "LEADS": {
        "valid_subject_types": NodeClassifications.SENTIENT
        | NodeClassifications.ORGANIZATIONAL,
        "valid_object_types": NodeClassifications.SENTIENT
        | NodeClassifications.ORGANIZATIONAL,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Leadership relationship",
        "examples_valid": [
            "Character:Captain | LEADS | Character:Soldier",
            "Character:King | LEADS | Faction:Army",
        ],
        "examples_invalid": ["WorldElement:Crown | LEADS | Character:King"],
    },
    "WORKS_FOR": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.SENTIENT
        | NodeClassifications.ORGANIZATIONAL,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Employment or service relationship",
        "examples_valid": [
            "Character:Guard | WORKS_FOR | Character:Lord",
            "Character:Merchant | WORKS_FOR | Faction:Guild",
        ],
        "examples_invalid": ["WorldElement:Tool | WORKS_FOR | Character:Craftsman"],
    },
    "SERVES": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.SENTIENT
        | NodeClassifications.ORGANIZATIONAL
        | NodeClassifications.ABSTRACT,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Service or allegiance relationship",
        "examples_valid": [
            "Character:Knight | SERVES | Character:King",
            "Character:Priest | SERVES | Lore:God",
        ],
        "examples_invalid": ["Location:Temple | SERVES | Lore:God"],
    },
}


INFORMATION_AND_RECORDING_RELATIONSHIPS: dict[str, dict[str, object]] = {
    "RECORDS": {
        "valid_subject_types": NodeClassifications.SENTIENT
        | NodeClassifications.SYSTEM_ENTITIES
        | NodeClassifications.INFORMATIONAL,
        "valid_object_types": NodeClassifications.INFORMATIONAL
        | NodeClassifications.ABSTRACT
        | {"ValueNode", "Entity"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Recording or documenting information",
        "examples_valid": [
            "Character:Scribe | RECORDS | Message:Event",
            "System:Database | RECORDS | Knowledge:Data",
        ],
        "examples_invalid": [],
    },
    "PRESERVES": {
        "valid_subject_types": NodeClassifications.CONTAINERS
        | NodeClassifications.SPATIAL
        | NodeClassifications.ORGANIZATIONAL,
        "valid_object_types": NodeClassifications.INFORMATIONAL
        | NodeClassifications.INANIMATE
        | NodeClassifications.ABSTRACT,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Preservation or archival relationship",
        "examples_valid": [
            "Archive:Library | PRESERVES | Document:Scroll",
            "Organization:Museum | PRESERVES | Artifact:Relic",
        ],
        "examples_invalid": [],
    },
    "HAS_METADATA": {
        "valid_subject_types": NodeClassifications.INFORMATIONAL
        | NodeClassifications.INANIMATE
        | NodeClassifications.SYSTEM_ENTITIES,
        "valid_object_types": NodeClassifications.INFORMATIONAL
        | NodeClassifications.ABSTRACT
        | {"ValueNode", "Attribute"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Contains metadata or descriptive information",
        "examples_valid": [
            "Document:File | HAS_METADATA | Attribute:CreationDate",
            "System:Database | HAS_METADATA | ValueNode:Schema",
        ],
        "examples_invalid": [],
    },
}


OPERATIONAL_RELATIONSHIPS: dict[str, dict[str, object]] = {
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


ORGANIZATIONAL_RELATIONSHIPS: dict[str, dict[str, object]] = {
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


PHYSICAL_RELATIONSHIPS: dict[str, dict[str, object]] = {
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


POSSESSION_RELATIONSHIPS: dict[str, dict[str, object]] = {
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


SOCIAL_RELATIONSHIPS: dict[str, dict[str, object]] = {
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


SPATIAL_RELATIONSHIPS: dict[str, dict[str, object]] = {
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


SPECIAL_ACTION_RELATIONSHIPS: dict[str, dict[str, object]] = {
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


STATUS_AND_STATE_CHANGE_RELATIONSHIPS: dict[str, dict[str, object]] = {
    "WAS_REPLACED_BY": {
        "valid_subject_types": NODE_LABELS,
        "valid_object_types": NODE_LABELS,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Past replacement relationship (inverse form)",
        "examples_valid": ["System:OldMagic | WAS_REPLACED_BY | System:NewMagic"],
        "examples_invalid": [],
    },
    "CHARACTERIZED_BY": {
        "valid_subject_types": NodeClassifications.TEMPORAL
        | NodeClassifications.ABSTRACT
        | NodeClassifications.SPATIAL,
        "valid_object_types": NodeClassifications.ABSTRACT
        | NodeClassifications.INFORMATIONAL
        | {"Concept", "Attribute"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Characterized or defined by certain traits",
        "examples_valid": ["Era:Medieval | CHARACTERIZED_BY | Concept:Feudalism"],
        "examples_invalid": [],
    },
    "IS_NOW": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.ORGANIZATIONAL | {"Role", "Status"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Current role or status",
        "examples_valid": ["Character:Hero | IS_NOW | Role:Leader"],
        "examples_invalid": [],
    },
    "IS_NO_LONGER": {
        "valid_subject_types": NodeClassifications.SENTIENT,
        "valid_object_types": NodeClassifications.ORGANIZATIONAL | {"Role", "Status"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Former role or status",
        "examples_valid": ["Character:Hero | IS_NO_LONGER | Status:Apprentice"],
        "examples_invalid": [],
    },
    "DIFFERS_FROM": {
        "valid_subject_types": NODE_LABELS,
        "valid_object_types": NODE_LABELS,
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Difference or distinction relationship",
        "examples_valid": ["Document:V1 | DIFFERS_FROM | Document:V2"],
        "examples_invalid": [],
    },
}


TEMPORAL_AND_STATE_RELATIONSHIPS: dict[str, dict[str, object]] = {
    "REPLACED_BY": {
        "valid_subject_types": NODE_LABELS,
        "valid_object_types": NODE_LABELS,
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Replacement or succession relationship",
        "examples_valid": [
            "System:OldMagic | REPLACED_BY | System:NewMagic",
            "Character:OldKing | REPLACED_BY | Character:NewKing",
        ],
        "examples_invalid": [],
    },
    "LINKED_TO": {
        "valid_subject_types": NODE_LABELS,
        "valid_object_types": NODE_LABELS,
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Connection or linkage relationship",
        "examples_valid": [
            "System:Network | LINKED_TO | System:Database",
            "Location:Portal | LINKED_TO | Location:Destination",
        ],
        "examples_invalid": [],
    },
}


TEMPORAL_CAUSAL_RELATIONSHIPS: dict[str, dict[str, object]] = {
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


CATEGORY_CONSTRAINTS: dict[str, dict[str, dict[str, object]]] = {
    "abstract_trait_relationships": ABSTRACT_TRAIT_RELATIONSHIPS,
    "accessibility_and_usage_relationships": ACCESSIBILITY_AND_USAGE_RELATIONSHIPS,
    "association_relationships": ASSOCIATION_RELATIONSHIPS,
    "cognitive_mental_relationships": COGNITIVE_MENTAL_RELATIONSHIPS,
    "communication_and_display_relationships": COMMUNICATION_AND_DISPLAY_RELATIONSHIPS,
    "default_fallback_restricted": DEFAULT_FALLBACK_RESTRICTED,
    "emotional_relationships": EMOTIONAL_RELATIONSHIPS,
    "hierarchical_relationships": HIERARCHICAL_RELATIONSHIPS,
    "information_and_recording_relationships": INFORMATION_AND_RECORDING_RELATIONSHIPS,
    "operational_relationships": OPERATIONAL_RELATIONSHIPS,
    "organizational_relationships": ORGANIZATIONAL_RELATIONSHIPS,
    "physical_relationships": PHYSICAL_RELATIONSHIPS,
    "possession_relationships": POSSESSION_RELATIONSHIPS,
    "social_relationships": SOCIAL_RELATIONSHIPS,
    "spatial_relationships": SPATIAL_RELATIONSHIPS,
    "special_action_relationships": SPECIAL_ACTION_RELATIONSHIPS,
    "status_and_state_change_relationships": STATUS_AND_STATE_CHANGE_RELATIONSHIPS,
    "temporal_and_state_relationships": TEMPORAL_AND_STATE_RELATIONSHIPS,
    "temporal_causal_relationships": TEMPORAL_CAUSAL_RELATIONSHIPS,
}


RELATIONSHIP_CONSTRAINTS: dict[str, dict[str, object]] = {}
for constraints in CATEGORY_CONSTRAINTS.values():
    RELATIONSHIP_CONSTRAINTS.update(constraints)
