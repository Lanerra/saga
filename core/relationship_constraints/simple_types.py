# core/relationship_constraints/simple_types.py
"""
Simple node type classifications for relationship validation.

Replaces the complex Enhanced Node Taxonomy with straightforward type sets.
"""

# Living entities capable of action and thought
SENTIENT = {"Character", "Person", "Deity", "Spirit", "Creature"}

# Entities with consciousness (subset of SENTIENT)
CONSCIOUS = {"Character", "Person", "Deity", "Creature", "Spirit"}

# Physical objects and items
INANIMATE = {
    "Object",
    "Item",
    "Artifact",
    "Document",
    "Relic",
    "WorldElement",
    "Resource",
    "Material",
    "Food",
    "Currency",
    "Structure",
}

# Abstract concepts and ideas
ABSTRACT = {
    "Concept",
    "Law",
    "Tradition",
    "Language",
    "Symbol",
    "Story",
    "Song",
    "Dream",
    "Memory",
    "Emotion",
    "Skill",
    "Lore",
    "Knowledge",
    "Secret",
    "Rumor",
    "News",
    "Trait",
    "Attribute",
    "Quality",
    "Reputation",
    "Status",
    "PlotPoint",
}

# Spatial locations and containers
SPATIAL = {
    "Location",
    "Structure",
    "Region",
    "Territory",
    "Settlement",
    "Landmark",
    "Path",
    "Room",
    "WorldContainer",
    "Library",
    "Archive",
    "Treasury",
    "Collection",
}

# Container entities
CONTAINERS = {
    "Location",
    "Structure",
    "Region",
    "Territory",
    "Settlement",
    "Room",
    "Library",
    "Archive",
    "Treasury",
    "Collection",
    "WorldContainer",
}

# Information and communication entities
INFORMATIONAL = {
    "Lore",
    "Knowledge",
    "Secret",
    "Rumor",
    "News",
    "Message",
    "Record",
    "ValueNode",
    "NovelInfo",
    "Document",
}

# Temporal entities (time-related)
TEMPORAL = {
    "Event",
    "Era",
    "Season",
    "Timeline",
    "Moment",
    "Chapter",
    "DevelopmentEvent",
    "WorldElaborationEvent",
}

# Organizational entities
ORGANIZATIONAL = {
    "Faction",
    "Organization",
    "Guild",
    "House",
    "Order",
    "Council",
    "Role",
    "Rank",
    "Government",
}

# System-level entities
SYSTEM_ENTITIES = {
    "System",
    "Magic",
    "Technology",
    "Religion",
    "Culture",
    "Education",
    "Government",
    "Economy",
}

# Physical presence (things that exist physically in space)
PHYSICAL_PRESENCE = (
    SENTIENT
    | INANIMATE
    | SPATIAL
    | {
        "Resource",
        "Material",
        "Food",
        "Currency",
        "Library",
        "Archive",
        "Treasury",
        "Collection",
    }
)

# Entities that can be located
LOCATABLE = {
    "Character",
    "Person",
    "Creature",
    "Object",
    "Artifact",
    "Document",
    "Item",
    "Relic",
    "Structure",
    "WorldElement",
}

# Entities that can be owned
OWNABLE = {
    "Object",
    "Artifact",
    "Document",
    "Item",
    "Relic",
    "Resource",
    "Material",
    "Food",
    "Currency",
    "Structure",
    "Territory",
    "WorldElement",
}
