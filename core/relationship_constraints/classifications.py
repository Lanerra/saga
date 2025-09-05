"""Node classifications for relationship constraint system."""


class NodeClassifications:
    """Enhanced classification of node types into semantic categories."""

    # Living entities capable of action and thought
    SENTIENT = {"Character", "Person", "Deity", "Spirit", "Creature"}

    # Entities with consciousness and self-awareness (subset of sentient)
    CONSCIOUS = {"Character", "Person", "Deity", "Creature", "Spirit"}

    # Physical objects and entities
    PHYSICAL_PRESENCE = {
        "Character",
        "Person",
        "Creature",
        "Spirit",
        "Deity",
        "Object",
        "Artifact",
        "Document",
        "Item",
        "Relic",
        "Location",
        "Structure",
        "Region",
        "Territory",
        "Landmark",
        "Path",
        "Room",
        "Settlement",
        "Resource",
        "Material",
        "Food",
        "Currency",
        "WorldElement",
    }

    # Entities that can be located in space
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

    # Entities capable of social relationships
    SOCIAL = {
        "Character",
        "Person",
        "Faction",
        "Organization",
        "Guild",
        "House",
        "Order",
        "Council",
        "Culture",
    }

    # Spatial entities that can contain others
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

    # Temporal entities
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

    # Abstract concepts
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
        "NovelInfo",
    }

    # System entities
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

    # Information entities
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

    # Legacy classifications for backward compatibility
    INANIMATE = PHYSICAL_PRESENCE - SENTIENT  # Non-living physical entities
    SPATIAL = CONTAINERS | {"Location", "Region", "Territory"}  # Spatial entities
