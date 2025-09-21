# core/enhanced_node_taxonomy.py
"""
Enhanced Node Type Taxonomy for SAGA Knowledge Graph

This module provides an expanded and hierarchical node type system that replaces
the current generic WorldElement + category approach with specific, semantically
meaningful node types.
"""

from enum import Enum


class NodeTypeCategory(Enum):
    """High-level categories for node type organization."""

    PHYSICAL = "physical"  # Tangible entities
    ABSTRACT = "abstract"  # Concepts, ideas
    TEMPORAL = "temporal"  # Time-related entities
    ORGANIZATIONAL = "organizational"  # Groups, structures
    SYSTEM = "system"  # Frameworks, processes
    RESOURCE = "resource"  # Materials, commodities
    INFORMATION = "information"  # Data, knowledge
    QUALITY = "quality"  # Attributes, traits
    CONTAINER = "container"  # Grouping entities


# Enhanced Node Type Definitions
ENHANCED_NODE_LABELS = {
    # === CORE EXISTING TYPES ===
    "Entity",  # Base label - all nodes inherit this
    "NovelInfo",  # Novel metadata
    "Chapter",  # Chapter entities
    # === LEGACY TYPES FOR BACKWARD COMPATIBILITY ===
    "WorldElement",  # Legacy generic type - being phased out
    # === PHYSICAL ENTITIES ===
    # Living Beings
    "Character",  # Main story characters (active participants)
    "Person",  # Historical figures, mentioned people (inactive)
    "Creature",  # Animals, monsters, non-human beings
    "Spirit",  # Ghosts, ethereal beings, supernatural entities
    "Deity",  # Gods, divine beings, worshipped entities
    # Objects & Items
    "Object",  # Physical items, tools, weapons
    "Artifact",  # Special/magical/historical objects
    "Document",  # Books, scrolls, letters, written records
    "Item",  # General items (synonym for Object)
    "Relic",  # Ancient/sacred objects
    # Locations & Structures
    "Location",  # General places
    "Structure",  # Buildings, constructions, man-made locations
    "Region",  # Large geographical areas, territories
    "Landmark",  # Notable geographical features
    "Territory",  # Claimed/controlled areas
    "Path",  # Roads, routes, passages, connections
    "Room",  # Interior spaces within structures
    "Settlement",  # Cities, towns, villages
    # === ABSTRACT ENTITIES ===
    "Concept",  # Ideas, philosophies, abstract notions
    "Law",  # Rules, regulations, natural laws, edicts
    "Tradition",  # Cultural practices, customs, rituals
    "Language",  # Languages, dialects, communication systems
    "Symbol",  # Symbolic representations, emblems, signs
    "Story",  # Tales, legends, narratives within the narrative
    "Song",  # Music, ballads, cultural expressions
    "Dream",  # Dreams, visions, prophecies
    "Memory",  # Specific memories, recollections
    "Emotion",  # Emotional states, feelings
    "Skill",  # Abilities, talents, competencies
    # === TEMPORAL ENTITIES ===
    "Event",  # Historical events, battles, ceremonies
    "Era",  # Time periods, ages, epochs
    "Timeline",  # Temporal sequences, chronologies
    "DevelopmentEvent",  # Character development events (existing)
    "WorldElaborationEvent",  # World expansion events (existing)
    "Season",  # Natural cycles, recurring periods
    "Moment",  # Specific points in time, instances
    # === ORGANIZATIONAL ENTITIES ===
    "Faction",  # Political groups, organizations with agendas
    "Organization",  # Generic organizations, institutions
    "Role",  # Positions, titles, functions within organizations
    "Rank",  # Hierarchical positions, military ranks
    "Guild",  # Professional organizations, trade groups
    "House",  # Noble houses, family organizations
    "Order",  # Religious or knightly orders
    "Council",  # Governing bodies, decision-making groups
    # === SYSTEM ENTITIES ===
    "System",  # General systems, frameworks
    "Magic",  # Magical systems, schools of magic
    "Technology",  # Technological systems, innovations
    "Religion",  # Religious systems, belief structures
    "Culture",  # Cultural systems, way of life
    "Education",  # Learning systems, schools, curricula
    "Government",  # Governing systems, political structures
    "Economy",  # Economic systems, trade networks
    # === RESOURCE ENTITIES ===
    "Resource",  # Materials, natural resources, commodities
    "Currency",  # Money, trade systems, units of value
    "Trade",  # Economic relationships, trade routes
    "Food",  # Consumable resources, sustenance
    "Material",  # Building materials, crafting components
    "Energy",  # Power sources, magical energy, fuel
    # === INFORMATION ENTITIES ===
    "Lore",  # Myths, legends, historical knowledge
    "Knowledge",  # Specific knowledge, information, data
    "Secret",  # Hidden information, classified knowledge
    "Rumor",  # Unconfirmed information, gossip
    "News",  # Current information, recent events
    "Message",  # Communications, signals, transmissions
    "ValueNode",  # Literal values, data (existing)
    "Record",  # Official records, documentation
    # === QUALITY ENTITIES ===
    "Trait",  # Character traits, personality aspects (existing)
    "Attribute",  # Physical or mental attributes
    "Quality",  # Characteristics, properties
    "Reputation",  # Social standing, renown
    "Status",  # Current state, condition
    # === CONTAINER ENTITIES ===
    "WorldContainer",  # Containers for world elements (existing)
    "PlotPoint",  # Plot/story points (existing)
    "Collection",  # Groups of related items
    "Archive",  # Storage of documents/information
    "Treasury",  # Storage of valuable resources
    "Library",  # Collection of documents/knowledge
}

# Node Type Hierarchical Relationships
NODE_TYPE_HIERARCHY = {
    "Entity": {  # Root - all nodes have this label
        "Physical": {
            "Living": {
                "Character": {"weight": 10},
                "Person": {"weight": 9},
                "Creature": {"weight": 8},
                "Spirit": {"weight": 7},
                "Deity": {"weight": 6},
            },
            "Object": {
                "Artifact": {"weight": 10},
                "Document": {"weight": 9},
                "Item": {"weight": 8},
                "Relic": {"weight": 7},
                "Object": {"weight": 6},  # Generic fallback
            },
            "Spatial": {
                "Settlement": {"weight": 10},
                "Structure": {"weight": 9},
                "Region": {"weight": 8},
                "Territory": {"weight": 7},
                "Landmark": {"weight": 6},
                "Location": {"weight": 5},  # Generic fallback
                "Room": {"weight": 4},
                "Path": {"weight": 3},
            },
        },
        "Abstract": {
            "Mental": {
                "Concept": {"weight": 10},
                "Dream": {"weight": 9},
                "Memory": {"weight": 8},
                "Emotion": {"weight": 7},
                "Skill": {"weight": 6},
            },
            "Cultural": {
                "Tradition": {"weight": 10},
                "Language": {"weight": 9},
                "Symbol": {"weight": 8},
                "Story": {"weight": 7},
                "Song": {"weight": 6},
            },
            "Legal": {
                "Law": {"weight": 10},
            },
        },
        "Temporal": {
            "Event": {"weight": 10},
            "Era": {"weight": 9},
            "Season": {"weight": 8},
            "Timeline": {"weight": 7},
            "Moment": {"weight": 6},
            "Chapter": {"weight": 5},
            "DevelopmentEvent": {"weight": 4},
            "WorldElaborationEvent": {"weight": 3},
        },
        "Organizational": {
            "Political": {
                "Faction": {"weight": 10},
                "House": {"weight": 9},
                "Government": {"weight": 8},
                "Council": {"weight": 7},
            },
            "Professional": {
                "Guild": {"weight": 10},
                "Order": {"weight": 9},
                "Organization": {"weight": 8},  # Generic fallback
            },
            "Hierarchical": {
                "Role": {"weight": 10},
                "Rank": {"weight": 9},
            },
        },
        "System": {
            "Magical": {
                "Magic": {"weight": 10},
            },
            "Technological": {
                "Technology": {"weight": 10},
            },
            "Social": {
                "Religion": {"weight": 10},
                "Culture": {"weight": 9},
                "Education": {"weight": 8},
                "Economy": {"weight": 7},
            },
            "System": {"weight": 5},  # Generic fallback
        },
        "Resource": {
            "Economic": {
                "Currency": {"weight": 10},
                "Trade": {"weight": 9},
            },
            "Physical": {
                "Food": {"weight": 10},
                "Material": {"weight": 9},
                "Energy": {"weight": 8},
                "Resource": {"weight": 7},  # Generic fallback
            },
        },
        "Information": {
            "Recorded": {
                "Document": {"weight": 10},  # Also in Physical/Object
                "Record": {"weight": 9},
                "Lore": {"weight": 8},
            },
            "Communication": {
                "Message": {"weight": 10},
                "News": {"weight": 9},
                "Rumor": {"weight": 8},
            },
            "Classified": {
                "Secret": {"weight": 10},
                "Knowledge": {"weight": 9},
            },
            "Data": {
                "ValueNode": {"weight": 10},
            },
            "NovelInfo": {"weight": 5},
        },
        "Quality": {
            "Personal": {
                "Trait": {"weight": 10},
                "Attribute": {"weight": 9},
                "Quality": {"weight": 8},  # Generic fallback
            },
            "Social": {
                "Reputation": {"weight": 10},
                "Status": {"weight": 9},
            },
        },
        "Container": {
            "Physical": {
                "Library": {"weight": 10},
                "Archive": {"weight": 9},
                "Treasury": {"weight": 8},
                "Collection": {"weight": 7},
            },
            "Conceptual": {
                "PlotPoint": {"weight": 10},
                "WorldContainer": {"weight": 9},
            },
        },
    }
}


# Type Classification for Constraint System
class NodeClassification:
    """Enhanced node classifications for relationship constraints."""

    # Living entities capable of action and thought
    SENTIENT = {"Character", "Person", "Deity", "Spirit", "Creature"}

    # Entities with consciousness and self-awareness
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
        # Container types that have physical presence
        "Library",
        "Archive",
        "Treasury",
        "Collection",
        # Legacy type for backward compatibility
        *( ["WorldElement"] if getattr(config, "ENABLE_LEGACY_WORLDELEMENT", True) else [] ),
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
        # Legacy type for backward compatibility
        *( ["WorldElement"] if getattr(config, "ENABLE_LEGACY_WORLDELEMENT", True) else [] ),
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
        "Territory",  # Can own buildings and land
        # Legacy type for backward compatibility
        *( ["WorldElement"] if getattr(config, "ENABLE_LEGACY_WORLDELEMENT", True) else [] ),
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


def get_node_type_priority(node_type: str) -> int:
    """Get priority weight for node type selection (higher = more specific)."""

    def find_weight(hierarchy: dict, target_type: str, current_weight: int = 0) -> int:
        for key, value in hierarchy.items():
            if key == target_type:
                return current_weight + value.get("weight", 0)
            elif isinstance(value, dict) and "weight" not in value:
                # Recurse into nested hierarchy
                result = find_weight(value, target_type, current_weight + 1)
                if result > 0:
                    return result
        return 0

    return find_weight(NODE_TYPE_HIERARCHY, node_type)


def get_all_node_classifications(node_type: str) -> set[str]:
    """Get all classifications that apply to a node type."""
    classifications = set()

    if node_type in NodeClassification.SENTIENT:
        classifications.add("SENTIENT")
    if node_type in NodeClassification.CONSCIOUS:
        classifications.add("CONSCIOUS")
    if node_type in NodeClassification.PHYSICAL_PRESENCE:
        classifications.add("PHYSICAL_PRESENCE")
    if node_type in NodeClassification.LOCATABLE:
        classifications.add("LOCATABLE")
    if node_type in NodeClassification.OWNABLE:
        classifications.add("OWNABLE")
    if node_type in NodeClassification.SOCIAL:
        classifications.add("SOCIAL")
    if node_type in NodeClassification.CONTAINERS:
        classifications.add("CONTAINERS")
    if node_type in NodeClassification.TEMPORAL:
        classifications.add("TEMPORAL")
    if node_type in NodeClassification.ABSTRACT:
        classifications.add("ABSTRACT")
    if node_type in NodeClassification.ORGANIZATIONAL:
        classifications.add("ORGANIZATIONAL")
    if node_type in NodeClassification.SYSTEM_ENTITIES:
        classifications.add("SYSTEM_ENTITIES")
    if node_type in NodeClassification.INFORMATIONAL:
        classifications.add("INFORMATIONAL")

    return classifications


def validate_node_type(node_type: str) -> bool:
    """Validate that a node type is in the enhanced type system."""
    return node_type in ENHANCED_NODE_LABELS
