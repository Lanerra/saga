# enhanced_node_taxonomy.py
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
    SENTIENT = {"Character", "Person", "Deity", "Spirit"}

    # Entities with consciousness and self-awareness
    CONSCIOUS = {"Character", "Person", "Deity"}

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


def infer_node_type_from_category(category: str) -> str:
    """Infer specific node type from WorldElement category."""
    category_mapping = {
        # Direct mappings
        "locations": "Location",
        "factions": "Faction",
        "systems": "System",
        "lore": "Lore",
        "history": "Lore",
        "society": "Culture",
        "magic": "Magic",
        "technology": "Technology",
        "religion": "Religion",
        "organizations": "Organization",
        "characters": "Character",
        "people": "Person",
        "objects": "Object",
        "items": "Item",
        "artifacts": "Artifact",
        "documents": "Document",
        "events": "Event",
        "traditions": "Tradition",
        "languages": "Language",
        "creatures": "Creature",
        "deities": "Deity",
        "spirits": "Spirit",
        "concepts": "Concept",
        "laws": "Law",
        "skills": "Skill",
        "traits": "Trait",
        "resources": "Resource",
        "materials": "Material",
        "currency": "Currency",
        "trade": "Trade",
        "food": "Food",
        "structures": "Structure",
        "regions": "Region",
        "territories": "Territory",
        "landmarks": "Landmark",
        "paths": "Path",
        "settlements": "Settlement",
        "guilds": "Guild",
        "houses": "House",
        "orders": "Order",
        "councils": "Council",
        "roles": "Role",
        "ranks": "Rank",
    }

    # Clean and normalize category
    clean_category = category.lower().strip()

    # Direct lookup
    if clean_category in category_mapping:
        return category_mapping[clean_category]

    # Partial matching for plurals and variations
    for cat_key, node_type in category_mapping.items():
        if clean_category in cat_key or cat_key in clean_category:
            return node_type

    # Fallback to WorldElement for unknown categories
    return "WorldElement"


def infer_node_type_from_name(name: str, context: str = "") -> str:
    """
    Infer node type from entity name and context using genre-agnostic linguistic patterns.

    Uses structural and grammatical analysis rather than domain-specific keywords.
    """
    name_lower = name.lower()
    context_lower = context.lower()
    words = name.split()

    # Order matters: Most specific to least specific

    # === SYSTEM RECOGNITION (Most specific patterns) ===
    if _is_system_by_structure(name, words, context_lower):
        return "System"

    # === EVENT RECOGNITION (Very specific patterns) ===
    if _is_event_by_structure(name, words, context_lower):
        return "Event"

    # === ORGANIZATION RECOGNITION (Specific suffixes) ===
    if _is_organization_by_structure(name, words, context_lower):
        return "Organization"

    # === OBJECT/ARTIFACT RECOGNITION (Specific patterns) ===
    object_type = _classify_object_by_structure(name, words, context_lower)
    if object_type:
        return object_type

    # === MEMORY RECOGNITION (Specific patterns) ===
    if _is_memory_by_structure(name, words, context_lower):
        return "Memory"

    # === ENERGY/POWER RECOGNITION (Specific patterns) ===
    if _is_energy_by_structure(name, words, context_lower):
        return "Energy"

    # === ABSTRACT CONCEPT RECOGNITION (Suffix-based) ===
    if _is_concept_by_structure(name, words, context_lower):
        return "Concept"

    # === LOCATION RECOGNITION (Geographic patterns) ===
    if _is_location_by_structure(name, words, context_lower):
        return "Location"

    # === CHARACTER/PERSON RECOGNITION (Last resort for proper nouns) ===
    if _is_character_by_structure(name, words, context_lower):
        return "Character"

    # Strip articles and re-evaluate
    if name_lower.startswith(("the ", "a ", "an ")):
        return infer_node_type_from_name(name[name.find(" ") + 1 :], context)

    # Final fallback
    return "Entity"


# === GENRE-AGNOSTIC STRUCTURAL CLASSIFICATION HELPERS ===


def _is_location_by_structure(name: str, words: list[str], context: str) -> bool:
    """Identify locations using structural patterns rather than keywords."""
    name_lower = name.lower()

    # Geographic compound patterns: [Name] + [Geographic descriptor] - MOST SPECIFIC
    if len(words) == 2 and words[0][0].isupper() and words[1][0].isupper():
        # Generic geographic endings (not genre-specific)
        geographic_suffixes = {
            # Natural features
            "Falls",
            "River",
            "Lake",
            "Bay",
            "Point",
            "Heights",
            "Hills",
            "Plains",
            "Valley",
            "Creek",
            "Stream",
            "Pond",
            "Beach",
            "Coast",
            "Shore",
            "Island",
            "Belt",  # Added for "Asteroid Belt" etc - but not "Field"
            # Elevation features
            "Ridge",
            "Peak",
            "Summit",
            "Cliff",
            "Mesa",
            "Gorge",
            "Canyon",
            "Pass",
            # Areas/regions
            "District",
            "Quarter",
            "Sector",
            "Zone",
            "Area",
            "Region",
            "Territory",
            # Settlements (structural, not fantasy-specific)
            "City",
            "Town",
            "Village",
            "Settlement",
            "Colony",
            "Outpost",
            "Base",
            # Buildings/structures
            "Hospital",
            "Park",
            "Center",
            "Square",
            "Plaza",
            "Mall",
            "Station",
        }
        if words[1] in geographic_suffixes:
            return True

    # Specific location words in compound names
    location_words = ["park", "hospital", "scene", "center", "station", "square"]
    if any(word in name_lower for word in location_words):
        return True

    # Context clues for location
    location_contexts = [
        "located",
        "situated",
        "found",
        "place",
        "area",
        "region",
        "site",
    ]
    if any(ctx in context for ctx in location_contexts):
        return True

    return False


def _is_character_by_structure(name: str, words: list[str], context: str) -> bool:
    """Identify characters using structural patterns."""
    name_lower = name.lower()

    # Single names with titles/roles (structural pattern recognition) - MOST SPECIFIC
    title_patterns = {
        # Formal titles (universal across genres)
        "Dr.",
        "Doctor",
        "Professor",
        "Captain",
        "Commander",
        "Lieutenant",
        "Sergeant",
        "Chief",
        "Director",
        "Manager",
        "President",
        "Chairman",
        # Honorifics (genre-neutral)
        "Mr.",
        "Mrs.",
        "Ms.",
        "Miss",
        "Sir",
        "Madam",
        "Lady",
        "Lord",
    }
    if any(title in name for title in title_patterns):
        return True

    # Two-word proper noun pattern (First Last) - but exclude organizations and compound objects
    if len(words) == 2 and all(word.isalpha() and word[0].isupper() for word in words):
        # Exclude if it looks like an organization or location
        org_indicators = ["Corp", "Inc", "LLC", "Ltd", "Group", "Company", "Co"]
        location_indicators = [
            "River",
            "Lake",
            "Bay",
            "Park",
            "Street",
            "Avenue",
            "City",
            "Town",
        ]
        if not any(ind in name for ind in org_indicators + location_indicators):
            return True

    # Context clues for characters
    character_contexts = [
        "person",
        "individual",
        "character",
        "who",
        "said",
        "spoke",
        "thinks",
    ]
    if any(ctx in context for ctx in character_contexts):
        return True

    return False


def _is_organization_by_structure(name: str, words: list[str], context: str) -> bool:
    """Identify organizations using structural patterns."""
    name_lower = name.lower()

    # Organizational suffixes (genre-neutral) - VERY SPECIFIC
    org_suffixes = {
        "Corporation",
        "Corp",
        "Company",
        "Co",
        "Inc",
        "LLC",
        "Ltd",
        "Group",
        "Organization",
        "Institute",
        "Foundation",
        "Association",
        "Society",
        "Union",
        "Alliance",
        "Federation",
        "Coalition",
        "Consortium",
        "Department",
        "Ministry",
        "Bureau",
        "Agency",
        "Office",
        "Division",
        "Council",
        "Committee",
        "Board",
        "Commission",
        "Empire",
        "Rebellion",
    }

    if any(suffix in name for suffix in org_suffixes):
        return True

    # "The [Organization]" patterns for collective entities
    if name.startswith("The ") and len(words) >= 2:
        org_words = ["Rebellion", "Empire", "Government", "Party", "Movement"]
        if any(word in name for word in org_words):
            return True

    # "[Name] of [Something]" pattern
    if " of " in name_lower and len(words) >= 3:
        return True

    # Context clues
    org_contexts = ["organization", "group", "company", "institution", "entity"]
    if any(ctx in context for ctx in org_contexts):
        return True

    return False


def _classify_object_by_structure(
    name: str, words: list[str], context: str
) -> str | None:
    """Classify objects using structural patterns, returns Object/Artifact or None."""
    name_lower = name.lower()

    # Object words - SPECIFIC DETECTION
    object_words = ["diary", "letters", "device", "blaster", "gun", "sword", "evidence"]
    if any(word in name_lower for word in object_words):
        # Check if context suggests special significance
        special_contexts = [
            "ancient",
            "legendary",
            "magical",
            "powerful",
            "sacred",
            "unique",
        ]
        if any(ctx in context for ctx in special_contexts):
            return "Artifact"
        return "Object"

    # Special handling for "core" - could be object or memory
    if "core" in name_lower:
        if "memory" in name_lower:
            return "Object"  # Memory Core = computer storage
        return "Object"

    # Possessive patterns: [Name]'s [Thing]
    if "'s " in name and len(words) >= 2:
        # Exclude memories and character names
        memory_indicators = [
            "memory",
            "death",
            "dream",
            "nightmare",
            "vision",
            "thought",
        ]
        if not any(mem in name_lower for mem in memory_indicators):
            # Check if context suggests special significance
            special_contexts = [
                "ancient",
                "legendary",
                "magical",
                "powerful",
                "sacred",
                "unique",
            ]
            if any(ctx in context for ctx in special_contexts):
                return "Artifact"
            return "Object"

    # Material composition patterns
    material_patterns = ["made of", "crafted from", "forged from", "built with"]
    if any(pattern in context for pattern in material_patterns):
        return "Object"

    # Tool/device patterns (structural)
    if name_lower.endswith(("er", "or", "device", "tool", "instrument", "apparatus")):
        return "Object"

    return None


def _is_memory_by_structure(name: str, words: list[str], context: str) -> bool:
    """Identify memories using linguistic patterns."""
    name_lower = name.lower()

    # Direct memory indicators
    memory_words = ["memory", "recollection", "remembrance", "flashback"]
    if any(mem in name_lower for mem in memory_words):
        return True

    # Possessive memory patterns: [Name]'s [memory-related]
    if "'s " in name:
        memory_objects = ["last", "first", "final", "dying", "childhood", "earliest"]
        if any(mem in name_lower for mem in memory_objects):
            return True

    # Personal experience patterns
    if name.startswith(("Her ", "His ", "Their ", "My ", "Your ")):
        return True

    # Context clues
    memory_contexts = ["remembered", "recalled", "thought back", "reminisced"]
    if any(ctx in context for ctx in memory_contexts):
        return True

    return False


def _is_concept_by_structure(name: str, words: list[str], context: str) -> bool:
    """Identify abstract concepts using linguistic patterns."""
    name_lower = name.lower()

    # Abstract noun suffixes - MOST RELIABLE
    abstract_suffixes = (
        "ism",
        "ity",
        "ness",
        "ment",
        "tion",
        "sion",
        "ence",
        "ance",
        "ship",
        "hood",
        "dom",
        "ology",
        "ics",
        "ure",
        "age",
    )
    if name_lower.endswith(abstract_suffixes):
        return True

    # Academic/analytical patterns
    analytical_words = [
        "analysis",
        "study",
        "theory",
        "principle",
        "concept",
        "idea",
        "philosophy",
        "strategy",
        "condition",
        "commentary",
        "metaphor",
        "alibi",
    ]
    if any(word in name_lower for word in analytical_words):
        return True

    # Two-word abstract concepts
    if len(words) == 2 and not name.startswith("The "):
        abstract_first = ["human", "social", "economic", "political"]
        abstract_second = ["condition", "commentary", "theory", "analysis", "strategy"]
        if any(word in words[0].lower() for word in abstract_first) or any(
            word in words[1].lower() for word in abstract_second
        ):
            return True

    # Context-dependent abstract concepts
    abstract_contexts = [
        "represents",
        "symbolizes",
        "embodies",
        "concept",
        "idea",
        "notion",
    ]
    if any(ctx in context for ctx in abstract_contexts):
        return True

    return False


def _is_event_by_structure(name: str, words: list[str], context: str) -> bool:
    """Identify events using linguistic patterns."""
    name_lower = name.lower()

    # Event suffixes - MOST SPECIFIC
    if name.endswith(("Event", "Incident", "Occurrence", "Happening", "Episode")):
        return True

    # Common event words
    event_words = [
        "battle",
        "war",
        "revolution",
        "murder",
        "accident",
        "wedding",
        "haunting",
    ]
    if any(word in name_lower for word in event_words):
        return True

    # "The [Event]" pattern with temporal context
    if name.startswith("The ") and len(words) >= 2:
        temporal_contexts = [
            "happened",
            "occurred",
            "took place",
            "during",
            "when",
            "after",
            "before",
        ]
        if any(ctx in context for ctx in temporal_contexts):
            return True

        # Event-like nouns after "The"
        event_nouns = [
            "Battle",
            "War",
            "Revolution",
            "Murder",
            "Accident",
            "Wedding",
            "Haunting",
        ]
        if any(noun in name for noun in event_nouns):
            return True

    return False


def _is_system_by_structure(name: str, words: list[str], context: str) -> bool:
    """Identify systems using structural patterns."""
    name_lower = name.lower()

    # System-related suffixes - VERY SPECIFIC
    system_suffixes = [
        "System",
        "Network",
        "Protocol",
        "Framework",
        "Method",
        "Process",
        "Investigation",
    ]
    if any(suffix in name for suffix in system_suffixes):
        return True

    # "The [System]" patterns
    if name.startswith("The ") and len(words) >= 2:
        system_words = ["Internet", "Network", "System", "Protocol"]
        if any(word in name for word in system_words):
            return True

    # Context clues
    system_contexts = [
        "operates",
        "functions",
        "works",
        "mechanism",
        "process",
        "procedure",
    ]
    if any(ctx in context for ctx in system_contexts):
        return True

    return False


def _is_energy_by_structure(name: str, words: list[str], context: str) -> bool:
    """Identify energy/power using linguistic patterns."""
    name_lower = name.lower()

    # Energy-related words (genre-neutral) - prioritize "field" for energy
    energy_words = ["energy", "power", "force", "wave", "current", "flow", "charge"]
    if any(word in name_lower for word in energy_words):
        return True

    # Special case: "[Type] Field" can be energy (Plasma Field, Force Field)
    if "field" in name_lower and len(words) == 2:
        field_types = ["plasma", "force", "energy", "magnetic", "electric", "psychic"]
        if any(ftype in name_lower for ftype in field_types):
            return True

    # Context clues
    energy_contexts = ["flows", "radiates", "emanates", "pulses", "vibrates"]
    if any(ctx in context for ctx in energy_contexts):
        return True

    return False


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


# Usage validation
def validate_node_type(node_type: str) -> bool:
    """Validate that a node type is in the enhanced type system."""
    return node_type in ENHANCED_NODE_LABELS


def suggest_better_node_type(
    current_type: str, name: str, category: str = "", context: str = ""
) -> str:
    """Suggest a better node type based on available information."""
    # Priority order for type inference
    suggestions = []

    # Try category-based inference first (most reliable)
    if category:
        category_suggestion = infer_node_type_from_category(category)
        if category_suggestion != "WorldElement":
            suggestions.append((category_suggestion, 10))

    # Try name-based inference
    name_suggestion = infer_node_type_from_name(name, context)
    if name_suggestion != "Entity":
        suggestions.append((name_suggestion, 8))

    # If we have suggestions, return the highest priority one
    if suggestions:
        return max(suggestions, key=lambda x: x[1])[0]

    # If current type is valid, keep it
    if validate_node_type(current_type):
        return current_type

    # Final fallback
    return "Entity"
