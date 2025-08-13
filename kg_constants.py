# kg_constants.py
"""Constants used for property names and the canonical schema in the knowledge graph."""

# Relationship and node property names
KG_REL_CHAPTER_ADDED = "chapter_added"
KG_NODE_CREATED_CHAPTER = "created_chapter"
KG_NODE_CHAPTER_UPDATED = "chapter_updated"
KG_IS_PROVISIONAL = "is_provisional"


# --- Canonical Schema Definition ---

# Set of known node labels used in the knowledge graph.
NODE_LABELS = {
    "Entity",  # Base label for all nodes
    "NovelInfo",
    "Chapter",
    "Character",
    "WorldElement",
    "WorldContainer",
    "PlotPoint",
    "Trait",
    "ValueNode",  # For literal-like values that need to be nodes
    "DevelopmentEvent",
    "WorldElaborationEvent",
    "Location",
    "Faction",
    "System",  # e.g. Magic System
    "Lore",
}


# Set of known relationship types used in the knowledge graph. This helps
# normalize variations and informs downstream consumers of valid types.
RELATIONSHIP_TYPES = {
    # Structural Relationships
    "HAS_PLOT_POINT",
    "NEXT_PLOT_POINT",
    "HAS_CHARACTER",
    "HAS_WORLD_META",
    "CONTAINS_ELEMENT",
    # Character-related Relationships
    "HAS_TRAIT",
    "DEVELOPED_IN_CHAPTER",
    "KNOWS",
    "FRIEND_OF",
    "RIVAL_OF",
    "MENTOR_OF",
    "PROTEGE_OF",
    "WORKS_FOR",
    "MEMBER_OF",
    "LOVES",
    "HATES",
    "IS_DEAD",
    # WorldElement-related Relationships
    "HAS_GOAL",
    "HAS_RULE",
    "HAS_KEY_ELEMENT",
    "HAS_TRAIT_ASPECT",
    "ELABORATED_IN_CHAPTER",
    # Generic & Dynamic Relationships
    "IS_A",
    "PART_OF",
    "LOCATED_IN",
    "NEAR",
    "HAS_STATUS",
    "HAS_ABILITY",
    "OWNS",
    "RELATED_TO",  # Generic fallback
    "DYNAMIC_REL",  # For KG triples where the relationship type is in a property
}

