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


# --- NARRATIVE-FOCUSED RELATIONSHIP TAXONOMY ---

# Structural Relationships (SAGA system architecture)
STRUCTURAL_RELATIONSHIPS = {
    "HAS_PLOT_POINT",
    "NEXT_PLOT_POINT",
    "HAS_CHARACTER",
    "HAS_WORLD_META",
    "CONTAINS_ELEMENT",
    "DEVELOPED_IN_CHAPTER",
    "ELABORATED_IN_CHAPTER",
}

# Character Social Relationships
CHARACTER_SOCIAL_RELATIONSHIPS = {
    "ALLY_OF",  # Strong positive alliance
    "ENEMY_OF",  # Active antagonism
    "FRIEND_OF",  # Personal friendship
    "RIVAL_OF",  # Competitive relationship
    "FAMILY_OF",  # Blood or adopted family
    "ROMANTIC_WITH",  # Romantic involvement
    "MENTOR_TO",  # Teaching/guidance relationship
    "STUDENT_OF",  # Learning relationship (inverse of mentor)
    "WORKS_FOR",  # Employment/service
    "LEADS",  # Authority/command
    "SERVES",  # Loyal service/allegiance
    "KNOWS",  # Basic acquaintance
    "TRUSTS",  # Trust relationship
    "DISTRUSTS",  # Mistrust relationship
    "OWES_DEBT_TO",  # Obligation relationship
}

# Character Emotional Relationships
CHARACTER_EMOTIONAL_RELATIONSHIPS = {
    "LOVES",  # Deep affection
    "HATES",  # Deep animosity
    "FEARS",  # Fear of person
    "RESPECTS",  # Admiration/respect
    "DESPISES",  # Contempt
    "ENVIES",  # Jealousy/envy
    "PITIES",  # Sympathy/pity
    "OBSESSED_WITH",  # Unhealthy fixation
}

# Plot Causal Relationships
PLOT_CAUSAL_RELATIONSHIPS = {
    "CAUSES",  # Direct causation
    "PREVENTS",  # Active prevention
    "ENABLES",  # Makes possible
    "TRIGGERS",  # Sets in motion
    "RESULTS_IN",  # Consequence of
    "DEPENDS_ON",  # Conditional dependency
    "CONFLICTS_WITH",  # Opposition/conflict
    "SUPPORTS",  # Aids/assists
    "THREATENS",  # Poses danger to
    "PROTECTS",  # Provides protection
}

# Spatial/Temporal Relationships
SPATIAL_TEMPORAL_RELATIONSHIPS = {
    "LOCATED_AT",  # Physical location
    "LOCATED_IN",  # Inside/within
    "NEAR",  # Proximity
    "ADJACENT_TO",  # Next to
    "OCCURS_DURING",  # Temporal overlap
    "HAPPENS_BEFORE",  # Temporal precedence
    "HAPPENS_AFTER",  # Temporal sequence
    "ORIGINATES_FROM",  # Place of origin
    "TRAVELS_TO",  # Movement destination
}

# Possession/Ownership Relationships
POSSESSION_RELATIONSHIPS = {
    "OWNS",  # Legal/practical ownership
    "POSSESSES",  # Physical possession
    "CREATED_BY",  # Creator relationship
    "INHERITED_FROM",  # Inheritance
    "STOLEN_FROM",  # Theft relationship
    "GIVEN_BY",  # Gift relationship
    "FOUND_AT",  # Discovery location
    "LOST_AT",  # Loss location
}

# Organizational Relationships
ORGANIZATIONAL_RELATIONSHIPS = {
    "MEMBER_OF",  # Membership
    "LEADER_OF",  # Leadership role
    "FOUNDED",  # Founder relationship
    "BELONGS_TO",  # Belonging/association
    "REPRESENTS",  # Representative role
    "OPPOSES",  # Organizational opposition
    "ALLIED_WITH",  # Organizational alliance
}

# Physical/Structural Relationships
PHYSICAL_RELATIONSHIPS = {
    "PART_OF",  # Component relationship
    "CONTAINS",  # Container relationship
    "CONNECTED_TO",  # Physical connection
    "BUILT_BY",  # Construction
    "DESTROYED_BY",  # Destruction
    "DAMAGED_BY",  # Partial destruction
    "REPAIRED_BY",  # Restoration
}

# Abstract/Thematic Relationships
THEMATIC_RELATIONSHIPS = {
    "SYMBOLIZES",  # Symbolic representation
    "REPRESENTS",  # Thematic representation
    "CONTRASTS_WITH",  # Thematic opposition
    "PARALLELS",  # Thematic similarity
    "FORESHADOWS",  # Narrative foreshadowing
    "ECHOES",  # Thematic echo/callback
    "EMBODIES",  # Physical manifestation of concept
}

# Ability/Trait Relationships
ABILITY_RELATIONSHIPS = {
    "HAS_ABILITY",  # Possesses skill/power
    "HAS_TRAIT",  # Character trait
    "HAS_GOAL",  # Motivation/objective
    "HAS_RULE",  # Governing principle
    "HAS_KEY_ELEMENT",  # Important component
    "HAS_TRAIT_ASPECT",  # Trait detail
    "SKILLED_IN",  # Competency
    "WEAK_IN",  # Deficiency
}

# Status/State Relationships
STATUS_RELATIONSHIPS = {
    "HAS_STATUS",  # Current state/condition
    "IS_DEAD",  # Death state
    "IS_ALIVE",  # Life state
    "IS_MISSING",  # Absence state
    "IS_INJURED",  # Harm state
    "IS_HEALTHY",  # Wellness state
    "IS_ACTIVE",  # Activity state
    "IS_INACTIVE",  # Inactivity state
}

# Combine all relationship categories
RELATIONSHIP_TYPES = (
    STRUCTURAL_RELATIONSHIPS
    | CHARACTER_SOCIAL_RELATIONSHIPS
    | CHARACTER_EMOTIONAL_RELATIONSHIPS
    | PLOT_CAUSAL_RELATIONSHIPS
    | SPATIAL_TEMPORAL_RELATIONSHIPS
    | POSSESSION_RELATIONSHIPS
    | ORGANIZATIONAL_RELATIONSHIPS
    | PHYSICAL_RELATIONSHIPS
    | THEMATIC_RELATIONSHIPS
    | ABILITY_RELATIONSHIPS
    | STATUS_RELATIONSHIPS
)

# Relationship category mapping for validation and normalization
RELATIONSHIP_CATEGORIES = {
    "structural": STRUCTURAL_RELATIONSHIPS,
    "character_social": CHARACTER_SOCIAL_RELATIONSHIPS,
    "character_emotional": CHARACTER_EMOTIONAL_RELATIONSHIPS,
    "plot_causal": PLOT_CAUSAL_RELATIONSHIPS,
    "spatial_temporal": SPATIAL_TEMPORAL_RELATIONSHIPS,
    "possession": POSSESSION_RELATIONSHIPS,
    "organizational": ORGANIZATIONAL_RELATIONSHIPS,
    "physical": PHYSICAL_RELATIONSHIPS,
    "thematic": THEMATIC_RELATIONSHIPS,
    "ability": ABILITY_RELATIONSHIPS,
    "status": STATUS_RELATIONSHIPS,
}

# Common relationship variations that should be normalized
# Maps variations -> canonical relationship
RELATIONSHIP_NORMALIZATIONS = {
    # Social relationship variations
    "is_friend_of": "FRIEND_OF",
    "friends_with": "FRIEND_OF",
    "befriends": "FRIEND_OF",
    "is_allied_with": "ALLY_OF",
    "allies_with": "ALLY_OF",
    "is_enemy_of": "ENEMY_OF",
    "enemies_with": "ENEMY_OF",
    "antagonizes": "ENEMY_OF",
    "is_rival_of": "RIVAL_OF",
    "rivals_with": "RIVAL_OF",
    "competes_with": "RIVAL_OF",
    # Family variations
    "is_family_of": "FAMILY_OF",
    "related_to": "FAMILY_OF",  # When clearly family context
    "is_parent_of": "FAMILY_OF",
    "is_child_of": "FAMILY_OF",
    "is_sibling_of": "FAMILY_OF",
    # Romance variations
    "in_love_with": "ROMANTIC_WITH",
    "dating": "ROMANTIC_WITH",
    "married_to": "ROMANTIC_WITH",
    "courting": "ROMANTIC_WITH",
    # Authority variations
    "is_leader_of": "LEADS",
    "commands": "LEADS",
    "is_boss_of": "LEADS",
    "supervises": "LEADS",
    "employs": "LEADS",
    "manages": "LEADS",
    "reports_to": "WORKS_FOR",
    "employed_by": "WORKS_FOR",
    # Emotional variations
    "is_in_love_with": "LOVES",
    "adores": "LOVES",
    "despises": "HATES",
    "loathes": "HATES",
    "is_afraid_of": "FEARS",
    "scared_of": "FEARS",
    # Spatial variations
    "is_located_at": "LOCATED_AT",
    "positioned_at": "LOCATED_AT",
    "situated_at": "LOCATED_AT",
    "is_in": "LOCATED_IN",
    "inside": "LOCATED_IN",
    "within": "LOCATED_IN",
    "lives": "LOCATED_IN",
    "resides": "LOCATED_IN",
    "dwells": "LOCATED_IN",
    # Ownership variations
    "possesses": "OWNS",
    "has": "OWNS",  # Context-dependent
    "belongs_to": "OWNED_BY",  # Note: this would be inverse
    # Causal variations
    "leads_to": "CAUSES",
    "results_in": "CAUSES",
    "brings_about": "CAUSES",
    "stops": "PREVENTS",
    "blocks": "PREVENTS",
    "hinders": "PREVENTS",
    "destroys": "DESTROYED_BY",
    "ruins": "DESTROYED_BY",
    # Generic fallbacks - these should be used sparingly
    "is_a": "IS_A",  # Only for true type relationships
    "part_of": "PART_OF",
}

# Inverse relationship mappings
INVERSE_RELATIONSHIPS = {
    "MENTOR_TO": "STUDENT_OF",
    "STUDENT_OF": "MENTOR_TO",
    "LEADS": "WORKS_FOR",
    "WORKS_FOR": "LEADS",
    "OWNS": "OWNED_BY",
    "OWNED_BY": "OWNS",
    "PART_OF": "CONTAINS",
    "CONTAINS": "PART_OF",
    "CREATED_BY": "CREATES",
    "CREATES": "CREATED_BY",
    # Add more as needed
}

# Relationship properties that can be used to add nuance without creating new relationship types
RELATIONSHIP_PROPERTIES = {
    "strength": ["weak", "moderate", "strong", "intense"],
    "duration": ["temporary", "ongoing", "permanent", "unknown"],
    "confidence": ["low", "medium", "high"],
    "public_knowledge": ["secret", "private", "known", "public"],
    "reciprocal": ["mutual", "one_sided"],
    "status": ["active", "inactive", "dormant", "resolved"],
}
