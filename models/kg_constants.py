# models/kg_constants.py
"""Constants used for property names and the canonical schema in the knowledge graph."""

# Relationship and node property names
KG_REL_CHAPTER_ADDED = "chapter_added"
KG_NODE_CREATED_CHAPTER = "created_chapter"
KG_NODE_CHAPTER_UPDATED = "chapter_updated"
KG_IS_PROVISIONAL = "is_provisional"


# --- Canonical Schema Definition ---

# Set of known node labels used in the knowledge graph.
NODE_LABELS = {
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
    "Signal",  # Electronic/magical signals, transmissions
    "ValueNode",  # Literal values, data (existing)
    "Record",  # Official records, documentation
    # === ACTION/EVENT ENTITIES ===
    "Action",  # Specific actions, deeds, activities
    "Reaction",  # Responses, reactions to events
    "Change",  # Transformations, alterations
    "Pattern",  # Behavioral patterns, recurring structures
    # === PHYSICAL/SENSORY ENTITIES ===
    "Sound",  # Auditory phenomena, noise, music
    "Pollen",  # Biological particles, airborne substances
    # === PURPOSE/INTENT ENTITIES ===
    "Purpose",  # Intentions, goals, objectives
    "Goal",  # Specific targets, aims, desired outcomes
    "Outcome",  # Results, consequences, end states
    # === RELATIONSHIP/ABSTRACT ===
    "Relationship",  # Social/conceptual connections (meta)
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
    # === LEGACY SUPPORT (gated at runtime) ===
    # "WorldElement",  # Legacy type - being phased out
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
    "IS_A",  # Type relationship
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
    "TEACHES",  # Instruction/education relationship
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
    "OCCURRED_IN",  # Event occurrence in a chapter/time
}

# Possession/Ownership Relationships
POSSESSION_RELATIONSHIPS = {
    "OWNS",  # Legal/practical ownership
    "POSSESSES",  # Physical possession
    "CREATED_BY",  # Creator relationship
    "CREATES",  # Active creation (inverse of CREATED_BY)
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
    "WORKS_AS",  # Employment role or position
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
    "OWNED_BY",  # Ownership (inverse of OWNS)
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
    "STATUS_IS",  # Alternative form of HAS_STATUS (legacy support)
    "IS_DEAD",  # Death state
    "IS_ALIVE",  # Life state
    "IS_MISSING",  # Absence state
    "IS_INJURED",  # Harm state
    "IS_HEALTHY",  # Wellness state
    "IS_ACTIVE",  # Activity state
    "IS_INACTIVE",  # Inactivity state
}

# Information and Recording Relationships
INFORMATION_RELATIONSHIPS = {
    "RECORDS",  # Recording or documenting information
    "PRESERVES",  # Preservation or archival relationship
    "HAS_METADATA",  # Contains metadata or descriptive information
    "DISCOVERS",  # Discovery of information, objects, or entities
    "REVEALS",  # Disclosure or revelation of information
    "IDENTIFIES",  # Recognition or identification of entities
}

# Usage and Accessibility Relationships
USAGE_RELATIONSHIPS = {
    "ACCESSIBLE_BY",  # Accessibility relationship
    "USED_IN",  # Usage in events or contexts
    "TARGETS",  # Targeting or directing toward something
}

# Communication and Display Relationships
COMMUNICATION_RELATIONSHIPS = {
    "DISPLAYS",  # Display or presentation of information
    "SHOWS",  # Demonstration or exhibition of information/objects
    "SPOKEN_BY",  # Communication originating from sentient beings
    "EMITS",  # Emission of energy, sound, or information
}

# Operational Relationships
OPERATIONAL_RELATIONSHIPS = {
    "EMPLOYS",  # Employment or hiring relationship
    "CONTROLS",  # Control or management relationship
    "REQUIRES",  # Dependency or requirement relationship
    "RUNS",  # Execution or operation of systems/processes
    "OPERATES",  # Operation or piloting of vehicles/systems
    "FLIES",  # Piloting or operating aircraft/spacecraft
    "SEALS",  # Securing, closing, or protecting
    "PULLS",  # Extraction, retrieval, or accessing data/objects
}

# Enhanced Temporal and State Relationships
ENHANCED_TEMPORAL_RELATIONSHIPS = {
    "REPLACED_BY",  # Replacement or succession relationship
    "LINKED_TO",  # Connection or linkage relationship
}

# Status and State Change Relationships
STATUS_CHANGE_RELATIONSHIPS = {
    "WAS_REPLACED_BY",  # Past replacement relationship
    "CHARACTERIZED_BY",  # Characterized or defined by traits
    "IS_NOW",  # Current role or status
    "IS_NO_LONGER",  # Former role or status
    "DIFFERS_FROM",  # Difference or distinction relationship
}

# Special Action Relationships
SPECIAL_ACTION_RELATIONSHIPS = {
    "WHISPERS",  # Quiet communication
    "WORE",  # Past wearing/carrying
    "DEPRECATED",  # Marked as obsolete
}

# Enhanced Association Relationships
ENHANCED_ASSOCIATION_RELATIONSHIPS = {
    "ASSOCIATED_WITH",  # General association relationship
}

# Relationship category mapping for validation and normalization
# Generated programmatically from the category sets above
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
    "information": INFORMATION_RELATIONSHIPS,
    "usage": USAGE_RELATIONSHIPS,
    "communication": COMMUNICATION_RELATIONSHIPS,
    "operational": OPERATIONAL_RELATIONSHIPS,
    "enhanced_temporal": ENHANCED_TEMPORAL_RELATIONSHIPS,
    "status_change": STATUS_CHANGE_RELATIONSHIPS,
    "special_action": SPECIAL_ACTION_RELATIONSHIPS,
    "enhanced_association": ENHANCED_ASSOCIATION_RELATIONSHIPS,
}

# Combine all relationship categories into a single set
# This is the single source of truth for all valid relationship types
RELATIONSHIP_TYPES = set()
for category_set in RELATIONSHIP_CATEGORIES.values():
    RELATIONSHIP_TYPES.update(category_set)

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
    # New relationship normalizations
    "presents": "DISPLAYS",
    "shows": "SHOWS",
    "demonstrates": "SHOWS",
    "exhibits": "SHOWS",
    "reveals": "REVEALS",
    "discloses": "REVEALS",
    "discovers": "DISCOVERS",
    "finds": "DISCOVERS",
    "uncovers": "DISCOVERS",
    "identifies": "IDENTIFIES",
    "recognizes": "IDENTIFIES",
    "detects": "IDENTIFIES",
    "said_by": "SPOKEN_BY",
    "uttered_by": "SPOKEN_BY",
    "voiced_by": "SPOKEN_BY",
    "radiates": "EMITS",
    "sends_out": "EMITS",
    "produces": "EMITS",
    "hires": "EMPLOYS",
    "takes_on": "EMPLOYS",
    # 'manages' canonicalized above to LEADS
    "operates": "OPERATES",
    "pilots": "FLIES",
    "navigates": "FLIES",
    "runs": "RUNS",
    "executes": "RUNS",
    "seals": "SEALS",
    "secures": "SEALS",
    "closes": "SEALS",
    "pulls": "PULLS",
    "extracts": "PULLS",
    "retrieves": "PULLS",
    "teaches": "TEACHES",
    "instructs": "TEACHES",
    "educates": "TEACHES",
    "works_as": "WORKS_AS",
    "serves_as": "WORKS_AS",
    # 'commands' canonicalized above to LEADS
    "needs": "REQUIRES",
    "depends_on": "REQUIRES",
    "substituted_by": "REPLACED_BY",
    "succeeded_by": "REPLACED_BY",
    "connected_to": "LINKED_TO",
    "joined_to": "LINKED_TO",
    # 'related_to' canonicalized above to FAMILY_OF
    "affiliated_with": "ASSOCIATED_WITH",
    # Information relationship normalizations
    "documents": "RECORDS",
    "logs": "RECORDS",
    "saves": "PRESERVES",
    "keeps": "PRESERVES",
    "contains_info": "HAS_METADATA",
    "describes": "HAS_METADATA",
    # Usage relationship normalizations
    "used_by": "ACCESSIBLE_BY",
    "available_to": "ACCESSIBLE_BY",
    "applied_in": "USED_IN",
    "utilized_in": "USED_IN",
    "aims_at": "TARGETS",
    "directed_at": "TARGETS",
    # Status relationship normalizations
    "status_is": "HAS_STATUS",
    "current_status": "HAS_STATUS",
    "state_is": "HAS_STATUS",
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
