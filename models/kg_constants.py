# models/kg_constants.py
"""
Constants used for property names and the canonical schema in the knowledge graph.

**IMPORTANT - PERMISSIVE MODE:**
As of the LangGraph migration, SAGA now operates in PERMISSIVE MODE for entity types
and relationships. The constants defined here serve as:

1. **Suggestions and Examples** - Common patterns for LLMs to reference
2. **Documentation** - A catalog of frequently used types in narratives
3. **Analytics** - For tracking and analyzing emergent patterns

They are **NOT** enforcement mechanisms. The LLM is free to create:
- Novel entity types not listed in NODE_LABELS
- Novel relationship types not listed in RELATIONSHIP_TYPES
- Custom combinations appropriate for the narrative context

This supports creative freedom, genre-specific conventions, and emergent patterns
that weren't anticipated when designing the schema.
"""

# Relationship and node property names
KG_REL_CHAPTER_ADDED = "chapter_added"
KG_NODE_CREATED_CHAPTER = "created_chapter"
KG_NODE_CHAPTER_UPDATED = "chapter_updated"
KG_IS_PROVISIONAL = "is_provisional"


# --- Canonical Schema Definition ---
# NOTE: These are SUGGESTED types, not required types.
# LLMs can create custom types as needed for narrative context.

# Set of commonly used node labels in the knowledge graph.
# The LLM can create new node types beyond this set.
VALID_NODE_LABELS = {
    "Character",
    "Location",
    "Event",
    "Trait",
    "Chapter",
    "Novel",
}

# Map common variations to the canonical labels
LABEL_NORMALIZATION_MAP = {
    "Person": "Character",
    "Creature": "Character",
    "Spirit": "Character",
    "Deity": "Character",
    "Human": "Character",
    "NPC": "Character",
    "Place": "Location",
    "City": "Location",
    "Town": "Location",
    "Building": "Location",
    "Region": "Location",
    "Landmark": "Location",
    "Room": "Location",
    "Object": "Item",
    "Artifact": "Item",
    "Weapon": "Item",
    "Tool": "Item",
    "Device": "Item",
    "Skill": "Trait",
    "Ability": "Trait",
    "Quality": "Trait",
    "Attribute": "Trait",
    "Scene": "Event",
    "Moment": "Event",
    "Incident": "Event",
    "Happening": "Event",
}

# Suggested categories for each valid label to guide the LLM
SUGGESTED_CATEGORIES = {
    "Character": [
        "Protagonist",
        "Antagonist",
        "Supporting",
        "Minor",
        "Background",
        "Historical",
    ],
    "Location": [
        "City",
        "Town",
        "Village",
        "Building",
        "Room",
        "Region",
        "Planet",
        "Landscape",
        "Landmark",
        "locations",
        "location",
    ],
    "Event": [
        "Battle",
        "Meeting",
        "Journey",
        "Ceremony",
        "Discovery",
        "Conflict",
        "Conversation",
        "Flashback",
        "events",
        "event",
    ],
    "Item": [
        "Weapon",
        "Tool",
        "Clothing",
        "Artifact",
        "Document",
        "Vehicle",
        "Resource",
        "Magical",
    ],
    "Trait": [
        "Personality",
        "Physical",
        "Skill",
        "Supernatural",
        "Status",
        "Background",
    ],
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

# Relationship category mapping for validation and normalization
# Generated programmatically from the category sets above
RELATIONSHIP_CATEGORIES = {
    "character_social": CHARACTER_SOCIAL_RELATIONSHIPS,
    "character_emotional": CHARACTER_EMOTIONAL_RELATIONSHIPS,
    "status": STATUS_RELATIONSHIPS,
}

# Combine all relationship categories into a single set
# NOTE: This set contains commonly used relationship types, but is NOT exhaustive.
# The LLM can and should create custom relationship types for narrative precision.
# This set is used for reference and analytics, not enforcement.
RELATIONSHIP_TYPES = set()
for category_set in RELATIONSHIP_CATEGORIES.values():
    RELATIONSHIP_TYPES.update(category_set)
