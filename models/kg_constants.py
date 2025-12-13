# models/kg_constants.py
"""
Constants used for property names and the canonical schema in the knowledge graph.

**Schema enforcement policy (contract):**

- **Node labels (STRICT for Cypher label interpolation / writes):**
  The application enforces a canonical set of node labels for entities when building
  Cypher queries that interpolate labels (e.g., via [`_get_cypher_labels()`](data_access/kg_queries.py:262)).
  Unknown or unsupported labels are rejected with a `ValueError` rather than silently
  creating new labels in the graph.

  Rationale: labels participate in Neo4j constraints/indexes and are commonly relied
  upon by query patterns; allowing arbitrary labels risks schema drift and unexpected
  query behavior.

- **Relationship types (PERMISSIVE membership, STRICT safety):**
  Relationship types are allowed to be novel (not present in [`RELATIONSHIP_TYPES`](models/kg_constants.py:211)),
  but any relationship type that is *directly interpolated into Cypher* must pass
  strict safety validation (uppercase and matching `^[A-Z0-9_]+$`) via
  [`validate_relationship_type_for_cypher_interpolation()`](data_access/kg_queries.py:135).

In other words: labels are a strict schema surface; relationship-type *names* may be
emergent, but must be safe for Cypher interpolation when used in queries.
"""

# Relationship and node property names
KG_REL_CHAPTER_ADDED = "chapter_added"
KG_NODE_CREATED_CHAPTER = "created_chapter"
KG_NODE_CHAPTER_UPDATED = "chapter_updated"
KG_IS_PROVISIONAL = "is_provisional"

# --- World item label taxonomy (P0.2) ---
# Canonical world "entity" labels that the system should use consistently across
# builder upserts, builder fetches, and read-by-id lookups.
#
# NOTE: This is intentionally narrower than VALID_NODE_LABELS; world items are a
# subset of the KG schema (Character/Trait/Chapter are not "world items").
WORLD_ITEM_CANONICAL_LABELS: tuple[str, ...] = (
    "Location",
    "Item",
    "Event",
    "Organization",
    "Concept",
)

# Legacy labels seen in older graphs and legacy read paths. These are supported
# for backwards compatibility in read/fetch predicates (but should not be newly
# written as primary labels by upsert).
WORLD_ITEM_LEGACY_LABELS: tuple[str, ...] = (
    "Object",
    "Artifact",
    "Document",
    "Relic",
)


# --- Canonical Schema Definition ---
#
# Contract (SAGA labeling strategy):
# - Node labels are a strict schema surface. Application code MUST NOT create arbitrary
#   new labels in the graph (prevents schema drift / index & constraint mismatch).
# - The canonical domain label set is the 9 labels below.
# - Subtypes (e.g., Faction, Settlement, Artifact, PlotPoint) MUST be represented via
#   properties (typically `category`), not as Neo4j node labels.
VALID_NODE_LABELS = {
    "Character",
    "Location",
    "Event",
    "Item",
    "Organization",
    "Concept",
    "Trait",
    "Chapter",
    "Novel",
}

# Map common variations / legacy labels / subtype "types" to canonical labels.
#
# Notes:
# - Keep this map intentionally explicit and auditable.
# - We include some lowercase variants because upstream may supply either casing.
LABEL_NORMALIZATION_MAP: dict[str, str] = {
    # ----------------------------
    # Character-like
    # ----------------------------
    "Person": "Character",
    "person": "Character",
    "Creature": "Character",
    "creature": "Character",
    "Spirit": "Character",
    "spirit": "Character",
    "Deity": "Character",
    "deity": "Character",
    "Human": "Character",
    "human": "Character",
    "NPC": "Character",
    "npc": "Character",

    # ----------------------------
    # Location-like subtypes
    # ----------------------------
    "Place": "Location",
    "place": "Location",
    "City": "Location",
    "city": "Location",
    "Town": "Location",
    "town": "Location",
    "Building": "Location",
    "building": "Location",
    "Settlement": "Location",
    "settlement": "Location",
    "Structure": "Location",
    "structure": "Location",
    "Region": "Location",
    "region": "Location",
    "Landmark": "Location",
    "landmark": "Location",
    "Room": "Location",
    "room": "Location",
    "Path": "Location",
    "path": "Location",
    "Territory": "Location",
    "territory": "Location",

    # ----------------------------
    # Legacy object-ish labels -> Item
    # ----------------------------
    "Object": "Item",
    "object": "Item",
    "Artifact": "Item",
    "artifact": "Item",
    "Document": "Item",
    "document": "Item",
    "Relic": "Item",
    "relic": "Item",

    # Common item-ish subtypes
    "Weapon": "Item",
    "weapon": "Item",
    "Tool": "Item",
    "tool": "Item",
    "Device": "Item",
    "device": "Item",
    "Resource": "Item",
    "resource": "Item",
    "Currency": "Item",
    "currency": "Item",

    # ----------------------------
    # Organization-like subtypes
    # ----------------------------
    "Organization": "Organization",
    "organization": "Organization",
    "Faction": "Organization",
    "faction": "Organization",
    "Guild": "Organization",
    "guild": "Organization",
    "House": "Organization",
    "house": "Organization",
    "Order": "Organization",
    "order": "Organization",
    "Council": "Organization",
    "council": "Organization",

    # ----------------------------
    # Concept-like
    # ----------------------------
    "Concept": "Concept",
    "concept": "Concept",

    # ----------------------------
    # Trait-like
    # ----------------------------
    "Skill": "Trait",
    "skill": "Trait",
    "Ability": "Trait",
    "ability": "Trait",
    "Quality": "Trait",
    "quality": "Trait",
    "Attribute": "Trait",
    "attribute": "Trait",

    # ----------------------------
    # Event-like subtypes
    # ----------------------------
    "Scene": "Event",
    "scene": "Event",
    "Moment": "Event",
    "moment": "Event",
    "Incident": "Event",
    "incident": "Event",
    "Happening": "Event",
    "happening": "Event",
    "PlotPoint": "Event",
    "plotpoint": "Event",
    "DevelopmentEvent": "Event",
    "developmentevent": "Event",
    "WorldElaborationEvent": "Event",
    "worldelaborationevent": "Event",
    "Era": "Event",
    "era": "Event",
    "Timeline": "Event",
    "timeline": "Event",
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
    # Subtypes are stored in `category` and are intentionally open-ended.
    # This list is advisory (soft validation) and should match what prompts
    # actively instruct the model to emit.
    "Location": [
        # Common location kinds
        "City",
        "Town",
        "Village",
        "Settlement",
        "Building",
        "Structure",
        "Room",
        "Region",
        "Landscape",
        "Landmark",
        "Territory",
        "Path",
        "Planet",
        "CelestialBody",
        # Legacy/variant bucket names sometimes seen in older outputs
        "locations",
        "location",
    ],
    "Event": [
        # Common narrative event kinds
        "Scene",
        "Battle",
        "Meeting",
        "Journey",
        "Travel",
        "Ceremony",
        "Discovery",
        "Conflict",
        "Conversation",
        "Flashback",
        # Narrative/KG subtypes (still Events; subtype lives in category)
        "PlotPoint",
        "DevelopmentEvent",
        "WorldElaborationEvent",
        "Era",
        "Moment",
        # Legacy/variant bucket names sometimes seen in older outputs
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


# --- NovelInfo property allowlist (P0.4: Cypher injection hardening) ---
# Neo4j/Cypher does not support parameterizing property keys, so any property-key
# used in a query must be strictly allowlisted before being interpolated.
#
# Keep this list small and auditable. If you add new NovelInfo properties that
# need to be read dynamically, update this allowlist and add a test.
NOVEL_INFO_ALLOWED_PROPERTY_KEYS: frozenset[str] = frozenset(
    {
        # Common novel metadata
        "title",
        "genre",
        "setting",
        # High-level story guidance
        "theme",
        "central_conflict",
        "thematic_progression",
    }
)
