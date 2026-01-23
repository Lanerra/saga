# models/kg_constants.py
"""Define constants for the knowledge-graph canonical schema.

This module centralizes:
- canonical node labels and label normalization rules,
- relationship type sets used for analytics/normalization, and
- property-key allowlists used for Cypher interpolation hardening.

Notes:
    Node labels are treated as a strict schema surface for writes that interpolate
    labels into Cypher. Unknown/unsupported labels are rejected to avoid schema drift
    and index/constraint mismatch.

    Relationship *type names* may be emergent, but any relationship type that is
    interpolated into Cypher must be validated for safety by the query layer.
    See [`validate_relationship_type_for_cypher_interpolation()`](data_access/kg_queries.py:135).
"""

# Relationship and node property names
KG_REL_CHAPTER_ADDED = "chapter_added"
KG_NODE_CREATED_CHAPTER = "created_chapter"
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
# - The canonical domain label set is the 6 labels below.
# - Subtypes (e.g., Faction, Settlement, Artifact, PlotPoint) MUST be represented via
#   properties (typically `category`), not as Neo4j node labels.
VALID_NODE_LABELS = {
    "Character",
    "Location",
    "Event",
    "Item",
    "Chapter",
    "Scene",
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
    "ALLIES_WITH",
    "FRIEND_OF",
    "RIVALS_WITH",
    "FAMILY_OF",
    "MENTORS",
    "PROTECTS",
    "SEEKS",
    "LOCATED_AT",
    "SERVES",
    "DEPENDS_ON",
    "OWES",
    "DECEIVES",
    "MANIPULATES",
}

# Character Emotional Relationships
CHARACTER_EMOTIONAL_RELATIONSHIPS = {
    "ADMIRES",  # Admiration
    "LOVES",  # Deep affection
    "CONFLICTS_WITH",  # Deep animosity
    "FEARS",  # Fear of person
    "TRUSTS",  # Admiration/respect
    "BETRAYS",  # Unhealthy fixation
    "RESENTS",  # Resentment
    "DISTRUSTS",  # Distrust/suspicion
}

# Character ↔ Event Relationships
CHARACTER_EVENT_RELATIONSHIPS = {
    "PARTICIPATES_IN",
    "WITNESSES",
    "CAUSES",
    "AFFECTED_BY",
}

# Character ↔ Item Relationships
CHARACTER_ITEM_RELATIONSHIPS = {
    "OWNS",
    "CARRIES",
    "WIELDS",
    "SEEKS",
    "GUARDS",
    "CREATED",
}

# Character ↔ Location Relationships (Extended)
CHARACTER_LOCATION_RELATIONSHIPS = {
    "LOCATED_AT",
    "ORIGINATES_FROM",
    "RULES",
    "OWNS_LOCATION",
    "SEEKS_LOCATION",
    "EXILED_FROM",
}

# Event ↔ Location Relationships
EVENT_LOCATION_RELATIONSHIPS = {
    "OCCURS_AT",
    "ORIGINATES_FROM_LOCATION",
    "AFFECTS_LOCATION",
}

# Event ↔ Item Relationships
EVENT_ITEM_RELATIONSHIPS = {
    "INVOLVES_ITEM",
    "CREATES_ITEM",
    "DESTROYS_ITEM",
    "TRANSFERS_ITEM",
}

# Event ↔ Event Relationships (Temporal)
EVENT_TEMPORAL_RELATIONSHIPS = {
    "HAPPENS_BEFORE",
    "HAPPENS_AFTER",
    "OCCURS_DURING",
    "CAUSES_EVENT",
    "INTERRUPTS",
    "PART_OF",
}

# Location ↔ Location Relationships (Spatial)
LOCATION_SPATIAL_RELATIONSHIPS = {
    "CONTAINS_LOCATION",
    "PART_OF_LOCATION",
    "BORDERS",
    "CONNECTED_TO",
    "VISIBLE_FROM",
}

# Location ↔ Item Relationships
LOCATION_ITEM_RELATIONSHIPS = {
    "CONTAINS_ITEM",
    "PRODUCES",
    "REQUIRES_ITEM",
}

# Scene ↔ Scene Relationships (Sequential)
SCENE_SEQUENTIAL_RELATIONSHIPS = {
    "FOLLOWS",
}

# Scene ↔ Chapter Relationships (Structural)
SCENE_CHAPTER_RELATIONSHIPS = {
    "PART_OF",
}

# Scene ↔ Event Relationships
SCENE_EVENT_RELATIONSHIPS = {
    "OCCURS_IN_SCENE",
}

# Scene ↔ Location Relationships
SCENE_LOCATION_RELATIONSHIPS = {
    "OCCURS_AT",
}

# Relationship category mapping for validation and normalization
# Generated programmatically from the category sets above
RELATIONSHIP_CATEGORIES = {
    "character_social": CHARACTER_SOCIAL_RELATIONSHIPS,
    "character_emotional": CHARACTER_EMOTIONAL_RELATIONSHIPS,
    "character_event": CHARACTER_EVENT_RELATIONSHIPS,
    "character_item": CHARACTER_ITEM_RELATIONSHIPS,
    "character_location": CHARACTER_LOCATION_RELATIONSHIPS,
    "event_location": EVENT_LOCATION_RELATIONSHIPS,
    "event_item": EVENT_ITEM_RELATIONSHIPS,
    "event_temporal": EVENT_TEMPORAL_RELATIONSHIPS,
    "location_spatial": LOCATION_SPATIAL_RELATIONSHIPS,
    "location_item": LOCATION_ITEM_RELATIONSHIPS,
    "scene_sequential": SCENE_SEQUENTIAL_RELATIONSHIPS,
    "scene_chapter": SCENE_CHAPTER_RELATIONSHIPS,
    "scene_event": SCENE_EVENT_RELATIONSHIPS,
    "scene_location": SCENE_LOCATION_RELATIONSHIPS,
}

# Combine all relationship categories into a single set
RELATIONSHIP_TYPES = set()
for category_set in RELATIONSHIP_CATEGORIES.values():
    RELATIONSHIP_TYPES.update(category_set)

# Static relationship mapping for exact synonyms
STATIC_RELATIONSHIP_MAP: dict[str, str] = {
    "HATES": "CONFLICTS_WITH",
    "DESPISES": "CONFLICTS_WITH",
    "LOATHES": "CONFLICTS_WITH",
    "DISLIKES": "CONFLICTS_WITH",
}

# Relationships that should be node properties instead of relationships
# NOTE: This set is now empty as status-related relationships have been deprecated.
# Status information (alive/dead/injured/active) is stored as properties on Character nodes.
PROPERTY_RELATIONSHIPS: set[str] = set()


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
