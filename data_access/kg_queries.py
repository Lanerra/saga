# data_access/kg_queries.py
import hashlib
import re
from typing import Any

import structlog
from async_lru import alru_cache

import config
from core.db_manager import neo4j_manager
from core.schema_validator import validate_node_labels
from models.kg_constants import (
    KG_IS_PROVISIONAL,
    KG_REL_CHAPTER_ADDED,
    NODE_LABELS,
    RELATIONSHIP_TYPES,
)

logger = structlog.get_logger(__name__)

# Cache to prevent repeated type upgrade logging for the same entity
_upgrade_logged = set()

# Valid relationship types for narrative knowledge graphs - use canonical constants
# NOTE: This is used for informational logging only, not enforcement.
# In permissive mode, novel relationship types are allowed and encouraged.
VALID_RELATIONSHIP_TYPES = RELATIONSHIP_TYPES

# Lookup table for canonical node labels to ensure consistent casing
_CANONICAL_NODE_LABEL_MAP: dict[str, str] = {lbl.lower(): lbl for lbl in NODE_LABELS}
_CANONICAL_NODE_LABEL_MAP.pop("worldelement", None)


# Preserve original static implementation for fallback
def _infer_specific_node_type_static(
    name: str, category: str = "", fallback_type: str = "Entity"
) -> str:
    """
    Original static node type inference (preserved as fallback).
    Upgrades generic types like 'WorldElement' to specific types like 'Artifact', 'Location', etc.
    """
    name_lower = name.lower()
    category_lower = category.lower()

    # Living beings - check name patterns
    if any(
        term in name_lower
        for term in [
            "king",
            "queen",
            "lord",
            "lady",
            "prince",
            "princess",
            "duke",
            "duchess",
            "baron",
            "earl",
            "knight",
            "sir",
            "master",
            "apprentice",
            "guard",
            "soldier",
            "captain",
            "general",
            "warrior",
            "hero",
            "villain",
            "mage",
            "wizard",
            "witch",
            "priest",
            "sage",
            "scholar",
            "merchant",
            "trader",
            "blacksmith",
            "farmer",
            "hunter",
        ]
    ):
        return "Character"

    if any(term in name_lower for term in ["god", "goddess", "deity", "divine"]):
        return "Deity"

    if any(
        term in name_lower
        for term in ["spirit", "ghost", "phantom", "wraith", "spectre", "soul"]
    ):
        return "Spirit"

    if any(
        term in name_lower
        for term in [
            "dragon",
            "beast",
            "monster",
            "creature",
            "wolf",
            "bear",
            "lion",
            "eagle",
            "serpent",
            "snake",
            "spider",
            "demon",
        ]
    ):
        return "Creature"

    # Locations and structures
    if any(
        term in name_lower
        for term in [
            "castle",
            "tower",
            "fortress",
            "palace",
            "manor",
            "hall",
            "temple",
            "church",
            "cathedral",
            "monastery",
            "house",
            "building",
            "bridge",
            "gate",
            "wall",
            "keep",
        ]
    ):
        return "Structure"

    if any(
        term in name_lower
        for term in [
            "city",
            "town",
            "village",
            "settlement",
            "hamlet",
            "capital",
            "metropolis",
        ]
    ):
        return "Settlement"

    if any(
        term in name_lower
        for term in [
            "forest",
            "mountain",
            "river",
            "lake",
            "sea",
            "ocean",
            "desert",
            "valley",
            "plain",
            "hill",
            "peak",
            "realm",
            "kingdom",
            "empire",
            "land",
            "territory",
            "region",
            "continent",
            "island",
            "cave",
            "cavern",
        ]
    ):
        return (
            "Location"
            if any(
                term in name_lower
                for term in [
                    "forest",
                    "mountain",
                    "river",
                    "lake",
                    "sea",
                    "ocean",
                    "desert",
                    "valley",
                    "plain",
                    "hill",
                    "peak",
                    "cave",
                    "cavern",
                ]
            )
            else "Region"
        )

    if any(
        term in name_lower
        for term in ["road", "path", "route", "trail", "passage", "corridor", "tunnel"]
    ):
        return "Path"

    if any(
        term in name_lower
        for term in [
            "room",
            "chamber",
            "quarters",
            "hall",
            "parlor",
            "study",
            "library",
            "kitchen",
            "bedroom",
            "attic",
            "cellar",
            "dungeon",
        ]
    ):
        return "Room"

    # Objects and items
    if any(
        term in name_lower
        for term in [
            "sword",
            "blade",
            "dagger",
            "knife",
            "axe",
            "mace",
            "hammer",
            "bow",
            "arrow",
            "spear",
            "lance",
            "shield",
            "armor",
            "helmet",
            "weapon",
            "tool",
            "staff",
            "rod",
            "wand",
        ]
    ):
        # Check if it's magical/special
        if any(
            term in name_lower
            for term in [
                "magic",
                "magical",
                "enchanted",
                "cursed",
                "ancient",
                "legendary",
                "divine",
                "sacred",
                "holy",
                "blessed",
                "artifact",
                "relic",
            ]
        ):
            return "Artifact"
        return "Object"

    if any(
        term in name_lower
        for term in [
            "book",
            "scroll",
            "tome",
            "grimoire",
            "manual",
            "journal",
            "diary",
            "letter",
            "document",
            "manuscript",
            "note",
            "parchment",
            "text",
        ]
    ):
        return "Document"

    if any(
        term in name_lower
        for term in [
            "ring",
            "crown",
            "amulet",
            "pendant",
            "necklace",
            "bracelet",
            "orb",
            "crystal",
            "gem",
            "stone",
            "jewel",
            "treasure",
            "artifact",
            "relic",
        ]
    ):
        return (
            "Artifact"
            if any(
                term in name_lower
                for term in [
                    "magic",
                    "magical",
                    "ancient",
                    "legendary",
                    "divine",
                    "sacred",
                    "artifact",
                    "relic",
                ]
            )
            else "Object"
        )

    if any(
        term in name_lower
        for term in [
            "gold",
            "silver",
            "copper",
            "coin",
            "money",
            "currency",
            "payment",
            "tribute",
            "tax",
        ]
    ):
        return "Currency"

    if any(
        term in name_lower
        for term in [
            "wood",
            "stone",
            "metal",
            "iron",
            "steel",
            "bronze",
            "material",
            "ore",
            "resource",
            "supply",
            "fuel",
            "food",
            "water",
            "grain",
            "bread",
            "meat",
        ]
    ):
        return (
            "Resource"
            if not any(
                term in name_lower
                for term in [
                    "food",
                    "water",
                    "grain",
                    "bread",
                    "meat",
                    "wine",
                    "ale",
                    "beer",
                ]
            )
            else "Food"
        )

    # Organizations
    if any(
        term in name_lower
        for term in [
            "guild",
            "order",
            "brotherhood",
            "sisterhood",
            "council",
            "court",
            "assembly",
            "faction",
            "clan",
            "tribe",
            "house",
            "family",
            "dynasty",
            "organization",
            "group",
            "band",
            "company",
            "army",
            "legion",
            "guard",
        ]
    ):
        if any(term in name_lower for term in ["guild"]):
            return "Guild"
        elif any(term in name_lower for term in ["order", "brotherhood", "sisterhood"]):
            return "Order"
        elif any(term in name_lower for term in ["council", "court", "assembly"]):
            return "Council"
        elif any(term in name_lower for term in ["house", "family", "dynasty"]):
            return "House"
        else:
            return "Faction"

    # Abstract concepts
    if any(
        term in name_lower
        for term in [
            "magic",
            "spell",
            "enchantment",
            "curse",
            "blessing",
            "power",
            "ability",
            "skill",
            "talent",
            "gift",
            "knowledge",
            "wisdom",
            "lore",
            "legend",
            "myth",
            "story",
            "tale",
            "prophecy",
            "vision",
            "dream",
            "memory",
            "tradition",
            "custom",
            "ritual",
            "ceremony",
            "law",
            "rule",
            "code",
            "principle",
        ]
    ):
        if any(
            term in name_lower
            for term in ["magic", "spell", "enchantment", "curse", "blessing", "power"]
        ):
            return "Magic"
        elif any(term in name_lower for term in ["skill", "talent", "gift", "ability"]):
            return "Skill"
        elif any(
            term in name_lower
            for term in ["knowledge", "wisdom", "lore", "legend", "myth"]
        ):
            return "Lore"
        elif any(term in name_lower for term in ["story", "tale"]):
            return "Story"
        elif any(term in name_lower for term in ["prophecy", "vision", "dream"]):
            return "Dream"
        elif any(term in name_lower for term in ["memory"]):
            return "Memory"
        elif any(
            term in name_lower for term in ["tradition", "custom", "ritual", "ceremony"]
        ):
            return "Tradition"
        elif any(term in name_lower for term in ["law", "rule", "code", "principle"]):
            return "Law"
        else:
            return "Concept"

    # Systems
    if any(
        term in name_lower
        for term in ["religion", "faith", "belief", "worship", "church", "cult"]
    ):
        return "Religion"

    if any(
        term in name_lower
        for term in ["culture", "society", "civilization", "people", "folk"]
    ):
        return "Culture"

    if any(
        term in name_lower
        for term in [
            "government",
            "kingdom",
            "empire",
            "republic",
            "democracy",
            "monarchy",
            "dictatorship",
        ]
    ):
        return "Government"

    if any(
        term in name_lower
        for term in [
            "technology",
            "invention",
            "device",
            "machine",
            "engine",
            "apparatus",
        ]
    ):
        return "Technology"

    # Use category information if available (expanded mapping)
    if category_lower:
        if any(
            term in category_lower
            for term in ["character", "person", "people", "npc", "hero", "villain"]
        ):
            return "Character"
        elif any(
            term in category_lower
            for term in [
                "location",
                "place",
                "city",
                "town",
                "village",
                "region",
                "territory",
                "landmark",
                "path",
                "room",
                "settlement",
            ]
        ):
            return "Location"
        elif any(
            term in category_lower
            for term in [
                "object",
                "item",
                "thing",
                "artifact",
                "relic",
                "weapon",
                "armor",
                "tool",
            ]
        ):
            return "Object"
        elif any(
            term in category_lower for term in ["organization", "faction", "group"]
        ):
            return "Faction"
        elif any(
            term in category_lower
            for term in [
                "concept",
                "idea",
                "abstract",
                "law",
                "tradition",
                "language",
                "symbol",
                "story",
                "song",
                "dream",
                "memory",
                "emotion",
                "skill",
            ]
        ):
            return "Concept"
        elif any(term in category_lower for term in ["event", "happening"]):
            return "Event"
        elif any(term in category_lower for term in ["system"]):
            return "System"

    # Name-based heuristics for common types
    if (
        name_lower.endswith(" temple")
        or name_lower.endswith(" hall")
        or name_lower.endswith(" keep")
    ):
        return "Structure"
    if name_lower.startswith("the ") and any(
        term in name_lower for term in [" inn", " tavern", " bar", " guild", " order"]
    ):
        return "Structure"
    if any(term in name_lower for term in ["district", "ward", "quarter"]):
        return "Region"
    if any(
        term in name_lower
        for term in ["empire", "kingdom", "duchy", "republic", "union"]
    ):
        return "Organization"
    if any(term in name_lower for term in ["university", "academy", "school"]):
        return "Education"
    if any(term in name_lower for term in ["church", "temple", "cult"]):
        return "Religion"
    if any(
        term in name_lower
        for term in ["sword", "bow", "axe", "shield", "armor", "ring", "amulet"]
    ):
        return "Artifact"
    if any(term in name_lower for term in ["road", "path", "bridge"]):
        return "Path"

    # Return fallback if no specific type inferred
    return fallback_type if fallback_type != "WorldElement" else "Object"


def _infer_specific_node_type(
    name: str, category: str = "", fallback_type: str = "Entity"
) -> str:
    """
    Node type inference using static rules.

    Simplified implementation for single-user deployment that directly uses
    static inference without dynamic schema overhead.
    """
    if not name or not name.strip():
        return fallback_type if fallback_type != "WorldElement" else "Entity"

    # Direct static inference - no dynamic fallback needed
    return _infer_specific_node_type_static(name, category, fallback_type)


def _to_pascal_case(text: str) -> str:
    """Convert underscore or space separated text to PascalCase."""
    parts = re.split(r"[_\s]+", text.strip())
    return "".join(part[:1].upper() + part[1:] for part in parts if part)


def validate_relationship_type(proposed_type: str) -> str:
    """
    Normalize a relationship type to a consistent format.

    Args:
        proposed_type: The relationship type to normalize

    Returns:
        Normalized relationship type (uppercase, underscores instead of spaces)
    """
    if not proposed_type or not proposed_type.strip():
        return "RELATES_TO"

    # Clean and normalize: strip whitespace, uppercase, replace spaces with underscores
    return proposed_type.strip().upper().replace(" ", "_")


def normalize_relationship_type(rel_type: str) -> str:
    """Return a canonical representation of a relationship type using predefined taxonomy."""
    # Use the enhanced validator which handles all the normalization logic
    return validate_relationship_type(rel_type)


def _find_best_relationship_match(rel_type: str) -> str | None:
    """Find the best matching relationship type using simple keyword matching."""

    rel_lower = rel_type.lower()

    # Define keyword mappings for common patterns
    keyword_mappings = {
        # Social relationships
        ("friend", "befriend"): "FRIEND_OF",
        ("enemy", "antagonize", "oppose"): "ENEMY_OF",
        ("ally", "allied"): "ALLY_OF",
        ("rival", "compete"): "RIVAL_OF",
        ("family", "related", "parent", "child", "sibling"): "FAMILY_OF",
        ("love", "romantic", "dating", "married"): "ROMANTIC_WITH",
        ("mentor", "teach", "guide"): "MENTOR_TO",
        ("student", "learn"): "STUDENT_OF",
        ("work", "employ", "job"): "WORKS_FOR",
        ("lead", "command", "boss", "supervise"): "LEADS",
        ("serve", "loyal"): "SERVES",
        ("know", "acquaint"): "KNOWS",
        ("trust"): "TRUSTS",
        ("distrust", "mistrust"): "DISTRUSTS",
        # Emotional relationships
        ("hate", "loath", "despise"): "HATES",
        ("fear", "afraid", "scare"): "FEARS",
        ("respect", "admire"): "RESPECTS",
        ("envy", "jealous"): "ENVIES",
        ("pity", "sympathy"): "PITIES",
        # Causal relationships
        ("cause", "lead to", "result"): "CAUSES",
        ("prevent", "stop", "block"): "PREVENTS",
        ("enable", "allow"): "ENABLES",
        ("trigger", "start"): "TRIGGERS",
        ("depend", "require"): "DEPENDS_ON",
        ("conflict", "clash"): "CONFLICTS_WITH",
        ("support", "help", "aid"): "SUPPORTS",
        ("threaten", "danger"): "THREATENS",
        ("protect", "guard", "defend"): "PROTECTS",
        # Spatial relationships
        ("located", "position", "situated"): "LOCATED_AT",
        ("inside", "within", "contained"): "LOCATED_IN",
        ("near", "close", "proximity"): "NEAR",
        ("adjacent", "next"): "ADJACENT_TO",
        ("origin", "from"): "ORIGINATES_FROM",
        ("travel", "move", "go"): "TRAVELS_TO",
        # Ownership relationships
        ("own", "possess", "have", "belong"): "OWNS",
        ("create", "make", "build"): "CREATED_BY",
        ("inherit"): "INHERITED_FROM",
        ("steal", "rob"): "STOLEN_FROM",
        ("give", "gift"): "GIVEN_BY",
        ("find", "discover"): "FOUND_AT",
        # Organizational relationships
        ("member", "join"): "MEMBER_OF",
        ("leader", "head"): "LEADER_OF",
        ("found", "establish"): "FOUNDED",
        ("represent"): "REPRESENTS",
        # Physical relationships
        ("part", "component"): "PART_OF",
        ("contain", "hold"): "CONTAINS",
        ("connect", "link"): "CONNECTED_TO",
        ("destroy", "ruin"): "DESTROYED_BY",
        ("damage", "harm"): "DAMAGED_BY",
        ("repair", "fix"): "REPAIRED_BY",
        # Additional mappings for better matching
        ("employ", "boss", "manage"): "LEADS",  # Not WORKS_FOR
        ("live", "reside", "stay"): "LOCATED_IN",
        ("travel", "journey", "move"): "TRAVELS_TO",
        ("own", "possess", "hold"): "OWNS",
    }

    # Find best match using whole word matching to avoid false positives
    import re

    for keywords, canonical_rel in keyword_mappings.items():
        for keyword in keywords:
            # Use word boundary matching to avoid partial matches
            # e.g., "trust" shouldn't match "destroys"
            if re.search(r"\b" + re.escape(keyword) + r"\b", rel_lower):
                return canonical_rel

    return None


async def normalize_existing_relationship_types() -> None:
    """Normalize all stored relationship types to canonical form."""
    query = "MATCH ()-[r:DYNAMIC_REL]->() RETURN DISTINCT r.type AS t"
    try:
        results = await neo4j_manager.execute_read_query(query)
    except Exception as exc:  # pragma: no cover - narrow DB errors
        logger.error(f"Error reading existing relationship types: {exc}")
        return

    statements: list[tuple[str, dict[str, Any]]] = []
    for record in results:
        current = record.get("t")
        if not current:
            continue
        normalized = normalize_relationship_type(str(current))
        if normalized != current:
            statements.append(
                (
                    "MATCH ()-[r:DYNAMIC_REL {type: $old}]->() SET r.type = $new",
                    {"old": current, "new": normalized},
                )
            )
    if statements:
        try:
            await neo4j_manager.execute_cypher_batch(statements)
            logger.info("Normalized %d relationship type variations", len(statements))
        except Exception as exc:  # pragma: no cover - log but continue
            logger.error(
                "Failed to update some relationship types: %s", exc, exc_info=True
            )


def _get_cypher_labels(entity_type: str | None) -> str:
    """Helper to create a Cypher label string (e.g., :Character:Entity or :Person:Character:Entity)."""

    entity_label_suffix = ":Entity"  # All nodes get this
    specific_labels_parts: list[str] = []

    if entity_type and entity_type.strip():
        cleaned = re.sub(r"[^a-zA-Z0-9_\s]+", "", entity_type)
        normalized_key = re.sub(r"[_\s]+", "", cleaned).lower()

        canonical = _CANONICAL_NODE_LABEL_MAP.get(normalized_key)
        if canonical is None:
            pascal = _to_pascal_case(cleaned)
            canonical = _CANONICAL_NODE_LABEL_MAP.get(pascal.lower(), pascal)

        if canonical and canonical != "Entity":
            if canonical != "Character":
                specific_labels_parts.append(f":{canonical}")

            if cleaned.strip().lower() == "person" or canonical == "Character":
                if ":Character" not in specific_labels_parts:
                    specific_labels_parts.append(":Character")

    # Order: :Character (if present), then other specific labels (e.g., :Person), then :Entity
    # Remove duplicates and establish order
    final_ordered_labels = []
    if ":Character" in specific_labels_parts:
        final_ordered_labels.append(":Character")

    for label in specific_labels_parts:
        if label not in final_ordered_labels:
            final_ordered_labels.append(label)

    if not final_ordered_labels:
        return entity_label_suffix  # Just ":Entity"

    return "".join(final_ordered_labels) + entity_label_suffix


def _get_constraint_safe_merge(
    labels_cypher: str,
    name_param: str,
    create_ts_var: str = "s",
    id_param: str | None = None,
) -> tuple[str, list[str]]:
    """Generate constraint-safe MERGE queries that handle multiple labels correctly.

    Returns:
        tuple: (merge_query, additional_set_labels)
    """
    # Parse the labels to identify constraint-sensitive ones
    labels = [label.strip() for label in labels_cypher.split(":") if label.strip()]
    constraint_labels = [
        "Character",
        "NovelInfo",
        "Chapter",
        "WorldElement",
    ]  # Labels with uniqueness constraints

    # Find the first constraint-sensitive label to use in MERGE
    primary_label = None
    additional_labels: list[str] = []

    # If we have a stable id to merge on, prefer the Entity label
    # to align with the unique constraint on Entity.id
    if id_param:
        primary_label = "Entity"
        additional_labels = [l for l in labels if l != "Entity"]
        merge_query = f"MERGE ({create_ts_var}:{primary_label} {{id: ${id_param}}})"
        return merge_query, additional_labels

    for label in labels:
        if label in constraint_labels:
            if primary_label is None:
                primary_label = label
            else:
                additional_labels.append(label)
        else:
            additional_labels.append(label)

    # If no constraint-sensitive label found, use Entity as default
    if primary_label is None:
        primary_label = "Entity"
        # Remove Entity from additional labels if it's there
        additional_labels = [l for l in additional_labels if l != "Entity"]

    # Build the MERGE query with only the primary constraint label
    merge_query = f"MERGE ({create_ts_var}:{primary_label} {{name: ${name_param}}})"

    return merge_query, additional_labels


async def add_kg_triples_batch_to_db(
    structured_triples_data: list[dict[str, Any]],
    chapter_number: int,
    is_from_flawed_draft: bool,
) -> None:
    if not structured_triples_data:
        logger.info("Neo4j: add_kg_triples_batch_to_db: No structured triples to add.")
        return

    statements_with_params: list[tuple[str, dict[str, Any]]] = []

    for i, triple_dict in enumerate(structured_triples_data):
        subject_info = triple_dict.get("subject")
        predicate_str = triple_dict.get("predicate")

        object_entity_info = triple_dict.get("object_entity")
        object_literal_val = triple_dict.get(
            "object_literal"
        )  # This will be a string from parsing
        is_literal_object = triple_dict.get("is_literal_object", False)

        if not (
            subject_info
            and isinstance(subject_info, dict)
            and subject_info.get("name")
            and predicate_str
        ):
            logger.warning(
                f"Neo4j (Batch): Invalid subject or predicate in triple dict: {triple_dict}"
            )
            continue

        subject_name = str(subject_info["name"]).strip()
        subject_type = subject_info.get(
            "type"
        )  # This is a string like "Character", "WorldElement", etc.
        predicate_clean = str(predicate_str).strip().upper().replace(" ", "_")

        # Enhanced Type Inference: Upgrade generic types to specific ones
        if subject_type in ["WorldElement", "Entity"] and subject_name:
            subject_category = subject_info.get("category", "")
            upgraded_type = _infer_specific_node_type(
                subject_name, subject_category, subject_type
            )
            if upgraded_type != subject_type:
                upgrade_key = f"{subject_name}:{subject_type}->{upgraded_type}"
                if upgrade_key not in _upgrade_logged:
                    logger.info(
                        f"Type inference upgraded {subject_type} -> {upgraded_type} for '{subject_name}'"
                    )
                    _upgrade_logged.add(upgrade_key)
                subject_type = upgraded_type

        if not all([subject_name, predicate_clean]):
            logger.warning(
                f"Neo4j (Batch): Empty subject name or predicate after stripping: {triple_dict}"
            )
            continue

        subject_labels_cypher = _get_cypher_labels(subject_type)

        # Base parameters for the relationship
        rel_props = {
            "type": predicate_clean,
            KG_REL_CHAPTER_ADDED: chapter_number,
            KG_IS_PROVISIONAL: is_from_flawed_draft,
            "confidence": 1.0,  # Default confidence
            # Add other relationship metadata if available
        }

        params = {"subject_name_param": subject_name, "rel_props_param": rel_props}

        if is_literal_object:
            if object_literal_val is None:
                logger.warning(
                    f"Neo4j (Batch): Literal object is None for triple: {triple_dict}"
                )
                continue

            # For literal objects, merge/create a ValueNode.
            # The ValueNode is unique by its string value and type 'Literal'.
            params["object_literal_value_param"] = str(
                object_literal_val
            )  # Ensure it's a string for ValueNode value
            params["value_node_type_param"] = (
                "Literal"  # Generic type for these literal ValueNodes
            )

            # Generate stable id for Characters if possible
            subject_id_param_key = None
            if subject_type == "Character":
                try:
                    from processing.entity_deduplication import generate_entity_id

                    params["subject_id_param"] = generate_entity_id(
                        subject_name, "character", int(chapter_number)
                    )
                    subject_id_param_key = "subject_id_param"
                except Exception:
                    subject_id_param_key = None

            # Generate constraint-safe MERGE for subject (prefer id when available)
            subject_merge, subject_additional_labels = _get_constraint_safe_merge(
                subject_labels_cypher, "subject_name_param", "s", subject_id_param_key
            )

            # Combine ON CREATE SET clauses for subject
            subject_create_sets = f"s.created_ts = timestamp(), s.updated_ts = timestamp(), s.type = '{subject_type}', s.name = $subject_name_param"
            if subject_additional_labels:
                label_clauses = [f"s:`{label}`" for label in subject_additional_labels]
                subject_create_sets += ", " + ", ".join(label_clauses)
            # If we generated a stable id for Character, set it on create
            if subject_id_param_key:
                subject_create_sets += ", s.id = coalesce(s.id, $subject_id_param)"

            # Create a stable relationship id to avoid flattening history across chapters
            rel_id_source = (
                f"{predicate_clean}|{subject_name.strip().lower()}|"
                f"{str(object_literal_val).strip()}|{chapter_number}"
            )
            rel_id = hashlib.sha1(rel_id_source.encode("utf-8")).hexdigest()[:16]
            params["rel_id_param"] = rel_id

            # Ensure any additional labels are applied even if node existed
            subject_label_set_clause = (
                "SET "
                + ", ".join([f"s:`{label}`" for label in subject_additional_labels])
                if subject_additional_labels
                else ""
            )

            query = f"""
            {subject_merge}
                ON CREATE SET {subject_create_sets}
            {subject_label_set_clause}
            MERGE (o:Entity:ValueNode {{value: $object_literal_value_param, type: $value_node_type_param}})
                ON CREATE SET o.created_ts = timestamp(), o.updated_ts = timestamp()

            MERGE (s)-[r:`{predicate_clean}` {{id: $rel_id_param}}]->(o)
                ON CREATE SET r = $rel_props_param, r.created_ts = timestamp()
                ON MATCH SET r += $rel_props_param, r.updated_ts = timestamp()
            """
            statements_with_params.append((query, params))

        elif (
            object_entity_info
            and isinstance(object_entity_info, dict)
            and object_entity_info.get("name")
        ):
            object_name = str(object_entity_info["name"]).strip()
            object_type = object_entity_info.get(
                "type"
            )  # String like "Location", "Item"
            if not object_name:
                logger.warning(
                    f"Neo4j (Batch): Empty object name for entity object in triple: {triple_dict}"
                )
                continue

            # Enhanced Type Inference for object as well
            if object_type in ["WorldElement", "Entity"] and object_name:
                object_category = object_entity_info.get("category", "")
                upgraded_object_type = _infer_specific_node_type(
                    object_name, object_category, object_type
                )
                if upgraded_object_type != object_type:
                    upgrade_key = (
                        f"object:{object_name}:{object_type}->{upgraded_object_type}"
                    )
                    if upgrade_key not in _upgrade_logged:
                        logger.info(
                            f"Type inference upgraded object {object_type} -> {upgraded_object_type} for '{object_name}'"
                        )
                        _upgrade_logged.add(upgrade_key)
                    object_type = upgraded_object_type

            object_labels_cypher = _get_cypher_labels(object_type)
            params["object_name_param"] = object_name

            # Prefer stable ids for Characters when available
            subject_id_param_key = None
            if subject_type == "Character":
                try:
                    from processing.entity_deduplication import generate_entity_id

                    params["subject_id_param"] = generate_entity_id(
                        subject_name, "character", int(chapter_number)
                    )
                    subject_id_param_key = "subject_id_param"
                except Exception:
                    subject_id_param_key = None

            object_id_param_key = None
            if object_type == "Character":
                try:
                    from processing.entity_deduplication import generate_entity_id

                    params["object_id_param"] = generate_entity_id(
                        object_name, "character", int(chapter_number)
                    )
                    object_id_param_key = "object_id_param"
                except Exception:
                    object_id_param_key = None

            # Generate constraint-safe MERGE for both subject and object
            subject_merge, subject_additional_labels = _get_constraint_safe_merge(
                subject_labels_cypher, "subject_name_param", "s", subject_id_param_key
            )
            object_merge, object_additional_labels = _get_constraint_safe_merge(
                object_labels_cypher, "object_name_param", "o", object_id_param_key
            )

            # Combine ON CREATE SET clauses for both nodes
            subject_create_sets = f"s.created_ts = timestamp(), s.updated_ts = timestamp(), s.type = '{subject_type}', s.name = $subject_name_param"
            if subject_additional_labels:
                label_clauses = [f"s:`{label}`" for label in subject_additional_labels]
                subject_create_sets += ", " + ", ".join(label_clauses)

            # If we generated a stable id for Character subject, set it
            if subject_id_param_key:
                subject_create_sets += ", s.id = coalesce(s.id, $subject_id_param)"

            object_create_sets = f"o.created_ts = timestamp(), o.updated_ts = timestamp(), o.type = '{object_type}', o.name = $object_name_param"
            if object_additional_labels:
                label_clauses = [f"o:`{label}`" for label in object_additional_labels]
                object_create_sets += ", " + ", ".join(label_clauses)

            # If we generated a stable id for Character object, set it
            if object_id_param_key:
                object_create_sets += ", o.id = coalesce(o.id, $object_id_param)"

            # Create a stable relationship id to avoid flattening history across chapters
            rel_id_source = (
                f"{predicate_clean}|{subject_name.strip().lower()}|"
                f"{object_name.strip().lower()}|{chapter_number}"
            )
            rel_id = hashlib.sha1(rel_id_source.encode("utf-8")).hexdigest()[:16]
            params["rel_id_param"] = rel_id

            # Ensure any additional labels are applied even if nodes existed
            subject_label_set_clause = (
                "SET "
                + ", ".join([f"s:`{label}`" for label in subject_additional_labels])
                if subject_additional_labels
                else ""
            )
            object_label_set_clause = (
                "SET "
                + ", ".join([f"o:`{label}`" for label in object_additional_labels])
                if object_additional_labels
                else ""
            )

            query = f"""
            {subject_merge}
                ON CREATE SET {subject_create_sets}
            {subject_label_set_clause}
            {object_merge}
                ON CREATE SET {object_create_sets}
            {object_label_set_clause}

            MERGE (s)-[r:`{predicate_clean}` {{id: $rel_id_param}}]->(o)
                ON CREATE SET r = $rel_props_param, r.created_ts = timestamp()
                ON MATCH SET r += $rel_props_param, r.updated_ts = timestamp()
            """
            statements_with_params.append((query, params))
        else:
            logger.warning(
                f"Neo4j (Batch): Invalid or missing object information in triple dict: {triple_dict}"
            )
            continue

    if not statements_with_params:
        logger.info(
            "Neo4j: add_kg_triples_batch_to_db: No valid statements generated from triples."
        )
        return

    try:
        await neo4j_manager.execute_cypher_batch(statements_with_params)

        logger.info(
            f"Neo4j: Batch processed {len(statements_with_params)} KG triple statements."
        )
    except Exception as e:
        # Log first few problematic params for debugging, if any
        first_few_params_str = (
            str([p_tuple[1] for p_tuple in statements_with_params[:2]])
            if statements_with_params
            else "N/A"
        )
        logger.error(
            f"Neo4j: Error in batch adding KG triples. First few params: {first_few_params_str}. Error: {e}",
            exc_info=True,
        )
        raise


@alru_cache(maxsize=256, ttl=300)  # Cache for 5 minutes with 256 entries max
async def query_kg_from_db(
    subject: str | None = None,
    predicate: str | None = None,
    obj_val: str | None = None,
    chapter_limit: int | None = None,
    include_provisional: bool = True,
    limit_results: int | None = None,
) -> list[dict[str, Any]]:
    conditions = []
    parameters: dict[str, Any] = {}
    match_clause = "MATCH (s:Entity)-[r]->(o) "

    if subject is not None:
        conditions.append("s.name = $subject_param")
        parameters["subject_param"] = subject.strip()
    if predicate is not None:
        normalized_predicate = validate_relationship_type(predicate)
        match_clause = f"MATCH (s:Entity)-[r:`{normalized_predicate}`]->(o) "
    if obj_val is not None:
        obj_val_stripped = obj_val.strip()
        conditions.append(
            """
            ( (o:ValueNode AND o.value = $object_param ) OR
              (NOT o:ValueNode AND o.name = $object_param)
            )
        """
        )
        parameters["object_param"] = obj_val_stripped
    if chapter_limit is not None:
        conditions.append(
            f"coalesce(r.{KG_REL_CHAPTER_ADDED}, -1) <= $chapter_limit_param"
        )
        parameters["chapter_limit_param"] = chapter_limit
    if not include_provisional:
        conditions.append(f"coalesce(r.{KG_IS_PROVISIONAL}, FALSE) = FALSE")

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

    return_clause = f"""
    RETURN s.name AS subject,
           type(r) AS predicate,
           CASE WHEN o:ValueNode THEN o.value ELSE o.name END AS object,
           CASE WHEN o:ValueNode THEN 'Literal' ELSE labels(o)[0] END AS object_type,
           coalesce(r.{KG_REL_CHAPTER_ADDED}, -1) AS {KG_REL_CHAPTER_ADDED},
           coalesce(r.confidence, 0.0) AS confidence,
           coalesce(r.{KG_IS_PROVISIONAL}, FALSE) AS {KG_IS_PROVISIONAL}
    """
    order_clause = f" ORDER BY coalesce(r.{KG_REL_CHAPTER_ADDED}, -1) DESC, coalesce(r.confidence, 0.0) DESC"
    limit_clause_str = (
        f" LIMIT {int(limit_results)}"
        if limit_results is not None and limit_results > 0
        else ""
    )

    full_query = (
        match_clause + where_clause + return_clause + order_clause + limit_clause_str
    )
    try:
        results = await neo4j_manager.execute_read_query(full_query, parameters)
        triples_list: list[dict[str, Any]] = (
            [dict(record) for record in results] if results else []
        )
        logger.debug(
            f"Neo4j: KG query returned {len(triples_list)} results. Query: '{full_query[:200]}...' Params: {parameters}"
        )
        return triples_list
    except Exception as e:
        logger.error(
            f"Neo4j: Error querying KG. Query: '{full_query[:200]}...', Params: {parameters}, Error: {e}",
            exc_info=True,
        )
        return []


async def get_most_recent_value_from_db(
    subject: str,
    predicate: str,
    chapter_limit: int | None = None,
    include_provisional: bool = False,
) -> Any | None:
    if not subject.strip() or not predicate.strip():
        logger.warning(
            f"Neo4j: get_most_recent_value_from_db: empty subject or predicate. S='{subject}', P='{predicate}'"
        )
        return None

    # Direct query for the most recent value to avoid dependency on query_kg_from_db
    if not subject.strip() or not predicate.strip():
        logger.warning(
            f"Neo4j: get_most_recent_value_from_db: empty subject or predicate. S='{subject}', P='{predicate}'"
        )
        return None

    conditions = []
    parameters: dict[str, Any] = {}
    normalized_predicate = validate_relationship_type(predicate)
    match_clause = f"MATCH (s:Entity)-[r:`{normalized_predicate}`]->(o) "

    conditions.append("s.name = $subject_param")
    parameters["subject_param"] = subject.strip()

    if chapter_limit is not None:
        conditions.append(
            f"coalesce(r.{KG_REL_CHAPTER_ADDED}, -1) <= $chapter_limit_param"
        )
        parameters["chapter_limit_param"] = chapter_limit
    if not include_provisional:
        conditions.append(
            f"(r.{KG_IS_PROVISIONAL} = FALSE OR r.{KG_IS_PROVISIONAL} IS NULL)"
        )

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

    return_clause = f"""
    RETURN
           CASE WHEN o:ValueNode THEN o.value ELSE o.name END AS object,
           coalesce(r.{KG_REL_CHAPTER_ADDED}, -1) AS {KG_REL_CHAPTER_ADDED},
           coalesce(r.confidence, 0.0) AS confidence,
           coalesce(r.{KG_IS_PROVISIONAL}, FALSE) AS {KG_IS_PROVISIONAL}
    """
    order_clause = f" ORDER BY coalesce(r.{KG_REL_CHAPTER_ADDED}, -1) DESC, coalesce(r.confidence, 0.0) DESC"
    limit_clause_str = " LIMIT 1"

    full_query = (
        match_clause + where_clause + return_clause + order_clause + limit_clause_str
    )
    try:
        results = await neo4j_manager.execute_read_query(full_query, parameters)
        if results and results[0] and "object" in results[0]:
            value = results[0]["object"]
            # Attempt to convert to number if it looks like one, as ValueNode.value stores as string from current triple parsing
            if isinstance(value, str):
                if re.match(r"^-?\d+$", value):
                    value = int(value)
                elif re.match(r"^-?\d*\.\d+$", value):
                    value = float(value)
                elif value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False

            logger.debug(
                f"Neo4j: Found most recent value for ('{subject}', '{predicate}'): '{value}' (type: {type(value)}) from Ch {results[0].get(KG_REL_CHAPTER_ADDED, 'N/A')}, Prov: {results[0].get(KG_IS_PROVISIONAL)}"
            )
            return value
    except Exception as e:
        logger.error(
            f"Neo4j: Error querying KG. Query: '{full_query[:200]}...', Params: {parameters}, Error: {e}",
            exc_info=True,
        )
    logger.debug(
        f"Neo4j: No value found for ({subject}, {predicate}) up to Ch {chapter_limit}, include_provisional={include_provisional}."
    )
    return None


@alru_cache(maxsize=64, ttl=600)  # Cache novel info properties for 10 minutes
async def get_novel_info_property_from_db(property_key: str) -> Any | None:
    """Return a property value from the NovelInfo node."""
    if not property_key.strip():
        logger.warning("Neo4j: empty property key for NovelInfo query")
        return None

    novel_id_param = config.MAIN_NOVEL_INFO_NODE_ID
    query = f"MATCH (ni:NovelInfo:Entity {{id: $novel_id_param}}) RETURN ni.{property_key} AS value"
    try:
        results = await neo4j_manager.execute_read_query(
            query, {"novel_id_param": novel_id_param}
        )
        if results and results[0] and "value" in results[0]:
            return results[0]["value"]
    except Exception as e:  # pragma: no cover - narrow DB errors
        logger.error(
            f"Neo4j: Error retrieving NovelInfo property '{property_key}': {e}",
            exc_info=True,
        )
    return None


async def get_chapter_context_for_entity(
    entity_name: str | None = None, entity_id: str | None = None
) -> list[dict[str, Any]]:
    """
    Finds chapters where an entity was mentioned or involved to provide context for enrichment.
    Searches by name for Characters/ValueNodes or by ID for WorldElements.
    """
    if not entity_name and not entity_id:
        return []

    match_clause = (
        "MATCH (e {id: $id_param})" if entity_id else "MATCH (e {name: $name_param})"
    )
    params = {"id_param": entity_id} if entity_id else {"name_param": entity_name}

    query = f"""
    {match_clause}

    // Get all paths to potential chapter number sources
    OPTIONAL MATCH (e)-[]->(event) WHERE (event:DevelopmentEvent OR event:WorldElaborationEvent) AND event.chapter_updated IS NOT NULL
    OPTIONAL MATCH (e)-[r]->() WHERE r.chapter_added IS NOT NULL

    // Collect all numbers into one list, then process
    WITH
      CASE WHEN e.created_chapter IS NOT NULL THEN [e.created_chapter] ELSE [] END as created_chapter_list,
      COLLECT(DISTINCT event.chapter_updated) as event_chapters,
      COLLECT(DISTINCT r.chapter_added) as rel_chapters

    // Combine, filter out nulls, unwind, get distinct
    WITH created_chapter_list + event_chapters + rel_chapters as all_chapters
    UNWIND all_chapters as chapter_num
    WITH DISTINCT chapter_num
    WHERE chapter_num IS NOT NULL AND chapter_num > 0

    // Now fetch the chapter data
    MATCH (c:{"Chapter"} {{number: chapter_num}})
    RETURN c.number as chapter_number, c.summary as summary, c.text as text
    ORDER BY c.number DESC
    LIMIT 5 // Limit context to most recent 5 chapters
    """
    try:
        results = await neo4j_manager.execute_read_query(query, params)
        return results if results else []
    except Exception as e:
        logger.error(
            f"Error getting chapter context for entity '{entity_name or entity_id}': {e}",
            exc_info=True,
        )
        return []


async def find_contradictory_trait_characters(
    contradictory_trait_pairs: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """
    Finds characters who have contradictory traits based on a provided list of pairs.
    e.g. [('Brave', 'Cowardly'), ('Honest', 'Deceitful')]
    """
    if not contradictory_trait_pairs:
        return []

    all_findings = []
    for trait1, trait2 in contradictory_trait_pairs:
        query = """
        MATCH (c:Character)-[:HAS_TRAIT_ASPECT]->(t1:Trait {name: $trait1_param}),
              (c)-[:HAS_TRAIT_ASPECT]->(t2:Trait {name: $trait2_param})
        RETURN c.name AS character_name, t1.name AS trait1, t2.name AS trait2
        """
        params = {"trait1_param": trait1, "trait2_param": trait2}
        try:
            results = await neo4j_manager.execute_read_query(query, params)
            if results:
                all_findings.extend(results)
        except Exception as e:
            logger.error(
                f"Error checking for contradictory traits '{trait1}' vs '{trait2}': {e}",
                exc_info=True,
            )

    return all_findings


async def find_post_mortem_activity() -> list[dict[str, Any]]:
    """
    Finds characters who have relationships or activities recorded in chapters
    after they were marked as dead.
    """
    query = """
    MATCH (c:Character)-[death_rel:`IS_DEAD`]->()
    WHERE death_rel.is_provisional = false OR death_rel.is_provisional IS NULL
    WITH c, death_rel.chapter_added AS death_chapter

    MATCH (c)-[activity_rel]->()
    WHERE activity_rel.chapter_added > death_chapter
      AND NOT type(activity_rel) IN ['IS_REMEMBERED_AS', 'WAS_FRIEND_OF'] // Exclude retrospective rels
    RETURN DISTINCT c.name as character_name,
           death_chapter,
           collect(
             {
               activity_type: type(activity_rel),
               activity_chapter: activity_rel.chapter_added
             }
           ) AS post_mortem_activities
    LIMIT 20
    """
    try:
        results = await neo4j_manager.execute_read_query(query)
        return results if results else []
    except Exception as e:
        logger.error(f"Error checking for post-mortem activity: {e}", exc_info=True)
        return []


async def find_candidate_duplicate_entities(
    similarity_threshold: float = 0.45,
    limit: int = 50,
    *,
    desc_threshold: float = 0.30,
    per_label_limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Find candidate duplicate entity pairs with tighter prefilters.

    Improvements vs. previous approach:
    - Restricts comparisons to like-with-like labels (Character↔Character, Typed Entities↔Typed Entities).
    - Normalizes names, gates by first-character and token overlap to prune pairs.
    - Optionally incorporates lightweight description token-overlap as a secondary check.
    - Keeps API compatible with existing caller ("similarity_threshold" and "limit").
    """
    # Use a per-label limit when provided; default to overall limit otherwise
    label_limit = per_label_limit or limit

    # The query runs two label-specific searches and unions results
    # Description similarity is a simple Jaccard-overlap on tokenized, cleaned text
    query = """
    // ---------- Character pairs ----------
    MATCH (e1:Character:Entity)
    WITH e1 WHERE e1.name IS NOT NULL AND e1.id IS NOT NULL
    MATCH (e2:Character:Entity)
    WHERE elementId(e1) < elementId(e2)
      AND e2.name IS NOT NULL AND e2.id IS NOT NULL
    WITH e1, e2,
         toLower(toString(e1.name)) AS n1,
         toLower(toString(e2.name)) AS n2,
         coalesce(toLower(toString(e1.description)), '') AS d1,
         coalesce(toLower(toString(e2.description)), '') AS d2,
         e1.category AS cat1,
         e2.category AS cat2
    WITH e1, e2, n1, n2, d1, d2,
         apoc.text.levenshteinSimilarity(n1, n2) AS name_sim,
         abs(size(n1) - size(n2)) AS len_diff,
         substring(n1, 0, 1) AS c1,
         substring(n2, 0, 1) AS c2,
         [w IN split(apoc.text.replace(n1, '[^a-z0-9 ]', ''), ' ') WHERE size(w) > 2] AS n1_tokens,
         [w IN split(apoc.text.replace(n2, '[^a-z0-9 ]', ''), ' ') WHERE size(w) > 2] AS n2_tokens,
         [w IN split(apoc.text.replace(d1, '[^a-z0-9 ]', ''), ' ') WHERE size(w) > 2] AS d1_tokens,
         [w IN split(apoc.text.replace(d2, '[^a-z0-9 ]', ''), ' ') WHERE size(w) > 2] AS d2_tokens,
         cat1, cat2
    WITH e1, e2, name_sim, len_diff,
         size([x IN n1_tokens WHERE x IN n2_tokens]) AS name_overlap,
         size([x IN d1_tokens WHERE x IN d2_tokens]) AS desc_overlap,
         CASE WHEN size(d1_tokens) >= size(d2_tokens) THEN size(d1_tokens) ELSE size(d2_tokens) END AS d_denom,
         c1, c2,
         cat1, cat2
    WITH e1, e2, name_sim, len_diff, name_overlap,
         CASE WHEN d_denom = 0 THEN 0.0 ELSE toFloat(desc_overlap) / toFloat(d_denom) END AS desc_sim,
         c1, c2,
         cat1, cat2
    WHERE len_diff <= 10
      AND c1 = c2
      AND name_overlap >= 1
      AND (cat1 IS NULL OR cat2 IS NULL OR cat1 = cat2)
      AND (
            name_sim >= $name_threshold
         OR (name_sim >= ($name_threshold * 0.8) AND desc_sim >= $desc_threshold)
      )
    RETURN e1.id AS id1, e1.name AS name1, labels(e1) AS labels1,
           e2.id AS id2, e2.name AS name2, labels(e2) AS labels2,
           name_sim AS similarity
    ORDER BY similarity DESC
    LIMIT $label_limit

    UNION

    // ---------- Typed Entity pairs (non-Character) ----------
    MATCH (e1:Entity)
    WHERE (e1:Object OR e1:Artifact OR e1:Location OR e1:Document OR e1:Item OR e1:Relic)
    WITH e1 WHERE e1.name IS NOT NULL AND e1.id IS NOT NULL
    MATCH (e2:Entity)
    WHERE (e2:Object OR e2:Artifact OR e2:Location OR e2:Document OR e2:Item OR e2:Relic)
      AND elementId(e1) < elementId(e2)
      AND e2.name IS NOT NULL AND e2.id IS NOT NULL
    WITH e1, e2,
         toLower(toString(e1.name)) AS n1,
         toLower(toString(e2.name)) AS n2,
         coalesce(toLower(toString(e1.description)), '') AS d1,
         coalesce(toLower(toString(e2.description)), '') AS d2,
         e1.category AS cat1,
         e2.category AS cat2
    WITH e1, e2, n1, n2, d1, d2,
         apoc.text.levenshteinSimilarity(n1, n2) AS name_sim,
         abs(size(n1) - size(n2)) AS len_diff,
         substring(n1, 0, 1) AS c1,
         substring(n2, 0, 1) AS c2,
         [w IN split(apoc.text.replace(n1, '[^a-z0-9 ]', ''), ' ') WHERE size(w) > 2] AS n1_tokens,
         [w IN split(apoc.text.replace(n2, '[^a-z0-9 ]', ''), ' ') WHERE size(w) > 2] AS n2_tokens,
         [w IN split(apoc.text.replace(d1, '[^a-z0-9 ]', ''), ' ') WHERE size(w) > 2] AS d1_tokens,
         [w IN split(apoc.text.replace(d2, '[^a-z0-9 ]', ''), ' ') WHERE size(w) > 2] AS d2_tokens,
         cat1, cat2
    WITH e1, e2, name_sim, len_diff,
         size([x IN n1_tokens WHERE x IN n2_tokens]) AS name_overlap,
         size([x IN d1_tokens WHERE x IN d2_tokens]) AS desc_overlap,
         CASE WHEN size(d1_tokens) >= size(d2_tokens) THEN size(d1_tokens) ELSE size(d2_tokens) END AS d_denom,
         c1, c2,
         cat1, cat2
    WITH e1, e2, name_sim, len_diff, name_overlap,
         CASE WHEN d_denom = 0 THEN 0.0 ELSE toFloat(desc_overlap) / toFloat(d_denom) END AS desc_sim,
         c1, c2,
         cat1, cat2
    WHERE len_diff <= 10
      AND c1 = c2
      AND name_overlap >= 1
      AND (cat1 IS NULL OR cat2 IS NULL OR cat1 = cat2)
      AND (
            name_sim >= $name_threshold
         OR (name_sim >= ($name_threshold * 0.8) AND desc_sim >= $desc_threshold)
      )
    RETURN e1.id AS id1, e1.name AS name1, labels(e1) AS labels1,
           e2.id AS id2, e2.name AS name2, labels(e2) AS labels2,
           name_sim AS similarity
    ORDER BY similarity DESC
    LIMIT $label_limit
    """

    params = {
        "name_threshold": similarity_threshold,
        "desc_threshold": desc_threshold,
        "label_limit": label_limit,
    }
    try:
        results = await neo4j_manager.execute_read_query(query, params)
        return results if results else []
    except Exception as e:
        logger.error(f"Error finding candidate duplicate entities: {e}", exc_info=True)
        return []


async def backfill_missing_entity_ids(max_updates: int = 1000) -> int:
    """Assign stable deterministic IDs to Character nodes missing `id`.

    Returns the number of nodes updated.
    """
    try:
        # Fetch candidate characters without IDs (limit to avoid huge batches)
        fetch_query = """
        MATCH (c:Character:Entity)
        WHERE c.id IS NULL OR c.id = ''
        RETURN c.name AS name, coalesce(c.created_chapter, 0) AS created_chapter
        LIMIT $limit
        """
        rows = await neo4j_manager.execute_read_query(
            fetch_query, {"limit": max_updates}
        )
        if not rows:
            return 0

        from processing.entity_deduplication import generate_entity_id

        items = []
        for r in rows:
            name = r.get("name")
            created_chapter = int(r.get("created_chapter", 0) or 0)
            if not name:
                continue
            items.append(
                {
                    "name": name,
                    "id": generate_entity_id(name, "character", created_chapter),
                }
            )

        if not items:
            return 0

        update_query = """
        UNWIND $items AS item
        MATCH (c:Character:Entity {name: item.name})
        WHERE c.id IS NULL OR c.id = ''
        SET c.id = item.id
        RETURN count(c) AS updated
        """
        result = await neo4j_manager.execute_write_query(update_query, {"items": items})
        return int((result or [{}])[0].get("updated", 0))
    except Exception as e:
        logger.error(f"Error backfilling missing entity IDs: {e}", exc_info=True)
        return 0


async def get_entity_context_for_resolution(
    entity_id: str,
) -> dict[str, Any] | None:
    """
    Gathers comprehensive context for an entity to help an LLM decide on a merge.
    """
    query = """
    MATCH (e)
    WHERE e.id = $entity_id OR e.name = $entity_id
    OPTIONAL MATCH (e)-[r]-(o)
    WHERE o.id IS NOT NULL
    WITH e,
         COUNT(r) as degree,
         COLLECT({
           rel_type: r.type,
           rel_props: properties(r),
           other_node_name: o.name,
           other_node_labels: labels(o)
         })[..10] AS relationships // Limit relationships for context brevity
    RETURN
      e.id AS id,
      e.name AS name,
      labels(e) AS labels,
      properties(e) AS properties,
      degree,
      relationships
    """
    params = {"entity_id": entity_id}
    try:
        results = await neo4j_manager.execute_read_query(query, params)
        if results:
            return results[0]
        else:
            # Debug: Check if entity exists with any labels
            debug_query = (
                "MATCH (e) WHERE e.id = $entity_id OR e.name = $entity_id "
                "RETURN e.name AS name, labels(e) AS labels"
            )
            debug_results = await neo4j_manager.execute_read_query(debug_query, params)
            if debug_results:
                logger.debug(
                    f"Entity {entity_id} exists with name '{debug_results[0]['name']}' "
                    f"and labels {debug_results[0]['labels']} but returned no context"
                )
            else:
                logger.debug(f"Entity {entity_id} does not exist in database")
            return None
    except Exception as e:
        logger.error(
            f"Error getting context for entity resolution (id: {entity_id}): {e}",
            exc_info=True,
        )
        return None


async def merge_entities(
    source_id: str, target_id: str, reason: str, max_retries: int = 3
) -> bool:
    """
    Merges one entity (source) into another (target) using atomic Neo4j operations with retry logic.
    The source node will be deleted after its relationships are moved.
    """
    import asyncio

    for attempt in range(max_retries):
        try:
            logger.info(
                f"Merge attempt {attempt + 1}/{max_retries} for {source_id} -> {target_id}"
            )
            return await _execute_atomic_merge(source_id, target_id, reason)
        except Exception as e:
            logger.error(
                f"Merge attempt {attempt + 1}/{max_retries} failed: {e}", exc_info=True
            )
            error_msg = str(e).lower()
            if (
                "entitynotfound" in error_msg
                or "transaction" in error_msg
                or "locked" in error_msg
                or "deadlock" in error_msg
            ) and attempt < max_retries - 1:
                logger.warning(
                    f"Entity merge attempt {attempt + 1}/{max_retries} failed, retrying: {e}"
                )
                await asyncio.sleep(0.1 * (2**attempt))  # Exponential backoff
                continue
            else:
                logger.error(
                    f"Entity merge failed after {attempt + 1} attempts: {e}",
                    exc_info=True,
                )
                return False

    return False


async def _execute_atomic_merge(source_id: str, target_id: str, reason: str) -> bool:
    """Execute entity merge using multiple queries in a single transaction to handle Neo4j constraints."""

    # Break the merge into separate queries that avoid complex Cypher constructs

    # Step 1: Copy properties
    copy_props_query = """
    MATCH (source {id: $source_id}), (target {id: $target_id})
    SET target.description = COALESCE(target.description + ', ' + source.description, source.description, target.description)
    RETURN count(*) as props_copied
    """

    # Step 2: Move outgoing relationships
    move_outgoing_query = """
    MATCH (source {id: $source_id}), (target {id: $target_id})
    MATCH (source)-[r]->(other)
    WHERE other.id <> target.id
    WITH source, target, r, other, properties(r) as rel_props
    CREATE (target)-[new_r:DYNAMIC_REL]->(other)
    SET new_r = rel_props,
        new_r.merged_from = $source_id,
        new_r.merge_reason = $reason,
        new_r.merge_timestamp = timestamp()
    DELETE r
    RETURN count(*) as outgoing_moved
    """

    # Step 3: Move incoming relationships
    move_incoming_query = """
    MATCH (source {id: $source_id}), (target {id: $target_id})
    MATCH (other)-[r]->(source)
    WHERE other.id <> target.id
    WITH source, target, r, other, properties(r) as rel_props
    CREATE (other)-[new_r:DYNAMIC_REL]->(target)
    SET new_r = rel_props,
        new_r.merged_from = $source_id,
        new_r.merge_reason = $reason,
        new_r.merge_timestamp = timestamp()
    DELETE r
    RETURN count(*) as incoming_moved
    """

    # Step 4: Delete source node
    delete_source_query = """
    MATCH (source {id: $source_id})
    DETACH DELETE source
    RETURN count(*) as deleted
    """

    params = {"target_id": target_id, "source_id": source_id, "reason": reason}

    try:
        logger.info(f"Attempting multi-step atomic merge: {source_id} -> {target_id}")

        # Execute all operations within the session's auto-commit transaction
        outgoing_count = 0
        incoming_count = 0

        # Copy properties
        await neo4j_manager.execute_write_query(copy_props_query, params)

        # Move outgoing relationships (may be zero)
        try:
            outgoing_result = await neo4j_manager.execute_write_query(
                move_outgoing_query, params
            )
            outgoing_count = (
                outgoing_result[0]["outgoing_moved"] if outgoing_result else 0
            )
        except Exception:
            # No outgoing relationships to move
            pass

        # Move incoming relationships (may be zero)
        try:
            incoming_result = await neo4j_manager.execute_write_query(
                move_incoming_query, params
            )
            incoming_count = (
                incoming_result[0]["incoming_moved"] if incoming_result else 0
            )
        except Exception:
            # No incoming relationships to move
            pass

        # Delete source node
        await neo4j_manager.execute_write_query(
            delete_source_query, {"source_id": source_id}
        )

        total_moved = outgoing_count + incoming_count
        logger.info(
            f"Successfully merged {source_id} -> {target_id} ({total_moved} relationships moved, reason: {reason})"
        )
        return True

    except Exception as e:
        logger.error(
            f"Multi-step merge failed ({source_id} -> {target_id}): {e}",
            exc_info=True,
        )
        raise


@alru_cache(maxsize=1)
async def get_defined_node_labels() -> list[str]:
    """
    Returns canonical node labels from schema constants, enhanced with database labels.
    Prioritizes schema-defined types for better entity extraction specificity.
    """
    try:
        # Start with our canonical schema labels
        from models.kg_constants import NODE_LABELS

        # Also explicitly import enhanced node labels for clarity
        # Verify that NODE_LABELS contains all enhanced labels
        schema_labels = sorted(list(NODE_LABELS))

        # Also get any additional labels from database (for backward compatibility)
        results = await neo4j_manager.execute_read_query("CALL db.labels() YIELD label")
        db_labels = [
            r["label"]
            for r in results
            if r.get("label")
            and not r["label"].startswith("_")
            and r["label"] not in NODE_LABELS
        ]

        # Combine schema labels (priority) + database labels (backward compatibility)
        all_labels = schema_labels + sorted(db_labels)

        # Validate combined labels
        errors = validate_node_labels(all_labels)
        if errors:
            logger.warning(f"Invalid node labels found: {errors}")

        return all_labels
    except Exception:
        logger.error("Failed to load defined node labels.", exc_info=True)
        # Fallback to constants if DB query fails
        from models.kg_constants import NODE_LABELS

        labels = sorted(list(NODE_LABELS))
        # Validate labels against schema
        errors = validate_node_labels(labels)
        if errors:
            logger.warning(f"Fallback node labels validation errors: {errors}")
        return labels


@alru_cache(maxsize=1)
async def get_defined_relationship_types() -> list[str]:
    """Queries the database for all defined relationship types and caches the result."""
    try:
        results = await neo4j_manager.execute_read_query(
            "CALL db.relationshipTypes() YIELD relationshipType"
        )
        rel_types = [
            r["relationshipType"] for r in results if r.get("relationshipType")
        ]
        # Validate relationship types against schema
        errors = []
        for rel_type in rel_types:
            if rel_type not in VALID_RELATIONSHIP_TYPES:
                errors.append(f"Invalid relationship type: {rel_type}")
        if errors:
            logger.warning(f"Invalid relationship types found: {errors}")
        return rel_types
    except Exception:
        logger.error(
            "Failed to query defined relationship types from Neo4j.", exc_info=True
        )
        # Fallback to constants if DB query fails
        rel_types = list(config.RELATIONSHIP_TYPES)
        # Validate relationship types against schema
        errors = []
        for rel_type in rel_types:
            if rel_type not in VALID_RELATIONSHIP_TYPES:
                errors.append(f"Invalid relationship type: {rel_type}")
        if errors:
            logger.warning(f"Invalid relationship types in config: {errors}")
        return rel_types


async def promote_dynamic_relationships() -> int:
    """
    Enhanced relationship type promotion with validation.

    First validates and corrects relationship types, then promotes valid types
    to proper relationship types.
    """
    # Add early return if normalization is disabled
    if config.settings.DISABLE_RELATIONSHIP_NORMALIZATION:
        logger.info(
            "Relationship normalization disabled - skipping dynamic relationship resolution"
        )
        return

    total_promoted = 0

    # Step 1: Validate and correct existing relationship types
    corrected_count = await _validate_and_correct_relationship_types()
    logger.info(f"Validated and corrected {corrected_count} relationship types")

    # Step 2: Promote DYNAMIC_REL to typed relationships
    promotion_query = """
    MATCH (s)-[r:DYNAMIC_REL]->(o)
    WHERE r.type IS NOT NULL 
      AND r.type <> 'UNKNOWN' 
      AND r.type <> 'DYNAMIC_REL'
      AND r.type IN $valid_types
    WITH s, r, o, r.type as rel_type, properties(r) as rel_props
    
    // Create new typed relationship
    CALL apoc.create.relationship(
        s,
        rel_type,
        apoc.map.removeKey(rel_props, 'type'),
        o
    ) YIELD rel
    
    // Delete old dynamic relationship  
    DELETE r
    RETURN count(rel) AS promoted
    """

    try:
        # Use our validated relationship types
        valid_types = list(VALID_RELATIONSHIP_TYPES)
        results = await neo4j_manager.execute_write_query(
            promotion_query, {"valid_types": valid_types}
        )
        promoted_count = results[0].get("promoted", 0) if results else 0
        total_promoted = corrected_count + promoted_count

        logger.info(
            f"Successfully promoted {promoted_count} dynamic relationships to typed relationships"
        )
        logger.info(f"Total relationship processing: {total_promoted} relationships")

        return total_promoted

    except Exception as exc:
        logger.error(f"Failed to promote dynamic relationships: {exc}", exc_info=True)
        return total_promoted  # Return partial success


async def _validate_and_correct_relationship_types() -> int:
    """Validate and correct existing relationship types."""
    # Find all DYNAMIC_REL relationships with type properties
    validation_query = """
    MATCH (s)-[r:DYNAMIC_REL]->(o)
    WHERE r.type IS NOT NULL 
      AND r.type <> 'UNKNOWN'
      AND r.type <> 'DYNAMIC_REL'
    RETURN elementId(r) as rel_id, r.type as current_type
    """

    try:
        results = await neo4j_manager.execute_read_query(validation_query)
        if not results:
            return 0

        corrected_count = 0

        for record in results:
            current_type = record["current_type"]
            validated_type = validate_relationship_type(current_type)

            if validated_type != current_type:
                # Update to validated type
                update_query = """
                MATCH ()-[r:DYNAMIC_REL]->()
                WHERE elementId(r) = $rel_id
                SET r.type = $new_type
                RETURN count(*) as updated
                """

                await neo4j_manager.execute_write_query(
                    update_query,
                    {"rel_id": record["rel_id"], "new_type": validated_type},
                )
                corrected_count += 1
                logger.debug(
                    f"Corrected relationship type: '{current_type}' -> '{validated_type}'"
                )

        return corrected_count

    except Exception as exc:
        logger.error(f"Failed to validate relationship types: {exc}", exc_info=True)
        return 0


async def deduplicate_relationships() -> int:
    """Merge duplicate relationships of the same type between nodes."""
    # Merge the relationships by combining properties and deleting duplicates
    query = """
    MATCH (s)-[r]->(o)
    WITH s, type(r) AS t, o, collect(r) AS rels
    WHERE size(rels) > 1
    WITH s, t, o, rels
    // Keep the first relationship and delete the rest
    WITH s, t, o, head(rels) AS keepRel, tail(rels) AS deleteRels
    UNWIND deleteRels AS deleteRel
    // Combine properties from deleteRel into keepRel
    WITH s, t, o, keepRel, deleteRel, properties(deleteRel) AS deleteProps
    // For simplicity, we'll just delete the duplicate relationships
    // A more complex implementation would combine properties
    DELETE deleteRel
    RETURN count(*) AS removed
    """

    try:
        results = await neo4j_manager.execute_write_query(query)
        return results[0].get("removed", 0) if results else 0
    except Exception as exc:  # pragma: no cover - narrow DB errors
        logger.error(f"Failed to deduplicate relationships: {exc}", exc_info=True)
        return 0


async def consolidate_similar_relationships() -> int:
    """Consolidate semantically similar relationships using the predefined taxonomy."""
    # Get all relationship types currently in the database
    query_current = """
    MATCH ()-[r]->()
    RETURN DISTINCT type(r) AS rel_type, count(r) AS count
    ORDER BY count DESC
    """

    try:
        current_results = await neo4j_manager.execute_read_query(query_current)
        current_types = [r["rel_type"] for r in current_results if r.get("rel_type")]

        consolidation_count = 0

        # Process each current relationship type
        for current_type in current_types:
            # Skip if already canonical
            if current_type in VALID_RELATIONSHIP_TYPES:
                continue

            # Find canonical version
            canonical_type = normalize_relationship_type(current_type)

            # Skip if no change needed
            if current_type == canonical_type:
                continue

            # Consolidate relationships
            consolidate_query = f"""
            MATCH (s)-[r:{current_type}]->(o)
            CREATE (s)-[new_r:{canonical_type}]->(o)
            SET new_r = properties(r)
            DELETE r
            RETURN count(*) AS consolidated
            """

            try:
                consolidate_results = await neo4j_manager.execute_write_query(
                    consolidate_query
                )
                count = (
                    consolidate_results[0].get("consolidated", 0)
                    if consolidate_results
                    else 0
                )
                consolidation_count += count

                if count > 0:
                    logger.info(
                        f"Consolidated {count} relationships: {current_type} -> {canonical_type}"
                    )

            except Exception as exc:
                logger.warning(
                    f"Failed to consolidate {current_type} -> {canonical_type}: {exc}"
                )

        return consolidation_count

    except Exception as exc:
        logger.error(
            "Failed to consolidate similar relationships: %s", exc, exc_info=True
        )
        return 0


async def fetch_unresolved_dynamic_relationships(
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Fetch dynamic relationships lacking a specific type."""
    query = """
    MATCH (s:Entity)-[r:DYNAMIC_REL]->(o:Entity)
    WHERE r.type IS NULL OR r.type = 'UNKNOWN'
    RETURN elementId(r) AS rel_id,
           s.name AS subject,
           labels(s) AS subject_labels,
           coalesce(s.description, '') AS subject_desc,
           o.name AS object,
           labels(o) AS object_labels,
           coalesce(o.description, '') AS object_desc,
           coalesce(r.type, 'UNKNOWN') AS type
    LIMIT $limit
    """
    try:
        results = await neo4j_manager.execute_read_query(query, {"limit": limit})
        records = [dict(record) for record in results] if results else []
        # Validate node labels in the results
        for record in records:
            subject_labels = record.get("subject_labels", [])
            object_labels = record.get("object_labels", [])
            errors = validate_node_labels(subject_labels + object_labels)
            if errors:
                logger.warning(
                    "Invalid node labels in unresolved relationship: %s", errors
                )
        return records
    except Exception as exc:  # pragma: no cover - narrow DB errors
        logger.error(
            "Failed to fetch unresolved dynamic relationships: %s", exc, exc_info=True
        )
        return []


async def update_dynamic_relationship_type(rel_id: int, new_type: str) -> None:
    """Update a dynamic relationship's type."""
    # Validate and normalize the new relationship type
    validated_type = validate_relationship_type(new_type)
    if validated_type != new_type:
        logger.info(
            f"Normalized relationship type for update: '{new_type}' -> '{validated_type}'"
        )

    query = "MATCH ()-[r:DYNAMIC_REL]->() WHERE elementId(r) = $id SET r.type = $type"
    try:
        await neo4j_manager.execute_write_query(
            query, {"id": rel_id, "type": validated_type}
        )
    except Exception as exc:  # pragma: no cover - narrow DB errors
        logger.error(
            "Failed to update dynamic relationship %s: %s", rel_id, exc, exc_info=True
        )


async def get_shortest_path_length_between_entities(
    name1: str, name2: str, max_depth: int = 4
) -> int | None:
    """Return the shortest path length between two entities if it exists."""
    if max_depth <= 0:
        return None

    query = f"""
    MATCH (a:Entity {{name: $name1}}), (b:Entity {{name: $name2}})
    MATCH p = shortestPath((a)-[*..{max_depth}]-(b))
    RETURN length(p) AS len
    """
    try:
        results = await neo4j_manager.execute_read_query(
            query, {"name1": name1, "name2": name2}
        )
        if results:
            return results[0].get("len")
    except Exception as exc:  # pragma: no cover - narrow DB errors
        logger.error(f"Failed to compute shortest path length: {exc}", exc_info=True)
    return None


async def find_orphaned_bootstrap_elements() -> list[dict[str, Any]]:
    """Find bootstrap elements with no relationships."""
    query = """
    MATCH (we:Entity)
    WHERE (we:Object OR we:Artifact OR we:Location OR we:Document OR we:Item OR we:Relic)
      AND (toString(we.source) CONTAINS 'bootstrap' OR we.created_chapter = 0)
      AND NOT (we)-[]->() AND NOT ()-[]->(we)
    RETURN we.name as name, we.category as category, we.id as id
    LIMIT 10
    """
    try:
        results = await neo4j_manager.execute_read_query(query)
        return [dict(record) for record in results] if results else []
    except Exception as e:
        logger.error(f"Error finding orphaned bootstrap elements: {e}", exc_info=True)
        return []


async def find_potential_bridges(element: dict[str, Any]) -> list[dict[str, Any]]:
    """Find characters/locations that could bridge to this element."""
    query = """
    MATCH (bridge)
    WHERE (bridge:Character OR bridge:Object OR bridge:Artifact OR bridge:Location OR bridge:Document OR bridge:Item OR bridge:Relic)
      AND bridge.name <> $element_name
      AND ((bridge)-[]->() OR ()-[]->(bridge))  // Has some connections
    RETURN bridge.name as name, bridge.id as id,
           size((bridge)-[]->()) + size(()-[]->(bridge)) as connection_count
    ORDER BY connection_count DESC
    LIMIT 5
    """
    try:
        params = {"element_name": element.get("name", "")}
        results = await neo4j_manager.execute_read_query(query, params)
        return [dict(record) for record in results] if results else []
    except Exception as e:
        logger.error(f"Error finding potential bridges: {e}", exc_info=True)
        return []


async def create_relationship_with_properties(
    subject_name: str,
    relationship_type: str,
    object_name: str,
    properties: dict[str, Any] | None = None,
) -> None:
    """Create a relationship between two entities by name with optional properties."""
    if not properties:
        properties = {}

    # Default properties for bootstrap relationships
    default_props = {
        "source": "bootstrap",
        "confidence": 0.8,
        "chapter_added": 0,
    }
    default_props.update(properties)

    # Create structured triple data for the batch processor
    triple_data = {
        "subject": {"name": subject_name.strip()},
        "predicate": relationship_type.upper().strip(),
        "object_entity": {"name": object_name.strip()},
        "is_literal_object": False,
        "subject_type": "Entity",  # Will be refined by validation
        "object_type": "Entity",  # Will be refined by validation
        "properties": default_props,
    }

    # Use existing batch infrastructure to create the relationship
    # Ensure chapter_added is an integer
    chapter_num = 0
    try:
        chapter_num = int(properties.get("chapter_added", 0))
    except (ValueError, TypeError):
        pass

    await add_kg_triples_batch_to_db(
        [triple_data],
        chapter_number=chapter_num,
        is_from_flawed_draft=False,
    )

    logger.debug(
        f"Created relationship: {subject_name} {relationship_type} {object_name} "
        f"with properties: {default_props}"
    )


async def create_typed_relationship_with_properties(
    subject_name: str,
    subject_type: str,
    relationship_type: str,
    object_name: str,
    object_type: str,
    properties: dict[str, Any] | None = None,
) -> None:
    """Create a relationship with explicit subject/object types and optional properties.

    This variant allows callers (e.g., bootstrap) to pass known node types to reduce
    inference overhead during constraint validation while preserving the existing
    batch processing path and semantics.
    """
    if not properties:
        properties = {}

    default_props = {
        "source": "bootstrap",
        "confidence": 0.8,
        "chapter_added": 0,
    }
    default_props.update(properties)

    triple_data = {
        "subject": {"name": subject_name.strip(), "type": subject_type.strip()},
        "predicate": relationship_type.upper().strip(),
        "object_entity": {"name": object_name.strip(), "type": object_type.strip()},
        "is_literal_object": False,
        "subject_type": subject_type.strip(),
        "object_type": object_type.strip(),
        "properties": default_props,
    }

    # Ensure chapter_added is an integer
    chapter_num = 0
    try:
        chapter_num = int(default_props.get("chapter_added", 0))
    except (ValueError, TypeError):
        pass

    await add_kg_triples_batch_to_db(
        [triple_data],
        chapter_number=chapter_num,
        is_from_flawed_draft=False,
    )

    logger.debug(
        f"Created TYPED relationship: ({subject_type}) {subject_name} {relationship_type} ({object_type}) {object_name} "
        f"with properties: {default_props}"
    )


async def create_contextual_relationship(
    element1: dict[str, Any],
    element2: dict[str, Any],
    relationship_type: str = "CONTEXTUALLY_RELATED",
) -> None:
    """Create a contextual relationship between two elements with constraint validation."""
    # Get node types for validation
    element1_type = element1.get("type", "Entity")
    element2_type = element2.get("type", "Entity")

    # Validate the relationship using constraint system
    validated_relationship = await validate_single_relationship(
        element1_type, relationship_type, element2_type
    )

    if not validated_relationship:
        logger.warning(
            f"Skipping invalid contextual relationship: "
            f"{element1_type}:{element1.get('name')} | {relationship_type} | "
            f"{element2_type}:{element2.get('name')}"
        )
        return

    # Use the validated relationship type (may have been corrected)
    query = f"""
    MATCH (e1), (e2)
    WHERE e1.id = $element1_id AND e2.id = $element2_id
    MERGE (e1)-[r:{validated_relationship}]->(e2)
    SET r.created_by = 'bootstrap_healing',
        r.created_ts = timestamp(),
        r.confidence = 0.6
    """
    try:
        params = {"element1_id": element1.get("id"), "element2_id": element2.get("id")}
        await neo4j_manager.execute_write_query(query, params)

        if validated_relationship != relationship_type:
            logger.info(
                f"Created validated relationship (corrected from {relationship_type} to {validated_relationship}) "
                f"between '{element1.get('name')}' and '{element2.get('name')}'"
            )
        else:
            logger.info(
                f"Created contextual relationship between '{element1.get('name')}' and '{element2.get('name')}'"
            )
    except Exception as e:
        logger.error(f"Error creating contextual relationship: {e}", exc_info=True)


async def validate_single_relationship(
    subject_type: str, relationship_type: str, object_type: str
) -> str | None:
    """
    Normalize a relationship type to a consistent format.

    Args:
        subject_type: Type of the subject node (unused, kept for API compatibility)
        relationship_type: Proposed relationship type
        object_type: Type of the object node (unused, kept for API compatibility)

    Returns:
        Normalized relationship type
    """
    # Simply normalize the relationship type
    return validate_relationship_type(relationship_type)
