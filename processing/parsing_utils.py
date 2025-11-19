# processing/parsing_utils.py
from __future__ import annotations

from typing import Any

# from rdflib import Graph, URIRef, Literal, BNode # No longer needed for triples
# from rdflib.namespace import RDF, RDFS # No longer needed for triples
import structlog

logger = structlog.get_logger(__name__)


class ParseError(Exception):
    """Custom exception for parsing errors (unused)."""


# --- New RDF Triple Parsing using rdflib ---
# Modified to be a custom plain-text triple parser


def _get_entity_type_and_name_from_text(entity_text: str) -> dict[str, str | None]:
    """
    Parses 'EntityType:EntityName' or just 'EntityName' string.
    If EntityType is missing, it's set to None.
    """
    name_part = entity_text
    type_part = None
    if ":" in entity_text:
        parts = entity_text.split(":", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            type_part = parts[0].strip()
            name_part = parts[1].strip()
        elif (
            parts[0].strip()
        ):  # Only one part before colon, might be a type or a name with an odd colon
            # Heuristic: if it starts with uppercase and has no spaces, assume it's a type and name is missing/error.
            # Or if it's a common entity type. For now, simpler: if only one part before ':', it's the type.
            # This logic might need refinement if LLM is inconsistent.
            # Let's assume if one part before ':', it's the type and the rest is name.
            # If no part after ':', then name is effectively empty.
            type_part = parts[0].strip()
            name_part = parts[1].strip() if len(parts) > 1 else ""

    return {
        "type": type_part if type_part else None,
        "name": name_part.strip() if name_part else None,
    }


# Blacklisted entity patterns that should not be created as entities
ENTITY_BLACKLIST_PATTERNS = [
    # Sensory/descriptive elements
    "resonant hum",
    "pulsing",
    "violet glow",
    "glow",
    "light",
    "sound",
    "hum",
    "color",
    "bright",
    "dim",
    "loud",
    "quiet",
    "warm",
    "cold",
    # Emotions and feelings
    "belonging",
    "fear",
    "joy",
    "sadness",
    "anger",
    "love",
    "hate",
    "feeling",
    "emotion",
    "sentiment",
    "mood",
    # Abstract descriptive concepts
    "memory as survival",
    "return is possible",
    "awakening",
    "recognition",
    # Pipeline/internal artifact terms to suppress
    "relationships",
    "relationship_updates",
    # Physical descriptions
    "height",
    "weight",
    "size",
    "shape",
    "appearance",
    # Temporal/ephemeral states
    "moment",
    "instant",
    "second",
    "minute",
    "now",
    "then",
    "current",
]


def _is_proper_noun(entity_name: str) -> bool:
    """
    Detect if an entity name is likely a proper noun.

    Proper nouns are names that:
    1. Have most words capitalized (excluding articles/prepositions)
    2. Are not generic descriptors like "the rebellion" or "the artifact"
    3. Represent specific named entities

    Args:
        entity_name: The entity name to check

    Returns:
        True if the name appears to be a proper noun
    """
    if not entity_name or not entity_name.strip():
        return False

    words = entity_name.strip().split()
    if not words:
        return False

    # Articles and prepositions that don't count toward proper noun status
    lowercase_allowed = {
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
    }

    # Count words that should be capitalized (excluding allowed lowercase words)
    significant_words = [w for w in words if w.lower() not in lowercase_allowed]
    if not significant_words:
        return False

    capitalized_count = sum(1 for w in significant_words if w and w[0].isupper())

    # Proper noun if 60%+ of significant words are capitalized
    is_mostly_capitalized = capitalized_count >= len(significant_words) * 0.6

    # Additional heuristic: filter out generic patterns even if capitalized
    name_lower = entity_name.lower().strip()
    generic_patterns = [
        "the ",  # "the Rebellion" might be generic vs "Rebellion of the North"
    ]

    # If it starts with "the " and only has 1-2 words total, likely generic
    if name_lower.startswith("the ") and len(words) <= 2:
        return False

    return is_mostly_capitalized


def _should_filter_entity(
    entity_name: str, entity_type: str = None, mention_count: int = 1
) -> bool:
    """
    Filter out problematic entity names that create noise in the knowledge graph.

    Uses a hybrid approach with proper noun preference:
    - Proper nouns: Accepted with 1+ mentions
    - Common nouns: Require 3+ mentions to be considered significant

    Args:
        entity_name: The name of the entity to check
        entity_type: Optional type for additional filtering
        mention_count: Number of times entity is mentioned (default: 1)

    Returns:
        True if entity should be filtered out (not created)
    """
    if not entity_name:
        return True

    name_lower = entity_name.lower().strip()

    # Filter very short or generic names
    if len(name_lower) <= 2:
        return True

    # Filter blacklisted patterns
    for pattern in ENTITY_BLACKLIST_PATTERNS:
        if pattern in name_lower:
            return True

    # Filter entities that are just adjectives or descriptive words
    descriptive_words = {
        "beautiful",
        "ugly",
        "big",
        "small",
        "fast",
        "slow",
        "new",
        "old",
        "good",
        "bad",
        "high",
        "low",
        "strong",
        "weak",
        "bright",
        "dark",
    }
    if name_lower in descriptive_words:
        return True

    # Filter ephemeral/internal placeholder ids like 'entity_4097e8ba'
    if name_lower.startswith("entity_"):
        return True

    # Proper noun preference: use tiered mention thresholds
    is_proper = _is_proper_noun(entity_name)

    try:
        import config

        proper_noun_threshold = getattr(
            config, "ENTITY_MENTION_THRESHOLD_PROPER_NOUN", 1
        )
        common_noun_threshold = getattr(
            config, "ENTITY_MENTION_THRESHOLD_COMMON_NOUN", 3
        )
    except ImportError:
        # Fallback defaults if config not available
        proper_noun_threshold = 1
        common_noun_threshold = 3

    if is_proper:
        # Proper nouns: keep if mentioned threshold or more times
        if mention_count < proper_noun_threshold:
            logger.debug(
                f"Filtered proper noun '{entity_name}' with {mention_count} mentions (threshold: {proper_noun_threshold})"
            )
            return True
    else:
        # Common nouns: require higher threshold
        if mention_count < common_noun_threshold:
            logger.debug(
                f"Filtered common noun '{entity_name}' with {mention_count} mentions (threshold: {common_noun_threshold})"
            )
            return True

    return False


def parse_llm_triples(
    text_block: str,
) -> list[dict[str, Any]]:
    """
    Custom parser for LLM-generated plain text triples.
    Expected format: 'SubjectEntityType:SubjectName | Predicate | ObjectEntityType:ObjectName'
                 OR 'SubjectEntityType:SubjectName | Predicate | LiteralValue'

    Includes filtering to prevent creation of problematic entities.
    """
    logger_func = logger
    triples_list: list[dict[str, Any]] = []
    if not text_block or not text_block.strip():
        return triples_list

    lines = text_block.strip().splitlines()

    for line_num, line in enumerate(lines):
        line = line.strip()
        if (
            not line or line.startswith("#") or line.startswith("//")
        ):  # Skip empty or comment lines
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            logger_func.warning(
                f"Line {line_num + 1}: Malformed triple (expected at least 3 parts separated by '|'): '{line}'"
            )
            continue

        subject_text = parts[0]
        predicate_text = parts[1]
        object_text = parts[2]

        if len(parts) > 3:
            logger_func.debug(
                f"Line {line_num + 1}: Triple had extra parts which are being ignored: '{' | '.join(parts[3:])}' from original line '{line}'"
            )

        subject_details = _get_entity_type_and_name_from_text(subject_text)
        pred_norm = predicate_text.strip().upper().replace(" ", "_")
        try:
            import config  # local import to avoid cycles

            if pred_norm == "STATUS_IS" and not getattr(
                config, "ENABLE_STATUS_IS_ALIAS", True
            ):
                pred_norm = "HAS_STATUS"
        except Exception:
            pass

        predicate_str = pred_norm  # Normalize predicate

        if not subject_details.get("name") or not predicate_str:
            logger_func.warning(
                f"Line {line_num + 1}: Missing subject name or predicate: S='{subject_text}', P='{predicate_text}'"
            )
            continue

        # Determine if object is an entity or a literal
        # If object_text contains 'EntityType:', assume it's an entity.
        # Otherwise, treat as a literal value.
        object_entity_payload: dict[str, str | None] | None = None
        object_literal_payload: str | None = None
        is_literal_object = True  # Default to literal

        if ":" in object_text:
            obj_parts_check = object_text.split(":", 1)
            # Heuristic: if part before colon is a known type or capitalized, assume entity
            # This can be made more robust by checking against a list of known types.
            potential_obj_type = obj_parts_check[0].strip()
            # A simple check: if it's capitalized and has no spaces, maybe it's a type.
            # Or if it matches any of the example types.
            # For now, if a colon is present and there's content on both sides, assume it's Type:Name
            if (
                len(obj_parts_check) == 2
                and obj_parts_check[0].strip()
                and obj_parts_check[1].strip()
            ):
                # Check if potential_obj_type is likely an entity type (e.g. starts with uppercase)
                if potential_obj_type[0].isupper() and " " not in potential_obj_type:
                    object_entity_payload = _get_entity_type_and_name_from_text(
                        object_text
                    )
                    is_literal_object = False

        if is_literal_object:
            object_literal_payload = object_text.strip()
            # Further clean if it's a string literal that might have quotes (LLM sometimes adds them)
            if object_literal_payload.startswith(
                '"'
            ) and object_literal_payload.endswith('"'):
                object_literal_payload = object_literal_payload[1:-1]
            if object_literal_payload.startswith(
                "'"
            ) and object_literal_payload.endswith("'"):
                object_literal_payload = object_literal_payload[1:-1]

        if not is_literal_object and (
            not object_entity_payload or not object_entity_payload.get("name")
        ):
            # This means we thought it was an entity due to ':', but parsing failed to get a name.
            # So, revert to treating it as a literal.
            logger_func.debug(
                f"Line {line_num + 1}: Object '{object_text}' looked like entity but parsed no name. Reverting to literal."
            )
            object_literal_payload = object_text.strip()
            is_literal_object = True
            object_entity_payload = None

        # Apply filtering to prevent problematic entities
        subject_name = subject_details.get("name")
        if _should_filter_entity(subject_name, subject_details.get("type")):
            logger_func.info(
                f"Line {line_num + 1}: Filtered out problematic subject entity: '{subject_name}'"
            )
            continue

        # Filter object entity if it exists
        if not is_literal_object and object_entity_payload:
            object_name = object_entity_payload.get("name")
            if _should_filter_entity(object_name, object_entity_payload.get("type")):
                logger_func.info(
                    f"Line {line_num + 1}: Filtered out problematic object entity: '{object_name}'"
                )
                continue

        # Drop triples with empty/null literal objects
        if is_literal_object:
            if (
                object_literal_payload is None
                or not str(object_literal_payload).strip()
                or str(object_literal_payload).strip().lower() in {"none", "null"}
            ):
                logger_func.info(
                    f"Line {line_num + 1}: Filtered out triple with null/empty literal object"
                )
                continue

        triples_list.append(
            {
                "subject": subject_details,
                "predicate": predicate_str,
                "object_entity": object_entity_payload,
                "object_literal": object_literal_payload,
                "is_literal_object": is_literal_object,
            }
        )

    return triples_list


"""
Note: The former parse_rdf_triples_with_rdflib wrapper has been removed.
Callers should use parse_llm_triples directly.
"""
