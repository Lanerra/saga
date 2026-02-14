# processing/parsing_utils.py
from __future__ import annotations

from typing import Any

import structlog

from models.kg_constants import VALID_NODE_LABELS
from utils.text_processing import normalize_entity_name

logger = structlog.get_logger(__name__)

# Pre-compute normalized labels for case-insensitive matching
_NORMALIZED_NODE_LABELS = {label.lower(): label for label in VALID_NODE_LABELS}


def _get_entity_type_and_name_from_text(entity_text: str) -> dict[str, str | None]:
    """Parse an entity reference into an optional type and name.

    This parser is designed for LLM-produced entity fields. It supports both explicit
    `"Type: Name"` references and implicit references where the first token may be a known type.

    Args:
        entity_text: Raw entity field text.

    Returns:
        A mapping with keys `type` and `name`. If no type is detected, `type` is `None`. If the
        input is blank or contains only a type marker, `name` is `None`.

    Notes:
        Heuristics (evaluated in order):
        - `"Type: Name"` yields `type="Type"` and `name="Name"`.
        - `"Type:"` yields `type="Type"` and `name=None`.
        - `"Type Name"` yields `type="Type"` and `name="Name"` when `Type` matches a known node
          label (case-insensitive).
        - `"Type"` yields `type="Type"` and `name=None` when `Type` matches a known node label.
        - Otherwise the entire input is treated as a name with no type.
    """
    text = entity_text.strip()
    if not text:
        return {"type": None, "name": None}

    name_part: str | None = text
    type_part: str | None = None

    if ":" in text:
        # Case 1 & 2: Explicit separator
        parts = text.split(":", 1)
        part1 = parts[0].strip()
        part2 = parts[1].strip() if len(parts) > 1 else ""

        if part1 and part2:
            type_part = part1
            name_part = part2
        elif part1 and not part2:
            # "Type:" case
            # Use heuristic to decide if part1 is a type or just a name ending in colon
            # If it's a known label or follows Type conventions (Capitalized, no spaces), assume Type.
            if part1.lower() in _NORMALIZED_NODE_LABELS:
                type_part = _NORMALIZED_NODE_LABELS[part1.lower()]
                name_part = None
            elif part1[0].isupper() and " " not in part1:
                type_part = part1
                name_part = None
            else:
                # Ambiguous, but previous logic favored Type. Keeping it as Type for consistency with "Type:" pattern.
                type_part = part1
                name_part = None
    else:
        # Case 3 & 4 & 5: No separator
        # Try to match the first word against known node labels
        parts = text.split(maxsplit=1)
        if parts:
            first_word = parts[0].strip()
            if first_word.lower() in _NORMALIZED_NODE_LABELS:
                # Found a known type prefix
                type_part = _NORMALIZED_NODE_LABELS[first_word.lower()]
                if len(parts) > 1:
                    name_part = parts[1].strip()
                else:
                    name_part = None
            else:
                # First word is not a known type, treat whole text as name
                type_part = None
                name_part = text

    # Clean up name_part to remove parenthetical explanations and normalize quotes
    if name_part:
        name_part = normalize_entity_name(name_part)

    return {
        "type": type_part if type_part else None,
        "name": name_part if name_part else None,
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
    # Abstract/Metaphysical concepts (Aggressive filtering)
    "memory",
    "voice",
    "signal",
    "echo",
    "ghost",
    "shadow",
    "silence",
    "darkness",
    "light",
    "dead",
    "alive",
    "awake",
    "asleep",
    "gone",
    "lost",
    "found",
    "pulse",
    "thrum",
    "beacon",
    "network",
    "void",
    "abyss",
    "truth",
    "lie",
    "hope",
    "despair",
]


def _is_proper_noun(entity_name: str) -> bool:
    """Detect whether an entity name looks like a proper noun.

    Args:
        entity_name: Candidate entity name.

    Returns:
        `True` when the name appears to be a specific named entity; otherwise `False`.

    Notes:
        Heuristics:
        - Proper nouns are inferred from capitalization: at least 60% of "significant" words are
          capitalized. Articles and common prepositions are excluded from the denominator.
        - Pure numbers (e.g., `"1984"`) are treated as consistent with proper-noun formatting.
        - Two-word generic determiners like `"The Room"` or `"A Man"` are treated as not proper.
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

    def _is_word_capitalized_or_special(w: str) -> bool:
        if not w:
            return False
        # Standard capitalization
        if w[0].isupper():
            return True

        # Check for mixed alphanumeric or pure numbers (e.g. "7-G", "Room 101")
        has_letters = any(c.isalpha() for c in w)
        if not has_letters:
            # Pure number or symbol like "7", "1984" - treat as consistent with proper noun
            return True

        # Mixed alphanumeric: "7-G" (starts with number but has upper letter)
        if not w[0].isalpha() and any(c.isupper() for c in w):
            return True

        return False

    capitalized_count = sum(1 for w in significant_words if _is_word_capitalized_or_special(w))

    # Proper noun if 60%+ of significant words are capitalized
    is_mostly_capitalized = capitalized_count >= len(significant_words) * 0.6

    # Additional heuristic: filter out generic patterns even if capitalized
    name_lower = entity_name.lower().strip()

    # If it starts with "the ", "a ", "an " and only has 1-2 words total, likely generic
    # e.g. "The Room", "A Man" -> False (not proper)
    # "The Order of the Phoenix" -> True (proper)
    if (name_lower.startswith("the ") or name_lower.startswith("a ") or name_lower.startswith("an ")) and len(words) <= 2:
        return False

    return is_mostly_capitalized


def _should_filter_entity(entity_name: str | None, entity_type: str | None = None, mention_count: int = 1) -> bool:
    """Filter out entity names that are likely to create noisy nodes.

    Args:
        entity_name: Candidate entity name.
        entity_type: Optional entity type, reserved for additional filtering.
        mention_count: Observed mention count used for thresholding.

    Returns:
        `True` if the entity should be suppressed; otherwise `False`.

    Notes:
        This filter is intentionally conservative about creating nodes. It suppresses:
        - Empty names, very short names (length <= 2), and names that look like internal IDs
          (prefix `"entity_"`).
        - Generic determiners (e.g., `"a X"`/`"an X"` with <= 2 words) and negations (`"not X"`).
        - Abstract phrase prefixes (e.g., `"sense of ..."`, `"memory of ..."`).
        - A configurable blacklist of substrings associated with sensory/emotional/abstract filler.
          Exact matches are always suppressed; substrings are allowed only when the name is inferred
          to be a proper noun.
        - A small set of single-word descriptive adjectives.

        Mention-count thresholds:
        - Proper nouns require at least `config.ENTITY_MENTION_THRESHOLD_PROPER_NOUN` mentions
          (default `1` if config cannot be imported).
        - Common nouns require at least `config.ENTITY_MENTION_THRESHOLD_COMMON_NOUN` mentions
          (default `3` if config cannot be imported).
    """
    if not entity_name:
        return True

    name_lower = entity_name.lower().strip()

    # Filter very short or generic names
    if len(name_lower) <= 2:
        return True

    # Proper noun preference: use tiered mention thresholds
    is_proper = _is_proper_noun(entity_name)

    # Filter blacklisted patterns
    for pattern in ENTITY_BLACKLIST_PATTERNS:
        if pattern in name_lower:
            # If exact match, always filter
            if pattern == name_lower:
                return True

            # If it's a substring...
            # If it's a Proper Noun containing the pattern (e.g. "Echo Protocol" contains "echo"),
            # we allow it.
            if is_proper:
                continue

            # If it's NOT a proper noun and contains the pattern, filter it.
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

    if name_lower.startswith("entity_"):
        return True

    # Filter generic "A [Noun]" patterns (e.g. "A Man", "An Apple")
    # These are usually not specific enough to be knowledge graph nodes
    if (name_lower.startswith("a ") or name_lower.startswith("an ")) and len(name_lower.split()) <= 2:
        return True

    # Filter "Not X" patterns (e.g., "Not Dead", "Not Gone")
    if name_lower.startswith("not "):
        return True

    # Filter abstract phrase starts
    abstract_prefixes = (
        "sense of",
        "feeling of",
        "sound of",
        "memory of",
        "vision of",
        "dream of",
        "thought of",
        "concept of",
        "idea of",
        "state of",
    )
    if name_lower.startswith(abstract_prefixes):
        return True

    try:
        import config

        proper_noun_threshold = getattr(config, "ENTITY_MENTION_THRESHOLD_PROPER_NOUN", 1)
        common_noun_threshold = getattr(config, "ENTITY_MENTION_THRESHOLD_COMMON_NOUN", 3)
    except ImportError:
        # Fallback defaults if config not available
        proper_noun_threshold = 1
        common_noun_threshold = 3

    if is_proper:
        # Proper nouns: keep if mentioned threshold or more times
        if mention_count < proper_noun_threshold:
            logger.debug(f"Filtered proper noun '{entity_name}' with {mention_count} mentions (threshold: {proper_noun_threshold})")
            return True
    else:
        # Common nouns: require higher threshold
        if mention_count < common_noun_threshold:
            logger.debug(f"Filtered common noun '{entity_name}' with {mention_count} mentions (threshold: {common_noun_threshold})")
            return True

    return False


def parse_llm_triples(
    text_block: str,
) -> list[dict[str, Any]]:
    """Parse LLM-generated plain-text triples into structured payloads.

    Args:
        text_block: A newline-delimited block of triples. Each non-empty, non-comment line is
            expected to contain at least three `|`-separated fields: subject, predicate, object.

    Returns:
        A list of triple payloads. Each payload contains:
        - `subject`: Mapping with `type` and `name` fields.
        - `predicate`: Normalized predicate string.
        - `object_entity`: Mapping with `type` and `name` when the object is treated as an entity.
        - `object_literal`: String when the object is treated as a literal.
        - `is_literal_object`: Whether the object was treated as a literal.

    Notes:
        Input format:
        - Subject and entity objects may be provided as `"Type:Name"` or `"Type Name"` where `Type`
          matches a known node label (case-insensitive). Otherwise they are treated as untyped.
        - Predicate is normalized by trimming, converting to uppercase, and replacing spaces with
          underscores.
        - Lines starting with `#` or `//` are treated as comments and skipped.
        - If a line contains more than three `|`-separated fields, extra fields are ignored.

        Entity vs literal objects:
        - Objects are treated as entities only when a type is detected by
          `_get_entity_type_and_name_from_text()`. Untyped objects default to literals to avoid
          turning arbitrary strings into nodes.
        - If an object looks typed but parses without a name, it is reverted to a literal.
        - Literal objects equal to empty/whitespace or case-insensitive `none`/`null` are dropped.

        Filtering:
        - Subjects and entity objects are filtered through `_should_filter_entity()` to suppress
          noisy node creation.

        Determinism:
        - The returned list preserves the order of accepted lines in `text_block`.
        - The only conditional predicate rewrite is `STATUS_IS` → `HAS_STATUS` when
          `config.ENABLE_STATUS_IS_ALIAS` is disabled; failures to read config fall back to the
          default behavior.
    """
    logger_func = logger
    triples_list: list[dict[str, Any]] = []
    if not text_block or not text_block.strip():
        return triples_list

    lines = text_block.strip().splitlines()

    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):  # Skip empty or comment lines
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            logger_func.warning(f"Line {line_num + 1}: Malformed triple (expected at least 3 parts separated by '|'): '{line}'")
            continue

        subject_text = parts[0]
        predicate_text = parts[1]
        object_text = parts[2]

        if len(parts) > 3:
            logger_func.debug(f"Line {line_num + 1}: Triple had extra parts which are being ignored: '{' | '.join(parts[3:])}' from original line '{line}'")

        subject_details = _get_entity_type_and_name_from_text(subject_text)
        pred_norm = predicate_text.strip().upper().replace(" ", "_")
        try:
            import config  # local import to avoid cycles

            if pred_norm == "STATUS_IS" and not getattr(config, "ENABLE_STATUS_IS_ALIAS", True):
                pred_norm = "HAS_STATUS"
        except Exception as e:
            logger.warning(
                "Failed to check ENABLE_STATUS_IS_ALIAS config, using default behavior",
                predicate=pred_norm,
                error=str(e),
            )

        predicate_str = pred_norm  # Normalize predicate

        if not subject_details.get("name") or not predicate_str:
            logger_func.warning(f"Line {line_num + 1}: Missing subject name or predicate: S='{subject_text}', P='{predicate_text}'")
            continue

        # Determine if object is an entity or a literal
        # Uses _get_entity_type_and_name_from_text to check for valid "Type: Name" or "Type Name" patterns.
        #
        # Heuristics:
        # 1. If it parses with a detected Type, it's an Entity.
        # 2. If it parses with only a Name (Type=None), it's a Literal.
        #    (We default to literal for objects to avoid turning every string into an entity node)

        object_entity_payload: dict[str, str | None] | None = None
        object_literal_payload: str | None = None
        is_literal_object = True  # Default to literal

        # Attempt to parse as entity
        potential_entity = _get_entity_type_and_name_from_text(object_text)

        # If we found a valid type, treat as entity
        if potential_entity.get("type"):
            object_entity_payload = potential_entity
            is_literal_object = False
        else:
            # No type found, treat as literal
            is_literal_object = True

        if is_literal_object:
            object_literal_payload = object_text.strip()
            # Further clean if it's a string literal that might have quotes (LLM sometimes adds them)
            if object_literal_payload.startswith('"') and object_literal_payload.endswith('"'):
                object_literal_payload = object_literal_payload[1:-1]
            if object_literal_payload.startswith("'") and object_literal_payload.endswith("'"):
                object_literal_payload = object_literal_payload[1:-1]

        if not is_literal_object and (not object_entity_payload or not object_entity_payload.get("name")):
            # This means we thought it was an entity due to ':', but parsing failed to get a name.
            # So, revert to treating it as a literal.
            logger_func.debug(f"Line {line_num + 1}: Object '{object_text}' looked like entity but parsed no name. Reverting to literal.")
            object_literal_payload = object_text.strip()
            is_literal_object = True
            object_entity_payload = None

        # Apply filtering to prevent problematic entities
        subject_name = subject_details.get("name")
        if subject_name and _should_filter_entity(subject_name, subject_details.get("type")):
            logger_func.info(f"Line {line_num + 1}: Filtered out problematic subject entity: '{subject_name}'")
            continue

        # Filter object entity if it exists
        if not is_literal_object and object_entity_payload:
            object_name = object_entity_payload.get("name")
            if object_name and _should_filter_entity(object_name, object_entity_payload.get("type")):
                logger_func.info(f"Line {line_num + 1}: Filtered out problematic object entity: '{object_name}'")
                continue

        # Drop triples with empty/null literal objects
        if is_literal_object:
            if object_literal_payload is None or not str(object_literal_payload).strip() or str(object_literal_payload).strip().lower() in {"none", "null"}:
                logger_func.info(f"Line {line_num + 1}: Filtered out triple with null/empty literal object")
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
