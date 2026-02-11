# utils/text_processing.py
import hashlib
import re
from typing import TYPE_CHECKING, Any

import structlog

# Optional imports. We keep them out of module import path to avoid hard deps.
try:  # pragma: no cover - optional dependency
    from rapidfuzz.fuzz import partial_ratio_alignment
except Exception:  # pragma: no cover - if rapidfuzz missing
    partial_ratio_alignment = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    pass


def normalize_entity_name(text: str) -> str:
    """
    Clean and normalize an entity name.

    1. Removes parenthetical content (e.g. "Name (description)" -> "Name")
    2. Normalizes smart quotes to straight quotes
    3. Strips whitespace
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    # Remove parenthetical content
    text = re.sub(r"\s*\(.*?\)", "", text)

    # Normalize smart quotes
    text = text.replace("'", "'").replace("‘", "'").replace("“", '"').replace("”", '"')

    return text.strip()


def _normalize_for_id(text: str) -> str:
    """Normalize a string for use in an ID."""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    # Handle empty string case
    if not text:
        return ""
    # Remove common leading articles to avoid ID duplicates
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"['\"()]", "", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text


def generate_entity_id(name: str, category: str) -> str:
    """Generate a deterministic entity ID.

    Args:
        name: Entity name.
        category: Entity category (for world items) or `"character"` for characters.

    Returns:
        A stable identifier derived from a normalized form of `name` and `category`.
    """
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if not isinstance(category, str):
        raise TypeError("category must be a string")

    normalized_name = re.sub(r"[^\w\s]", "", name.lower().strip())
    normalized_name = re.sub(r"\s+", "_", normalized_name)

    content_hash = hashlib.md5(f"{normalized_name}_{category}".encode()).hexdigest()[:12]

    return f"entity_{content_hash}"


async def get_context_snippet_for_patch(original_text: str, problem: dict[str, Any], max_chars: int) -> str:
    """Return a context snippet around the problem's quote or start of text.

    Replaces the old _get_context_window_for_patch_llm shim with a proper helper
    in utils.text_processing. If the problem contains a quote, take a window
    centered on that quote; otherwise return the head of the text up to max_chars.
    """
    if not isinstance(original_text, str) or not original_text:
        return ""
    quote = None
    if isinstance(problem, dict):
        quote = problem.get("original_problem_quote_text") or problem.get("quote_from_original_text")
    if isinstance(quote, str) and quote:
        idx = original_text.find(quote)
        if idx != -1:
            left = max_chars // 2
            start = max(0, idx - left)
            end = min(len(original_text), idx + len(quote) + (max_chars - (idx - start)))
            snippet = original_text[start:end]
            return snippet[:max_chars]
    return original_text[:max_chars]


def validate_world_item_fields(category: str, name: str, item_id: str, allow_empty_name: bool = False) -> tuple[str, str, str]:
    """Validate and normalize WorldItem core fields.

    Raises:
        ValueError: If category is missing/empty, or if name is missing/empty
            (when allow_empty_name is False).
    """
    if not isinstance(category, str) or not category.strip():
        raise ValueError(f"WorldItem category must be a non-empty string, got {category!r}")

    if not allow_empty_name and (not isinstance(name, str) or not name.strip()):
        raise ValueError(f"WorldItem name must be a non-empty string, got {name!r}")

    if not item_id or not isinstance(item_id, str) or not item_id.strip():
        import hashlib

        norm_cat = _normalize_for_id(category)
        norm_name = _normalize_for_id(name) if name else ""
        base = f"{norm_cat}_{norm_name}" if norm_name else norm_cat
        suffix = hashlib.sha1(f"{category}:{name}".encode()).hexdigest()[:12]
        item_id = f"{base}_{suffix}"

    return category, name, item_id


def classify_category_label(category: str) -> str:
    """
    Classify a category string to a canonical Neo4j node label.

    This is the authoritative category classification function used across the codebase
    for consistent label assignment. Supports 118 keyword variants across 7 label types.

    Args:
        category: The category string to classify

    Returns:
        Canonical Neo4j label: "Location", "Item" "Event",
        "Character", or "Chapter". Defaults to "Item" if ambiguous.

    Examples:
        >>> classify_category_label("place")
        'Location'
        >>> classify_category_label("weapon")
        'Item'
        >>> classify_category_label("battle")
        'Event'
    """
    c = (category or "").strip().lower()

    if c.title() in [
        "Location",
        "Event",
        "Item",
        "Character",
        "Chapter",
    ]:
        return c.title()

    if c in [
        "location",
        "locations",
        "place",
        "site",
        "area",
        "region",
        "settlement",
        "landmark",
        "room",
        "structure",
        "territory",
        "city",
        "town",
        "village",
        "path",
        "road",
        "planet",
        "landscape",
        "building",
    ]:
        return "Location"

    if c in [
        "item",
        "items",
        "object",
        "objects",
        "thing",
        "things",
        "artifact",
        "artifacts",
        "tool",
        "tools",
        "weapon",
        "weapons",
        "document",
        "documents",
        "relic",
        "relics",
        "book",
        "books",
        "scroll",
        "scrolls",
        "letter",
        "letters",
        "device",
        "devices",
        "vehicle",
        "vehicles",
        "resource",
        "resources",
        "material",
        "materials",
        "food",
        "clothing",
        "currency",
        "money",
        "treasure",
    ]:
        return "Item"

    if c in [
        "event",
        "events",
        "happening",
        "happenings",
        "occurrence",
        "occurrences",
        "incident",
        "incidents",
        "development",
        "developments",
        "battle",
        "battles",
        "meeting",
        "meetings",
        "journey",
        "journeys",
        "ceremony",
        "ceremonies",
        "discovery",
        "discoveries",
        "conflict",
        "conflicts",
        "conversation",
        "conversations",
        "flashback",
        "flashbacks",
        "scene",
        "scenes",
        "moment",
        "moments",
    ]:
        return "Event"

    if c in [
        "skill",
        "ability",
        "quality",
        "attribute",
        "status",
        "personality",
        "physical",
        "supernatural",
        "background",
    ]:
        return "Trait"

    if c in [
        "person",
        "people",
        "creature",
        "being",
        "spirit",
        "deity",
        "human",
        "npc",
        "protagonist",
        "antagonist",
        "supporting",
        "minor",
        "historical",
    ]:
        return "Character"

    return "Item"


def is_single_word_trait(trait: str) -> bool:
    """
    Check if a trait is a valid single-word trait.

    A valid single-word trait:
    - Contains only letters, numbers, and hyphens (no spaces)
    - Starts with a letter
    - Is not empty

    Args:
        trait: The trait to validate

    Returns:
        True if the trait is a valid single-word trait, False otherwise
    """
    if not isinstance(trait, str) or not trait:
        return False

    # Remove quotes if present (from JSON)
    trait = trait.strip('"').strip("'").strip()

    # Check pattern: must start with letter, can contain letters, numbers, hyphens
    pattern = r"^[a-zA-Z][a-zA-Z0-9-]*$"
    return bool(re.match(pattern, trait)) and " " not in trait


def normalize_trait_name(trait: str) -> str:
    """
    Return a canonical representation of a trait name.

    Enforces single-word trait requirement by:
    1. Converting to lowercase
    2. Removing spaces and replacing with hyphens (for multi-word traits)
    3. Removing special characters except hyphens
    4. Returning empty string if invalid

    Args:
        trait: The trait to normalize

    Returns:
        Normalized trait name or empty string if invalid
    """
    if not isinstance(trait, str):
        trait = str(trait)

    # Strip quotes and whitespace
    cleaned = trait.strip('"').strip("'").strip().lower()

    # Remove special characters except hyphens and spaces (temporarily)
    cleaned = re.sub(r"[^a-z0-9 -]", "", cleaned)

    # Replace multiple spaces with single hyphen
    cleaned = re.sub(r"\s+", "-", cleaned)

    # Remove leading/trailing hyphens
    cleaned = cleaned.strip("-")

    # Validate it's a single word (no spaces after normalization)
    if not cleaned or not is_single_word_trait(cleaned):
        return ""

    return cleaned


def validate_and_filter_traits(traits: list[str]) -> list[str]:
    """
    Validate and filter a list of traits to ensure all are single-word.

    Args:
        traits: List of trait strings to validate

    Returns:
        List of valid, normalized, unique traits
    """
    if not isinstance(traits, list):
        return []

    valid_traits = []
    seen = set()

    for trait in traits:
        normalized = normalize_trait_name(trait)
        if normalized and normalized not in seen:
            valid_traits.append(normalized)
            seen.add(normalized)

    return valid_traits


def _get_spacy_nlp() -> Any | None:
    """Return the shared spaCy Language model, or None if unavailable."""
    from core.spacy_service import get_spacy_service

    service = get_spacy_service()
    return service.nlp


def _normalize_text_for_matching(text: str) -> str:
    """Normalize text for more robust matching."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"^[ '\"\(]*(\.\.\.)?[ '\"\(]*|[ '\"\(]*(\.\.\.)?[ '\"\(]*$", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_similarity(a: str, b: str) -> float:
    """Return Jaccard similarity between token sets of ``a`` and ``b``."""
    tokens_a = set(_normalize_text_for_matching(a).split())
    tokens_b = set(_normalize_text_for_matching(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


async def find_quote_and_sentence_offsets_with_spacy(doc_text: str, quote_text_from_llm: str) -> tuple[int, int, int, int] | None:
    """Locate quote and sentence offsets within ``doc_text``."""
    if not quote_text_from_llm.strip() or not doc_text.strip():
        logger.debug("find_quote_offsets: Empty quote_text or doc_text.")
        return None

    if "N/A - General Issue" in quote_text_from_llm:
        logger.debug("Quote is '%s', treating as general issue. No offsets.", quote_text_from_llm)
        return None

    cleaned_llm_quote_for_direct_search = quote_text_from_llm.strip(" \"'.")
    if not cleaned_llm_quote_for_direct_search:
        logger.debug("LLM quote became empty after basic stripping for direct search, cannot match.")
        return None

    # Prepare sentence segments regardless of spaCy availability
    sentences = get_text_segments(doc_text, segment_level="sentence")

    current_pos = 0
    while current_pos < len(doc_text):
        match_start = doc_text.lower().find(cleaned_llm_quote_for_direct_search.lower(), current_pos)
        if match_start == -1:
            break

        match_end = match_start + len(cleaned_llm_quote_for_direct_search)
        found_sentence_span = None
        for _sent_text, s_start, s_end in sentences:
            if s_start <= match_start < s_end and s_start < match_end <= s_end:
                found_sentence_span = (s_start, s_end)
                break

        if found_sentence_span:
            logger.debug(
                "Direct Substring Match: Found LLM quote (approx) '%s...' at %d-%d in sentence %d-%d",
                cleaned_llm_quote_for_direct_search[:30],
                match_start,
                match_end,
                found_sentence_span[0],
                found_sentence_span[1],
            )
            return (
                match_start,
                match_end,
                found_sentence_span[0],
                found_sentence_span[1],
            )

        current_pos = match_end

    if partial_ratio_alignment is not None:
        alignment = partial_ratio_alignment(cleaned_llm_quote_for_direct_search, doc_text)
        if alignment is not None and getattr(alignment, "score", 0.0) >= 85.0:
            match_start = alignment.dest_start
            match_end = alignment.dest_end
            for _sent_text, s_start, s_end in sentences:
                if s_start <= match_start < s_end and s_start < match_end <= s_end:
                    logger.debug(
                        "Fuzzy Match: Found LLM quote (approx) '%s...' at %d-%d in sentence %d-%d (Score: %.2f)",
                        cleaned_llm_quote_for_direct_search[:30],
                        match_start,
                        match_end,
                        s_start,
                        s_end,
                        alignment.score,
                    )
                    return (
                        match_start,
                        match_end,
                        s_start,
                        s_end,
                    )

    # Token similarity fallback before expensive semantic search
    best_span = None
    best_sim = 0.0
    for sent_text, s_start, s_end in sentences:
        sim = _token_similarity(cleaned_llm_quote_for_direct_search, sent_text)
        if sim > best_sim:
            best_sim = sim
            best_span = (s_start, s_end)
    if best_span and best_sim >= 0.45:
        logger.debug(
            "Token Similarity Match: '%s...' most similar to sentence %d-%d (%.2f)",
            cleaned_llm_quote_for_direct_search[:30],
            best_span[0],
            best_span[1],
            best_sim,
        )
        return (
            best_span[0],
            best_span[1],
            best_span[0],
            best_span[1],
        )

    logger.debug(
        "Direct substring match failed for LLM quote '%s...'. Falling back to semantic sentence search.",
        quote_text_from_llm[:50],
    )
    from .similarity import find_semantically_closest_segment

    semantic_sentence_match = await find_semantically_closest_segment(
        original_doc=doc_text,
        query_text=quote_text_from_llm,
        segment_type="sentence",
        min_similarity_threshold=0.6,
    )

    if semantic_sentence_match:
        s_start, s_end, similarity = semantic_sentence_match
        logger.debug(
            "Semantic Match: Found sentence for LLM quote '%s...' from %d-%d (Similarity: %.2f). Using whole sentence as target.",
            quote_text_from_llm[:30],
            s_start,
            s_end,
            similarity,
        )
        return s_start, s_end, s_start, s_end

    logger.info(
        "Could not confidently locate quote TEXT from LLM: '%s...' in document using direct or semantic search.",
        quote_text_from_llm[:50],
    )
    return None


def get_text_segments(text: str, segment_level: str = "paragraph") -> list[tuple[str, int, int]]:
    """Segment text into paragraphs or sentences with offsets."""
    segments: list[tuple[str, int, int]] = []

    if not text.strip():
        return segments

    if segment_level == "paragraph":
        current_paragraph_lines: list[str] = []
        current_paragraph_start_char = -1

        for line_match in re.finditer(r"([^\r\n]*(?:\r\n|\r|\n)?)", text):
            line_text = line_match.group(0)
            line_text_stripped = line_text.strip()

            if line_text_stripped:
                if not current_paragraph_lines:
                    current_paragraph_start_char = line_match.start()
                current_paragraph_lines.append(line_text)
            else:
                if current_paragraph_lines:
                    full_para_text = "".join(current_paragraph_lines)
                    segments.append(
                        (
                            full_para_text.strip(),
                            current_paragraph_start_char,
                            current_paragraph_start_char + len(full_para_text),
                        )
                    )
                    current_paragraph_lines = []
                    current_paragraph_start_char = -1

        if current_paragraph_lines:
            full_para_text = "".join(current_paragraph_lines)
            segments.append(
                (
                    full_para_text.strip(),
                    current_paragraph_start_char,
                    current_paragraph_start_char + len(full_para_text),
                )
            )

        if not segments and text.strip():
            segments.append((text.strip(), 0, len(text)))

    elif segment_level == "sentence":
        nlp = _get_spacy_nlp()
        if nlp:
            doc = nlp(text)
            for sent in doc.sents:
                sent_text_stripped = sent.text.strip()
                if sent_text_stripped:
                    segments.append((sent_text_stripped, sent.start_char, sent.end_char))
        else:
            logger.warning("get_text_segments: spaCy model not loaded. Falling back to basic sentence segmentation (less accurate).")
            for match in re.finditer(r"([^\.!?]+(?:[\.!?]|$))", text):
                sent_text_stripped = match.group(1).strip()
                if sent_text_stripped:
                    segments.append((sent_text_stripped, match.start(), match.end()))
            if not segments and text.strip():
                segments.append((text.strip(), 0, len(text)))
    else:
        raise ValueError(f"Unsupported segment_level for get_text_segments: {segment_level}")

    return segments
