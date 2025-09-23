# utils/text_processing.py
import logging
import re
from typing import TYPE_CHECKING, Any

# Optional imports. We keep them out of module import path to avoid hard deps.
try:  # pragma: no cover - optional dependency
    from rapidfuzz.fuzz import partial_ratio_alignment  # type: ignore
except Exception:  # pragma: no cover - if rapidfuzz missing
    partial_ratio_alignment = None  # type: ignore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    pass


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


async def get_context_snippet_for_patch(
    original_text: str, problem: dict[str, Any], max_chars: int
) -> str:
    """Return a context snippet around the problem’s quote or start of text.

    Replaces the old _get_context_window_for_patch_llm shim with a proper helper
    in utils.text_processing. If the problem contains a quote, take a window
    centered on that quote; otherwise return the head of the text up to max_chars.
    """
    if not isinstance(original_text, str) or not original_text:
        return ""
    quote = None
    if isinstance(problem, dict):
        quote = problem.get("original_problem_quote_text") or problem.get(
            "quote_from_original_text"
        )
    if isinstance(quote, str) and quote:
        idx = original_text.find(quote)
        if idx != -1:
            left = max_chars // 2
            start = max(0, idx - left)
            end = min(len(original_text), idx + len(quote) + (max_chars - (idx - start)))
            snippet = original_text[start:end]
            return snippet[:max_chars]
    return original_text[:max_chars]


def validate_world_item_fields(
    category: str, name: str, item_id: str, allow_empty_name: bool = False
) -> tuple[str, str, str]:
    """Validate and normalize WorldItem core fields, providing defaults for missing values."""
    # Validate category
    if not category or not isinstance(category, str) or not category.strip():
        category = "other"

    # Validate name
    # Only set default name if allow_empty_name is False and name is actually missing/empty
    if (not allow_empty_name) and (
        not name or not isinstance(name, str) or not name.strip()
    ):
        name = "unnamed_element"

    # Validate ID: ensure deterministic, human-readable if possible
    if not item_id or not isinstance(item_id, str) or not item_id.strip():
        import hashlib

        norm_cat = _normalize_for_id(category) or "other"
        norm_name = _normalize_for_id(name) or "unnamed"
        base = f"{norm_cat}_{norm_name}"
        # If either part is too short or generic, append a short stable hash for uniqueness
        if norm_name in {"", "unnamed"} or norm_cat in {"", "other"}:
            suffix = hashlib.sha1(f"{category}:{name}".encode()).hexdigest()[:8]
            item_id = f"{base}_{suffix}"
        else:
            item_id = base

    return category, name, item_id


def normalize_trait_name(trait: str) -> str:
    """Return a canonical representation of a trait name."""
    if not isinstance(trait, str):
        trait = str(trait)
    cleaned = re.sub(r"[^a-z0-9 ]", "", trait.strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


class SpaCyModelManager:
    """Lazily loads and stores the spaCy model used across the project."""

    def __init__(self) -> None:
        self._nlp: Any | None = None

    @property
    def nlp(self) -> Any | None:
        return self._nlp

    def load(self) -> None:
        """Load the spaCy model if it hasn't been loaded yet.

        The import is performed lazily to avoid hard dependency and heavy
        startup costs in single-user CLI mode. Defaults to a small model,
        overridable by config (settings.SPACY_MODEL) or env.
        """
        if self._nlp is not None:
            return
        try:  # import spacy lazily
            import spacy  # type: ignore
        except Exception:
            logger.error(
                "spaCy library not installed. Install with: pip install spacy. "
                "spaCy-dependent features will be disabled."
            )
            self._nlp = None
            return

        # Choose model: prefer config setting, default to lightweight model
        model_name = None
        try:
            import config  # local import to avoid cycles

            model_name = getattr(config.settings, "SPACY_MODEL", None)
        except Exception:
            model_name = None
        if not model_name:
            model_name = "en_core_web_sm"

        try:
            self._nlp = spacy.load(model_name)  # type: ignore[name-defined]
            logger.info("spaCy model '%s' loaded.", model_name)
        except OSError:
            logger.error(
                "spaCy model '%s' not found. Install with: python -m spacy download %s. "
                "spaCy-dependent features will be disabled.",
                model_name,
                model_name,
            )
            self._nlp = None
        except Exception as e:
            logger.error("Failed to load spaCy model '%s': %s", model_name, e)
            self._nlp = None


spacy_manager = SpaCyModelManager()


def load_spacy_model_if_needed() -> None:
    """Load the spaCy model using the shared manager if needed."""
    spacy_manager.load()


def _normalize_text_for_matching(text: str) -> str:
    """Normalize text for more robust matching."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(
        r"^[ '\"\(]*(\.\.\.)?[ '\"\(]*|[ '\"\(]*(\.\.\.)?[ '\"\(]*$", "", text
    )
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


async def find_quote_and_sentence_offsets_with_spacy(
    doc_text: str, quote_text_from_llm: str
) -> tuple[int, int, int, int] | None:
    """Locate quote and sentence offsets within ``doc_text``."""
    load_spacy_model_if_needed()
    if not quote_text_from_llm.strip() or not doc_text.strip():
        logger.debug("find_quote_offsets: Empty quote_text or doc_text.")
        return None

    if "N/A - General Issue" in quote_text_from_llm:
        logger.debug(
            "Quote is '%s', treating as general issue. No offsets.", quote_text_from_llm
        )
        return None

    cleaned_llm_quote_for_direct_search = quote_text_from_llm.strip(" \"'.")
    if not cleaned_llm_quote_for_direct_search:
        logger.debug(
            "LLM quote became empty after basic stripping for direct search, cannot match."
        )
        return None

    # Prepare sentence segments regardless of spaCy availability
    sentences = get_text_segments(doc_text, segment_level="sentence")

    current_pos = 0
    while current_pos < len(doc_text):
        match_start = doc_text.lower().find(
            cleaned_llm_quote_for_direct_search.lower(), current_pos
        )
        if match_start == -1:
            break

        match_end = match_start + len(cleaned_llm_quote_for_direct_search)
        found_sentence_span = None
        for sent_text, s_start, s_end in sentences:
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
        alignment = partial_ratio_alignment(
            cleaned_llm_quote_for_direct_search, doc_text
        )
        if getattr(alignment, "score", 0.0) >= 85.0:
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
        min_similarity_threshold=0.65,
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


def get_text_segments(
    text: str, segment_level: str = "paragraph"
) -> list[tuple[str, int, int]]:
    """Segment text into paragraphs or sentences with offsets."""
    load_spacy_model_if_needed()
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
        if spacy_manager.nlp:
            doc = spacy_manager.nlp(text)
            for sent in doc.sents:
                sent_text_stripped = sent.text.strip()
                if sent_text_stripped:
                    segments.append(
                        (sent_text_stripped, sent.start_char, sent.end_char)
                    )
        else:
            logger.warning(
                "get_text_segments: spaCy model not loaded. Falling back to basic sentence segmentation (less accurate)."
            )
            for match in re.finditer(r"([^\.!?]+(?:[\.!?]|$))", text):
                sent_text_stripped = match.group(1).strip()
                if sent_text_stripped:
                    segments.append((sent_text_stripped, match.start(), match.end()))
            if not segments and text.strip():
                segments.append((text.strip(), 0, len(text)))
    else:
        raise ValueError(
            f"Unsupported segment_level for get_text_segments: {segment_level}"
        )

    return segments
