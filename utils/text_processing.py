import logging
import re
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import spacy
from rapidfuzz.fuzz import partial_ratio_alignment

import config

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    pass


def _normalize_for_id(text: str) -> str:
    """Normalize a string for use in an ID."""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    text = re.sub(r"['\"()]", "", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text


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
        self._nlp: Optional[spacy.language.Language] = None

    @property
    def nlp(self) -> Optional[spacy.language.Language]:
        return self._nlp

    def load(self) -> None:
        """Load the spaCy model if it hasn't been loaded yet."""
        if self._nlp is not None:
            return
        try:
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
        except OSError:
            logger.error(
                "spaCy model 'en_core_web_sm' not found. "
                "Please run: python -m spacy download en_core_web_sm. "
                "spaCy dependent features will be disabled."
            )
            self._nlp = None
        except ImportError:
            logger.error(
                "spaCy library not installed. Please install it: pip install spacy. "
                "spaCy dependent features will be disabled."
            )
            self._nlp = None


spacy_manager = SpaCyModelManager()


def _is_fill_in(value: Any) -> bool:
    """Return True if ``value`` is the fill-in placeholder."""
    return isinstance(value, str) and value == config.FILL_IN


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


async def find_quote_and_sentence_offsets_with_spacy(
    doc_text: str, quote_text_from_llm: str
) -> Optional[Tuple[int, int, int, int]]:
    """Locate quote and sentence offsets within ``doc_text``."""
    load_spacy_model_if_needed()
    if (
        spacy_manager.nlp is None
        or not quote_text_from_llm.strip()
        or not doc_text.strip()
    ):
        if spacy_manager.nlp is None:
            logger.debug("find_quote_offsets: spaCy model not loaded.")
        else:
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

    spacy_doc = spacy_manager.nlp(doc_text) if spacy_manager.nlp else None
    if spacy_doc is None:
        return None

    current_pos = 0
    while current_pos < len(doc_text):
        match_start = doc_text.lower().find(
            cleaned_llm_quote_for_direct_search.lower(), current_pos
        )
        if match_start == -1:
            break

        match_end = match_start + len(cleaned_llm_quote_for_direct_search)
        found_sentence_span = None
        for sent in spacy_doc.sents:
            if (
                sent.start_char <= match_start < sent.end_char
                and sent.start_char < match_end <= sent.end_char
            ):
                found_sentence_span = sent
                break

        if found_sentence_span:
            logger.info(
                "Direct Substring Match: Found LLM quote (approx) '%s...' at %d-%d in sentence %d-%d",
                cleaned_llm_quote_for_direct_search[:30],
                match_start,
                match_end,
                found_sentence_span.start_char,
                found_sentence_span.end_char,
            )
            return (
                match_start,
                match_end,
                found_sentence_span.start_char,
                found_sentence_span.end_char,
            )

        current_pos = match_end

    alignment = partial_ratio_alignment(cleaned_llm_quote_for_direct_search, doc_text)
    if alignment.score >= 85.0:
        match_start = alignment.dest_start
        match_end = alignment.dest_end
        for sent in spacy_doc.sents:
            if (
                sent.start_char <= match_start < sent.end_char
                and sent.start_char < match_end <= sent.end_char
            ):
                logger.info(
                    "Fuzzy Match: Found LLM quote (approx) '%s...' at %d-%d in sentence %d-%d (Score: %.2f)",
                    cleaned_llm_quote_for_direct_search[:30],
                    match_start,
                    match_end,
                    sent.start_char,
                    sent.end_char,
                    alignment.score,
                )
                return (
                    match_start,
                    match_end,
                    sent.start_char,
                    sent.end_char,
                )

    logger.warning(
        "Direct substring match failed for LLM quote '%s...'. Falling back to semantic sentence search.",
        quote_text_from_llm[:50],
    )
    from .similarity import find_semantically_closest_segment

    semantic_sentence_match = await find_semantically_closest_segment(
        original_doc=doc_text,
        query_text=quote_text_from_llm,
        segment_type="sentence",
        min_similarity_threshold=0.75,
    )

    if semantic_sentence_match:
        s_start, s_end, similarity = semantic_sentence_match
        logger.info(
            "Semantic Match: Found sentence for LLM quote '%s...' from %d-%d (Similarity: %.2f). Using whole sentence as target.",
            quote_text_from_llm[:30],
            s_start,
            s_end,
            similarity,
        )
        return s_start, s_end, s_start, s_end

    logger.warning(
        "Could not confidently locate quote TEXT from LLM: '%s...' in document using direct or semantic search.",
        quote_text_from_llm[:50],
    )
    return None


def get_text_segments(
    text: str, segment_level: str = "paragraph"
) -> List[Tuple[str, int, int]]:
    """Segment text into paragraphs or sentences with offsets."""
    load_spacy_model_if_needed()
    segments: List[Tuple[str, int, int]] = []

    if not text.strip():
        return segments

    if segment_level == "paragraph":
        current_paragraph_lines: List[str] = []
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
