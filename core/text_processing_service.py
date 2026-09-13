# core/text_processing_service.py
"""Process and normalize LLM-related text in SAGA.

This module provides utilities for:
- Token counting and token-budget truncation.
- Cleaning LLM responses to remove provider artifacts and wrapper phrases.

Notes:
    Tokenization uses `tiktoken` when available for the requested model; otherwise it
    falls back to a character-based heuristic. The fallback is less accurate and should
    be treated as an approximation.
"""

import re
from typing import Any

import structlog
import tiktoken

import config
from core.spacy_service import get_spacy_service

logger = structlog.get_logger(__name__)


class TokenizerService:
    """Count tokens and truncate text to token budgets."""

    def __init__(self) -> None:
        """Initialize the tokenizer service."""
        self._tokenizer_cache: dict[str, tiktoken.Encoding] = {}
        self._stats = {
            "tokenizer_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "fallback_used": 0,
        }

    def get_tokenizer(self, model_name: str) -> tiktoken.Encoding | None:
        """Return a cached `tiktoken` encoder for a model name.

        Args:
            model_name: Provider model identifier.

        Returns:
            Encoder instance when available, otherwise None.

        Notes:
            When no model-specific encoding exists, this falls back to the configured
            default encoding.
        """
        self._stats["tokenizer_requests"] += 1

        if model_name in self._tokenizer_cache:
            self._stats["cache_hits"] += 1
            return self._tokenizer_cache[model_name]

        self._stats["cache_misses"] += 1

        try:
            # Try model-specific encoding first
            try:
                encoder = tiktoken.encoding_for_model(model_name)
            except KeyError:
                logger.debug(f"No direct tiktoken encoding for '{model_name}'. " f"Using default '{config.TIKTOKEN_DEFAULT_ENCODING}'.")
                encoder = tiktoken.get_encoding(config.TIKTOKEN_DEFAULT_ENCODING)

            self._tokenizer_cache[model_name] = encoder
            logger.debug(f"Tokenizer for model '{model_name}' (using actual encoder '{encoder.name}') found and cached.")
            return encoder

        except KeyError:
            logger.error(f"Default tiktoken encoding '{config.TIKTOKEN_DEFAULT_ENCODING}' also not found. " f"Token counting will fall back to character-based heuristic for '{model_name}'.")
            return None

        except Exception as e:
            logger.error(
                f"Unexpected error getting tokenizer for '{model_name}': {e}",
                exc_info=True,
            )
            return None

    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens in a text input for a given model.

        Args:
            text: Input text.
            model_name: Provider model identifier.

        Returns:
            Token count.

        Notes:
            When `tiktoken` encoding is unavailable, this uses a character-based heuristic.
        """
        if not text:
            return 0

        encoder = self.get_tokenizer(model_name)

        if encoder:
            return len(encoder.encode(text, allowed_special="all"))
        else:
            # Fallback to character-based estimation
            self._stats["fallback_used"] += 1
            token_estimate = len(text.encode("utf-8"))
            logger.warning("Token encoding unavailable; using conservative UTF-8 byte budget", model=model_name)
            return token_estimate

    def truncate_text_by_tokens(
        self,
        text: str,
        model_name: str,
        max_tokens: int,
        truncation_marker: str = "\n... (truncated)",
    ) -> str:
        """Truncate a text input to a token budget.

        Args:
            text: Input text.
            model_name: Provider model identifier.
            max_tokens: Maximum tokens to keep, including the truncation marker tokens.
            truncation_marker: Marker appended when truncation occurs.

        Returns:
            Possibly truncated text.

        Notes:
            When `tiktoken` encoding is unavailable, this uses a character-based fallback
            approximation.
        """
        if type(max_tokens) is not int or max_tokens < 0:
            raise ValueError("Text token budget must be a nonnegative integer")
        if not text or max_tokens == 0:
            return ""
        encoder = self.get_tokenizer(model_name)

        def measure(value: str) -> int:
            return len(encoder.encode(value, allowed_special="all")) if encoder else len(value.encode("utf-8"))

        if measure(text) <= max_tokens:
            return text
        marker = truncation_marker if measure(truncation_marker) < max_tokens else ""
        lower, upper = 0, len(text)
        retained = marker
        while lower <= upper:
            middle = (lower + upper) // 2
            candidate = text[:middle] + marker
            if measure(candidate) <= max_tokens:
                retained = candidate
                lower = middle + 1
            else:
                upper = middle - 1
        return retained

    def get_statistics(self) -> dict[str, Any]:
        """Get tokenizer service statistics."""
        total_requests = self._stats["tokenizer_requests"]
        return {
            **self._stats,
            "cache_hit_rate": (self._stats["cache_hits"] / total_requests * 100) if total_requests > 0 else 0,
            "fallback_rate": (self._stats["fallback_used"] / total_requests * 100) if total_requests > 0 else 0,
        }


class ResponseCleaningService:
    """Remove common LLM response artifacts and wrapper phrases."""

    def __init__(self) -> None:
        """Initialize the response cleaning service."""
        self._stats = {
            "responses_cleaned": 0,
            "think_tags_removed": 0,
            "code_blocks_cleaned": 0,
            "phrases_removed": 0,
            "significant_reductions": 0,  # >0.5% reduction
        }

        # Pre-compile regex patterns for better performance
        self._think_tags = [
            "think",
            "thought",
            "thinking",
            "reasoning",
            "rationale",
            "meta",
            "reflection",
            "internal_monologue",
            "plan",
            "analysis",
            "no_think",
        ]

        self._patterns: dict[str, re.Pattern[str]] = {}
        self._phrase_patterns: list[re.Pattern[str]] = []
        self._compile_cleaning_patterns()

    def _compile_cleaning_patterns(self) -> None:
        """Compile patterns for outer artifacts, never answer-interior rewriting."""
        tag_alternation = "|".join(re.escape(tag) for tag in self._think_tags)

        self._patterns = {
            "reasoning_tag": re.compile(
                rf"<\s*(/?)\s*({tag_alternation})\s*(/?)\s*>",
                flags=re.IGNORECASE,
            ),
            "think_boundary": re.compile(
                r"^[ \t]*<\s*/\s*think\s*>[ \t]*(?:\r?\n|$)",
                flags=re.IGNORECASE | re.MULTILINE,
            ),
            "code_blocks": re.compile(
                r"\A\s*```(?:[a-zA-Z0-9_-]+)?[ \t]*\r?\n(.*?)\r?\n```\s*\Z",
                flags=re.DOTALL,
            ),
            "chapter_headers": re.compile(
                r"\A[ \t]*Chapter \d+[ \t]*[:\-—]?[ \t]*([^\r\n]*)(?:\r?\n|$)",
                flags=re.IGNORECASE,
            ),
        }

        phrase_pattern_strings = [
            r"^\s*(Okay,\s*)?(Sure,\s*)?(Here's|Here is)\s+(the|your)\s+[\w\s]+?:\s*",
            r"^\s*I've written the\s+[\w\s]+?\s+as requested:\s*",
            r"^\s*Certainly! Here is the text:\s*",
            r"^\s*(?:Output|Result|Response|Answer)\s*:\s*",
            r"^\s*\[SYSTEM OUTPUT\]\s*",
            r"^\s*USER:\s*.*?ASSISTANT:\s*",
            r"\s*Let me know if you (need|have) any(thing else| other questions| further revisions| adjustments)\b.*?\.?[^\w\n]*$",
            r"\s*I hope this (meets your expectations|helps|is what you were looking for)\b.*?\.?[^\w\n]*$",
            r"\s*Feel free to ask for (adjustments|anything else)\b.*?\.?[^\w\n]*$",
            r"\s*Is there anything else I can help you with\b.*?(\?|.)[^\w\n]*$",
            r"\s*\[END SYSTEM OUTPUT\]\s*$",
        ]

        self._phrase_patterns = [re.compile(pattern_string, flags=re.IGNORECASE) for pattern_string in phrase_pattern_strings]

    def _remove_reasoning_prefix(self, text: str) -> str:
        """Consume balanced leading reasoning blocks or fail on incomplete markup."""
        text = text.strip()
        pattern = self._patterns["reasoning_tag"]
        if pattern.match(text) is None and not text.startswith(("{", "[")):
            boundary = self._patterns["think_boundary"].search(text)
            if boundary is not None:
                text = text[boundary.end():].lstrip()
        while (opening := pattern.match(text)) is not None:
            if opening.group(1):
                raise ValueError("Unexpected closing reasoning tag")
            stack: list[str] = []
            for tag in pattern.finditer(text):
                closing, name, self_closing = tag.groups()
                name = name.lower()
                if closing:
                    if self_closing or not stack or stack.pop() != name:
                        raise ValueError("Mismatched reasoning tags")
                elif not self_closing:
                    stack.append(name)
                if not stack:
                    text = text[tag.end():].lstrip()
                    break
            else:
                raise ValueError("Incomplete reasoning block")
        return text

    def clean_response(self, text: str) -> str:
        """Clean common artifacts from an LLM text response.

        Args:
            text: Raw model response.

        Returns:
            Cleaned response text.

        Notes:
            This removes:
            - Provider "think"/analysis style tags.
            - Markdown code fences while preserving fenced content.
            - Common lead-in and sign-off phrases.
        """
        if not isinstance(text, str):
            logger.warning(f"clean_response received non-string input: {type(text)}. Returning empty string.")
            return ""

        self._stats["responses_cleaned"] += 1
        original_length = len(text)
        cleaned_text = text.strip()

        # Removing an outer fence or lead-in can expose a reasoning prefix.
        # Each pass only consumes boundaries; answer interiors remain literal.
        while True:
            before_pass = cleaned_text
            cleaned_text = self._remove_reasoning_prefix(cleaned_text)
            if cleaned_text != before_pass:
                self._stats["think_tags_removed"] += 1

            if self._patterns["code_blocks"].search(cleaned_text):
                self._stats["code_blocks_cleaned"] += 1
            cleaned_text = self._patterns["code_blocks"].sub(r"\1", cleaned_text).strip()
            cleaned_text = self._patterns["chapter_headers"].sub("\\1\n", cleaned_text).strip()

            for pattern in self._phrase_patterns:
                original_text = cleaned_text
                cleaned_text = pattern.sub("", cleaned_text, count=1).strip()
                if cleaned_text != original_text:
                    self._stats["phrases_removed"] += 1

            if cleaned_text == before_pass:
                break

        # Final normalization
        final_text = cleaned_text
        # Track significant reductions
        if original_length > 0 and len(final_text) < original_length:
            reduction_percentage = ((original_length - len(final_text)) / original_length) * 100
            if reduction_percentage > 0.5:
                self._stats["significant_reductions"] += 1
                logger.debug(f"Cleaning reduced text length from {original_length} to {len(final_text)} " f"({reduction_percentage:.1f}% reduction).")

        return final_text

    def get_statistics(self) -> dict[str, Any]:
        """Get response cleaning service statistics."""
        total_cleaned = self._stats["responses_cleaned"]
        return {
            **self._stats,
            "think_removal_rate": (self._stats["think_tags_removed"] / total_cleaned * 100) if total_cleaned > 0 else 0,
            "code_cleaning_rate": (self._stats["code_blocks_cleaned"] / total_cleaned * 100) if total_cleaned > 0 else 0,
            "phrase_removal_rate": (self._stats["phrases_removed"] / total_cleaned * 100) if total_cleaned > 0 else 0,
            "significant_reduction_rate": (self._stats["significant_reductions"] / total_cleaned * 100) if total_cleaned > 0 else 0,
        }


# Streaming processing removed; only non-streaming responses are supported.


class TextProcessingService:
    """Coordinate tokenization and response cleaning utilities."""

    def __init__(self) -> None:
        """Initialize the text processing service with all sub-services."""
        self.tokenizer = TokenizerService()
        self.response_cleaner = ResponseCleaningService()
        self.spacy_service = get_spacy_service()

        logger.info("TextProcessingService initialized with all sub-services")

    def load_spacy_model(self, model_name: str | None = None) -> bool:
        """Load the spaCy model for NLP operations.

        Args:
            model_name: Optional model name override. If None, uses config.SPACY_MODEL or defaults.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        return self.spacy_service.load_model(model_name)

    def clean_text_with_spacy(self, text: str, aggressive: bool = False) -> str:
        """Clean text using spaCy NLP processing.

        Uses spaCy for advanced text cleaning including whitespace normalization,
        punctuation handling, and optional stop word removal.

        Args:
            text: Input text to clean.
            aggressive: If True, remove stop words and lemmatize.
                      If False, only normalize whitespace and basic punctuation.

        Returns:
            Cleaned text. Falls back to regex-based cleaning if spaCy not available.
        """
        return self.spacy_service.clean_text(text, aggressive)

    def extract_sentences_with_spacy(self, text: str) -> list[str]:
        """Extract sentences using spaCy's sentence boundary detection.

        Args:
            text: Input text to process.

        Returns:
            List of sentences. Falls back to regex-based splitting if spaCy not available.
        """
        return self.spacy_service.extract_sentences(text)

    def get_combined_statistics(self) -> dict[str, Any]:
        """Return combined statistics for tokenization and response cleaning."""
        return {
            "tokenizer": self.tokenizer.get_statistics(),
            "response_cleaner": self.response_cleaner.get_statistics(),
            "spacy_service": {
                "model_loaded": self.spacy_service.is_loaded(),
                "model_name": self.spacy_service.get_model_name(),
            },
        }


# Module-level convenience functions for backward compatibility
_default_tokenizer = TokenizerService()


def count_tokens(text: str, model_name: str) -> int:
    """Count tokens using the module-default tokenizer service."""
    return _default_tokenizer.count_tokens(text, model_name)


def truncate_text_by_tokens(
    text: str,
    model_name: str,
    max_tokens: int,
    truncation_marker: str = "\n... (truncated)",
) -> str:
    """Truncate text to a token budget using the module-default tokenizer service."""
    return _default_tokenizer.truncate_text_by_tokens(text, model_name, max_tokens, truncation_marker)


def clean_text_with_spacy(text: str, aggressive: bool = False) -> str:
    """Clean text using spaCy NLP processing (module-level convenience function).

    Args:
        text: Input text to clean.
        aggressive: If True, remove stop words and lemmatize.

    Returns:
        Cleaned text. Falls back to regex-based cleaning if spaCy not available.
    """
    return get_spacy_service().clean_text(text, aggressive)


def extract_sentences_with_spacy(text: str) -> list[str]:
    """Extract sentences using spaCy's sentence boundary detection (module-level convenience function).

    Args:
        text: Input text to process.

    Returns:
        List of sentences. Falls back to regex-based splitting if spaCy not available.
    """
    return get_spacy_service().extract_sentences(text)
