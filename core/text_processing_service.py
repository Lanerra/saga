# core/text_processing_service.py
"""
Text processing service for LLM content manipulation.

This module provides text processing functionality including tokenization,
truncation, and response cleaning, extracted from the monolithic LLMService
to improve separation of concerns and maintainability.

REFACTORED: Extracted from core.llm_interface as part of Phase 3 architectural improvements.
- Focuses solely on text processing concerns
- Token counting and text truncation
- Response cleaning and normalization
"""

import functools
import re
from collections.abc import Generator
from typing import Any

import structlog
import tiktoken

import config

logger = structlog.get_logger(__name__)


class TokenizerService:
    """
    Service for handling tokenization operations.

    Provides cached tokenizer access and token-related utilities
    with proper fallback mechanisms.
    """

    def __init__(self):
        """Initialize the tokenizer service."""
        self._tokenizer_cache: dict[str, tiktoken.Encoding] = {}
        self._stats = {
            "tokenizer_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "fallback_used": 0,
        }

    @functools.lru_cache(maxsize=config.TOKENIZER_CACHE_SIZE)
    def get_tokenizer(self, model_name: str) -> tiktoken.Encoding | None:
        """
        Get a tiktoken encoder for the given model name, with caching.

        Args:
            model_name: Name of the model to get tokenizer for

        Returns:
            Tokenizer encoding or None if unavailable
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
                logger.debug(
                    f"No direct tiktoken encoding for '{model_name}'. "
                    f"Using default '{config.TIKTOKEN_DEFAULT_ENCODING}'."
                )
                encoder = tiktoken.get_encoding(config.TIKTOKEN_DEFAULT_ENCODING)

            self._tokenizer_cache[model_name] = encoder
            logger.debug(
                f"Tokenizer for model '{model_name}' (using actual encoder '{encoder.name}') found and cached."
            )
            return encoder

        except KeyError:
            logger.error(
                f"Default tiktoken encoding '{config.TIKTOKEN_DEFAULT_ENCODING}' also not found. "
                f"Token counting will fall back to character-based heuristic for '{model_name}'."
            )
            return None

        except Exception as e:
            logger.error(
                f"Unexpected error getting tokenizer for '{model_name}': {e}",
                exc_info=True,
            )
            return None

    def count_tokens(self, text: str, model_name: str) -> int:
        """
        Count the number of tokens in a string for a given model.

        Args:
            text: Text to count tokens for
            model_name: Model to use for tokenization

        Returns:
            Number of tokens in the text
        """
        if not text:
            return 0

        encoder = self.get_tokenizer(model_name)

        if encoder:
            return len(encoder.encode(text, allowed_special="all"))
        else:
            # Fallback to character-based estimation
            self._stats["fallback_used"] += 1
            char_count = len(text)
            token_estimate = int(char_count / config.FALLBACK_CHARS_PER_TOKEN)
            logger.warning(
                f"count_tokens: Failed to get tokenizer for '{model_name}'. "
                f"Falling back to character-based estimate: {char_count} chars -> ~{token_estimate} tokens."
            )
            return token_estimate

    def truncate_text_by_tokens(
        self,
        text: str,
        model_name: str,
        max_tokens: int,
        truncation_marker: str = "\n... (truncated)",
    ) -> str:
        """
        Truncate text to a maximum number of tokens for a given model.

        Args:
            text: Text to truncate
            model_name: Model to use for tokenization
            max_tokens: Maximum number of tokens to keep
            truncation_marker: Marker to add when text is truncated

        Returns:
            Truncated text with marker if applicable
        """
        if not text:
            return ""

        encoder = self.get_tokenizer(model_name)

        if not encoder:
            # Fallback to character-based truncation
            self._stats["fallback_used"] += 1
            max_chars = int(max_tokens * config.FALLBACK_CHARS_PER_TOKEN)
            logger.warning(
                f"truncate_text_by_tokens: Failed to get tokenizer for '{model_name}'. "
                f"Falling back to character-based truncation: {max_tokens} tokens -> ~{max_chars} chars."
            )
            if len(text) > max_chars:
                effective_max_chars = max_chars - len(truncation_marker)
                if effective_max_chars < 0:
                    effective_max_chars = 0
                return text[:effective_max_chars] + truncation_marker
            return text

        tokens = encoder.encode(text, allowed_special="all")
        if len(tokens) <= max_tokens:
            return text

        # Calculate tokens needed for truncation marker
        marker_tokens_len = 0
        if truncation_marker:
            marker_tokens_len = len(
                encoder.encode(truncation_marker, allowed_special="all")
            )

        content_tokens_to_keep = max_tokens - marker_tokens_len
        effective_truncation_marker = truncation_marker

        if content_tokens_to_keep < 0:
            logger.debug(
                f"Truncation marker ('{truncation_marker}' -> {marker_tokens_len} tokens) "
                f"is longer than max_tokens ({max_tokens}). Using empty marker."
            )
            content_tokens_to_keep = max_tokens
            effective_truncation_marker = ""

        truncated_content_tokens = tokens[:content_tokens_to_keep]

        # Ensure we keep at least one token if possible
        if not truncated_content_tokens and max_tokens > 0 and tokens:
            logger.debug(
                "Truncated content to 0 tokens due to marker length. "
                "Attempting to keep 1 token of content."
            )
            truncated_content_tokens = tokens[:1]
            effective_truncation_marker = ""

        try:
            decoded_text = encoder.decode(truncated_content_tokens)
            return decoded_text + effective_truncation_marker
        except Exception as e:
            logger.error(
                f"Error decoding truncated tokens for model '{model_name}': {e}. "
                f"Falling back to simpler char-based truncation.",
                exc_info=True,
            )
            # Fallback to character-based truncation
            avg_chars_per_token = (
                len(text) / len(tokens)
                if len(tokens) > 0
                else config.FALLBACK_CHARS_PER_TOKEN
            )
            estimated_char_limit_for_content = int(
                content_tokens_to_keep * avg_chars_per_token
            )
            return text[:estimated_char_limit_for_content] + effective_truncation_marker

    def get_statistics(self) -> dict[str, Any]:
        """Get tokenizer service statistics."""
        total_requests = self._stats["tokenizer_requests"]
        return {
            **self._stats,
            "cache_hit_rate": (self._stats["cache_hits"] / total_requests * 100)
            if total_requests > 0
            else 0,
            "fallback_rate": (self._stats["fallback_used"] / total_requests * 100)
            if total_requests > 0
            else 0,
        }


class ResponseCleaningService:
    """
    Service for cleaning and normalizing LLM responses.

    Handles removal of common artifacts, think tags, and other
    unwanted content from LLM outputs.
    """

    def __init__(self):
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

        self._compiled_patterns = self._compile_cleaning_patterns()

    def _compile_cleaning_patterns(self) -> dict[str, list[re.Pattern]]:
        """Pre-compile regex patterns for better performance."""
        patterns = {
            "think_blocks": [],
            "think_self_closing": [],
            "think_opening": [],
            "think_closing": [],
            "code_blocks": [],
            "chapter_headers": [],
            "common_phrases": [],
        }

        # Think tag patterns
        for tag_name in self._think_tags:
            patterns["think_blocks"].append(
                re.compile(
                    rf"<\s*{tag_name}\s*>.*?<\s*/\s*{tag_name}\s*>",
                    flags=re.DOTALL | re.IGNORECASE,
                )
            )
            patterns["think_self_closing"].append(
                re.compile(rf"<\s*{tag_name}\s*/\s*>", flags=re.IGNORECASE)
            )
            patterns["think_opening"].append(
                re.compile(rf"<\s*{tag_name}\s*>", flags=re.IGNORECASE)
            )
            patterns["think_closing"].append(
                re.compile(rf"<\s*/\s*{tag_name}\s*>", flags=re.IGNORECASE)
            )

        # Code blocks
        patterns["code_blocks"].append(
            re.compile(
                r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)\s*```",
                flags=re.DOTALL,
            )
        )

        # Chapter headers
        patterns["chapter_headers"].append(
            re.compile(
                r"^\s*Chapter \d+\s*[:\-—]?\s*(.*?)\s*$",
                flags=re.MULTILINE | re.IGNORECASE,
            )
        )

        # Common phrase patterns
        phrase_patterns = [
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

        for pattern_str in phrase_patterns:
            patterns["common_phrases"].append(
                re.compile(pattern_str, flags=re.IGNORECASE | re.MULTILINE)
            )

        return patterns

    def clean_response(self, text: str) -> str:
        """
        Clean common artifacts from LLM text responses.

        Args:
            text: Raw LLM response text

        Returns:
            Cleaned and normalized text
        """
        if not isinstance(text, str):
            logger.warning(
                f"clean_response received non-string input: {type(text)}. Returning empty string."
            )
            return ""

        self._stats["responses_cleaned"] += 1
        original_length = len(text)
        cleaned_text = text

        # Remove think tags and similar content
        text_before_think_removal = cleaned_text
        for pattern_list in [
            self._compiled_patterns["think_blocks"],
            self._compiled_patterns["think_self_closing"],
            self._compiled_patterns["think_opening"],
            self._compiled_patterns["think_closing"],
        ]:
            for pattern in pattern_list:
                cleaned_text = pattern.sub("", cleaned_text)

        if len(cleaned_text) < len(text_before_think_removal):
            self._stats["think_tags_removed"] += 1
            logger.debug(
                f"clean_response: Removed think tag content. "
                f"Length before: {len(text_before_think_removal)}, after: {len(cleaned_text)}."
            )

        # Remove code blocks
        for pattern in self._compiled_patterns["code_blocks"]:
            if pattern.search(cleaned_text):
                self._stats["code_blocks_cleaned"] += 1
            cleaned_text = pattern.sub(r"\1", cleaned_text)

        # Remove chapter headers
        for pattern in self._compiled_patterns["chapter_headers"]:
            cleaned_text = pattern.sub(r"\1", cleaned_text).strip()

        # Remove common phrases
        for pattern in self._compiled_patterns["common_phrases"]:
            original_text = cleaned_text
            if pattern.pattern.startswith("^"):
                # Apply repeatedly for patterns that should be removed from start
                while True:
                    new_text = pattern.sub("", cleaned_text, count=1).strip()
                    if new_text == cleaned_text:
                        break
                    cleaned_text = new_text
            else:
                cleaned_text = pattern.sub("", cleaned_text, count=1).strip()

            if len(cleaned_text) < len(original_text):
                self._stats["phrases_removed"] += 1

        # Final normalization
        final_text = cleaned_text.strip()

        # Normalize multiple newlines
        final_text = re.sub(r"\n\s*\n(\s*\n)+", "\n\n", final_text)
        final_text = re.sub(r"\n{3,}", "\n\n", final_text)

        # Track significant reductions
        if original_length > 0 and len(final_text) < original_length:
            reduction_percentage = (
                (original_length - len(final_text)) / original_length
            ) * 100
            if reduction_percentage > 0.5:
                self._stats["significant_reductions"] += 1
                logger.debug(
                    f"Cleaning reduced text length from {original_length} to {len(final_text)} "
                    f"({reduction_percentage:.1f}% reduction)."
                )

        return final_text

    def get_statistics(self) -> dict[str, Any]:
        """Get response cleaning service statistics."""
        total_cleaned = self._stats["responses_cleaned"]
        return {
            **self._stats,
            "think_removal_rate": (
                self._stats["think_tags_removed"] / total_cleaned * 100
            )
            if total_cleaned > 0
            else 0,
            "code_cleaning_rate": (
                self._stats["code_blocks_cleaned"] / total_cleaned * 100
            )
            if total_cleaned > 0
            else 0,
            "phrase_removal_rate": (
                self._stats["phrases_removed"] / total_cleaned * 100
            )
            if total_cleaned > 0
            else 0,
            "significant_reduction_rate": (
                self._stats["significant_reductions"] / total_cleaned * 100
            )
            if total_cleaned > 0
            else 0,
        }


# Streaming processing removed; only non-streaming responses are supported.


class TextProcessingService:
    """
    Main text processing service that coordinates all text-related operations.

    This service provides a unified interface for all text processing needs,
    combining tokenization and cleaning functionality.
    """

    def __init__(self):
        """Initialize the text processing service with all sub-services."""
        self.tokenizer = TokenizerService()
        self.response_cleaner = ResponseCleaningService()

        logger.info("TextProcessingService initialized with all sub-services")

    def get_combined_statistics(self) -> dict[str, Any]:
        """Get combined statistics from all sub-services."""
        return {
            "tokenizer": self.tokenizer.get_statistics(),
            "response_cleaner": self.response_cleaner.get_statistics(),
        }


# Module-level convenience functions for backward compatibility
_default_tokenizer = TokenizerService()


def count_tokens(text: str, model_name: str) -> int:
    """Convenience function for token counting using default service."""
    return _default_tokenizer.count_tokens(text, model_name)


def truncate_text_by_tokens(
    text: str,
    model_name: str,
    max_tokens: int,
    truncation_marker: str = "\n... (truncated)",
) -> str:
    """Convenience function for text truncation using default service."""
    return _default_tokenizer.truncate_text_by_tokens(
        text, model_name, max_tokens, truncation_marker
    )
