# core/llm_interface_refactored.py
"""
Refactored LLM interface with separated concerns.

This module provides the new LLM service architecture using direct instantiation
and separated concerns for HTTP communication and text processing.

Licensed under the Apache License, Version 2.0
"""

import asyncio
import hashlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

import numpy as np
import structlog

import config
from core.http_client_service import (
    CompletionHTTPClient,
    EmbeddingHTTPClient,
    HTTPClientService,
)
from core.lightweight_cache import (
    get_cached_value,
    register_cache_service,
    set_cached_value,
)
from core.text_processing_service import TextProcessingService

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def async_llm_context(
    batch_size: int | None = None,
    clear_cache_on_exit: bool = False,
) -> AsyncGenerator[tuple["RefactoredLLMService", "EmbeddingService"], None]:
    """
    Unified async context manager for LLM operations with guaranteed cleanup.

    This single context manager replaces the previous nested context managers:
    - async_llm_service
    - async_batch_embedding_session
    - async_managed_cache_session

    Args:
        batch_size: Size of batches to process (for embedding operations)
        clear_cache_on_exit: Whether to clear cache when exiting context

    Usage:
        async with async_llm_context(batch_size=8, clear_cache_on_exit=True) as (llm_service, embedding_service):
            # Use both services
            result = await llm_service.async_call_llm(model_name, prompt)
            embeddings = await embedding_service.get_embeddings_batch(texts, batch_size)
    """
    # Direct instantiation instead of service locator
    http_client = HTTPClientService()
    embedding_client = EmbeddingHTTPClient(http_client)
    completion_client = CompletionHTTPClient(http_client)
    text_processor = TextProcessingService()

    embedding_service = EmbeddingService(embedding_client)
    completion_service = CompletionService(completion_client, text_processor)
    llm_service = RefactoredLLMService(completion_service, embedding_service, text_processor)

    # Cache size tracking not directly available via service attribute
    initial_cache_size = 0

    try:
        yield llm_service, embedding_service
    finally:
        # Cleanup
        try:
            await http_client.aclose()
            logger.debug("HTTP client closed successfully")
        except Exception as cleanup_error:
            logger.error(f"Failed to cleanup HTTP client: {cleanup_error}")

        # Handle cache management
        if clear_cache_on_exit:
            from core.lightweight_cache import clear_service_cache

            clear_service_cache("llm_embedding")
            logger.debug("Cleared llm_embedding cache on exit")

        # Log performance metrics
        try:
            stats = embedding_service.get_statistics()
            cache_growth = stats.get("cache_size", 0) - initial_cache_size
            logger.debug(
                f"LLM session completed: "
                f"{stats.get('cache_hit_rate', 0):.1f}% cache hit rate, "
                f"grew by {cache_growth} entries, "
                f"{stats.get('cache_size', 0)}/{stats.get('cache_max_size', 0)} cache entries"
            )
        except Exception as stats_error:
            logger.error(f"Failed to log session statistics: {stats_error}")


class EmbeddingService:
    """
    Service for generating embeddings using the Ollama API.

    REFACTORED: Simplified to focus only on embedding logic,
    delegating HTTP concerns to HTTPClientService.
    """

    def __init__(self, embedding_client: EmbeddingHTTPClient):
        """
        Initialize embedding service.

        Args:
            embedding_client: HTTP client for embedding requests
        """
        self._embedding_client = embedding_client
        self._service_name = "llm_embedding"
        # Register with cache coordinator
        register_cache_service(self._service_name)
        self._stats = {
            "embeddings_requested": 0,
            "embeddings_successful": 0,
            "embeddings_failed": 0,
            "validation_failures": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def _compute_text_hash(self, text: str) -> str:
        """Compute a hash of the text for caching purposes."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    async def get_embedding(self, text: str) -> np.ndarray | None:
        """
        Get embedding vector for text with coordinated caching.

        Args:
            text: Text to get embedding for

        Returns:
            Embedding vector as numpy array or None if failed
        """
        self._stats["embeddings_requested"] += 1

        if not text or not isinstance(text, str) or not text.strip():
            logger.warning(f"get_embedding: empty or invalid text provided. Text repr: {repr(text)}")
            self._stats["embeddings_failed"] += 1
            return None

        # Check coordinated cache first
        text_hash = self._compute_text_hash(text.strip())
        cached_embedding = get_cached_value(text_hash, self._service_name)
        if cached_embedding is not None:
            self._stats["cache_hits"] += 1
            logger.debug(f"Cache hit for embedding (hash: {text_hash[:8]})")
            return cached_embedding

        self._stats["cache_misses"] += 1

        try:
            response_data = await self._embedding_client.get_embedding(text, config.EMBEDDING_MODEL)

            # Extract and validate embedding
            embedding = self._extract_and_validate_embedding(response_data)
            if embedding is not None:
                # Cache the successful embedding
                set_cached_value(text_hash, embedding, self._service_name)
                self._stats["embeddings_successful"] += 1
                return embedding
            else:
                self._stats["validation_failures"] += 1
                return None

        except Exception as e:
            logger.error(f"Failed to get embedding: {e}", exc_info=True)
            self._stats["embeddings_failed"] += 1
            return None

    async def get_embeddings_batch(self, texts: list[str], batch_size: int | None = None) -> list[np.ndarray | None]:
        """
        Get embeddings for multiple texts in batches for better performance.

        Args:
            texts: List of texts to get embeddings for
            batch_size: Size of batches to process (defaults to MAX_CONCURRENT_LLM_CALLS)

        Returns:
            List of embedding vectors (same order as input texts)
        """
        if not texts:
            return []

        batch_size = batch_size or config.MAX_CONCURRENT_LLM_CALLS
        results: list[np.ndarray | None] = [None] * len(texts)

        # Process in batches to control concurrency and memory usage
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_tasks = [self.get_embedding(text) for text in batch_texts]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for j, result in enumerate(batch_results):
                if not isinstance(result, Exception):
                    results[i + j] = cast(np.ndarray | None, result)

        return results

    def _extract_and_validate_embedding(self, response_data: dict[str, Any]) -> np.ndarray | None:
        """Extract and validate embedding from API response."""
        # Try primary key first
        primary_key = "embedding"
        if primary_key in response_data and isinstance(response_data[primary_key], list):
            embedding = self._validate_embedding_list(response_data[primary_key])
            if embedding is not None:
                return embedding

        # Try fallback keys
        logger.warning(f"Primary embedding key '{primary_key}' not found, trying fallbacks")
        for key, value in response_data.items():
            if isinstance(value, list) and all(isinstance(item, float | int) for item in value):
                embedding = self._validate_embedding_list(value)
                if embedding is not None:
                    logger.info(f"Found embedding using fallback key '{key}'")
                    return embedding

        logger.error(f"No suitable embedding found in response: {response_data}")
        return None

    def _validate_embedding_list(self, embedding_list: list[float | int]) -> np.ndarray | None:
        """Validate and convert embedding list to numpy array."""
        try:
            embedding = np.array(embedding_list).astype(config.EMBEDDING_DTYPE)
            if embedding.ndim > 1:
                logger.warning(f"Embedding had unexpected ndim > 1: {embedding.ndim}. Flattening.")
                embedding = embedding.flatten()

            if embedding.shape == (config.EXPECTED_EMBEDDING_DIM,):
                logger.debug(f"Embedding validated: shape={embedding.shape}, dtype={embedding.dtype}")
                return embedding

            logger.error(f"Embedding dimension mismatch: Expected ({config.EXPECTED_EMBEDDING_DIM},), " f"Got {embedding.shape}. List length: {len(embedding_list)}")

        except (TypeError, ValueError) as e:
            logger.error(f"Failed to convert embedding list to numpy array: {e}")

        return None

    def get_statistics(self) -> dict[str, Any]:
        """Get embedding service statistics."""
        total = self._stats["embeddings_requested"]
        # Get cache size from lightweight cache service
        from core.lightweight_cache import get_cache_size

        cache_size = get_cache_size(self._service_name)

        return {
            **self._stats,
            "cache_size": cache_size,
            "cache_hit_rate": (self._stats["cache_hits"] / total * 100) if total > 0 else 0,
            "cache_miss_rate": (self._stats["cache_misses"] / total * 100) if total > 0 else 0,
            "success_rate": (self._stats["embeddings_successful"] / total * 100) if total > 0 else 0,
            "failure_rate": (self._stats["embeddings_failed"] / total * 100) if total > 0 else 0,
            "validation_failure_rate": (self._stats["validation_failures"] / total * 100) if total > 0 else 0,
        }


class CompletionService:
    """
    Service for generating text completions using OpenAI-compatible APIs.

    REFACTORED: Simplified to focus only on completion logic,
    delegating HTTP and text processing concerns to specialized services.
    """

    def __init__(
        self,
        completion_client: CompletionHTTPClient,
        text_processor: TextProcessingService,
    ):
        """
        Initialize completion service.

        Args:
            completion_client: HTTP client for completion requests
            text_processor: Service for text processing operations
        """
        self._completion_client = completion_client
        self._text_processor = text_processor
        self._stats = {
            "completions_requested": 0,
            "completions_successful": 0,
            "completions_failed": 0,
            "fallback_used": 0,
            "streaming_requests": 0,
        }

    async def get_completion(
        self,
        model_name: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        allow_fallback: bool = False,
        auto_clean_response: bool = True,
        grammar: str | None = None,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, int] | None]:
        """
        Get completion from LLM.

        Args:
            model_name: Model to use for completion
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            allow_fallback: Whether to allow fallback model
            auto_clean_response: Whether to clean the response
            **kwargs: Additional completion parameters

        Returns:
            Tuple of (response_text, usage_data)
        """
        self._stats["completions_requested"] += 1

        if not model_name or not prompt:
            logger.error("get_completion: model_name and prompt are required")
            self._stats["completions_failed"] += 1
            return "", None

        effective_temperature = temperature if temperature is not None else config.Temperatures.DEFAULT
        effective_max_tokens = max_tokens if max_tokens is not None else config.MAX_GENERATION_TOKENS

        # Build messages with optional system prompt
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": prompt}]

        # Try primary model
        try:
            response_data = await self._completion_client.get_completion(
                model_name,
                messages,
                effective_temperature,
                effective_max_tokens,
                grammar=grammar,
                **kwargs,
            )

            content = self._extract_completion_content(response_data)
            usage_data = response_data.get("usage")

            if auto_clean_response:
                content = self._text_processor.response_cleaner.clean_response(content)

            self._stats["completions_successful"] += 1
            return content, usage_data

        except Exception as e:
            logger.error(f"Primary model '{model_name}' failed: {e}")

            # Try fallback if enabled
            if allow_fallback and config.MEDIUM_MODEL:
                logger.info(f"Attempting fallback with '{config.MEDIUM_MODEL}'")
                self._stats["fallback_used"] += 1

                try:
                    response_data = await self._completion_client.get_completion(
                        config.MEDIUM_MODEL,
                        messages,
                        effective_temperature,
                        effective_max_tokens,
                        grammar=grammar,
                        **kwargs,
                    )

                    content = self._extract_completion_content(response_data)
                    usage_data = response_data.get("usage")

                    if auto_clean_response:
                        content = self._text_processor.response_cleaner.clean_response(content)

                    self._stats["completions_successful"] += 1
                    return content, usage_data

                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")

            self._stats["completions_failed"] += 1
            return "", None

    # Streaming completion path removed to simplify the API.

    def _extract_completion_content(self, response_data: dict[str, Any]) -> str:
        """Extract completion content from API response."""
        # Prefer the standard OpenAI schema first
        try:
            if response_data.get("choices") and len(response_data["choices"]) > 0:
                choice0 = response_data["choices"][0]
                message = choice0.get("message") or {}

                # 1) Standard content
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content

                # 2) Some providers (e.g., Qwen reasoning models) expose 'reasoning_content'
                reasoning_content = message.get("reasoning_content")
                if isinstance(reasoning_content, str) and reasoning_content.strip():
                    logger.warning("LLM response missing 'content'; using 'reasoning_content' fallback")
                    return reasoning_content

                # 3) Occasionally providers place content directly under the choice
                direct_choice_content = choice0.get("content")
                if isinstance(direct_choice_content, str) and direct_choice_content.strip():
                    logger.warning("LLM response missing message.content; using choice['content'] fallback")
                    return direct_choice_content

                # 4) Last resorts: top-level convenience fields sometimes appear
                for key in ("output_text", "text", "response", "content"):
                    top = response_data.get(key)
                    if isinstance(top, str) and top.strip():
                        logger.warning(f"LLM response using top-level '{key}' fallback for content")
                        return top

        except Exception as e:
            logger.error(f"Completion content extraction failed: {e}", exc_info=True)

        logger.error(f"Invalid response structure - missing choices/content: {response_data}")
        return ""

    def get_statistics(self) -> dict[str, Any]:
        """Get completion service statistics."""
        total = self._stats["completions_requested"]
        return {
            **self._stats,
            "success_rate": (self._stats["completions_successful"] / total * 100) if total > 0 else 0,
            "failure_rate": (self._stats["completions_failed"] / total * 100) if total > 0 else 0,
            "fallback_rate": (self._stats["fallback_used"] / total * 100) if total > 0 else 0,
            "streaming_rate": (self._stats["streaming_requests"] / total * 100) if total > 0 else 0,
        }


class RefactoredLLMService:
    """
    Main LLM service with separated concerns architecture.

    REFACTORED: Complete rewrite using direct instantiation and separated services.
    - HTTP communication handled by HTTPClientService
    - Text processing handled by TextProcessingService
    - Clear separation of concerns
    - Better testability and maintainability
    """

    def __init__(
        self,
        completion_service: CompletionService,
        embedding_service: EmbeddingService,
        text_processor: TextProcessingService,
    ):
        """
        Initialize the refactored LLM service.

        Args:
            completion_service: Service for text completions
            embedding_service: Service for embeddings
            text_processor: Service for text processing
        """
        self._completion_service = completion_service
        self._embedding_service = embedding_service
        self._text_processor = text_processor

        logger.info("RefactoredLLMService initialized with separated components")

    async def async_call_llm(
        self,
        model_name: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        allow_fallback: bool = False,
        stream_to_disk: bool = False,
        auto_clean_response: bool = True,
        grammar: str | None = None,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, int] | None]:
        """
        Call LLM with comprehensive options.

        REFACTORED: Simplified orchestration, delegating to specialized services.
        """
        return await self._completion_service.get_completion(
            model_name,
            prompt,
            temperature,
            max_tokens,
            allow_fallback,
            auto_clean_response,
            grammar=grammar,
            system_prompt=system_prompt,
            **kwargs,
        )

    async def async_get_embedding(self, text: str) -> np.ndarray | None:
        """
        Get embedding for text.

        REFACTORED: Simple delegation to embedding service.
        """
        return await self._embedding_service.get_embedding(text)

    async def async_get_embeddings_batch(self, texts: list[str], batch_size: int | None = None) -> list[np.ndarray | None]:
        """
        Get embeddings for multiple texts in batches.

        REFACTORED: Simple delegation to embedding service.
        """
        return await self._embedding_service.get_embeddings_batch(texts, batch_size)

    def count_tokens(self, text: str, model_name: str) -> int:
        """
        Count tokens in text.

        REFACTORED: Simple delegation to text processing service.
        """
        return self._text_processor.tokenizer.count_tokens(text, model_name)

    def truncate_text_by_tokens(
        self,
        text: str,
        model_name: str,
        max_tokens: int,
        truncation_marker: str = "\n... (truncated)",
    ) -> str:
        """
        Truncate text by tokens.

        REFACTORED: Simple delegation to text processing service.
        """
        return self._text_processor.tokenizer.truncate_text_by_tokens(text, model_name, max_tokens, truncation_marker)

    def get_combined_statistics(self) -> dict[str, Any]:
        """Get combined statistics from all services."""
        return {
            "completion_service": self._completion_service.get_statistics(),
            "embedding_service": self._embedding_service.get_statistics(),
            "text_processor": self._text_processor.get_combined_statistics(),
        }


# Direct instantiation functions for simplified API
def create_llm_service() -> RefactoredLLMService:
    """
    Create and return a new LLM service instance with direct instantiation.

    This replaces the service locator pattern with direct dependency injection.
    """
    http_client = HTTPClientService()
    embedding_client = EmbeddingHTTPClient(http_client)
    completion_client = CompletionHTTPClient(http_client)
    text_processor = TextProcessingService()

    embedding_service = EmbeddingService(embedding_client)
    completion_service = CompletionService(completion_client, text_processor)
    return RefactoredLLMService(completion_service, embedding_service, text_processor)


# Module-level service instance
llm_service = create_llm_service()
