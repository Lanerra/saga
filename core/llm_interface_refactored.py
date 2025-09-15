# core/llm_interface_refactored.py
"""
Refactored LLM interface with separated concerns.

This module provides the new LLM service architecture using dependency injection
and separated concerns for HTTP communication and text processing.

REFACTORED: Complete rewrite as part of Phase 3 architectural improvements.
- Separated HTTP client and text processing concerns
- Proper abstraction layers with protocols
- Dependency injection for better testability
- Clean service composition

Licensed under the Apache License, Version 2.0
"""

import asyncio
import hashlib
import os
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

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
from core.llm_service_interfaces import LLMServiceFactory, initialize_service_locator
from core.text_processing_service import TextProcessingService

logger = structlog.get_logger(__name__)


@contextmanager
def secure_temp_file(suffix: str = ".tmp", text: bool = True):
    """
    Context manager for secure temporary file handling.
    Guarantees cleanup even if exceptions occur.
    """
    temp_fd = None
    temp_path = None
    try:
        temp_fd, temp_path = tempfile.mkstemp(suffix=suffix, text=text)
        os.close(temp_fd)  # Close the file descriptor immediately
        yield temp_path
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"Cleaned up temporary file: {temp_path}")
            except Exception as cleanup_error:
                logger.error(
                    f"Failed to cleanup temporary file {temp_path}: {cleanup_error}"
                )


@asynccontextmanager
async def async_llm_service() -> AsyncGenerator["RefactoredLLMService", None]:
    """
    Async context manager for LLM service with guaranteed cleanup.

    Usage:
        async with async_llm_service() as service:
            result = await service.async_get_embedding("text")
    """
    service = get_llm_service()
    try:
        yield service
    finally:
        try:
            await service.aclose()
            logger.debug("LLM service closed successfully")
        except Exception as cleanup_error:
            logger.error(f"Failed to cleanup LLM service: {cleanup_error}")


@asynccontextmanager
async def async_batch_embedding_session(
    batch_size: int | None = None,
) -> AsyncGenerator["EmbeddingService", None]:
    """
    Async context manager for batch embedding operations with resource cleanup.

    Args:
        batch_size: Size of batches to process

    Usage:
        async with async_batch_embedding_session(batch_size=8) as embedding_service:
            embeddings = await embedding_service.get_embeddings_batch(texts)
    """
    service = get_llm_service()
    embedding_service = service._embedding_service

    # Note: Could store original cache state here for restoration if needed

    try:
        yield embedding_service
    finally:
        # Log performance metrics
        stats = embedding_service.get_statistics()
        logger.info(
            f"Batch embedding session completed: "
            f"{stats.get('cache_hit_rate', 0):.1f}% cache hit rate, "
            f"{stats.get('cache_size', 0)}/{stats.get('cache_max_size', 0)} cache entries"
        )


@asynccontextmanager
async def async_managed_cache_session(
    clear_on_exit: bool = False,
) -> AsyncGenerator["EmbeddingService", None]:
    """
    Async context manager for cache management during processing sessions.

    Args:
        clear_on_exit: Whether to clear cache when exiting context

    Usage:
        async with async_managed_cache_session(clear_on_exit=True) as embedding_service:
            # Perform operations that might benefit from fresh cache state
            embeddings = await embedding_service.get_embedding("text")
    """
    service = get_llm_service()
    embedding_service = service._embedding_service

    # Store initial cache state
    initial_cache_size = len(embedding_service._embedding_cache)

    try:
        yield embedding_service
    finally:
        final_stats = embedding_service.get_statistics()
        cache_growth = final_stats.get("cache_size", 0) - initial_cache_size

        logger.debug(
            f"Cache session: grew by {cache_growth} entries, "
            f"final hit rate: {final_stats.get('cache_hit_rate', 0):.1f}%"
        )

        if clear_on_exit:
            embedding_service._embedding_cache.clear()
            logger.debug("Cache cleared on session exit")


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
            logger.warning("get_embedding: empty or invalid text provided")
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
            response_data = await self._embedding_client.get_embedding(
                text, config.EMBEDDING_MODEL
            )

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

    async def get_embeddings_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[np.ndarray | None]:
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
        results = [None] * len(texts)

        # Process in batches to control concurrency and memory usage
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_tasks = [self.get_embedding(text) for text in batch_texts]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for j, result in enumerate(batch_results):
                if not isinstance(result, Exception):
                    results[i + j] = result

        return results

    def _extract_and_validate_embedding(
        self, response_data: dict[str, Any]
    ) -> np.ndarray | None:
        """Extract and validate embedding from API response."""
        # Try primary key first
        primary_key = "embedding"
        if primary_key in response_data and isinstance(
            response_data[primary_key], list
        ):
            embedding = self._validate_embedding_list(response_data[primary_key])
            if embedding is not None:
                return embedding

        # Try fallback keys
        logger.warning(
            f"Primary embedding key '{primary_key}' not found, trying fallbacks"
        )
        for key, value in response_data.items():
            if isinstance(value, list) and all(
                isinstance(item, float | int) for item in value
            ):
                embedding = self._validate_embedding_list(value)
                if embedding is not None:
                    logger.info(f"Found embedding using fallback key '{key}'")
                    return embedding

        logger.error(f"No suitable embedding found in response: {response_data}")
        return None

    def _validate_embedding_list(
        self, embedding_list: list[float | int]
    ) -> np.ndarray | None:
        """Validate and convert embedding list to numpy array."""
        try:
            embedding = np.array(embedding_list).astype(config.EMBEDDING_DTYPE)
            if embedding.ndim > 1:
                logger.warning(
                    f"Embedding had unexpected ndim > 1: {embedding.ndim}. Flattening."
                )
                embedding = embedding.flatten()

            if embedding.shape == (config.EXPECTED_EMBEDDING_DIM,):
                logger.debug(
                    f"Embedding validated: shape={embedding.shape}, dtype={embedding.dtype}"
                )
                return embedding

            logger.error(
                f"Embedding dimension mismatch: Expected ({config.EXPECTED_EMBEDDING_DIM},), "
                f"Got {embedding.shape}. List length: {len(embedding_list)}"
            )

        except (TypeError, ValueError) as e:
            logger.error(f"Failed to convert embedding list to numpy array: {e}")

        return None

    def get_statistics(self) -> dict[str, Any]:
        """Get embedding service statistics."""
        total = self._stats["embeddings_requested"]
        # Get cache metrics from coordinated cache
        from core.lightweight_cache import get_cache_metrics

        cache_metrics = get_cache_metrics(self._service_name)
        cache_size = 0
        if cache_metrics and self._service_name in cache_metrics:
            cache_size = cache_metrics[self._service_name].total_entries

        return {
            **self._stats,
            "cache_size": cache_size,
            "cache_hit_rate": (self._stats["cache_hits"] / total * 10)
            if total > 0
            else 0,
            "cache_miss_rate": (self._stats["cache_misses"] / total * 100)
            if total > 0
            else 0,
            "success_rate": (self._stats["embeddings_successful"] / total * 100)
            if total > 0
            else 0,
            "failure_rate": (self._stats["embeddings_failed"] / total * 100)
            if total > 0
            else 0,
            "validation_failure_rate": (
                self._stats["validation_failures"] / total * 100
            )
            if total > 0
            else 0,
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
        **kwargs,
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

        effective_temperature = (
            temperature if temperature is not None else config.Temperatures.DEFAULT
        )
        effective_max_tokens = (
            max_tokens if max_tokens is not None else config.MAX_GENERATION_TOKENS
        )

        messages = [{"role": "user", "content": prompt}]

        # Try primary model
        try:
            response_data = await self._completion_client.get_completion(
                model_name,
                messages,
                effective_temperature,
                effective_max_tokens,
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
            if allow_fallback and config.FALLBACK_GENERATION_MODEL:
                logger.info(
                    f"Attempting fallback with '{config.FALLBACK_GENERATION_MODEL}'"
                )
                self._stats["fallback_used"] += 1

                try:
                    response_data = await self._completion_client.get_completion(
                        config.FALLBACK_GENERATION_MODEL,
                        messages,
                        effective_temperature,
                        effective_max_tokens,
                        **kwargs,
                    )

                    content = self._extract_completion_content(response_data)
                    usage_data = response_data.get("usage")

                    if auto_clean_response:
                        content = self._text_processor.response_cleaner.clean_response(
                            content
                        )

                    self._stats["completions_successful"] += 1
                    return content, usage_data

                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")

            self._stats["completions_failed"] += 1
            return "", None

    async def get_streaming_completion(
        self,
        model_name: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        auto_clean_response: bool = True,
        **kwargs,
    ) -> tuple[str, dict[str, int] | None]:
        """
        Get streaming completion and write to disk for memory efficiency.

        Args:
            model_name: Model to use for completion
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            auto_clean_response: Whether to clean the response
            **kwargs: Additional completion parameters

        Returns:
            Tuple of (response_text, usage_data)
        """
        self._stats["streaming_requests"] += 1

        effective_temperature = (
            temperature if temperature is not None else config.Temperatures.DEFAULT
        )
        effective_max_tokens = (
            max_tokens if max_tokens is not None else config.MAX_GENERATION_TOKENS
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            with secure_temp_file(suffix=".llmstream.txt", text=True) as temp_file_path:
                accumulated_content = ""
                usage_data = None

                # Process streaming chunks and write to temp file
                async for (
                    chunk_data
                ) in self._completion_client.get_streaming_completion(
                    model_name,
                    messages,
                    effective_temperature,
                    effective_max_tokens,
                    **kwargs,
                ):
                    if chunk_data.get("choices"):
                        delta = chunk_data["choices"][0].get("delta", {})
                        content_piece = delta.get("content")

                        if content_piece:
                            accumulated_content += content_piece
                            # Write to temp file for large responses
                            with open(temp_file_path, "a", encoding="utf-8") as tmp_f:
                                tmp_f.write(content_piece)

                        # Check for completion and extract usage
                        if chunk_data["choices"][0].get("finish_reason") is not None:
                            potential_usage = chunk_data.get("usage")
                            if (
                                not potential_usage
                                and chunk_data.get("x_groq")
                                and chunk_data["x_groq"].get("usage")
                            ):
                                potential_usage = chunk_data["x_groq"]["usage"]
                            if potential_usage:
                                usage_data = potential_usage

                if auto_clean_response:
                    accumulated_content = (
                        self._text_processor.response_cleaner.clean_response(
                            accumulated_content
                        )
                    )

                self._stats["completions_successful"] += 1
                return accumulated_content, usage_data

        except Exception as e:
            logger.error(f"Streaming completion failed: {e}", exc_info=True)
            self._stats["completions_failed"] += 1
            return "", None

    def _extract_completion_content(self, response_data: dict[str, Any]) -> str:
        """Extract completion content from API response."""
        if response_data.get("choices") and len(response_data["choices"]) > 0:
            message = response_data["choices"][0].get("message")
            if message and message.get("content"):
                return message["content"]

        logger.error(
            f"Invalid response structure - missing choices/content: {response_data}"
        )
        return ""

    def get_statistics(self) -> dict[str, Any]:
        """Get completion service statistics."""
        total = self._stats["completions_requested"]
        return {
            **self._stats,
            "success_rate": (self._stats["completions_successful"] / total * 100)
            if total > 0
            else 0,
            "failure_rate": (self._stats["completions_failed"] / total * 100)
            if total > 0
            else 0,
            "fallback_rate": (self._stats["fallback_used"] / total * 100)
            if total > 0
            else 0,
            "streaming_rate": (self._stats["streaming_requests"] / total * 100)
            if total > 0
            else 0,
        }


class RefactoredLLMService:
    """
    Main LLM service with separated concerns architecture.

    REFACTORED: Complete rewrite using dependency injection and separated services.
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
        **kwargs,
    ) -> tuple[str, dict[str, int] | None]:
        """
        Call LLM with comprehensive options.

        REFACTORED: Simplified orchestration, delegating to specialized services.
        """
        if stream_to_disk:
            return await self._completion_service.get_streaming_completion(
                model_name,
                prompt,
                temperature,
                max_tokens,
                auto_clean_response,
                **kwargs,
            )
        else:
            return await self._completion_service.get_completion(
                model_name,
                prompt,
                temperature,
                max_tokens,
                allow_fallback,
                auto_clean_response,
                **kwargs,
            )

    async def async_get_embedding(self, text: str) -> np.ndarray | None:
        """
        Get embedding for text.

        REFACTORED: Simple delegation to embedding service.
        """
        return await self._embedding_service.get_embedding(text)

    async def async_get_embeddings_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[np.ndarray | None]:
        """
        Get embeddings for multiple texts in batches.

        REFACTORED: Simple delegation to embedding service.
        """
        return await self._embedding_service.get_embeddings_batch(texts, batch_size)

    def clean_model_response(self, text: str) -> str:
        """
        Clean LLM response text.

        REFACTORED: Simple delegation to text processing service.
        """
        return self._text_processor.response_cleaner.clean_response(text)

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
        return self._text_processor.tokenizer.truncate_text_by_tokens(
            text, model_name, max_tokens, truncation_marker
        )

    async def aclose(self) -> None:
        """Close service and cleanup resources."""
        # The HTTP client will be closed by the service locator
        logger.debug("RefactoredLLMService closed")

    def get_combined_statistics(self) -> dict[str, Any]:
        """Get combined statistics from all services."""
        return {
            "completion_service": self._completion_service.get_statistics(),
            "embedding_service": self._embedding_service.get_statistics(),
            "text_processor": self._text_processor.get_combined_statistics(),
        }


class DefaultLLMServiceFactory(LLMServiceFactory):
    """
    Default factory implementation for LLM services.

    Creates concrete implementations of all services with proper dependencies.
    """

    def create_http_client(self, timeout: float | None = None) -> HTTPClientService:
        """Create HTTP client service instance."""
        effective_timeout = timeout if timeout is not None else config.HTTPX_TIMEOUT
        return HTTPClientService(effective_timeout)

    def create_tokenizer(self) -> TextProcessingService:
        """Create tokenizer service instance."""
        return TextProcessingService()

    def create_response_cleaner(self) -> TextProcessingService:
        """Create response cleaner service instance."""
        return TextProcessingService()

    def create_stream_processor(self, response_cleaner) -> TextProcessingService:
        """Create stream processor service instance."""
        return TextProcessingService()

    def create_embedding_service(
        self, http_client: HTTPClientService
    ) -> EmbeddingService:
        """Create embedding service instance."""
        embedding_client = EmbeddingHTTPClient(http_client)
        return EmbeddingService(embedding_client)

    def create_completion_service(
        self,
        http_client: HTTPClientService,
        tokenizer,
        response_cleaner,
        stream_processor,
    ) -> CompletionService:
        """Create completion service instance."""
        completion_client = CompletionHTTPClient(http_client)
        text_processor = TextProcessingService()  # Creates all text processing services
        return CompletionService(completion_client, text_processor)

    def create_llm_service(
        self,
        completion_service: CompletionService,
        embedding_service: EmbeddingService,
        tokenizer,
        response_cleaner,
    ) -> RefactoredLLMService:
        """Create unified LLM service instance."""
        text_processor = TextProcessingService()
        return RefactoredLLMService(
            completion_service, embedding_service, text_processor
        )


# Initialize the service locator with default factory
_factory = DefaultLLMServiceFactory()
_service_locator = initialize_service_locator(_factory)


def get_llm_service() -> RefactoredLLMService:
    """
    Get the main LLM service instance.

    REFACTORED: Uses service locator instead of global singleton.
    """
    return _service_locator.get_llm_service()


# Create module-level instance for backward compatibility
llm_service = get_llm_service()


# Backward compatibility functions
def count_tokens(text: str, model_name: str) -> int:
    """Backward compatibility function for token counting."""
    return llm_service.count_tokens(text, model_name)
