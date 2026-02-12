# core/llm_interface_refactored.py
"""Provide the primary LLM client interface for SAGA.

This module centralizes:
- Completion calls (OpenAI-compatible chat completion APIs).
- Embedding calls (Ollama-compatible embedding APIs).
- Coordinated caching for embeddings.
- Consistent error contracts for strict vs best-effort call sites.

Notes:
    This module avoids logging raw prompt contents on completion failures. It logs only
    prompt hashes and lengths to support debugging without leaking user content.
"""

import asyncio
import hashlib
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

import numpy as np
import structlog

import config
from core.exceptions import LLMServiceError, create_error_context
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
from core.text_processing_service import TextProcessingService, truncate_text_by_tokens

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def async_llm_context(
    batch_size: int | None = None,
    clear_cache_on_exit: bool = False,
) -> AsyncGenerator[tuple["RefactoredLLMService", "EmbeddingService"], None]:
    """Create LLM services with guaranteed HTTP client cleanup.

    Args:
        batch_size: Default batch size for embedding batch calls when the caller does not
            provide one.
        clear_cache_on_exit: Whether to clear the embedding cache namespace on exit.

    Yields:
        Tuple of (`RefactoredLLMService`, `EmbeddingService`) instances.

    Notes:
        This context manager owns the underlying HTTP client lifecycle. It should be used
        for workflows that create many embeddings/completions and want deterministic
        cleanup.

        If `clear_cache_on_exit=True`, the `llm_embedding` cache namespace is cleared when
        the context exits.
    """
    # Direct instantiation instead of service locator
    http_client = HTTPClientService()
    embedding_client = EmbeddingHTTPClient(http_client)
    completion_client = CompletionHTTPClient(http_client)
    text_processor = TextProcessingService()

    embedding_service = EmbeddingService(embedding_client)
    completion_service = CompletionService(completion_client, text_processor)
    llm_service = RefactoredLLMService(completion_service, embedding_service, text_processor, http_client)

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
    """Generate and cache embedding vectors for text."""

    def __init__(self, embedding_client: EmbeddingHTTPClient):
        """Initialize the embedding service.

        Args:
            embedding_client: HTTP client used to perform embedding requests.
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
        """Compute a stable cache key for an embedding input string."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    async def get_embedding(self, text: str) -> np.ndarray | None:
        """Get an embedding vector for a text input.

        Args:
            text: Input text to embed. Must be a non-empty string after stripping.

        Returns:
            Embedding vector, or None when the input is invalid, the request fails, or the
            provider returns an invalid embedding payload.

        Notes:
            Successful embeddings are cached in the `llm_embedding` namespace keyed by a
            hash of the stripped input text.
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

        truncated_text = truncate_text_by_tokens(
            text=text,
            model_name=config.EMBEDDING_MODEL,
            max_tokens=config.EMBEDDING_MAX_INPUT_TOKENS,
        )

        try:
            response_data = await self._embedding_client.get_embedding(truncated_text, config.EMBEDDING_MODEL)

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
        """Get embeddings for many inputs with bounded concurrency.

        Args:
            texts: Inputs to embed. Empty inputs are allowed but may yield None results,
                depending on per-item validation.
            batch_size: Maximum number of concurrent embedding requests in one batch.

        Returns:
            List of embeddings aligned to the input order.
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
        """Extract an embedding vector from a provider response and validate its shape."""
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
    """Generate text completions via OpenAI-compatible APIs."""

    def __init__(
        self,
        completion_client: CompletionHTTPClient,
        text_processor: TextProcessingService,
    ):
        """Initialize the completion service.

        Args:
            completion_client: HTTP client used to perform completion requests.
            text_processor: Service used for response cleanup and token operations.
        """
        self._completion_client = completion_client
        self._text_processor = text_processor
        self._stats = {
            "completions_requested": 0,
            "completions_successful": 0,
            "completions_failed": 0,
            "fallback_used": 0,
        }

    async def get_completion(
        self,
        model_name: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        allow_fallback: bool = False,
        auto_clean_response: bool = True,
        spacy_cleanup: bool = True,
        *,
        system_prompt: str | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> tuple[str, dict[str, int] | None]:
        """Request a completion for a prompt.

        Args:
            model_name: Provider model identifier.
            prompt: User prompt content.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            allow_fallback: Whether to attempt a fallback model after a primary failure.
            auto_clean_response: Whether to apply response cleanup.
            spacy_cleanup: Whether to apply spaCy-based text cleaning after regex cleanup.
            system_prompt: Optional system prompt injected as a system message.
            strict: Whether to raise a typed exception on failure.
            **kwargs: Provider-specific completion parameters forwarded to the HTTP client.

        Returns:
            Tuple of `(response_text, usage_data)`.

        Raises:
            LLMServiceError: When `strict=True` and the request fails or required inputs
                are missing.

        Notes:
            This method avoids logging raw prompt contents on failures. It logs only a
            hash and length.
        """
        self._stats["completions_requested"] += 1

        if not model_name or not prompt:
            self._stats["completions_failed"] += 1
            error_details = create_error_context(
                model_name=model_name,
                prompt_len=len(prompt) if isinstance(prompt, str) else None,
                allow_fallback=allow_fallback,
            )
            if strict:
                raise LLMServiceError("get_completion requires non-empty model_name and prompt", details=error_details)
            logger.error("get_completion: model_name and prompt are required", **error_details)
            return "", None

        # Respect global temperature override if set
        if config.Temperatures.OVERRIDE is not None:
            effective_temperature = config.Temperatures.OVERRIDE
        else:
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
                **kwargs,
            )

            content = self._extract_completion_content(response_data)

            finish_reason = None
            try:
                if response_data.get("choices") and len(response_data["choices"]) > 0:
                    finish_reason = response_data["choices"][0].get("finish_reason")
            except Exception:  # pragma: no cover
                finish_reason = None

            usage_data = response_data.get("usage")
            if usage_data is None:
                usage_data = {}

            if isinstance(usage_data, dict) and finish_reason is not None:
                usage_data = {**usage_data, "finish_reason": finish_reason}

            if auto_clean_response:
                content = self._text_processor.response_cleaner.clean_response(content)

            if spacy_cleanup:
                content = self._text_processor.clean_text_with_spacy(content, aggressive=False)

            self._stats["completions_successful"] += 1
            return content, usage_data

        except Exception as primary_error:
            # Never log raw prompt; capture only hash+length to aid debugging.
            try:
                prompt_sha1 = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
                prompt_len = len(prompt)
            except Exception:  # pragma: no cover
                prompt_sha1 = None
                prompt_len = None

            logger.error(
                "get_completion: primary model failed",
                model=model_name,
                prompt_sha1=prompt_sha1,
                prompt_len=prompt_len,
                error=str(primary_error),
                exc_info=True,
            )

            fallback_error: Exception | None = None

            # Try fallback if enabled
            if allow_fallback and config.MEDIUM_MODEL:
                logger.info(
                    "get_completion: attempting fallback model",
                    fallback_model=config.MEDIUM_MODEL,
                    primary_model=model_name,
                    prompt_sha1=prompt_sha1,
                )
                self._stats["fallback_used"] += 1

                try:
                    response_data = await self._completion_client.get_completion(
                        config.MEDIUM_MODEL,
                        messages,
                        effective_temperature,
                        effective_max_tokens,
                        **kwargs,
                    )

                    content = self._extract_completion_content(response_data)

                    finish_reason = None
                    try:
                        if response_data.get("choices") and len(response_data["choices"]) > 0:
                            finish_reason = response_data["choices"][0].get("finish_reason")
                    except Exception:  # pragma: no cover
                        finish_reason = None

                    usage_data = response_data.get("usage")
                    if usage_data is None:
                        usage_data = {}

                    if isinstance(usage_data, dict) and finish_reason is not None:
                        usage_data = {**usage_data, "finish_reason": finish_reason}

                    if auto_clean_response:
                        content = self._text_processor.response_cleaner.clean_response(content)

                    if spacy_cleanup:
                        content = self._text_processor.clean_text_with_spacy(content, aggressive=False)

                    self._stats["completions_successful"] += 1
                    return content, usage_data

                except Exception as exc:
                    fallback_error = exc
                    logger.error(
                        "get_completion: fallback model failed",
                        primary_model=model_name,
                        fallback_model=config.MEDIUM_MODEL,
                        prompt_sha1=prompt_sha1,
                        prompt_len=prompt_len,
                        error=str(exc),
                        exc_info=True,
                    )

            self._stats["completions_failed"] += 1

            error_details = create_error_context(
                primary_model=model_name,
                fallback_model=config.MEDIUM_MODEL if allow_fallback else None,
                allow_fallback=allow_fallback,
                prompt_sha1=prompt_sha1,
                prompt_len=prompt_len,
                primary_error=str(primary_error),
                primary_error_type=type(primary_error).__name__,
                fallback_error=str(fallback_error) if fallback_error else None,
                fallback_error_type=type(fallback_error).__name__ if fallback_error else None,
            )

            if strict:
                raise LLMServiceError("LLM completion failed", details=error_details) from primary_error

            # Compatibility: explicit non-strict mode preserves legacy sentinel return.
            return "", None

    # Streaming completion path removed to simplify the API.

    def _extract_completion_content(self, response_data: dict[str, Any]) -> str:
        """Extract completion text from a provider response.

        Returns:
            Extracted text, or an empty string when no usable content is present.

        Notes:
            Providers sometimes return structured content as a list of parts. This method
            accepts common variants used by OpenAI-compatible APIs.
        """

        def _extract_text_from_content(content_value: Any) -> str | None:
            if isinstance(content_value, str):
                return content_value if content_value.strip() else None

            if isinstance(content_value, list):
                text_parts: list[str] = []
                for item in content_value:
                    if isinstance(item, str):
                        if item:
                            text_parts.append(item)
                        continue

                    if not isinstance(item, dict):
                        continue

                    item_type = item.get("type")
                    if isinstance(item_type, str) and item_type != "text":
                        continue

                    text_field = item.get("text")
                    if isinstance(text_field, str):
                        if text_field:
                            text_parts.append(text_field)
                        continue

                    if isinstance(text_field, dict):
                        nested_value = text_field.get("value")
                        if isinstance(nested_value, str) and nested_value:
                            text_parts.append(nested_value)
                        continue

                combined_text = "".join(text_parts)
                return combined_text if combined_text.strip() else None

            return None

        # Prefer the standard OpenAI schema first
        try:
            if response_data.get("choices") and len(response_data["choices"]) > 0:
                choice0 = response_data["choices"][0]
                message = choice0.get("message") or {}

                # 1) Standard content (string or list-of-parts)
                content_text = _extract_text_from_content(message.get("content"))
                if content_text is not None:
                    return content_text

                # 2) Occasionally providers place content directly under the choice
                direct_choice_content_text = _extract_text_from_content(choice0.get("content"))
                if direct_choice_content_text is not None:
                    logger.warning("LLM response missing message.content; using choice['content'] fallback")
                    return direct_choice_content_text

                # 3) Last resorts: top-level convenience fields sometimes appear
                for key in ("output_text", "text", "response", "content"):
                    top_text = _extract_text_from_content(response_data.get(key))
                    if top_text is not None:
                        logger.warning(f"LLM response using top-level '{key}' fallback for content")
                        return top_text

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
        }


class RefactoredLLMService:
    """Expose completions, embeddings, and token utilities behind a single interface."""

    def __init__(
        self,
        completion_service: CompletionService,
        embedding_service: EmbeddingService,
        text_processor: TextProcessingService,
        http_client: "HTTPClientService | None" = None,
    ):
        """Initialize the service with explicit dependencies.

        Args:
            completion_service: Completion provider wrapper.
            embedding_service: Embedding provider wrapper.
            text_processor: Text cleanup and tokenization utilities.
            http_client: Underlying HTTP client for lifecycle management.
        """
        self._completion_service = completion_service
        self._embedding_service = embedding_service
        self._text_processor = text_processor
        self._http_client = http_client

        logger.info("RefactoredLLMService initialized with separated components")

    async def aclose(self) -> None:
        """Close the underlying HTTP client and release connection resources."""
        if self._http_client is not None:
            await self._http_client.aclose()

    async def async_call_llm(
        self,
        model_name: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        allow_fallback: bool = False,
        auto_clean_response: bool = True,
        spacy_cleanup: bool = False,
        *,
        system_prompt: str | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> tuple[str, dict[str, int] | None]:
        """Call the LLM completion API.

        Args:
            model_name: Provider model identifier.
            prompt: User prompt content.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            allow_fallback: Whether to attempt a fallback model after a primary failure.
            auto_clean_response: Whether to apply response cleanup.
            spacy_cleanup: Whether to apply spaCy-based text cleaning after regex cleanup.
            system_prompt: Optional system prompt injected as a system message.
            strict: Whether to raise a typed exception on failure.
            **kwargs: Provider-specific completion parameters forwarded to the HTTP client.

        Returns:
            Tuple of `(response_text, usage_data)`.

        Raises:
            LLMServiceError: When `strict=True` and the completion call fails.

        Notes:
            CORE-007 error contract:
            - When `strict=True`, failures raise a typed exception instead of returning
              ambiguous sentinels like `("", None)`.
            - When `strict=False`, failures return `("", None)` for compatibility with
              best-effort call sites.
        """
        return await self._completion_service.get_completion(
            model_name,
            prompt,
            temperature,
            max_tokens,
            allow_fallback,
            auto_clean_response,
            spacy_cleanup,
            system_prompt=system_prompt,
            strict=strict,
            **kwargs,
        )

    async def async_call_llm_json_object(
        self,
        model_name: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        allow_fallback: bool = False,
        auto_clean_response: bool = True,
        *,
        system_prompt: str | None = None,
        strict: bool = True,
        max_attempts: int = config.JSON_PARSE_RETRY_ATTEMPTS,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, int] | None]:
        """Call the LLM and parse the response as a JSON object.

        Args:
            model_name: Provider model identifier.
            prompt: User prompt content.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            allow_fallback: Whether to attempt a fallback model after a primary failure.
            auto_clean_response: Whether to apply response cleanup before parsing JSON.
            system_prompt: Optional system prompt injected as a system message.
            strict: Whether to raise a typed exception on completion failure.
            max_attempts: Maximum number of attempts to obtain valid JSON.
            **kwargs: Provider-specific completion parameters forwarded to the HTTP client.

        Returns:
            Tuple of `(data, usage_data)` where `data` is a JSON object.

        Raises:
            ValueError: If `max_attempts < 1`, or if the model does not return a valid JSON
                object after all attempts.
            LLMServiceError: When `strict=True` and the underlying completion call fails.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        last_decode_error: json.JSONDecodeError | None = None

        for attempt in range(1, max_attempts + 1):
            text, usage = await self.async_call_llm(
                model_name=model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                allow_fallback=allow_fallback,
                auto_clean_response=auto_clean_response,
                system_prompt=system_prompt,
                strict=strict,
                **kwargs,
            )

            try:
                data = json.loads(text)
            except json.JSONDecodeError as decode_error:
                last_decode_error = decode_error

                response_sha1 = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
                prompt_sha1 = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
                finish_reason = usage.get("finish_reason") if isinstance(usage, dict) else None
                completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
                prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
                total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None

                logger.warning(
                    "LLM returned invalid JSON (object expected)",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    model=model_name,
                    requested_max_tokens=max_tokens,
                    temperature=temperature,
                    auto_clean_response=auto_clean_response,
                    finish_reason=finish_reason,
                    prompt_sha1=prompt_sha1,
                    prompt_len=len(prompt),
                    response_sha1=response_sha1,
                    response_len=len(text),
                    completion_tokens=completion_tokens,
                    prompt_tokens=prompt_tokens,
                    total_tokens=total_tokens,
                    line=decode_error.lineno,
                    column=decode_error.colno,
                )
                continue

            if not isinstance(data, dict):
                response_sha1 = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
                logger.warning(
                    "LLM returned JSON but root value was not an object",
                    model=model_name,
                    response_sha1=response_sha1,
                    response_len=len(text),
                    root_type=type(data).__name__,
                )
                raise ValueError("LLM returned JSON but root value was not an object")

            return data, usage

        if last_decode_error is not None:
            raise ValueError("LLM returned invalid JSON") from last_decode_error

        raise ValueError("LLM returned invalid JSON")

    async def async_call_llm_json_array(
        self,
        model_name: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        allow_fallback: bool = False,
        auto_clean_response: bool = True,
        *,
        system_prompt: str | None = None,
        strict: bool = True,
        max_attempts: int = config.JSON_PARSE_RETRY_ATTEMPTS,
        **kwargs: Any,
    ) -> tuple[list[Any], dict[str, int] | None]:
        """Call the LLM and parse the response as a JSON array.

        Args:
            model_name: Provider model identifier.
            prompt: User prompt content.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            allow_fallback: Whether to attempt a fallback model after a primary failure.
            auto_clean_response: Whether to apply response cleanup before parsing JSON.
            system_prompt: Optional system prompt injected as a system message.
            strict: Whether to raise a typed exception on completion failure.
            max_attempts: Maximum number of attempts to obtain valid JSON.
            **kwargs: Provider-specific completion parameters forwarded to the HTTP client.

        Returns:
            Tuple of `(data, usage_data)` where `data` is a JSON array.

        Raises:
            ValueError: If `max_attempts < 1`, or if the model does not return a valid JSON
                array after all attempts.
            LLMServiceError: When `strict=True` and the underlying completion call fails.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        last_decode_error: json.JSONDecodeError | None = None

        for attempt in range(1, max_attempts + 1):
            text, usage = await self.async_call_llm(
                model_name=model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                allow_fallback=allow_fallback,
                auto_clean_response=auto_clean_response,
                system_prompt=system_prompt,
                strict=strict,
                **kwargs,
            )

            try:
                data = json.loads(text)
            except json.JSONDecodeError as decode_error:
                last_decode_error = decode_error

                response_sha1 = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
                logger.warning(
                    "LLM returned invalid JSON (array expected)",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    response_sha1=response_sha1,
                    response_len=len(text),
                    line=decode_error.lineno,
                    column=decode_error.colno,
                )
                continue

            if not isinstance(data, list):
                raise ValueError("LLM returned JSON but root value was not an array")

            return data, usage

        if last_decode_error is not None:
            raise ValueError("LLM returned invalid JSON") from last_decode_error

        raise ValueError("LLM returned invalid JSON")

    async def async_get_embedding(self, text: str) -> np.ndarray | None:
        """Get an embedding for a single text input."""
        return await self._embedding_service.get_embedding(text)

    async def async_get_embeddings_batch(self, texts: list[str], batch_size: int | None = None) -> list[np.ndarray | None]:
        """Get embeddings for many inputs with bounded concurrency."""
        return await self._embedding_service.get_embeddings_batch(texts, batch_size)

    def count_tokens(self, text: str, model_name: str) -> int:
        """Count model tokens for a text input."""
        return self._text_processor.tokenizer.count_tokens(text, model_name)

    def truncate_text_by_tokens(
        self,
        text: str,
        model_name: str,
        max_tokens: int,
        truncation_marker: str = "\n... (truncated)",
    ) -> str:
        """Truncate a text input to a token budget."""
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
    """Construct a new LLM service instance with direct dependency injection."""
    http_client = HTTPClientService()
    embedding_client = EmbeddingHTTPClient(http_client)
    completion_client = CompletionHTTPClient(http_client)
    text_processor = TextProcessingService()

    embedding_service = EmbeddingService(embedding_client)
    completion_service = CompletionService(completion_client, text_processor)
    return RefactoredLLMService(completion_service, embedding_service, text_processor, http_client)


# Module-level service instance
llm_service = create_llm_service()


async def close_llm_service() -> None:
    """Close the module-level LLM service singleton and release HTTP resources."""
    await llm_service.aclose()
