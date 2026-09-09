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
from typing import Any, cast

import numpy as np
import structlog
from pydantic import ValidationError

import config
from config.settings import EffectiveSettings
from core.embedding_contract import embedding_identity, validate_embedding
from core.exceptions import LLMServiceError, create_error_context
from core.http_client_service import (
    CompletionHTTPClient,
    EmbeddingHTTPClient,
    EmbeddingResponse,
    HTTPClientService,
    bounded_request,
    completion_content,
)
from core.lightweight_cache import (
    get_cached_value,
    register_cache_service,
    set_cached_value,
)
from core.text_processing_service import TextProcessingService, truncate_text_by_tokens

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """Generate and cache embedding vectors for text."""

    def __init__(self, embedding_client: EmbeddingHTTPClient, *, configuration: EffectiveSettings | None = None):
        """Initialize the embedding service.

        Args:
            embedding_client: HTTP client used to perform embedding requests.
        """
        self._embedding_client = embedding_client
        self.configuration = configuration if configuration is not None else config.snapshot_settings()
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
        return hashlib.sha256((embedding_identity(self.configuration) + "\n" + text).encode("utf-8")).hexdigest()

    @bounded_request
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
            logger.warning("get_embedding: empty or invalid text provided")
            self._stats["embeddings_failed"] += 1
            return None

        # Check coordinated cache first
        text_hash = self._compute_text_hash(text.strip())
        cached_embedding = get_cached_value(text_hash, self._service_name)
        if cached_embedding is not None:
            self._stats["cache_hits"] += 1
            logger.debug(f"Cache hit for embedding (hash: {text_hash[:8]})")
            return validate_embedding(cached_embedding, model=self.configuration.EMBEDDING_MODEL, configuration=self.configuration)

        self._stats["cache_misses"] += 1

        truncated_text = truncate_text_by_tokens(
            text=text,
            model_name=self.configuration.EMBEDDING_MODEL,
            max_tokens=self.configuration.EMBEDDING_MAX_INPUT_TOKENS,
        )

        try:
            response_data = await self._embedding_client.get_embedding(truncated_text, self.configuration.EMBEDDING_MODEL)

            # Extract and validate embedding
            embedding = self._extract_and_validate_embedding(response_data)
            if embedding is not None:
                # Cache the successful embedding
                set_cached_value(text_hash, embedding.copy(), self._service_name)
                self._stats["embeddings_successful"] += 1
                return embedding
            else:
                self._stats["validation_failures"] += 1
                return None

        except TimeoutError:
            raise
        except Exception as e:
            logger.error("Failed to get embedding", error_type=type(e).__name__)
            self._stats["embeddings_failed"] += 1
            return None

    @bounded_request
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

        batch_size = self.configuration.MAX_CONCURRENT_LLM_CALLS if batch_size is None else batch_size
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Embedding batch size must be a positive integer")
        results: list[np.ndarray | None] = [None] * len(texts)

        # Process in batches to control concurrency and memory usage
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_tasks = [self.get_embedding(text) for text in batch_texts]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for j, result in enumerate(batch_results):
                if isinstance(result, (asyncio.CancelledError, TimeoutError)):
                    raise result
                if not isinstance(result, Exception):
                    results[i + j] = cast(np.ndarray | None, result)

        return results

    def _extract_and_validate_embedding(self, response_data: dict[str, Any]) -> np.ndarray | None:
        """Extract an embedding vector from a provider response and validate its shape."""
        try:
            response = EmbeddingResponse.model_validate(response_data)
        except ValidationError:
            logger.error("Invalid embedding provider response schema")
            return None
        return self._validate_embedding_list(response.embedding)

    def _validate_embedding_list(self, embedding_list: list[float | int]) -> np.ndarray | None:
        """Validate and convert embedding list to numpy array."""
        try:
            return validate_embedding(embedding_list, model=self.configuration.EMBEDDING_MODEL, configuration=self.configuration)

        except (TypeError, ValueError):
            logger.error("Failed to convert embedding list to numpy array")

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
        *, configuration: EffectiveSettings | None = None,
    ):
        """Initialize the completion service.

        Args:
            completion_client: HTTP client used to perform completion requests.
            text_processor: Service used for response cleanup and token operations.
        """
        self._completion_client = completion_client
        self.configuration = configuration if configuration is not None else config.snapshot_settings()
        self._text_processor = text_processor
        self._stats = {
            "completions_requested": 0,
            "completions_successful": 0,
            "completions_failed": 0,
            "fallback_used": 0,
        }

    @bounded_request
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
        if self.configuration.TEMPERATURE_OVERRIDE is not None:
            effective_temperature = self.configuration.TEMPERATURE_OVERRIDE
        else:
            effective_temperature = temperature if temperature is not None else config.Temperatures.DEFAULT
        effective_max_tokens = max_tokens if max_tokens is not None else self.configuration.MAX_GENERATION_TOKENS

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

        except TimeoutError:
            raise
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
                error_type=type(primary_error).__name__,
            )

            fallback_error: Exception | None = None

            # Try fallback if enabled
            if allow_fallback and self.configuration.MEDIUM_MODEL:
                logger.info(
                    "get_completion: attempting fallback model",
                    fallback_model=self.configuration.MEDIUM_MODEL,
                    primary_model=model_name,
                    prompt_sha1=prompt_sha1,
                )
                self._stats["fallback_used"] += 1

                try:
                    response_data = await self._completion_client.get_completion(
                        self.configuration.MEDIUM_MODEL,
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

                except TimeoutError:
                    raise
                except Exception as exc:
                    fallback_error = exc
                    logger.error(
                        "get_completion: fallback model failed",
                        primary_model=model_name,
                        fallback_model=self.configuration.MEDIUM_MODEL,
                        prompt_sha1=prompt_sha1,
                        prompt_len=prompt_len,
                        error_type=type(exc).__name__,
                    )

            self._stats["completions_failed"] += 1

            error_details = create_error_context(
                primary_model=model_name,
                fallback_model=self.configuration.MEDIUM_MODEL if allow_fallback else None,
                allow_fallback=allow_fallback,
                prompt_sha1=prompt_sha1,
                prompt_len=prompt_len,
                primary_error_type=type(primary_error).__name__,
                fallback_error_type=type(fallback_error).__name__ if fallback_error else None,
            )

            if strict:
                raise LLMServiceError("LLM completion failed", details=error_details) from None

            # Compatibility: explicit non-strict mode preserves legacy sentinel return.
            return "", None

    # Streaming completion path removed to simplify the API.

    def _extract_completion_content(self, response_data: dict[str, Any]) -> str:
        """Extract text using the configured provider contract."""
        return completion_content(response_data, self.configuration)

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
        self.configuration = completion_service.configuration
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

    @bounded_request
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
        max_attempts: int | None = None,
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
        max_attempts = self.configuration.JSON_PARSE_RETRY_ATTEMPTS if max_attempts is None else max_attempts
        if type(max_attempts) is not int or max_attempts < 1:
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

    @bounded_request
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
        max_attempts: int | None = None,
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
        max_attempts = self.configuration.JSON_PARSE_RETRY_ATTEMPTS if max_attempts is None else max_attempts
        if type(max_attempts) is not int or max_attempts < 1:
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
def create_llm_service(http_client: HTTPClientService | None = None, *, configuration: EffectiveSettings | None = None) -> RefactoredLLMService:
    """Construct a new LLM service instance with direct dependency injection."""
    if http_client is None:
        http_client = HTTPClientService(configuration=configuration)
    elif configuration is not None and http_client.configuration is not configuration:
        raise ValueError("HTTP and LLM services must share one effective configuration")
    embedding_client = EmbeddingHTTPClient(http_client)
    completion_client = CompletionHTTPClient(http_client)
    text_processor = TextProcessingService()

    embedding_service = EmbeddingService(embedding_client, configuration=http_client.configuration)
    completion_service = CompletionService(completion_client, text_processor, configuration=http_client.configuration)
    return RefactoredLLMService(completion_service, embedding_service, text_processor, http_client)
