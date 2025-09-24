# core/http_client_service.py
"""
HTTP client service for LLM API interactions.

This module provides the low-level HTTP client functionality for communicating
with LLM APIs, extracted from the monolithic LLMService to improve separation
of concerns and maintainability.

REFACTORED: Extracted from core.llm_interface as part of Phase 3 architectural improvements.
- Focuses solely on HTTP communication concerns
- Proper error handling and retry logic
- Streaming and non-streaming request support
- Concurrent request management
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import structlog

import config

logger = structlog.get_logger(__name__)


class HTTPClientService:
    """
    Service for handling HTTP communications with LLM APIs.

    This service encapsulates all HTTP-related concerns including:
    - Connection management
    - Retry logic
    - Error handling
    - Concurrency control
    - Streaming support
    """

    def __init__(self, timeout: float = config.HTTPX_TIMEOUT):
        """
        Initialize the HTTP client service.

        Args:
            timeout: HTTP request timeout in seconds
        """
        self._client = httpx.AsyncClient(timeout=timeout)
        self._semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_LLM_CALLS)
        self.request_count = 0
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retry_attempts": 0,
        }

        logger.info(
            f"HTTPClientService initialized with timeout={timeout}s, "
            f"concurrency_limit={config.MAX_CONCURRENT_LLM_CALLS}"
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client and cleanup resources."""
        await self._client.aclose()
        logger.debug("HTTPClientService closed")

    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        max_retries: int | None = None,
    ) -> httpx.Response:
        """
        Make a POST request with JSON payload and retry logic.

        Args:
            url: Target URL for the request
            payload: JSON payload to send
            headers: Optional HTTP headers
            max_retries: Maximum retry attempts (defaults to config value)

        Returns:
            HTTP response object

        Raises:
            httpx.HTTPError: For unrecoverable HTTP errors
            Exception: For other unexpected errors
        """
        async with self._semaphore:
            self._stats["total_requests"] += 1
            self.request_count += 1

            effective_headers = headers or {}
            effective_max_retries = (
                max_retries if max_retries is not None else config.LLM_RETRY_ATTEMPTS
            )

            last_exception: Exception | None = None

            for attempt in range(effective_max_retries):
                try:
                    logger.debug(
                        f"HTTP POST to {url} (attempt {attempt + 1}/{effective_max_retries})"
                    )

                    response = await self._client.post(
                        url, json=payload, headers=effective_headers
                    )
                    response.raise_for_status()

                    self._stats["successful_requests"] += 1
                    logger.debug(f"HTTP POST successful: {response.status_code}")
                    return response

                except httpx.TimeoutException as e:
                    last_exception = e
                    logger.warning(f"HTTP timeout (attempt {attempt + 1}): {e}")

                except httpx.HTTPStatusError as e:
                    last_exception = e
                    status_code = e.response.status_code if e.response else 0
                    response_text = e.response.text[:200] if e.response else "N/A"

                    logger.warning(
                        f"HTTP status error (attempt {attempt + 1}): {status_code} - {response_text}"
                    )

                    # Don't retry on client errors (except 429 rate limit)
                    if 400 <= status_code < 500 and status_code != 429:
                        logger.error(
                            f"Non-retryable client error {status_code}, aborting"
                        )
                        break

                except httpx.RequestError as e:
                    last_exception = e
                    logger.warning(f"HTTP request error (attempt {attempt + 1}): {e}")

                except Exception as e:
                    last_exception = e
                    logger.error(
                        f"Unexpected HTTP error (attempt {attempt + 1}): {e}",
                        exc_info=True,
                    )

                # Apply retry delay if not the last attempt
                if attempt < effective_max_retries - 1:
                    delay = config.LLM_RETRY_DELAY_SECONDS * (2**attempt)
                    logger.info(
                        f"Retrying in {delay:.2f}s due to: {type(last_exception).__name__}"
                    )
                    await asyncio.sleep(delay)
                    self._stats["retry_attempts"] += 1

            # All retries failed
            self._stats["failed_requests"] += 1
            logger.error(
                f"HTTP POST failed after {effective_max_retries} attempts: {last_exception}"
            )

            if last_exception:
                raise last_exception
            else:
                raise Exception("HTTP request failed with no specific error")

    # Streaming support removed to simplify HTTP client and standardize on non-streaming calls.

    def get_statistics(self) -> dict[str, Any]:
        """Get HTTP client statistics for monitoring and debugging."""
        total = self._stats["total_requests"]
        return {
            **self._stats,
            "success_rate": (self._stats["successful_requests"] / total * 100)
            if total > 0
            else 0,
            "failure_rate": (self._stats["failed_requests"] / total * 100)
            if total > 0
            else 0,
            "avg_retries_per_request": (self._stats["retry_attempts"] / total)
            if total > 0
            else 0,
        }

    def reset_statistics(self) -> None:
        """Reset statistics (useful for testing)."""
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retry_attempts": 0,
        }
        self.request_count = 0


class EmbeddingHTTPClient:
    """
    Specialized HTTP client for embedding API requests.

    This client provides embedding-specific functionality built on top
    of the base HTTPClientService.
    """

    def __init__(self, http_client: HTTPClientService):
        """
        Initialize embedding client.

        Args:
            http_client: Base HTTP client service to use
        """
        self._http_client = http_client

    async def get_embedding(self, text: str, model: str) -> dict[str, Any]:
        """
        Get embedding for text from Ollama API.

        Args:
            text: Text to get embedding for
            model: Embedding model to use

        Returns:
            API response dictionary

        Raises:
            ValueError: If text is empty or invalid
            Exception: For HTTP or API errors
        """
        if not text or not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string")

        payload = {"model": model, "prompt": text.strip()}

        logger.debug(
            f"Requesting embedding from Ollama for model '{model}': "
            f"'{text[:80].replace(chr(10), ' ')}...'"
        )

        response = await self._http_client.post_json(
            f"{config.EMBEDDING_API_BASE}/api/embeddings", payload
        )

        return response.json()


class CompletionHTTPClient:
    """
    Specialized HTTP client for completion API requests.

    This client provides completion-specific functionality built on top
    of the base HTTPClientService.
    """

    def __init__(self, http_client: HTTPClientService):
        """
        Initialize completion client.

        Args:
            http_client: Base HTTP client service to use
        """
        self._http_client = http_client

    async def get_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Get completion from OpenAI-compatible API.

        Args:
            model: Model to use for completion
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional completion parameters

        Returns:
            API response dictionary
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": config.LLM_TOP_P,
            "max_tokens": max_tokens,
            "stream": False,
            **kwargs,
        }

        headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        logger.debug(
            f"Requesting completion from {config.OPENAI_API_BASE} "
            f"for model '{model}' with {len(messages)} messages"
        )

        response = await self._http_client.post_json(
            f"{config.OPENAI_API_BASE}/chat/completions", payload, headers
        )

        return response.json()

    # Streaming completion removed; use get_completion() only.
