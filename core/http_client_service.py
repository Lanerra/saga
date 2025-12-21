# core/http_client_service.py
"""Perform HTTP I/O for LLM provider integrations.

This module provides a small HTTP layer used by higher-level LLM services. It
centralizes concurrency limits, retry behavior, and response handling so call
sites do not re-implement network concerns.

Notes:
    - Requests are concurrency-limited via a semaphore.
    - Retries are applied for transient failures and server/rate-limit responses.
"""

import asyncio
from typing import Any

import httpx
import structlog

import config

logger = structlog.get_logger(__name__)


class HTTPClientService:
    """Perform concurrency-limited HTTP requests with retries."""

    def __init__(self, timeout: float = config.HTTPX_TIMEOUT):
        """Initialize the HTTP client.

        Args:
            timeout: Request timeout in seconds.
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

        logger.info(f"HTTPClientService initialized with timeout={timeout}s, " f"concurrency_limit={config.MAX_CONCURRENT_LLM_CALLS}")

    async def aclose(self) -> None:
        """Close the underlying HTTP client and release resources."""
        await self._client.aclose()
        logger.debug("HTTPClientService closed")

    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        max_retries: int | None = None,
    ) -> httpx.Response:
        """POST a JSON payload with retry behavior.

        Args:
            url: Target URL for the request.
            payload: JSON payload to send.
            headers: Optional HTTP headers.
            max_retries: Maximum retry attempts. When omitted, defaults to the
                configured value.

        Returns:
            The successful HTTP response.

        Raises:
            httpx.TimeoutException: When all attempts time out.
            httpx.HTTPStatusError: When a non-retryable status occurs or retries are
                exhausted.
            httpx.RequestError: When the request fails and retries are exhausted.
        """
        async with self._semaphore:
            self._stats["total_requests"] += 1
            self.request_count += 1

            effective_headers = headers or {}
            effective_max_retries = max_retries if max_retries is not None else config.LLM_RETRY_ATTEMPTS

            last_exception: Exception | None = None

            for attempt in range(effective_max_retries):
                try:
                    logger.debug(f"HTTP POST to {url} (attempt {attempt + 1}/{effective_max_retries})")

                    response = await self._client.post(url, json=payload, headers=effective_headers)
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

                    logger.warning(f"HTTP status error (attempt {attempt + 1}): {status_code} - {response_text}")

                    # Don't retry on client errors (except 429 rate limit)
                    if 400 <= status_code < 500 and status_code != 429:
                        logger.error(f"Non-retryable client error {status_code}, aborting")
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
                    logger.info(f"Retrying in {delay:.2f}s due to: {type(last_exception).__name__}")
                    await asyncio.sleep(delay)
                    self._stats["retry_attempts"] += 1

            # All retries failed
            self._stats["failed_requests"] += 1
            logger.error(f"HTTP POST failed after {effective_max_retries} attempts: {last_exception}")

            if last_exception:
                raise last_exception
            else:
                raise Exception("HTTP request failed with no specific error")

    # Streaming support removed to simplify HTTP client and standardize on non-streaming calls.

    def get_statistics(self) -> dict[str, Any]:
        """Return HTTP request statistics for monitoring."""
        total = self._stats["total_requests"]
        return {
            **self._stats,
            "success_rate": (self._stats["successful_requests"] / total * 100) if total > 0 else 0,
            "failure_rate": (self._stats["failed_requests"] / total * 100) if total > 0 else 0,
            "avg_retries_per_request": (self._stats["retry_attempts"] / total) if total > 0 else 0,
        }


class EmbeddingHTTPClient:
    """Call the embedding API using a shared HTTP client."""

    def __init__(self, http_client: HTTPClientService):
        """Initialize the embedding client.

        Args:
            http_client: Shared HTTP client used for requests.
        """
        self._http_client = http_client

    async def get_embedding(self, text: str, model: str) -> dict[str, Any]:
        """Request an embedding for the provided text.

        Args:
            text: Text to embed.
            model: Embedding model identifier.

        Returns:
            Parsed JSON response from the embedding provider.

        Raises:
            ValueError: If `text` is empty or whitespace.
            httpx.HTTPError: If the HTTP request fails.
        """
        if not text or not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string")

        payload = {"model": model, "prompt": text.strip()}

        logger.debug(f"Requesting embedding from Ollama for model '{model}': " f"'{text[:80].replace(chr(10), ' ')}...'")

        response = await self._http_client.post_json(f"{config.EMBEDDING_API_BASE}/api/embeddings", payload)

        return response.json()


class CompletionHTTPClient:
    """Call the chat completion API using a shared HTTP client."""

    def __init__(self, http_client: HTTPClientService):
        """Initialize the completion client.

        Args:
            http_client: Shared HTTP client used for requests.
        """
        self._http_client = http_client

    async def get_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Request a chat completion from an OpenAI-compatible API.

        Args:
            model: Model identifier.
            messages: Chat messages payload.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific parameters merged into the request.

        Returns:
            Parsed JSON response from the completion provider.

        Raises:
            httpx.HTTPError: If the HTTP request fails.
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

        logger.debug(f"Requesting completion from {config.OPENAI_API_BASE} " f"for model '{model}' with {len(messages)} messages")

        response = await self._http_client.post_json(f"{config.OPENAI_API_BASE}/chat/completions", payload, headers)

        return response.json()

    # Streaming completion removed; use get_completion() only.
