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
import hashlib
import json
import math
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from functools import wraps
from typing import Annotated, Any, Concatenate, Literal

import httpx
import structlog
import tiktoken
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import config
from config.settings import EffectiveSettings

logger = structlog.get_logger(__name__)

_request_deadline: ContextVar[float | None] = ContextVar("saga_request_deadline", default=None)



def bounded_request[**P, R](function: Callable[Concatenate[Any, P], Coroutine[Any, Any, R]]) -> Callable[Concatenate[Any, P], Coroutine[Any, Any, R]]:
    """Share one absolute HTTPX_TIMEOUT across retries, fallback and JSON repair.

    Cancellation propagates unchanged. Expiry raises TimeoutError, including in
    best-effort callers. Child tasks inherit the deadline, never a fresh allowance.
    CPU-only work is checked on return; asyncio cancellation is cooperative.
    """
    @wraps(function)
    async def bounded(self: Any, *arguments: P.args, **keywords: P.kwargs) -> R:
        loop = asyncio.get_running_loop()
        seconds = getattr(self, "_total_request_timeout", self.configuration.HTTPX_TIMEOUT)
        deadline = loop.time() + seconds
        inherited = _request_deadline.get()
        if inherited is not None:
            deadline = min(deadline, inherited)
        token = _request_deadline.set(deadline)
        try:
            if loop.time() >= deadline:
                raise TimeoutError("Total provider request deadline exceeded")
            async with asyncio.timeout_at(deadline):
                result = await function(self, *arguments, **keywords)
                if loop.time() >= deadline:
                    raise TimeoutError("Total provider request deadline exceeded")
                return result
        finally:
            _request_deadline.reset(token)
    return bounded


class _ProviderModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


class EmbeddingResponse(_ProviderModel):
    embedding: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(min_length=1)


class _TextPart(_ProviderModel):
    type: Literal["text"]
    text: str


class _Message(_ProviderModel):
    role: Literal["assistant"] = "assistant"
    content: str = Field(min_length=1)
    reasoning_content: str | None = None
    refusal: str | None = None


class _PartsMessage(_ProviderModel):
    role: Literal["assistant"] = "assistant"
    content: list[_TextPart] = Field(min_length=1)
    reasoning_content: str | None = None
    refusal: str | None = None


class _Choice(_ProviderModel):
    index: int = Field(default=0, ge=0)
    message: _Message
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls", "function_call"] | None = None
    logprobs: None = None


class _PartsChoice(_ProviderModel):
    index: int = Field(default=0, ge=0)
    message: _PartsMessage
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls", "function_call"] | None = None
    logprobs: None = None


class _PromptTokensDetails(_ProviderModel):
    cached_tokens: int = Field(ge=0)


class _LlamaCppTimings(_ProviderModel):
    cache_n: int = Field(ge=0)
    prompt_n: int = Field(ge=0)
    prompt_ms: float = Field(ge=0, allow_inf_nan=False)
    prompt_per_token_ms: float = Field(ge=0, allow_inf_nan=False)
    prompt_per_second: float = Field(ge=0, allow_inf_nan=False)
    predicted_n: int = Field(ge=0)
    predicted_ms: float = Field(ge=0, allow_inf_nan=False)
    predicted_per_token_ms: float = Field(ge=0, allow_inf_nan=False)
    predicted_per_second: float = Field(ge=0, allow_inf_nan=False)
    draft_n: int = Field(ge=0)
    draft_n_accepted: int = Field(ge=0)


class _Usage(_ProviderModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    prompt_tokens_details: _PromptTokensDetails | None = None


class _CompletionMetadata(_ProviderModel):
    id: str | None = None
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default=0, ge=0)
    model: str | None = None
    usage: _Usage | None = None
    timings: _LlamaCppTimings | None = None
    system_fingerprint: str | None = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None


class CompletionResponse(_CompletionMetadata):
    choices: list[_Choice] = Field(min_length=1, max_length=1)


class PartsCompletionResponse(_CompletionMetadata):
    choices: list[_PartsChoice] = Field(min_length=1, max_length=1)


def completion_content(response: dict[str, Any], configuration: EffectiveSettings) -> str:
    """Accept only the configured chat content schema, without field guessing."""
    try:
        if configuration.COMPLETION_CONTENT_FORMAT == "text_parts":
            parts_response = PartsCompletionResponse.model_validate(response)
            text = "".join(part.text for part in parts_response.choices[0].message.content)
        else:
            text = CompletionResponse.model_validate(response).choices[0].message.content
    except ValidationError:
        raise ValueError("Invalid completion provider response schema") from None
    if not text.strip():
        raise ValueError("Completion provider response is empty")
    return text


class HTTPClientService:
    """Perform concurrency-limited HTTP requests with retries."""

    def __init__(self, timeout: float | None = None, *, client: httpx.AsyncClient | None = None, configuration: EffectiveSettings | None = None) -> None:
        """Initialize the HTTP client.

        Args:
            timeout: Request timeout in seconds.
        """
        self.configuration = configuration if configuration is not None else config.snapshot_settings()
        effective_timeout = self.configuration.HTTPX_TIMEOUT if timeout is None else timeout
        if not math.isfinite(effective_timeout) or effective_timeout <= 0:
            raise ValueError("HTTP timeout must be finite and positive")
        self._total_request_timeout = effective_timeout
        self._client = client if client is not None else httpx.AsyncClient(timeout=effective_timeout, follow_redirects=False, trust_env=False)
        self._semaphore = asyncio.Semaphore(self.configuration.MAX_CONCURRENT_LLM_CALLS)
        self.request_count = 0
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retry_attempts": 0,
        }

        logger.info("HTTPClientService initialized", timeout=effective_timeout, concurrency_limit=self.configuration.MAX_CONCURRENT_LLM_CALLS)

    async def aclose(self) -> None:
        """Close the underlying HTTP client and release resources."""
        await self._client.aclose()
        logger.debug("HTTPClientService closed")

    @bounded_request
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
        effective_max_retries = max_retries if max_retries is not None else self.configuration.LLM_RETRY_ATTEMPTS
        if type(effective_max_retries) is not int or effective_max_retries < 1:
            raise ValueError("HTTP attempts must be a positive integer")
        # Use the same exact UTF-8 serialization admitted at the completion seam.
        # Detach before waiting so retries cannot observe caller mutation.
        serialized_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        async with self._semaphore:
            self._stats["total_requests"] += 1
            self.request_count += 1

            effective_headers = {"Content-Type": "application/json", **(headers or {})}


            last_exception: Exception | None = None

            for attempt in range(effective_max_retries):
                try:
                    logger.debug("HTTP POST", attempt=attempt + 1, max_attempts=effective_max_retries)

                    response = await self._client.post(url, content=serialized_payload, headers=effective_headers, follow_redirects=False)
                    response.raise_for_status()

                    self._stats["successful_requests"] += 1
                    logger.debug(f"HTTP POST successful: {response.status_code}")
                    return response

                except httpx.TimeoutException as e:
                    last_exception = e
                    logger.warning("HTTP timeout", attempt=attempt + 1)

                except httpx.HTTPStatusError as e:
                    last_exception = e
                    status_code = e.response.status_code if e.response else 0
                    logger.warning("HTTP status error", attempt=attempt + 1, status_code=status_code, response_length=len(e.response.content))

                    # Don't retry on client errors (except 429 rate limit)
                    if 400 <= status_code < 500 and status_code != 429:
                        logger.error(f"Non-retryable client error {status_code}, aborting")
                        break

                except httpx.RequestError as e:
                    last_exception = e
                    logger.warning("HTTP request error", attempt=attempt + 1, error_type=type(e).__name__)

                except Exception as e:
                    last_exception = e
                    logger.error("Unexpected HTTP error", attempt=attempt + 1, error_type=type(e).__name__)

                # Apply retry delay if not the last attempt
                if attempt < effective_max_retries - 1:
                    delay = self.configuration.LLM_RETRY_DELAY_SECONDS * (2**attempt)
                    logger.info(f"Retrying in {delay:.2f}s due to: {type(last_exception).__name__}")
                    await asyncio.sleep(delay)
                    self._stats["retry_attempts"] += 1

            # All retries failed
            self._stats["failed_requests"] += 1
            logger.error("HTTP POST failed", max_attempts=effective_max_retries, error_type=type(last_exception).__name__)

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

        logger.debug("Requesting embedding", text_length=len(text), text_sha256=hashlib.sha256(text.encode()).hexdigest())
        key = self._http_client.configuration.EMBEDDING_API_KEY.get_secret_value()
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        response = await self._http_client.post_json(f"{self._http_client.configuration.EMBEDDING_API_BASE}/api/embeddings", payload, headers)
        try:
            return EmbeddingResponse.model_validate_json(response.content).model_dump()
        except ValidationError:
            raise ValueError("Invalid embedding provider response schema") from None


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
        allowed_options = {"top_p", "frequency_penalty", "presence_penalty", "stop", "seed", "response_format", "tools", "tool_choice"}
        if kwargs.keys() - allowed_options:
            raise ValueError("Unsupported completion options cannot override the request contract")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("Completion budget must be a positive integer")
        if not messages or any(set(message) != {"role", "content"} or message["role"] not in {"system", "user", "assistant"} or not isinstance(message["content"], str) for message in messages):
            raise ValueError("Completion messages require text and an explicit role")
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": self._http_client.configuration.LLM_TOP_P,
            "max_tokens": max_tokens,
            "stream": False,
            **kwargs,
        }

        # This is a configured encoding budget, not a claim about native chat templates.
        configuration = self._http_client.configuration
        encoder = tiktoken.get_encoding(configuration.TIKTOKEN_DEFAULT_ENCODING)
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        request_tokens = len(encoder.encode(serialized, disallowed_special=()))
        framing_tokens = len(messages) * configuration.REQUEST_MESSAGE_OVERHEAD_TOKENS + configuration.REQUEST_REPLY_OVERHEAD_TOKENS
        if request_tokens + framing_tokens + max_tokens > configuration.MAX_CONTEXT_TOKENS:
            raise ValueError("Serialized request plus framing and completion exceeds context budget")
        # Detach mutable caller structures before semaphore waiting or retrying.
        payload = json.loads(serialized)

        headers = {
            "Authorization": f"Bearer {self._http_client.configuration.OPENAI_API_KEY.get_secret_value()}",
            "Content-Type": "application/json",
        }

        logger.debug("Requesting completion", message_count=len(messages))
        response = await self._http_client.post_json(f"{self._http_client.configuration.OPENAI_API_BASE}/chat/completions", payload, headers)
        response_data = response.json()
        completion_content(response_data, self._http_client.configuration)
        return response_data

    # Streaming completion removed; use get_completion() only.
