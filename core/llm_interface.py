# core/llm_interface.py
# llm_interface.py
"""
DEPRECATED: This module is deprecated as part of Phase 3 architectural improvements.

The original monolithic LLMService has been refactored into separate, focused services
to improve maintainability, testability, and separation of concerns.

MIGRATION GUIDE:
- For new code, use: from core.llm_interface_refactored import get_llm_service
- The API remains compatible, but new features will only be added to the refactored version
- HTTP client concerns: core.http_client_service
- Text processing concerns: core.text_processing_service
- Service interfaces: core.llm_service_interfaces

The refactored architecture provides:
- Better separation of HTTP client and text processing concerns
- Proper abstraction layers with dependency injection
- Improved testability with mock service support
- Cleaner service composition and lifecycle management

ORIGINAL DESCRIPTION:
Handles all direct interactions with Large Language Models (LLMs)
and embedding models (via Ollama). Includes functions for API calls,
response cleaning, and embedding generation with caching.
Also includes asynchronous versions of API call functions.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright 2025 Dennis Lewis
"""

# Standard library imports
import asyncio
import functools
import json
import os
import re
import tempfile
from contextlib import contextmanager

# Type hints
from typing import Any

import httpx

# Third-party imports
import numpy as np
import structlog
import tiktoken
from async_lru import alru_cache

# Local imports
import config

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


# Token parameter handling
def _completion_token_param(api_base: str) -> str:
    """Return the token count parameter expected by the provider."""
    if "api.openai.com" in api_base or "api.anthropic.com" in api_base:
        return "max_completion_tokens"
    return "max_tokens"


# --- Tokenizer Cache and Utility Functions (Module Level) ---
_tokenizer_cache: dict[str, tiktoken.Encoding] = {}


@functools.lru_cache(maxsize=config.TOKENIZER_CACHE_SIZE)
def _get_tokenizer(model_name: str) -> tiktoken.Encoding | None:
    """
    Gets a tiktoken encoder for the given model name, with caching.
    Tries model-specific encoding, then a default, then returns None.
    """
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    try:
        try:
            encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.debug(
                f"No direct tiktoken encoding for '{model_name}'. Using default '{config.TIKTOKEN_DEFAULT_ENCODING}'."
            )
            encoder = tiktoken.get_encoding(config.TIKTOKEN_DEFAULT_ENCODING)

        _tokenizer_cache[model_name] = encoder
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


def count_tokens(text: str, model_name: str) -> int:
    """
    Counts the number of tokens in a string for a given model.
    Uses tiktoken with caching and fallbacks.
    """
    if not text:
        return 0

    encoder = _get_tokenizer(model_name)

    if encoder:
        return len(encoder.encode(text, allowed_special="all"))
    else:
        char_count = len(text)
        token_estimate = int(char_count / config.FALLBACK_CHARS_PER_TOKEN)
        logger.warning(
            f"count_tokens: Failed to get tokenizer for '{model_name}'. "
            f"Falling back to character-based estimate: {char_count} chars -> ~{token_estimate} tokens."
        )
        return token_estimate


def truncate_text_by_tokens(
    text: str,
    model_name: str,
    max_tokens: int,
    truncation_marker: str = "\n... (truncated)",
) -> str:
    """
    Truncates text to a maximum number of tokens for a given model.
    Adds a truncation marker if truncation occurs.
    """
    if not text:
        return ""

    encoder = _get_tokenizer(model_name)

    if not encoder:
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

    marker_tokens_len = 0
    if truncation_marker:
        marker_tokens_len = len(
            encoder.encode(truncation_marker, allowed_special="all")
        )

    content_tokens_to_keep = max_tokens - marker_tokens_len
    effective_truncation_marker = truncation_marker

    if content_tokens_to_keep < 0:
        logger.debug(
            f"Truncation marker ('{truncation_marker}' -> {marker_tokens_len} tokens) is longer than max_tokens ({max_tokens}). Using empty marker."
        )
        content_tokens_to_keep = max_tokens
        effective_truncation_marker = ""

    truncated_content_tokens = tokens[:content_tokens_to_keep]

    if not truncated_content_tokens and max_tokens > 0 and tokens:
        logger.debug(
            "Truncated content to 0 tokens due to marker length. Attempting to"
            " keep 1 token of content."
        )
        truncated_content_tokens = tokens[:1]
        effective_truncation_marker = ""

    try:
        decoded_text = encoder.decode(truncated_content_tokens)
        return decoded_text + effective_truncation_marker
    except Exception as e:
        logger.error(
            f"Error decoding truncated tokens for model '{model_name}': {e}. Falling back to simpler char-based truncation.",
            exc_info=True,
        )
        avg_chars_per_token = (
            len(text) / len(tokens)
            if len(tokens) > 0
            else config.FALLBACK_CHARS_PER_TOKEN
        )
        estimated_char_limit_for_content = int(
            content_tokens_to_keep * avg_chars_per_token
        )
        return text[:estimated_char_limit_for_content] + effective_truncation_marker


class LLMService:
    """Utility class for interacting with LLM and embedding endpoints."""

    def __init__(self, timeout: float = config.HTTPX_TIMEOUT):
        # Use a single async client for all requests to reuse connections
        self._client = httpx.AsyncClient(timeout=timeout)
        # Add a semaphore to limit concurrent requests
        self._semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_LLM_CALLS)
        self.request_count = 0
        logger.info(
            f"LLMService initialized with a concurrency limit of {config.MAX_CONCURRENT_LLM_CALLS}."
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    def _validate_embedding(
        self,
        embedding_list: list[float | int],
        expected_dim: int,
        dtype: np.dtype,
    ) -> np.ndarray | None:
        """Helper to validate and convert a list to a 1D numpy embedding."""
        try:
            embedding = np.array(embedding_list).astype(dtype)
            if embedding.ndim > 1:
                logger.warning(
                    f"Embedding from source had unexpected ndim > 1: {embedding.ndim}. Flattening."
                )
                embedding = embedding.flatten()
            if embedding.shape == (expected_dim,):
                logger.debug(
                    f"Embedding validated successfully. Shape: {embedding.shape}, Dtype: {embedding.dtype}"
                )
                return embedding
            logger.error(
                f"Embedding dimension mismatch: Expected ({expected_dim},), Got {embedding.shape}. Original list length: {len(embedding_list)}"
            )
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to convert embedding list to numpy array: {e}")
        return None

    @alru_cache(maxsize=config.EMBEDDING_CACHE_SIZE)
    async def async_get_embedding(self, text: str) -> np.ndarray | None:
        """
        Asynchronously retrieves an embedding for the given text from Ollama with retry logic.
        """
        async with self._semaphore:
            if not text or not isinstance(text, str) or not text.strip():
                logger.warning(
                    "async_get_embedding: empty or invalid text provided. Returning None."
                )
                return None

            payload = {"model": config.EMBEDDING_MODEL, "prompt": text.strip()}
            logger.debug(
                f"Async Embedding req to Ollama for model '{config.EMBEDDING_MODEL}': '{text[:80].replace(chr(10), ' ')}...'"
            )

            last_exception: Exception | None = None
            for attempt in range(config.LLM_RETRY_ATTEMPTS):
                api_response: httpx.Response | None = None
                try:
                    self.request_count += 1
                    api_response = await self._client.post(
                        f"{config.OLLAMA_EMBED_URL}/api/embeddings", json=payload
                    )
                    api_response.raise_for_status()
                    data = api_response.json()

                    primary_key = "embedding"
                    if primary_key in data and isinstance(data[primary_key], list):
                        embedding = self._validate_embedding(
                            data[primary_key],
                            config.EXPECTED_EMBEDDING_DIM,
                            config.EMBEDDING_DTYPE,
                        )
                        if embedding is not None:
                            return embedding
                    else:
                        logger.warning(
                            f"Ollama (Attempt {attempt + 1}): Primary embedding key '{primary_key}' not found or not a list. Data: {data}"
                        )
                        for key, value in data.items():
                            if isinstance(value, list) and all(
                                isinstance(item, (float, int)) for item in value
                            ):
                                embedding = self._validate_embedding(
                                    value,
                                    config.EXPECTED_EMBEDDING_DIM,
                                    config.EMBEDDING_DTYPE,
                                )
                                if embedding is not None:
                                    logger.info(
                                        f"Ollama (Attempt {attempt + 1}): Found embedding using fallback key '{key}'."
                                    )
                                    return embedding

                    logger.error(
                        f"Ollama (Attempt {attempt + 1}): Embedding extraction failed. No suitable embedding list found in response: {data}"
                    )
                    last_exception = ValueError(
                        "No suitable embedding list found in Ollama response after parsing."
                    )

                except httpx.TimeoutException as e_timeout:
                    last_exception = e_timeout
                    logger.warning(
                        f"Ollama Embedding (Attempt {attempt + 1}/{config.LLM_RETRY_ATTEMPTS}): Request timed out: {e_timeout}"
                    )
                except httpx.HTTPStatusError as e_status:
                    last_exception = e_status
                    error_message_detail = f"HTTP status {e_status.response.status_code}: {e_status}. Body: {e_status.response.text[:200]}"
                    logger.warning(
                        f"Ollama Embedding (Attempt {attempt + 1}/{config.LLM_RETRY_ATTEMPTS}): {error_message_detail}"
                    )
                    if 400 <= e_status.response.status_code < 500:
                        logger.error(
                            f"Ollama Embedding: Client-side error {e_status.response.status_code}. Aborting retries."
                        )
                        break
                except httpx.RequestError as e_req:
                    last_exception = e_req
                    logger.warning(
                        f"Ollama Embedding (Attempt {attempt + 1}/{config.LLM_RETRY_ATTEMPTS}): Request error: {e_req}"
                    )
                except json.JSONDecodeError as e_json:
                    last_exception = e_json
                    response_text_snippet = (
                        api_response.text[:200]
                        if api_response and hasattr(api_response, "text")
                        else "N/A"
                    )
                    logger.warning(
                        f"Ollama Embedding (Attempt {attempt + 1}/{config.LLM_RETRY_ATTEMPTS}): Failed to decode JSON response: {e_json}. "
                        f"Response text: {response_text_snippet}"
                    )
                except Exception as e_exc:
                    last_exception = e_exc
                    logger.warning(
                        f"Ollama Embedding (Attempt {attempt + 1}/{config.LLM_RETRY_ATTEMPTS}): Unexpected error: {e_exc}",
                        exc_info=True,
                    )

                if attempt < config.LLM_RETRY_ATTEMPTS - 1:
                    delay = config.LLM_RETRY_DELAY_SECONDS * (2**attempt)
                    retry_reason = (
                        type(last_exception).__name__
                        if last_exception
                        else "Unknown reason"
                    )
                    logger.info(
                        f"Ollama Embedding: Retrying in {delay:.2f} seconds due to: {retry_reason}."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Ollama Embedding: All {config.LLM_RETRY_ATTEMPTS} retry attempts failed. Last error: {last_exception}"
                    )
                    return None
            return None

    async def async_call_llm(
        self,
        model_name: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        allow_fallback: bool = False,
        stream_to_disk: bool = False,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        auto_clean_response: bool = True,
    ) -> tuple[str, dict[str, int] | None]:
        """
        Asynchronously calls the LLM (OpenAI-compatible API) with retry and optional model fallback.
        Returns the LLM's text response as a string (optionally cleaned), and a dictionary containing token usage.
        """
        async with self._semaphore:
            if not model_name:
                logger.error("async_call_llm: model_name is required.")
                return "", None
            if not prompt or not isinstance(prompt, str) or not prompt.strip():
                logger.error("async_call_llm: empty or invalid prompt.")
                return "", None

            prompt_token_count = count_tokens(prompt, model_name)
            effective_max_output_tokens = (
                max_tokens if max_tokens is not None else config.MAX_GENERATION_TOKENS
            )
            effective_temperature = (
                temperature if temperature is not None else config.Temperatures.DEFAULT
            )

            headers = {
                "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }

            current_model_to_try = model_name
            is_fallback_attempt = False
            current_usage_data: dict[str, int] | None = None
            final_text_response = ""

            for attempt_num_overall in range(2):
                if is_fallback_attempt:
                    if not allow_fallback or not config.FALLBACK_GENERATION_MODEL:
                        logger.warning(
                            f"Primary model '{model_name}' failed. Fallback not allowed or no fallback model configured. Aborting call."
                        )
                        if auto_clean_response:
                            final_text_response = self.clean_model_response(
                                final_text_response
                            )
                        return final_text_response, current_usage_data
                    current_model_to_try = config.FALLBACK_GENERATION_MODEL
                    logger.info(
                        f"Primary model '{model_name}' failed. Attempting fallback with '{current_model_to_try}'."
                    )
                    prompt_token_count = count_tokens(prompt, current_model_to_try)
                    current_usage_data = None

                token_param_name = _completion_token_param(config.OPENAI_API_BASE)
                payload: dict[str, Any] = {
                    "model": current_model_to_try,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": effective_temperature,
                    "top_p": config.LLM_TOP_P,
                    token_param_name: effective_max_output_tokens,
                }

                if frequency_penalty is not None:
                    payload["frequency_penalty"] = frequency_penalty
                if presence_penalty is not None:
                    payload["presence_penalty"] = presence_penalty

                last_exception_for_current_model: Exception | None = None

                for retry_attempt in range(config.LLM_RETRY_ATTEMPTS):
                    penalty_log_str = ""
                    if frequency_penalty is not None:
                        penalty_log_str += f" FreqPen: {frequency_penalty}"
                    if presence_penalty is not None:
                        penalty_log_str += f" PresPen: {presence_penalty}"

                    logger.debug(
                        f"Async Calling LLM '{current_model_to_try}' (Attempt {retry_attempt + 1}/{config.LLM_RETRY_ATTEMPTS}, OverallAttempt: {attempt_num_overall + 1}). "
                        f"StreamToDisk: {stream_to_disk}. Prompt tokens (est.): {prompt_token_count}. Max output tokens: {effective_max_output_tokens}. "
                        f"Temp: {effective_temperature}, TopP: {config.LLM_TOP_P}{penalty_log_str}"
                    )
                    api_response_obj: httpx.Response | None = None

                    try:
                        if stream_to_disk:
                            payload["stream"] = True

                            # Use secure context manager for temporary file handling
                            with secure_temp_file(
                                suffix=".llmstream.txt", text=True
                            ) as temp_file_path:
                                accumulated_stream_content = ""
                                stream_usage_data: dict[str, int] | None = None

                                try:
                                    self.request_count += 1
                                    async with self._client.stream(
                                        "POST",
                                        f"{config.OPENAI_API_BASE}/chat/completions",
                                        json=payload,
                                        headers=headers,
                                    ) as response_stream:
                                        response_stream.raise_for_status()

                                        with open(
                                            temp_file_path, "w", encoding="utf-8"
                                        ) as tmp_f_write:
                                            async for (
                                                line
                                            ) in response_stream.aiter_lines():
                                                if line.startswith("data: "):
                                                    data_json_str = line[
                                                        len("data: ") :
                                                    ].strip()
                                                    if data_json_str == "[DONE]":
                                                        break
                                                    try:
                                                        chunk_data = json.loads(
                                                            data_json_str
                                                        )
                                                        if chunk_data.get("choices"):
                                                            delta = chunk_data[
                                                                "choices"
                                                            ][0].get("delta", {})
                                                            content_piece = delta.get(
                                                                "content"
                                                            )
                                                            if content_piece:
                                                                accumulated_stream_content += content_piece
                                                                tmp_f_write.write(
                                                                    content_piece
                                                                )

                                                            if (
                                                                chunk_data["choices"][
                                                                    0
                                                                ].get("finish_reason")
                                                                is not None
                                                            ):
                                                                potential_usage = (
                                                                    chunk_data.get(
                                                                        "usage"
                                                                    )
                                                                )
                                                                if (
                                                                    not potential_usage
                                                                    and chunk_data.get(
                                                                        "x_groq"
                                                                    )
                                                                    and chunk_data[
                                                                        "x_groq"
                                                                    ].get("usage")
                                                                ):
                                                                    potential_usage = (
                                                                        chunk_data[
                                                                            "x_groq"
                                                                        ]["usage"]
                                                                    )
                                                                if (
                                                                    potential_usage
                                                                    and isinstance(
                                                                        potential_usage,
                                                                        dict,
                                                                    )
                                                                ):
                                                                    stream_usage_data = potential_usage
                                                    except json.JSONDecodeError:
                                                        logger.warning(
                                                            f"Async LLM Stream: Could not decode JSON from line: {line}"
                                                        )

                                    final_text_response = accumulated_stream_content
                                    current_usage_data = stream_usage_data
                                    if auto_clean_response:
                                        final_text_response = self.clean_model_response(
                                            final_text_response
                                        )
                                    return final_text_response, current_usage_data

                                except Exception as stream_err:
                                    last_exception_for_current_model = stream_err
                                    logger.warning(
                                        f"Async LLM Stream ('{current_model_to_try}' Attempt {retry_attempt + 1}): Error during stream: {stream_err}",
                                        exc_info=True,
                                    )
                                    # If there was an error, try to read partial content from temp file
                                    try:
                                        with open(
                                            temp_file_path, encoding="utf-8"
                                        ) as f_read_partial:
                                            final_text_response = f_read_partial.read()
                                    except Exception as read_error:
                                        logger.warning(
                                            f"Could not read partial content from temp file: {read_error}"
                                        )
                                    # Context manager will handle cleanup automatically
                        else:
                            payload["stream"] = False
                            self.request_count += 1
                            api_response_obj = await self._client.post(
                                f"{config.OPENAI_API_BASE}/chat/completions",
                                json=payload,
                                headers=headers,
                            )
                            api_response_obj.raise_for_status()
                            response_data = api_response_obj.json()

                            raw_text_non_stream = ""
                            if (
                                response_data.get("choices")
                                and len(response_data["choices"]) > 0
                            ):
                                message = response_data["choices"][0].get("message")
                                if message and message.get("content"):
                                    raw_text_non_stream = message["content"]
                            else:
                                logger.error(
                                    f"Async LLM ('{current_model_to_try}') Invalid response structure - missing choices/content despite 200 OK: {response_data}"
                                )

                            final_text_response = raw_text_non_stream
                            current_usage_data = response_data.get("usage")
                            if auto_clean_response:
                                final_text_response = self.clean_model_response(
                                    final_text_response
                                )
                            return final_text_response, current_usage_data

                    except httpx.TimeoutException as e_timeout:
                        last_exception_for_current_model = e_timeout
                        logger.warning(
                            f"Async LLM ('{current_model_to_try}' Attempt {retry_attempt + 1}): API request timed out: {e_timeout}"
                        )
                    except httpx.HTTPStatusError as e_status:
                        last_exception_for_current_model = e_status
                        response_text_snippet = (
                            e_status.response.text[:200] if e_status.response else "N/A"
                        )
                        error_message_detail = f"API HTTP status error: {e_status}. Status: {e_status.response.status_code if e_status.response else 'N/A'}, Body: {response_text_snippet}"
                        logger.warning(
                            f"Async LLM ('{current_model_to_try}' Attempt {retry_attempt + 1}): {error_message_detail}"
                        )
                        if (
                            e_status.response
                            and 400 <= e_status.response.status_code < 500
                        ):
                            if e_status.response.status_code == 429:
                                logger.warning(
                                    f"Async LLM ('{current_model_to_try}' Attempt {retry_attempt + 1}): Rate limit error (429). Will retry after delay."
                                )
                                # Do not break, allow the retry loop to continue
                            else:
                                if (
                                    e_status.response.status_code == 400
                                    and "context_length_exceeded"
                                    in response_text_snippet.lower()
                                ):
                                    logger.error(
                                        f"Async LLM ('{current_model_to_try}'): Context length exceeded. Prompt tokens (est.): {prompt_token_count}. "
                                        f"Aborting retries for this model."
                                    )
                                else:
                                    logger.error(
                                        f"Async LLM ('{current_model_to_try}'): Non-retryable client-side error {e_status.response.status_code}. Aborting retries for this model."
                                    )
                                break
                    except httpx.RequestError as e_req:
                        last_exception_for_current_model = e_req
                        logger.warning(
                            f"Async LLM ('{current_model_to_try}' Attempt {retry_attempt + 1}): API request error (network/connection): {e_req}"
                        )
                    except json.JSONDecodeError as e_json:
                        last_exception_for_current_model = e_json
                        response_text_snippet = ""
                        if api_response_obj and hasattr(api_response_obj, "text"):
                            response_text_snippet = api_response_obj.text[:200]
                        logger.warning(
                            f"Async LLM ('{current_model_to_try}' Attempt {retry_attempt + 1}): Failed to decode API JSON response: {e_json}. "
                            f"Response text snippet (if available): {response_text_snippet}"
                        )
                    except Exception as e_exc:
                        last_exception_for_current_model = e_exc
                        logger.warning(
                            f"Async LLM ('{current_model_to_try}' Attempt {retry_attempt + 1}): Unexpected error: {e_exc}",
                            exc_info=True,
                        )
                    finally:
                        # Temp file cleanup is now handled by the secure_temp_file context manager
                        pass

                    if (
                        retry_attempt < config.LLM_RETRY_ATTEMPTS - 1
                        and last_exception_for_current_model is not None
                    ):
                        delay = config.LLM_RETRY_DELAY_SECONDS * (2**retry_attempt)
                        retry_reason = (
                            type(last_exception_for_current_model).__name__
                            if last_exception_for_current_model
                            else "Unknown reason"
                        )
                        logger.info(
                            f"Async LLM ('{current_model_to_try}'): Retrying in {delay:.2f} seconds due to: {retry_reason}."
                        )
                        await asyncio.sleep(delay)
                    elif last_exception_for_current_model is not None:
                        logger.error(
                            f"Async LLM ('{current_model_to_try}'): All {config.LLM_RETRY_ATTEMPTS} retries failed for this model. Last error: {last_exception_for_current_model}"
                        )

                if last_exception_for_current_model is None:
                    break

                is_fallback_attempt = True

                if (
                    attempt_num_overall == 0
                    and isinstance(
                        last_exception_for_current_model, httpx.HTTPStatusError
                    )
                    and last_exception_for_current_model.response
                    and 400
                    <= last_exception_for_current_model.response.status_code
                    < 500
                    and last_exception_for_current_model.response.status_code != 429
                ):
                    logger.error(
                        f"Async LLM: Primary model '{model_name}' failed with non-429 client error. Checking fallback conditions."
                    )

            logger.error(
                f"Async LLM: Call failed for '{model_name}' after all primary and potential fallback attempts. Returning last captured text ('{final_text_response[:50]}...') and usage."
            )
            # Final temp file cleanup is now handled by the secure_temp_file context manager

            if auto_clean_response:
                final_text_response = self.clean_model_response(final_text_response)

            return final_text_response, current_usage_data

    def clean_model_response(self, text: str) -> str:
        """Cleans common artifacts from LLM text responses, including content within <think> tags and normalizes newlines."""
        if not isinstance(text, str):
            logger.warning(
                f"clean_model_response received non-string input: {type(text)}. Returning empty string."
            )
            return ""

        original_length = len(text)
        cleaned_text = text

        text_before_think_removal = cleaned_text
        think_tags_to_remove = [
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
        for tag_name in think_tags_to_remove:
            block_pattern = re.compile(
                rf"<\s*{tag_name}\s*>.*?<\s*/\s*{tag_name}\s*>",
                flags=re.DOTALL | re.IGNORECASE,
            )
            cleaned_text = block_pattern.sub("", cleaned_text)

            self_closing_pattern = re.compile(
                rf"<\s*{tag_name}\s*/\s*>", flags=re.IGNORECASE
            )
            cleaned_text = self_closing_pattern.sub("", cleaned_text)

            lone_opening_pattern = re.compile(
                rf"<\s*{tag_name}\s*>", flags=re.IGNORECASE
            )
            cleaned_text = lone_opening_pattern.sub("", cleaned_text)

            lone_closing_pattern = re.compile(
                rf"<\s*/\s*{tag_name}\s*>", flags=re.IGNORECASE
            )
            cleaned_text = lone_closing_pattern.sub("", cleaned_text)

        if len(cleaned_text) < len(text_before_think_removal):
            logger.debug(
                f"clean_model_response: Removed content associated with <think>/similar tags. Length before: {len(text_before_think_removal)}, after: {len(cleaned_text)}."
            )

        cleaned_text = re.sub(
            r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)\s*```",
            r"\1",
            cleaned_text,
            flags=re.DOTALL,
        )

        cleaned_text = re.sub(
            r"^\s*Chapter \d+\s*[:\-—]?\s*(.*?)\s*$",
            r"\1",
            cleaned_text,
            flags=re.MULTILINE | re.IGNORECASE,
        ).strip()

        common_phrases_patterns = [
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
        for pattern_str in common_phrases_patterns:
            if pattern_str.startswith("^"):
                while True:
                    new_text = re.sub(
                        pattern_str,
                        "",
                        cleaned_text,
                        count=1,
                        flags=re.IGNORECASE | re.MULTILINE,
                    ).strip()
                    if new_text == cleaned_text:
                        break
                    cleaned_text = new_text
            else:
                cleaned_text = re.sub(
                    pattern_str,
                    "",
                    cleaned_text,
                    count=1,
                    flags=re.IGNORECASE | re.MULTILINE,
                ).strip()

        final_text = cleaned_text.strip()
        final_text = re.sub(r"\n\s*\n(\s*\n)+", "\n\n", final_text)
        final_text = re.sub(r"\n{3,}", "\n\n", final_text)

        if original_length > 0 and len(final_text) < original_length:
            reduction_percentage = (
                (original_length - len(final_text)) / original_length
            ) * 100
            if reduction_percentage > 0.5:
                logger.debug(
                    f"Cleaning reduced text length from {original_length} to {len(final_text)} ({reduction_percentage:.1f}% reduction)."
                )

        return final_text


# Instantiate the service for other modules to import and use
llm_service = LLMService()
