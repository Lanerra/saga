# llm_interface.py
"""
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
import functools
import logging
import json
import re
import asyncio
import time
import tempfile
import os

# Third-party imports
import numpy as np
import httpx
from async_lru import alru_cache
import tiktoken

# Type hints
from typing import List, Optional, Dict, Any, Union, Tuple

# Local imports
import config

logger = logging.getLogger(__name__)

# --- Tokenizer Cache ---
_tokenizer_cache: Dict[str, tiktoken.Encoding] = {}

@functools.lru_cache(maxsize=config.TOKENIZER_CACHE_SIZE)
def _get_tokenizer(model_name: str) -> Optional[tiktoken.Encoding]:
    """
    Gets a tiktoken encoder for the given model name, with caching.
    Tries model-specific encoding, then a default, then returns None.
    """
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    try:
        try:
            # First, try to get the encoder directly for the model name.
            # This works for models like "gpt-4", "gpt-3.5-turbo".
            encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # If model_name is not directly recognized (e.g., custom or local model alias),
            # fall back to a default encoding suitable for many OpenAI-compatible models.
            logger.debug(f"No direct tiktoken encoding for '{model_name}'. Using default '{config.TIKTOKEN_DEFAULT_ENCODING}'.")
            encoder = tiktoken.get_encoding(config.TIKTOKEN_DEFAULT_ENCODING)

        _tokenizer_cache[model_name] = encoder
        logger.debug(f"Tokenizer for model '{model_name}' (using actual encoder '{encoder.name}') found and cached.")
        return encoder
    except KeyError: # If the default encoding itself is not found (highly unlikely for cl100k_base)
        logger.error(
            f"Default tiktoken encoding '{config.TIKTOKEN_DEFAULT_ENCODING}' also not found. "
            f"Token counting will fall back to character-based heuristic for '{model_name}'."
        )
        return None
    except Exception as e: # Catch any other unexpected errors during tokenizer loading
        logger.error(f"Unexpected error getting tokenizer for '{model_name}': {e}", exc_info=True)
        return None

def count_tokens(text: str, model_name: str) -> int:
    """
    Counts the number of tokens in a string for a given model.
    Uses tiktoken with caching and fallbacks.
    `model_name` here is used to select the appropriate tokenizer.
    """
    if not text: # An empty string has zero tokens
        return 0
    
    # Use the model_name provided to get the appropriate tokenizer.
    # _get_tokenizer handles fallbacks if model_name isn't directly known by tiktoken.
    encoder = _get_tokenizer(model_name) 
    
    if encoder:
        return len(encoder.encode(text, allowed_special="all")) # "all" ensures special tokens are counted if present
    else:
        # Fallback heuristic if no tokenizer could be loaded
        char_count = len(text)
        # Estimate tokens based on average characters per token. This is a rough guide.
        token_estimate = int(char_count / config.FALLBACK_CHARS_PER_TOKEN)
        logger.warning(
            f"count_tokens: Failed to get tokenizer for '{model_name}'. "
            f"Falling back to character-based estimate: {char_count} chars -> ~{token_estimate} tokens."
        )
        return token_estimate

def truncate_text_by_tokens(text: str, model_name: str, max_tokens: int, truncation_marker: str = "\n... (truncated)") -> str:
    """
    Truncates text to a maximum number of tokens for a given model.
    Adds a truncation marker if truncation occurs.
    `model_name` helps select the tokenizer.
    """
    if not text: # No text to truncate
        return ""

    encoder = _get_tokenizer(model_name)

    if not encoder:
        # Fallback to character-based truncation if tokenizer fails
        max_chars = int(max_tokens * config.FALLBACK_CHARS_PER_TOKEN) # Estimate max characters
        logger.warning(
            f"truncate_text_by_tokens: Failed to get tokenizer for '{model_name}'. "
            f"Falling back to character-based truncation: {max_tokens} tokens -> ~{max_chars} chars."
        )
        if len(text) > max_chars:
            # Ensure marker fits
            effective_max_chars = max_chars - len(truncation_marker)
            if effective_max_chars < 0: effective_max_chars = 0 # Avoid negative slice
            return text[:effective_max_chars] + truncation_marker
        return text # Text is already within estimated char limit

    tokens = encoder.encode(text, allowed_special="all")
    if len(tokens) <= max_tokens:
        return text # No truncation needed
    
    # Account for truncation marker tokens
    marker_tokens_len = 0
    if truncation_marker: # Only calculate if marker is non-empty
        marker_tokens_len = len(encoder.encode(truncation_marker, allowed_special="all"))
    
    content_tokens_to_keep = max_tokens - marker_tokens_len
    effective_truncation_marker = truncation_marker

    # If marker is too long for the max_tokens budget, we might need to shorten or omit it
    if content_tokens_to_keep < 0:
        logger.debug(f"Truncation marker ('{truncation_marker}' -> {marker_tokens_len} tokens) is longer than max_tokens ({max_tokens}). Using empty marker.")
        content_tokens_to_keep = max_tokens # Use all tokens for content, no marker
        effective_truncation_marker = ""       # This means marker was longer than max_tokens

    truncated_content_tokens = tokens[:content_tokens_to_keep]
    
    # Handle edge case: if content_tokens_to_keep is 0 but max_tokens > 0 (marker took all space)
    # It's better to return a tiny bit of content than just the marker, or an empty string if even that's not possible.
    if not truncated_content_tokens and max_tokens > 0 and tokens: # If original tokens existed
         logger.debug(f"Truncated content to 0 tokens due to marker length. Attempting to keep 1 token of content.")
         truncated_content_tokens = tokens[:1] # Keep at least one token of content if possible
         effective_truncation_marker = "" # And sacrifice the marker

    try:
        decoded_text = encoder.decode(truncated_content_tokens)
        return decoded_text + effective_truncation_marker
    except Exception as e:
        # Extremely rare, but if decoding fails, fallback to a simpler char-based slice
        logger.error(f"Error decoding truncated tokens for model '{model_name}': {e}. Falling back to simpler char-based truncation.", exc_info=True)
        avg_chars_per_token = len(text) / len(tokens) if len(tokens) > 0 else config.FALLBACK_CHARS_PER_TOKEN
        estimated_char_limit_for_content = int(content_tokens_to_keep * avg_chars_per_token)
        return text[:estimated_char_limit_for_content] + effective_truncation_marker


def _validate_embedding(embedding_list: List[Union[float, int]], expected_dim: int, dtype: np.dtype) -> Optional[np.ndarray]:
    """Helper to validate and convert a list to a 1D numpy embedding."""
    try:
        embedding = np.array(embedding_list).astype(dtype)
        if embedding.ndim > 1: # Should be 1D
            logger.warning(f"Embedding from source had unexpected ndim > 1: {embedding.ndim}. Flattening.")
            embedding = embedding.flatten()
        if embedding.shape == (expected_dim,):
            logger.debug(f"Embedding validated successfully. Shape: {embedding.shape}, Dtype: {embedding.dtype}") 
            return embedding
        logger.error(f"Embedding dimension mismatch: Expected ({expected_dim},), Got {embedding.shape}. Original list length: {len(embedding_list)}")
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to convert embedding list to numpy array: {e}")
    return None


@alru_cache(maxsize=config.EMBEDDING_CACHE_SIZE)
async def async_get_embedding(text: str) -> Optional[np.ndarray]:
    """
    Asynchronously retrieves an embedding for the given text from Ollama with retry logic.
    """
    if not text or not isinstance(text, str) or not text.strip(): # Check for empty or invalid text
        logger.warning("async_get_embedding: empty or invalid text provided. Returning None.")
        return None

    payload = {"model": config.EMBEDDING_MODEL, "prompt": text.strip()}
    logger.debug(f"Async Embedding req to Ollama for model '{config.EMBEDDING_MODEL}': '{text[:80].replace(chr(10), ' ')}...'")


    last_exception: Optional[Exception] = None 
    for attempt in range(config.LLM_RETRY_ATTEMPTS):
        api_response: Optional[httpx.Response] = None 
        try:
            async with httpx.AsyncClient(timeout=120.0) as client: 
                api_response = await client.post(f"{config.OLLAMA_EMBED_URL}/api/embeddings", json=payload)
                api_response.raise_for_status() 
                data = api_response.json()

            primary_key = "embedding" 
            if primary_key in data and isinstance(data[primary_key], list):
                embedding = _validate_embedding(data[primary_key], config.EXPECTED_EMBEDDING_DIM, config.EMBEDDING_DTYPE)
                if embedding is not None:
                    return embedding
            else: 
                logger.warning(f"Ollama (Attempt {attempt+1}): Primary embedding key '{primary_key}' not found or not a list. Data: {data}")
                for key, value in data.items(): 
                    if isinstance(value, list) and all(isinstance(item, (float, int)) for item in value):
                        embedding = _validate_embedding(value, config.EXPECTED_EMBEDDING_DIM, config.EMBEDDING_DTYPE)
                        if embedding is not None:
                            logger.info(f"Ollama (Attempt {attempt+1}): Found embedding using fallback key '{key}'.")
                            return embedding

            logger.error(f"Ollama (Attempt {attempt+1}): Embedding extraction failed. No suitable embedding list found in response: {data}")
            last_exception = ValueError("No suitable embedding list found in Ollama response after parsing.") 

        except httpx.TimeoutException as e_timeout:
            last_exception = e_timeout
            logger.warning(f"Ollama Embedding (Attempt {attempt+1}/{config.LLM_RETRY_ATTEMPTS}): Request timed out: {e_timeout}")
        except httpx.HTTPStatusError as e_status:
            last_exception = e_status
            error_message_detail = f"HTTP status {e_status.response.status_code}: {e_status}. Body: {e_status.response.text[:200]}" 
            logger.warning(
                f"Ollama Embedding (Attempt {attempt+1}/{config.LLM_RETRY_ATTEMPTS}): {error_message_detail}" 
            )
            if 400 <= e_status.response.status_code < 500: 
                logger.error(f"Ollama Embedding: Client-side error {e_status.response.status_code}. Aborting retries.")
                break 
        except httpx.RequestError as e_req: 
            last_exception = e_req
            logger.warning(f"Ollama Embedding (Attempt {attempt+1}/{config.LLM_RETRY_ATTEMPTS}): Request error: {e_req}")
        except json.JSONDecodeError as e_json:
            last_exception = e_json
            response_text_snippet = api_response.text[:200] if api_response and hasattr(api_response, 'text') else 'N/A'
            logger.warning(
                 f"Ollama Embedding (Attempt {attempt+1}/{config.LLM_RETRY_ATTEMPTS}): Failed to decode JSON response: {e_json}. "
                 f"Response text: {response_text_snippet}"
            )
        except Exception as e_exc: 
            last_exception = e_exc
            logger.warning(f"Ollama Embedding (Attempt {attempt+1}/{config.LLM_RETRY_ATTEMPTS}): Unexpected error: {e_exc}", exc_info=True)

        if attempt < config.LLM_RETRY_ATTEMPTS - 1: 
            delay = config.LLM_RETRY_DELAY_SECONDS * (2 ** attempt) 
            retry_reason = type(last_exception).__name__ if last_exception else "Unknown reason" 
            logger.info(f"Ollama Embedding: Retrying in {delay:.2f} seconds due to: {retry_reason}.") 
            await asyncio.sleep(delay)
        else: 
            logger.error(f"Ollama Embedding: All {config.LLM_RETRY_ATTEMPTS} retry attempts failed. Last error: {last_exception}")
            return None 
    return None 


def _log_llm_usage(model_name: str, usage_data: Optional[Dict[str, int]], async_mode: bool = False, streamed: bool = False):
    """Helper to log LLM token usage if available in the response."""
    prefix = "Async: " if async_mode else ""
    stream_prefix = "Streamed " if streamed else ""
    if usage_data and isinstance(usage_data, dict):
        logger.info(
            f"{prefix}{stream_prefix}LLM ('{model_name}') Usage - Prompt: {usage_data.get('prompt_tokens', 'N/A')} tk, "
            f"Comp: {usage_data.get('completion_tokens', 'N/A')} tk, Total: {usage_data.get('total_tokens', 'N/A')} tk"
        )
    else:
        logger.debug(f"{prefix}{stream_prefix}LLM ('{model_name}') response missing 'usage' information or 'usage' was not a dictionary.")

async def async_call_llm(
    model_name: str,
    prompt: str,
    temperature: Optional[float] = None, # MODIFIED: Allow optional temperature override
    max_tokens: Optional[int] = None, 
    allow_fallback: bool = False,
    stream_to_disk: bool = False 
) -> Tuple[str, Optional[Dict[str, int]]]:
    """
    Asynchronously calls the LLM (OpenAI-compatible API) with retry and optional model fallback.
    Returns the LLM's text response as a string, and a dictionary containing token usage.
    """
    if not model_name:
        logger.error("async_call_llm: model_name is required.")
        return "", None
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        logger.error("async_call_llm: empty or invalid prompt.")
        return "", None

    prompt_token_count = count_tokens(prompt, model_name) 
    effective_max_output_tokens = max_tokens if max_tokens is not None else config.MAX_GENERATION_TOKENS
    
    # MODIFIED: Use provided temperature or default from config
    effective_temperature = temperature if temperature is not None else config.TEMPERATURE_DEFAULT 
    
    headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}", "Content-Type": "application/json"}
    
    current_model_to_try = model_name
    is_fallback_attempt = False
    current_usage_data: Optional[Dict[str, int]] = None
    final_text_response = "" 

    for attempt_num_overall in range(2):
        if is_fallback_attempt:
            if not allow_fallback or not config.FALLBACK_GENERATION_MODEL:
                logger.warning(f"Primary model '{model_name}' failed. Fallback not allowed or no fallback model configured. Aborting call.")
                return "", current_usage_data 
            current_model_to_try = config.FALLBACK_GENERATION_MODEL
            logger.info(f"Primary model '{model_name}' failed. Attempting fallback with '{current_model_to_try}'.")
            prompt_token_count = count_tokens(prompt, current_model_to_try)
            current_usage_data = None 

        payload = {
            "model": current_model_to_try,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": effective_temperature, # MODIFIED: Use effective_temperature
            "top_p": config.LLM_TOP_P,
            "max_tokens": effective_max_output_tokens
        }

        last_exception_for_current_model: Optional[Exception] = None 
        temp_file_path_for_stream: Optional[str] = None

        for retry_attempt in range(config.LLM_RETRY_ATTEMPTS):
            logger.debug(
                f"Async Calling LLM '{current_model_to_try}' (Attempt {retry_attempt+1}/{config.LLM_RETRY_ATTEMPTS}, OverallAttempt: {attempt_num_overall+1}). "
                f"StreamToDisk: {stream_to_disk}. Prompt tokens (est.): {prompt_token_count}. Max output tokens: {effective_max_output_tokens}. Temp: {effective_temperature}, TopP: {config.LLM_TOP_P}" # MODIFIED: Log effective_temperature
            )
            api_response_obj: Optional[httpx.Response] = None
            
            try:
                if stream_to_disk:
                    payload["stream"] = True
                    with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8", suffix=".llmstream.txt") as tmp_f:
                        temp_file_path_for_stream = tmp_f.name
                    
                    accumulated_stream_content = ""
                    stream_usage_data: Optional[Dict[str, int]] = None
                    try:
                        async with httpx.AsyncClient(timeout=600.0) as client:
                            async with client.stream("POST", f"{config.OPENAI_API_BASE}/chat/completions", json=payload, headers=headers) as response_stream:
                                response_stream.raise_for_status() 
                                
                                async for line in response_stream.aiter_lines():
                                    if line.startswith("data: "):
                                        data_json_str = line[len("data: "):].strip()
                                        if data_json_str == "[DONE]":
                                            break
                                        try:
                                            chunk_data = json.loads(data_json_str)
                                            if chunk_data.get("choices"):
                                                delta = chunk_data["choices"][0].get("delta", {})
                                                content_piece = delta.get("content")
                                                if content_piece:
                                                    accumulated_stream_content += content_piece
                                                
                                                if chunk_data["choices"][0].get("finish_reason") is not None:
                                                    potential_usage = chunk_data.get("usage")
                                                    if not potential_usage and chunk_data.get("x_groq") and chunk_data["x_groq"].get("usage"):
                                                        potential_usage = chunk_data["x_groq"]["usage"]
                                                    if potential_usage and isinstance(potential_usage, dict):
                                                        stream_usage_data = potential_usage 
                                        except json.JSONDecodeError:
                                            logger.warning(f"Async LLM Stream: Could not decode JSON from line: {line}")
                        
                        if temp_file_path_for_stream: 
                            with open(temp_file_path_for_stream, "w", encoding="utf-8") as f_out:
                                f_out.write(accumulated_stream_content)

                        final_text_response = accumulated_stream_content
                        current_usage_data = stream_usage_data 
                        _log_llm_usage(current_model_to_try, current_usage_data, async_mode=True, streamed=True)
                        return final_text_response, current_usage_data
                    finally:
                        if temp_file_path_for_stream and os.path.exists(temp_file_path_for_stream):
                            try:
                                os.remove(temp_file_path_for_stream)
                            except Exception as e_clean:
                                logger.error(f"Error cleaning up temp file {temp_file_path_for_stream} in stream success/finally: {e_clean}")
                else: 
                    payload["stream"] = False
                    async with httpx.AsyncClient(timeout=600.0) as client:
                        api_response_obj = await client.post(f"{config.OPENAI_API_BASE}/chat/completions", json=payload, headers=headers)
                        api_response_obj.raise_for_status()
                        response_data = api_response_obj.json()

                        raw_text_non_stream = ""
                        if response_data.get("choices") and len(response_data["choices"]) > 0:
                            message = response_data["choices"][0].get("message")
                            if message and message.get("content"):
                                raw_text_non_stream = message["content"]
                        else:
                            logger.error(f"Async LLM ('{current_model_to_try}') Invalid response structure - missing choices/content despite 200 OK: {response_data}")
                        
                        final_text_response = raw_text_non_stream
                        current_usage_data = response_data.get("usage")
                        _log_llm_usage(current_model_to_try, current_usage_data, async_mode=True, streamed=False)
                        return final_text_response, current_usage_data

            except httpx.TimeoutException as e_timeout:
                last_exception_for_current_model = e_timeout
                logger.warning(f"Async LLM ('{current_model_to_try}' Attempt {retry_attempt+1}): API request timed out: {e_timeout}")
            except httpx.HTTPStatusError as e_status:
                last_exception_for_current_model = e_status
                response_text_snippet = e_status.response.text[:200] if e_status.response else "N/A"
                error_message_detail = f"API HTTP status error: {e_status}. Status: {e_status.response.status_code if e_status.response else 'N/A'}, Body: {response_text_snippet}" 
                logger.warning(
                    f"Async LLM ('{current_model_to_try}' Attempt {retry_attempt+1}): {error_message_detail}" 
                )
                if e_status.response and 400 <= e_status.response.status_code < 500:
                    if e_status.response.status_code == 400 and "context_length_exceeded" in response_text_snippet.lower():
                         logger.error(
                             f"Async LLM ('{current_model_to_try}'): Context length exceeded. Prompt tokens (est.): {prompt_token_count}. "
                             f"Aborting retries for this model."
                         )
                    else:
                        logger.error(f"Async LLM ('{current_model_to_try}'): Client-side error {e_status.response.status_code}. Aborting retries for this model.")
                    break 
            except httpx.RequestError as e_req:
                last_exception_for_current_model = e_req
                logger.warning(f"Async LLM ('{current_model_to_try}' Attempt {retry_attempt+1}): API request error (network/connection): {e_req}")
            except json.JSONDecodeError as e_json:
                last_exception_for_current_model = e_json
                response_text_snippet = ""
                if api_response_obj and hasattr(api_response_obj, 'text'): response_text_snippet = api_response_obj.text[:200]
                logger.warning(
                    f"Async LLM ('{current_model_to_try}' Attempt {retry_attempt+1}): Failed to decode API JSON response: {e_json}. "
                    f"Response text snippet (if available): {response_text_snippet}"
                )
            except Exception as e_exc:
                last_exception_for_current_model = e_exc
                logger.warning(f"Async LLM ('{current_model_to_try}' Attempt {retry_attempt+1}): Unexpected error: {e_exc}", exc_info=True)
            finally:
                if stream_to_disk and temp_file_path_for_stream and os.path.exists(temp_file_path_for_stream):
                    try:
                        os.remove(temp_file_path_for_stream)
                    except Exception as e_clean_err:
                        logger.error(f"Error cleaning up temp file {temp_file_path_for_stream} after failed LLM attempt: {e_clean_err}")

            if retry_attempt < config.LLM_RETRY_ATTEMPTS - 1:
                delay = config.LLM_RETRY_DELAY_SECONDS * (2 ** retry_attempt)
                retry_reason = type(last_exception_for_current_model).__name__ if last_exception_for_current_model else "Unknown reason" 
                logger.info(f"Async LLM ('{current_model_to_try}'): Retrying in {delay:.2f} seconds due to: {retry_reason}.") 
                await asyncio.sleep(delay)
            else:
                logger.error(f"Async LLM ('{current_model_to_try}'): All {config.LLM_RETRY_ATTEMPTS} retries failed for this model. Last error: {last_exception_for_current_model}")
        
        if last_exception_for_current_model is None: 
            return final_text_response, current_usage_data 

        is_fallback_attempt = True 

        if attempt_num_overall == 0 and isinstance(last_exception_for_current_model, httpx.HTTPStatusError) and \
           last_exception_for_current_model.response and 400 <= last_exception_for_current_model.response.status_code < 500:
            logger.error(f"Async LLM: Primary model '{model_name}' failed with client error. Not attempting fallback if this was the primary and fallback is disabled or unconfigured.")
            
    logger.error(f"Async LLM: Call failed for '{model_name}' after all primary and potential fallback attempts.")
    return "", current_usage_data 


def clean_model_response(text: str) -> str:
    """Cleans common artifacts from LLM text responses, including content within <think> tags and normalizes newlines."""
    if not isinstance(text, str):
        logger.warning(f"clean_model_response received non-string input: {type(text)}. Returning empty string.")
        return ""

    original_length = len(text)
    cleaned_text = text

    text_before_think_removal = cleaned_text 
    think_tags_to_remove = [
        "think", "thought", "thinking", "reasoning", "rationale",
        "meta", "reflection", "internal_monologue", "plan", "analysis",
        "no_think" 
    ]
    for tag_name in think_tags_to_remove:
        block_pattern = re.compile(
            rf'<\s*{tag_name}\s*>.*?<\s*/\s*{tag_name}\s*>',
            flags=re.DOTALL | re.IGNORECASE
        )
        cleaned_text = block_pattern.sub('', cleaned_text)
        
        self_closing_pattern = re.compile(
            rf'<\s*{tag_name}\s*/\s*>',
            flags=re.IGNORECASE
        )
        cleaned_text = self_closing_pattern.sub('', cleaned_text)
        
        lone_opening_pattern = re.compile(rf'<\s*{tag_name}\s*>', flags=re.IGNORECASE)
        cleaned_text = lone_opening_pattern.sub('', cleaned_text)

        lone_closing_pattern = re.compile(rf'<\s*/\s*{tag_name}\s*>', flags=re.IGNORECASE)
        cleaned_text = lone_closing_pattern.sub('', cleaned_text)

    if len(cleaned_text) < len(text_before_think_removal): 
        logger.debug(f"clean_model_response: Removed content associated with <think>/similar tags. Length before: {len(text_before_think_removal)}, after: {len(cleaned_text)}.")


    cleaned_text = re.sub(r'```(?:[a-zA-Z0-9]+)?\s*.*?\s*```', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r'^```\s*\n', '', cleaned_text, flags=re.MULTILINE) 
    cleaned_text = re.sub(r'\n\s*```$', '', cleaned_text, flags=re.MULTILINE)

    cleaned_text = re.sub(r'^\s*Chapter \d+\s*[:\-—]?\s*(.*?)\s*$', r'\1', cleaned_text, flags=re.MULTILINE | re.IGNORECASE).strip()
    
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
                new_text = re.sub(pattern_str, "", cleaned_text, count=1, flags=re.IGNORECASE | re.MULTILINE).strip()
                if new_text == cleaned_text: break
                cleaned_text = new_text
        else:
            cleaned_text = re.sub(pattern_str, "", cleaned_text, count=1, flags=re.IGNORECASE | re.MULTILINE).strip()

    final_text = cleaned_text.strip()
    final_text = re.sub(r'\n\s*\n(\s*\n)+', '\n\n', final_text)
    final_text = re.sub(r'\n{3,}', '\n\n', final_text)

    if original_length > 0 and len(final_text) < original_length:
        reduction_percentage = ((original_length - len(final_text)) / original_length) * 100
        if reduction_percentage > 0.5: 
            logger.debug(f"Cleaning reduced text length from {original_length} to {len(final_text)} ({reduction_percentage:.1f}% reduction).")

    return final_text