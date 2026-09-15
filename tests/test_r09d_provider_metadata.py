"""Offline replay of R09D's actual Ornith provider bytes; no live calls."""
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from config.settings import EffectiveSettings
from core.http_client_service import CompletionHTTPClient, CompletionResponse, HTTPClientService, PartsCompletionResponse, completion_content

RESPONSE_BYTES = b'{"choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"{\\"status\\":\\"ready\\"}","reasoning_content":"The user wants me to return exactly the JSON object {\\"status\\":\\"ready\\"}. This is a simple, direct request. I should return exactly that JSON object.\\n"}}],"created":1789281403,"model":"saga-r09s-ornith-1.5-35b-a3b-apex-i-compact","system_fingerprint":"b10931-3057bb66c","object":"chat.completion","usage":{"completion_tokens":41,"prompt_tokens":21,"total_tokens":62,"prompt_tokens_details":{"cached_tokens":0}},"id":"chatcmpl-PfY1vpY4dIVKhAL1TI8Q11nDxDwO7CsS","timings":{"cache_n":0,"prompt_n":21,"prompt_ms":222.464,"prompt_per_token_ms":10.59352380952381,"prompt_per_second":94.39729574223246,"predicted_n":41,"predicted_ms":619.914,"predicted_per_token_ms":15.49785,"predicted_per_second":64.52507928519117}}'
REQUEST_BYTES = b'{"model":"saga-r09s-ornith-1.5-35b-a3b-apex-i-compact","messages":[{"role":"user","content":"Return exactly the JSON object {\\"status\\":\\"ready\\"}."}],"temperature":1.0,"top_p":0.95,"max_tokens":65536,"stream":false,"response_format":{"type":"json_schema","json_schema":{"name":"qualification","strict":false,"schema":{"additionalProperties":false,"properties":{"status":{"const":"ready","title":"Status","type":"string"}},"required":["status"],"title":"Answer","type":"object"}}}}'


def test_captured_response_retains_exact_metadata() -> None:
    assert hashlib.sha256(RESPONSE_BYTES).hexdigest() == "47ebc7056d3f81eca1fc9019235badc36667b99dac44cb55f1e960dc5cb4dd9b"
    validated = CompletionResponse.model_validate_json(RESPONSE_BYTES)
    assert validated.model_dump(exclude_unset=True) == json.loads(RESPONSE_BYTES)
    assert validated.timings is not None
    assert validated.timings.draft_n is None
    assert validated.timings.draft_n_accepted is None
    assert completion_content(json.loads(RESPONSE_BYTES), EffectiveSettings(_env_file=None)) == '{"status":"ready"}'


@pytest.mark.parametrize("format_name", ["text", "text_parts"])
@pytest.mark.parametrize("field", ["draft_n", "draft_n_accepted"])
@pytest.mark.parametrize("value", [-1, 1.5, "1", True, None])
def test_supplied_speculative_counts_remain_strict(format_name: str, field: str, value: Any) -> None:
    response = json.loads(RESPONSE_BYTES)
    response["timings"][field] = value
    if format_name == "text_parts":
        response["choices"][0]["message"]["content"] = [{"type": "text", "text": '{"status":"ready"}'}]
    with pytest.raises(ValidationError) as caught:
        (PartsCompletionResponse if format_name == "text_parts" else CompletionResponse).model_validate(response)
    assert [error["loc"] for error in caught.value.errors()] == [("timings", field)]


@pytest.mark.parametrize("format_name", ["text", "text_parts"])
@pytest.mark.parametrize("value", [0, 7])
def test_supplied_speculative_counts_are_preserved(format_name: str, value: int) -> None:
    response = json.loads(RESPONSE_BYTES)
    response["timings"].update(draft_n=value, draft_n_accepted=value)
    if format_name == "text_parts":
        response["choices"][0]["message"]["content"] = [{"type": "text", "text": '{"status":"ready"}'}]
    validated = (PartsCompletionResponse if format_name == "text_parts" else CompletionResponse).model_validate(response)
    assert validated.model_dump(exclude_unset=True) == response


@pytest.mark.parametrize("defect", ["extra", "missing_prompt", "length", "refusal", "empty", "reasoning_only"])
def test_answer_and_other_metadata_gates_stay_closed(defect: str) -> None:
    response = json.loads(RESPONSE_BYTES)
    choice = response["choices"][0]
    if defect == "extra":
        response["timings"]["unexpected"] = 0
    elif defect == "missing_prompt":
        del response["timings"]["prompt_n"]
    elif defect == "length":
        choice["finish_reason"] = "length"
    elif defect == "refusal":
        choice["message"]["refusal"] = "No"
    elif defect == "empty":
        choice["message"]["content"] = " "
    elif defect == "reasoning_only":
        del choice["message"]["content"]
    with pytest.raises(ValueError):
        completion_content(response, EffectiveSettings(_env_file=None))


async def test_captured_bytes_through_production_adapter() -> None:
    payload = json.loads(REQUEST_BYTES)
    sent = []

    def replay(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        assert request.content == REQUEST_BYTES
        return httpx.Response(200, content=RESPONSE_BYTES)

    configuration = EffectiveSettings(_env_file=None, OPENAI_API_BASE="http://127.0.0.1:18080/v1", HTTPX_TIMEOUT=300.0, LLM_RETRY_ATTEMPTS=1, TEMPERATURE_OVERRIDE=1.0, STRUCTURED_OUTPUT_STRICT=False, MAX_CONTEXT_TOKENS=131072)
    service = HTTPClientService(configuration=configuration, client=httpx.AsyncClient(transport=httpx.MockTransport(replay)))
    try:
        assert Path(inspect.getfile(CompletionHTTPClient)).resolve() == Path(__file__).resolve().parents[1] / "core/http_client_service.py"
        result = await CompletionHTTPClient(service).get_completion(payload["model"], payload["messages"], 1.0, 65536, response_format=payload["response_format"])
        assert result == json.loads(RESPONSE_BYTES)
        assert len(sent) == 1
        assert service.get_statistics()["retry_attempts"] == 0
    finally:
        await service.aclose()
