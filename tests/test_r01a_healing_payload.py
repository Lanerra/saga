"""Synthetic payload admission at the database/transport boundaries; no live services."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

import config
import core.graph_healing_service as healing
from core.exceptions import ValidationError
from core.service_context import RunServices, get_services, inject_services

NODE: dict[str, Any] = {"element_id": "internal-synthetic", "id": "character_synthetic", "name": "Synthetic Keeper", "type": "Character", "description": "Short", "traits": [], "created_chapter": 1}


def payload(**changes: Any) -> dict[str, Any]:
    return {"inferred_description": "A synthetic keeper.", "inferred_traits": ["Brave"], "inferred_role": "Ally", "confidence": 0.9, **changes}


class Database:
    def __init__(self, *, mentions: bool = True) -> None:
        self.mentions = mentions
        self.writes: list[dict[str, Any]] = []
        self.reads: list[dict[str, Any]] = []

    async def execute_read_query(self, query: str, parameters: Any = None) -> list[dict[str, Any]]:
        self.reads.append(parameters or {})
        if "ORDER BY n.created_chapter ASC" in query:
            return [dict(NODE)]
        if "count(r) AS rel_count" in query:
            return [{"rel_count": 3, "status": "Unknown"}]
        if (parameters or {}).get("id_param"):
            return [{"chapter_number": 1, "summary": "The synthetic keeper bravely guided the group."}] if self.mentions else []
        return []

    async def execute_write_query(self, query: str, parameters: Any = None) -> list[dict[str, Any]]:
        self.writes.append(dict(parameters or {}))
        return [{"name": NODE["name"]}]


class Transport:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def async_call_llm(self, **arguments: Any) -> tuple[str, None]:
        self.calls.append(arguments)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response, None


@pytest.mark.parametrize("field,value", [
    ("inferred_description", ["bad"]), ("inferred_description", {"bad": "value"}),
    ("inferred_role", ["Ally"]), ("inferred_role", {"role": "Ally"}),
    ("inferred_traits", "Brave"), ("inferred_traits", [1]), ("inferred_traits", [None]),
    ("inferred_traits", [{"trait": "Brave"}]), ("inferred_traits", ("Brave",)),
    ("inferred_traits", ["two words"]), ("inferred_traits", [""]),
    ("inferred_description", None), ("inferred_role", None), ("inferred_traits", None),
    ("unexpected", "ignored"),
])
@pytest.mark.parametrize("confidence", [0.9, 0.1])
async def test_invalid_fields_are_rejected_before_any_write(field: str, value: Any, confidence: float) -> None:
    database = Database()
    with inject_services(RunServices(cast(Any, object()), cast(Any, database))):
        try:
            with pytest.raises(ValidationError):
                await healing.GraphHealingService().apply_enrichment(NODE["element_id"], payload(**{field: value, "confidence": confidence}))
        finally:
            assert database.writes == []


@pytest.mark.parametrize("field", list(payload()))
async def test_nonempty_payload_requires_every_producer_field(field: str) -> None:
    enriched = payload()
    del enriched[field]
    database = Database()
    with inject_services(RunServices(cast(Any, object()), cast(Any, database))):
        with pytest.raises(ValidationError):
            await healing.GraphHealingService().apply_enrichment(NODE["element_id"], enriched)
    assert database.writes == []


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), float("-inf"), -0.01, 1.01, True, "0.9", None])
async def test_confidence_remains_strict_finite_and_bounded(confidence: Any) -> None:
    database = Database()
    with inject_services(RunServices(cast(Any, object()), cast(Any, database))):
        with pytest.raises(ValidationError):
            await healing.GraphHealingService().apply_enrichment(NODE["element_id"], payload(confidence=confidence))
    assert database.writes == []


@pytest.mark.parametrize("confidence,applied", [(0, False), (0.59, False), (0.6, True), (1, True)])
async def test_valid_payload_preserves_values_and_threshold(confidence: float, applied: bool) -> None:
    database = Database()
    enriched = payload(confidence=confidence)
    before = json.dumps(enriched)
    with inject_services(RunServices(cast(Any, object()), cast(Any, database))):
        assert await healing.GraphHealingService().apply_enrichment(NODE["element_id"], enriched) is applied
    assert json.dumps(enriched) == before
    assert database.writes == ([{"element_id": NODE["element_id"], "description": enriched["inferred_description"], "new_traits": ["Brave"], "role": "Ally", "confidence": confidence}] if applied else [])


@pytest.mark.parametrize("enriched", [{}, payload(inferred_description="", inferred_traits=[], inferred_role="")])
async def test_explicit_no_inference_is_a_zero_write_noop(enriched: dict[str, Any]) -> None:
    database = Database()
    with inject_services(RunServices(cast(Any, object()), cast(Any, database))):
        assert await healing.GraphHealingService().apply_enrichment(NODE["element_id"], enriched) is False
    assert database.writes == []


@pytest.mark.parametrize("response", [
    json.dumps(payload(inferred_description=["bad"])), json.dumps(payload(inferred_traits="Brave")),
    json.dumps(payload(unexpected=True)), '{"confidence":0.1,' + json.dumps(payload())[1:],
    json.dumps(payload()).replace('"Brave"', '{"trait":"Brave","trait":"Calm"}'),
    json.dumps(payload()).replace('0.9', '1e400'), json.dumps(payload()).replace('0.9', 'NaN'),
    '{}', '[]', '```json\n' + json.dumps(payload()) + '\n```',
])
async def test_invalid_producer_payload_is_advisory_failure_not_success(response: str) -> None:
    database, transport = Database(), Transport(response)
    with inject_services(RunServices(cast(Any, transport), cast(Any, database))):
        receipt = await healing.GraphHealingService().heal_graph(2, "offline-synthetic")
    assert receipt["status"] == "partial"
    assert receipt["nodes_enriched"] == 0
    assert receipt["nodes_graduated"] == 0
    assert receipt["warnings"]
    assert receipt["actions"][0]["type"] == "enrich_error"
    assert receipt["actions"][0]["raw_responses"] == [response] * len(transport.calls)
    assert database.writes == []
    assert 1 <= len(transport.calls) <= config.JSON_PARSE_RETRY_ATTEMPTS


async def test_producer_schema_and_raw_response_are_not_cleaned() -> None:
    database, transport = Database(), Transport(json.dumps(payload()))
    with inject_services(RunServices(cast(Any, transport), cast(Any, database))):
        assert await healing.GraphHealingService().enrich_node_from_context(NODE, "offline-synthetic") == payload()
    arguments = transport.calls[0]
    assert arguments["auto_clean_response"] is False
    assert arguments["spacy_cleanup"] is False
    contract = arguments["response_format"]["json_schema"]
    assert contract["strict"] is config.STRUCTURED_OUTPUT_STRICT is False
    schema = contract["schema"]
    assert schema["additionalProperties"] is False
    assert list(schema["properties"]) == list(payload())
    assert schema["required"] == list(payload())
    assert schema["properties"]["inferred_description"]["type"] == "string"
    assert schema["properties"]["inferred_role"]["type"] == "string"
    assert schema["properties"]["inferred_traits"]["items"]["type"] == "string"
    assert schema["properties"]["confidence"]["minimum"] == 0
    assert schema["properties"]["confidence"]["maximum"] == 1
    assert database.reads[0]["id_param"] == NODE["id"]
    assert database.writes == []


async def test_no_mentions_skips_transport_and_writes() -> None:
    database, transport = Database(mentions=False), Transport(RuntimeError("must not call"))
    with inject_services(RunServices(cast(Any, transport), cast(Any, database))):
        enriched = await healing.GraphHealingService().enrich_node_from_context(NODE, "offline-synthetic")
        assert enriched == {}
        assert await healing.GraphHealingService().apply_enrichment(NODE["element_id"], enriched) is False
    assert transport.calls == []
    assert database.writes == []


async def test_unavailable_transport_remains_advisory_partial() -> None:
    database, transport = Database(), Transport(RuntimeError("synthetic transport unavailable"))
    with inject_services(RunServices(cast(Any, transport), cast(Any, database))):
        receipt = await healing.GraphHealingService().heal_graph(2, "offline-synthetic")
    assert receipt["status"] == "partial"
    assert receipt["nodes_enriched"] == 0
    assert "synthetic transport unavailable" in receipt["warnings"][0]
    assert database.writes == []


def test_import_provenance() -> None:
    assert Path(healing.__file__).resolve() == Path(__file__).resolve().parents[1] / "core" / "graph_healing_service.py"


@pytest.mark.parametrize("response", ["not JSON", json.dumps(payload(inferred_role=["Ally"]))])
async def test_rejected_raw_response_is_available_without_logging_it(response: str) -> None:
    database, transport = Database(), Transport(response)
    with inject_services(RunServices(cast(Any, transport), cast(Any, database))):
        with pytest.raises(ValidationError) as caught:
            await healing.GraphHealingService().enrich_node_from_context(NODE, "offline-synthetic")
    assert getattr(caught.value, "raw_responses", None) == tuple(response for _ in transport.calls)
    assert response not in str(caught.value)
    assert database.writes == []


@pytest.mark.parametrize("enriched", [None, [], "", False])
async def test_only_empty_object_is_the_internal_no_mention_sentinel(enriched: Any) -> None:
    database = Database()
    with inject_services(RunServices(cast(Any, object()), cast(Any, database))):
        with pytest.raises(ValidationError):
            await healing.GraphHealingService().apply_enrichment(NODE["element_id"], enriched)
    assert database.writes == []


@pytest.mark.parametrize("response", [json.dumps(payload()), "```json\n" + json.dumps(payload()) + "\n```"])
async def test_existing_language_model_preserves_raw_json_at_http_boundary(monkeypatch: pytest.MonkeyPatch, response: str) -> None:
    calls: list[dict[str, Any]] = []

    async def send(client: httpx.AsyncClient, request: httpx.Request, **arguments: Any) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, request=request, json={"choices": [{"message": {"role": "assistant", "content": response}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    database = Database()
    with inject_services(RunServices(get_services().language_model, cast(Any, database))):
        service = healing.GraphHealingService()
        if response.startswith("```"):
            with pytest.raises(ValidationError) as caught:
                await service.enrich_node_from_context(NODE, "offline-synthetic")
            assert getattr(caught.value, "raw_responses", None) == (response,) * config.JSON_PARSE_RETRY_ATTEMPTS
        else:
            assert await service.enrich_node_from_context(NODE, "offline-synthetic") == payload()
    assert calls
    for call in calls:
        assert call["response_format"]["json_schema"]["strict"] is False
        assert call["temperature"] == config.TEMPERATURE_OVERRIDE == 1.0
        assert call["max_tokens"] == config.MAX_GENERATION_TOKENS == 65536
    assert database.writes == []
