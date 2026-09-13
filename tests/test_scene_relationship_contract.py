"""Scene relationship producer contracts at the real adapter boundary."""
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from core.http_client_service import HTTPClientService
from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization.catalog import select_catalog
from core.langgraph.nodes.scene_extraction import _extract_relationships_from_scene, extract_from_scenes
from core.langgraph.nodes.scene_extraction_parsing import parse_kg_triples
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import create_llm_service
from core.relationship_validation import validate_relationship_semantics_strict
from core.service_context import get_services
from models.kg_constants import RELATIONSHIP_TYPES
from prompts.prompt_renderer import render_prompt
from tests.test_r08g_catalog_fixtures import catalog_state

ROW = {"subject": "Elara", "predicate": "LOCATED_AT", "object_entity": "Library", "description": "Elara enters the Library."}


def test_scene_prompt_examples_and_guidance_use_only_canonical_predicates() -> None:
    prompt = render_prompt("knowledge_agent/extract_relationships.j2", {
        "chapter_number": 1, "novel_title": "Synthetic", "novel_genre": "Fantasy",
        "protagonist": "Elara", "chapter_text": "Elara enters the Library.",
        "canonical_relationship_types": sorted(RELATIONSHIP_TYPES),
    })
    examples = prompt.split("Examples demonstrating valid relationships:", 1)[1]
    predicates = set(re.findall(r"\b[A-Z]+(?:_[A-Z]+)+\b", examples))
    assert predicates <= RELATIONSHIP_TYPES
    assert "ALLIES_WITH" in predicates
    example, _ = json.JSONDecoder().raw_decode(examples[examples.index('{'):])
    labels = {"Elara": "Character", "Marcus": "Character", "Sunken Library": "Location", "Enchanted Blade": "Item"}
    assert all(validate_relationship_semantics_strict(row["predicate"], labels[row["subject"]], labels[row["object_entity"]])[0] for row in example["kg_triples"])


@pytest.mark.parametrize("mode", ["success", "json_retry", "fallback", "exhausted_json", "exhausted_fallback"])
@pytest.mark.run_settings(LLM_RETRY_ATTEMPTS=1)
async def test_scene_schema_reaches_every_wire_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    catalog = select_catalog(catalog_state(tmp_path, characters=("Elara",), locations=("Library",)))
    bodies: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if mode == "exhausted_fallback" or (mode == "fallback" and len(bodies) == 1):
            return httpx.Response(400)
        text = "not JSON" if mode == "exhausted_json" or (mode == "json_retry" and len(bodies) == 1) else '{"kg_triples": []}'
        return httpx.Response(200, json={"choices": [{"message": {"content": text}, "finish_reason": "stop"}]})

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    monkeypatch.setattr(get_services(), "language_model", service)
    try:
        if mode.startswith("exhausted"):
            from core.exceptions import LLMServiceError
            with pytest.raises((ValueError, LLMServiceError)):
                await _extract_relationships_from_scene("Elara enters the Library.", 0, 1, "Synthetic", "Fantasy", "Elara", "synthetic", catalog=catalog)
        else:
            assert await _extract_relationships_from_scene("Elara enters the Library.", 0, 1, "Synthetic", "Fantasy", "Elara", "synthetic", catalog=catalog) == []
        assert len(bodies) == (1 if mode == "success" else 2)
        contract = bodies[0]["response_format"]
        assert all(body["response_format"] == contract for body in bodies)
        assert contract["type"] == "json_schema"
        assert contract["json_schema"]["strict"] is False
        assert all(body["temperature"] == 1.0 for body in bodies)
        schema = contract["json_schema"]["schema"]
        assert schema["required"] == ["kg_triples"]
        assert schema["additionalProperties"] is False
        rows = schema["properties"]["kg_triples"]
        assert rows["maxItems"] == 15
        row_schema = rows["items"]
        if "$ref" in row_schema:
            row_schema = schema["$defs"][row_schema["$ref"].rsplit("/", 1)[1]]
        labels = {"Elara": "Character", "Library": "Location"}
        assert len(row_schema["oneOf"]) == 4
        for variant in row_schema["oneOf"]:
            assert set(variant["properties"]) == set(ROW)
            assert set(variant["required"]) == set(ROW)
            assert variant["additionalProperties"] is False
            properties = variant["properties"]
            subject, = properties["subject"]["enum"]
            target, = properties["object_entity"]["enum"]
            assert properties["predicate"]["enum"] == sorted(predicate for predicate in RELATIONSHIP_TYPES if validate_relationship_semantics_strict(predicate, labels[subject], labels[target])[0])
        if not mode.startswith("exhausted"):
            await service.async_call_llm("synthetic", "Write prose.", max_tokens=100, auto_clean_response=False)
            assert "response_format" not in bodies[-1]
    finally:
        await service.aclose()


def invalid_response(defect: str) -> str:
    row: dict[str, Any] = dict(ROW)
    if defect.startswith("missing_"):
        del row[defect.removeprefix("missing_")]
    elif defect == "unknown_field":
        row["source_id"] = "not-a-scene-field"
    elif defect in {"unknown_predicate", "alias", "lowercase"}:
        row["predicate"] = {"unknown_predicate": "INVENTED", "alias": "ALLY_OF", "lowercase": "located_at"}[defect]
    elif defect == "blank":
        row["subject"] = " "
    elif defect == "wrong_type":
        row["description"] = 1
    data: dict[str, Any] = {"kg_triples": [row] * (16 if defect == "too_many" else 1)}
    if defect == "unknown_wrapper":
        data["relationships"] = []
    raw = json.dumps(data)
    if defect == "duplicate_row_key":
        raw = raw.replace('"predicate":', '"predicate": "INVENTED", "predicate":')
    if defect == "duplicate_wrapper":
        raw = raw.replace('"kg_triples":', '"kg_triples": [], "kg_triples":')
    return raw


@pytest.mark.parametrize("defect", [
    "missing_subject", "missing_predicate", "missing_object_entity", "missing_description", "unknown_field",
    "unknown_wrapper", "unknown_predicate", "alias", "lowercase", "blank", "wrong_type", "too_many",
    "duplicate_row_key", "duplicate_wrapper",
])
async def test_invalid_scene_relationship_blocks_persistence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    manager = ContentManager(str(tmp_path))
    state: NarrativeState = catalog_state(tmp_path, characters=("Elara",), locations=("Library",), events=("Arrival",))
    state["scene_drafts_ref"] = manager.save_list_of_texts(["Elara enters the Library for Arrival."], "scenes", "chapter_1", 1)
    replies = iter([json.dumps({"character_updates": {}}), json.dumps({"world_updates": {"Location": {}}}), json.dumps({"world_updates": {"Event": {}}}), invalid_response(defect)])
    bodies: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": next(replies)}, "finish_reason": "stop"}]})

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    monkeypatch.setattr(get_services(), "language_model", service)
    try:
        result = await extract_from_scenes(state)
        assert result["extraction_status"] == "failed"
        assert result["has_fatal_error"] is True
        assert result["extracted_relationships_ref"] is None
        assert result["extracted_entities_ref"] is None
        assert result["extraction_outcomes"][-1]["status"] == "failed"
        assert len(bodies) == 4
        assert manager.get_latest_version("extracted_relationships", "chapter_1") == 0
        assert manager.get_latest_version("extracted_entities", "chapter_1") == 0
    finally:
        await service.aclose()


@pytest.mark.parametrize("count", [0, 1, 15])
def test_scene_relationship_admission_preserves_all_valid_rows(count: int) -> None:
    rows = [dict(ROW) for _ in range(count)]
    assert parse_kg_triples({"kg_triples": rows}, 0, 1) == rows
