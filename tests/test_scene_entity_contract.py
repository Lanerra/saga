"""Closed scene producer wire contracts; transport fakes are not provider proof."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from core.exceptions import LLMServiceError
from core.http_client_service import HTTPClientService
from core.langgraph.initialization.catalog import select_catalog
from core.langgraph.nodes import scene_extraction, scene_extraction_parsing
from core.langgraph.nodes.scene_extraction_validation import scene_name_authority
from core.llm_interface_refactored import create_llm_service
from core.relationship_validation import validate_relationship_semantics_strict
from core.service_context import get_services
from models.kg_constants import RELATIONSHIP_TYPES
from tests.test_r08g_catalog_fixtures import catalog_state

NAMES = {"characters": "Father O'Brien", "locations": "St. James’ Gate (West)", "events": "The King's Return"}
LABELS = {"characters": "Character", "locations": "Location", "events": "Event"}
SCENE = "Father O'Brien visits St. James’ Gate (West) for The King's Return."


def payload(kind: str, updates: dict[str, Any]) -> dict[str, Any]:
    return {"character_updates": updates} if kind == "characters" else {"world_updates": {LABELS[kind]: updates}}


def details(kind: str) -> dict[str, Any]:
    if kind == "characters":
        return {"description": "A named participant", "traits": [], "status": "Present", "relationships": {}}
    return {"description": "Named in the scene", "category": "Structure" if kind == "locations" else "Ceremony", "goals": [], "rules": [], "key_elements": []}


def named_mapping(schema: dict[str, Any], kind: str) -> dict[str, Any]:
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    wrapper = "character_updates" if kind == "characters" else "world_updates"
    assert schema["required"] == [wrapper]
    assert set(schema["properties"]) == {wrapper}
    mapping = schema["properties"][wrapper]
    if kind != "characters":
        assert mapping["additionalProperties"] is False
        assert mapping["required"] == [LABELS[kind]]
        assert set(mapping["properties"]) == {LABELS[kind]}
        mapping = mapping["properties"][LABELS[kind]]
    assert mapping["type"] == "object" and mapping["additionalProperties"] is False
    assert mapping.get("required", []) == []
    return dict(mapping)


@pytest.mark.parametrize("kind", NAMES)
@pytest.mark.parametrize("mode", ["success", "json_retry", "fallback", "exhausted_json", "exhausted_fallback"])
@pytest.mark.run_settings(LLM_RETRY_ATTEMPTS=1)
async def test_entity_schema_reaches_every_wire_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, mode: str) -> None:
    catalog = select_catalog(catalog_state(tmp_path, characters=(NAMES["characters"], "Absent Person"), locations=(NAMES["locations"], "Absent Place"), events=(NAMES["events"], "Absent Event")))
    bodies: list[dict[str, Any]] = []
    data = payload(kind, {NAMES[kind]: details(kind)})

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if mode == "exhausted_fallback" or (mode == "fallback" and len(bodies) == 1):
            return httpx.Response(400)
        text = "not JSON" if mode == "exhausted_json" or (mode == "json_retry" and len(bodies) == 1) else json.dumps(data)
        return httpx.Response(200, json={"choices": [{"message": {"content": text}, "finish_reason": "stop"}]})

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    monkeypatch.setattr(get_services(), "language_model", service)
    try:
        extractor = getattr(scene_extraction, f"_extract_{kind}_from_scene")
        if mode.startswith("exhausted"):
            with pytest.raises((ValueError, LLMServiceError)):
                await extractor(SCENE, 0, 1, "Synthetic", "Fantasy", NAMES["characters"], "synthetic", catalog=catalog)
        else:
            rows = await extractor(SCENE, 0, 1, "Synthetic", "Fantasy", NAMES["characters"], "synthetic", catalog=catalog)
            assert [row["name"] for row in rows] == [NAMES[kind]]
            assert rows[0]["type"] == LABELS[kind]
        assert len(bodies) == (1 if mode == "success" else 2)
        contract = bodies[0]["response_format"]
        assert all(body["response_format"] == contract for body in bodies)
        assert contract["type"] == "json_schema" and contract["json_schema"]["strict"] is False
        mapping = named_mapping(contract["json_schema"]["schema"], kind)
        assert set(mapping["properties"]) == {NAMES[kind]}
        entry = mapping["properties"][NAMES[kind]]
        assert entry["additionalProperties"] is False
        assert set(entry["properties"]) == set(details(kind))
        assert set(entry["required"]) == set(details(kind))
        assert entry["properties"]["description"]["type"] == "string"
        for field in ("traits",) if kind == "characters" else ("goals", "rules", "key_elements"):
            assert entry["properties"][field] == {"type": "array", "items": {"type": "string"}}
        if kind == "characters":
            relationships = entry["properties"]["relationships"]
            assert relationships["additionalProperties"] is False
            assert set(relationships["properties"]) == {NAMES["characters"]}
            relation = relationships["properties"][NAMES["characters"]]
            assert relation["additionalProperties"] is False
            assert set(relation["required"]) == {"type", "description"}
            assert relation["properties"]["type"]["enum"] == sorted(predicate for predicate in RELATIONSHIP_TYPES if validate_relationship_semantics_strict(predicate, "Character", "Character")[0])
        assert all(SCENE in body["messages"][-1]["content"] for body in bodies)
    finally:
        await service.aclose()


@pytest.mark.parametrize("kind", NAMES)
@pytest.mark.parametrize("defect", ["duplicate_wrapper", "duplicate_name", "duplicate_detail", "wrong_description", "wrong_array", "unknown_name", "wrong_label", "fenced"])
async def test_entity_bad_response_is_never_partial_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, defect: str) -> None:
    catalog = select_catalog(catalog_state(tmp_path, characters=(NAMES["characters"],), locations=(NAMES["locations"],), events=(NAMES["events"],)))
    data = payload(kind, {NAMES[kind]: details(kind)})
    if defect == "wrong_description":
        data = payload(kind, {NAMES[kind]: {**details(kind), "description": 1}})
    elif defect == "wrong_array":
        data = payload(kind, {NAMES[kind]: {**details(kind), "traits" if kind == "characters" else "goals": [1]}})
    elif defect == "unknown_name":
        data = payload(kind, {NAMES[kind]: details(kind), "Invented Name": details(kind)})
    elif defect == "wrong_label":
        other = NAMES["locations"] if kind != "locations" else NAMES["characters"]
        data = payload(kind, {other: details(kind)})
    raw = json.dumps(data)
    if defect.startswith("duplicate_"):
        key = {"duplicate_wrapper": "character_updates" if kind == "characters" else "world_updates", "duplicate_name": NAMES[kind], "duplicate_detail": "description"}[defect]
        token = json.dumps(key) + ":"
        raw = raw.replace(token, token + " null, " + token, 1)
    elif defect == "fenced":
        raw = "```json\n" + raw + "\n```"
    bodies: list[bytes] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": raw}, "finish_reason": "stop"}]})

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    monkeypatch.setattr(get_services(), "language_model", service)
    try:
        with pytest.raises(ValueError):
            await getattr(scene_extraction, f"_extract_{kind}_from_scene")(SCENE, 0, 1, "Synthetic", "Fantasy", NAMES["characters"], "synthetic", catalog=catalog)
        assert bodies
    finally:
        await service.aclose()


@pytest.mark.parametrize("kind", [*NAMES, "relationships"])
async def test_validated_empty_candidates_need_no_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    catalog = select_catalog(catalog_state(tmp_path))
    calls: list[bytes] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.content)
        return httpx.Response(400)

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    monkeypatch.setattr(get_services(), "language_model", service)
    try:
        assert await getattr(scene_extraction, f"_extract_{kind}_from_scene")("Nothing named here.", 0, 1, "Synthetic", "Fantasy", "Elara", "synthetic", catalog=catalog) == []
        assert calls == []
    finally:
        await service.aclose()


@pytest.mark.parametrize("kind", NAMES)
@pytest.mark.parametrize("names", [(), ("The Hague", "Father O'Brien", "St. James’ Gate (West)")])
def test_finite_schema_empty_geometry_and_parser_contract(kind: str, names: tuple[str, ...]) -> None:
    contract = scene_extraction_parsing.scene_entity_response_format(LABELS[kind], names)
    mapping = named_mapping(contract["json_schema"]["schema"], kind)
    assert set(mapping["properties"]) == set(names)
    data = payload(kind, {name: details(kind) for name in names})
    with scene_name_authority(names):
        if kind == "characters":
            parsed = scene_extraction_parsing.parse_character_updates(data, 0, 1)
        else:
            parsed = scene_extraction_parsing.parse_world_updates(data, LABELS[kind], 0, 1)
    assert parsed == [(name, details(kind)) for name in names]
    assert mapping["additionalProperties"] is False


def test_imports_are_owned_worktree_source() -> None:
    root = Path(__file__).resolve().parents[1]
    assert Path(scene_extraction.__file__).resolve() == root / "core/langgraph/nodes/scene_extraction.py"
    assert Path(scene_extraction_parsing.__file__).resolve() == root / "core/langgraph/nodes/scene_extraction_parsing.py"


@pytest.mark.parametrize("kind", [*NAMES, "relationships"])
@pytest.mark.parametrize("scene", ["", "   "])
async def test_blank_scene_is_not_a_known_empty_selection(tmp_path: Path, kind: str, scene: str) -> None:
    catalog = select_catalog(catalog_state(tmp_path))
    with pytest.raises(ValueError, match="blank"):
        await getattr(scene_extraction, f"_extract_{kind}_from_scene")(scene, 0, 1, "Synthetic", "Fantasy", "Elara", "synthetic", catalog=catalog)


@pytest.mark.parametrize("kind", [*NAMES, "relationships"])
async def test_missing_authority_is_not_a_known_empty_selection(kind: str) -> None:
    with pytest.raises(ValueError, match="eligible identity catalog"):
        await getattr(scene_extraction, f"_extract_{kind}_from_scene")("Nothing named here.", 0, 1, "Synthetic", "Fantasy", "Elara", "synthetic")


def test_relationship_empty_schema_keeps_only_empty_array() -> None:
    schema = scene_extraction_parsing.SceneRelationships.response_format()["json_schema"]["schema"]
    assert schema["properties"]["kg_triples"]["maxItems"] == 0
    assert "enum" not in schema["$defs"]["SceneRelationship"]["properties"]["subject"]
    assert scene_extraction_parsing.parse_kg_triples({"kg_triples": []}, 0, 1) == []
