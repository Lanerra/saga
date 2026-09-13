"""Synthetic catalog-to-scene admission, not natural-language quality evidence."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import config
from core.http_client_service import HTTPClientService
from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization.catalog import materialize_entities, select_catalog
from core.langgraph.initialization.snapshot import select_inputs
from core.langgraph.nodes import scene_extraction
from core.langgraph.nodes.scene_extraction_validation import _validate_entity_with_spacy, validate_named_entity
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import create_llm_service
from core.service_context import get_services
from models.kg_models import WorldItem
from tests.test_staged_initialization import example_state

SCENE = "Neighbors gathered at the community hall. Father O'Brien visits The Hague and King's Cross. The Blackwood Family assists Father O'Brien. The Arrival ceremony begins."


def selected_scene(tmp_path: Path, scene: str = SCENE) -> NarrativeState:
    state = example_state(tmp_path)
    manager = ContentManager(str(tmp_path))
    reference = state["character_sheets_ref"]
    assert reference is not None
    sheet = manager.load_json_strict(reference)["Ada"]
    sheets = {name: {**sheet, "name": name} for name in ["Father O'Brien", "The Blackwood Family", "Absent Name"]}
    state["character_sheets_ref"] = manager.save_json(sheets, "character_sheets", "named", 1)
    chapter_reference = state["chapter_outlines_ref"]
    assert chapter_reference is not None
    chapters = manager.load_json_strict(chapter_reference)
    chapters["1"]["key_beats"] = ["The Arrival"]
    chapters["1"]["version"] = 1
    state["chapter_outlines_ref"] = manager.save_json(chapters, "chapter_outlines", "named", 1)
    catalog = materialize_entities(select_inputs(state), [
        WorldItem(name=name, category="location", description="Synthetic named place", id=identity)
        for name, identity in [("The Hague", "place-exact-17"), ("King's Cross", "place-exact-18")]
    ], ())
    state["initialization_catalog_ref"] = manager.save_json(catalog.model_dump(mode="json"), "initialization_catalog", catalog.inputs.identity, 2)
    state["scene_drafts_ref"] = manager.save_json([scene], "scene_drafts", "chapter_1", 1)
    state["current_chapter"] = 1
    return state


def responses() -> list[dict[str, Any]]:
    return [
        {"character_updates": {name: {"description": "Synthetic named participant", "traits": [], "status": "Active", "relationships": {}}
                               for name in ["Father O'Brien", "The Blackwood Family"]}},
        {"world_updates": {"Location": {name: {"description": "Synthetic named place", "category": "Settlement", "goals": [], "rules": [], "key_elements": []}
                                         for name in ["The Hague", "King's Cross"]}}},
        {"world_updates": {"Event": {}}},
        {"kg_triples": [{"subject": "Father O'Brien", "predicate": "LOCATED_AT", "object_entity": "The Hague", "description": "A visit."}]},
    ]


async def run_extraction(monkeypatch: pytest.MonkeyPatch, state: NarrativeState, replies: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []
    iterator = iter(replies)

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(next(iterator))}, "finish_reason": "stop"}]})

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    monkeypatch.setattr(get_services(), "language_model", service)
    try:
        return await scene_extraction.extract_from_scenes(state), requests
    finally:
        await service.aclose()


@pytest.mark.parametrize("name", ["Neighbors", "Bystanders", "Travelers"])
@pytest.mark.parametrize("endpoint", ["subject", "object_entity", "character", "nested_relationship"])
async def test_generic_group_fails_entire_extraction_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, endpoint: str) -> None:
    state = selected_scene(tmp_path, SCENE.replace("Neighbors", name))
    replies = responses()
    if endpoint == "character":
        replies[0]["character_updates"][name] = {"description": "An unnamed crowd", "traits": [], "status": "Present", "relationships": {}}
    elif endpoint == "nested_relationship":
        replies[0]["character_updates"]["Father O'Brien"]["relationships"][name] = {"type": "ALLIES_WITH", "description": "An unsupported group"}
    else:
        replies[3]["kg_triples"].append({**replies[3]["kg_triples"][0], endpoint: name})
    result, requests = await run_extraction(monkeypatch, state, replies)
    assert requests
    assert result["extraction_status"] == "failed"
    assert result["has_fatal_error"] is True
    assert result["extracted_entities_ref"] is None
    assert result["extracted_relationships_ref"] is None
    assert not list((tmp_path / ".saga/content/extracted_relationships").glob("*.json"))
    failure = next(outcome for outcome in result["extraction_outcomes"] if outcome["status"] == "failed")
    assert failure["error_type"] == "ValueError"
    assert "eligible" in failure["error"]


async def test_catalog_names_ids_and_choices_survive_full_handoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = selected_scene(tmp_path)
    replies = responses()
    replies[0]["character_updates"]["The Blackwood Family"]["relationships"] = {"Father O'Brien": {"type": "ALLIES_WITH", "description": "Support"}}
    result, requests = await run_extraction(monkeypatch, state, replies)
    assert result["extraction_status"] == "complete"
    assert len(requests) == 4
    manager = ContentManager(str(tmp_path))
    entities = manager.load_json_strict(result["extracted_entities_ref"])
    relationships = manager.load_json_strict(result["extracted_relationships_ref"])
    catalog = {row["name"]: row for row in select_catalog(state).candidates("Character", "Location")}
    for row in entities["characters"] + entities["world_items"]:
        assert row["attributes"]["id"] == catalog[row["name"]]["id"]
    assert relationships[0]["source_id"] == catalog["Father O'Brien"]["id"]
    assert relationships[0]["target_id"] == "place-exact-17"
    assert relationships[0]["source_type"] == "Character"
    assert relationships[0]["target_type"] == "Location"
    expected = {"Father O'Brien", "The Blackwood Family", "The Hague", "King's Cross", "The Arrival"}
    for request in requests:
        prompt = request["messages"][-1]["content"]
        assert "ELIGIBLE SCENE IDENTITIES" in prompt
        assert SCENE in prompt
        assert "Absent Name" not in prompt
        assert request["temperature"] == 1.0
        assert request["max_tokens"] == 65536
    contract = requests[3]["response_format"]["json_schema"]
    assert contract["strict"] is False
    variants = contract["schema"]["$defs"]["SceneRelationship"]["oneOf"]
    assert len(variants) == 9
    assert {name for variant in variants for name in variant["properties"]["subject"]["enum"]} == expected
    assert {name for variant in variants for name in variant["properties"]["object_entity"]["enum"]} == expected


@pytest.mark.parametrize("case", ["missing", "corrupt", "wrong_parent"])
async def test_invalid_candidate_source_fails_before_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    state = selected_scene(tmp_path)
    manager = ContentManager(str(tmp_path))
    if case == "missing":
        state["initialization_catalog_ref"] = None
    elif case == "corrupt":
        reference = state["initialization_catalog_ref"]
        assert reference is not None
        (tmp_path / reference["path"]).write_text("{}", encoding="utf-8")
    else:
        state["character_sheets_ref"] = manager.save_json({"Ada": {"name": "Ada"}}, "character_sheets", "wrong", 1)
    result, requests = await run_extraction(monkeypatch, state, responses())
    assert result["extraction_status"] == "failed"
    assert result["extracted_relationships_ref"] is None
    assert requests == []


def test_capitalized_span_is_not_named_admission_without_candidates() -> None:
    with pytest.raises(ValueError, match="eligible"):
        validate_named_entity("Neighbors")
    assert not _validate_entity_with_spacy("Neighbors gathered at the community hall.", "Neighbors")


def test_import_provenance_and_settings() -> None:
    assert Path(scene_extraction.__file__).resolve() == Path(__file__).resolve().parents[1] / "core/langgraph/nodes/scene_extraction.py"
    assert config.STRUCTURED_OUTPUT_STRICT is False
    assert config.TEMPERATURE_OVERRIDE == 1.0
    assert config.MAX_KG_TRIPLE_TOKENS == 65536
    assert config.MAX_CONTEXT_TOKENS == 131072
