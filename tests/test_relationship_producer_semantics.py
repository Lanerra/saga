"""Offline producer/strict-admission parity; no model or grammar-engine proof."""

import json
from itertools import product
from pathlib import Path
from typing import Any

import httpx
import pytest

from core.exceptions import LLMServiceError
from core.http_client_service import HTTPClientService
from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization.catalog import EntityCatalog, materialize_entities, select_catalog
from core.langgraph.initialization.outline_relationships_node import extract_outline_relationships
from core.langgraph.initialization.snapshot import select_inputs
from core.langgraph.nodes import scene_extraction
from core.langgraph.nodes.commit_node import _build_relationship_statements
from core.langgraph.nodes.scene_extraction_parsing import SceneRelationships, scene_entity_response_format
from core.langgraph.nodes.scene_extraction_validation import scene_identity_candidates
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import create_llm_service
from core.relationship_validation import validate_relationship_semantics_strict
from core.service_context import get_services
from models.kg_constants import RELATIONSHIP_TYPES
from models.kg_models import WorldItem
from tests.test_catalog_output_contract import wire_accepts
from tests.test_r02_extraction_identity import entity
from tests.test_r02_extraction_identity import relationship as example_relationship
from tests.test_r08g_catalog_fixtures import catalog_state

LABELS = ("Character", "Location", "Item", "Event")
NAMES = ("Father O'Brien", "Miren", "St. James’ Gate (West)", "The King's Key", "The King's Return")
SCENE = "; ".join(NAMES) + "."


def selected_geometry(tmp_path: Path) -> tuple[NarrativeState, EntityCatalog]:
    state = catalog_state(tmp_path, characters=NAMES[:2], locations=(), events=(NAMES[4],))
    catalog = materialize_entities(select_inputs(state), [
        WorldItem(name=NAMES[2], category="location", description="Synthetic location", id="location-Exact.ID"),
        WorldItem(name=NAMES[3], category="object", description="Synthetic item", id="item-Exact.ID"),
    ], ())
    state["initialization_catalog_ref"] = ContentManager(str(tmp_path)).save_json(catalog.model_dump(mode="json"), "initialization_catalog", "semantic_geometry", 2)
    state["outline_relationships_ref"] = None
    return state, select_catalog(state)


def expected_predicates(source: str, target: str) -> list[str]:
    return sorted(predicate for predicate in RELATIONSHIP_TYPES if validate_relationship_semantics_strict(predicate, source, target)[0])


def relationship_variants(schema: dict[str, Any]) -> list[dict[str, Any]]:
    row = schema["properties"]["kg_triples"]["items"]
    if "$ref" in row:
        row = schema["$defs"][row["$ref"].rsplit("/", 1)[1]]
    return list(row["oneOf"])


def assert_predicate_parity(schema: dict[str, Any], candidates: dict[str, dict[str, Any]], producer: str) -> None:
    variants = relationship_variants(schema)
    labels = sorted({candidate["label"] for candidate in candidates.values()})
    assert len(variants) == len(labels) ** 2
    pairs = set()
    for variant in variants:
        properties = variant["properties"]
        assert variant["additionalProperties"] is False
        assert set(variant["required"]) == set(properties)
        if producer == "outline":
            source_label, = properties["source_label"]["enum"]
            target_label, = properties["target_label"]["enum"]
            for endpoint, label in (("source", source_label), ("target", target_label)):
                assert set(properties[endpoint + "_id"]["enum"]) == {row["id"] for row in candidates.values() if row["label"] == label}
            predicate_field = "relationship_type"
        else:
            source_labels = {candidates[name]["label"] for name in properties["subject"]["enum"]}
            target_labels = {candidates[name]["label"] for name in properties["object_entity"]["enum"]}
            assert len(source_labels) == len(target_labels) == 1
            source_label, = source_labels
            target_label, = target_labels
            for endpoint, label in (("subject", source_label), ("object_entity", target_label)):
                assert properties[endpoint]["enum"] == sorted(name for name, row in candidates.items() if row["label"] == label)
            predicate_field = "predicate"
        assert properties[predicate_field]["enum"] == expected_predicates(source_label, target_label)
        assert (source_label, target_label) not in pairs
        pairs.add((source_label, target_label))
    assert pairs == set(product(labels, repeat=2))


@pytest.mark.parametrize("source,target", product(LABELS, repeat=2))
def test_scene_predicate_enum_matches_strict_validator_for_every_pair(source: str, target: str) -> None:
    candidates = {"First Name": {"label": source, "id": "exact-first"}, "Second Name": {"label": target, "id": "exact-second"}}
    schema = SceneRelationships.response_format(candidates)["json_schema"]["schema"]
    assert_predicate_parity(schema, candidates, "scene")


def test_character_nested_predicates_match_strict_character_pair() -> None:
    schema = scene_entity_response_format("Character", NAMES[:2])["json_schema"]["schema"]
    for details in schema["properties"]["character_updates"]["properties"].values():
        for relationship in details["properties"]["relationships"]["properties"].values():
            assert relationship["properties"]["type"]["enum"] == expected_predicates("Character", "Character")


def test_catalog_predicates_match_strict_validator_without_identity_pair_expansion(tmp_path: Path) -> None:
    _, catalog = selected_geometry(tmp_path)
    candidates = {row["id"]: row for row in catalog.candidates(*LABELS)}
    schema = catalog.response_format("extract_outline_relationships")["json_schema"]["schema"]
    assert_predicate_parity(schema, candidates, "outline")
    for source, target in product(candidates.values(), repeat=2):
        for predicate in RELATIONSHIP_TYPES:
            row = {"source_id": source["id"], "source_label": source["label"], "target_id": target["id"], "target_label": target["label"], "relationship_type": predicate, "description": "Synthetic assertion"}
            assert wire_accepts(schema, {"kg_triples": [row]}) is validate_relationship_semantics_strict(predicate, source["label"], target["label"])[0]


@pytest.mark.parametrize("producer", ["scene", "characters", "outline"])
@pytest.mark.parametrize("mode", ["success", "json_retry", "fallback", "exhausted_json", "exhausted_fallback", "transport_retry"])
@pytest.mark.run_settings(LLM_RETRY_ATTEMPTS=2, STRUCTURED_OUTPUT_STRICT=False)
async def test_semantic_contract_survives_actual_http_serialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, producer: str, mode: str) -> None:
    state, catalog = selected_geometry(tmp_path)
    candidates = scene_identity_candidates(catalog, SCENE)
    bodies: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if mode == "transport_retry" and len(bodies) == 1:
            return httpx.Response(503)
        if mode == "exhausted_fallback" or (mode == "fallback" and len(bodies) == 1):
            return httpx.Response(400)
        raw = '{"character_updates": {}}' if producer == "characters" else '{"kg_triples": []}'
        if mode == "exhausted_json" or (mode == "json_retry" and len(bodies) == 1):
            raw = "not JSON"
        return httpx.Response(200, json={"choices": [{"message": {"content": raw}, "finish_reason": "stop"}]})

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    monkeypatch.setattr(get_services(), "language_model", service)
    try:
        async def produce() -> Any:
            if producer == "outline":
                return await extract_outline_relationships(state)
            extractor = scene_extraction._extract_characters_from_scene if producer == "characters" else scene_extraction._extract_relationships_from_scene
            return await extractor(SCENE, 0, 1, "Synthetic", "Fantasy", NAMES[0], "synthetic", catalog=catalog)

        failure = mode.startswith("exhausted") or (producer == "outline" and mode in {"json_retry", "fallback"})
        if failure:
            with pytest.raises((ValueError, LLMServiceError)):
                await produce()
        else:
            await produce()
        expected_calls = 1 if mode == "success" or (producer == "outline" and mode != "transport_retry") else 2
        assert len(bodies) == expected_calls
        contract = bodies[0]["response_format"]
        assert contract["json_schema"]["strict"] is False
        assert all(body["response_format"] == contract for body in bodies)
        assert all(body["temperature"] == 1.0 for body in bodies)
        schema = contract["json_schema"]["schema"]
        if producer == "characters":
            listing = bodies[0]["messages"][-1]["content"].split("Allowed Character-to-Character relationship types (use ONLY these):", 1)[1].split("Example:", 1)[0]
            assert [line.removeprefix("- ") for line in listing.splitlines() if line.startswith("- ")] == expected_predicates("Character", "Character")
            for details in schema["properties"]["character_updates"]["properties"].values():
                for relationship in details["properties"]["relationships"]["properties"].values():
                    assert relationship["properties"]["type"]["enum"] == expected_predicates("Character", "Character")
        else:
            eligible = {row["id"]: row for row in catalog.candidates(*LABELS)} if producer == "outline" else candidates
            assert_predicate_parity(schema, eligible, producer)
        if mode == "success":
            path = tmp_path / f"actual-wire-{producer}.json"
            path.write_text(json.dumps(bodies, indent=2) + "\n")
            (tmp_path / f"wire-schema-{producer}.json").write_text(json.dumps(schema, indent=2) + "\n")
            print("ACTUAL_SERIALIZED_HTTP_REQUESTS", path)
    finally:
        await service.aclose()


def scene_predicates(schema: dict[str, Any], subject: str, target: str) -> set[str]:
    return {
        predicate for variant in relationship_variants(schema)
        if subject in variant["properties"]["subject"]["enum"] and target in variant["properties"]["object_entity"]["enum"]
        for predicate in variant["properties"]["predicate"]["enum"]
    }


def test_scene_all_predicates_and_exact_names_not_ids(tmp_path: Path) -> None:
    _, catalog = selected_geometry(tmp_path)
    candidates = scene_identity_candidates(catalog, SCENE)
    schema = SceneRelationships.response_format(candidates)["json_schema"]["schema"]
    for source, target in product(candidates.values(), repeat=2):
        assert scene_predicates(schema, source["name"], target["name"]) == set(expected_predicates(source["label"], target["label"]))
        assert scene_predicates(schema, source["id"], target["name"]) == set()
        assert scene_predicates(schema, source["name"], target["id"]) == set()
        assert scene_predicates(schema, source["name"] + " ", target["name"]) == set()
        assert scene_predicates(schema, source["name"], target["name"].replace("’", "'")) == (scene_predicates(schema, source["name"], target["name"]) if "’" not in target["name"] else set())


@pytest.mark.parametrize("source,predicate,target", [
    ("Pell", "OWNS", "Pell"), ("Miren", "OWNS", "Miren"),
    ("Iona", "AFFECTED_BY", "Corran"), ("Iona", "AFFECTED_BY", "Miren"), ("Pell", "AFFECTS_LOCATION", "Corran"),
])
@pytest.mark.run_settings(STRUCTURED_OUTPUT_STRICT=False)
async def test_captured_bad_triples_excluded_from_wire_and_rejected_at_commit(source: str, predicate: str, target: str) -> None:
    names = sorted({source, target})
    candidates = {name: {"label": "Character", "id": "exact-" + name} for name in names}
    contract = SceneRelationships.response_format(candidates)
    assert contract["json_schema"]["strict"] is False
    assert predicate not in scene_predicates(contract["json_schema"]["schema"], source, target)
    character_schema = scene_entity_response_format("Character", candidates)["json_schema"]["schema"]
    nested = character_schema["properties"]["character_updates"]["properties"][source]["properties"]["relationships"]["properties"][target]
    assert not wire_accepts(nested, {"type": predicate, "description": "Captured semantic mismatch"})
    assert validate_relationship_semantics_strict(predicate, "Character", "Character")[0] is False
    with pytest.raises(ValueError, match="Relationship semantic validation failed"):
        await _build_relationship_statements(
            [example_relationship(source, target, predicate, source_type="Character", target_type="Character")],
            [entity(name, "Character", id=candidates[name]["id"]) for name in names], [], {}, {}, 1, False,
        )


async def test_valid_cross_label_rows_preserve_exact_names_ids_and_predicates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, catalog = selected_geometry(tmp_path)
    candidates = scene_identity_candidates(catalog, SCENE)
    triples = [
        {"subject": NAMES[source], "predicate": predicate, "object_entity": NAMES[target], "description": "Synthetic valid assertion"}
        for source, predicate, target in [(0, "OWNS", 3), (3, "LOCATED_AT", 2), (0, "PARTICIPATES_IN", 4), (4, "OCCURS_AT", 2)]
    ]
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"kg_triples": triples})}, "finish_reason": "stop"}]})

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    monkeypatch.setattr(get_services(), "language_model", service)
    try:
        rows = await scene_extraction._extract_relationships_from_scene(SCENE, 0, 1, "Synthetic", "Fantasy", NAMES[0], "synthetic", catalog=catalog)
        assert len(requests) == 1 and len(rows) == len(triples)
        for row, triple in zip(rows, triples, strict=True):
            assert (row["source_name"], row["relationship_type"], row["target_name"], row["description"]) == (triple["subject"], triple["predicate"], triple["object_entity"], triple["description"])
            assert (row["source_id"], row["target_id"]) == (candidates[triple["subject"]]["id"], candidates[triple["object_entity"]]["id"])
            assert validate_relationship_semantics_strict(row["relationship_type"], row["source_type"], row["target_type"])[0]
            assert row["relationship_type"] in scene_predicates(requests[0]["response_format"]["json_schema"]["schema"], row["source_name"], row["target_name"])
    finally:
        await service.aclose()


def test_schema_growth_is_label_bounded_and_unruled_predicates_survive() -> None:
    candidates = {f"{label} Name {index}": {"label": label, "id": f"{label}-{index}"} for label in LABELS for index in range(40)}
    schema = SceneRelationships.response_format(candidates)["json_schema"]["schema"]
    assert_predicate_parity(schema, candidates, "scene")
    serialized = json.dumps(schema)
    assert all(serialized.count(json.dumps(name)) == 8 for name in candidates)


def test_scene_schema_rejects_nonsemantic_candidate_labels() -> None:
    with pytest.raises(ValueError, match="selected semantic labels"):
        SceneRelationships.response_format({"Chapter Name": {"label": "Chapter", "id": "chapter-1"}})


def test_character_prompt_types_and_examples_match_wire(tmp_path: Path) -> None:
    from prompts.prompt_renderer import render_prompt

    prompt = render_prompt("knowledge_agent/extract_characters.j2", {
        "eligible_entities": [], "chapter_number": 1, "novel_title": "Synthetic", "novel_genre": "Fantasy", "protagonist": "Elara", "chapter_text": "Elara.",
        "canonical_relationship_types": expected_predicates("Character", "Character"),
    })
    listing = prompt.split("Allowed Character-to-Character relationship types (use ONLY these):", 1)[1].split("Example:", 1)[0]
    assert [line.removeprefix("- ") for line in listing.splitlines() if line.startswith("- ")] == expected_predicates("Character", "Character")
    example = json.loads(prompt.split("Example:\n", 1)[1].split("Per-character entry requirements:", 1)[0])
    for details in example["character_updates"].values():
        assert all(row["type"] in expected_predicates("Character", "Character") for row in details["relationships"].values())
