"""Constrained producers still require strict application admission."""
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest

from core.http_client_service import HTTPClientService
from core.langgraph.initialization.catalog import select_catalog
from core.langgraph.initialization.character_sheets_node import _admit_character_sheet, _character_sheet_contract, _generate_character_sheet, generate_character_sheets
from core.langgraph.initialization.outline_relationships_node import extract_outline_relationships
from core.langgraph.initialization.snapshot import encoded
from core.langgraph.initialization.staged_import import InitializationImport
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import create_llm_service
from core.relationship_validation import validate_relationship_semantics_strict
from core.service_context import get_services
from models.kg_constants import RELATIONSHIP_TYPES
from tests.test_initialization_catalog import SyntheticSelector, selected_state
from tests.test_staged_initialization import example_state, with_catalog


def wire_accepts(schema: dict[str, Any], value: Any) -> bool:
    """Evaluate the selector's JSON Schema subset without optional dependencies."""
    assert set(schema) <= {"type", "enum", "oneOf", "properties", "required", "additionalProperties", "items", "maxItems"}
    if "oneOf" in schema and sum(wire_accepts(branch, value) for branch in schema["oneOf"]) != 1:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    types = {"string": isinstance(value, str), "null": value is None, "object": isinstance(value, dict), "array": isinstance(value, list)}
    if "type" in schema:
        allowed = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(types[name] for name in allowed):
            return False
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if not set(schema.get("required", [])).issubset(value):
            return False
        if schema.get("additionalProperties") is False and not set(value).issubset(properties):
            return False
        if not all(wire_accepts(properties[name], item) for name, item in value.items() if name in properties):
            return False
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return False
        if "items" in schema and not all(wire_accepts(schema["items"], item) for item in value):
            return False
    return True


@pytest.mark.parametrize("endpoint", ["source", "target"])
async def test_relationship_wire_couples_literal_identity_and_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str) -> None:
    state = cast(NarrativeState, await selected_state(tmp_path, monkeypatch, SyntheticSelector()))
    catalog = select_catalog(state)
    schema = catalog.response_format("extract_outline_relationships")["json_schema"]["schema"]
    candidates = catalog.candidates("Character", "Location", "Item", "Event")
    serialized = json.dumps(schema)
    # A bounded label vocabulary must not expand into per-entity-pair alternatives.
    assert all(serialized.count(json.dumps(candidate["id"])) <= 8 for candidate in candidates)
    for source in candidates:
        for target in candidates:
            predicate = next(predicate for predicate in sorted(RELATIONSHIP_TYPES) if validate_relationship_semantics_strict(predicate, source["label"], target["label"])[0])
            row = {"source_id": source["id"], "source_label": source["label"], "target_id": target["id"], "target_label": target["label"], "relationship_type": predicate, "description": "Synthetic assertion"}
            assert wire_accepts(schema, {"kg_triples": [row]})
            for wrong_label in {"Character", "Location", "Item", "Event"} - {row[endpoint + "_label"]}:
                bad = row | {endpoint + "_label": wrong_label}
                with pytest.raises(ValueError, match="Unknown catalog ID or wrong label"):
                    catalog.endpoint(bad[endpoint + "_id"], wrong_label)
                assert not wire_accepts(schema, {"kg_triples": [bad]}), bad
    assert wire_accepts(schema, {"kg_triples": []})
    for candidate in catalog.candidates("Chapter", "Scene"):
        assert not wire_accepts(schema, {"kg_triples": [row | {endpoint + "_id": candidate["id"]}]})


async def test_location_prompt_projection_omits_world_storage_discriminator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = cast(NarrativeState, await selected_state(tmp_path, monkeypatch, SyntheticSelector()))
    catalog = select_catalog(state)
    before = catalog.model_dump_json()
    original = catalog.candidates("Location")[0]
    assert original["label"] == "Location"
    assert original["type"] == "Item"
    projected = catalog.model_candidates("Location")[0]
    assert "type" not in projected
    for field in ("id", "label", "name", "category", "description"):
        assert projected[field] == original[field]
    assert catalog.candidates("Location")[0] == original
    assert catalog.model_dump_json() == before


async def test_relationship_contract_reaches_serialized_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = with_catalog(example_state(tmp_path))
    state['outline_relationships_ref'] = None
    bodies: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'content': '{"kg_triples": []}'}, 'finish_reason': 'stop'}]})

    transport = HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    service = create_llm_service(transport)
    monkeypatch.setattr(get_services(), 'language_model', service)
    try:
        await extract_outline_relationships(state)
        contract = bodies[0]['response_format']
        assert contract['type'] == 'json_schema'
        assert contract['json_schema']['strict'] is False
        assert bodies[0]['temperature'] == 1.0
        schema = contract['json_schema']['schema']
        catalog = select_catalog(state)
        assert encoded(catalog.model_candidates('Character', 'Location', 'Item', 'Event')) in bodies[0]['messages'][-1]['content']
        character = catalog.candidates('Character')[0]['id']
        event = catalog.candidates('Event')[0]['id']
        row = {'source_id': character, 'source_label': 'Character', 'target_id': event, 'target_label': 'Event', 'relationship_type': 'PARTICIPATES_IN', 'description': 'Synthetic assertion'}
        assert wire_accepts(schema, {'kg_triples': []})
        assert wire_accepts(schema, {'kg_triples': [row] * 20})
        assert not wire_accepts(schema, {'kg_triples': [row] * 21})
        assert not wire_accepts(schema, {})
        assert not wire_accepts(schema, {'kg_triples': [], 'extra': True})
        for field in row:
            assert not wire_accepts(schema, {'kg_triples': [{key: value for key, value in row.items() if key != field}]})
            assert not wire_accepts(schema, {'kg_triples': [row | {field: None}]})
        for bad in [row | {'extra': True}, row | {'relationship_type': 'AFFECTS'}, row | {'source_label': 'Event'}, row | {'target_label': 'Character'}, row | {'target_id': event.lower() + ' '}]:
            assert not wire_accepts(schema, {'kg_triples': [bad]})
        # A subsequent narrative request must not inherit the extraction contract.
        await service.async_call_llm('synthetic', 'Write prose.', max_tokens=100, auto_clean_response=False)
        assert 'response_format' not in bodies[1]
    finally:
        await transport.aclose()


async def test_every_catalog_selector_has_typed_choices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SyntheticSelector()
    state = cast(NarrativeState, await selected_state(tmp_path, monkeypatch, provider))
    calls: list[dict[str, Any]] = []

    async def record(**arguments: Any) -> tuple[str, dict[str, Any]]:
        calls.append(arguments)
        return await provider(**arguments)

    monkeypatch.setattr(get_services().language_model, 'async_call_llm', record)
    state.update(await extract_outline_relationships(state))
    await InitializationImport(str(tmp_path)).prepare(state)
    catalog = select_catalog(state)
    expected = {
        'catalog_possessions': ('possessions', {'character_id': 'Character', 'item_id': 'Item'}),
        'catalog_event_characters': (None, {'character_id': 'Character'}),
        'catalog_event_location': ('location_id', {'location_id': 'Location'}),
        'catalog_event_items': ('featured_items', {'item_id': 'Item'}),
    }
    observed = set()
    for call in calls[1:]:
        contract = call['response_format']['json_schema']
        name = contract['name']
        observed.add(name)
        wrapper, fields = expected[name]
        schema = contract['schema']
        row = schema if wrapper == 'location_id' else schema['items'] if wrapper is None else schema['properties'][wrapper]['items']
        assert row['additionalProperties'] is False
        assert set(row['required']) == set(row['properties'])
        for field, label in fields.items():
            assert encoded(catalog.model_candidates(label)) in call['prompt']
            choices = row['properties'][field]['enum']
            assert choices == [entity.identity for entity in catalog.entities if entity.label == label] + ([None] if field == 'location_id' else [])
        if 'role' in row['properties']:
            assert row['properties']['role']['type'] == ['string', 'null']
        sample = {field: catalog.candidates(label)[0]['id'] for field, label in fields.items()}
        if 'role' in row['properties']:
            sample['role'] = None
        assert wire_accepts(row, sample)
        empty = {'location_id': None} if wrapper == 'location_id' else [] if wrapper is None else {wrapper: []}
        assert wire_accepts(schema, empty)
        for field, label in fields.items():
            for candidate in catalog.candidates('Character', 'Location', 'Item', 'Event'):
                assert wire_accepts(row, sample | {field: candidate['id']}) is (candidate['label'] == label)
    assert observed == set(expected)


@pytest.mark.parametrize('defect', ['duplicate', 'missing', 'unknown_field', 'unknown_id', 'wrong_label', 'too_many'])
async def test_raw_relationship_rejection_not_repaired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    state = with_catalog(example_state(tmp_path))
    state['outline_relationships_ref'] = None
    catalog = select_catalog(state)
    character = catalog.candidates('Character')[0]['id']
    event = catalog.candidates('Event')[0]['id']
    row = {'source_id': character, 'source_label': 'Character', 'target_id': event, 'target_label': 'Event', 'relationship_type': 'OCCURS_AT', 'description': 'Synthetic assertion'}
    if defect == 'missing':
        del row['source_label']
    elif defect == 'unknown_field':
        row['untrusted'] = 'assertion'
    elif defect == 'unknown_id':
        row['target_id'] = event.replace('event_', 'entity_')
    elif defect == 'wrong_label':
        row['target_label'] = 'Character'
    response = json.dumps({'kg_triples': [row] * (21 if defect == 'too_many' else 1)})
    if defect == 'duplicate':
        response = response.replace('"source_id":', '"source_id": "invented", "source_id":')
    monkeypatch.setattr(get_services().language_model, 'async_call_llm', AsyncMock(return_value=(response, {})))
    with pytest.raises(ValueError):
        await extract_outline_relationships(state)
    assert state['outline_relationships_ref'] is None
    assert not list((tmp_path / '.saga/initialization').glob('relationships-*.json'))


def test_model_candidates_compact_only_storage_metadata(tmp_path: Path) -> None:
    catalog = select_catalog(with_catalog(example_state(tmp_path)))
    before = catalog.model_dump_json()
    full = catalog.candidates('Character', 'Event')
    compact = catalog.model_candidates('Character', 'Event')
    assert len(json.dumps(compact)) < len(json.dumps(full))
    for original, candidate in zip(full, compact, strict=True):
        assert candidate['id'] == original['id']
        assert candidate['label'] == original['label']
        assert candidate['name'] == original['name']
        for key, value in candidate.items():
            assert value == original[key]
        for field in ('description', 'cause', 'effect', 'event_type', 'sequence_in_act', 'personality_description', 'motivations', 'background'):
            if original.get(field):
                assert candidate[field] == original[field]
    assert catalog.model_dump_json() == before


@pytest.mark.parametrize('predicate', ['AFFECTS', 'TRUSTS'])
async def test_character_relationship_contract_precedes_retention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, predicate: str) -> None:
    state = example_state(tmp_path)
    raw = {'name': 'Ada', 'description': 'An explorer', 'traits': ['brave'], 'status': 'Active', 'motivations': 'Discover', 'background': 'Harbor', 'skills': ['navigation'], 'relationships': {'Bea': {'type': predicate, 'description': 'Synthetic assertion'}}, 'internal_conflict': 'Duty'}
    calls: list[dict[str, Any]] = []

    async def respond(**arguments: Any) -> tuple[str, dict[str, Any]]:
        calls.append(arguments)
        return json.dumps(raw), {}

    monkeypatch.setattr(get_services().language_model, 'async_call_llm', respond)
    result = await _generate_character_sheet(state, 'Ada', ['Ada', 'Bea'])
    if predicate == 'AFFECTS':
        assert result is None
    else:
        assert result is not None and result['relationships'] == raw['relationships']
    contract = calls[0]['response_format']['json_schema']
    assert contract['strict'] is False
    schema = contract['schema']
    assert schema['properties']['name']['enum'] == ['Ada']
    relations = schema['properties']['relationships']
    assert set(relations['properties']) == {'Bea'}
    assert relations['additionalProperties'] is False
    assert 'AFFECTS' not in relations['properties']['Bea']['properties']['type']['enum']
    assert calls[0]['auto_clean_response'] is False


@pytest.mark.parametrize('change', [
    {'name': 'Other'}, {'traits': ['two words']}, {'skills': [1]}, {'unknown': True},
    {'relationships': {'Other': {'type': 'TRUSTS', 'description': 'test'}}},
    {'relationships': {'Bea': {'type': 'TRUSTS', 'description': 1}}},
])
def test_character_sheet_rejects_contract_violations(change: dict[str, Any]) -> None:
    raw = {'name': 'Ada', 'description': 'test', 'traits': ['Brave'], 'status': 'Active', 'motivations': '', 'background': '', 'skills': [], 'relationships': {}, 'internal_conflict': ''}
    contract = _character_sheet_contract('Ada', ['Bea'])
    assert _admit_character_sheet(json.dumps(raw), contract)['traits'] == ['Brave']
    with pytest.raises(ValueError):
        _admit_character_sheet(json.dumps(raw | change), contract)
    with pytest.raises(ValueError):
        _admit_character_sheet(json.dumps(raw)[:-1] + ',"name":"Ada"}', contract)


async def test_character_collection_cannot_drop_failed_sheet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = example_state(tmp_path)
    state['genre'] = 'Adventure'
    state['protagonist_name'] = 'Ada'
    async def respond(**arguments: Any) -> tuple[str, dict[str, Any]]:
        if 'response_format' not in arguments:
            return '["Ada","Bea","Cora"]', {}
        name = arguments['response_format']['json_schema']['schema']['properties']['name']['enum'][0]
        if name != 'Ada':
            return '', {}
        return json.dumps({'name': 'Ada', 'description': 'test', 'traits': [], 'status': 'Active', 'motivations': '', 'background': '', 'skills': [], 'relationships': {}, 'internal_conflict': ''}), {}
    monkeypatch.setattr(get_services().language_model, 'async_call_llm', respond)
    result = await generate_character_sheets(state)
    assert result['last_error']
    assert 'character_sheets_ref' not in result
