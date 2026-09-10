"""Constrained producers still require strict application admission."""
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest

from core.http_client_service import HTTPClientService
from core.langgraph.initialization.catalog import select_catalog
from core.langgraph.initialization.outline_relationships_node import extract_outline_relationships
from core.langgraph.initialization.staged_import import InitializationImport
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import create_llm_service
from core.service_context import get_services
from tests.test_initialization_catalog import SyntheticSelector, selected_state
from tests.test_staged_initialization import example_state, with_catalog


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
        assert contract['json_schema']['strict'] is True
        schema = contract['json_schema']['schema']
        assert schema['required'] == ['kg_triples']
        assert schema['additionalProperties'] is False
        rows = schema['properties']['kg_triples']
        assert rows['maxItems'] == 20
        properties = rows['items']['properties']
        assert set(rows['items']['required']) == set(properties)
        assert rows['items']['additionalProperties'] is False
        catalog = select_catalog(state)
        assert properties['source_id']['enum'] == [entity.identity for entity in catalog.entities if entity.label in {'Character', 'Location', 'Item', 'Event'}]
        assert properties['target_id'] == properties['source_id']
        assert properties['source_label']['enum'] == ['Character', 'Location', 'Item', 'Event']
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
            choices = row['properties'][field]['enum']
            assert choices == [entity.identity for entity in catalog.entities if entity.label == label] + ([None] if field == 'location_id' else [])
        if 'role' in row['properties']:
            assert row['properties']['role']['type'] == ['string', 'null']
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
