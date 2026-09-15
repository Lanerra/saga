"""Provider sampling controls do not relax local output admission."""
import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

import config
from config.settings import SagaSettings
from core.http_client_service import HTTPClientService
from core.langgraph.initialization.catalog import select_catalog
from core.langgraph.initialization.character_sheets_node import _character_sheet_contract
from core.langgraph.nodes.scene_extraction_parsing import SceneRelationships
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import create_llm_service
from tests.test_initialization_catalog import SyntheticSelector, selected_state


def test_requested_sampling_defaults() -> None:
    assert SagaSettings.model_fields["TEMPERATURE_OVERRIDE"].default == 1.0
    assert SagaSettings.model_fields["STRUCTURED_OUTPUT_STRICT"].default is False


@pytest.mark.parametrize("requested_temperature", [0.1, 0.3, 0.65, 0.7])
async def test_global_temperature_reaches_wire(requested_temperature: float) -> None:
    bodies: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "An answer."}, "finish_reason": "stop"}]})

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    try:
        await service.async_call_llm("synthetic", "A prompt.", temperature=requested_temperature, auto_clean_response=False)
        assert bodies[0]["temperature"] == 1.0
    finally:
        await service.aclose()


@pytest.mark.parametrize("strict", [False, True])
async def test_all_schema_producers_use_provider_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strict: bool) -> None:
    state = cast(NarrativeState, await selected_state(tmp_path, monkeypatch, SyntheticSelector()))
    effective = config.snapshot_settings().model_copy(update={"STRUCTURED_OUTPUT_STRICT": strict})
    with config.bind_settings(effective):
        catalog = select_catalog(state)
        contracts = [SceneRelationships.response_format(), _character_sheet_contract("Ada", ["Ada", "Bo"])]
        contracts.extend(catalog.response_format(name) for name in [
            "extract_outline_relationships", "catalog_possessions", "catalog_event_characters",
            "catalog_event_location", "catalog_event_items",
        ])
        assert all(contract["json_schema"]["strict"] is strict for contract in contracts)
        assert SceneRelationships.model_config["strict"] is True
        assert SceneRelationships.model_config["extra"] == "forbid"
