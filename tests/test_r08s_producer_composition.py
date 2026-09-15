"""Real initialization/enrichment/extraction composed with only an HTTP transport fake."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import config
from core.http_client_service import HTTPClientService
from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization.catalog import materialize_initialization_catalog, select_catalog
from core.langgraph.initialization.chapter_outline_node import generate_chapter_outline
from core.langgraph.nodes.scene_extraction import extract_from_scenes
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import create_llm_service
from core.service_context import get_services
from tests.test_staged_initialization import example_state


async def prepare_enriched_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[NarrativeState, list[dict[str, Any]]]:
    state = example_state(tmp_path)
    state["outline_relationships_ref"] = None
    state["current_chapter"] = 1
    requests: list[dict[str, Any]] = []
    responses = iter([
        [{"name": "Weather Station", "category": "location", "description": "Synthetic station"}],
        {"scene_description": "Ada visits Weather Station.", "key_beats": ["Ada chooses", "Ada visits", "Ada remains"], "plot_point": "Choice"},
    ])

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(next(responses))}, "finish_reason": "stop"}]})

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    monkeypatch.setattr(get_services(), "language_model", service)
    try:
        state.update(await materialize_initialization_catalog(state))
        original = state["chapter_outlines_ref"]
        select_catalog(state)
        update = await generate_chapter_outline(state)
        assert update.get("last_error") is None
        state.update(update)
        assert state["chapter_outlines_ref"] != original
        manager = ContentManager(str(tmp_path))
        assert original is not None
        assert manager.load_json_strict(original)["1"]["version"] == 0
        state["scene_drafts_ref"] = manager.save_json(["Ada chooses to visit Weather Station.", "Ada chooses to remain at Weather Station."], "scene_drafts", "chapter_1", 1)
        state["chapter_plan_scene_count"] = 2
        return state, requests
    finally:
        await service.aclose()


async def extract_selected(state: NarrativeState, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []
    replies = iter([reply for description in ["Ada visits", "Ada remains"] for reply in [
        {"character_updates": {"Ada": {"description": description, "traits": [], "status": "Active", "relationships": {}}}},
        {"world_updates": {"Location": {}}}, {"world_updates": {"Event": {}}},
        {"kg_triples": [{"subject": "Ada", "predicate": "LOCATED_AT", "object_entity": "Weather Station", "description": description}]},
    ]])

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(next(replies))}, "finish_reason": "stop"}]})

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    monkeypatch.setattr(get_services(), "language_model", service)
    try:
        return await extract_from_scenes(state), requests
    finally:
        await service.aclose()


async def test_catalog_enrichment_extraction_real_composition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state, requests = await prepare_enriched_state(tmp_path, monkeypatch)
    result, extraction_requests = await extract_selected(state, monkeypatch)
    assert result["extraction_status"] == "complete", result.get("last_error")
    assert len(extraction_requests) == 8
    manager = ContentManager(str(tmp_path))
    relationship = manager.load_json_strict(result["extracted_relationships_ref"])[0]
    assert relationship["description"] == "Ada remains"
    assert relationship["scene_index"] == 1
    assert [row["scene_index"] for row in relationship["scene_assertions"]] == [0, 1]
    assert [row["description"] for row in relationship["scene_assertions"]] == ["Ada visits", "Ada remains"]
    from core.langgraph.nodes.commit_node import _build_relationship_statements
    from core.langgraph.state import ExtractedRelationship

    _, parameters = (await _build_relationship_statements([ExtractedRelationship(**relationship)], [], [], {}, {}, 1, False))[1]
    assert parameters["subject_id"] == relationship["source_id"]
    assert parameters["object_id"] == relationship["target_id"]
    assert json.loads(parameters["relationship_properties"]["scene_assertions"]) == relationship["scene_assertions"]
    for request in requests + extraction_requests:
        assert request["temperature"] == 1.0
        assert request["max_tokens"] == 65536
        if "response_format" in request and request["response_format"]["type"] == "json_schema":
            assert request["response_format"]["json_schema"]["strict"] is False
    assert config.MAX_CONTEXT_TOKENS == 131072


@pytest.mark.parametrize("case", ["source_corrupt", "catalog_corrupt", "foreign_project", "wrong_parent"])
async def test_retained_authority_still_verifies_sources_and_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    state, _ = await prepare_enriched_state(tmp_path, monkeypatch)
    manager = ContentManager(str(tmp_path))
    reference = state["initialization_catalog_ref"]
    assert reference is not None
    catalog = manager.load_json_strict(reference)
    if case == "source_corrupt":
        source = next(artifact for artifact in catalog["inputs"]["artifacts"] if artifact["content_type"] == "chapter_outlines")
        (tmp_path / source["path"]).write_text("{}")
    elif case == "catalog_corrupt":
        (tmp_path / reference["path"]).write_text("{}")
    elif case == "foreign_project":
        state["graph_project_id"] = "f" * 64
    else:
        state["character_sheets_ref"] = manager.save_json({"Ada": {"name": "Ada"}}, "character_sheets", "wrong", 1)
    result, requests = await extract_selected(state, monkeypatch)
    assert result["extraction_status"] == "failed"
    assert result["extracted_relationships_ref"] is None
    assert requests == []


async def test_unadmitted_planned_character_fails_before_extraction_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state, _ = await prepare_enriched_state(tmp_path, monkeypatch)
    manager = ContentManager(str(tmp_path))
    state["chapter_plan_ref"] = manager.save_json([{"characters": ["Ada", "Novel Person"]}], "chapter_plan", "chapter_1", 1)
    result, requests = await extract_selected(state, monkeypatch)
    assert result["extraction_status"] == "failed"
    assert "upstream admission" in result["last_error"]
    assert requests == []


@pytest.mark.parametrize("novel", [False, True])
async def test_planner_catalog_contract_precedes_stub_writes_and_extraction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, novel: bool) -> None:
    from core.langgraph.nodes.scene_planning_node import plan_scenes

    state, _ = await prepare_enriched_state(tmp_path, monkeypatch)
    requests: list[dict[str, Any]] = []
    database_calls: list[str] = []
    count = config.TARGET_SCENES_MIN
    names = ["Ada", "Novel Person"] if novel else ["Ada"]
    plan = [{"title": f"Scene {index}", "pov_character": "Ada", "setting": "Weather Station", "characters": names,
             "plot_point": "Choice", "conflict": "Uncertainty", "outcome": "A decision",
             "beats": ["Ada chooses", "Ada visits", "Ada remains"] if index == 0 else []} for index in range(count)]

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(plan)}, "finish_reason": "stop"}]})

    async def read_graph(query: str, parameters: Any = None, **kwargs: Any) -> list[dict[str, Any]]:
        database_calls.append(query)
        return [{"name": "Ada"}]

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    monkeypatch.setattr(get_services(), "language_model", service)
    monkeypatch.setattr(get_services().database, "execute_read_query", read_graph)
    try:
        result = await plan_scenes(state)
    finally:
        await service.aclose()
    if novel:
        assert result["has_fatal_error"] is True
        assert "upstream admission" in (result["last_error"] or "")
        assert database_calls == []
        assert result["chapter_plan_ref"] is None
    else:
        assert not result.get("has_fatal_error"), result.get("last_error")
        state.update(result)
        manager = ContentManager(str(tmp_path))
        plan_reference = state["chapter_plan_ref"]
        assert plan_reference is not None
        assert manager.load_json_strict(plan_reference) == plan
    assert all("ELIGIBLE PLANNED CHARACTER IDENTITIES" in request["messages"][-1]["content"] for request in requests)


@pytest.mark.parametrize("existing", [False, True])
async def test_admitted_provisional_stub_preserves_identity_and_existing_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool) -> None:
    from core.langgraph.nodes.scene_planning_node import _ensure_scene_characters_exist

    state, _ = await prepare_enriched_state(tmp_path, monkeypatch)
    catalog = select_catalog(state, retained_chapter_outline=True)
    eligible = {candidate["name"]: candidate for candidate in catalog.candidates("Character")}
    writes: list[tuple[str, dict[str, Any]]] = []

    async def read(query: str, parameters: Any = None) -> list[dict[str, Any]]:
        assert query == "MATCH (c:Character) RETURN c.name AS name ORDER BY c.name"
        return [{"name": "Ada"}] if existing else []

    async def batch(statements: list[tuple[str, dict[str, Any]]]) -> None:
        writes.extend(statements)

    monkeypatch.setattr(get_services().database, "execute_read_query", read)
    monkeypatch.setattr(get_services().database, "execute_cypher_batch", batch)
    await _ensure_scene_characters_exist([{"characters": ["Ada"]}], 1, eligible_characters=eligible)
    if existing:
        assert writes == []
    else:
        assert len(writes) == 1
        assert writes[0][1]["id"] == eligible["Ada"]["id"]
        assert writes[0][1]["name"] == "Ada"
        assert writes[0][1]["is_provisional"] is True
