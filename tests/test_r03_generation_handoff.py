"""Synthetic wire-to-scene/context/assembly/SQLite handoff, not novel acceptance."""
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from langgraph.graph import END, StateGraph  # type: ignore[attr-defined]

import config
from core.http_client_service import HTTPClientService
from core.langgraph.content_manager import ContentManager, get_scene_drafts
from core.langgraph.initialization.catalog import select_catalog
from core.langgraph.nodes.assemble_chapter_node import assemble_chapter
from core.langgraph.state import NarrativeState
from core.langgraph.subgraphs.generation import create_generation_subgraph
from core.langgraph.workflow import create_checkpointer
from core.llm_interface_refactored import create_llm_service
from core.service_context import RunServices, get_services, inject_services
from data_access.cache_coordinator import clear_all_data_access_caches
from prompts.prompt_data_getters import clear_context_cache
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager
from tests.fakes.graph_ownership import OwnershipDriver
from tests.test_generation_failure_contract import seeded_state
from tests.test_r03_context_contracts import SCENE
from tests.test_r08t_migration_contracts import selected_authority_state


@pytest.mark.parametrize("draft_response", ["prose", "reasoning", "length"])
@pytest.mark.run_settings(TARGET_SCENES_MIN=2, DEFAULT_PROTAGONIST_NAME="Hero")
async def test_generation_handoff_and_sqlite_reopen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, draft_response: str, caplog: pytest.LogCaptureFixture) -> None:
    database = FakeNeo4jManager()
    database.configure_response(r"RETURN c.name AS name", [{"name": "Hero"}])
    owner = get_services().database
    monkeypatch.setattr(owner, "execute_read_query", database.execute_read_query)
    monkeypatch.setattr(owner, "execute_write_query", database.execute_write_query)
    monkeypatch.setattr(owner, "execute_cypher_batch", database.execute_cypher_batch)
    prose = ["Iven's ledger's clasp wouldn't open.\n\n“You’d know,” she said.", "He'd tried.  “Don’t,” she said.\n\nThe sailors' map stayed shut."]
    plan = [SCENE, {**SCENE, "title": "Departure", "beats": ["Hero leaves"]}]
    bodies: list[dict[str, Any]] = []
    draft_calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal draft_calls
        body = json.loads(request.content)
        if request.url.path == "/api/embeddings":
            return httpx.Response(200, json={"embedding": [0.25] * config.EXPECTED_EMBEDDING_DIM})
        bodies.append(body)
        if body["messages"][-1]["content"].startswith("You are breaking a chapter outline into"):
            text = json.dumps(plan)
            finish = "stop"
        else:
            text = prose[draft_calls] if draft_response != "reasoning" else "<think>reasoning-canary"
            draft_calls += 1
            finish = "length" if draft_response == "length" else "stop"
        return httpx.Response(200, json={"choices": [{"message": {"content": text, "reasoning_content": "not-an-answer"}, "finish_reason": finish}]})

    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    workflow = StateGraph(NarrativeState)
    workflow.add_node("generation", create_generation_subgraph())
    workflow.set_entry_point("generation")
    workflow.add_node("assembly", assemble_chapter)
    workflow.add_conditional_edges("generation", lambda state: "failure" if state.get("has_fatal_error") else "success", {"failure": END, "success": "assembly"})
    workflow.add_edge("assembly", END)
    checkpoint = str(tmp_path / "r03.sqlite")
    run = {"configurable": {"thread_id": "r03-synthetic"}}
    state = seeded_state(tmp_path)
    state = selected_authority_state(tmp_path, ("Hero",), event_names=("Hero opens the door", "Hero leaves"), initial_state=state)
    catalog = select_catalog(state)
    character = catalog.candidates("Character")[0]
    events = catalog.candidates("Event")
    database.configure_response(r"RETURN\s+c,\s+coalesce\(c.traits", [{"c": {key: value for key, value in character.items() if key != "label"}, "traits": character["traits"], "relationships": []}])
    database.configure_response(r"AS current_status", [{"description": character["personality_description"], "current_status": character["status"], "char_is_provisional": False, "provisional_rel_count": 0}])
    database.configure_response(r"RETURN ni.theme AS value", [{"value": state["theme"]}])
    database.configure_response(r"RETURN ni.central_conflict AS value", [{"value": "Locked door"}])
    database.configure_response(r"as major_points", [{"major_points": [event for event in events if event["event_type"] == "MajorPlotPoint"]}])
    database.configure_response(r"as act_events", [{"act_events": [event for event in events if event["event_type"] == "ActKeyEvent"]}])

    # The selected synthetic graph has no items, social edges or earlier chapters.
    assert catalog.candidates("Item") == []
    assert character["relationships"] == {}
    for pattern in (r"\[r:POSSESSES\]", r"\[:FEATURES_ITEM\]", r"MATCH \(c1:Character\)-\[r\]->\(c2:Character\)", r"RETURN s.name AS subject", r"\[r:`LOCATED_AT`\]"):
        database.configure_response(pattern, [])
    database.configure_response(r"AS context_chapters", [{"context_chapters": []}])

    async def read_graph(query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if 'e:Event {event_type: "SceneEvent"}' in query:
            assert parameters is not None and parameters["chapter_number"] == 1
            database.executed_queries.append((query, parameters))
            return [event for event in events if event["event_type"] == "SceneEvent" and event["scene_index"] == parameters["scene_index"]]
        return await database.execute_read_query(query, parameters)

    monkeypatch.setattr(owner, "execute_read_query", read_graph)
    for field in ("_project_id", "_database", "_uri"):
        monkeypatch.setattr(owner, field, None)
    owner.bind_project(state["graph_project_id"])
    driver = OwnershipDriver()
    driver.transaction.owner = state["graph_project_id"]
    monkeypatch.setattr(owner, "driver", driver)
    clear_all_data_access_caches()
    clear_context_cache()
    try:
        with inject_services(RunServices(service, owner)):
            async with create_checkpointer(checkpoint) as saver:
                result = await workflow.compile(checkpointer=saver).ainvoke(state, run)
            async with create_checkpointer(checkpoint) as saver:
                reopened = await workflow.compile(checkpointer=saver).aget_state(run)
        assert reopened.values == result
        assert reopened.next == ()
        manager = ContentManager(str(tmp_path))
        if draft_response == "prose":
            assert result["has_fatal_error"] is False, result.get("last_error")
            assert result["current_scene_index"] == 2
            assert get_scene_drafts(result, manager) == prose
            assert manager.load_text(result["draft_ref"]) == "\n\n# ***\n\n".join(prose)
            assert draft_calls == 2
            assert prose[0] in bodies[2]["messages"][-1]["content"]
            assert "Previous Scenes in This Chapter:" in bodies[2]["messages"][-1]["content"]
            assert character["personality_description"] in bodies[1]["messages"][-1]["content"]
        else:
            assert result["has_fatal_error"] is True
            assert result["error_node"] == "draft_scene"
            assert result["draft_ref"] is None
            assert result["scene_drafts_ref"] is None
            assert result["current_scene_index"] == 0
            assert draft_calls == 1
        assert all(body["temperature"] == 1.0 and body["max_tokens"] == 65536 for body in bodies)
        assert "Unconfigured synthetic query" not in caplog.text
        assert not any("non-fatal error" in record.getMessage() for record in caplog.records)
    finally:
        clear_all_data_access_caches()
        clear_context_cache()
        await service.aclose()
