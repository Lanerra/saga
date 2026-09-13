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
from core.langgraph.nodes.assemble_chapter_node import assemble_chapter
from core.langgraph.state import NarrativeState
from core.langgraph.subgraphs.generation import create_generation_subgraph
from core.langgraph.workflow import create_checkpointer
from core.llm_interface_refactored import create_llm_service
from core.service_context import RunServices, get_services, inject_services
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager
from tests.test_generation_failure_contract import seeded_state
from tests.test_r03_context_contracts import SCENE


@pytest.mark.parametrize("draft_response", ["prose", "reasoning", "length"])
async def test_generation_handoff_and_sqlite_reopen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, draft_response: str) -> None:
    database = FakeNeo4jManager()
    database.configure_response(r"RETURN c.name AS name", [{"name": "Hero"}])
    owner = get_services().database
    monkeypatch.setattr(owner, "execute_read_query", database.execute_read_query)
    monkeypatch.setattr(owner, "execute_write_query", database.execute_write_query)
    monkeypatch.setattr(owner, "execute_cypher_batch", database.execute_cypher_batch)
    monkeypatch.setattr(owner, "driver", None)
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
            assert result["has_fatal_error"] is False
            assert result["current_scene_index"] == 2
            assert get_scene_drafts(result, manager) == prose
            assert manager.load_text(result["draft_ref"]) == "\n\n# ***\n\n".join(prose)
            assert draft_calls == 2
            assert prose[0] in bodies[2]["messages"][-1]["content"]
            assert "Previous Scenes in This Chapter:" in bodies[2]["messages"][-1]["content"]
        else:
            assert result["has_fatal_error"] is True
            assert result["error_node"] == "draft_scene"
            assert result["draft_ref"] is None
            assert result["scene_drafts_ref"] is None
            assert result["current_scene_index"] == 0
            assert draft_calls == 1
        assert all(body["temperature"] == 1.0 and body["max_tokens"] == 65536 for body in bodies)
    finally:
        await service.aclose()
