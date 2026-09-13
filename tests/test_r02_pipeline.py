"""Adapter-to-query synthetic handoff; no Neo4j engine or narrative-quality claim."""
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import config
from core.http_client_service import HTTPClientService
from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.commit_node import commit_to_graph
from core.langgraph.nodes.scene_extraction import extract_from_scene
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import create_llm_service
from core.service_context import get_services
from tests.test_r02_extraction_identity import RETAINED, entity, relationship


class BatchRecorder:
    """Record the transaction API boundary without pretending to execute Cypher."""
    def __init__(self) -> None:
        self.batches: list[list[tuple[str, dict[str, Any]]]] = []

    async def execute_read_query(self, query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        assert "RETURN DISTINCT toLower(n.name) AS name" in query
        return []

    async def execute_cypher_batch(self, statements: list[tuple[str, dict[str, Any]]]) -> None:
        self.batches.append(statements)


async def test_commit_uses_same_graph_identity_inputs_for_entity_and_relationship(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    recorder = BatchRecorder()
    monkeypatch.setattr(get_services(), "database", recorder)
    manager = ContentManager(str(tmp_path))
    state: NarrativeState = {
        "project_dir": str(tmp_path), "current_chapter": 1,
        "extracted_entities_ref": manager.save_json({"characters": [entity("Mara", "Character").model_dump()],
            "world_items": [entity(name, "Location").model_dump() for name in ["King's Cross", "Kings Cross"]]}, "extracted_entities", "chapter_1", 1),
        "extracted_relationships_ref": manager.save_json([relationship("Mara", name).model_dump() for name in ["King's Cross", "Kings Cross"]], "extracted_relationships", "chapter_1", 1),
        "draft_ref": manager.save_text("Mara visits King's Cross and Kings Cross.", "draft", "chapter_1", 1),
    }
    result = await commit_to_graph(state)
    assert result["last_error"] is None
    assert len(recorder.batches) == 1
    world_parameters = [parameters for _, parameters in recorder.batches[0] if parameters.get("category") == "location"]
    assert {parameters["name"] for parameters in world_parameters} == {"King's Cross", "Kings Cross"}
    # Absent IDs must use the same canonical graph resolver as characters, not
    # a second Python punctuation-stripping identity scheme.
    assert all(parameters["id"] is None for parameters in world_parameters)
    relationships = [parameters for _, parameters in recorder.batches[0] if "object_name" in parameters]
    assert {parameters["object_name"] for parameters in relationships} == {"King's Cross", "Kings Cross"}
    assert all(parameters["object_id"] is None for parameters in relationships)


async def test_original_scene_punctuation_reaches_all_producers(monkeypatch: pytest.MonkeyPatch) -> None:
    scene = "Father O'Brien visits The Hague during The King's Return. King's Cross is nearby."
    replies = iter([
        {"character_updates": {"Father O'Brien": {"description": "A visitor", "traits": [], "status": "Active", "relationships": {}}}},
        {"world_updates": {"Location": {name: {"description": "Named place", "category": "Settlement", "goals": [], "rules": [], "key_elements": []} for name in ["The Hague", "King's Cross"]}}},
        {"world_updates": {"Event": {"The King's Return": {"description": "Named event", "category": "Ceremony", "goals": [], "rules": [], "key_elements": []}}}},
        {"kg_triples": [{"subject": "Father O'Brien", "predicate": "LOCATED_AT", "object_entity": "The Hague", "description": "The visitor arrives."}]},
    ])
    requests: list[dict[str, Any]] = []
    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(next(replies))}, "finish_reason": "stop"}]})
    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    monkeypatch.setattr(get_services(), "language_model", service)
    try:
        result = await extract_from_scene(scene, 0, 1, "Synthetic", "Fantasy", "Father O'Brien", "synthetic")
        assert result["extraction_status"] == "complete"
        assert result["characters"][0]["name"] == "Father O'Brien"
        assert [item["name"] for item in result["world_items"]] == ["The Hague", "King's Cross", "The King's Return"]
        assert result["relationships"][0]["source_name"] == "Father O'Brien"
        assert len(requests) == 4
        assert all(scene in request["messages"][-1]["content"] for request in requests)
    finally:
        await service.aclose()


async def test_retained_scene_failure_publishes_no_partial_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter([item["response"] for item in RETAINED[:4]])
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": next(replies)}, "finish_reason": "stop"}]})
    service = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    monkeypatch.setattr(get_services(), "language_model", service)
    try:
        result = await extract_from_scene(RETAINED[0]["scene"], 0, 1, "Synthetic", "Literary Fiction", "Mara", "synthetic")
        assert result["extraction_status"] == "failed"
        assert set(result) == {"extraction_status", "extraction_outcomes"}
        assert all(outcome["status"] == "failed" for outcome in result["extraction_outcomes"])
        assert all(outcome["error_type"] in {"ValueError", "ValidationError"} for outcome in result["extraction_outcomes"])
    finally:
        await service.aclose()
