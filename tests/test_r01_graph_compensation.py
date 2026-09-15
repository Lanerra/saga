"""Exact graph inverse contracts; Cypher execution is separately engine-qualified."""
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from core.graph_migration import EDGE_SNAPSHOT_QUERY, NODE_SNAPSHOT_QUERY
from core.langgraph.chapter_lifecycle import ChapterLifecycle, extraction_binding
from core.langgraph.nodes.revision_node import _rollback_chapter_data
from core.service_context import RunServices, inject_services
from tests.test_langgraph.test_chapter_lifecycle import example_state


class Rows(list[dict[str, Any]]):
    def consume(self) -> None:
        pass


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {
            "scene": {"element_id": "scene", "labels": ["Scene"], "properties": {"id": "init-scene", "chapter_number": 1}},
            "chapter": {"element_id": "chapter", "labels": ["Chapter"], "properties": {"id": "init-chapter", "number": 1, "generation_status": "planned"}},
            "character": {"element_id": "character", "labels": ["Character"], "properties": {"id": "init-character", "is_provisional": False}},
        }
        self.edges: dict[str, dict[str, Any]] = {"init": {"element_id": "init", "source": "scene", "target": "chapter", "type": "PART_OF", "properties": {"chapter_added": 0}}}
        self.writes: list[str] = []

    def run(self, query: str, parameters: Any = None, **keywords: Any) -> Rows:
        if query == NODE_SNAPSHOT_QUERY:
            return Rows(deepcopy(list(self.nodes.values())))
        if query == EDGE_SNAPSHOT_QUERY:
            return Rows(deepcopy(list(self.edges.values())))
        from data_access.cypher_builders import graph_compensation as compensation

        self.writes.append(query)
        if query == compensation.DELETE_EDGE:
            del self.edges[parameters["element_id"]]
        elif query == compensation.RESTORE_NODE:
            self.nodes[parameters["element_id"]]["properties"] = deepcopy(parameters["properties"])
        elif query == compensation.RESTORE_EDGE:
            self.edges[parameters["element_id"]]["properties"] = deepcopy(parameters["properties"])
        elif query == compensation.DELETE_NODE:
            identity = parameters["element_id"]
            assert all(identity not in (edge["source"], edge["target"]) for edge in self.edges.values())
            del self.nodes[identity]
        elif query == compensation.CREATE_EDGE:
            self.edges["restored"] = {"element_id": "restored", **deepcopy(parameters)}
        else:
            raise AssertionError(query)
        return Rows()


def test_compensation_restores_plan_and_properties_without_touching_unrelated_data() -> None:
    from data_access.cypher_builders.graph_compensation import apply_compensation, capture_graph, encode_compensation

    graph = Graph()
    before = capture_graph(graph)
    graph.nodes["chapter"]["properties"].update(generation_status="committed", attempt_id="attempt")
    graph.nodes["character"]["properties"]["updated_ts"] = 10
    graph.nodes["new"] = {"element_id": "new", "labels": ["Item"], "properties": {"id": "new-item"}}
    graph.edges["new-edge"] = {"element_id": "new-edge", "source": "scene", "target": "new", "type": "FEATURES_ITEM", "properties": {"chapter_added": 1}}
    journal = encode_compensation(before, capture_graph(graph))
    graph.nodes["unrelated"] = {"element_id": "unrelated", "labels": ["Item"], "properties": {"id": "unrelated"}}
    apply_compensation(graph, journal)
    assert graph.nodes == {**{row["element_id"]: row for row in before["nodes"]}, "unrelated": graph.nodes["unrelated"]}
    assert list(graph.edges.values()) == before["edges"]
    assert not any("DETACH" in query for query in graph.writes)


@pytest.mark.parametrize("drift", ["properties", "edge", "missing", "labels"])
def test_compensation_conflict_fails_before_writes(drift: str) -> None:
    from data_access.cypher_builders.graph_compensation import apply_compensation, capture_graph, encode_compensation

    graph = Graph()
    before = capture_graph(graph)
    graph.nodes["new"] = {"element_id": "new", "labels": ["Item"], "properties": {"id": "new"}}
    journal = encode_compensation(before, capture_graph(graph))
    if drift == "properties":
        graph.nodes["new"]["properties"]["description"] = "Later legitimate work"
    elif drift == "edge":
        graph.edges["foreign"] = {"element_id": "foreign", "source": "scene", "target": "new", "type": "FEATURES_ITEM", "properties": {}}
    elif drift == "labels":
        graph.nodes["new"]["labels"].append("Protected")
    else:
        del graph.nodes["new"]
    with pytest.raises(ValueError, match="Compensation conflict"):
        apply_compensation(graph, journal)
    assert graph.writes == []


def test_deleted_assertion_restores_occurrence_and_existing_modified_edge() -> None:
    from data_access.cypher_builders.graph_compensation import apply_compensation, capture_graph, encode_compensation

    graph = Graph()
    graph.edges["old"] = {"element_id": "old", "source": "character", "target": "chapter", "type": "KNOWS", "properties": {"id": "prior-assertion", "description": "Prior evidence"}}
    before = capture_graph(graph)
    del graph.edges["old"]
    graph.edges["init"]["properties"]["changed"] = True
    journal = encode_compensation(before, capture_graph(graph))
    apply_compensation(graph, journal)
    expected = [{key: value for key, value in edge.items() if key != "element_id"} for edge in before["edges"]]
    actual = [{key: value for key, value in edge.items() if key != "element_id"} for edge in graph.edges.values()]
    assert actual == expected


def test_journal_refuses_unrestorable_node_deletion() -> None:
    from data_access.cypher_builders.graph_compensation import capture_graph, encode_compensation

    graph = Graph()
    before = capture_graph(graph)
    del graph.nodes["character"]
    with pytest.raises(ValueError, match="node deletion or label change"):
        encode_compensation(before, capture_graph(graph))


async def test_unjournaled_rollback_requires_reconciliation_without_graph_writes() -> None:
    with pytest.raises(ValueError, match="durable.*journal"):
        await _rollback_chapter_data(1)


def test_compensation_requires_graph_readback() -> None:
    from data_access.cypher_builders.graph_compensation import apply_compensation, capture_graph, encode_compensation

    class LostWriteGraph(Graph):
        def run(self, query: str, parameters: Any = None, **keywords: Any) -> Rows:
            if query in (NODE_SNAPSHOT_QUERY, EDGE_SNAPSHOT_QUERY):
                return super().run(query, parameters)
            return Rows()

    graph = LostWriteGraph()
    before = capture_graph(graph)
    graph.nodes["chapter"]["properties"]["generation_status"] = "committed"
    journal = encode_compensation(before, capture_graph(graph))
    with pytest.raises(ValueError, match="Compensation readback"):
        apply_compensation(graph, journal)


async def test_compensation_intent_is_durable_even_when_graph_read_fails(tmp_path: Path) -> None:
    state = example_state(tmp_path)
    state["extraction_source"] = extraction_binding(state, ["A synthetic traveler returns.\\n\nUnicode: 雨\n"])
    lifecycle = ChapterLifecycle(state).stage()

    class UnavailableDatabase:
        def require_project_binding(self) -> str:
            return state["graph_project_id"]

        async def execute_read_query(self, *arguments: Any) -> Any:
            raise RuntimeError("synthetic graph unavailable")

    with inject_services(RunServices(cast(Any, object()), cast(Any, UnavailableDatabase()))):
        with pytest.raises(RuntimeError, match="synthetic graph unavailable"):
            await _rollback_chapter_data(1, lifecycle=lifecycle)
    reopened = ChapterLifecycle(state).stage()
    assert reopened.files.exists(reopened.phase_path("compensation_required"))
