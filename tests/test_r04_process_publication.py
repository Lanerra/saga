"""Run with -m tests.test_r04_process_publication under offline_run.py.

Three fresh interpreters share synthetic files and real SQLite checkpoints.
The driver-boundary fake persists graph receipts, not a substitute SQLite graph.
Every child installs the Python offline boundary; the outer kernel/filesystem
sandbox remains inherited by all children. No provider transport is permitted.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


def file_identities(directory: Path) -> dict[str, Any]:
    return {
        str(path.relative_to(directory)): [hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_ino, path.stat().st_mtime_ns]
        for base in ("chapters", "exports") for path in sorted((directory / base).rglob("*")) if path.is_file()
    }


async def phase(name: str, directory: Path) -> None:
    import httpx
    from langgraph.graph import END, StateGraph  # type: ignore[attr-defined]

    import config
    from core.db_manager import Neo4jManagerSingleton
    from core.http_client_service import HTTPClientService
    from core.langgraph.chapter_lifecycle import extraction_binding
    from core.langgraph.content_manager import ContentManager
    from core.langgraph.export import generate_full_export
    from core.langgraph.nodes.commit_node import commit_to_graph
    from core.langgraph.nodes.finalize_node import finalize_chapter
    from core.langgraph.nodes.quality_assurance_node import check_quality
    from core.langgraph.state import NarrativeState, create_initial_state
    from core.langgraph.workflow import advance_chapter, create_checkpointer, create_full_workflow_graph
    from core.llm_interface_refactored import create_llm_service
    from core.service_context import RunServices, inject_services
    from orchestration.langgraph_orchestrator import LangGraphOrchestrator
    from tests.test_langgraph.test_chapter_lifecycle import DriverExample, Rows, TransactionExample, example_state

    class ProcessTransaction(TransactionExample):
        def run(self, query: str, parameters: Any = None, **keywords: Any) -> Rows:
            if " ".join(query.split()) == "MATCH (c:Chapter) RETURN c.number AS chapter_number, c.generation_status AS generation_status, c.is_provisional AS is_provisional":
                return Rows([{"chapter_number": node["properties"]["number"], "generation_status": node["properties"].get("generation_status"), "is_provisional": node["properties"].get("is_provisional")} for node in self.nodes.values() if "Chapter" in node["labels"]])
            return super().run(query, parameters, **keywords)

    class ProcessDriver(DriverExample):
        def begin_transaction(self) -> ProcessTransaction:
            return ProcessTransaction(self)

        def execute_read(self, callback: Any, *arguments: Any) -> Any:
            return callback(ProcessTransaction(self), *arguments)

    provider_calls: list[str] = []

    def provider_request(request: httpx.Request) -> httpx.Response:
        provider_calls.append(request.method)
        raise AssertionError(f"Unexpected provider call: {request.method}")

    effective = config.snapshot_settings()
    configuration = type(effective).model_validate({**{name: getattr(effective, name) for name in type(effective).model_fields}, "ENABLE_QA_CHECKS": False})
    with config.bind_settings(configuration):
        language_model = create_llm_service(HTTPClientService(client=httpx.AsyncClient(transport=httpx.MockTransport(provider_request))))
        database = Neo4jManagerSingleton()
        services = RunServices(language_model, database)
    try:
        with inject_services(services):
            state: NarrativeState
            if name == "seed":
                state = {
                    **create_initial_state(project_id="synthetic", project_dir=str(directory), title="Synthetic", genre="Mystery", theme="Trust", setting="Archive", protagonist_name="Traveler", target_word_count=2000, total_chapters=1),
                    **example_state(directory),
                }
                assert state["scene_drafts_ref"] is not None
                scenes = ContentManager(str(directory)).load_list_of_texts(state["scene_drafts_ref"])
                state["extraction_source"] = extraction_binding(state, scenes)
                driver = ProcessDriver(state["graph_project_id"])
            else:
                persisted = json.loads((directory / "synthetic-graph.json").read_text())
                driver = ProcessDriver(persisted["project_id"])
                driver.receipts = persisted["receipts"]
                driver.nodes = persisted["nodes"]
                driver.edges = persisted["edges"]
                assert driver.nodes and driver.edges
                assert driver.receipts and all(row["compensation"] and row["compensation_sha256"] for row in driver.receipts.values())
                driver.writes = persisted["writes"]
                driver.commits = persisted["commits"]
            database.bind_project(driver.project_id)
            database.driver = driver  # type: ignore[assignment]
            workflow = StateGraph(NarrativeState)
            workflow.add_node("summarize", lambda value: value)
            workflow.add_node("finalize", finalize_chapter)
            workflow.add_node("check_quality", check_quality)
            workflow.add_node("advance_chapter", advance_chapter)
            workflow.set_entry_point("summarize")
            workflow.add_edge("summarize", "finalize")
            workflow.add_edge("finalize", "check_quality")
            workflow.add_edge("check_quality", END)
            workflow.add_edge("advance_chapter", END)
            graph_configuration = {"configurable": {"thread_id": "saga_synthetic"}}
            async with create_checkpointer(str(directory / "checkpoints/saga.db")) as saver:
                graph = create_full_workflow_graph(saver) if name.startswith("reopen") else workflow.compile(checkpointer=saver)
                if name == "seed":
                    state = {**state, **await commit_to_graph(state), "current_node": "summarize"}
                    assert not state.get("has_fatal_error"), state.get("last_error")
                    saved = await graph.aupdate_state(graph_configuration, state, as_node="summarize")
                    snapshot = await graph.aget_state(saved)
                    task, = snapshot.tasks
                    publication = await finalize_chapter(state)
                    assert not publication.get("has_fatal_error"), publication.get("last_error")
                    await saver.aput_writes(saved, [*publication.items(), ("branch:to:check_quality", None)], task.id)
                    assert (await graph.aget_state(saved)).next == ("finalize",)
                    assert (await graph.aget_state(graph_configuration)).values["current_node"] == "finalize"
                    (directory / "synthetic-graph.json").write_text(json.dumps({
                        "project_id": driver.project_id, "receipts": driver.receipts, "nodes": driver.nodes, "edges": driver.edges,
                        "writes": driver.writes, "commits": driver.commits,
                    }))
                    (directory / "seed-files.json").write_text(json.dumps(file_identities(directory)))
                else:
                    before_writes = list(driver.writes)
                    before_graph = deepcopy((driver.nodes, driver.edges, driver.receipts))
                    before_snapshot = await graph.aget_state(graph_configuration)
                    orchestrator = LangGraphOrchestrator(project_dir=directory)
                    state = await orchestrator._load_state_for_run(graph=graph, requested_project_id="synthetic", thread_id="saga_synthetic", narrative_config=None)
                    await orchestrator._run_chapter_generation_loop(graph, state)
                    snapshot = await graph.aget_state(graph_configuration)
                    assert snapshot.next == snapshot.tasks == ()
                    assert snapshot.values["current_node"] == "check_quality"
                    assert driver.writes == before_writes
                    assert (driver.nodes, driver.edges, driver.receipts) == before_graph
                    assert driver.commits == 2
                    output = generate_full_export(directory, expected_chapters=1)
                    assert output.read_bytes() == b"A synthetic traveler returns.\\n\nUnicode: \xe9\x9b\xa8\n"
                    if name == "resume":
                        seed_files = json.loads((directory / "seed-files.json").read_text())
                        after_files = file_identities(directory)
                        assert all(after_files[path] == identity for path, identity in seed_files.items())
                        (directory / "completed-files.json").write_text(json.dumps(after_files))
                    else:
                        assert snapshot.config == before_snapshot.config
                        assert file_identities(directory) == json.loads((directory / "completed-files.json").read_text())
            assert provider_calls == []
            (directory / f"{name}-result.json").write_text(json.dumps({"phase": name, "status": "passed", "pid": os.getpid(), "graph_commits": driver.commits, "provider_calls": len(provider_calls), "full_workflow": name.startswith("reopen"), "source": str(Path(finalize_chapter.__code__.co_filename).resolve())}))
    finally:
        await language_model.aclose()


def main() -> None:
    if len(sys.argv) == 3:
        from tests.offline import OfflineBoundary

        boundary = OfflineBoundary()
        boundary.install(change_directory=False)
        try:
            asyncio.run(phase(sys.argv[1], Path(sys.argv[2])))
        finally:
            boundary.close()
        return
    directory = Path(os.environ["TMPDIR"]) / "process-publication"
    directory.mkdir()
    for name in ("seed", "resume", "reopen", "reopen_again"):
        result = subprocess.run([sys.executable, "-B", "-m", "tests.test_r04_process_publication", name, str(directory)], capture_output=True, text=True, timeout=90)
        (directory / f"{name}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, f"{name}: {result.stdout}\n{result.stderr}"
    results = [json.loads((directory / f"{name}-result.json").read_text()) for name in ("seed", "resume", "reopen", "reopen_again")]
    assert len({result["pid"] for result in results}) == 4
    print(json.dumps({"status": "passed", "results": results, "directory": str(directory)}))


if __name__ == "__main__":
    main()
