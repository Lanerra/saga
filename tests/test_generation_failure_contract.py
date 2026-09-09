"""Exercise failure contracts with real nodes, graphs, content files and SQLite."""
import asyncio
import inspect
import json
import os
import runpy
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
import spacy
from langgraph.graph import END, StateGraph  # type: ignore[attr-defined]
from pydantic import ValidationError

import config
import core.exceptions as exceptions
from config.settings import SagaSettings
from core.db_manager import neo4j_manager
from core.graph_ownership import GraphOwnershipError
from core.http_client_service import CompletionHTTPClient, EmbeddingHTTPClient
from core.langgraph.content_manager import ContentManager, get_scene_drafts
from core.langgraph.nodes.scene_generation_node import draft_scene
from core.langgraph.nodes.scene_planning_node import plan_scenes
from core.langgraph.state import NarrativeState, RevisionRollbackFailure, create_initial_state
from core.langgraph.state_helpers import clear_error_state, clear_generation_artifacts
from core.langgraph.subgraphs._shared import _should_continue_or_error
from core.langgraph.subgraphs.generation import create_generation_subgraph, should_continue_scenes
from core.langgraph.workflow import create_checkpointer, create_full_workflow_graph
from core.project_config import NarrativeProjectConfig
from core.project_manager import ProjectManager
from core.service_context import managed_services
from core.spacy_service import get_spacy_service
from orchestration.langgraph_orchestrator import LangGraphOrchestrator
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager

SCENE = {
    "title": "Arrival", "pov_character": "Hero", "setting": "Room",
    "characters": ["Hero"], "plot_point": "Arrival", "conflict": "Locked door",
    "outcome": "Door opens", "beats": "Hero opens the door",
}
ROLLBACK: RevisionRollbackFailure = {
    "chapter_number": 1, "iteration_count": 1, "error": "rollback acknowledgement failed",
    "previous_error": "contradiction", "previous_error_node": "validate",
}


class ProviderBoundary:
    def __init__(self) -> None:
        self.plan_responses = [json.dumps([SCENE])]
        self.draft_response = "Hero opened the door."
        self.fail_draft = False
        self.cancel_draft = False
        self.plan_calls = 0
        self.draft_calls = 0
        self.embedding_calls = 0

    async def completion(self, client: CompletionHTTPClient, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int, **keywords: Any) -> dict[str, Any]:
        prompt = messages[-1]["content"]
        if prompt.startswith("You are breaking a chapter outline into"):
            self.plan_calls += 1
            response = self.plan_responses[min(self.plan_calls - 1, len(self.plan_responses) - 1)]
        else:
            self.draft_calls += 1
            if self.cancel_draft:
                raise asyncio.CancelledError("synthetic cancellation")
            if self.fail_draft:
                raise RuntimeError("synthetic provider exhausted")
            response = self.draft_response
        return {"choices": [{"message": {"content": response}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    async def embedding(self, client: EmbeddingHTTPClient, text: str, model: str) -> dict[str, Any]:
        self.embedding_calls += 1
        return {"embedding": [0.0] * config.EXPECTED_EMBEDDING_DIM}


@pytest.fixture
def boundaries(monkeypatch: pytest.MonkeyPatch) -> tuple[ProviderBoundary, FakeNeo4jManager]:
    # Synthetic CLI projects keep production binding checks without owning session state.
    monkeypatch.setattr(neo4j_manager, "__dict__", {**neo4j_manager.__dict__, "_project_id": None, "_database": None, "_uri": None})
    provider = ProviderBoundary()
    database = FakeNeo4jManager()
    database.configure_response(r"RETURN c.name AS name", [{"name": "Hero"}])
    monkeypatch.setattr(neo4j_manager, "execute_read_query", database.execute_read_query)
    monkeypatch.setattr(neo4j_manager, "execute_cypher_batch", database.execute_cypher_batch)
    monkeypatch.setattr(neo4j_manager, "execute_write_query", database.execute_write_query)
    monkeypatch.setattr(neo4j_manager, "connect", database.connect)
    monkeypatch.setattr(neo4j_manager, "create_db_schema", database.connect)
    monkeypatch.setattr(neo4j_manager, "driver", None)
    monkeypatch.setattr(CompletionHTTPClient, "get_completion", lambda client, *args, **kwargs: provider.completion(client, *args, **kwargs))
    monkeypatch.setattr(EmbeddingHTTPClient, "get_embedding", lambda client, *args, **kwargs: provider.embedding(client, *args, **kwargs))
    pipeline = spacy.blank("en")
    pipeline.add_pipe("sentencizer")
    monkeypatch.setattr(get_spacy_service(), "_nlp", pipeline)
    monkeypatch.setattr(config, "ENABLE_RICH_PROGRESS", False)
    return provider, database


def seeded_state(directory: Path) -> NarrativeState:
    state = create_initial_state(project_id=directory.name, title="Synthetic", genre="Fantasy", theme="Discovery", setting="Room", target_word_count=100, total_chapters=1, project_dir=str(directory), protagonist_name="Hero")
    manager = ContentManager(str(directory))
    state["initialization_complete"] = True
    state["chapter_outlines_ref"] = manager.save_json({"1": {"scene_description": "Arrival", "key_beats": ["Open door"], "version": 1}}, "chapter_outlines", "all", 1)
    state["chapter_plan_ref"] = manager.save_json([SCENE], "chapter_plan", "chapter_1", 1)
    state["chapter_plan_scene_count"] = 1
    return state


@pytest.mark.parametrize("response", ["not json", "[]"])
async def test_exhausted_plan_stops_before_stale_plan_use(tmp_path: Path, boundaries: tuple[ProviderBoundary, FakeNeo4jManager], response: str) -> None:
    provider, database = boundaries
    provider.plan_responses = [response]
    state = seeded_state(tmp_path)
    original = dict(state)
    result = await create_generation_subgraph().ainvoke(state, {"recursion_limit": 12})
    assert result["has_fatal_error"] is True
    assert result["error_node"] == "plan_scenes"
    assert result["chapter_plan_ref"] is None
    assert result["chapter_plan_scene_count"] == 0
    assert result["current_scene_index"] == 0
    assert (provider.plan_calls, provider.draft_calls, provider.embedding_calls) == (3, 0, 0)
    assert database.executed_queries == []
    assert state == original


async def test_missing_outline_clears_plan_and_stops(tmp_path: Path, boundaries: tuple[ProviderBoundary, FakeNeo4jManager]) -> None:
    provider, database = boundaries
    state = seeded_state(tmp_path)
    state["chapter_outlines_ref"] = None
    result = await plan_scenes(state)
    assert result["has_fatal_error"] is True
    assert result["chapter_plan_ref"] is None
    assert result["last_error"] == "No outline found for chapter 1"
    assert (provider.plan_calls, provider.draft_calls) == (0, 0)
    assert database.executed_queries == []


@pytest.mark.parametrize("attempts", [1, 3, 10])
@pytest.mark.parametrize("succeed", [False, True])
async def test_plan_attempt_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundaries: tuple[ProviderBoundary, FakeNeo4jManager], attempts: int, succeed: bool) -> None:
    provider, _ = boundaries
    settings = SagaSettings(SCENE_PLAN_MAX_ATTEMPTS=attempts)
    monkeypatch.setitem(vars(config), "settings", settings)
    provider.plan_responses = ["not json"] * (attempts - 1) + [json.dumps([SCENE]) if succeed else "not json"]
    result = await plan_scenes(seeded_state(tmp_path))
    assert provider.plan_calls == attempts
    assert result.get("has_fatal_error", False) is (not succeed)
    if succeed:
        assert result["chapter_plan_scene_count"] == 1
        reference = result["chapter_plan_ref"]
        assert reference is not None
        assert ContentManager(str(tmp_path)).load_json(reference) == [SCENE]


@pytest.mark.parametrize("attempts", [0, -1, 11, 1.5])
def test_invalid_plan_attempt_budget_is_rejected(attempts: Any) -> None:
    with pytest.raises(ValidationError):
        SagaSettings(SCENE_PLAN_MAX_ATTEMPTS=attempts)


async def test_draft_failure_has_no_graph_retry(tmp_path: Path, boundaries: tuple[ProviderBoundary, FakeNeo4jManager]) -> None:
    provider, _ = boundaries
    provider.fail_draft = True
    result = await create_generation_subgraph().ainvoke(seeded_state(tmp_path), {"recursion_limit": 12})
    assert result["has_fatal_error"] is True
    assert result["error_node"] == "draft_scene"
    assert result["current_scene_index"] == 0
    assert result["scene_drafts_ref"] is None
    assert (provider.plan_calls, provider.draft_calls, provider.embedding_calls) == (1, 1, 1)
    assert "LLM completion failed" in result["last_error"]
    assert "synthetic provider exhausted" not in result["last_error"]


async def test_empty_draft_is_terminal(tmp_path: Path, boundaries: tuple[ProviderBoundary, FakeNeo4jManager]) -> None:
    provider, _ = boundaries
    provider.draft_response = "   "
    result = await draft_scene(seeded_state(tmp_path))
    assert result["has_fatal_error"] is True
    assert isinstance(result["last_error"], str)
    assert result["last_error"].startswith("Error generating scene: LLM completion failed")
    assert provider.draft_calls == 1


async def test_generation_success_advances_each_scene(tmp_path: Path, boundaries: tuple[ProviderBoundary, FakeNeo4jManager]) -> None:
    provider, _ = boundaries
    provider.plan_responses = [json.dumps([SCENE, SCENE])]
    result = await create_generation_subgraph().ainvoke(seeded_state(tmp_path), {"recursion_limit": 12})
    assert result["has_fatal_error"] is False
    assert result["current_scene_index"] == 2
    assert get_scene_drafts(result, ContentManager(str(tmp_path))) == [provider.draft_response, provider.draft_response]
    assert (provider.plan_calls, provider.draft_calls) == (1, 2)


def test_revision_clearing_removes_plan_without_clearing_barrier() -> None:
    state = {"chapter_plan_ref": {"old": True}, "revision_rollback_failure": ROLLBACK, **clear_generation_artifacts(), **clear_error_state()}
    assert state["chapter_plan_ref"] is None
    assert state["revision_rollback_failure"] == ROLLBACK


async def test_scene_nodes_preserve_rollback_barrier(tmp_path: Path, boundaries: tuple[ProviderBoundary, FakeNeo4jManager]) -> None:
    provider, database = boundaries
    state = seeded_state(tmp_path)
    state["revision_rollback_failure"] = ROLLBACK
    state["force_continue"] = True
    for node in (plan_scenes, draft_scene):
        result = {**state, **await node(state)}
        assert result == state
    assert _should_continue_or_error(state) == "error"
    assert should_continue_scenes(state) == "error"
    assert (provider.plan_calls, provider.draft_calls) == (0, 0)
    assert database.executed_queries == []


@pytest.mark.parametrize("scenario", ["plan", "draft", "rollback", "cancel"])
def test_real_cli_propagates_compiled_graph_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundaries: tuple[ProviderBoundary, FakeNeo4jManager], scenario: str, caplog: pytest.LogCaptureFixture) -> None:
    provider, _ = boundaries
    monkeypatch.setattr(ProjectManager, "projects_root", tmp_path / "projects")
    project = NarrativeProjectConfig(title="Synthetic", genre="Fantasy", theme="Discovery", setting="Room", protagonist_name="Hero", narrative_style="Third person", total_chapters=1, created_from="settings", original_prompt="")
    directory = ProjectManager.save_config(project, review=False)
    state = seeded_state(directory)
    from core.graph_ownership import load_graph_project_id

    state.update({"lifecycle_version": 1, "graph_project_id": load_graph_project_id(directory)})
    if scenario == "plan":
        provider.plan_responses = ["not json"]
    elif scenario == "draft":
        provider.fail_draft = True
    elif scenario == "cancel":
        provider.cancel_draft = True
    else:
        state["revision_rollback_failure"] = ROLLBACK
        state["force_continue"] = True
    checkpoint_path = directory / "checkpoints/saga.db"
    configuration = {"configurable": {"thread_id": f"saga_{directory.name}"}}

    async def seed_checkpoint() -> None:
        async with create_checkpointer(str(checkpoint_path)) as checkpointer:
            graph = create_full_workflow_graph(checkpointer)
            await graph.aupdate_state(configuration, state, as_node="route")

    asyncio.run(seed_checkpoint())
    root = Path(__file__).resolve().parents[1]
    assert Path(inspect.getfile(plan_scenes)).resolve() == root / "core/langgraph/nodes/scene_planning_node.py"
    monkeypatch.setattr(sys, "argv", [str(root / "main.py"), "generate", "--project-dir", str(directory)])
    output, errors = StringIO(), StringIO()
    with pytest.raises(SystemExit) as caught, redirect_stdout(output), redirect_stderr(errors):
        runpy.run_path(str(root / "main.py"), run_name="__main__")
    assert caught.value.code == (130 if scenario == "cancel" else 1)
    assert [line for line in output.getvalue().splitlines() if line.startswith("Scope:")] == [
        "Scope: resume the selected project's durable workflow, or initialize only if fresh. No reset or deletion is requested."
    ]
    assert "SAGA generation invocation succeeded:" not in output.getvalue()
    if scenario == "cancel":
        assert errors.getvalue() == "SAGA generate cancelled; no completion claimed. Retained artifacts may include partial progress; resume the same project without resetting.\n"
    else:
        failure = caught.value.__context__
        assert isinstance(failure, exceptions.WorkflowExecutionError)
        assert errors.getvalue() == f"SAGA generate failed: {failure}. No completion claimed; retained artifacts may include partial progress.\n"

    async def read_checkpoint() -> dict[str, Any]:
        async with create_checkpointer(str(checkpoint_path)) as checkpointer:
            checkpoint = await checkpointer.aget(configuration)
            assert checkpoint is not None
            return checkpoint["channel_values"]

    terminal = asyncio.run(read_checkpoint())
    if scenario != "cancel":
        assert terminal["has_fatal_error"] is True
        assert terminal["current_node"] == "error_handler"
        assert terminal["error_node"] == {"plan": "plan_scenes", "draft": "draft_scene", "rollback": "revise"}[scenario]
    if scenario == "rollback":
        assert terminal["revision_rollback_failure"] == ROLLBACK
        assert (provider.plan_calls, provider.draft_calls) == (0, 0)
    if scenario == "plan":
        assert (provider.plan_calls, provider.draft_calls) == (3, 0)
        assert terminal["chapter_plan_ref"] is None
    if scenario in {"draft", "cancel"}:
        assert (provider.plan_calls, provider.draft_calls) == (1, 1)
    assert terminal["draft_ref"] is None
    assert terminal["extracted_entities_ref"] is None
    assert not (directory / "chapters").exists()
    assert "SAGA: LangGraph Generation Complete" not in caplog.text


@pytest.mark.parametrize("previously_bound", [False, True])
@pytest.mark.parametrize("scenarios", [("plan", "draft", "rollback", "cancel", "plan"), ("cancel", "rollback", "draft", "plan", "cancel")])
def test_cli_boundary_restores_prior_singleton(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, previously_bound: bool, scenarios: tuple[str, ...]) -> None:
    monkeypatch.setattr(neo4j_manager, "_project_id", None)
    monkeypatch.setattr(neo4j_manager, "_database", None)
    monkeypatch.setattr(neo4j_manager, "_uri", None)
    if previously_bound:
        neo4j_manager.bind_project(str(uuid4()))
    monkeypatch.setattr(neo4j_manager, "driver", object())
    monkeypatch.setattr(neo4j_manager, "_property_keys_cache", {"sentinel_property"})
    monkeypatch.setattr(neo4j_manager, "_property_keys_cache_ts", 123.0)
    monkeypatch.setattr(neo4j_manager, "_apoc_available_cache", True)
    monkeypatch.setattr(neo4j_manager, "_apoc_availability_warning_logged", True)
    monkeypatch.setattr(neo4j_manager, "_neo4j_server_info", {"sentinel": "server"})
    monkeypatch.setattr(neo4j_manager, "_neo4j_server_info_logged", True)
    original = neo4j_manager.__dict__
    original_values = original.copy()
    original_pipeline = get_spacy_service()._nlp
    for number, scenario in enumerate(scenarios):
        directory = tmp_path / str(number)
        directory.mkdir()
        with pytest.MonkeyPatch.context() as patches:
            synthetic_boundaries = inspect.unwrap(boundaries)(patches)
            test_real_cli_propagates_compiled_graph_failure(directory, patches, synthetic_boundaries, scenario, caplog)
            assert neo4j_manager.require_project_binding() != original_values["_project_id"]
            with pytest.raises(GraphOwnershipError, match="start a fresh process"):
                neo4j_manager.bind_project(str(uuid4()))
            assert synthetic_boundaries[1].batch_statements == []
        assert neo4j_manager.__dict__ is original
        assert neo4j_manager.__dict__ == original_values
        assert neo4j_manager._property_keys_cache is original_values["_property_keys_cache"]
        assert neo4j_manager._property_keys_cache == {"sentinel_property"}
        assert neo4j_manager._neo4j_server_info is original_values["_neo4j_server_info"]
        assert neo4j_manager._neo4j_server_info == {"sentinel": "server"}
        assert get_spacy_service()._nlp is original_pipeline


@pytest.mark.parametrize("fatal", [False, True])
async def test_terminal_node_name_cannot_override_fatal_state(tmp_path: Path, boundaries: tuple[ProviderBoundary, FakeNeo4jManager], fatal: bool) -> None:
    async def finish(state: NarrativeState) -> NarrativeState:
        return {"has_fatal_error": fatal, "last_error": "finalization failed" if fatal else None, "error_node": "finalize" if fatal else None, "current_node": "finalize"}

    graph = StateGraph(NarrativeState)
    graph.add_node("finalize", finish)
    graph.set_entry_point("finalize")
    graph.add_edge("finalize", END)
    orchestrator = LangGraphOrchestrator(project_dir=tmp_path)
    if fatal:
        with pytest.raises(exceptions.SAGACoreError) as caught:
            await orchestrator._run_chapter_generation_loop(graph.compile(), seeded_state(tmp_path))
        assert type(caught.value).__name__ == "WorkflowExecutionError"
        assert caught.value.details["error_node"] == "finalize"
        assert caught.value.details["last_error"] == "finalization failed"
    else:
        await orchestrator._run_chapter_generation_loop(graph.compile(), seeded_state(tmp_path))


@pytest.mark.parametrize("attempts", [1, 3])
@pytest.mark.parametrize("succeed", [False, True])
async def test_transport_retry_budget_does_not_restart_draft(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attempts: int, succeed: bool) -> None:
    calls = 0

    async def post(client: httpx.AsyncClient, url: str, **keywords: Any) -> httpx.Response:
        nonlocal calls
        calls += 1
        if not succeed or calls < attempts:
            raise httpx.ConnectError("synthetic transport exhausted", request=httpx.Request("POST", url))
        return httpx.Response(200, request=httpx.Request("POST", url), json={"choices": [{"message": {"content": "Hero opened the door."}}]})

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    monkeypatch.setattr(config, "LLM_RETRY_ATTEMPTS", attempts)
    monkeypatch.setattr(config, "LLM_RETRY_DELAY_SECONDS", 0.001)
    pipeline = spacy.blank("en")
    pipeline.add_pipe("sentencizer")
    monkeypatch.setattr(get_spacy_service(), "_nlp", pipeline)
    async with managed_services():
        result = await draft_scene(seeded_state(tmp_path))
    assert calls == attempts
    assert result.get("has_fatal_error", False) is (not succeed)
    if succeed:
        assert result["current_scene_index"] == 1
    else:
        assert result["error_node"] == "draft_scene"
        message = result["last_error"]
        assert isinstance(message, str)
        assert "ConnectError" in message
        assert "synthetic transport exhausted" not in message


async def test_draft_storage_failure_preserves_prior_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundaries: tuple[ProviderBoundary, FakeNeo4jManager]) -> None:
    provider, _ = boundaries
    state = seeded_state(tmp_path)
    manager = ContentManager(str(tmp_path))
    state["chapter_plan_ref"] = manager.save_json([SCENE, SCENE], "chapter_plan", "chapter_1", 2)
    state["scene_drafts_ref"] = manager.save_list_of_texts(["Earlier scene."], "scenes", "chapter_1", 1)
    state["current_scene_index"] = 1
    state["chapter_plan_scene_count"] = 2
    original = dict(state)

    def fail_scene_write(*arguments: Any, **keywords: Any) -> None:
        raise OSError("synthetic storage failure")

    monkeypatch.setattr(os, "link", fail_scene_write)
    result = {**state, **await draft_scene(state)}
    assert result["has_fatal_error"] is True
    assert result["error_node"] == "draft_scene"
    assert result["last_error"] == "Error generating scene: synthetic storage failure"
    assert result["current_scene_index"] == 1
    assert result["scene_drafts_ref"] == state["scene_drafts_ref"]
    assert provider.draft_calls == 1
    assert state == original


@pytest.mark.parametrize("index", [-1, 1])
async def test_invalid_scene_indices_stop_without_provider(tmp_path: Path, boundaries: tuple[ProviderBoundary, FakeNeo4jManager], index: int) -> None:
    provider, _ = boundaries
    state = seeded_state(tmp_path)
    state["current_scene_index"] = index
    result = await draft_scene(state)
    assert result == {"last_error": f"Invalid scene index {index} for chapter plan with 1 scenes", "has_fatal_error": True, "error_node": "draft_scene", "current_node": "draft_scene"}
    assert provider.draft_calls == 0


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("from_candidate", [False, True])
def test_cli_terminal_invocation_reports_actual_manuscripts_without_reinitializing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundaries: tuple[ProviderBoundary, FakeNeo4jManager], complete: bool, from_candidate: bool,
) -> None:
    from core.graph_ownership import load_graph_project_id
    from core.langgraph.manuscript import ManuscriptStore

    provider, database = boundaries
    monkeypatch.setattr(ProjectManager, "projects_root", tmp_path / "projects")
    project = NarrativeProjectConfig(title="Synthetic", genre="Fantasy", theme="Discovery", setting="Room", protagonist_name="Hero", narrative_style="Third person", total_chapters=1)
    directory = ProjectManager.save_config(project, review=from_candidate)
    unrelated = ProjectManager.save_config(project.model_copy(update={"title": "Unrelated"}), review=True)
    unrelated_bytes = (unrelated / "config.candidate.json").read_bytes()
    original_config = (directory / ("config.candidate.json" if from_candidate else "config.json")).read_bytes()
    state = seeded_state(directory)
    state.update({"current_chapter": 2 if complete else 1, "current_node": "check_quality", "lifecycle_version": 1, "graph_project_id": load_graph_project_id(directory)})
    if complete:
        store = ManuscriptStore(directory)
        store.accept(store.prepare(1, "Synthetic retained prose.  \r\n"))
        database.configure_response(r"RETURN c.number AS chapter_number", [{"chapter_number": 1, "generation_status": "finalized", "is_provisional": False}])
    configuration = {"configurable": {"thread_id": "saga_synthetic"}}

    async def seed_terminal() -> Any:
        async with create_checkpointer(str(directory / "checkpoints/saga.db")) as saver:
            graph = create_full_workflow_graph(saver)
            await graph.aupdate_state(configuration, state, as_node="check_quality")
            snapshot = await graph.aget_state(configuration)
            assert snapshot.next == ()
            return snapshot.config

    original_checkpoint = asyncio.run(seed_terminal())
    entrypoint = Path(__file__).resolve().parents[1] / "main.py"
    for invocation in range(2):
        arguments = [str(entrypoint), "generate", "--project-dir", str(directory)]
        if from_candidate and invocation == 0:
            arguments.append("--from-candidate")
        monkeypatch.setattr(sys, "argv", arguments)
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            runpy.run_path(str(entrypoint), run_name="__main__")
        assert errors.getvalue() == ""
        assert [line for line in output.getvalue().splitlines() if line.startswith("SAGA generation invocation")] == [
            f"SAGA generation invocation succeeded: {directory}; accepted manuscripts {int(complete)}/1. Export is a separate command."
        ]
    assert (provider.plan_calls, provider.draft_calls, provider.embedding_calls) == (0, 0, 0)
    assert database.batch_statements == []
    assert (directory / "config.json").read_bytes() == original_config
    assert not (directory / "config.candidate.json").exists()
    assert (unrelated / "config.candidate.json").read_bytes() == unrelated_bytes
    assert not (unrelated / "config.json").exists()

    async def read_terminal() -> Any:
        async with create_checkpointer(str(directory / "checkpoints/saga.db")) as saver:
            graph = create_full_workflow_graph(saver)
            snapshot = await graph.aget_state(configuration)
            assert snapshot.next == ()
            return snapshot.config

    assert asyncio.run(read_terminal()) == original_checkpoint


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
def test_cli_bootstrap_outcome_matches_saved_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundaries: tuple[ProviderBoundary, FakeNeo4jManager], outcome: str,
) -> None:
    provider, _ = boundaries
    monkeypatch.setattr(ProjectManager, "projects_root", tmp_path / "projects")
    provider.draft_response = json.dumps({"title": "Synthetic", "genre": "Mystery", "theme": "Discovery", "setting": "Archive", "protagonist_name": "Ada", "narrative_style": "First person", "total_chapters": 2, "target_word_count": 1000})
    if outcome == "failure":
        provider.draft_response = "{}"
    provider.cancel_draft = outcome == "cancel"
    entrypoint = Path(__file__).resolve().parents[1] / "main.py"
    monkeypatch.setattr(sys, "argv", [str(entrypoint), "bootstrap", "Synthetic premise"])
    output, errors = StringIO(), StringIO()
    with redirect_stdout(output), redirect_stderr(errors):
        if outcome == "success":
            runpy.run_path(str(entrypoint), run_name="__main__")
        else:
            with pytest.raises(SystemExit) as caught:
                runpy.run_path(str(entrypoint), run_name="__main__")
            assert caught.value.code == (130 if outcome == "cancel" else 1)
    candidate = tmp_path / "projects/synthetic/config.candidate.json"
    summaries = [line for line in output.getvalue().splitlines() if line.startswith("SAGA bootstrap succeeded:")]
    assert summaries == ([f"SAGA bootstrap succeeded: {candidate}"] if outcome == "success" else [])
    assert candidate.exists() is (outcome == "success")
    assert not candidate.with_name("config.json").exists()
    if outcome == "success":
        assert ProjectManager.load_candidate_config(candidate.parent).original_prompt == "Synthetic premise"
        assert errors.getvalue() == ""
    elif outcome == "cancel":
        assert errors.getvalue() == "SAGA bootstrap cancelled; no completion claimed. Retained artifacts may include partial progress; resume the same project without resetting.\n"
    else:
        failure = caught.value.__context__
        assert isinstance(failure, ValueError)
        assert errors.getvalue() == f"SAGA bootstrap failed: {failure}. No completion claimed; retained artifacts may include partial progress.\n"
