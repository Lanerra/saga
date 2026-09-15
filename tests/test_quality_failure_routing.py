"""Persist quality rejections through native workflow, recovery and CLI boundaries."""
import asyncio
import json
import runpy
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from core.db_manager import neo4j_manager
from core.exceptions import WorkflowExecutionError
from core.langgraph.chapter_lifecycle import extraction_binding
from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.commit_node import commit_to_graph
from core.langgraph.quality_policy import SCORE_FIELDS, validation_decision
from core.langgraph.state import NarrativeState, create_initial_state
from core.langgraph.workflow import create_checkpointer, create_full_workflow_graph, handle_fatal_error
from core.project_config import NarrativeProjectConfig
from core.project_manager import ProjectManager
from core.service_context import get_services
from data_access import chapter_queries
from data_access.chapter_queries import ChapterProgress
from orchestration.langgraph_orchestrator import LangGraphOrchestrator
from tests.test_langgraph.test_chapter_lifecycle import DriverExample, example_state, lifecycle_example
from tests.test_quality_acceptance import evaluate_example

__all__ = ["lifecycle_example"]


@pytest.mark.parametrize("mandatory", [False, True])
async def test_trait_query_failure_is_not_completed(lifecycle_example: tuple[NarrativeState, DriverExample], monkeypatch: pytest.MonkeyPatch, mandatory: bool) -> None:
    from neo4j.exceptions import CypherTypeError

    from core.langgraph.chapter_lifecycle import ChapterLifecycle
    from tests.test_langgraph.test_chapter_lifecycle import TransactionExample

    state, driver = lifecycle_example
    state["quality_policy"].update(graph_quality="mandatory" if mandatory else "advisory", graph_quality_frequency=1)
    state.update(await commit_to_graph(state))
    original_run = TransactionExample.run

    def run(transaction: TransactionExample, query: str, parameters: Any = None) -> Any:
        if "ANY(t IN c.traits" in query:
            raise CypherTypeError("Synthetic invalid trait value")
        return original_run(transaction, query, parameters)

    monkeypatch.setattr(TransactionExample, "run", run)
    lifecycle = ChapterLifecycle(state).stage()
    if mandatory:
        with pytest.raises(ValueError, match="Mandatory graph quality gate failed: Synthetic invalid trait value"):
            await lifecycle.publish()
        assert driver.commits == 1
        assert not lifecycle.files.exists("chapters/chapter_001.accepted.json")
    else:
        await lifecycle.publish()
        decision = json.loads(lifecycle.files.read_bytes(lifecycle.phase_path("acceptance")))["quality"]
        assert decision["graph_quality_check"]["status"] == "failed"
        assert decision["graph_quality_check"]["reason"] == "Synthetic invalid trait value"
        assert decision["exceptions"] == ["advisory_graph_quality:Synthetic invalid trait value"]
        assert decision["status"] == "accepted_with_exceptions"


async def failing_evaluation(state: NarrativeState, monkeypatch: pytest.MonkeyPatch, case: str) -> NarrativeState:
    # A retained workflow checkpoint includes authoring inputs, unlike the narrow lifecycle fixture.
    state = {
        **create_initial_state(
            project_id=state["project_id"], project_dir=state["project_dir"], title="Synthetic",
            genre="Fantasy", theme="Discovery", setting="Room", protagonist_name="Hero",
            target_word_count=100, total_chapters=state["total_chapters"],
        ),
        **state,
    }
    state["quality_policy"]["identity"] = "author" if case == "author_incomplete" else "strict"
    state.update({"force_continue": case == "strict_force", "max_iterations": 0})
    state.update(await commit_to_graph(state))
    response = "malformed" if case == "author_incomplete" else json.dumps({**dict.fromkeys(SCORE_FIELDS, 0.1), "feedback": "Synthetic failing quality"})
    return await evaluate_example(state, monkeypatch, response)


async def empty_progress() -> ChapterProgress:
    return ChapterProgress(0, ())


@pytest.mark.parametrize("case", ["strict_score", "strict_force", "author_incomplete"])
@pytest.mark.parametrize("boundary", ["commit", "validate"])
async def test_quality_failure_is_persisted(lifecycle_example: tuple[NarrativeState, DriverExample], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str, boundary: str) -> None:
    state, driver = lifecycle_example
    state = await failing_evaluation(state, monkeypatch, case)
    with pytest.raises(ValueError) as rejection:
        validation_decision(state)
    expected_reason = str(rejection.value)
    policy = deepcopy(state["quality_policy"])
    provider_calls = 0
    evaluate = get_services().language_model.async_call_llm

    async def counted_evaluation(**arguments: Any) -> Any:
        nonlocal provider_calls
        provider_calls += 1
        return await evaluate(**arguments)

    monkeypatch.setattr(get_services().language_model, "async_call_llm", counted_evaluation)
    monkeypatch.setattr(chapter_queries, "load_chapter_progress_from_db", empty_progress)
    if boundary == "commit":
        state.update({"quality_checks": {}, "contradictions": []})
    configuration = {"configurable": {"thread_id": "saga_synthetic"}}
    checkpoint = str(tmp_path / "quality.sqlite")
    async with create_checkpointer(checkpoint) as saver:
        graph = create_full_workflow_graph(saver)
        await graph.aupdate_state(configuration, state, as_node=boundary)
        assert (await graph.aget_state(configuration)).next == (("validate",) if boundary == "commit" else ("error_handler",))
    async with create_checkpointer(checkpoint) as saver:
        graph = create_full_workflow_graph(saver)
        orchestrator = LangGraphOrchestrator(project_dir=tmp_path)
        loaded = await orchestrator._load_state_for_run(graph=graph, requested_project_id="synthetic", thread_id="saga_synthetic", narrative_config=None)
        with pytest.raises(WorkflowExecutionError) as failure:
            await orchestrator._run_chapter_generation_loop(graph, loaded)
        final = await graph.aget_state(configuration)
        assert failure.value.details["last_error"] == expected_reason
        assert failure.value.details["error_node"] == "validate"
        assert final.values["has_fatal_error"] is True
        assert final.values["last_error"] == expected_reason
        assert final.values["error_node"] == "validate"
        assert final.values["quality_policy"] == policy
        assert final.next == ()
    retained = {str(path.relative_to(tmp_path)): path.read_bytes() for path in (tmp_path / ".saga").rglob("*") if path.is_file()}
    async with create_checkpointer(checkpoint) as saver:
        graph = create_full_workflow_graph(saver)
        orchestrator = LangGraphOrchestrator(project_dir=tmp_path)
        loaded = await orchestrator._load_state_for_run(graph=graph, requested_project_id="synthetic", thread_id="saga_synthetic", narrative_config=None)
        with pytest.raises(ValueError, match="outside the chapter lifecycle recovery boundary"):
            await orchestrator._run_chapter_generation_loop(graph, loaded)
        assert (await graph.aget_state(configuration)).values == final.values
    assert retained == {str(path.relative_to(tmp_path)): path.read_bytes() for path in (tmp_path / ".saga").rglob("*") if path.is_file()}
    assert provider_calls == (1 if boundary == "commit" else 0)
    assert driver.commits == 1
    assert not (tmp_path / "chapters/chapter_001.accepted.json").exists()
    assert not (tmp_path / "chapters/chapter_001.md").exists()


@pytest.mark.parametrize("policy,force,maximum,expected", [("strict", False, 3, "revise"), ("author", False, 3, "revise"), ("author", True, 3, "summarize"), ("author", False, 0, "summarize")])
async def test_revision_and_author_routes_remain_available(lifecycle_example: tuple[NarrativeState, DriverExample], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, policy: str, force: bool, maximum: int, expected: str) -> None:
    state, driver = lifecycle_example
    state = await failing_evaluation(state, monkeypatch, "strict_score")
    state["quality_policy"]["identity"] = policy
    state.update({"force_continue": force, "max_iterations": maximum, "quality_checks": {}, "contradictions": []})
    configuration = {"configurable": {"thread_id": "routes"}}
    async with create_checkpointer(str(tmp_path / "routes.sqlite")) as saver:
        graph = create_full_workflow_graph(saver)
        await graph.aupdate_state(configuration, state, as_node="commit")
        await graph.ainvoke(None, configuration, interrupt_before=["revise", "summarize", "error_handler"])
        snapshot = await graph.aget_state(configuration)
        assert snapshot.next == (expected,)
        assert snapshot.values["has_fatal_error"] is False
    assert driver.commits == 1


def test_quality_handler_preserves_existing_failure() -> None:
    state: NarrativeState = {"has_fatal_error": True, "last_error": "Original failure", "error_node": "commit", "quality_checks": {"invalid": {}}}
    before = deepcopy(state)
    assert handle_fatal_error(state) == {"current_node": "error_handler"}
    assert state == before
    state["revision_rollback_failure"] = {"chapter_number": 1, "iteration_count": 0, "error": "Rollback failed", "previous_error": "Original failure", "previous_error_node": "commit"}
    assert handle_fatal_error(state) == {"current_node": "error_handler", "has_fatal_error": True, "last_error": "Rollback failed", "error_node": "revise"}


@pytest.mark.parametrize("case", ["strict_score", "strict_force", "author_incomplete"])
def test_cli_reports_quality_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, case: str) -> None:
    monkeypatch.setattr(ProjectManager, "projects_root", tmp_path / "projects")
    project = NarrativeProjectConfig(title="Synthetic", genre="Fantasy", theme="Discovery", setting="Room", protagonist_name="Hero", narrative_style="Third person", total_chapters=1, created_from="settings", original_prompt="")
    directory = ProjectManager.save_config(project, review=False)
    state = example_state(directory)
    state["project_id"] = directory.name
    manager = ContentManager(str(directory))
    assert state["scene_drafts_ref"] is not None
    state["extraction_source"] = extraction_binding(state, manager.load_list_of_texts(state["scene_drafts_ref"]))
    monkeypatch.setattr(neo4j_manager, "__dict__", {**neo4j_manager.__dict__, "_project_id": None, "_database": None, "_uri": None})
    neo4j_manager.bind_project(state["graph_project_id"])
    driver = DriverExample(state["graph_project_id"])
    monkeypatch.setattr(neo4j_manager, "driver", driver)

    async def connect() -> None:
        return None

    monkeypatch.setattr(neo4j_manager, "connect", connect)
    monkeypatch.setattr(neo4j_manager, "create_db_schema", connect)
    monkeypatch.setattr(chapter_queries, "load_chapter_progress_from_db", empty_progress)
    configuration = {"configurable": {"thread_id": "saga_" + directory.name}}
    checkpoint = str(directory / "checkpoints/saga.db")

    async def prepare() -> str:
        evaluated = await failing_evaluation(state, monkeypatch, case)
        with pytest.raises(ValueError) as rejection:
            validation_decision(evaluated)
        async with create_checkpointer(checkpoint) as saver:
            await create_full_workflow_graph(saver).aupdate_state(configuration, evaluated, as_node="validate")
        return str(rejection.value)

    reason = asyncio.run(prepare())
    source = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(sys, "argv", [str(source / "main.py"), "generate", "--project-dir", str(directory)])
    for invocation in range(2):
        with pytest.raises(SystemExit) as failure:
            runpy.run_path(str(source / "main.py"), run_name="__main__")
        assert failure.value.code == 1, invocation

    async def retained_failure() -> None:
        async with create_checkpointer(checkpoint) as saver:
            snapshot = await create_full_workflow_graph(saver).aget_state(configuration)
            assert snapshot.values["has_fatal_error"] is True
            assert snapshot.values["last_error"] == reason
            assert snapshot.values["error_node"] == "validate"

    asyncio.run(retained_failure())
    assert driver.commits == 1
    assert not (directory / "chapters/chapter_001.accepted.json").exists()
    assert "SAGA: LangGraph Generation Complete" not in caplog.text
