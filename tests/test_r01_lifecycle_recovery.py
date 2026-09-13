"""Durable journal lifecycle at transaction acknowledgement and SQLite boundaries."""
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from core.langgraph.chapter_lifecycle import ATTEMPT_QUERY, CHAPTER_QUERY, CREATE_ATTEMPT, STORE_COMPENSATION, UPDATE_ATTEMPT, ChapterLifecycle, extraction_binding
from core.langgraph.nodes.revision_node import _rollback_chapter_data
from core.langgraph.state import NarrativeState
from core.langgraph.state_helpers import clear_error_state
from core.langgraph.workflow import create_checkpointer, create_full_workflow_graph
from core.service_context import RunServices, inject_services
from data_access.chapter_queries import build_chapter_upsert_statement
from tests.test_langgraph.test_chapter_lifecycle import example_state
from tests.test_r01_graph_compensation import Graph, Rows


class AttemptGraph(Graph):
    def __init__(self) -> None:
        super().__init__()
        self.receipts: dict[str, dict[str, Any]] = {}

    def run(self, query: str, parameters: Any = None, **keywords: Any) -> Rows:
        if query == ATTEMPT_QUERY:
            return Rows(deepcopy(list(self.receipts.values())))
        if query == CHAPTER_QUERY:
            return Rows([{"status": node["properties"].get("generation_status"), "attempt_id": node["properties"].get("attempt_id")} for node in self.nodes.values() if "Chapter" in node["labels"]])
        if query == CREATE_ATTEMPT:
            self.receipts[parameters["attempt_id"]] = {"id": parameters["attempt_id"], "manifest": parameters["manifest"], "phase": "committed", "acceptance": None}
            self.nodes["chapter"]["properties"].update(attempt_id=parameters["attempt_id"], graph_project_id=parameters["project_id"])
        elif query == STORE_COMPENSATION:
            self.receipts[parameters["attempt_id"]].update(compensation=parameters["compensation"], compensation_sha256=parameters["compensation_sha256"])
        elif query == UPDATE_ATTEMPT:
            self.receipts[parameters["attempt_id"]].update(phase=parameters["phase"], acceptance=parameters["acceptance"])
        elif "MERGE (c:Chapter" in query:
            self.nodes["chapter"]["properties"].update(generation_status=parameters["generation_status_param"], is_provisional=parameters["is_provisional_param"], updated_ts=10)
        else:
            return super().run(query, parameters, **keywords)
        self.writes.append(query)
        return Rows()


class JournalDatabase:
    def __init__(self, identity: str) -> None:
        self.identity = identity
        self.graph = AttemptGraph()
        self.failure = ""

    def require_project_binding(self) -> str:
        return self.identity

    async def execute_read_query(self, query: str, parameters: Any = None) -> Rows:
        return self.graph.run(query, parameters)

    async def execute_in_transaction(self, callback: Any) -> None:
        candidate = deepcopy(self.graph)
        callback(candidate)
        failure, self.failure = self.failure, ""
        if failure == "before_commit":
            raise RuntimeError("synthetic before_commit")
        self.graph = candidate
        if failure == "after_commit":
            raise RuntimeError("synthetic after_commit acknowledgement")


def staged_state(directory: Path) -> NarrativeState:
    state = example_state(directory)
    state.update({"max_iterations": 3, "needs_revision": True})
    state["extraction_source"] = extraction_binding(state, ["A synthetic traveler returns.\\n\nUnicode: 雨\n"])
    return state


@pytest.mark.parametrize("boundary", ["before_commit", "after_commit", ""])
async def test_journal_commit_and_compensation_reconcile_acknowledgement(tmp_path: Path, boundary: str) -> None:
    state = staged_state(tmp_path)
    database = JournalDatabase(state["graph_project_id"])
    before = deepcopy((database.graph.nodes, database.graph.edges))
    with inject_services(RunServices(cast(Any, object()), cast(Any, database))):
        lifecycle = ChapterLifecycle(state).stage()
        statements = [build_chapter_upsert_statement(chapter_number=1, generation_status="committed", is_provisional=True)]
        database.failure = boundary
        if boundary:
            with pytest.raises(RuntimeError, match=boundary):
                await lifecycle.commit(statements)
        else:
            await lifecycle.commit(statements)
        await ChapterLifecycle(state).stage().commit(statements)
        assert database.graph.writes.count(CREATE_ATTEMPT) == 1
        assert database.graph.writes.count(STORE_COMPENSATION) == 1
        database.failure = boundary
        if boundary:
            with pytest.raises(RuntimeError, match=boundary):
                await _rollback_chapter_data(1, lifecycle=lifecycle)
        else:
            await _rollback_chapter_data(1, lifecycle=lifecycle)
        reopened = ChapterLifecycle(state).stage()
        assert reopened.files.exists(reopened.phase_path("compensation_required"))
        await _rollback_chapter_data(1, lifecycle=reopened)
        writes = deepcopy(database.graph.writes)
        await _rollback_chapter_data(1, lifecycle=ChapterLifecycle(state).stage())
        assert database.graph.writes == writes
        assert (database.graph.nodes, database.graph.edges) == before
        assert reopened.files.exists(reopened.phase_path("compensated"))
        with pytest.raises(ValueError, match="Rejected attempt"):
            await reopened.commit(statements)
        with pytest.raises(ValueError, match="eligible"):
            await reopened.publish()


@pytest.mark.parametrize("journal", ["missing", "corrupt"])
async def test_legacy_or_corrupt_journal_stays_fail_closed(tmp_path: Path, journal: str) -> None:
    state = staged_state(tmp_path)
    database = JournalDatabase(state["graph_project_id"])
    with inject_services(RunServices(cast(Any, object()), cast(Any, database))):
        lifecycle = ChapterLifecycle(state).stage()
        await lifecycle.commit([build_chapter_upsert_statement(chapter_number=1, generation_status="committed", is_provisional=True)])
        row = database.graph.receipts[lifecycle.manifest.attempt_id]
        if journal == "missing":
            del row["compensation"]
        else:
            row["compensation"] += " "
        before = deepcopy(database.graph.__dict__)
        with pytest.raises(ValueError, match="durable compensation journal"):
            await _rollback_chapter_data(1, lifecycle=lifecycle)
        assert database.graph.__dict__ == before
        assert lifecycle.files.exists(lifecycle.phase_path("compensation_required"))


async def test_rollback_failure_survives_sqlite_reopen_and_force_continue(tmp_path: Path) -> None:
    state = staged_state(tmp_path / "project")
    database = JournalDatabase(state["graph_project_id"])
    configuration = {"configurable": {"thread_id": "r01-rollback"}}
    checkpoint_path = str(tmp_path / "checkpoint.sqlite")
    with inject_services(RunServices(cast(Any, object()), cast(Any, database))):
        lifecycle = ChapterLifecycle(state).stage()
        await lifecycle.commit([build_chapter_upsert_statement(chapter_number=1, generation_status="committed", is_provisional=True)])
        state.update(lifecycle.state_update("committed"))
        database.failure = "before_commit"
        async with create_checkpointer(checkpoint_path) as checkpointer:
            workflow = create_full_workflow_graph(checkpointer)
            await workflow.aupdate_state(configuration, state, as_node="validate")
            events = [event async for event in workflow.astream(None, configuration, stream_mode="updates")]
            assert [name for event in events for name in event] == ["revise", "error_handler"]
            saved = await workflow.aget_state(configuration)
            assert saved.values["revision_rollback_failure"]
            assert saved.values["has_fatal_error"] is True
        graph_before = deepcopy(database.graph.__dict__)
        async with create_checkpointer(checkpoint_path) as checkpointer:
            workflow = create_full_workflow_graph(checkpointer)
            restored = await workflow.aget_state(configuration)
            assert restored.values == saved.values
            replay = {**restored.values, **clear_error_state(), "force_continue": True}
            events = [event async for event in workflow.astream(replay, configuration, stream_mode="updates")]
            assert [name for event in events for name in event] == ["route", "error_handler"]
            final = await workflow.aget_state(configuration)
            assert final.values["revision_rollback_failure"] == saved.values["revision_rollback_failure"]
            assert final.values["has_fatal_error"] is True
        assert database.graph.__dict__ == graph_before
