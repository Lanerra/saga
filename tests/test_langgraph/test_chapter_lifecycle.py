"""Behavioral chapter lifecycle regressions using synthetic retained content."""

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
from langgraph.constants import END
from langgraph.graph.state import StateGraph
from structlog.testing import capture_logs

from core.db_manager import neo4j_manager
from core.exceptions import WorkflowExecutionError
from core.graph_ownership import OWNER_QUERY, load_graph_project_id
from core.langgraph.chapter_lifecycle import ATTEMPT_QUERY, CHAPTER_QUERY, CREATE_ATTEMPT, UPDATE_ATTEMPT, ChapterLifecycle, extraction_binding
from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.commit_node import commit_to_graph
from core.langgraph.nodes.finalize_node import finalize_chapter
from core.langgraph.nodes.revision_node import _rollback_chapter_data
from core.langgraph.state import NarrativeState
from core.langgraph.workflow import advance_chapter, create_checkpointer
from core.service_context import get_services
from data_access import character_queries, kg_queries, world_queries
from orchestration.langgraph_orchestrator import LangGraphOrchestrator
from tests.fakes.quality import example_quality_state
from tests.fakes.schema_catalog import schema_catalog
from utils.file_io import ContainedFiles


def example_state(directory: Path) -> NarrativeState:
    directory.mkdir(exist_ok=True)
    manager = ContentManager(str(directory))
    scenes = ["A synthetic traveler returns.\\n\nUnicode: 雨\n"]
    state = cast(
        NarrativeState,
        {
            "project_dir": str(directory),
            "project_id": "synthetic",
            "lifecycle_version": 1,
            "graph_project_id": load_graph_project_id(directory),
            "current_chapter": 1,
            "iteration_count": 0,
            "initialization_complete": True,
            "total_chapters": 1,
            "draft_ref": manager.save_text(scenes[0], "draft", "chapter_1"),
            "scene_drafts_ref": manager.save_list_of_texts(scenes, "scenes", "chapter_1"),
            "extracted_entities_ref": manager.save_json({"characters": [], "world_items": []}, "extracted_entities", "chapter_1"),
            "extracted_relationships_ref": manager.save_json([], "extracted_relationships", "chapter_1"),
            "chapter_plan_scene_count": 1,
            "current_summary": "Synthetic return.",
            "extraction_status": "complete",
            "extraction_policy": "fail_closed",
            "extraction_outcomes": [
                {"chapter_number": 1, "scene_index": 0, "extraction_type": kind, "status": "succeeded", "item_count": 0, "error": "", "error_type": ""}
                for kind in ("characters", "locations", "events", "relationships")
            ],
        },
    )
    return example_quality_state(state)


async def test_stale_completion_cannot_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = example_state(tmp_path)
    calls: list[str] = []

    async def read(query: str, parameters: Any = None) -> list[dict[str, Any]]:
        calls.append("read")
        return []

    async def batch(statements: Any) -> None:
        calls.append("batch")

    monkeypatch.setattr(neo4j_manager, "execute_read_query", read)
    monkeypatch.setattr(neo4j_manager, "execute_cypher_batch", batch)
    result = await commit_to_graph(state)
    assert result.get("has_fatal_error") is True
    assert calls == []


class Rows(list[dict[str, Any]]):
    def consume(self) -> None:
        pass


class TransactionExample:
    def __init__(self, driver: "DriverExample") -> None:
        self.driver = driver
        self.receipts = deepcopy(driver.receipts)
        self.chapters = deepcopy(driver.chapters)
        self.statements: list[str] = []
        self.is_closed = False

    def run(self, query: str, parameters: Any = None) -> Rows:
        catalog = schema_catalog()
        if query in catalog:
            return Rows(catalog[query])
        if query == OWNER_QUERY:
            return Rows([{"key": "exclusive", "project_id": self.driver.project_id, "version": 1}])
        if query == ATTEMPT_QUERY:
            return Rows(list(self.receipts.values()))
        if query == CHAPTER_QUERY:
            return Rows(self.chapters)
        if query == CREATE_ATTEMPT:
            self.receipts[parameters["attempt_id"]] = {
                "id": parameters["attempt_id"],
                "manifest": parameters["manifest"],
                "phase": "committed",
                "acceptance": None,
            }
            self.chapters = [{"status": "committed", "attempt_id": parameters["attempt_id"]}]
        elif query == UPDATE_ATTEMPT:
            self.receipts[parameters["attempt_id"]].update(phase=parameters["phase"], acceptance=parameters["acceptance"])
        elif "MERGE (c:Chapter" in query:
            self.chapters = [{"status": parameters["generation_status_param"], "attempt_id": self.chapters[0]["attempt_id"] if self.chapters else None}]
        elif query.strip().startswith("MATCH (ch:Chapter") and "DELETE r, ch" in query:
            self.chapters = []
        elif "DELETE r" in query or "SET e.is_provisional" in query:
            pass
        elif "RETURN" in query and ("name" in query or "labels" in query):
            return Rows()
        else:
            raise AssertionError(f"Unimplemented synthetic query: {query}")
        self.statements.append(query)
        return Rows()

    def commit(self) -> None:
        if self.driver.failure == "before_commit":
            self.driver.failure = ""
            raise RuntimeError("interrupted before commit")
        self.driver.receipts = self.receipts
        self.driver.chapters = self.chapters
        self.driver.writes.extend(self.statements)
        self.driver.commits += 1
        self.is_closed = True
        if self.driver.failure == "after_commit":
            self.driver.failure = ""
            raise RuntimeError("lost acknowledgement")

    def rollback(self) -> None:
        self.is_closed = True

    def closed(self) -> bool:
        return self.is_closed


class DriverExample:
    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self.receipts: dict[str, dict[str, Any]] = {}
        self.chapters: list[dict[str, Any]] = []
        self.writes: list[str] = []
        self.commits = 0
        self.failure = ""

    def session(self, **arguments: Any) -> "DriverExample":
        return self

    def __enter__(self) -> "DriverExample":
        return self

    def __exit__(self, *arguments: Any) -> None:
        pass

    def begin_transaction(self) -> TransactionExample:
        return TransactionExample(self)

    def execute_read(self, callback: Any, *arguments: Any) -> Any:
        return callback(TransactionExample(self), *arguments)


@pytest.fixture
def lifecycle_example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[NarrativeState, DriverExample]:
    state = example_state(tmp_path)
    assert state["scene_drafts_ref"] is not None
    scenes = ContentManager(str(tmp_path)).load_list_of_texts(state["scene_drafts_ref"])
    state["extraction_source"] = extraction_binding(state, scenes)
    for name in ("_project_id", "_database", "_uri"):
        monkeypatch.setattr(neo4j_manager, name, None)
    neo4j_manager.bind_project(state["graph_project_id"])
    driver = DriverExample(state["graph_project_id"])
    monkeypatch.setattr(neo4j_manager, "driver", driver)
    return state, driver


@pytest.mark.parametrize("boundary", ["before_commit", "after_commit", ""])
async def test_commit_reentry_reconciles_acknowledgement(lifecycle_example: tuple[NarrativeState, DriverExample], boundary: str) -> None:
    state, driver = lifecycle_example
    driver.failure = boundary
    first = await commit_to_graph(state)
    assert first.get("has_fatal_error") is bool(boundary)
    second = await commit_to_graph(state)
    assert second.get("has_fatal_error") is False
    assert second["lifecycle_phase"] == "committed"
    assert driver.commits == 1
    assert len(driver.receipts) == 1
    assert second["draft_ref"] is not None and state["draft_ref"] is not None
    assert second["draft_ref"]["path"] != state["draft_ref"]["path"]


async def test_committed_replay_retries_cache_maintenance_without_graph_writes(
    lifecycle_example: tuple[NarrativeState, DriverExample], monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, driver = lifecycle_example
    first = await commit_to_graph(state)
    assert first["has_fatal_error"] is False
    before = deepcopy((driver.receipts, driver.chapters, driver.writes))
    calls: list[str] = []
    world_clear = world_queries.get_world_item_by_id.cache_clear
    graph_clear = cast(Any, kg_queries.query_kg_from_db).cache_clear

    def fail_character() -> None:
        calls.append("character")
        raise RuntimeError("synthetic replay cache failure")

    def clear_world() -> None:
        calls.append("world")
        world_clear()

    def clear_graph() -> None:
        calls.append("kg")
        graph_clear()

    with monkeypatch.context() as context:
        context.setattr(character_queries.get_character_profile_by_name, "cache_clear", fail_character)
        context.setattr(world_queries.get_world_item_by_id, "cache_clear", clear_world)
        context.setattr(kg_queries.query_kg_from_db, "cache_clear", clear_graph)
        with capture_logs() as logs:
            replay = await commit_to_graph(state)
        assert replay == first
        assert calls == ["character", "world", "kg"]
        assert [entry for entry in logs if entry["event"] == "commit_to_graph: postcommit cache invalidation failed"] == [{
            "event": "commit_to_graph: postcommit cache invalidation failed", "chapter": 1, "cache": "character",
            "error": "synthetic replay cache failure", "log_level": "warning",
        }]

    with capture_logs() as logs:
        assert await commit_to_graph(state) == first
    assert [entry["log_level"] for entry in logs if entry["event"] == "commit_to_graph: postcommit caches invalidated"] == ["info"]
    assert (driver.receipts, driver.chapters, driver.writes) == before
    assert driver.commits == 1


@pytest.mark.parametrize("boundary", ["before_commit", "after_commit", ""])
async def test_publication_reentry_uses_exact_accepted_candidate(lifecycle_example: tuple[NarrativeState, DriverExample], boundary: str) -> None:
    state, driver = lifecycle_example
    state = {**state, **await commit_to_graph(state)}
    driver.failure = boundary
    first = await finalize_chapter(state)
    assert first.get("has_fatal_error") is bool(boundary)
    lifecycle = ChapterLifecycle(state).stage()
    intent = lifecycle.files.read_bytes(lifecycle.phase_path("acceptance"))
    second = await finalize_chapter(state)
    assert second.get("has_fatal_error") is False
    assert second["lifecycle_phase"] == "published"
    assert driver.commits == 2
    assert lifecycle.files.read_bytes(lifecycle.phase_path("acceptance")) == intent
    assert advance_chapter({**state, **second})["lifecycle_phase"] == "advanced"
    assert lifecycle.files.exists(lifecycle.phase_path("advanced")) is True


@pytest.mark.parametrize("boundary", ["before_commit", "after_commit", ""])
async def test_compensation_reentry_preserves_verified_barrier(lifecycle_example: tuple[NarrativeState, DriverExample], boundary: str) -> None:
    state, driver = lifecycle_example
    state = {**state, **await commit_to_graph(state)}
    lifecycle = ChapterLifecycle(state).stage()
    driver.failure = boundary
    if boundary:
        with pytest.raises(Exception, match="Transaction failed"):
            await _rollback_chapter_data(1, lifecycle=lifecycle)
    else:
        await _rollback_chapter_data(1, lifecycle=lifecycle)
    assert lifecycle.files.exists(lifecycle.phase_path("compensation_required")) is True
    await _rollback_chapter_data(1, lifecycle=lifecycle)
    assert lifecycle.files.exists(lifecycle.phase_path("compensated")) is True
    assert driver.commits == 2
    assert driver.chapters == []
    assert (await commit_to_graph(state))["has_fatal_error"] is True


def test_staging_retains_bytes_after_source_overwrite(lifecycle_example: tuple[NarrativeState, DriverExample]) -> None:
    state, _ = lifecycle_example
    lifecycle = ChapterLifecycle(state).stage()
    manager = ContentManager(state["project_dir"])
    with pytest.raises(FileExistsError, match="Immutable content conflict"):
        manager.save_text("Different synthetic prose", "draft", "chapter_1")
    manager.save_text("Different synthetic prose", "draft", "chapter_1", 2)
    source_reference = state["draft_ref"]
    assert source_reference is not None
    assert manager.load_text_strict(source_reference) == "A synthetic traveler returns.\\n\nUnicode: 雨\n"
    (manager.project_dir / source_reference["path"]).write_text("External synthetic corruption")
    replay = ChapterLifecycle(state).stage()
    assert replay.manifest == lifecycle.manifest
    assert replay.files.read_bytes(replay.artifact_ref("draft_ref")["path"]) == b"A synthetic traveler returns.\\n\nUnicode: \xe9\x9b\xa8\n"


def test_unpublished_attempt_cannot_advance(lifecycle_example: tuple[NarrativeState, DriverExample]) -> None:
    state, _ = lifecycle_example
    lifecycle = ChapterLifecycle(state).stage()
    with pytest.raises(ValueError, match="unpublished"):
        advance_chapter({**state, **lifecycle.state_update("staged")})


async def test_native_resume_starts_at_pending_node(lifecycle_example: tuple[NarrativeState, DriverExample], tmp_path: Path) -> None:
    state, driver = lifecycle_example
    visited: list[str] = []

    def route(value: NarrativeState) -> NarrativeState:
        visited.append("route")
        return {"current_node": "route"}

    def validate(value: NarrativeState) -> NarrativeState:
        visited.append("validate")
        return {"current_node": "validate"}

    workflow = StateGraph(NarrativeState)
    workflow.add_node("route", route)
    workflow.add_node("commit", commit_to_graph)
    workflow.add_node("validate", validate)
    workflow.add_node("finalize", finalize_chapter)
    workflow.set_entry_point("route")
    workflow.add_edge("route", "commit")
    workflow.add_edge("commit", "validate")
    workflow.add_edge("validate", "finalize")
    workflow.add_edge("finalize", END)
    configuration = {"configurable": {"thread_id": "saga_synthetic"}}
    async with create_checkpointer(str(tmp_path / "checkpoint.db")) as saver:
        graph = workflow.compile(checkpointer=saver)
        await graph.ainvoke(state, configuration, interrupt_after=["commit"])
        snapshot = await graph.aget_state(configuration)
        assert snapshot.next == ("validate",)
    visited.clear()
    async with create_checkpointer(str(tmp_path / "checkpoint.db")) as saver:
        graph = workflow.compile(checkpointer=saver)
        orchestrator = LangGraphOrchestrator(project_dir=tmp_path)
        orchestrator._resume_checkpoint = True
        await orchestrator._run_chapter_generation_loop(graph, snapshot.values)
    assert visited == ["validate"]
    assert driver.commits == 2


async def test_acceptance_recovery_skips_lagging_validation(lifecycle_example: tuple[NarrativeState, DriverExample], tmp_path: Path) -> None:
    from core.langgraph.chapter_lifecycle import reconcile_checkpoint

    state, _ = lifecycle_example
    state = {**state, **await commit_to_graph(state)}
    visits: list[str] = []

    def validate(value: NarrativeState) -> NarrativeState:
        visits.append("validate")
        return {"current_node": "validate"}

    workflow = StateGraph(NarrativeState)
    workflow.add_node("commit", commit_to_graph)
    workflow.add_node("validate", validate)
    workflow.add_node("finalize", finalize_chapter)
    workflow.add_node("advance_chapter", advance_chapter)
    workflow.set_entry_point("commit")
    workflow.add_edge("commit", "validate")
    workflow.add_edge("validate", "finalize")
    workflow.add_edge("finalize", "advance_chapter")
    workflow.add_edge("advance_chapter", END)
    configuration = {"configurable": {"thread_id": "lagging"}}
    async with create_checkpointer(str(tmp_path / "lagging.db")) as saver:
        graph = workflow.compile(checkpointer=saver)
        await graph.aupdate_state(configuration, state, as_node="commit")
        assert (await finalize_chapter(state))["lifecycle_phase"] == "published"
        await reconcile_checkpoint(graph, state, configuration)
        await graph.ainvoke(None, configuration)
    assert visits == []


def test_explicit_attempt_cannot_cross_revision_iteration(lifecycle_example: tuple[NarrativeState, DriverExample]) -> None:
    state, _ = lifecycle_example
    lifecycle = ChapterLifecycle(state).stage()
    with pytest.raises(ValueError, match="iteration"):
        ChapterLifecycle({**state, **lifecycle.state_update("staged"), "iteration_count": 1}).stage()


async def test_publication_rejects_divergent_graph_projection(lifecycle_example: tuple[NarrativeState, DriverExample]) -> None:
    state, driver = lifecycle_example
    state = {**state, **await commit_to_graph(state)}
    assert (await finalize_chapter(state))["lifecycle_phase"] == "published"
    driver.chapters[0]["status"] = "planned"
    result = await finalize_chapter(state)
    assert result["has_fatal_error"] is True


async def test_publication_does_not_replace_another_accepted_selection(lifecycle_example: tuple[NarrativeState, DriverExample]) -> None:
    state, _ = lifecycle_example
    state = {**state, **await commit_to_graph(state)}
    lifecycle = ChapterLifecycle(state).stage()
    assert (await finalize_chapter(state))["lifecycle_phase"] == "published"
    other = lifecycle.manuscripts.prepare(1, "Previously accepted synthetic canon.")
    lifecycle.manuscripts.accept(other)
    result = await finalize_chapter(state)
    assert result["has_fatal_error"] is True
    assert lifecycle.manuscripts.accepted(1) == other


@pytest.mark.parametrize("field", ["draft_ref", "scene_drafts_ref", "extracted_entities_ref", "extracted_relationships_ref"])
def test_retained_corruption_rejected(lifecycle_example: tuple[NarrativeState, DriverExample], field: str) -> None:
    state, _ = lifecycle_example
    lifecycle = ChapterLifecycle(state).stage()
    lifecycle.files.write_bytes(lifecycle.artifact_ref(field)["path"], b"corruption")
    with pytest.raises(ValueError, match="checksum or size"):
        ChapterLifecycle(state).stage()


async def test_real_extraction_assembly_binds_commit(lifecycle_example: tuple[NarrativeState, DriverExample], monkeypatch: pytest.MonkeyPatch) -> None:
    from core.langgraph.nodes import scene_extraction
    from core.langgraph.nodes.assemble_chapter_node import assemble_chapter

    state, _ = lifecycle_example
    calls: list[str] = []

    async def provider(*args: Any, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        calls.append("extraction")
        return '{"character_updates": {}, "world_updates": {"Location": {}, "Event": {}}, "kg_triples": []}', {}

    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    state = {**state, **await assemble_chapter(state)}
    state = cast(NarrativeState, {**state, **await scene_extraction.extract_from_scenes(state)})
    assert state["extraction_status"] == "complete"
    assert len(calls) == 4
    result = await commit_to_graph(state)
    assert result["has_fatal_error"] is False
    assert result["lifecycle_phase"] == "committed"


async def test_missing_checkpoint_cannot_adopt_graph_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from langgraph.types import StateSnapshot

    from core.exceptions import CheckpointResumeConflictError

    class EmptyGraph:
        async def aget_state(self, configuration: Any) -> StateSnapshot:
            return StateSnapshot(values={}, next=(), config=configuration, metadata=None, created_at=None, parent_config=None, tasks=(), interrupts=())

    orchestrator = LangGraphOrchestrator(project_dir=tmp_path)

    async def seed(**kwargs: Any) -> NarrativeState:
        return {"project_id": "synthetic", "project_dir": str(tmp_path), "current_chapter": 2}

    monkeypatch.setattr(orchestrator, "_load_or_create_state", seed)
    with pytest.raises(CheckpointResumeConflictError, match="checkpoint"):
        await orchestrator._load_state_for_run(graph=EmptyGraph(), requested_project_id="synthetic", thread_id="saga_synthetic", narrative_config=None)


def test_exclusive_checkpoint_writer_lock(tmp_path: Path) -> None:
    files = ContainedFiles(tmp_path, durable=True)
    with files.exclusive_lock("checkpoints/writer.lock"):
        with pytest.raises(BlockingIOError):
            with files.exclusive_lock("checkpoints/writer.lock"):
                raise AssertionError("Second writer admitted")
    with files.exclusive_lock("checkpoints/writer.lock"):
        assert files.exists("checkpoints/writer.lock") is True


@pytest.mark.parametrize("total_chapters", [1, 2])
async def test_pending_finalize_preserves_native_successors(lifecycle_example: tuple[NarrativeState, DriverExample], tmp_path: Path, total_chapters: int) -> None:
    from core.langgraph.chapter_lifecycle import reconcile_checkpoint

    state, driver = lifecycle_example
    state = {**state, **await commit_to_graph(state), "total_chapters": total_chapters, "run_start_chapter": 1, "current_node": "summarize"}
    visited: list[str] = []

    async def finalize(value: NarrativeState) -> NarrativeState:
        visited.append("finalize")
        return await finalize_chapter(value)

    def heal(value: NarrativeState) -> NarrativeState:
        visited.append("heal_graph")
        return {"current_node": "heal_graph"}

    def quality(value: NarrativeState) -> NarrativeState:
        visited.append("check_quality")
        return {"current_node": "check_quality"}

    workflow = StateGraph(NarrativeState)
    workflow.add_node("summarize", lambda value: value)
    workflow.add_node("finalize", finalize)
    workflow.add_node("heal_graph", heal)
    workflow.add_node("check_quality", quality)
    workflow.add_node("advance_chapter", advance_chapter)
    workflow.set_entry_point("summarize")
    workflow.add_edge("summarize", "finalize")
    workflow.add_edge("finalize", "heal_graph")
    workflow.add_edge("heal_graph", "check_quality")
    workflow.add_edge("check_quality", END)
    workflow.add_edge("advance_chapter", END)
    configuration = {"configurable": {"thread_id": "pending-finalize"}}
    async with create_checkpointer(str(tmp_path / "pending-finalize.db")) as saver:
        graph = workflow.compile(checkpointer=saver)
        saved_configuration = await graph.aupdate_state(configuration, state, as_node="summarize")
        raw = await graph.aget_state(saved_configuration)
        task, = raw.tasks
        publication = await finalize_chapter(state)
        await saver.aput_writes(saved_configuration, [*publication.items(), ("branch:to:heal_graph", None)], task.id)
        effective = await graph.aget_state(configuration)
        assert raw.next == ("finalize",)
        assert effective.next == ()
        assert effective.values["current_node"] == "finalize"
        reconciled = await reconcile_checkpoint(graph, effective.values, configuration)
        assert reconciled["run_start_chapter"] == 1
        assert reconciled["current_chapter"] == 1
        assert (await graph.aget_state(configuration)).config == saved_configuration
        result = await graph.ainvoke(None, configuration)
        assert result["run_start_chapter"] == 1
        assert result["current_chapter"] == 1
    assert visited == ["heal_graph", "check_quality"]
    assert driver.commits == 2


@pytest.mark.parametrize("total_chapters", [1, 2])
async def test_physical_end_starts_only_nonterminal_next_chapter(lifecycle_example: tuple[NarrativeState, DriverExample], tmp_path: Path, total_chapters: int) -> None:
    state, driver = lifecycle_example
    state = {**state, **await commit_to_graph(state), "total_chapters": total_chapters, "run_start_chapter": 1}
    state = {**state, **await finalize_chapter(state), "current_node": "check_quality"}
    visited: list[str] = []

    def outline(value: NarrativeState) -> NarrativeState:
        visited.append("chapter_outline")
        return {"current_node": "chapter_outline"}

    workflow = StateGraph(NarrativeState)
    workflow.add_node("check_quality", lambda value: value)
    workflow.add_node("advance_chapter", advance_chapter)
    workflow.add_node("chapter_outline", outline)
    workflow.set_entry_point("check_quality")
    workflow.add_edge("check_quality", END)
    workflow.add_edge("advance_chapter", "chapter_outline")
    workflow.add_edge("chapter_outline", END)
    configuration = {"configurable": {"thread_id": "saga_synthetic"}}
    async with create_checkpointer(str(tmp_path / "physical-end.db")) as saver:
        graph = workflow.compile(checkpointer=saver)
        saved_configuration = await graph.aupdate_state(configuration, state, as_node="check_quality")
        raw = await graph.aget_state(saved_configuration)
        assert raw.next == raw.tasks == ()
    lifecycle = ChapterLifecycle(state).stage()
    accepted = lifecycle.manuscripts.accepted(1)
    writes = list(driver.writes)
    async with create_checkpointer(str(tmp_path / "physical-end.db")) as saver:
        graph = workflow.compile(checkpointer=saver)
        orchestrator = LangGraphOrchestrator(project_dir=tmp_path)
        orchestrator._resume_checkpoint = True
        if total_chapters == 1:
            for _ in range(2):
                await orchestrator._run_chapter_generation_loop(graph, raw.values)
                assert (await graph.aget_state(configuration)).config == saved_configuration
        else:
            with pytest.raises(WorkflowExecutionError, match="incomplete"):
                await orchestrator._run_chapter_generation_loop(graph, raw.values)
        result = (await graph.aget_state(configuration)).values
    assert result["current_chapter"] == total_chapters
    assert result["run_start_chapter"] == total_chapters
    assert visited == ([] if total_chapters == 1 else ["chapter_outline"])
    assert driver.commits == 2
    assert driver.writes == writes
    assert lifecycle.manuscripts.accepted(1) == accepted


@pytest.mark.parametrize("successful_pending_write", [False, True])
async def test_accepted_advance_can_end_a_native_invocation(
    lifecycle_example: tuple[NarrativeState, DriverExample], tmp_path: Path, successful_pending_write: bool,
) -> None:
    state, driver = lifecycle_example
    state = {**state, **await commit_to_graph(state), "total_chapters": 2, "run_start_chapter": 1}
    state = {**state, **await finalize_chapter(state)}
    workflow = StateGraph(NarrativeState)
    workflow.add_node("finalize", lambda value: value)
    workflow.add_node("advance_chapter", advance_chapter)
    workflow.set_entry_point("finalize")
    workflow.add_edge("finalize", "advance_chapter")
    workflow.add_edge("advance_chapter", END)
    configuration = {"configurable": {"thread_id": "saga_synthetic"}}
    checkpoint_path = str(tmp_path / "advance.db")
    async with create_checkpointer(checkpoint_path) as saver:
        graph = workflow.compile(checkpointer=saver)
        saved = await graph.aupdate_state(configuration, state, as_node="finalize")
        if successful_pending_write:
            snapshot = await graph.aget_state(saved)
            task, = snapshot.tasks
            await saver.aput_writes(saved, list(advance_chapter(state).items()), task.id)
    async with create_checkpointer(checkpoint_path) as saver:
        graph = workflow.compile(checkpointer=saver)
        orchestrator = LangGraphOrchestrator(project_dir=tmp_path)
        orchestrator._resume_checkpoint = True
        await orchestrator._run_chapter_generation_loop(graph, (await graph.aget_state(configuration)).values)
        final = await graph.aget_state(configuration)
        assert final.next == final.tasks == ()
        assert final.values["current_chapter"] == 2
        assert final.values["run_start_chapter"] == 1
        assert final.values["lifecycle_phase"] == "advanced"
    assert driver.commits == 2
