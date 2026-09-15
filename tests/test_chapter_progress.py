"""Chapter progress uses explicit finalization, never outline cardinality."""

import json
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph  # type: ignore[attr-defined]
from neo4j.exceptions import Neo4jError

from core.exceptions import CheckpointResumeConflictError, DatabaseError
from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes import finalize_node
from core.langgraph.state import NarrativeState
from core.parsers.chapter_outline_parser import ChapterOutlineParser
from core.service_context import get_services
from data_access import chapter_queries
from orchestration.langgraph_orchestrator import LangGraphOrchestrator
from tests.fakes.quality import example_quality_state


class ChapterRows:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.queries: list[str] = []

    async def execute_read_query(self, query: str, parameters: Any = None) -> list[dict[str, Any]]:
        self.queries.append(" ".join(query.split()))
        if "count(c)" in query:
            return [{"chapter_count": len(self.rows)}]
        return self.rows


def chapter(number: Any, status: Any = "finalized", provisional: Any = False) -> dict[str, Any]:
    return {"chapter_number": number, "generation_status": status, "is_provisional": provisional}


@pytest.fixture
def chapter_rows(monkeypatch: pytest.MonkeyPatch) -> ChapterRows:
    provider = ChapterRows([])
    monkeypatch.setattr(get_services().database, "execute_read_query", provider.execute_read_query)
    return provider


@pytest.mark.parametrize("numbers, expected", [([], 1), ([1, 2, 3], 4), ([3, 1], 2), ([3, 2, 1], 4), ([2, 3], 1)])
async def test_fresh_generation_uses_contiguous_prefix(tmp_path: Path, chapter_rows: ChapterRows, numbers: list[int], expected: int) -> None:
    chapter_rows.rows = [chapter(number) for number in numbers]
    orchestrator = LangGraphOrchestrator()
    orchestrator.project_dir = tmp_path
    state = await orchestrator._load_or_create_state(project_id="synthetic", narrative_config=None)
    assert (state["current_chapter"], state["run_start_chapter"]) == (expected, expected)


async def test_planned_chapters_start_and_resume_at_one(tmp_path: Path, chapter_rows: ChapterRows) -> None:
    chapter_rows.rows = [chapter(number, "planned") for number in [1, 2, 3]]
    orchestrator = LangGraphOrchestrator()
    orchestrator.project_dir = tmp_path
    state = await orchestrator._load_or_create_state(project_id="synthetic", narrative_config=None)
    assert (state["current_chapter"], state["initialization_complete"]) == (1, False)
    await orchestrator._validate_resume_state_or_raise_async(checkpoint_state=state, requested_project_id="synthetic")


@pytest.mark.parametrize("status", ["planned", "staged", "committed", "validated", "provisional", "rejected"])
async def test_nonfinal_status_is_not_progress(chapter_rows: ChapterRows, status: str) -> None:
    chapter_rows.rows = [chapter(1, status), chapter(2)]
    progress = await chapter_queries.load_chapter_progress_from_db()
    assert (progress.last_finalized_chapter, progress.finalized_chapters) == (0, (2,))


@pytest.mark.parametrize("provisional", [True, None, 0, "false"])
async def test_finalized_requires_explicit_nonprovisional(chapter_rows: ChapterRows, provisional: Any) -> None:
    chapter_rows.rows = [chapter(1, provisional=provisional)]
    progress = await chapter_queries.load_chapter_progress_from_db()
    assert (progress.last_finalized_chapter, progress.finalized_chapters) == (0, ())


@pytest.mark.parametrize("rows", [
    [chapter(1), chapter(1)], [chapter(1), chapter(1, "planned")],
    [chapter(0)], [chapter(-1)], [chapter(True)], [chapter(1.0)], [chapter("1")], [chapter(None)],
    [chapter(1, None)], [chapter(1, "unknown")], [chapter(1, ["finalized"])],
    [{"chapter_number": 1, "is_provisional": False, "summary": "A long legacy summary"}],
    [{"generation_status": "finalized", "is_provisional": False}],
])
async def test_ambiguous_progress_fails_closed(chapter_rows: ChapterRows, rows: list[dict[str, Any]]) -> None:
    chapter_rows.rows = rows
    with pytest.raises(ValueError):
        await chapter_queries.load_chapter_progress_from_db()


async def test_read_contract_keeps_all_rows(chapter_rows: ChapterRows) -> None:
    chapter_rows.rows = [chapter(3), chapter(1), chapter(2, "planned")]
    progress = await chapter_queries.load_chapter_progress_from_db()
    assert (progress.last_finalized_chapter, progress.finalized_chapters) == (1, (1, 3))
    assert chapter_rows.queries == [
        "MATCH (c:Chapter) RETURN c.number AS chapter_number, c.generation_status AS generation_status, c.is_provisional AS is_provisional"
    ]


async def test_progress_database_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail(query: str) -> list[dict[str, Any]]:
        raise Neo4jError("synthetic failure")

    monkeypatch.setattr(get_services().database, "execute_read_query", fail)
    with pytest.raises(DatabaseError):
        await chapter_queries.load_chapter_progress_from_db()


@pytest.mark.parametrize("numbers, current_chapter", [([1, 3], 3), ([1, 3], 2), ([1, 2, 3], 3), ([], 2)])
async def test_resume_rejects_missing_prefix_or_finalized_ahead(tmp_path: Path, chapter_rows: ChapterRows, numbers: list[int], current_chapter: int) -> None:
    chapter_rows.rows = [chapter(number) for number in numbers]
    orchestrator = LangGraphOrchestrator()
    orchestrator.project_dir = tmp_path
    state: NarrativeState = {"project_id": "synthetic", "project_dir": str(tmp_path), "current_chapter": current_chapter}
    with pytest.raises(CheckpointResumeConflictError):
        await orchestrator._validate_resume_state_or_raise_async(checkpoint_state=state, requested_project_id="synthetic")


async def test_resume_accepts_prefix_and_future_plans(tmp_path: Path, chapter_rows: ChapterRows) -> None:
    chapter_rows.rows = [chapter(3, "planned"), chapter(1), chapter(2, "staged")]
    orchestrator = LangGraphOrchestrator()
    orchestrator.project_dir = tmp_path
    state: NarrativeState = {"project_id": "synthetic", "project_dir": str(tmp_path), "current_chapter": 2}
    await orchestrator._validate_resume_state_or_raise_async(checkpoint_state=state, requested_project_id="synthetic")


async def test_outline_write_marks_only_new_nodes_planned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[tuple[str, dict[str, Any]]] = []

    async def write(query: str, parameters: dict[str, Any]) -> None:
        writes.append((query, parameters))

    monkeypatch.setattr(get_services().database, "execute_write_query", write)
    parser = ChapterOutlineParser(str(tmp_path / "outline.json"))
    chapters = [parser._parse_chapter({"chapter_number": number, "act_number": 1}) for number in [1, 2, 3]]
    assert await parser.create_chapter_nodes(chapters) is True
    assert [parameters["number"] for _, parameters in writes] == [1, 2, 3]
    for query, _ in writes:
        create, match = query.split("ON MATCH SET")
        assert "c.generation_status = 'planned'" in create
        assert "c.generation_status" not in match
        assert "c.is_provisional" not in match
        assert "c.summary =" not in match


def test_generic_writes_do_not_finalize() -> None:
    query, parameters = chapter_queries.build_chapter_upsert_statement(chapter_number=1, summary="Generated summary", is_provisional=False)
    assert parameters["generation_status_param"] is None
    assert "c.generation_status = 'staged'" in query
    assert "CASE WHEN $generation_status_param IS NULL THEN [] ELSE [1] END" in query


@pytest.mark.parametrize("status, provisional", [("unknown", False), ("finalized", True), ("finalized", None)])
def test_invalid_status_writes_raise(status: str, provisional: Any) -> None:
    with pytest.raises(ValueError):
        chapter_queries.build_chapter_upsert_statement(chapter_number=1, generation_status=status, is_provisional=provisional)


async def test_finalization_write_explicitly_sets_status(monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[tuple[str, dict[str, Any]]] = []

    async def write(query: str, parameters: dict[str, Any]) -> None:
        writes.append((query, parameters))

    monkeypatch.setattr(get_services().database, "execute_write_query", write)
    await chapter_queries.save_finalized_chapter_to_db(chapter_number=1, summary="Synthetic summary")
    assert len(writes) == 1
    query, parameters = writes[0]
    assert (parameters["chapter_number_param"], parameters["generation_status_param"], parameters["is_provisional_param"]) == (1, "finalized", False)
    assert "SET c.generation_status = $generation_status_param" in query


def test_imports_resolve_to_worker_tree() -> None:
    root = Path(__file__).resolve().parents[1]
    assert Path(chapter_queries.__file__).resolve() == root / "data_access/chapter_queries.py"
    assert Path(finalize_node.__file__).resolve() == root / "core/langgraph/nodes/finalize_node.py"


@pytest.mark.parametrize("number", [0, -1, True, 1.0, "1"])
async def test_finalized_write_rejects_invalid_numbers(number: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[str] = []

    async def write(query: str, parameters: dict[str, Any]) -> None:
        writes.append(query)

    monkeypatch.setattr(get_services().database, "execute_write_query", write)
    with pytest.raises(ValueError, match="positive integer"):
        await chapter_queries.save_finalized_chapter_to_db(chapter_number=number)
    assert writes == []


class ChapterWriteJournal(ChapterRows):
    """Record the narrow Chapter query contract; not a Cypher engine."""

    async def execute_read_query(self, query: str, parameters: Any = None) -> list[dict[str, Any]]:
        if any(label in query for label in [":Character", ":Location", ":Item"]):
            return []
        return await super().execute_read_query(query, parameters)

    async def execute_write_query(self, query: str, parameters: dict[str, Any]) -> None:
        if "MERGE (c:Chapter {number: $number})" in query:
            create, match = query.split("ON MATCH SET")
            assert "c.generation_status = 'planned'" in create
            assert "c.generation_status" not in match
            assert "c.is_provisional" not in match
            number = parameters["number"]
            if not any(row["chapter_number"] == number for row in self.rows):
                self.rows.append(chapter(number, "planned", parameters["is_provisional"]))
            return
        assert "MERGE (c:Chapter {number: $chapter_number_param})" in query
        assert "SET c.generation_status = $generation_status_param" in query
        number = parameters["chapter_number_param"]
        row = next(row for row in self.rows if row["chapter_number"] == number)
        if parameters["generation_status_param"] is not None:
            row["generation_status"] = parameters["generation_status_param"]
        row["is_provisional"] = parameters["is_provisional_param"]


async def test_parser_finalizer_and_reopened_checkpoint_use_real_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    journal = ChapterWriteJournal([])
    monkeypatch.setattr(get_services().database, "execute_read_query", journal.execute_read_query)
    monkeypatch.setattr(get_services().database, "execute_write_query", journal.execute_write_query)
    outline_path = tmp_path / "outlines.json"
    outline_path.write_text(json.dumps({str(number): {"chapter_number": number, "act_number": 1} for number in [1, 2, 3]}))
    parser = ChapterOutlineParser(str(outline_path))
    assert await parser.parse_and_persist() == (True, "Successfully parsed and persisted 3 Chapter, 0 Scenes, 0 SceneEvents, 0 location references, and 0 relationships")
    assert journal.rows == [chapter(number, "planned") for number in [1, 2, 3]]
    orchestrator = LangGraphOrchestrator(project_dir=tmp_path)
    state = await orchestrator._load_or_create_state(project_id="synthetic", narrative_config=None)
    assert (state["current_chapter"], state["run_start_chapter"]) == (1, 1)
    from core.graph_ownership import load_graph_project_id

    state.update({"lifecycle_version": 1, "graph_project_id": load_graph_project_id(tmp_path)})

    def preserve_state(state: NarrativeState) -> NarrativeState:
        return {"current_chapter": state["current_chapter"]}

    checkpoint_path = tmp_path / "checkpoints.db"
    thread_id = orchestrator._checkpoint_thread_id("synthetic")
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        graph = StateGraph(NarrativeState)
        graph.add_node("preserve", preserve_state)
        graph.set_entry_point("preserve")
        graph.add_edge("preserve", END)
        await graph.compile(checkpointer=saver).ainvoke(state, config={"configurable": {"thread_id": thread_id}})

    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        resumed = await orchestrator._load_state_for_run(graph=graph.compile(checkpointer=saver), requested_project_id="synthetic", thread_id=thread_id, narrative_config=None)
    assert (resumed["current_chapter"], resumed["run_start_chapter"]) == (1, 1)

    # Exercise the retained legacy node API independently of orchestrated v1.
    del state["lifecycle_version"]
    content_manager = ContentManager(str(tmp_path))
    draft = "A short synthetic chapter."
    state["draft_ref"] = content_manager.save_text(draft, "draft", "chapter_1", 1)

    async def no_embedding(text: str) -> None:
        return None

    monkeypatch.setattr(get_services().language_model, "async_get_embedding", no_embedding)
    outcome = await finalize_node.finalize_chapter(example_quality_state(state))
    assert (outcome["current_node"], outcome["last_error"]) == ("finalize", None)
    assert (tmp_path / "chapters/chapter_001.txt").read_text() == draft
    assert (tmp_path / "chapters/chapter_001.md").read_text().split("---\n", 2)[2] == draft
    assert journal.rows == [chapter(1), chapter(2, "planned"), chapter(3, "planned")]
    assert await parser.parse_and_persist() == (True, "Successfully parsed and persisted 3 Chapter, 0 Scenes, 0 SceneEvents, 0 location references, and 0 relationships")
    assert journal.rows == [chapter(1), chapter(2, "planned"), chapter(3, "planned")]
    next_state = await orchestrator._load_or_create_state(project_id="synthetic", narrative_config=None)
    assert (next_state["current_chapter"], next_state["initialization_complete"]) == (2, True)
    await orchestrator._validate_resume_state_or_raise_async(checkpoint_state=next_state, requested_project_id="synthetic")


@pytest.mark.parametrize("failed_suffix", [".md", ".txt"])
async def test_partial_filesystem_failure_cannot_finalize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_suffix: str) -> None:
    journal = ChapterWriteJournal([chapter(1, "planned")])
    monkeypatch.setattr(get_services().database, "execute_read_query", journal.execute_read_query)
    monkeypatch.setattr(get_services().database, "execute_write_query", journal.execute_write_query)
    import os

    real_replace = os.replace
    writes: list[str] = []

    def failing_replace(source: Any, destination: Any, **keywords: Any) -> None:
        path = Path(destination)
        if path.name in {"chapter_001.md", "chapter_001.txt"}:
            writes.append(path.suffix)
            if path.suffix == failed_suffix:
                raise OSError("synthetic disk failure")
        real_replace(source, destination, **keywords)

    async def no_embedding(text: str) -> None:
        return None

    monkeypatch.setattr(os, "replace", failing_replace)
    monkeypatch.setattr(get_services().language_model, "async_get_embedding", no_embedding)
    content_manager = ContentManager(str(tmp_path))
    state: NarrativeState = {
        "project_dir": str(tmp_path), "current_chapter": 1,
        "draft_ref": content_manager.save_text("Synthetic prose.", "draft", "chapter_1", 1),
    }
    result = await finalize_node.finalize_chapter(example_quality_state(state))
    assert result == {
        "last_error": "Error saving chapter to filesystem: synthetic disk failure",
        "has_fatal_error": True, "error_node": "finalize", "current_node": "finalize",
    }
    assert writes == ([".md"] if failed_suffix == ".md" else [".md", ".txt"])
    assert journal.rows == [chapter(1, "planned")]
    assert await chapter_queries.load_chapter_progress_from_db() == chapter_queries.ChapterProgress(0, ())
    assert (tmp_path / "chapters/chapter_001.md").exists() is (failed_suffix == ".txt")
    assert (tmp_path / "chapters/chapter_001.txt").exists() is False

    monkeypatch.setattr(os, "replace", real_replace)
    result = await finalize_node.finalize_chapter(example_quality_state(state))
    assert (result["current_node"], result["last_error"]) == ("finalize", None)
    assert journal.rows == [chapter(1)]
    assert (tmp_path / "chapters/chapter_001.txt").read_text() == "Synthetic prose."
