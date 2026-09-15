# tests/test_langgraph/test_revision_node.py
"""
Tests for LangGraph revision node behavior in the scene-first pipeline.

This suite validates that `revise_chapter()` produces externalized revision guidance and
clears stale chapter artifacts so that the workflow regenerates scenes.
"""

import inspect
from collections.abc import Iterator
from contextlib import aclosing
from copy import deepcopy
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.db_manager import Neo4jManagerSingleton
from core.langgraph.chapter_lifecycle import ChapterLifecycle, canonical_bytes, extraction_binding
from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.revision_node import revise_chapter
from core.langgraph.state import Contradiction, NarrativeState, create_initial_state
from core.langgraph.state_helpers import clear_error_state
from core.langgraph.workflow import create_checkpointer, create_full_workflow_graph
from core.service_context import get_services
from data_access.chapter_queries import build_chapter_upsert_statement
from tests.fakes.service_context import patch_service
from tests.test_langgraph.test_chapter_lifecycle import DriverExample, example_state


@pytest.fixture(autouse=True)
def revision_database(monkeypatch: pytest.MonkeyPatch) -> Neo4jManagerSingleton:
    monkeypatch.setattr(Neo4jManagerSingleton, "_instance", None)
    database = Neo4jManagerSingleton()
    monkeypatch.setattr(get_services(), 'database', database)
    return database


@pytest.fixture
async def sample_revision_state(tmp_path: Path, revision_database: Neo4jManagerSingleton) -> NarrativeState:
    project_dir = str(tmp_path / "test-project")

    state = create_initial_state(
        project_id="test-project",
        title="Test Novel",
        genre="Fantasy",
        theme="Adventure",
        setting="Medieval world",
        target_word_count=80000,
        total_chapters=20,
        project_dir=project_dir,
        protagonist_name="Hero",
        medium_model="test-model",
        revision_model="test-revision-model",
    )
    state.update(example_state(Path(project_dir)))
    state["project_id"] = "test-project"
    state["total_chapters"] = 20

    content_manager = ContentManager(project_dir)

    chapter_plan = [
        {
            "title": "Into the Forest",
            "pov_character": "Hero",
            "setting": "Dark forest",
            "characters_involved": ["Hero"],
            "plot_point": "The hero begins their journey",
            "conflict": "Unease and ominous signs",
            "outcome": "Hero continues forward",
        },
        {
            "title": "The Dragon Appears",
            "pov_character": "Hero",
            "setting": "Forest clearing",
            "characters_involved": ["Hero", "Dragon"],
            "plot_point": "The dragon threatens the hero",
            "conflict": "Dragon confrontation",
            "outcome": "Hero survives and retreats",
        },
    ]
    chapter_plan_ref = content_manager.save_json(chapter_plan, "chapter_plan", "chapter_1", 1)
    state["chapter_plan_ref"] = chapter_plan_ref

    chapter_outlines = {
        1: {
            "plot_point": "The hero begins their journey",
            "chapter_summary": "Introduction to the protagonist",
        }
    }
    chapter_outlines_ref = content_manager.save_json(chapter_outlines, "chapter_outlines", "all", 1)
    state["chapter_outlines_ref"] = chapter_outlines_ref

    state["contradictions"] = [
        Contradiction(
            type="character_trait",
            description="Hero acts cowardly, contradicting established brave trait",
            conflicting_chapters=[1],
            severity="major",
            suggested_fix="Revise to show Hero's courage",
        ),
        Contradiction(
            type="plot_consistency",
            description="Dragon appears without foreshadowing",
            conflicting_chapters=[1],
            severity="minor",
            suggested_fix="Add earlier hints of dragon presence",
        ),
    ]

    state["iteration_count"] = 0
    state["max_iterations"] = 3
    state["needs_revision"] = True

    state["scene_embeddings_ref"] = content_manager.save_json([[0.1, 0.2]], "scene_embeddings", "chapter_1", 1)
    state["embedding_ref"] = content_manager.save_json([0.1, 0.2], "embedding", "chapter_1", 1)
    state["generated_embedding"] = [0.1, 0.2]
    assert state["scene_drafts_ref"] is not None
    scenes = content_manager.load_list_of_texts(state["scene_drafts_ref"])
    state["extraction_source"] = extraction_binding(state, scenes)
    revision_database.bind_project(state["graph_project_id"])
    revision_database.driver = cast(Any, DriverExample(state["graph_project_id"]))
    lifecycle = ChapterLifecycle(state).stage()
    await lifecycle.commit([build_chapter_upsert_statement(chapter_number=1, generation_status="committed", is_provisional=True)])
    state.update(lifecycle.state_update("committed"))

    return state


@pytest.fixture
def mock_llm_guidance() -> Iterator[MagicMock]:
    with patch_service('language_model') as mock_llm:
        mock_llm.async_call_llm = AsyncMock(return_value=("- Fix Scene 1: add foreshadowing.\n- Fix Scene 2: show courage.", {}))
        mock_llm.count_tokens = lambda text, model: 800
        yield mock_llm


@pytest.mark.asyncio
async def test_revise_chapter_sets_revision_guidance_and_clears_artifacts(
    sample_revision_state: NarrativeState,
    mock_llm_guidance: MagicMock,
) -> None:
    result = await revise_chapter(sample_revision_state)

    assert result["has_fatal_error"] is False
    assert result["last_error"] is None
    assert result["current_node"] == "revise"
    assert result["iteration_count"] == 1
    assert result["needs_revision"] is False
    assert result["contradictions"] == []

    assert result["revision_guidance_ref"] is not None
    assert isinstance(result["revision_guidance_ref"], dict)
    assert isinstance(result["revision_guidance_ref"].get("path"), str)

    assert result["scene_drafts_ref"] is None
    assert result["scene_embeddings_ref"] is None
    assert result["draft_ref"] is None
    assert result["embedding_ref"] is None
    assert result["extracted_entities_ref"] is None
    assert result["extracted_relationships_ref"] is None
    assert result["generated_embedding"] is None
    assert result["current_scene_index"] == 0

    content_manager = ContentManager(sample_revision_state["project_dir"])
    guidance_text = content_manager.load_text(result["revision_guidance_ref"])
    assert guidance_text == "- Fix Scene 1: add foreshadowing.\n- Fix Scene 2: show courage."

    mock_llm_guidance.async_call_llm.assert_called_once()


@pytest.mark.asyncio
async def test_revise_chapter_guidance_generation_failure_sets_fatal_error(
    sample_revision_state: NarrativeState,
) -> None:
    with patch_service('language_model') as mock_llm:
        mock_llm.count_tokens = lambda text, model: 800
        mock_llm.async_call_llm = AsyncMock(side_effect=RuntimeError("LLM exploded"))

        result = await revise_chapter(sample_revision_state)

    assert result["has_fatal_error"] is True
    assert result["error_node"] == "revise"
    assert result["current_node"] == "revise"
    assert result["last_error"] == "Revision guidance generation failed"


class FailingRollbackDatabase:
    def __init__(self, database: Neo4jManagerSingleton, monkeypatch: pytest.MonkeyPatch) -> None:
        self.reads: list[tuple[str, Any]] = []
        self.fail = True
        self.read = database.execute_read_query
        monkeypatch.setattr(database, "execute_read_query", self.execute_read_query)

    async def execute_read_query(self, query: str, parameters: Any = None) -> list[dict[str, Any]]:
        self.reads.append((query, deepcopy(parameters)))
        if self.fail:
            raise RuntimeError("rollback acknowledgement lost")
        return await self.read(query, parameters)


@pytest.fixture
def rollback_failure_state(sample_revision_state: dict[str, Any]) -> dict[str, Any]:
    assert Path(inspect.getfile(revise_chapter)).resolve() == Path(__file__).resolve().parents[2] / "core/langgraph/nodes/revision_node.py"
    state = sample_revision_state
    manager = ContentManager(state["project_dir"])

    state["revision_guidance_ref"] = manager.save_text("Prior guidance evidence", "revision_guidance", "chapter_1", 1)
    state["last_error"] = "Prior validation diagnostic"
    state["error_node"] = "validate"
    state["initialization_complete"] = True
    return state


@pytest.mark.asyncio
async def test_failed_rollback_preserves_evidence_and_blocks_direct_retry(
    rollback_failure_state: dict[str, Any], monkeypatch: pytest.MonkeyPatch, mock_llm_guidance: AsyncMock, revision_database: Neo4jManagerSingleton,
) -> None:
    database = FailingRollbackDatabase(revision_database, monkeypatch)
    driver = cast(DriverExample, revision_database.driver)
    graph_before = deepcopy((driver.nodes, driver.edges, driver.receipts, driver.writes, driver.commits))
    original = deepcopy({key: dict(value) if key.endswith("_ref") and value is not None else value for key, value in rollback_failure_state.items()})
    result = await revise_chapter(cast(NarrativeState, rollback_failure_state))

    assert mock_llm_guidance.async_call_llm.call_count == 0
    assert rollback_failure_state == original
    assert result == {
        "revision_rollback_failure": {
            "chapter_number": 1,
            "iteration_count": 0,
            "error": "Revision rollback failed; reconciliation required: rollback acknowledgement lost",
            "previous_error": "Prior validation diagnostic",
            "previous_error_node": "validate",
        },
        "last_error": "Revision rollback failed; reconciliation required: rollback acknowledgement lost",
        "has_fatal_error": True,
        "error_node": "revise",
        "current_node": "revise_blocked",
    }
    blocked = {**original, **result}
    for retry_controls in ({}, clear_error_state(), {"force_continue": True}):
        retried = await revise_chapter(cast(NarrativeState, {**blocked, **retry_controls}))
        expected = deepcopy(result)
        assert expected["revision_rollback_failure"] is not None
        expected["revision_rollback_failure"]["previous_error"] = retry_controls.get("last_error", blocked["last_error"])
        expected["revision_rollback_failure"]["previous_error_node"] = retry_controls.get("error_node", blocked["error_node"])
        assert retried == expected
    assert len(database.reads) == 4
    assert (driver.nodes, driver.edges, driver.receipts, driver.writes, driver.commits) == graph_before
    assert mock_llm_guidance.async_call_llm.call_count == 0
    database.fail = False
    recovered = await revise_chapter(cast(NarrativeState, blocked))
    assert recovered["has_fatal_error"] is False
    assert recovered["revision_rollback_failure"] is None
    assert recovered["iteration_count"] == 1
    assert driver.commits == 2
    assert driver.chapters == [{"status": "planned", "attempt_id": None}]
    assert mock_llm_guidance.async_call_llm.call_count == 1
    assert await revise_chapter(cast(NarrativeState, blocked)) == recovered
    assert driver.commits == 2
    assert mock_llm_guidance.async_call_llm.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("clear_transient_errors", [False, True])
async def test_failed_rollback_stays_blocked_across_sqlite_reopen_and_workflow_replay(
    rollback_failure_state: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_llm_guidance: AsyncMock,
    clear_transient_errors: bool,
    revision_database: Neo4jManagerSingleton,
) -> None:
    database = FailingRollbackDatabase(revision_database, monkeypatch)
    driver = cast(DriverExample, revision_database.driver)
    graph_before = deepcopy((driver.nodes, driver.edges, driver.receipts, driver.writes, driver.commits))
    project = Path(rollback_failure_state["project_dir"])
    artifact_bytes = {str(path.relative_to(project)): path.read_bytes() for path in project.rglob("*") if path.is_file()}
    configuration = {"configurable": {"thread_id": "rollback-recovery"}}
    checkpoint_path = str(tmp_path / "rollback.sqlite")
    async with create_checkpointer(checkpoint_path) as checkpointer:
        graph = create_full_workflow_graph(checkpointer)
        await graph.aupdate_state(configuration, rollback_failure_state, as_node="validate")
        assert (await graph.aget_state(configuration)).next == ("revise",)
        events = []
        async with aclosing(graph.astream(None, configuration, stream_mode="updates")) as stream:
            async for event in stream:
                events.append(event)
                assert mock_llm_guidance.async_call_llm.call_count == 0
        assert [node for event in events for node in event] == ["revise", "error_handler"]
        saved = await graph.aget_state(configuration)
        assert saved.next == ()
        assert saved.values["has_fatal_error"] is True
        assert saved.values["current_node"] == "error_handler"
        for key, value in rollback_failure_state.items():
            if key not in {"has_fatal_error", "last_error", "error_node", "current_node", "revision_rollback_failure"}:
                assert saved.values[key] == value

    database.fail = False
    async with create_checkpointer(checkpoint_path) as checkpointer:
        graph = create_full_workflow_graph(checkpointer)
        restored = await graph.aget_state(configuration)
        assert restored.values == saved.values
        assert [event async for event in graph.astream(None, configuration, stream_mode="updates")] == []
        replay = dict(restored.values)
        if clear_transient_errors:
            replay.update(clear_error_state())
        replay["force_continue"] = True
        events = [event async for event in graph.astream(replay, configuration, stream_mode="updates")]
        assert [node for event in events for node in event] == ["route", "error_handler"]
        final = (await graph.aget_state(configuration)).values
        assert final["revision_rollback_failure"] == saved.values["revision_rollback_failure"]
        assert final["has_fatal_error"] is True
        assert final["last_error"] == saved.values["last_error"]
        for key, value in rollback_failure_state.items():
            if key not in {"has_fatal_error", "last_error", "error_node", "current_node", "force_continue", "revision_rollback_failure"}:
                assert final[key] == value

    assert len(database.reads) == 1
    assert (driver.nodes, driver.edges, driver.receipts, driver.writes, driver.commits) == graph_before
    assert mock_llm_guidance.async_call_llm.call_count == 0
    lifecycle = ChapterLifecycle(cast(NarrativeState, rollback_failure_state)).stage()
    expected_marker = canonical_bytes({"schema_version": 1, "attempt_id": lifecycle.manifest.attempt_id, "phase": "compensation_required"})
    assert {str(path.relative_to(project)): path.read_bytes() for path in project.rglob("*") if path.is_file()} == {
        **artifact_bytes, lifecycle.phase_path("compensation_required"): expected_marker,
    }
