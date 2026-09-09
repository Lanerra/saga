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

from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.revision_node import revise_chapter
from core.langgraph.state import Contradiction, NarrativeState, create_initial_state
from core.langgraph.state_helpers import clear_error_state
from core.langgraph.workflow import create_checkpointer, create_full_workflow_graph
from core.service_context import get_services
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager
from tests.fakes.service_context import patch_service


@pytest.fixture(autouse=True)
def revision_database(monkeypatch: pytest.MonkeyPatch) -> FakeNeo4jManager:
    database = FakeNeo4jManager()
    monkeypatch.setattr(get_services(), 'database', database)
    return database


@pytest.fixture
def sample_revision_state(tmp_path: Path) -> NarrativeState:
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

    state["scene_drafts_ref"] = {"path": "mock_scene_drafts.json"}
    state["scene_embeddings_ref"] = {"path": "mock_scene_embeddings.json"}
    state["draft_ref"] = {"path": "mock_draft.txt"}
    state["embedding_ref"] = {"path": "mock_embedding.json"}
    state["extracted_entities_ref"] = {"path": "mock_entities.json"}
    state["extracted_relationships_ref"] = {"path": "mock_relationships.json"}
    state["generated_embedding"] = [0.1, 0.2]

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
    def __init__(self) -> None:
        self.batches: list[list[tuple[str, dict[str, Any]]]] = []
        self.fail = True

    async def execute_cypher_batch(self, queries: list[tuple[str, dict[str, Any]]]) -> None:
        self.batches.append(deepcopy(queries))
        if self.fail:
            raise RuntimeError("rollback acknowledgement lost")


@pytest.fixture
def rollback_failure_state(sample_revision_state: dict[str, Any]) -> dict[str, Any]:
    assert Path(inspect.getfile(revise_chapter)).resolve() == Path(__file__).resolve().parents[2] / "core/langgraph/nodes/revision_node.py"
    state = sample_revision_state
    manager = ContentManager(state["project_dir"])
    for key in ("scene_drafts_ref", "scene_embeddings_ref", "embedding_ref", "extracted_entities_ref", "extracted_relationships_ref"):
        state[key] = manager.save_json({"rejected_attempt": key}, key.removesuffix("_ref"), "chapter_1", 1)
    state["draft_ref"] = manager.save_text("Rejected draft evidence", "draft", "chapter_1", 1)
    state["revision_guidance_ref"] = manager.save_text("Prior guidance evidence", "revision_guidance", "chapter_1", 1)
    state["last_error"] = "Prior validation diagnostic"
    state["error_node"] = "validate"
    state["initialization_complete"] = True
    return state


@pytest.mark.asyncio
async def test_failed_rollback_preserves_evidence_and_blocks_direct_retry(
    rollback_failure_state: dict[str, Any], monkeypatch: pytest.MonkeyPatch, mock_llm_guidance: AsyncMock
) -> None:
    database = FailingRollbackDatabase()
    monkeypatch.setattr(get_services(), 'database', database)
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
    database.fail = False
    for retry_controls in ({}, clear_error_state(), {"force_continue": True, "iteration_count": 3}):
        retried = await revise_chapter(cast(NarrativeState, {**blocked, **retry_controls}))
        assert retried == {
            "last_error": result["last_error"],
            "has_fatal_error": True,
            "error_node": "revise",
            "current_node": "revise_blocked",
        }
    assert len(database.batches) == 1
    assert [parameters for _, parameters in database.batches[0]] == [{"chapter": 1}] * 4
    assert mock_llm_guidance.async_call_llm.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("clear_transient_errors", [False, True])
async def test_failed_rollback_stays_blocked_across_sqlite_reopen_and_workflow_replay(
    rollback_failure_state: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_llm_guidance: AsyncMock,
    clear_transient_errors: bool,
) -> None:
    database = FailingRollbackDatabase()
    monkeypatch.setattr(get_services(), 'database', database)
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

    assert len(database.batches) == 1
    assert mock_llm_guidance.async_call_llm.call_count == 0
    assert {str(path.relative_to(project)): path.read_bytes() for path in project.rglob("*") if path.is_file()} == artifact_bytes
