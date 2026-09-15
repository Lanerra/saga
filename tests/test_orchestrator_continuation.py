"""Native SQLite state loading and pending initialization continuation."""

from pathlib import Path
from typing import Any

import pytest
from langgraph.graph import END, StateGraph  # type: ignore[attr-defined]
from langgraph.types import interrupt

from core.exceptions import WorkflowExecutionError
from core.graph_ownership import load_graph_project_id
from core.langgraph import initialization
from core.langgraph.content_manager import ContentManager
from core.langgraph.state import NarrativeState, create_initial_state
from core.langgraph.workflow import create_checkpointer, create_full_workflow_graph
from data_access import chapter_queries
from data_access.chapter_queries import ChapterProgress
from orchestration.langgraph_orchestrator import LangGraphOrchestrator


@pytest.mark.parametrize("successful_pending_write", [False, True])
async def test_loader_preserves_pending_initialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, successful_pending_write: bool) -> None:
    calls: list[str] = []
    manager = ContentManager(str(tmp_path))
    character_update: NarrativeState = {}

    async def characters(state: NarrativeState) -> NarrativeState:
        calls.append("init_character_sheets")
        return character_update

    async def outline(state: NarrativeState) -> NarrativeState:
        calls.append("init_global_outline")
        return {"initialization_step": "global_outline"}

    async def progress() -> ChapterProgress:
        return ChapterProgress(0, ())

    monkeypatch.setattr(initialization, "generate_character_sheets", characters)
    monkeypatch.setattr(initialization, "generate_global_outline", outline)
    monkeypatch.setattr(chapter_queries, "load_chapter_progress_from_db", progress)
    state: NarrativeState = {
        **create_initial_state(
            project_id="synthetic", project_dir=str(tmp_path), title="Synthetic",
            genre="Mystery", theme="Trust", setting="Archive", protagonist_name="Mara",
            target_word_count=101, total_chapters=2,
        ),
        "project_id": "synthetic", "project_dir": str(tmp_path), "graph_project_id": load_graph_project_id(tmp_path),
        "lifecycle_version": 1, "current_chapter": 1, "total_chapters": 2, "run_start_chapter": 1,
        "initialization_complete": False,
    }
    configuration: dict[str, Any] = {"configurable": {"thread_id": "saga_synthetic"}}
    checkpoint_path = str(tmp_path / "checkpoints/saga.db")
    async with create_checkpointer(checkpoint_path) as saver:
        graph = create_full_workflow_graph(saver)
        # Seed the real scheduled task without rerunning the filesystem admission guard.
        await graph.aupdate_state(configuration, state, as_node="route")
        snapshot = await graph.aget_state(configuration)
        assert snapshot.next == ("init_character_sheets",)
        reference = manager.save_json({"Traveler": "Synthetic character"}, "characters", "initialization")
        character_update = {"character_sheets_ref": reference, "initialization_step": "character_sheets"}
        if successful_pending_write:
            task, = snapshot.tasks
            writes = [*character_update.items(), ("branch:to:init_global_outline", None)]
            await saver.aput_writes(snapshot.config, writes, task.id)

    async with create_checkpointer(checkpoint_path) as saver:
        graph = create_full_workflow_graph(saver)
        graph.interrupt_after_nodes = ["init_global_outline"]
        raw = await saver.aget(configuration)
        effective = await graph.aget_state(configuration)
        assert raw is not None
        assert raw["channel_values"].get("character_sheets_ref") is None
        orchestrator = LangGraphOrchestrator(project_dir=tmp_path)
        loaded = await orchestrator._load_state_for_run(graph=graph, requested_project_id="synthetic", thread_id="saga_synthetic", narrative_config=None)
        assert loaded == effective.values
        assert loaded.get("character_sheets_ref") == (reference if successful_pending_write else None)
        with pytest.raises(WorkflowExecutionError) as caught:
            await orchestrator._run_chapter_generation_loop(graph, loaded)
        assert caught.value.details["outcome"] == "interrupted"
        final = await graph.aget_state(configuration)
        assert calls == (["init_global_outline"] if successful_pending_write else ["init_character_sheets", "init_global_outline"])
        assert final.next == ("init_act_outlines",)
        assert final.values["initialization_step"] == "global_outline"
        assert final.values["run_start_chapter"] == 1
        assert manager.load_json(reference) == {"Traveler": "Synthetic character"}


async def test_dynamic_interrupt_never_becomes_completed_resume(tmp_path: Path) -> None:
    def pause(state: NarrativeState) -> NarrativeState:
        interrupt("Synthetic operator input required")
        raise AssertionError("Interrupted work must not reach generation or publication")

    workflow = StateGraph(NarrativeState)
    workflow.add_node("finalize", lambda state: {"current_node": "finalize"})
    workflow.add_node("pause", pause)
    workflow.set_entry_point("finalize")
    workflow.add_edge("finalize", "pause")
    workflow.add_edge("pause", END)
    state: NarrativeState = {
        "project_id": "synthetic", "project_dir": str(tmp_path), "graph_project_id": load_graph_project_id(tmp_path),
        "lifecycle_version": 1, "current_chapter": 1, "total_chapters": 1,
    }
    configuration = {"configurable": {"thread_id": "saga_synthetic"}}
    for resume in (False, True):
        async with create_checkpointer(str(tmp_path / "saga.db")) as saver:
            graph = workflow.compile(checkpointer=saver)
            orchestrator = LangGraphOrchestrator(project_dir=tmp_path)
            orchestrator._resume_checkpoint = resume
            if resume:
                state = (await graph.aget_state(configuration)).values
            with pytest.raises(WorkflowExecutionError) as caught:
                await orchestrator._run_chapter_generation_loop(graph, state)
            assert caught.value.details["outcome"] == "interrupted"
            snapshot = await graph.aget_state(configuration)
            assert snapshot.next == ("pause",)
            assert snapshot.values["current_node"] == "finalize"
            assert len(snapshot.interrupts) == 1
            assert snapshot.interrupts[0].value == "Synthetic operator input required"
    assert not (tmp_path / "chapters").exists()
