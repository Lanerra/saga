"""Phase admission and deprecated state compatibility through real SQLite."""
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from core.exceptions import CheckpointResumeConflictError
from core.graph_ownership import load_graph_project_id
from core.langgraph import initialization
from core.langgraph.content_manager import ContentManager
from core.langgraph.state import NarrativeState, create_initial_state
from core.langgraph.workflow import create_checkpointer, create_full_workflow_graph
from data_access.chapter_queries import ChapterProgress
from orchestration.langgraph_orchestrator import LangGraphOrchestrator


def synthetic_state(directory: Path) -> NarrativeState:
    return {
        **create_initial_state(
            project_id="synthetic", project_dir=str(directory), title="A Key",
            genre="Mystery", theme="Trust", setting="Archive", protagonist_name="Mara",
            narrative_style="First person present", target_word_count=101, total_chapters=3,
        ),
        "lifecycle_version": 1, "graph_project_id": load_graph_project_id(directory), "attempt_id": None,
    }


@pytest.mark.parametrize("field,value", [("target_word_count", 0), ("total_chapters", True), ("narrative_style", "")])
def test_fresh_author_policy_rejects_invalid_values(tmp_path: Path, field: str, value: Any) -> None:
    arguments: dict[str, Any] = dict(
        project_id="synthetic", project_dir=str(tmp_path), title="A Key", genre="Mystery",
        theme="Trust", setting="Archive", protagonist_name="Mara", narrative_style="First person",
        target_word_count=101, total_chapters=3,
    )
    arguments[field] = value
    with pytest.raises(ValueError, match=field):
        create_initial_state(**arguments)


def test_fresh_state_does_not_seed_unproduced_telemetry(tmp_path: Path) -> None:
    state = synthetic_state(tmp_path)
    assert set(state).intersection({"generated_embedding", "qa_results", "qa_history", "healing_history", "coherence_score", "nodes_enriched"}) == set()
    assert (state["project_id"], state["medium_model"]) == ("synthetic", state["revision_model"])


@pytest.mark.parametrize("legacy", [[1.0, 2.0], [], {}, False, {"path": "old-vector.json"}])
async def test_checkpoint_rejects_legacy_embedding_without_rewriting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: Any) -> None:
    from data_access import chapter_queries

    async def progress() -> ChapterProgress:
        return ChapterProgress(0, ())

    monkeypatch.setattr(chapter_queries, "load_chapter_progress_from_db", progress)
    state = synthetic_state(tmp_path)
    state["generated_embedding"] = legacy
    configuration = {"configurable": {"thread_id": "saga_synthetic"}}
    path = str(tmp_path / "checkpoints/saga.db")
    async with create_checkpointer(path) as saver:
        graph = create_full_workflow_graph(saver)
        await graph.aupdate_state(configuration, state, as_node="route")
    async with create_checkpointer(path) as saver:
        graph = create_full_workflow_graph(saver)
        before = await saver.aget_tuple(configuration)
        orchestrator = LangGraphOrchestrator(project_dir=tmp_path)
        with pytest.raises(CheckpointResumeConflictError, match="generated_embedding"):
            await orchestrator._load_state_for_run(graph=graph, requested_project_id="synthetic", thread_id="saga_synthetic", narrative_config=None)
        assert await saver.aget_tuple(configuration) == before
        assert (await graph.aget_state(configuration)).values["generated_embedding"] == legacy


@pytest.mark.parametrize("initialization_complete,missing", [(False, "large_model"), (True, "narrative_model"), (True, "narrative_style"), (True, "target_word_count")])
async def test_checkpoint_requires_its_phase_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, initialization_complete: bool, missing: str) -> None:
    from data_access import chapter_queries

    async def progress() -> ChapterProgress:
        return ChapterProgress(0, ())

    monkeypatch.setattr(chapter_queries, "load_chapter_progress_from_db", progress)
    state = synthetic_state(tmp_path)
    state["initialization_complete"] = initialization_complete
    cast(dict[str, Any], state).pop(missing)
    configuration = {"configurable": {"thread_id": "saga_synthetic"}}
    async with create_checkpointer(str(tmp_path / "checkpoints/saga.db")) as saver:
        graph = create_full_workflow_graph(saver)
        await graph.aupdate_state(configuration, state, as_node="init_complete" if initialization_complete else "init_character_sheets")
        before = await saver.aget_tuple(configuration)
        with pytest.raises(CheckpointResumeConflictError, match=missing):
            await LangGraphOrchestrator(project_dir=tmp_path)._load_state_for_run(
                graph=graph, requested_project_id="synthetic", thread_id="saga_synthetic", narrative_config=None,
            )
        assert await saver.aget_tuple(configuration) == before


@pytest.mark.parametrize("legacy_null", [False, True])
async def test_checkpoint_continues_with_retained_policy_and_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_null: bool) -> None:
    from data_access import chapter_queries

    calls: list[str] = []

    async def progress() -> ChapterProgress:
        return ChapterProgress(0, ())

    async def outline(state: NarrativeState) -> NarrativeState:
        calls.append("init_global_outline")
        assert (state["narrative_style"], state["target_word_count"], state["medium_model"]) == ("First person present", 101, "retained-medium")
        return {"initialization_step": "global_outline"}

    monkeypatch.setattr(chapter_queries, "load_chapter_progress_from_db", progress)
    monkeypatch.setattr(initialization, "generate_global_outline", outline)
    state = synthetic_state(tmp_path)
    state["medium_model"] = "retained-medium"
    state["qa_history"] = [{"chapter": 1, "issues_found": 2}]
    state["healing_history"] = [{"chapter": 1, "nodes_enriched": 2}]
    if legacy_null:
        state["generated_embedding"] = None
    manager = ContentManager(str(tmp_path))
    state["character_sheets_ref"] = manager.save_json({"Mara": "Synthetic"}, "characters", "initialization")
    expected = deepcopy(state)
    configuration = {"configurable": {"thread_id": "saga_synthetic"}}
    path = str(tmp_path / "checkpoints/saga.db")
    async with create_checkpointer(path) as saver:
        graph = create_full_workflow_graph(saver)
        await graph.aupdate_state(configuration, state, as_node="init_character_sheets")
    async with create_checkpointer(path) as saver:
        graph = create_full_workflow_graph(saver)
        graph.interrupt_after_nodes = ["init_global_outline"]
        before = await saver.aget_tuple(configuration)
        orchestrator = LangGraphOrchestrator(project_dir=tmp_path)
        loaded = await orchestrator._load_state_for_run(graph=graph, requested_project_id="synthetic", thread_id="saga_synthetic", narrative_config=None)
        assert loaded == expected
        assert await saver.aget_tuple(configuration) == before
        await orchestrator._run_chapter_generation_loop(graph, loaded)
        snapshot = await graph.aget_state(configuration)
        assert calls == ["init_global_outline"]
        assert snapshot.next == ("init_act_outlines",)
        assert snapshot.values == {**expected, "initialization_step": "global_outline"}
        assert manager.load_json(snapshot.values["character_sheets_ref"]) == {"Mara": "Synthetic"}


async def test_initialization_handoff_requires_generation_policy(tmp_path: Path) -> None:
    state = synthetic_state(tmp_path)
    del state["narrative_model"]
    configuration = {"configurable": {"thread_id": "saga_synthetic"}}
    async with create_checkpointer(str(tmp_path / "checkpoints/saga.db")) as saver:
        graph = create_full_workflow_graph(saver)
        graph.interrupt_after_nodes = ["init_complete"]
        await graph.aupdate_state(configuration, state, as_node="init_run_parsers")
        with pytest.raises(ValueError, match="narrative_model"):
            await graph.ainvoke(None, configuration)
        snapshot = await graph.aget_state(configuration)
        assert snapshot.values["initialization_complete"] is False
        assert snapshot.next == ("init_complete",)


async def test_generation_reopens_identified_artifact_and_drafts_with_retained_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from langgraph.graph import END, StateGraph  # type: ignore[attr-defined]

    import config
    from core.langgraph.content_manager import get_scene_drafts, load_embedding, save_embedding
    from core.langgraph.nodes.scene_generation_node import draft_scene
    from core.project_config import NarrativeProjectConfig
    from core.service_context import get_services
    from data_access import chapter_queries

    prompts: list[tuple[str, str]] = []

    async def provider(model_name: str, prompt: str, **options: Any) -> tuple[str, dict[str, Any]]:
        prompts.append((model_name, prompt))
        return "I open the archive.", {}

    async def progress() -> ChapterProgress:
        return ChapterProgress(0, ())

    def prepare(state: NarrativeState) -> NarrativeState:
        raise AssertionError("Native continuation must not replay prepare")

    def workflow(saver: Any) -> Any:
        builder = StateGraph(NarrativeState)
        builder.add_node("prepare", prepare)
        builder.add_node("draft_scene", draft_scene)
        builder.set_entry_point("prepare")
        builder.add_edge("prepare", "draft_scene")
        builder.add_edge("draft_scene", END)
        return builder.compile(checkpointer=saver)

    monkeypatch.setattr(chapter_queries, "load_chapter_progress_from_db", progress)
    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    manager = ContentManager(str(tmp_path))
    state = synthetic_state(tmp_path)
    state.update({"initialization_complete": True, "narrative_model": "retained-narrative", "chapter_plan_scene_count": 1})
    state["chapter_plan_ref"] = manager.save_json([{
        "title": "The archive", "pov_character": "Mara", "setting": "Archive", "characters": ["Mara"],
        "plot_point": "Read ledger", "conflict": "Locked door", "outcome": "Mara reads the ledger", "beats": ["Mara opens the archive"],
    }], "chapter_plan", "chapter_1", 1)
    vector = [1.0] * config.EXPECTED_EMBEDDING_DIM
    reference = save_embedding(manager, vector, 1, embedding_model=config.EMBEDDING_MODEL)
    state["embedding_ref"] = reference
    artifact = manager.load_json(reference)
    monkeypatch.setattr(config, "DEFAULT_NARRATIVE_STYLE", "Current global style must not replace retained style")
    different_policy = NarrativeProjectConfig(
        title="Different", genre="Mystery", theme="Trust", setting="Archive", protagonist_name="Mara",
        total_chapters=3, target_word_count=999, narrative_style="Different requested style",
    )
    configuration = {"configurable": {"thread_id": "saga_synthetic"}}
    path = str(tmp_path / "checkpoints/saga.db")
    async with create_checkpointer(path) as saver:
        await workflow(saver).aupdate_state(configuration, state, as_node="prepare")
    async with create_checkpointer(path) as saver:
        graph = workflow(saver)
        orchestrator = LangGraphOrchestrator(project_dir=tmp_path)
        before = await saver.aget_tuple(configuration)
        loaded = await orchestrator._load_state_for_run(
            graph=graph, requested_project_id="synthetic", thread_id="saga_synthetic", narrative_config=different_policy,
        )
        assert loaded == state
        assert await saver.aget_tuple(configuration) == before
        await orchestrator._run_chapter_generation_loop(graph, loaded)
        final = await graph.aget_state(configuration)
        assert final.next == ()
        assert final.values["current_scene_index"] == 1
        assert (final.values["narrative_style"], final.values["target_word_count"], final.values["run_start_chapter"]) == ("First person present", 101, 1)
        assert get_scene_drafts(final.values, manager) == ["I open the archive."]
        assert manager.load_json(final.values["embedding_ref"]) == artifact
        assert load_embedding(manager, final.values["embedding_ref"]) == vector
        assert len(prompts) == 1
        assert prompts[0][0] == "retained-narrative"
        assert prompts[0][1].split("Narrative Style & Voice:\n")[1].split("\n\n")[0] == "First person present"


def test_imports_resolve_to_candidate() -> None:
    import inspect

    root = Path(__file__).resolve().parents[1]
    assert Path(inspect.getfile(create_initial_state)).resolve() == root / "core/langgraph/state.py"
    assert Path(inspect.getfile(create_full_workflow_graph)).resolve() == root / "core/langgraph/workflow.py"
    assert Path(inspect.getfile(LangGraphOrchestrator)).resolve() == root / "orchestration/langgraph_orchestrator.py"
