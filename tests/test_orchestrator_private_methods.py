# tests/test_orchestrator_private_methods.py
from unittest.mock import AsyncMock

import pytest
import utils
from chapter_generation.drafting_service import DraftResult
from chapter_generation.prerequisites_service import PrerequisiteData
from data_access import character_queries, world_queries
from initialization.models import PlotOutline
from orchestration.nana_orchestrator import NANA_Orchestrator, RevisionOutcome

from models.agent_models import ChapterEndState
from models.user_input_models import (
    KeyLocationModel,
    NovelConceptModel,
    ProtagonistModel,
    SettingModel,
    UserStoryInputModel,
)


@pytest.fixture
def orchestrator(monkeypatch):
    monkeypatch.setattr(utils, "load_spacy_model_if_needed", lambda: None)
    orch = NANA_Orchestrator()
    monkeypatch.setattr(orch, "_update_rich_display", lambda *a, **k: None)
    monkeypatch.setattr(orch, "refresh_plot_outline", AsyncMock())
    monkeypatch.setattr(orch, "refresh_knowledge_cache", AsyncMock())
    monkeypatch.setattr(
        orch.context_service,
        "build_hybrid_context",
        AsyncMock(return_value="ctx"),
    )
    return orch


@pytest.mark.asyncio
async def test_validate_plot_outline_missing(orchestrator):
    orchestrator.plot_outline = PlotOutline()
    assert not await orchestrator._validate_plot_outline(1)


@pytest.mark.asyncio
async def test_process_prereq_result_failure(orchestrator):
    result = await orchestrator._process_prereq_result(
        1, PrerequisiteData(None, -1, None, None)
    )
    assert result is None


@pytest.mark.asyncio
async def test_process_initial_draft_failure(orchestrator):
    result = await orchestrator._process_initial_draft(1, DraftResult(None, None))
    assert result is None


@pytest.mark.asyncio
async def test_process_revision_result_failure(orchestrator):
    result = await orchestrator._process_revision_result(
        1, RevisionOutcome(None, None, False)
    )
    assert result is None


@pytest.mark.asyncio
async def test_finalize_and_log_success(orchestrator, monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "_finalize_and_save_chapter",
        AsyncMock(return_value="final"),
    )
    result = await orchestrator._finalize_and_log(1, "text", None, False)
    assert result == "final"


@pytest.mark.asyncio
async def test_finalize_and_log_failure(orchestrator, monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "_finalize_and_save_chapter",
        AsyncMock(return_value=None),
    )
    result = await orchestrator._finalize_and_log(1, "text", None, True)
    assert result is None


def test_load_state_from_user_model(orchestrator):
    model = UserStoryInputModel(
        novel_concept=NovelConceptModel(title="My Tale"),
        protagonist=ProtagonistModel(name="Hero"),
        setting=SettingModel(key_locations=[KeyLocationModel(name="Town")]),
    )
    orchestrator.load_state_from_user_model(model)
    assert orchestrator.plot_outline.get("title") == "My Tale"
    assert orchestrator.plot_outline.get("protagonist_name") == "Hero"


@pytest.mark.asyncio
async def test_prepare_prerequisites_uses_plan(orchestrator, monkeypatch):
    orchestrator.plot_outline = PlotOutline(plot_points=["Intro"])
    orchestrator.next_chapter_context = "prefetched"
    monkeypatch.setattr(orchestrator, "_update_novel_props_cache", lambda: None)
    monkeypatch.setattr(
        orchestrator.pre_flight_agent,
        "perform_core_checks",
        AsyncMock(),
    )

    async def fake_plan(*_args, **_kwargs):
        return ([{"scene_number": 1}], {"total_tokens": 1})

    monkeypatch.setattr(
        orchestrator.planner_agent,
        "plan_chapter_scenes",
        AsyncMock(side_effect=fake_plan),
    )
    monkeypatch.setattr(
        character_queries,
        "get_character_profiles_from_db",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        world_queries,
        "get_world_building_from_db",
        AsyncMock(return_value={}),
    )
    await orchestrator.refresh_knowledge_cache()
    monkeypatch.setattr(
        character_queries,
        "get_character_profiles_from_db",
        AsyncMock(side_effect=Exception("cache not used")),
    )
    monkeypatch.setattr(
        world_queries,
        "get_world_building_from_db",
        AsyncMock(side_effect=Exception("cache not used")),
    )
    monkeypatch.setattr(
        orchestrator.evaluator_agent,
        "check_scene_plan_consistency",
        AsyncMock(return_value=([], {})),
    )
    build_ctx_mock = AsyncMock(side_effect=Exception("should not call"))
    monkeypatch.setattr(
        orchestrator.context_service,
        "build_hybrid_context",
        build_ctx_mock,
    )

    result = await orchestrator._prepare_chapter_prerequisites(1)
    assert result == PrerequisiteData(
        plot_point_focus="Intro",
        plot_point_index=0,
        chapter_plan=[{"scene_number": 1}],
        hybrid_context_for_draft="prefetched",
    )
    assert orchestrator.next_chapter_context is None
    build_ctx_mock.assert_not_called()


@pytest.mark.asyncio
async def test_prepare_prerequisites_runs_preflight(orchestrator, monkeypatch):
    orchestrator.plot_outline = PlotOutline(
        plot_points=["Intro"], protagonist_name="Hero"
    )
    orchestrator.next_chapter_context = "prefetched"
    monkeypatch.setattr(orchestrator, "_update_novel_props_cache", lambda: None)
    monkeypatch.setattr(
        orchestrator.planner_agent,
        "plan_chapter_scenes",
        AsyncMock(return_value=([], {})),
    )
    monkeypatch.setattr(
        character_queries, "get_character_profiles_from_db", AsyncMock(return_value={})
    )
    monkeypatch.setattr(
        world_queries, "get_world_building_from_db", AsyncMock(return_value={})
    )
    await orchestrator.refresh_knowledge_cache()
    preflight_mock = AsyncMock()
    monkeypatch.setattr(
        orchestrator.pre_flight_agent,
        "perform_core_checks",
        preflight_mock,
    )
    monkeypatch.setattr(
        orchestrator.evaluator_agent,
        "check_scene_plan_consistency",
        AsyncMock(return_value=([], {})),
    )
    await orchestrator._prepare_chapter_prerequisites(1)
    preflight_mock.assert_awaited_with(
        orchestrator.plot_outline,
        orchestrator.knowledge_cache.characters,
        orchestrator.knowledge_cache.world,
    )


@pytest.mark.asyncio
async def test_perform_initial_setup_sets_next_context(monkeypatch, orchestrator):
    plot_outline = PlotOutline(title="T", plot_points=["p"], protagonist_name="Hero")
    monkeypatch.setattr(
        "orchestration.nana_orchestrator.run_genesis_phase",
        AsyncMock(return_value=(plot_outline, {}, {"source": "w"}, {})),
    )
    monkeypatch.setattr(orchestrator, "_accumulate_tokens", lambda *_a, **_k: None)
    monkeypatch.setattr(orchestrator, "_update_novel_props_cache", lambda: None)
    ctx_mock = AsyncMock(return_value="ctx0")
    monkeypatch.setattr(orchestrator.context_service, "build_hybrid_context", ctx_mock)
    monkeypatch.setattr(
        "orchestration.nana_orchestrator.neo4j_manager.driver", None, raising=False
    )

    result = await orchestrator.perform_initial_setup()

    assert result is True
    ctx_mock.assert_awaited_once_with(orchestrator, 1, None, None)
    assert orchestrator.next_chapter_context == "ctx0"


@pytest.mark.asyncio
async def test_perform_initial_setup_loads_ch0_state(monkeypatch, orchestrator):
    plot_outline = PlotOutline(title="T", plot_points=["p"], protagonist_name="Hero")
    monkeypatch.setattr(
        "orchestration.nana_orchestrator.run_genesis_phase",
        AsyncMock(return_value=(plot_outline, {}, {"source": "w"}, {})),
    )
    monkeypatch.setattr(orchestrator, "_accumulate_tokens", lambda *_a, **_k: None)
    monkeypatch.setattr(orchestrator, "_update_novel_props_cache", lambda: None)
    ctx_mock = AsyncMock(return_value="ctx0")
    monkeypatch.setattr(orchestrator.context_service, "build_hybrid_context", ctx_mock)
    monkeypatch.setattr(
        "orchestration.nana_orchestrator.neo4j_manager.driver", None, raising=False
    )

    ch0_data = {"end_state_json": "{}"}
    get_mock = AsyncMock(return_value=ch0_data)
    monkeypatch.setattr(
        "data_access.chapter_queries.get_chapter_data_from_db", get_mock
    )
    monkeypatch.setattr(
        "models.agent_models.ChapterEndState.model_validate_json",
        lambda *_a, **_k: ChapterEndState(
            chapter_number=0,
            character_states=[],
            unresolved_cliffhanger=None,
            key_world_changes={},
        ),
    )

    result = await orchestrator.perform_initial_setup()

    assert result is True
    get_mock.assert_awaited_once_with(0)
    ctx_mock.assert_awaited_once_with(
        orchestrator,
        1,
        None,
        {"chapter_zero_end_state": orchestrator.chapter_zero_end_state},
    )


@pytest.mark.asyncio
async def test_finalize_and_save_chapter_prefetches_context(orchestrator, monkeypatch):
    ctx_mock = AsyncMock(return_value="ctx1")
    monkeypatch.setattr(orchestrator.context_service, "build_hybrid_context", ctx_mock)
    monkeypatch.setattr(orchestrator, "refresh_plot_outline", AsyncMock())
    monkeypatch.setattr(
        "orchestration.nana_orchestrator.neo4j_manager.driver", None, raising=False
    )
    monkeypatch.setattr(orchestrator, "refresh_knowledge_cache", AsyncMock())
    monkeypatch.setattr(
        orchestrator,
        "_finalize_and_save_chapter",
        AsyncMock(return_value="text"),
    )

    result = await orchestrator._finalize_and_log(1, "text", None, False)

    assert result == "text"
    ctx_mock.assert_awaited_with(orchestrator, 2, None, None)
    assert orchestrator.next_chapter_context == "ctx1"
