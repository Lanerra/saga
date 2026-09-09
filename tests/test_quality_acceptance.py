"""Quality gates exercise production publication with disposable retained content."""
from pathlib import Path
from typing import Any

import pytest

from core.langgraph.chapter_lifecycle import ChapterLifecycle
from core.langgraph.nodes.commit_node import commit_to_graph
from core.langgraph.subgraphs.validation import _parse_quality_scores
from core.service_context import get_services
from tests.test_langgraph.test_chapter_lifecycle import DriverExample, lifecycle_example

__all__ = ["lifecycle_example"]


@pytest.mark.parametrize("response", ["not an evaluation", '{"coherence_score": 0.9}', '{"coherence_score": true}'])
def test_malformed_evaluation_cannot_invent_scores(response: str) -> None:
    with pytest.raises(ValueError):
        _parse_quality_scores(response)


async def test_unvalidated_attempt_cannot_publish(lifecycle_example: tuple[Any, DriverExample]) -> None:
    state, driver = lifecycle_example
    state["quality_checks"] = {}
    state.update(await commit_to_graph(state))
    lifecycle = ChapterLifecycle(state).stage()
    with pytest.raises(ValueError, match="[Qq]uality|validation"):
        await lifecycle.publish()
    assert driver.commits == 1
    assert not lifecycle.files.exists("chapters/chapter_001.accepted.json")
    assert not lifecycle.files.exists("chapters/chapter_001.md")


async def test_legacy_force_does_not_bypass_missing_checks(tmp_path: Path) -> None:
    from core.langgraph.content_manager import ContentManager
    from core.langgraph.nodes.finalize_node import finalize_chapter
    from core.langgraph.state import NarrativeState

    manager = ContentManager(str(tmp_path))
    state: NarrativeState = {"project_dir": str(tmp_path), "current_chapter": 1, "force_continue": True,
             "draft_ref": manager.save_text("Synthetic prose.", "draft", "chapter_1")}
    result = await finalize_chapter(state)
    assert result["has_fatal_error"] is True
    assert not (tmp_path / "chapters/chapter_001.md").exists()


async def evaluate_example(state: Any, monkeypatch: pytest.MonkeyPatch, response: str) -> Any:
    import config
    from core.db_manager import neo4j_manager
    from core.langgraph.subgraphs.validation import create_validation_subgraph
    from data_access.validation_queries import PRIOR_ACCEPTED_ATTEMPTS_QUERY

    state = {**state, "quality_checks": {}, "quality_feedback": None}
    monkeypatch.setitem(vars(config), "settings", config.settings.model_copy(update={"PLOT_STAGNATION_MIN_WORD_COUNT": 0, "PLOT_STAGNATION_MIN_ENTITIES": 0}))
    original_read = neo4j_manager.execute_read_query

    async def read(query: str, parameters: Any = None) -> Any:
        if query == PRIOR_ACCEPTED_ATTEMPTS_QUERY:
            return []
        return await original_read(query, parameters)

    async def evaluate(**arguments: Any) -> Any:
        return response, {}

    monkeypatch.setattr(neo4j_manager, "execute_read_query", read)
    monkeypatch.setattr(get_services().language_model, "async_call_llm", evaluate)
    return await create_validation_subgraph().ainvoke(state)


@pytest.mark.parametrize("policy,force,maximum,score,allowed", [
    ("strict", True, 0, 0.1, False), ("strict", False, 0, 0.1, False),
    ("author", True, 3, 0.1, True), ("author", False, 0, 0.1, True),
    ("author", False, 3, 0.1, False), ("strict", False, 3, 0.9, True),
])
async def test_policy_findings_are_durable(lifecycle_example: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, policy: str, force: bool, maximum: int, score: float, allowed: bool) -> None:
    import json

    from core.langgraph.quality_policy import SCORE_FIELDS

    state, driver = lifecycle_example
    state["quality_policy"]["identity"] = policy
    state.update(force_continue=force, max_iterations=maximum)
    state.update(await commit_to_graph(state))
    state = await evaluate_example(state, monkeypatch, json.dumps({**dict.fromkeys(SCORE_FIELDS, score), "feedback": "Synthetic evaluation"}))
    lifecycle = ChapterLifecycle(state).stage()
    if not allowed:
        with pytest.raises(ValueError, match="quality gate"):
            await lifecycle.publish()
        assert driver.commits == 1
        assert not lifecycle.files.exists("chapters/chapter_001.md")
        return
    await lifecycle.publish()
    receipt = json.loads(lifecycle.files.read_bytes(lifecycle.phase_path("acceptance")))
    decision = receipt["quality"]
    assert decision["scores"] == dict.fromkeys(SCORE_FIELDS, score)
    assert decision["feedback"] == "Synthetic evaluation"
    assert decision["policy"]["identity"] == policy
    assert decision["status"] == "accepted_with_exceptions"
    assert decision["checks"]["evaluation"]["status"] == "completed"
    assert decision["graph_quality_check"]["reason"] == "cadence"
    assert "accepted_with_exceptions" in caplog.text
    assert lifecycle.phase_path("acceptance") in caplog.text
    assert json.loads(driver.receipts[lifecycle.manifest.attempt_id]["acceptance"]) == receipt
    before = lifecycle.files.read_bytes(lifecycle.phase_path("acceptance"))
    await ChapterLifecycle(state).stage().publish()
    assert lifecycle.files.read_bytes(lifecycle.phase_path("acceptance")) == before
    assert driver.commits == 2


@pytest.mark.parametrize("policy,force,allowed", [("strict", True, False), ("author", False, False), ("author", True, True)])
async def test_incomplete_evaluation_requires_explicit_author_exception(lifecycle_example: Any, monkeypatch: pytest.MonkeyPatch, policy: str, force: bool, allowed: bool) -> None:
    import json

    from core.langgraph.quality_policy import SCORE_FIELDS

    state, driver = lifecycle_example
    state["quality_policy"]["identity"] = policy
    state.update(force_continue=force, max_iterations=0)
    state.update(await commit_to_graph(state))
    state = await evaluate_example(state, monkeypatch, "malformed")
    lifecycle = ChapterLifecycle(state).stage()
    if not allowed:
        with pytest.raises(ValueError, match="incomplete"):
            await lifecycle.publish()
        assert driver.commits == 1
        return
    await lifecycle.publish()
    decision = json.loads(lifecycle.files.read_bytes(lifecycle.phase_path("acceptance")))["quality"]
    assert decision["scores"] == dict.fromkeys(SCORE_FIELDS)
    assert decision["exceptions"] == ["force_continue_incomplete_evaluation", "advisory_graph_quality:cadence"]
    assert decision["checks"]["evaluation"]["status"] == "failed"


@pytest.mark.parametrize("mandatory,outcome", [(True, "failure"), (False, "failure"), (True, "findings"), (False, "findings"), (True, "success"), (True, "disabled")])
async def test_graph_quality_gate_precedes_files(lifecycle_example: Any, monkeypatch: pytest.MonkeyPatch, mandatory: bool, outcome: str) -> None:
    import json

    from core.langgraph.nodes import quality_assurance_node

    state, driver = lifecycle_example
    state["force_continue"] = True
    state["max_iterations"] = 0
    state["quality_policy"].update(graph_quality="mandatory" if mandatory else "advisory", graph_quality_frequency=1, graph_quality_enabled=outcome != "disabled")
    async def query(pairs: Any) -> Any:
        if outcome == "failure":
            raise RuntimeError("synthetic graph query failure")
        return [{"character_name": "Example", "trait1": "brave", "trait2": "cowardly"}] if outcome == "findings" else []
    monkeypatch.setattr(quality_assurance_node, "find_contradictory_trait_characters", query)
    state.update(await commit_to_graph(state))
    lifecycle = ChapterLifecycle(state).stage()
    if mandatory and outcome != "success":
        with pytest.raises(ValueError, match="Mandatory graph quality"):
            await lifecycle.publish()
        assert not lifecycle.files.exists("chapters/chapter_001.md")
        assert driver.commits == 1
    else:
        await lifecycle.publish()
        decision = json.loads(lifecycle.files.read_bytes(lifecycle.phase_path("acceptance")))["quality"]
        assert decision["status"] == ("accepted" if outcome == "success" else "accepted_with_exceptions")
        assert decision["graph_quality_check"]["status"] == ("failed" if outcome == "failure" else "completed")


async def test_healing_failure_has_durable_advisory_receipt(lifecycle_example: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from core.langgraph.nodes.graph_healing_node import graph_healing_service, heal_graph

    state, _ = lifecycle_example
    state.update(await commit_to_graph(state))
    state.update(await ChapterLifecycle(state).stage().publish())
    async def fail(**arguments: Any) -> Any:
        raise RuntimeError("synthetic advisory failure")
    monkeypatch.setattr(graph_healing_service, "heal_graph", fail)
    result = await heal_graph(state)
    assert result.get("has_fatal_error", False) is False
    directory = Path(state["project_dir"]) / ".saga/attempts" / state["attempt_id"] / "maintenance"
    receipts = [json.loads(path.read_bytes()) for path in directory.glob("healing-*.json")]
    assert len(receipts) == 1
    assert receipts[0]["status"] == "accepted_with_exceptions"
    assert receipts[0]["outcome"]["errors"] == ["synthetic advisory failure"]


@pytest.mark.parametrize("mutation", ["findings", "policy", "fatal"])
async def test_prepared_acceptance_cannot_override_conflicting_checkpoint(lifecycle_example: Any, mutation: str) -> None:
    from core.langgraph.nodes.quality_assurance_node import assess_graph_quality
    from core.langgraph.state import Contradiction

    state, driver = lifecycle_example
    state.update(await commit_to_graph(state))
    state["graph_quality_check"] = await assess_graph_quality(state)
    lifecycle = ChapterLifecycle(state).stage()
    lifecycle._acceptance()
    if mutation == "findings":
        state["contradictions"] = [Contradiction(type="quality_issue", description="Unresolved", severity="critical", conflicting_chapters=[1])]
    elif mutation == "policy":
        state["quality_policy"] = {**state["quality_policy"], "identity": "strict"}
    else:
        state.update(has_fatal_error=True, error_node="validate")
    with pytest.raises(ValueError):
        await ChapterLifecycle(state).stage().publish()
    assert driver.commits == 1
    assert not lifecycle.files.exists("chapters/chapter_001.accepted.json")


@pytest.mark.parametrize("force", [False, True])
async def test_native_checkpoint_cannot_finalize_mandatory_failure(lifecycle_example: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, force: bool) -> None:
    import json

    from langgraph.graph.state import END, StateGraph

    from core.langgraph.nodes.finalize_node import finalize_chapter
    from core.langgraph.quality_policy import SCORE_FIELDS
    from core.langgraph.state import NarrativeState
    from core.langgraph.workflow import create_checkpointer

    state, driver = lifecycle_example
    state["quality_policy"]["identity"] = "strict"
    state.update(force_continue=force, max_iterations=0)
    state.update(await commit_to_graph(state))
    state = await evaluate_example(state, monkeypatch, json.dumps({**dict.fromkeys(SCORE_FIELDS, 0.1), "feedback": "Unresolved"}))
    workflow = StateGraph(NarrativeState)
    workflow.add_node("finalize", finalize_chapter)
    workflow.set_entry_point("finalize")
    workflow.add_edge("finalize", END)
    configuration = {"configurable": {"thread_id": "policy"}}
    async with create_checkpointer(str(tmp_path / "policy.db")) as saver:
        graph = workflow.compile(checkpointer=saver)
        await graph.ainvoke(state, configuration, interrupt_before=["finalize"])
        assert (await graph.aget_state(configuration)).next == ("finalize",)
    async with create_checkpointer(str(tmp_path / "policy.db")) as saver:
        result = await workflow.compile(checkpointer=saver).ainvoke(None, configuration)
    assert result["has_fatal_error"] is True
    assert driver.commits == 1
    assert not (tmp_path / "chapters/chapter_001.accepted.json").exists()


@pytest.mark.parametrize("policy", ["strict", "author"])
async def test_full_workflow_resume_quality_policy(lifecycle_example: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, policy: str) -> None:
    import json

    from core.langgraph.quality_policy import SCORE_FIELDS
    from core.langgraph.workflow import create_checkpointer, create_full_workflow_graph

    state, driver = lifecycle_example
    state["quality_policy"]["identity"] = policy
    state.update(force_continue=True, max_iterations=0)
    state.update(await commit_to_graph(state))
    state = await evaluate_example(state, monkeypatch, json.dumps({**dict.fromkeys(SCORE_FIELDS, 0.1), "feedback": "Synthetic findings"}))
    configuration = {"configurable": {"thread_id": "full-policy"}}
    checkpoint = str(tmp_path / "full-policy.db")
    async with create_checkpointer(checkpoint) as saver:
        graph = create_full_workflow_graph(saver)
        await graph.aupdate_state(configuration, state, as_node="summarize")
        assert (await graph.aget_state(configuration)).next == ("finalize",)
    async with create_checkpointer(checkpoint) as saver:
        graph = create_full_workflow_graph(saver)
        await graph.ainvoke(None, configuration, interrupt_after=["finalize"])
        snapshot = await graph.aget_state(configuration)
        assert snapshot.next == (("error_handler",) if policy == "strict" else ("heal_graph",))
    if policy == "strict":
        assert driver.commits == 1
        assert not (tmp_path / "chapters/chapter_001.accepted.json").exists()
    else:
        lifecycle = ChapterLifecycle(state).stage()
        receipt = json.loads(lifecycle.files.read_bytes(lifecycle.phase_path("acceptance")))
        assert "force_continue_findings" in receipt["quality"]["exceptions"]
        assert receipt["quality"]["status"] == "accepted_with_exceptions"
        assert driver.commits == 2


async def test_advisory_maintenance_findings_are_exceptions(lifecycle_example: Any) -> None:
    import json

    from core.langgraph.quality_policy import retain_maintenance

    state, _ = lifecycle_example
    state.update(await commit_to_graph(state))
    state.update(await ChapterLifecycle(state).stage().publish())
    retain_maintenance(state, "graph_quality", {"issues_found": 1, "contradictory_traits": [{"character_name": "Example"}]})
    directory = Path(state["project_dir"]) / ".saga/attempts" / state["attempt_id"] / "maintenance"
    receipt, = [json.loads(path.read_bytes()) for path in directory.glob("graph_quality-*.json")]
    assert receipt["status"] == "accepted_with_exceptions"
