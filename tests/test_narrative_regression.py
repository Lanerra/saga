"""Synthetic controls establish detector behavior, never measured model quality."""
import json
from pathlib import Path
from typing import Any

import pytest

from core.langgraph.content_manager import ContentManager, get_scene_drafts
from core.langgraph.nodes.scene_generation_node import draft_scene
from core.langgraph.state import create_initial_state
from core.narrative_regression import NarrativeExpectations, assess_narrative
from core.service_context import get_services

FIXTURES = Path(__file__).parent / "fixtures/narrative_regressions.json"
DIMENSIONS = ["continuity", "fact_coverage", "repetition", "perspective", "prose_fidelity", "usefulness"]


def fixture_stories() -> list[dict[str, Any]]:
    return json.loads(FIXTURES.read_text())["stories"]


@pytest.mark.parametrize("number", [0, 1])
async def test_fixed_stories_through_drafting(number: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    story = fixture_stories()[number]
    state = create_initial_state(
        project_id=tmp_path.name, project_dir=str(tmp_path),
        title=story["title"], genre=story["genre"], theme="Trust", setting=story["setting"],
        protagonist_name=story["protagonist"], narrative_style=story["style"],
        total_chapters=1, target_word_count=120,
    )
    manager = ContentManager(str(tmp_path))
    state["chapter_plan_ref"] = manager.save_json(story["scene_plan"], "chapter_plan", "chapter_1", 1)
    state["hybrid_context_ref"] = manager.save_text(story["prior_story"], "context", "prior", 1)
    prompts: list[str] = []

    async def synthetic_completion(model_name: str, prompt: str, **options: Any) -> tuple[str, dict[str, int]]:
        prompts.append(prompt)
        return story["synthetic_scenes"][len(prompts) - 1], {}

    monkeypatch.setattr(get_services().language_model, "async_call_llm", synthetic_completion)
    for _ in story["scene_plan"]:
        update = await draft_scene(state)
        assert update.get("has_fatal_error", False) is False
        state.update(update)
    drafts = get_scene_drafts(state, manager)
    assert drafts == story["synthetic_scenes"]
    for prompt in prompts:
        assert story["style"] in prompt
        assert story["prior_story"] in prompt
    assert story["scene_plan"][0]["outcome"] in prompts[1]
    assessment = assess_narrative("\n\n".join(drafts), NarrativeExpectations.model_validate(story["expectations"]))
    assert assessment["scope"] == "mechanical_only"
    assert assessment["issues"] == {dimension: [] for dimension in DIMENSIONS}
    assert assessment["human_review_required"] == DIMENSIONS
    (tmp_path / "mechanical-assessment.json").write_text(json.dumps({
        "fixture_id": story["identifier"], "provider": "synthetic-fixed-control",
        "model_quality_measured": False, "assessment": assessment,
        "prompts": prompts, "scenes": drafts, "human_review_questions": story["human_review_questions"],
    }, indent=2))


@pytest.mark.parametrize("number", [0, 1])
@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_each_negative_control_is_detected(number: int, dimension: str) -> None:
    story = fixture_stories()[number]
    candidate = "\n\n".join(story["synthetic_scenes"])
    control = story["negative_controls"][dimension]
    candidate = candidate.replace(control["remove"], "") if control["remove"] else candidate
    candidate += control["append"]
    assessment = assess_narrative(candidate, NarrativeExpectations.model_validate(story["expectations"]))
    assert assessment["issues"][dimension] == control["expected_issues"]
    assert assessment["human_review_required"] == DIMENSIONS


def test_empty_prose_cannot_pass_mechanical_screen() -> None:
    story = fixture_stories()[0]
    result = assess_narrative("", NarrativeExpectations.model_validate(story["expectations"]))
    assert result["issues"]["usefulness"] == ["word_count_outside_range", "missing_outcome"]
    assert result["issues"]["fact_coverage"] == story["expectations"]["required_facts"]


def test_fixture_collection_is_finite_and_explicitly_synthetic() -> None:
    document = json.loads(FIXTURES.read_text())
    assert document["provenance"] == "authored_synthetic_controls_not_model_outputs"
    assert [story["identifier"] for story in document["stories"]] == ["archive-key", "canal-signal"]
    for story in document["stories"]:
        assert sorted(story["human_review_questions"]) == sorted(DIMENSIONS)
        assert sorted(story["negative_controls"]) == sorted(DIMENSIONS)
