"""Synthetic producer-to-admission contracts; no narrative-quality claims."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization.act_outlines_node import _get_act_role, generate_act_outlines
from core.langgraph.initialization.chapter_outline_node import _parse_chapter_outline, generate_chapter_outline
from core.langgraph.initialization.global_outline_node import _parse_global_outline
from core.langgraph.state import NarrativeState
from core.service_context import get_services
from tests.test_staged_initialization import example_state


def global_response(total: int = 3, count: int | None = None) -> dict[str, Any]:
    count = min(total, 3) if count is None else count
    cursor = 1
    acts = []
    for number in range(1, count + 1):
        size = total // count + int(number <= total % count)
        acts.append(dict(act_number=number, title=f"Act {number}", summary="Ada explores", key_events=["Ada chooses"], chapters_start=cursor, chapters_end=cursor + size - 1))
        cursor += size
    return dict(act_count=count, acts=acts, inciting_incident="Ada departs", midpoint="Ada learns", climax="Ada chooses", resolution="Ada returns", character_arcs=[], thematic_progression="Courage", pacing_notes="Steady")


def chapter_response() -> dict[str, Any]:
    return dict(scene_description="Ada enters the harbor.", key_beats=["Ada arrives", "Ada reads", "Ada chooses"], plot_point="Ada chooses to stay.")


@pytest.mark.parametrize("case", ["extra", "nested_extra", "coercion", "empty_act", "reverse_acts", "duplicate_arc", "duplicate_json"])
def test_global_producer_rejects_lossy_or_invalid_output(case: str) -> None:
    data = global_response()
    if case == "extra":
        data["selected_world"] = "Must not disappear"
    elif case == "nested_extra":
        data["acts"][0]["selected_location"] = "Harbor"
    elif case == "coercion":
        data["act_count"] = "3"
    elif case == "empty_act":
        data["acts"][0]["chapters_end"] = 0
    elif case == "reverse_acts":
        data["acts"].reverse()
    elif case == "duplicate_arc":
        arc = dict(character_name="Ada", starting_state="Afraid", ending_state="Brave", key_moments=[])
        data["character_arcs"] = [arc, arc]
    raw = json.dumps(data)
    if case == "duplicate_json":
        raw = raw.replace('"act_count": 3', '"act_count": 2, "act_count": 3')
    with pytest.raises(ValueError):
        _parse_global_outline(raw, {"total_chapters": 3})


@pytest.mark.parametrize("case", ["text", "extra", "missing", "wrong_type", "too_many", "duplicate", "empty"])
def test_chapter_producer_rejects_instead_of_repairing(case: str) -> None:
    data = chapter_response()
    if case == "extra":
        data["selected_location"] = "Harbor"
    elif case == "missing":
        del data["plot_point"]
    elif case == "wrong_type":
        data["key_beats"] = "Ada chooses"
    elif case == "too_many":
        data["key_beats"] = [f"Beat {number}" for number in range(11)]
    elif case == "empty":
        data["scene_description"] = ""
    raw = json.dumps(data)
    if case == "text":
        raw = "Summary\nAda explores\nBeats\n- Ada arrives"
    elif case == "duplicate":
        raw = raw.replace('"plot_point":', '"plot_point": "discarded", "plot_point":')
    with pytest.raises(ValueError):
        _parse_chapter_outline(raw, 1, 1)


async def test_act_failure_does_not_publish_partial_collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state: NarrativeState = {"project_dir": str(tmp_path), "total_chapters": 3}
    manager = ContentManager(str(tmp_path))
    state["global_outline_ref"] = manager.save_json(_parse_global_outline(json.dumps(global_response()), state), "global_outline", "main", version=2)
    state["act_outlines_ref"] = None
    calls = 0

    async def transport(**arguments: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("synthetic act schema failure")
        sections: dict[str, Any] = {key: "Ada explores" for key in ("act_summary", "opening_situation", "character_development", "stakes_and_tension", "act_ending_turn", "thematic_thread", "pacing_notes")}
        sections["key_events"] = [dict(sequence=number, event="Ada explores", cause="Choice", effect="Discovery") for number in range(1, 6)]
        return dict(act_number=calls, total_acts=3, act_role=_get_act_role(calls, 3), chapters_in_act=1, sections=sections), {}

    monkeypatch.setattr(get_services().language_model, "async_call_llm_json_object", transport)
    result = await generate_act_outlines(state)
    assert result["initialization_step"] == "act_outlines_failed"
    assert "act_outlines_ref" not in result
    assert calls == 2


async def test_failed_enrichment_is_not_successful_existing_outline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = example_state(tmp_path)
    state["current_chapter"] = 1

    async def transport(**arguments: Any) -> tuple[str, dict[str, Any]]:
        raise ValueError("synthetic enrichment failure")

    monkeypatch.setattr(get_services().language_model, "async_call_llm", transport)
    result = await generate_chapter_outline(state)
    assert result["initialization_step"] == "chapter_outline_1_failed"
    assert result["last_error"]
    assert "chapter_outlines_ref" not in result


def test_valid_chapter_preserves_exact_fields() -> None:
    data = chapter_response()
    raw = json.dumps(data)
    result = _parse_chapter_outline(raw, 2, 1)
    assert {key: result[key] for key in data} == data
    assert result["raw_text"] == raw
    assert result["chapter_number"] == 2
    assert result["act_number"] == 1
