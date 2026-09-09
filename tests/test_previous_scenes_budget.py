"""Aggregate rendering invariants under an explicit synthetic BPE contract.

These exercise tiktoken and the production TokenizerService, not configured-model
compatibility. Configured cl100k_base assets require separate provisioning.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import tiktoken

from core import text_processing_service
from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes import context_scene_retrieval
from core.service_context import get_services

MODEL = "b13-synthetic-bpe"
SECTION = "\n\n**Previous Scenes in This Chapter:**\n"


@pytest.fixture(autouse=True)
def offline_language_assets() -> None:
    """Override the global synthetic-byte fixture explicitly."""


@pytest.fixture
def encoder(monkeypatch: pytest.MonkeyPatch) -> tiktoken.Encoding:
    ranks = {bytes([number]): number for number in range(256)}
    ranks.update({b"aa": 256, b" a": 257, b"\n\n": 258, b"--": 259, b"---": 260})
    encoding = tiktoken.Encoding(name=MODEL, pat_str=r"(?s:.+)", mergeable_ranks=ranks, special_tokens={"<|endoftext|>": 1000})
    tokenizer = text_processing_service.TokenizerService()
    tokenizer._tokenizer_cache[MODEL] = encoding
    monkeypatch.setattr(text_processing_service, "_default_tokenizer", tokenizer)
    assert len(encoding.encode("aa")) < len(encoding.encode("a")) * 2
    return encoding


@pytest.fixture
def summary_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def generate_summary(**arguments: Any) -> tuple[str, dict[str, Any]]:
        calls.append(arguments)
        return "Résumé 界🌙 " * 120, {}

    monkeypatch.setattr(get_services().language_model, "async_call_llm", generate_summary)
    return calls


@pytest.fixture
def build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, encoder: tiktoken.Encoding, summary_calls: list[dict[str, Any]]) -> Callable[..., Any]:
    manager = ContentManager(str(tmp_path))

    async def render(drafts: list[str], titles: list[str], budget: int, scene_index: int | None = None) -> str | None:
        monkeypatch.setattr(context_scene_retrieval, "PREVIOUS_SCENES_TOKEN_BUDGET", budget)
        return await context_scene_retrieval.get_previous_scenes_context(
            state={}, scene_drafts=drafts, chapter_plan=[{"title": title} for title in titles],
            scene_index=len(drafts) if scene_index is None else scene_index,
            model_name=MODEL, extraction_model=MODEL, content_manager=manager,
        )

    return render


@pytest.mark.parametrize("budget", [0, 1, 8, 40, 80, 180, 512])
@pytest.mark.parametrize("scene_count", [1, 4, 80])
@pytest.mark.parametrize("title", ["Arrival", "海🌙 café", "Oversized heading " * 80], ids=["ordinary-heading", "unicode-heading", "oversized-heading"])
async def test_complete_rendering_obeys_one_budget(
    build: Callable[..., Any], encoder: tiktoken.Encoding, budget: int,
    scene_count: int, title: str, request: pytest.FixtureRequest,
) -> None:
    result = await build(["a a café 🌙 界\n" * 15] * scene_count, [title] * scene_count, budget)
    measured = len(encoder.encode(result or "", allowed_special="all"))
    request.node.user_properties.extend([
        ("budget", budget), ("measured_tokens", measured),
        ("encoder", encoder.name), ("scene_count", scene_count),
    ])
    assert measured <= budget
    if title == "Arrival" and budget >= 180:
        assert result is not None
        assert result.startswith(SECTION)


async def test_exact_fit_preserves_every_character(build: Callable[..., Any], encoder: tiktoken.Encoding, summary_calls: list[dict[str, Any]]) -> None:
    expected = SECTION + "\n--- Arrival ---\naa café 🌙\n\n--- Departure ---\n界\n"
    budget = len(encoder.encode(expected))
    assert await build(["aa café 🌙", "界"], ["Arrival", "Departure"], budget) == expected
    assert summary_calls == []


async def test_recent_scenes_have_priority_with_chronological_display(build: Callable[..., Any], encoder: tiktoken.Encoding) -> None:
    expected = SECTION + "\n--- Third ---\nlatest\n"
    budget = len(encoder.encode(expected))
    assert await build(["earlier", "middle", "latest"], ["First", "Second", "Third"], budget) == expected
    expected = SECTION + "\n--- Second ---\nmiddle\n\n--- Third ---\nlatest\n"
    assert await build(["earlier", "middle", "latest"], ["First", "Second", "Third"], len(encoder.encode(expected))) == expected


async def test_current_and_future_drafts_do_not_consume_budget(build: Callable[..., Any], encoder: tiktoken.Encoding) -> None:
    expected = SECTION + "\n--- First ---\npast\n"
    assert await build(["past", "current", "future"], ["First", "Second", "Third"], len(encoder.encode(expected)), scene_index=1) == expected


@pytest.mark.parametrize("drafts,scene_index", [([], 0), ([""], 1), (["   "], 1), (["future"], 0)])
async def test_no_previous_content_returns_none(build: Callable[..., Any], summary_calls: list[dict[str, Any]], drafts: list[str], scene_index: int) -> None:
    assert await build(drafts, ["Title"], 500, scene_index=scene_index) is None
    assert summary_calls == []


async def test_zero_budget_never_requests_summary(build: Callable[..., Any], summary_calls: list[dict[str, Any]]) -> None:
    assert await build(["long " * 20000], ["Arrival"], 0) is None
    assert summary_calls == []


async def test_oversized_heading_does_not_displace_usable_scene(build: Callable[..., Any]) -> None:
    assert await build(["past", "latest"], ["First", "heading " * 1000], 80) == SECTION + "\n--- First ---\npast\n"


async def test_summary_overrun_and_marker_are_measured(build: Callable[..., Any], encoder: tiktoken.Encoding, summary_calls: list[dict[str, Any]]) -> None:
    result = await build(["long " * 20000], ["Arrival"], 180)
    assert len(summary_calls) == 1
    assert 0 < summary_calls[0]["max_tokens"] <= 180
    assert result is not None
    assert result.startswith(SECTION + "\n--- Arrival (Summary) ---\n")
    assert "[...]" in result
    assert "\ufffd" not in result
    assert len(encoder.encode(result)) <= 180


async def test_negative_budget_is_rejected(build: Callable[..., Any]) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        await build(["past"], ["First"], -1)


async def test_imports_and_counter_use_this_tree(encoder: tiktoken.Encoding) -> None:
    root = Path(__file__).resolve().parents[1]
    assert Path(context_scene_retrieval.__file__).resolve() == root / "core/langgraph/nodes/context_scene_retrieval.py"
    assert Path(text_processing_service.__file__).resolve() == root / "core/text_processing_service.py"
    text = "aa café 🌙\n\n<|endoftext|>"
    assert text_processing_service.count_tokens(text, MODEL) == len(encoder.encode(text, allowed_special="all"))
    assert text_processing_service._default_tokenizer._stats["fallback_used"] == 0
