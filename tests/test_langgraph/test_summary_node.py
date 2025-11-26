# tests/test_langgraph/test_summary_node.py
import asyncio
from pathlib import Path

import yaml

from core.langgraph.nodes.summary_node import summarize_chapter
from core.langgraph.state import NarrativeState


async def _run_summarize_chapter(
    tmp_path: Path, summary_text: str = "This is a test summary."
) -> Path:
    """
    Helper to invoke summarize_chapter with a minimal deterministic state.

    Uses a fake LLM output by monkeypatching llm_service.async_call_llm via a simple
    shim state where we directly call the internal behavior by constructing the
    expected summary response shape.
    """
    # Build minimal state resembling NarrativeState where it matters.
    state: NarrativeState = {
        "project_dir": str(tmp_path),
        "current_chapter": 1,
        "draft_text": "Chapter 1 draft text.",
        "extraction_model": "test-model",
        "previous_chapter_summaries": [],
    }

    # We monkeypatch by temporarily importing and overriding llm_service.async_call_llm.
    # Import inside function to avoid test discovery side-effects.
    from core.langgraph.nodes import summary_node as summary_module

    async def fake_async_call_llm(
        model_name: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        allow_fallback: bool,
        stream_to_disk: bool,
        auto_clean_response: bool,
        system_prompt: str,
    ):
        # Return deterministic summary_text and dummy usage.
        return summary_text, {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    original_call = summary_module.llm_service.async_call_llm

    try:
        summary_module.llm_service.async_call_llm = fake_async_call_llm  # type: ignore[assignment]
        new_state = await summarize_chapter(state)
    finally:
        summary_module.llm_service.async_call_llm = original_call  # type: ignore[assignment]

    # Ensure node updated state as expected
    assert new_state["current_node"] == "summarize"
    # Check that summary was added (list is not empty)
    assert new_state.get(
        "previous_chapter_summaries"
    ), "Summary list should not be empty"
    # Verify the last summary matches
    assert new_state["previous_chapter_summaries"][-1] == summary_text

    # Return path to generated summary file
    return tmp_path / "summaries" / "chapter_001.md"


def test_summary_file_written_with_front_matter_and_body(tmp_path: Path) -> None:
    """
    Verify summarize_chapter writes summaries/chapter_001.md with YAML front matter
    and the expected summary body, and that no literal '\\n' sequences appear.
    """
    summary_text = "This is a deterministic test summary about chapter 1."

    summary_path = asyncio.run(_run_summarize_chapter(tmp_path, summary_text))

    # File should exist
    assert summary_path.is_file(), f"Expected summary file at {summary_path}"

    content = summary_path.read_text(encoding="utf-8")
    # Must start with YAML front matter
    assert content.startswith("---\n")

    # Split front matter and body
    parts = content.split("---")
    # parts: ["", "\nfront-matter...\n", "\nbody..."]
    assert len(parts) >= 3, "Expected YAML front matter delimited by ---"

    front_matter_raw = parts[1]
    # Body starts after closing --- and optional leading newline
    body = "---".join(parts[2:]).lstrip("\n")

    meta = yaml.safe_load(front_matter_raw)
    assert isinstance(meta, dict)
    assert meta.get("chapter") == 1
    # generated_at might be parsed as datetime object by yaml.safe_load
    assert meta.get("generated_at"), "generated_at should be present"

    # Title is optional; no strict assertion needed here.

    # Body should match the summary text (ignoring trailing newline)
    assert body.strip() == summary_text

    # Ensure no literal "\n" sequences; all newlines must be real
    assert "\\n" not in body
