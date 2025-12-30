# tests/test_langgraph/test_summary_node.py
import asyncio
from pathlib import Path

import yaml

from core.langgraph.content_manager import ContentManager, get_previous_summaries
from core.langgraph.nodes.summary_node import summarize_chapter
from core.langgraph.state import NarrativeState
from data_access import chapter_queries


async def _run_summarize_chapter(tmp_path: Path, summary_text: str = "This is a test summary.") -> Path:
    """
    Helper to invoke summarize_chapter with a minimal deterministic state.

    Uses a fake LLM output by monkeypatching llm_service.async_call_llm via a simple
    shim state where we directly call the internal behavior by constructing the
    expected summary response shape.
    """
    project_dir = str(tmp_path)
    content_manager = ContentManager(project_dir)

    # Save draft text
    draft_ref = content_manager.save_text("Chapter 1 draft text.", "draft", "chapter_1", 1)

    # Save summaries
    sum_ref = content_manager.save_list_of_texts([], "summaries", "all", 1)

    # Build minimal state resembling NarrativeState where it matters.
    state: NarrativeState = {
        "project_dir": project_dir,
        "current_chapter": 1,
        "draft_ref": draft_ref,
        "extraction_model": "test-model",
        "small_model": "test-model",
        "summaries_ref": sum_ref,
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
        auto_clean_response: bool,
        system_prompt: str,
    ):
        import json

        return json.dumps({"summary": summary_text}), {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    original_call = summary_module.llm_service.async_call_llm
    original_write = summary_module.neo4j_manager.execute_write_query

    write_calls: list[tuple[str, dict]] = []

    async def fake_write(query: str, params=None):
        p = params or {}
        write_calls.append((query, p))

        # Canonical chapter upsert must always set Chapter.id.
        assert "c.id" in query
        assert "chapter_id_param" in query or "chapter_id" in query

        assert p.get("chapter_number_param") == 1
        assert p.get("chapter_id_param") == chapter_queries.compute_chapter_id(1)
        assert p.get("summary_param") == summary_text

        return []

    try:
        summary_module.llm_service.async_call_llm = fake_async_call_llm  # type: ignore[assignment]
        summary_module.neo4j_manager.execute_write_query = fake_write  # type: ignore[assignment]
        new_state = await summarize_chapter(state)
    finally:
        summary_module.llm_service.async_call_llm = original_call  # type: ignore[assignment]
        summary_module.neo4j_manager.execute_write_query = original_write  # type: ignore[assignment]

    # Ensure node updated state as expected
    assert new_state["current_node"] == "summarize"

    # Check that summary was added via content manager
    summaries = get_previous_summaries(new_state, content_manager)
    assert summaries, "Summary list should not be empty"

    # Verify the last summary matches
    assert summaries[-1] == summary_text

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
