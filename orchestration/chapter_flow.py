# orchestration/chapter_flow.py
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hint import
    from .nana_orchestrator import NANA_Orchestrator


async def run_chapter_pipeline(
    orchestrator: NANA_Orchestrator, novel_chapter_number: int
) -> str | None:
    """High-level pipeline for generating a chapter."""
    orchestrator._update_rich_display(
        chapter_num=novel_chapter_number, step="Starting Chapter"
    )

    # Optional NarrativeState lifecycle (backward compatible for tests/mocks)
    state = None
    if hasattr(orchestrator, "_begin_chapter_state"):
        try:
            state = orchestrator._begin_chapter_state(novel_chapter_number)
        except Exception:
            state = None

    if not await orchestrator._validate_plot_outline(novel_chapter_number):
        return None

    # Prefer state-aware prerequisites if available
    if hasattr(orchestrator, "_prepare_chapter_prerequisites_with_state"):
        prereq_result = await orchestrator._prepare_chapter_prerequisites_with_state(
            novel_chapter_number, state
        )
    else:
        prereq_result = await orchestrator._prepare_chapter_prerequisites(
            novel_chapter_number
        )
    processed_prereqs = await orchestrator._process_prereq_result(
        novel_chapter_number, prereq_result
    )
    if processed_prereqs is None:
        return None
    plot_point_focus, plot_point_index, chapter_plan, hybrid_context_for_draft = (
        processed_prereqs
    )

    # Build/refresh snapshot once after prerequisites if supported
    if state is not None and hasattr(orchestrator, "_refresh_snapshot"):
        try:
            await orchestrator._refresh_snapshot(state, chapter_plan, novel_chapter_number)
            # Prefer snapshot context for the rest of the chapter
            if state.snapshot and getattr(state.snapshot, "hybrid_context", None):
                hybrid_context_for_draft = state.snapshot.hybrid_context
        except Exception:
            pass

    draft_result = await orchestrator._draft_initial_chapter_text(
        novel_chapter_number,
        plot_point_focus,
        hybrid_context_for_draft,
        chapter_plan,
        state,
    )
    processed_draft = await orchestrator._process_initial_draft(
        novel_chapter_number, draft_result
    )
    if processed_draft is None:
        return None
    initial_draft_text, initial_raw_llm_text = processed_draft

    revision_result = await orchestrator._process_and_revise_draft(
        novel_chapter_number,
        initial_draft_text,
        initial_raw_llm_text,
        plot_point_focus,
        plot_point_index,
        hybrid_context_for_draft,
        chapter_plan,
        state,
    )
    processed_revision = await orchestrator._process_revision_result(
        novel_chapter_number, revision_result
    )
    if processed_revision is None:
        return None
    processed_text, processed_raw_llm, is_flawed = processed_revision

    return await orchestrator._finalize_and_log(
        novel_chapter_number, processed_text, processed_raw_llm, is_flawed, state
    )
