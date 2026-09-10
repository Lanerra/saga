# core/langgraph/nodes/revision_node.py
"""Revise a chapter draft based on validation feedback.

This module defines the revision node used in the LangGraph workflow. It applies
feedback (contradictions) to the current draft, generates a revised version, and
externalizes the updated draft to keep workflow state small.

Migration Reference: docs/phase2_migration_plan.md - Step 2.2
"""

from __future__ import annotations

import structlog

import config
from core.langgraph.chapter_lifecycle import ChapterLifecycle, canonical_bytes
from core.langgraph.content_manager import (
    ContentManager,
    get_chapter_outlines,
    get_chapter_plan,
    get_hybrid_context,
    require_project_dir,
)
from core.langgraph.state import Contradiction, NarrativeState
from core.langgraph.state_helpers import (
    clear_error_state,
    clear_extraction_state,
    clear_generation_artifacts,
)
from core.service_context import get_services
from data_access.cypher_builders.native_builders import chapter_assertion_delete_statement
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)


async def _rollback_chapter_data(chapter_number: int, *, lifecycle: ChapterLifecycle | None = None) -> None:
    """Delete entities and relationships committed for a chapter that needs revision.

    This function performs a compensating transaction to rollback data committed
    before validation determined that revision was needed.

    Strategy:
        1. Delete all relationships added in this chapter
        2. Mark entities created in this chapter as provisional (graph healing will clean up orphans)
        3. Delete the chapter node itself

    Args:
        chapter_number: The chapter number to rollback.

    Raises:
        Exception: Rollback failure requires reconciliation before revision.
    """
    logger.info(
        "rollback_chapter_data: removing committed data for revision",
        chapter=chapter_number,
    )

    queries = [
        chapter_assertion_delete_statement(chapter_number),
        (
            """
            MATCH (e)
            WHERE coalesce(e.created_chapter, -1) = $chapter
            SET e.is_provisional = true
            """,
            {"chapter": chapter_number},
        ),
        (
            """
            MATCH (s:Scene)-[r:PART_OF]->(ch:Chapter {number: $chapter})
            DELETE r, s
            """,
            {"chapter": chapter_number},
        ),
        (
            """
            MATCH (ch:Chapter {number: $chapter})
            OPTIONAL MATCH (ch)-[r]-()
            DELETE r, ch
            """,
            {"chapter": chapter_number},
        ),
    ]

    try:
        if lifecycle is None:
            await get_services().database.execute_cypher_batch(queries)
        else:
            if lifecycle.chapter_number != chapter_number:
                raise ValueError("Compensation chapter mismatch")
            await lifecycle.compensate(queries)
        logger.info(
            "rollback_chapter_data: successfully rolled back chapter data",
            chapter=chapter_number,
        )
    except Exception as exc:
        logger.error(
            "rollback_chapter_data: failed to rollback chapter data",
            chapter=chapter_number,
            error=str(exc),
            exc_info=True,
        )
        raise


async def revise_chapter(state: NarrativeState) -> NarrativeState:
    """Produce revision guidance and reset chapter artifacts for scene-level regeneration.

    Revision semantics (scene-first pipeline):
        - This node does NOT rewrite the chapter draft.
        - It produces externalized `revision_guidance_ref` that downstream scene drafting
          can incorporate.
        - It clears stale artifacts (`scene_drafts_ref`, `scene_embeddings_ref`, `draft_ref`,
          `embedding_ref`, extraction refs) so the regenerated pipeline cannot reuse them.
        - It resets `current_scene_index` to 0 so the generation subgraph drafts scenes again.
        - ROLLBACK: Since commit now happens before validation, this node deletes
          committed entities/relationships for this chapter before regenerating.
    """
    chapter_number = state.get("current_chapter", 1)

    rollback_failure = state.get("revision_rollback_failure")
    if rollback_failure is not None and "lifecycle_version" not in state:
        return {
            "last_error": rollback_failure["error"],
            "has_fatal_error": True,
            "error_node": "revise",
            "current_node": "revise_blocked",
        }

    logger.info(
        "revise_chapter: generating revision guidance",
        chapter=chapter_number,
        iteration=state.get("iteration_count", 0),
        contradictions=len(state.get("contradictions", [])),
    )

    if state.get("iteration_count", 0) >= state.get("max_iterations", 3):
        error_msg = f"Max revision attempts ({state.get('max_iterations', 3)}) reached"
        logger.error(
            "revise_chapter: iteration limit exceeded",
            iterations=state.get("iteration_count", 0),
            max_iterations=state.get("max_iterations", 3),
        )
        return {
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "revise",
            "needs_revision": False,
            "current_node": "revise_failed",
        }

    lifecycle = None
    try:
        if "lifecycle_version" in state:
            lifecycle = ChapterLifecycle(state).stage()
            await _rollback_chapter_data(chapter_number, lifecycle=lifecycle)
            previous_result = lifecycle.revision_result()
            if previous_result is not None:
                return previous_result
            state = {**state, **lifecycle.state_update("compensated"), "revision_rollback_failure": None}
        else:
            await _rollback_chapter_data(chapter_number)
    except Exception as exc:
        error_msg = f"Revision rollback failed; reconciliation required: {exc}"
        logger.error(
            "revise_chapter: rollback failed, blocking revision",
            chapter=chapter_number,
            error=str(exc),
            exc_info=True,
        )

        return {
            "revision_rollback_failure": {
                "chapter_number": chapter_number,
                "iteration_count": state.get("iteration_count", 0),
                "error": error_msg,
                "previous_error": state.get("last_error"),
                "previous_error_node": state.get("error_node"),
            },
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "revise",
            "current_node": "revise_blocked",
        }

    content_manager = ContentManager(require_project_dir(state))

    try:
        hybrid_context = get_hybrid_context(state, content_manager)
        chapter_outlines = get_chapter_outlines(state, content_manager)
        chapter_plan = get_chapter_plan(state, content_manager)
        protagonist_name = state.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)

        if not chapter_plan:
            raise ValueError("No chapter plan available for revision guidance")

        scene_lines: list[str] = []
        for scene_index, scene in enumerate(chapter_plan, 1):
            title = scene.get("title", "")
            pov_character = scene.get("pov_character", "")
            setting = scene.get("setting", "")
            plot_point = scene.get("plot_point", "")
            conflict = scene.get("conflict", "")
            outcome = scene.get("outcome", "")

            scene_lines.append(
                "\n".join(
                    [
                        f"{scene_index}. {title}".strip(),
                        f"   POV: {pov_character}".strip(),
                        f"   Setting: {setting}".strip(),
                        f"   Plot point: {plot_point}".strip(),
                        f"   Conflict: {conflict}".strip(),
                        f"   Outcome: {outcome}".strip(),
                    ]
                )
            )

        plot_point_focus = ""
        chapter_outline = chapter_outlines.get(chapter_number, {})
        if isinstance(chapter_outline, dict):
            plot_point_focus = str(chapter_outline.get("plot_point", "") or "")

        revision_reason = _format_contradictions_for_prompt(state.get("contradictions", []))

        prompt = render_prompt(
            "revision_agent/revision_guidance.j2",
            {
                "title": state.get("title", ""),
                "genre": state.get("genre", ""),
                "protagonist_name": protagonist_name,
                "chapter_number": chapter_number,
                "plot_point_focus": plot_point_focus,
                "hybrid_context": hybrid_context or "",
                "scene_lines": scene_lines,
                "revision_reason": revision_reason,
            },
        )

    except Exception as exc:
        error_msg = f"Revision guidance prompt construction failed: {str(exc)}"
        logger.error(
            "revise_chapter: fatal error",
            error=error_msg,
            exc_info=True,
        )
        return {
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "revise",
            "current_node": "revise",
        }

    model_name = state.get("revision_model", config.MEDIUM_MODEL)
    prompt_tokens = get_services().language_model.count_tokens(prompt, model_name)

    max_context = config.MAX_CONTEXT_TOKENS
    token_buffer = getattr(config.settings, "NARRATIVE_TOKEN_BUFFER", 16384)
    max_generation = config.MAX_GENERATION_TOKENS

    available_tokens = max_context - prompt_tokens - token_buffer
    max_gen_tokens = min(max_generation, available_tokens)

    if max_gen_tokens < 200:
        error_msg = "Insufficient token space for revision guidance. " f"Prompt tokens: {prompt_tokens}, available: {available_tokens}"
        logger.error("revise_chapter: fatal error - token budget exceeded", error=error_msg)
        return {
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "revise",
            "current_node": "revise",
        }

    try:
        revision_guidance, _ = await get_services().language_model.async_call_llm(
            model_name=model_name,
            prompt=prompt,
            temperature=getattr(config.Temperatures, "REVISION", 0.5),
            max_tokens=max_gen_tokens,
            allow_fallback=True,
            frequency_penalty=getattr(config, "FREQUENCY_PENALTY_DRAFTING", 0.3),
            presence_penalty=getattr(config, "PRESENCE_PENALTY_DRAFTING", 0.3),
            auto_clean_response=True,
            system_prompt=get_system_prompt("revision_agent"),
        )
    except Exception:
        logger.error(
            "revise_chapter: revision guidance generation failed",
            chapter=chapter_number,
            exc_info=True,
        )
        return {
            "last_error": "Revision guidance generation failed",
            "has_fatal_error": True,
            "error_node": "revise",
            "current_node": "revise",
        }

    if not revision_guidance or not revision_guidance.strip():
        return {
            "last_error": "Revision guidance generation failed",
            "has_fatal_error": True,
            "error_node": "revise",
            "current_node": "revise",
        }

    current_version = (
        content_manager.get_latest_version(
            "revision_guidance",
            f"chapter_{chapter_number}",
        )
        + 1
    )
    revision_guidance_ref = content_manager.save_text(
        revision_guidance.strip(),
        "revision_guidance",
        f"chapter_{chapter_number}",
        version=current_version,
    )

    result: NarrativeState = {
        "revision_guidance_ref": revision_guidance_ref,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "contradictions": [],
        "needs_revision": False,
        "current_node": "revise",
        **clear_generation_artifacts(),
        **clear_extraction_state(),
        **clear_error_state(),
    }
    if lifecycle is not None:
        result.update({"attempt_id": None, "lifecycle_phase": "compensated", "extraction_source": None, "extraction_status": "failed", "extraction_outcomes": [], "revision_rollback_failure": None})
        lifecycle.retain_revision_result(canonical_bytes(result))
    return result


def _format_contradictions_for_prompt(contradictions: list[Contradiction]) -> str:
    """
    Format contradictions into a human-readable revision reason.

    Args:
        contradictions: List of Contradiction objects

    Returns:
        Formatted string describing all contradictions
    """
    if not contradictions:
        return "General quality improvements needed."

    # Group by severity
    critical = [c for c in contradictions if c.severity == "critical"]
    major = [c for c in contradictions if c.severity == "major"]
    minor = [c for c in contradictions if c.severity == "minor"]

    parts = []

    if critical:
        parts.append("**CRITICAL ISSUES:**")
        for c in critical:
            parts.append(f"- {c.description}")

    if major:
        parts.append("\n**MAJOR ISSUES:**")
        for c in major:
            parts.append(f"- {c.description}")

    if minor:
        parts.append("\n**MINOR ISSUES:**")
        for c in minor:
            parts.append(f"- {c.description}")

    return "\n".join(parts)


__all__ = ["revise_chapter"]
