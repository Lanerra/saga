# core/langgraph/nodes/summary_node.py
"""Summarize a chapter for use as future drafting context.

This module defines the summarization node that produces a short chapter summary,
persists it to Neo4j, and externalizes the rolling summary window to keep workflow
state small.

Migration Reference: docs/phase2_migration_plan.md - Step 2.3

Notes:
    This node performs LLM I/O and Neo4j writes. Summarization is treated as
    best-effort and should not block the pipeline on failures.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import structlog

import config
from core.db_manager import neo4j_manager
from core.langgraph.content_manager import (
    ContentManager,
    get_draft_text,
    get_previous_summaries,
    require_project_dir,
)
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from data_access import chapter_queries
from prompts.prompt_renderer import get_system_prompt, render_prompt
from utils.common import try_load_json_from_response
from utils.file_io import write_text_file

logger = structlog.get_logger(__name__)


class ChapterSummaryContractError(ValueError):
    """Raised when the chapter summary response violates the strict JSON contract."""


_SUMMARY_MAX_ATTEMPTS = 3
_SUMMARY_CORRECTION_INSTRUCTION = "\n\nCORRECTION:\n" 'Return ONLY valid JSON. Output MUST be a single JSON object with exactly one key: "summary".\n' "No markdown. No code fences. No extra text.\n"


async def summarize_chapter(state: NarrativeState) -> NarrativeState:
    """Generate and persist a concise summary of the current chapter.

    The summary is used as compact context in later chapters. The rolling window of
    summaries is externalized via `ContentManager.save_list_of_texts()` to avoid
    bloating workflow state.

    Args:
        state: Workflow state.

    Returns:
        Updated state containing:
        - summaries_ref: Content reference for the rolling summary window.
        - current_summary: Summary text for the current chapter.
        - current_node: `"summarize"`.

        If the draft is missing/empty, returns a no-op update.
        If summarization fails, returns a best-effort no-op update and does not set
        `has_fatal_error`.

    Notes:
        This node performs LLM I/O and Neo4j writes. Failures are non-fatal by design.
    """
    logger.info(
        "summarize_chapter: starting summarization",
        chapter=state.get("current_chapter", 1),
    )

    # Initialize content manager for reading externalized content
    content_manager = ContentManager(require_project_dir(state))

    from core.exceptions import MissingDraftReferenceError

    try:
        draft_text = get_draft_text(state, content_manager)
    except MissingDraftReferenceError:
        logger.info(
            "summarize_chapter: missing draft_ref; skipping summarization",
            chapter=state.get("current_chapter", 1),
        )
        return {
            "current_node": "summarize",
        }

    # Validate we have text to summarize
    if not draft_text:
        logger.warning("summarize_chapter: no draft text to summarize")
        return {
            "current_node": "summarize",
        }

    # Step 1: Build summary prompt
    prompt = render_prompt(
        "knowledge_agent/chapter_summary.j2",
        {
            "chapter_number": state.get("current_chapter", 1),
            "chapter_text": draft_text,
        },
    )

    # Step 2: Generate summary using fast extraction model
    logger.info(
        "summarize_chapter: calling LLM for summary",
        chapter=state.get("current_chapter", 1),
        model=state.get("extraction_model", config.SMALL_MODEL),
    )

    try:
        last_contract_error: ChapterSummaryContractError | None = None

        for attempt_index in range(_SUMMARY_MAX_ATTEMPTS):
            attempt_prompt = prompt
            if attempt_index > 0:
                attempt_prompt = attempt_prompt + _SUMMARY_CORRECTION_INSTRUCTION

            summary_text, usage = await llm_service.async_call_llm(
                model_name=state.get("small_model", config.SMALL_MODEL),  # Use fast model
                prompt=attempt_prompt,
                temperature=0.3,  # Low temperature for consistency
                max_tokens=config.MAX_GENERATION_TOKENS,
                allow_fallback=True,
                auto_clean_response=True,
                system_prompt=get_system_prompt("knowledge_agent"),
            )

            if not summary_text or not summary_text.strip():
                last_contract_error = ChapterSummaryContractError("Chapter summary JSON contract violated: model returned an empty response.")
            else:
                try:
                    summary = _parse_summary_response(summary_text)
                    break
                except ChapterSummaryContractError as error:
                    last_contract_error = error
            if attempt_index < _SUMMARY_MAX_ATTEMPTS - 1:
                logger.warning(
                    "summarize_chapter: chapter summary response violated JSON contract; retrying",
                    chapter=state.get("current_chapter", 1),
                    attempt=attempt_index + 1,
                    max_attempts=_SUMMARY_MAX_ATTEMPTS,
                    error=str(last_contract_error),
                )
                continue

            raise last_contract_error

        logger.info(
            "summarize_chapter: summary generated",
            chapter=state.get("current_chapter", 1),
            summary_length=len(summary),
        )

        # Step 4: Persist to Neo4j
        await _save_summary_to_neo4j(
            chapter_number=state.get("current_chapter", 1),
            summary=summary,
        )

        # Step 5: Persist per-chapter summary file (best-effort, non-fatal)
        try:
            _write_chapter_summary_file(
                chapter_number=state.get("current_chapter"),
                summary_text=summary,
                project_dir=state.get("project_dir"),
            )
        except Exception as e:
            logger.error(
                "summarize_chapter: failed to write summary file",
                chapter=state.get("current_chapter"),
                error=str(e),
                exc_info=True,
            )

        # Step 6: Update state with summary
        # Keep rolling window of last 5 summaries
        previous_summaries = list(get_previous_summaries(state, content_manager))[-4:]
        previous_summaries.append(summary)

        # Get current version (for revision tracking)
        # Note: content_manager already initialized earlier in the function
        current_version = content_manager.get_latest_version("summaries", "all") + 1

        # Externalize previous_chapter_summaries to reduce state bloat
        summaries_ref = content_manager.save_list_of_texts(
            previous_summaries,
            "summaries",
            "all",
            current_version,
        )

        logger.info(
            "summarize_chapter: complete",
            chapter=state.get("current_chapter", 1),
            total_summaries_in_state=len(previous_summaries),
            summaries_externalized=True,
            size=summaries_ref["size_bytes"],
        )

        return {
            "summaries_ref": summaries_ref,
            "current_summary": summary,
            "current_node": "summarize",
        }

    except ChapterSummaryContractError:
        raise
    except Exception as e:
        logger.error(
            "summarize_chapter: exception during summarization",
            chapter=state.get("current_chapter", 1),
            error=str(e),
            exc_info=True,
        )
        # Continue workflow even if summarization fails
        # This is non-critical, so we don't block the pipeline
        return {
            "current_node": "summarize",
        }


def _parse_summary_response(response_text: str) -> str:
    """Parse a chapter summary from an LLM response.

    Contract (see prompt template):
    - Return ONLY valid JSON.
    - Output MUST be a single JSON object with exactly one key: "summary".

    Args:
        response_text: Raw LLM response.

    Returns:
        The summary string.

    Raises:
        ChapterSummaryContractError: When the response is not a JSON object matching the contract.
    """
    parsed, _candidates, _parse_errors = try_load_json_from_response(
        response_text,
        expected_root=(dict,),
    )

    if not isinstance(parsed, dict):
        raise ChapterSummaryContractError("Chapter summary JSON contract violated: could not parse a JSON object from the model response.")

    if set(parsed.keys()) != {"summary"}:
        keys = ", ".join(sorted(str(k) for k in parsed.keys()))
        raise ChapterSummaryContractError('Chapter summary JSON contract violated: expected a single JSON object with exactly one key: "summary". ' f"Found keys: {keys}")

    summary = parsed.get("summary")
    if not isinstance(summary, str):
        raise ChapterSummaryContractError('Chapter summary JSON contract violated: key "summary" must be a string.')

    cleaned = summary.strip()
    if not cleaned:
        raise ChapterSummaryContractError('Chapter summary JSON contract violated: key "summary" must be a non-empty string.')

    return cleaned


async def _save_summary_to_neo4j(
    chapter_number: int,
    summary: str,
) -> None:
    """Persist the chapter summary to Neo4j.

    Args:
        chapter_number: Chapter number to update.
        summary: Summary text to save.

    Notes:
        Errors are logged and swallowed because summarization is a non-critical step.
    """
    # Canonical Chapter persistence: ensure Chapter.id is always present and stable.
    query, parameters = chapter_queries.build_chapter_upsert_statement(
        chapter_number=chapter_number,
        summary=summary,
        # Summary node should not touch embedding/provisional flags; leave them unchanged.
        embedding_vector=None,
        is_provisional=None,
    )

    try:
        await neo4j_manager.execute_write_query(query, parameters)
        logger.info(
            "summarize_chapter: summary saved to Neo4j (canonical chapter upsert)",
            chapter=chapter_number,
            chapter_id=parameters.get("chapter_id_param"),
        )
    except Exception as e:
        logger.error(
            "summarize_chapter: failed to save summary to Neo4j",
            chapter=chapter_number,
            error=str(e),
            exc_info=True,
        )
        # Don't raise - summarization is non-critical


def _write_chapter_summary_file(
    chapter_number: int | None,
    summary_text: str,
    project_dir: str | None,
) -> None:
    """Write a per-chapter Markdown summary file.

    Format:
        summaries/chapter_{chapter_number:03d}.md

    Args:
        chapter_number: Chapter number used in the filename. If missing/invalid, this
            is a no-op.
        summary_text: Summary body text.
        project_dir: Base project directory containing the `summaries/` folder.

    Notes:
        - Overwrites existing files for idempotence.
        - Normalizes literal `"\\n"` sequences into real newlines in the body.
    """
    if not chapter_number or chapter_number <= 0:
        # Graceful no-op: invalid or missing chapter number
        return

    if not project_dir:
        # Graceful no-op: cannot determine target directory
        return

    summaries_dir = Path(project_dir) / "summaries"

    # Normalize summary text: convert literal "\\n" sequences to actual newlines.
    body = str(summary_text)
    if "\\n" in body:
        body = body.replace("\\n", "\n")

    # Build YAML front matter
    generated_at = datetime.now(UTC).isoformat()
    front_matter_lines = [
        "---",
        f"chapter: {int(chapter_number)}",
        f"generated_at: {generated_at}",
        "---",
        "",
    ]
    content = "\n".join(front_matter_lines) + body

    summary_path = summaries_dir / f"chapter_{int(chapter_number):03d}.md"
    write_text_file(summary_path, content)

    logger.info(
        "summarize_chapter: summary file written",
        chapter=int(chapter_number),
        path=str(summary_path),
    )


__all__ = ["summarize_chapter"]
