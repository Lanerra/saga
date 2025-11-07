"""
Summarization node for LangGraph workflow.

This module contains the chapter summarization logic for the LangGraph-based
narrative generation workflow.

Migration Reference: docs/phase2_migration_plan.md - Step 2.3

New Functionality:
- Not in SAGA 1.0, but needed for long context management
- Generates concise summaries for use in future chapter context
"""

from __future__ import annotations

import json

import structlog

from core.db_manager import neo4j_manager
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)


async def summarize_chapter(state: NarrativeState) -> NarrativeState:
    """
    Generate chapter summary for context in future chapters.

    This is the main LangGraph node function for chapter summarization.
    It takes the final chapter text, generates a concise 2-3 sentence summary,
    persists it to Neo4j, and updates the state's summary history.

    NEW FUNCTIONALITY (not in SAGA 1.0)

    Process Flow:
    1. Validate draft text exists
    2. Generate summary using fast extraction model
    3. Parse summary from LLM response
    4. Persist summary to Neo4j Chapter node
    5. Update state with summary added to rolling window

    Args:
        state: Current narrative state containing final draft_text

    Returns:
        Updated state with summary added to previous_chapter_summaries
    """
    logger.info(
        "summarize_chapter: starting summarization",
        chapter=state["current_chapter"],
    )

    # Validate we have text to summarize
    if not state.get("draft_text"):
        logger.warning("summarize_chapter: no draft text to summarize")
        return {
            **state,
            "current_node": "summarize",
        }

    # Step 1: Build summary prompt
    prompt = render_prompt(
        "knowledge_agent/chapter_summary.j2",
        {
            "chapter_number": state["current_chapter"],
            "chapter_text": state["draft_text"],
        },
    )

    # Step 2: Generate summary using fast extraction model
    logger.info(
        "summarize_chapter: calling LLM for summary",
        chapter=state["current_chapter"],
        model=state["extraction_model"],
    )

    try:
        summary_text, usage = await llm_service.async_call_llm(
            model_name=state["extraction_model"],  # Use fast model
            prompt=prompt,
            temperature=0.3,  # Low temperature for consistency
            max_tokens=200,  # Short summary
            allow_fallback=True,
            stream_to_disk=False,
            auto_clean_response=True,
            system_prompt=get_system_prompt("knowledge_agent"),
        )

        if not summary_text or not summary_text.strip():
            logger.error(
                "summarize_chapter: LLM returned empty summary",
                chapter=state["current_chapter"],
            )
            # Continue without summary rather than fail
            return {
                **state,
                "current_node": "summarize",
            }

        # Step 3: Parse summary from response
        summary = _parse_summary_response(summary_text)

        if not summary:
            logger.warning(
                "summarize_chapter: failed to parse summary, using raw text",
                chapter=state["current_chapter"],
                raw_text=summary_text[:100],
            )
            # Use first few sentences of raw text as fallback
            summary = " ".join(summary_text.strip().split(".")[:3]) + "."

        logger.info(
            "summarize_chapter: summary generated",
            chapter=state["current_chapter"],
            summary_length=len(summary),
        )

        # Step 4: Persist to Neo4j
        await _save_summary_to_neo4j(
            chapter_number=state["current_chapter"],
            summary=summary,
        )

        # Step 5: Update state with summary
        # Keep rolling window of last 5 summaries
        previous_summaries = list(state.get("previous_chapter_summaries", []))[-4:]
        previous_summaries.append(summary)

        logger.info(
            "summarize_chapter: complete",
            chapter=state["current_chapter"],
            total_summaries_in_state=len(previous_summaries),
        )

        return {
            **state,
            "previous_chapter_summaries": previous_summaries,
            "current_node": "summarize",
        }

    except Exception as e:
        logger.error(
            "summarize_chapter: exception during summarization",
            chapter=state["current_chapter"],
            error=str(e),
            exc_info=True,
        )
        # Continue workflow even if summarization fails
        # This is non-critical, so we don't block the pipeline
        return {
            **state,
            "current_node": "summarize",
        }


def _parse_summary_response(response_text: str) -> str | None:
    """
    Parse summary from LLM JSON response.

    The template expects a JSON object with a "summary" key.
    Falls back to treating the entire response as the summary.

    Args:
        response_text: Raw LLM response text

    Returns:
        Extracted summary string, or None if parsing fails
    """
    # Try to parse as JSON first
    try:
        # Clean common JSON wrapper issues
        cleaned = response_text.strip()

        # Remove markdown code blocks if present
        if cleaned.startswith("```"):
            # Find the first newline after ```
            start = cleaned.find("\n")
            if start != -1:
                # Find the last ```
                end = cleaned.rfind("```")
                if end != -1:
                    cleaned = cleaned[start + 1 : end].strip()

        data = json.loads(cleaned)
        if isinstance(data, dict) and "summary" in data:
            return data["summary"].strip()
        elif isinstance(data, str):
            # Sometimes LLM returns just a string
            return data.strip()
    except json.JSONDecodeError:
        # Not valid JSON, treat entire response as summary
        pass

    # Fallback: return the response as-is if it looks reasonable
    cleaned = response_text.strip()
    if cleaned and len(cleaned) > 10:  # Minimum reasonable summary length
        return cleaned

    return None


async def _save_summary_to_neo4j(
    chapter_number: int,
    summary: str,
) -> None:
    """
    Save chapter summary to Neo4j.

    Updates the Chapter node with the summary text.
    Creates the Chapter node if it doesn't exist.

    Args:
        chapter_number: Chapter number to update
        summary: Summary text to save
    """
    query = """
    MERGE (c:Chapter {number: $chapter_number})
    SET c.summary = $summary,
        c.last_updated = timestamp()
    """

    parameters = {
        "chapter_number": chapter_number,
        "summary": summary,
    }

    try:
        await neo4j_manager.execute_write_query(query, parameters)
        logger.info(
            "summarize_chapter: summary saved to Neo4j",
            chapter=chapter_number,
        )
    except Exception as e:
        logger.error(
            "summarize_chapter: failed to save summary to Neo4j",
            chapter=chapter_number,
            error=str(e),
            exc_info=True,
        )
        # Don't raise - summarization is non-critical


__all__ = ["summarize_chapter"]
