# core/langgraph/initialization/global_outline_node.py
"""Generate the global story outline during initialization.

This module defines the initialization node responsible for creating the
high-level narrative plan (acts, major turning points, and character arcs). The
resulting outline is externalized to keep workflow state small.
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic import BaseModel, Field

import config
from core.langgraph.content_manager import ContentManager, get_character_sheets
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from prompts.prompt_renderer import get_system_prompt, render_prompt
from utils.common import try_load_json_from_response

logger = structlog.get_logger(__name__)


class ActOutline(BaseModel):
    """Structured outline for a single act."""

    act_number: int = Field(description="Act number (1-5)")
    title: str = Field(description="Title or name of the act")
    summary: str = Field(description="Brief summary of the act (1-2 sentences)")
    key_events: list[str] = Field(default_factory=list, description="List of major events in this act")
    chapters_start: int = Field(description="First chapter number in this act")
    chapters_end: int = Field(description="Last chapter number in this act")


class CharacterArc(BaseModel):
    """Character arc progression throughout the story."""

    character_name: str = Field(description="Name of the character")
    starting_state: str = Field(description="Character's state at story start")
    ending_state: str = Field(description="Character's state at story end")
    key_moments: list[str] = Field(default_factory=list, description="Key transformation moments")


class GlobalOutlineSchema(BaseModel):
    """Structured global story outline."""

    act_count: int = Field(description="Number of acts (3 or 5)")
    acts: list[ActOutline] = Field(description="List of act outlines")
    inciting_incident: str = Field(description="The event that starts the story")
    midpoint: str = Field(description="Major midpoint event or revelation")
    climax: str = Field(description="The story's climax")
    resolution: str = Field(description="How the story resolves")
    character_arcs: list[CharacterArc] = Field(default_factory=list, description="Character arc progressions")
    thematic_progression: str = Field(description="How the theme develops throughout the story")
    pacing_notes: str = Field(default="", description="Notes on pacing and structure")


async def generate_global_outline(state: NarrativeState) -> NarrativeState:
    """Generate and externalize a global story outline.

    Args:
        state: Workflow state. Reads character sheets (preferring externalized refs)
            and story metadata (title, genre, theme, setting, total chapters).

    Returns:
        Updated state containing:
        - global_outline_ref: Externalized outline artifact.
        - initialization_step: `"global_outline_complete"` on success.
        - current_node: `"global_outline"`.

        If outline generation fails, returns an error update without setting
        `has_fatal_error`.

    Notes:
        This node performs LLM I/O and writes the outline to disk via
        `ContentManager.save_json()`.
    """
    logger.info(
        "generate_global_outline: starting outline generation",
        title=state.get("title", ""),
        total_chapters=state.get("total_chapters", 0),
    )

    # Initialize content manager for reading externalized content
    content_manager = ContentManager(state.get("project_dir", ""))

    # Get character sheets (prefers externalized content, falls back to in-state)
    character_sheets = get_character_sheets(state, content_manager)

    # Validate inputs
    if not character_sheets:
        logger.warning("generate_global_outline: no character sheets available, " "generating without character context")

    # Build character context for outline generation
    character_context = _build_character_context_from_sheets(character_sheets)

    # Step 1: Generate global outline
    prompt = render_prompt(
        "initialization/generate_global_outline.j2",
        {
            "title": state.get("title", ""),
            "genre": state.get("genre", ""),
            "theme": state.get("theme", ""),
            "setting": state.get("setting", ""),
            "target_word_count": state.get("target_word_count", 80000),
            "total_chapters": state.get("total_chapters", 20),
            "protagonist_name": state.get("protagonist_name", ""),
            "character_context": character_context,
            "character_names": list(character_sheets.keys()),
        },
    )

    try:
        response, usage = await llm_service.async_call_llm(
            model_name=state.get("large_model", config.LARGE_MODEL),
            prompt=prompt,
            temperature=0.7,
            max_tokens=16384,
            allow_fallback=True,
            auto_clean_response=True,
            system_prompt=get_system_prompt("initialization"),
        )

        if not response or not response.strip():
            error_msg = "LLM returned empty global outline"
            logger.error("generate_global_outline: empty response")
            return {
                **state,
                "last_error": error_msg,
                "current_node": "global_outline",
                "initialization_step": "global_outline_failed",
            }

        # Parse outline structure
        global_outline = _parse_global_outline(response, state)

        logger.info(
            "generate_global_outline: generation complete",
            outline_length=len(response),
            acts=global_outline.get("act_count", 0),
        )

        # Externalize global_outline to reduce state bloat
        global_outline_ref = content_manager.save_json(
            global_outline,
            "global_outline",
            "main",
            version=1,
        )

        logger.info(
            "generate_global_outline: content externalized",
            size=global_outline_ref["size_bytes"],
        )

        return {
            **state,
            "global_outline_ref": global_outline_ref,
            "current_node": "global_outline",
            "last_error": None,
            "initialization_step": "global_outline_complete",
        }

    except Exception as e:
        error_msg = f"Error generating global outline: {str(e)}"
        logger.error(
            "generate_global_outline: exception during generation",
            error=str(e),
            exc_info=True,
        )
        return {
            **state,
            "last_error": error_msg,
            "current_node": "global_outline",
            "initialization_step": "global_outline_failed",
        }


def _build_character_context_from_sheets(character_sheets: dict) -> str:
    """Build a concise character context block for outline generation.

    Args:
        character_sheets: Character sheets mapping.

    Returns:
        Short, human-readable context string summarizing each character.
    """
    if not character_sheets:
        return "No characters defined yet."

    context_parts = []
    for name, sheet in character_sheets.items():
        is_protag = sheet.get("is_protagonist", False)
        role = "Protagonist" if is_protag else "Main Character"
        # Truncate description to first 200 chars for context
        desc = sheet.get("description", "")
        context_parts.append(f"**{name}** ({role}): {desc}...")

    return "\n\n".join(context_parts)


def _validate_chapter_allocations(outline: GlobalOutlineSchema, total_chapters: int) -> list[str]:
    """Validate chapter range coverage and act numbering.

    Args:
        outline: Parsed outline schema.
        total_chapters: Total number of chapters expected.

    Returns:
        Validation error messages. An empty list indicates the allocations are valid.
    """
    errors = []

    if not outline.acts:
        errors.append("No acts defined in outline")
        return errors

    # Check chapter coverage
    all_chapters = set()
    for act in outline.acts:
        for chapter in range(act.chapters_start, act.chapters_end + 1):
            if chapter in all_chapters:
                errors.append(f"Chapter {chapter} is assigned to multiple acts")
            all_chapters.add(chapter)

    # Check for gaps
    expected_chapters = set(range(1, total_chapters + 1))
    missing = expected_chapters - all_chapters
    extra = all_chapters - expected_chapters

    if missing:
        errors.append(f"Missing chapter allocations: {sorted(missing)}")
    if extra:
        errors.append(f"Extra chapter allocations beyond total: {sorted(extra)}")

    # Check sequential act numbering
    act_numbers = [act.act_number for act in outline.acts]
    expected_numbers = list(range(1, outline.act_count + 1))
    if sorted(act_numbers) != expected_numbers:
        errors.append(f"Act numbers should be {expected_numbers}, got {act_numbers}")

    return errors


def _extract_json_from_response(response: str) -> str:
    """Extract a JSON payload from an LLM response.

    This is a lightweight compatibility helper for tests that expect a raw JSON
    string, optionally wrapped in markdown code fences or surrounded by other
    text.

    Args:
        response: Raw LLM response text.

    Returns:
        JSON string content, without code fences and without surrounding text when
        possible.

    Raises:
        ValueError: When no JSON object boundaries can be detected.
    """
    if "```" in response:
        lines = response.strip().splitlines()
        if not lines:
            raise ValueError("Response is empty")

        if lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].lstrip().startswith("```"):
            lines = lines[:-1]

        extracted = "\n".join(lines).strip()
        if extracted:
            return extracted

    stripped = response.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        if response == stripped:
            return response
        return stripped

    start_index = response.find("{")
    end_index = response.rfind("}")
    if start_index == -1 or end_index == -1 or end_index <= start_index:
        raise ValueError("No JSON object found in response")

    return response[start_index : end_index + 1].strip()


def _parse_global_outline(response: str, state: NarrativeState) -> dict[str, Any]:
    """Parse and validate a global outline response.

    Args:
        response: LLM response expected to contain a JSON object.
        state: Workflow state used for validation context (for example, total chapters).

    Returns:
        Parsed outline data as a JSON-serializable dictionary.

    Notes:
        This function validates structure via Pydantic and records any detected
        validation issues in the returned outline payload.
    """
    total_chapters = state.get("total_chapters", 20)

    parsed, candidates, parse_errors = try_load_json_from_response(
        response,
        expected_root=dict,
    )
    if parsed is None:
        logger.warning(
            "_parse_global_outline: JSON parsing failed, using fallback",
            tried_sources=[source for source, _candidate in candidates[:5]],
            errors=parse_errors[:5],
        )
        return _fallback_parse_outline(response, state)

    try:
        # Validate with Pydantic schema
        outline = GlobalOutlineSchema.model_validate(parsed)

        # Validate chapter allocations
        validation_errors = _validate_chapter_allocations(outline, total_chapters)
        if validation_errors:
            logger.warning(
                "_parse_global_outline: validation issues",
                errors=validation_errors,
            )

        # Convert to dictionary for state storage
        global_outline = {
            "act_count": outline.act_count,
            "acts": [act.model_dump() for act in outline.acts],
            "inciting_incident": outline.inciting_incident,
            "midpoint": outline.midpoint,
            "climax": outline.climax,
            "resolution": outline.resolution,
            "character_arcs": [arc.model_dump() for arc in outline.character_arcs],
            "thematic_progression": outline.thematic_progression,
            "pacing_notes": outline.pacing_notes,
            "total_chapters": total_chapters,
            "structure_type": f"{outline.act_count}-act",
            "generated_at": "initialization",
            "validation_errors": validation_errors,
            "raw_text": response,  # Keep for reference
        }

        logger.info(
            "_parse_global_outline: successfully parsed structured outline",
            act_count=outline.act_count,
            num_character_arcs=len(outline.character_arcs),
        )

        return global_outline

    except Exception as e:
        logger.warning(
            "_parse_global_outline: schema validation failed, using fallback",
            error=str(e),
        )
        return _fallback_parse_outline(response, state)


def _fallback_parse_outline(response: str, state: NarrativeState) -> dict[str, Any]:
    """
    Fallback parser for when JSON parsing fails.

    Uses pattern matching to extract what structure we can from free-form text.

    Args:
        response: LLM-generated outline text
        state: Current narrative state

    Returns:
        Dictionary containing basic outline data
    """
    # Try to detect act structure from the response
    act_count = 3  # Default three-act structure
    if "act 1" in response.lower() or "act i" in response.lower():
        if "act 5" in response.lower() or "act v" in response.lower():
            act_count = 5
        elif "act 4" in response.lower() or "act iv" in response.lower():
            act_count = 4

    global_outline = {
        "raw_text": response,
        "act_count": act_count,
        "acts": [],  # Empty, will need manual parsing
        "inciting_incident": "",
        "midpoint": "",
        "climax": "",
        "resolution": "",
        "character_arcs": [],
        "thematic_progression": "",
        "pacing_notes": "",
        "total_chapters": state.get("total_chapters", 20),
        "structure_type": f"{act_count}-act",
        "generated_at": "initialization",
        "validation_errors": ["Fallback parsing used - JSON parsing failed"],
    }

    logger.debug(
        "_fallback_parse_outline: used fallback parsing",
        act_count=act_count,
        length=len(response),
    )

    return global_outline


__all__ = ["generate_global_outline"]
