# core/langgraph/initialization/act_outlines_node.py
"""Generate act-level outlines from the global outline.

This module defines the initialization node that expands the global outline into
structured act outlines. Act chapter ranges are derived from explicit ranges in the
global outline when present, otherwise a balanced fallback allocation is used.
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, ConfigDict, Field, field_validator

import config
from core.langgraph.content_manager import (
    ContentManager,
    get_character_sheets,
    get_global_outline,
    require_project_dir,
)
from core.langgraph.initialization.chapter_allocation import choose_act_ranges
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)


class ActOutlineKeyEventSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    sequence: int = Field(description="1-indexed sequence number")
    event: str = Field(description="On-page, concrete event")
    cause: str = Field(description="What directly causes the event")
    effect: str = Field(description="What directly changes because of the event")


class ActOutlineSectionsSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    act_summary: str
    opening_situation: str
    key_events: list[ActOutlineKeyEventSchema]
    character_development: str
    stakes_and_tension: str
    act_ending_turn: str
    thematic_thread: str
    pacing_notes: str

    @field_validator("key_events")
    @classmethod
    def validate_key_events_length_and_sequence(
        cls,
        key_events: list[ActOutlineKeyEventSchema],
    ) -> list[ActOutlineKeyEventSchema]:
        if not (5 <= len(key_events) <= 7):
            raise ValueError("Act outline sections.key_events must contain 5-7 items")

        expected_sequence_numbers = list(range(1, len(key_events) + 1))
        actual_sequence_numbers = [event.sequence for event in key_events]
        if actual_sequence_numbers != expected_sequence_numbers:
            raise ValueError("Act outline sections.key_events sequence numbers must be contiguous starting at 1. " f"expected={expected_sequence_numbers} actual={actual_sequence_numbers}")

        return key_events


class ActOutlineSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    act_number: int
    total_acts: int
    act_role: str
    chapters_in_act: int
    sections: ActOutlineSectionsSchema


def _synthesize_act_outline_raw_text(outline: ActOutlineSchema) -> str:
    """Synthesize a human-readable act outline from structured sections.

    Args:
        outline: Parsed act outline schema.

    Returns:
        A formatted text representation suitable for display and file persistence.
    """
    sections = outline.sections

    parts: list[str] = [
        f"Act {outline.act_number}: {outline.act_role}",
        "",
        "1. Act Summary (2-3 sentences)",
        sections.act_summary.strip(),
        "",
        "2. Opening Situation",
        sections.opening_situation.strip(),
        "",
        "3. Key Events",
    ]

    for key_event in sections.key_events:
        parts.append(
            "\n".join(
                [
                    f"{key_event.sequence}. {key_event.event.strip()}",
                    f"Cause: {key_event.cause.strip()}",
                    f"Effect: {key_event.effect.strip()}",
                ]
            ).strip()
        )
        parts.append("")

    parts.extend(
        [
            "4. Character Development (focus on internal conflict + choices)",
            sections.character_development.strip(),
            "",
            "5. Stakes and Tension (how pressure escalates)",
            sections.stakes_and_tension.strip(),
            "",
            "6. Act Ending / Turn (cliffhanger or decisive shift)",
            sections.act_ending_turn.strip(),
            "",
            "7. Thematic Thread (how theme appears through action/imagery)",
            sections.thematic_thread.strip(),
            "",
            "8. Pacing Notes (fast vs slow chapters)",
            sections.pacing_notes.strip(),
        ]
    )

    return "\n".join(parts).strip()


async def generate_act_outlines(state: NarrativeState) -> NarrativeState:
    """Generate and externalize act outlines for initialization.

    Args:
        state: Workflow state. Requires a global outline (prefer externalized refs).

    Returns:
        Updated state containing:
        - act_outlines_ref: Externalized act outlines artifact (when generation succeeds).
        - current_node: `"act_outlines"`.
        - initialization_step: Progress marker.

        If the global outline is missing, returns an error update and does not set
        `has_fatal_error`.

    Notes:
        This node performs LLM I/O and writes act outlines to disk via `ContentManager`.
        JSON/schema contract violations for a single act are handled by skipping that
        act outline rather than crashing the entire initialization run.
    """
    logger.info(
        "generate_act_outlines: starting act outline generation",
        title=state.get("title", ""),
    )

    # Initialize content manager for reading externalized content
    content_manager = ContentManager(require_project_dir(state))

    # Get global outline (prefers externalized content, falls back to in-state)
    global_outline = get_global_outline(state, content_manager)

    # Validate inputs
    if not global_outline:
        error_msg = "No global outline available for act outline generation"
        logger.error("generate_act_outlines: missing global outline")
        return {
            "last_error": error_msg,
            "current_node": "act_outlines",
            "initialization_step": "act_outlines_failed",
        }

    act_count_raw = global_outline.get("act_count", 3)
    act_count = act_count_raw if isinstance(act_count_raw, int) and not isinstance(act_count_raw, bool) else 3
    if act_count <= 0:
        act_count = 3

    total_chapters = state.get("total_chapters", 20)
    if not isinstance(total_chapters, int) or isinstance(total_chapters, bool) or total_chapters < 0:
        total_chapters = 20

    # Prefer explicit act ranges from global outline when present; otherwise compute
    # balanced ranges that cover all chapters exactly once and distribute remainder.
    act_ranges = choose_act_ranges(global_outline=global_outline, total_chapters=total_chapters)

    logger.info(
        "generate_act_outlines: generating outlines",
        act_count=act_count,
        total_chapters=total_chapters,
        act_ranges={k: {"chapters_start": v.chapters_start, "chapters_end": v.chapters_end, "chapters_in_act": v.chapters_in_act} for k, v in act_ranges.items()},
    )

    # Generate outline for each act
    act_outlines = {}
    for act_num in range(1, act_count + 1):
        logger.info("generate_act_outlines: generating act", act_number=act_num)

        act_range = act_ranges.get(act_num)
        chapters_in_act = act_range.chapters_in_act if act_range else 0

        try:
            act_outline = await _generate_single_act_outline(
                state=state,
                act_number=act_num,
                total_acts=act_count,
                chapters_in_act=chapters_in_act,
            )
        except ValueError as error:
            logger.warning(
                "generate_act_outlines: act outline JSON/schema contract violated",
                act_number=act_num,
                error=str(error),
            )
            act_outline = None

        if act_outline:
            # Record chapter allocation alongside the outline so downstream logic can
            # introspect without recomputing allocation.
            if act_range:
                act_outline["chapters_start"] = act_range.chapters_start
                act_outline["chapters_end"] = act_range.chapters_end

            act_outlines[act_num] = act_outline
        else:
            logger.warning(
                "generate_act_outlines: failed to generate act",
                act_number=act_num,
            )

    if not act_outlines:
        error_msg = "Failed to generate any act outlines"
        logger.error("generate_act_outlines: no outlines generated")
        return {
            "last_error": error_msg,
            "current_node": "act_outlines",
            "initialization_step": "act_outlines_failed",
        }

    logger.info(
        "generate_act_outlines: generation complete",
        act_count=len(act_outlines),
    )

    # Externalize act_outlines to reduce state bloat.
    #
    # PR4: Use a JSON-safe externalized container format (v2) that is deterministic and
    # does not rely on dict keys that may be coerced to strings during JSON serialization.
    acts_externalized: list[dict[str, object]] = []
    for act_number in sorted(act_outlines.keys()):
        act_outline = act_outlines[act_number]

        # Ensure each entry is self-describing (required for v2 list format).
        if act_outline.get("act_number") != act_number:
            act_outline = {**act_outline, "act_number": act_number}

        acts_externalized.append(act_outline)

    act_outlines_externalized = {
        "format_version": 2,
        "acts": acts_externalized,
    }

    act_outlines_ref = content_manager.save_json(
        act_outlines_externalized,
        "act_outlines",
        "all",
        version=1,
    )

    logger.info(
        "generate_act_outlines: content externalized",
        size=act_outlines_ref["size_bytes"],
    )

    return {
        "act_outlines_ref": act_outlines_ref,
        "current_node": "act_outlines",
        "last_error": None,
        "initialization_step": "act_outlines_complete",
    }


async def _generate_single_act_outline(
    state: NarrativeState,
    act_number: int,
    total_acts: int,
    chapters_in_act: int,
) -> dict[str, object] | None:
    """
    Generate an outline for a single act.

    Args:
        state: Current narrative state
        act_number: Act number to generate (1-indexed)
        total_acts: Total number of acts in the story
        chapters_in_act: Approximate number of chapters in this act

    Returns:
        Dictionary containing act outline details or None on failure
    """
    # Initialize content manager for reading externalized content
    content_manager = ContentManager(require_project_dir(state))

    # Get global outline and character sheets
    global_outline = get_global_outline(state, content_manager) or {}
    character_sheets = get_character_sheets(state, content_manager)

    # Determine act role in story structure
    act_role = _get_act_role(act_number, total_acts)

    # Build context strings
    character_context = _build_character_summary(character_sheets)
    global_outline_text = global_outline.get("raw_text", "")

    prompt = render_prompt(
        "initialization/generate_act_outline.j2",
        {
            "title": state.get("title", ""),
            "genre": state.get("genre", ""),
            "theme": state.get("theme", ""),
            "setting": state.get("setting", ""),
            "act_number": act_number,
            "total_acts": total_acts,
            "act_role": act_role,
            "chapters_in_act": chapters_in_act,
            "global_outline": global_outline_text,  # Truncate if too long
            "character_context": character_context,
            "protagonist_name": state.get("protagonist_name", ""),
        },
    )

    try:
        data, usage = await llm_service.async_call_llm_json_object(
            model_name=state.get("large_model", config.LARGE_MODEL),
            prompt=prompt,
            temperature=0.7,
            max_tokens=config.MAX_GENERATION_TOKENS,
            allow_fallback=True,
            auto_clean_response=True,
            system_prompt=get_system_prompt("initialization"),
            max_attempts=2,
        )

        outline = ActOutlineSchema.model_validate(data)

        if outline.act_number != act_number:
            raise ValueError(f"Act outline act_number mismatch. expected={act_number} actual={outline.act_number}")
        if outline.total_acts != total_acts:
            raise ValueError(f"Act outline total_acts mismatch. expected={total_acts} actual={outline.total_acts}")
        if outline.act_role != act_role:
            raise ValueError(f"Act outline act_role mismatch. expected={act_role!r} actual={outline.act_role!r}")
        if outline.chapters_in_act != chapters_in_act:
            raise ValueError(f"Act outline chapters_in_act mismatch. expected={chapters_in_act} actual={outline.chapters_in_act}")

        raw_text = _synthesize_act_outline_raw_text(outline)

        act_outline = {
            "act_number": outline.act_number,
            "total_acts": outline.total_acts,
            "act_role": outline.act_role,
            "chapters_in_act": outline.chapters_in_act,
            "sections": outline.sections.model_dump(),
            "raw_text": raw_text,
            "generated_at": "initialization",
        }

        logger.debug(
            "_generate_single_act_outline: act generated",
            act_number=act_number,
            length=len(raw_text),
        )

        return act_outline

    except ValueError:
        raise
    except Exception as e:
        logger.error(
            "_generate_single_act_outline: exception during generation",
            act_number=act_number,
            error=str(e),
            exc_info=True,
        )
        return None


def _get_act_role(act_number: int, total_acts: int) -> str:
    """
    Determine the narrative role of an act.

    Args:
        act_number: Act number (1-indexed)
        total_acts: Total number of acts

    Returns:
        String describing the act's role in the story structure
    """
    if act_number == 1:
        return "Setup/Introduction"
    elif act_number == total_acts:
        return "Resolution/Climax"
    elif total_acts == 3 and act_number == 2:
        return "Confrontation/Rising Action"
    elif total_acts == 5:
        if act_number == 2:
            return "Rising Action"
        elif act_number == 3:
            return "Midpoint/Crisis"
        elif act_number == 4:
            return "Falling Action"
        return "Development"
    else:
        return "Development"


def _build_character_summary(character_sheets: dict[str, dict]) -> str:
    """
    Build a concise summary of characters for act outline generation.

    Args:
        character_sheets: Dictionary of character sheets

    Returns:
        Formatted string summarizing characters
    """
    if not character_sheets:
        return "No characters defined."

    summaries = []
    for name, sheet in character_sheets.items():
        is_protag = sheet.get("is_protagonist", False)
        role = "Protagonist" if is_protag else "Character"
        # Get first sentence or 100 chars of description
        desc = sheet.get("description", "")
        summaries.append(f"- **{name}** ({role}): {desc}")

    return "\n".join(summaries)


__all__ = ["generate_act_outlines"]
