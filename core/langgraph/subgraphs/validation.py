# core/langgraph/subgraphs/validation.py
"""
Build the validation subgraph for SAGA's LangGraph workflow.

Migration Reference: docs/langgraph-architecture.md - Section 3.4

This subgraph runs a sequence of checks over a chapter draft and extracted
signals:
- Consistency validation (graph- and heuristic-based).
- LLM-based prose quality evaluation.
- Additional contradiction detection (relationship evolution).

Notes:
    These nodes are async and may perform I/O (Neo4j queries, LLM calls, and
    filesystem reads via externalized content refs).
"""

from __future__ import annotations

from typing import Any, cast

import structlog
from langgraph.graph import END, StateGraph  # type: ignore[import-not-found, attr-defined]

import config
from core.langgraph.content_manager import (
    ContentManager,
    get_chapter_outlines,
    get_draft_text,
    get_previous_summaries,
    get_scene_drafts,
    require_project_dir,
)
from core.langgraph.nodes.validation_node import (
    validate_consistency as original_validate_consistency,
)
from core.langgraph.quality_policy import SCORE_FIELDS, policy_for, record_check
from core.langgraph.state import Contradiction, NarrativeState
from core.langgraph.subgraphs._shared import _should_continue_or_error
from core.service_context import get_services
from data_access.validation_queries import fetch_prior_accepted_facts, get_candidate_relationship_assertions
from prompts.prompt_renderer import get_system_prompt, render_prompt
from utils.common import try_load_json_from_response

logger = structlog.get_logger(__name__)


async def validate_consistency(state: NarrativeState) -> NarrativeState:
    """Validate chapter consistency against graph-derived constraints.

    This is a thin wrapper around
    [`validate_consistency()`](core/langgraph/nodes/validation_node.py:149) to
    keep the validation subgraph as the canonical composition point.

    Args:
        state: Workflow state.

    Returns:
        Updated state containing any detected contradictions and the derived
        `needs_revision` flag.
    """
    logger.info("validate_consistency: checking graph constraints")
    return await original_validate_consistency(state)


async def evaluate_quality(state: NarrativeState) -> NarrativeState:
    """Evaluate chapter prose quality using an LLM and record scores in state.

    Args:
        state: Workflow state. Reads `draft_ref` via
            [`get_draft_text()`](core/langgraph/content_manager.py:637) and uses
            metadata such as `genre`/`theme`/`current_chapter`.

    Returns:
        Updated state with quality score fields and `quality_feedback`. When the
        average quality is below threshold, a `quality_issue` contradiction is
        appended.

    Notes:
        This function performs LLM I/O and may be slow relative to purely local
        validation. Failures degrade gracefully by returning `None` scores and a
        descriptive `quality_feedback` message.
    """
    logger.info(
        "evaluate_quality: analyzing prose quality",
        chapter=state.get("current_chapter"),
        word_count=state.get("draft_word_count", 0),
    )

    content_manager = ContentManager(require_project_dir(state))

    from core.exceptions import MissingDraftReferenceError

    try:
        draft_text = get_draft_text(state, content_manager)
    except MissingDraftReferenceError:
        draft_text = ""

    if not draft_text:
        logger.warning("evaluate_quality: no draft text to evaluate")
        return {
            "coherence_score": None,
            "prose_quality_score": None,
            "plot_advancement_score": None,
            "pacing_score": None,
            "tone_consistency_score": None,
            "quality_feedback": "No draft text available for evaluation",
            "quality_checks": record_check(state, "evaluation", "failed", "No draft text available for evaluation"),
        }

    # Build evaluation prompt
    evaluation_prompt = _build_quality_evaluation_prompt(
        draft_text=draft_text,
        chapter_number=state.get("current_chapter", 1),
        genre=state.get("genre", ""),
        theme=state.get("theme", ""),
        previous_summaries=get_previous_summaries(state, content_manager),
        chapter_outline=get_chapter_outlines(state, content_manager).get(state.get("current_chapter", 1), {}),
    )

    try:
        # Call LLM for quality evaluation
        model_name = state.get("medium_model", config.MEDIUM_MODEL)

        response, usage = await get_services().language_model.async_call_llm(
            model_name=model_name,
            prompt=evaluation_prompt,
            temperature=0.1,
            max_tokens=config.MAX_GENERATION_TOKENS,
            auto_clean_response=True,
            system_prompt=get_system_prompt("validation_agent"),
        )

        # Parse the evaluation response
        scores = _parse_quality_scores(response)

        logger.info(
            "evaluate_quality: evaluation complete",
            chapter=state.get("current_chapter"),
            coherence=scores.get("coherence_score"),
            prose_quality=scores.get("prose_quality_score"),
            plot_advancement=scores.get("plot_advancement_score"),
            pacing=scores.get("pacing_score"),
            tone=scores.get("tone_consistency_score"),
        )

        min_quality_threshold = policy_for(state).minimum_score if state.get("quality_policy") else config.MIN_QUALITY_THRESHOLD
        quality_scores = [
            scores.get("coherence_score", 0.0),
            scores.get("prose_quality_score", 0.0),
            scores.get("plot_advancement_score", 0.0),
        ]

        avg_quality = sum(quality_scores) / len(quality_scores)

        # Add quality-based revision trigger
        current_contradictions = state.get("contradictions", [])
        if avg_quality < min_quality_threshold:
            current_contradictions = [
                *current_contradictions,
                Contradiction(
                    type="quality_issue",
                    description=f"Overall quality score ({avg_quality:.2f}) below threshold ({min_quality_threshold})",
                    conflicting_chapters=[state.get("current_chapter", 1)],
                    severity="major",
                    suggested_fix=scores.get("feedback", "Improve prose quality and coherence"),
                ),
            ]

        return {
            "coherence_score": scores.get("coherence_score"),
            "prose_quality_score": scores.get("prose_quality_score"),
            "plot_advancement_score": scores.get("plot_advancement_score"),
            "pacing_score": scores.get("pacing_score"),
            "tone_consistency_score": scores.get("tone_consistency_score"),
            "quality_feedback": scores.get("feedback"),
            "contradictions": current_contradictions,
            "quality_checks": record_check(state, "evaluation", "completed", details={"scores": {name: scores[name] for name in SCORE_FIELDS}, "evaluated_characters": min(len(draft_text), 8000), "draft_characters": len(draft_text)}),
        }

    except Exception as e:
        logger.error(
            "evaluate_quality: error during evaluation",
            error=str(e),
            exc_info=True,
        )
        # Return state with no scores on error
        return {
            "coherence_score": None,
            "prose_quality_score": None,
            "plot_advancement_score": None,
            "pacing_score": None,
            "tone_consistency_score": None,
            "quality_feedback": f"Evaluation failed: {str(e)}",
            "quality_checks": record_check(state, "evaluation", "failed", str(e)),
        }


def _build_quality_evaluation_prompt(
    draft_text: str,
    chapter_number: int,
    genre: str,
    theme: str,
    previous_summaries: list[str],
    chapter_outline: dict[str, Any],
) -> str:
    """Build the prompt for LLM-based quality evaluation.

    Args:
        draft_text: Chapter text to evaluate (may be truncated for context limits).
        chapter_number: Chapter number being evaluated.
        genre: Novel genre.
        theme: Novel theme.
        previous_summaries: Summaries used as continuity context.
        chapter_outline: Outline for the chapter being evaluated.

    Returns:
        Rendered evaluation prompt.
    """
    # Truncate draft if too long (keep first and last parts)
    max_text_length = 8000
    if len(draft_text) > max_text_length:
        half_length = max_text_length // 2
        draft_text = draft_text[:half_length] + "\n\n[... middle section truncated for evaluation ...]\n\n" + draft_text[-half_length:]

    # Format previous summaries
    summary_context = ""
    if previous_summaries:
        recent_summaries = previous_summaries[-3:]  # Last 3 chapters
        summary_context = "\n".join([f"Chapter {chapter_number - len(recent_summaries) + i}: {s}" for i, s in enumerate(recent_summaries)])

    # Format chapter outline
    outline_context = ""
    if chapter_outline:
        outline_context = f"""
Scene Description: {chapter_outline.get('scene_description', 'N/A')}
Key Beats: {', '.join(chapter_outline.get('key_beats', ['N/A']))}
Plot Point: {chapter_outline.get('plot_point', 'N/A')}
"""

    return render_prompt(
        "validation_agent/evaluate_quality.j2",
        {
            "draft_text": draft_text,
            "chapter_number": chapter_number,
            "genre": genre,
            "theme": theme,
            "summary_context": summary_context,
            "outline_context": outline_context,
        },
    )


def _parse_quality_scores(response: str) -> dict[str, Any]:
    """Parse an evaluation payload from an LLM response.

    Args:
        response: Raw LLM response text.

    Returns:
        All five score fields in the range [0.0, 1.0] plus a `feedback` string.

    Raises:
        ValueError: Missing, malformed, boolean or out-of-range scores. Failed
            parsing never invents a passing score or certifies completion.
    """
    parsed, _candidates, _parse_errors = try_load_json_from_response(
        response,
        expected_root=dict,
    )
    if not isinstance(parsed, dict) or set(parsed) != {*SCORE_FIELDS, "feedback"}:
        raise ValueError("Quality evaluation requires all five scores and feedback")
    if any(type(parsed[name]) not in (int, float) or not 0 <= parsed[name] <= 1 for name in SCORE_FIELDS):
        raise ValueError("Quality evaluation scores must be finite numbers between zero and one")
    if not isinstance(parsed["feedback"], str):
        raise ValueError("Quality evaluation feedback must be text")
    return parsed


async def _fetch_validation_data(state: NarrativeState | int) -> dict[str, Any]:
    """Fetch verified snapshots; chapter-only legacy invocations fail closed."""
    if isinstance(state, int):
        raise ValueError("Prior canon requires a verified NarrativeState, not a chapter number")
    return await fetch_prior_accepted_facts(state)


def _check_scene_duplication(state: NarrativeState, content_manager: ContentManager) -> list[Contradiction]:
    """Check for duplicate or highly similar scenes in the chapter draft.

    Args:
        state: Workflow state containing scene_drafts_ref.
        content_manager: Content manager for loading scene drafts.

    Returns:
        List of contradictions flagging duplicate scenes.
    """
    scene_drafts = get_scene_drafts(state, content_manager)

    if len(scene_drafts) < 2:
        return []

    duplicates = []

    for i in range(len(scene_drafts)):
        for j in range(i + 1, len(scene_drafts)):
            scene_i = scene_drafts[i]
            scene_j = scene_drafts[j]

            sample_length = min(800, len(scene_i), len(scene_j))
            sample_i = scene_i[:sample_length].lower().strip()
            sample_j = scene_j[:sample_length].lower().strip()

            if len(sample_i) < 100 or len(sample_j) < 100:
                continue

            similarity = _calculate_text_similarity(sample_i, sample_j)

            if similarity > 0.7:
                duplicates.append(
                    Contradiction(
                        type="scene_duplication",
                        description=f"Scene {i + 1} and Scene {j + 1} are highly similar ({int(similarity * 100)}% match). Each scene should be distinct with unique content and purpose.",
                        conflicting_chapters=[state.get("current_chapter", 1)],
                        severity="critical",
                        suggested_fix=f"Rewrite Scene {j + 1} to ensure it covers different content, POV, or plot beats than Scene {i + 1}.",
                    )
                )

                logger.warning(
                    "Scene duplication detected",
                    scene_i_index=i,
                    scene_j_index=j,
                    similarity=round(similarity, 2),
                    sample_i_start=sample_i[:100],
                    sample_j_start=sample_j[:100],
                )

    return duplicates


def _calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple character-level similarity between two texts.

    Uses a basic approach: count common bigrams and trigrams.

    Args:
        text1: First text sample.
        text2: Second text sample.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if not text1 or not text2:
        return 0.0

    def get_ngrams(text: str, n: int) -> set[str]:
        return set(text[i : i + n] for i in range(len(text) - n + 1))

    bigrams1 = get_ngrams(text1, 2)
    bigrams2 = get_ngrams(text2, 2)

    trigrams1 = get_ngrams(text1, 3)
    trigrams2 = get_ngrams(text2, 3)

    if not bigrams1 or not bigrams2 or not trigrams1 or not trigrams2:
        return 0.0

    bigram_overlap = len(bigrams1 & bigrams2) / max(len(bigrams1), len(bigrams2))
    trigram_overlap = len(trigrams1 & trigrams2) / max(len(trigrams1), len(trigrams2))

    return (bigram_overlap + trigram_overlap) / 2


async def detect_contradictions(state: NarrativeState) -> NarrativeState:
    """Detect additional narrative contradictions and update revision decision.

    This step augments the contradictions produced by
    [`validate_consistency()`](core/langgraph/subgraphs/validation.py:49) with
    additional checks (relationship evolution, scene duplication).

    Args:
        state: Workflow state.

    Returns:
        Updated state with an extended `contradictions` list and a recalculated
        `needs_revision` flag.
    """
    logger.info(
        "detect_contradictions: checking for narrative contradictions",
        chapter=state.get("current_chapter"),
    )

    contradictions = list(state.get("contradictions", []))
    current_chapter = state.get("current_chapter", 1)

    content_manager = ContentManager(require_project_dir(state))

    validation_data = await _fetch_validation_data(state)

    extracted_relationships = get_candidate_relationship_assertions(state, content_manager)
    relationship_issues = await _check_relationship_evolution(
        extracted_relationships,
        current_chapter,
        validation_data.get("relationships", {}),
    )
    contradictions = [
        *contradictions,
        *relationship_issues,
    ]

    scene_duplication_issues = _check_scene_duplication(state, content_manager)
    contradictions = [
        *contradictions,
        *scene_duplication_issues,
    ]

    logger.info(
        "detect_contradictions: contradiction detection complete",
        chapter=current_chapter,
        total_contradictions=len(contradictions),
        new_contradictions=len(contradictions) - len(state.get("contradictions", [])),
    )

    critical_issues = [c for c in contradictions if c.severity == "critical"]
    major_issues = [c for c in contradictions if c.severity == "major"]
    has_issues = len(critical_issues) > 0 or len(major_issues) > 0
    force_continue = state.get("force_continue", False)

    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    if iteration_count >= max_iterations and has_issues and not force_continue:
        logger.warning(
            "detect_contradictions: max iterations reached, accepting best-effort draft",
            iteration_count=iteration_count,
            max_iterations=max_iterations,
            critical_issues=len(critical_issues),
            major_issues=len(major_issues),
        )
        return {
            "needs_revision": False,
            "contradictions": contradictions,
            "current_node": "detect_contradictions",
            "quality_checks": record_check(state, "contradictions", "completed", details={"findings": [item.model_dump(mode="json") for item in contradictions]}),
        }

    needs_revision = has_issues and not force_continue

    return {
        "contradictions": contradictions,
        "needs_revision": needs_revision,
        "current_node": "detect_contradictions",
        "quality_checks": record_check(state, "contradictions", "completed", details={"findings": [item.model_dump(mode="json") for item in contradictions]}),
    }


async def _check_relationship_evolution(
    extracted_relationships: list[Any],
    current_chapter: int,
    existing_relationships: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[Contradiction]:
    """Flag abrupt relationship shifts that may require narrative development.

    Args:
        extracted_relationships: Relationships extracted from the current chapter.
        current_chapter: Chapter number being validated.
        existing_relationships: Verified prior accepted assertion snapshots.

    Returns:
        Informational contradictions (typically `minor`) for abrupt transitions.

    """
    if not extracted_relationships:
        return []

    contradictions = []

    # Define relationship transitions that require development
    requires_development = {
        ("HATES", "LOVES"): "dramatic emotional shift",
        ("ENEMIES_WITH", "ALLIES_WITH"): "allegiance change",
        ("DISTRUSTS", "TRUSTS"): "trust development",
        ("FEARS", "PROTECTS"): "relationship reversal",
    }

    candidates: set[tuple[str, str, str]] = set()
    for relationship in extracted_relationships:
        values = tuple(relationship.get(name) if isinstance(relationship, dict) else getattr(relationship, name) for name in ("source_name", "target_name", "relationship_type"))
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("Candidate relationship identity missing")
        candidates.add(cast(tuple[str, str, str], values))
    for source, target, relationship_type in sorted(candidates):
        history = existing_relationships.get((source, target), [])
        prior = {(item["first_chapter"], item["rel_type"]) for item in history if type(item["first_chapter"]) is int and 0 < item["first_chapter"] < current_chapter}
        latest_chapter = max((chapter for chapter, _ in prior), default=0)
        for previous_chapter, previous_type in sorted(prior):
            if previous_chapter != latest_chapter:
                continue
            description = requires_development.get((previous_type, relationship_type))
            chapters_between = current_chapter - previous_chapter
            if description is not None and chapters_between < 3:
                contradictions.append(Contradiction(
                    type="relationship",
                    description=f"{source} and {target}: {description} from '{previous_type}' to '{relationship_type}' without sufficient development (only {chapters_between} chapters since chapter {previous_chapter})",
                    conflicting_chapters=[previous_chapter, current_chapter],
                    severity="minor",
                    suggested_fix=f"Add intermediate scenes showing the {description}",
                ))
    return contradictions


def create_validation_subgraph() -> StateGraph:
    """Create and compile the validation subgraph.

    Order of operations:
        1. `validate_consistency`
        2. `evaluate_quality`
        3. `detect_contradictions`

    Returns:
        A compiled `StateGraph` implementing the validation phase.
    """
    workflow = StateGraph(NarrativeState)

    workflow.add_node("validate_consistency", validate_consistency)
    workflow.add_node("evaluate_quality", evaluate_quality)
    workflow.add_node("detect_contradictions", detect_contradictions)

    workflow.set_entry_point("validate_consistency")

    workflow.add_conditional_edges(
        "validate_consistency",
        _should_continue_or_error,
        {"continue": "evaluate_quality", "error": END},
    )
    workflow.add_conditional_edges(
        "evaluate_quality",
        _should_continue_or_error,
        {"continue": "detect_contradictions", "error": END},
    )
    workflow.add_edge("detect_contradictions", END)

    return workflow.compile()


__all__ = [
    "validate_consistency",
    "evaluate_quality",
    "detect_contradictions",
    "create_validation_subgraph",
]
