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

import re
from typing import Any

import structlog
from langgraph.graph import END, StateGraph  # type: ignore[import-not-found, attr-defined]

import config
from core.db_manager import neo4j_manager
from core.langgraph.content_manager import (
    ContentManager,
    get_chapter_outlines,
    get_draft_text,
    get_extracted_relationships,
    get_previous_summaries,
    get_scene_drafts,
    require_project_dir,
)
from core.langgraph.nodes.validation_node import (
    validate_consistency as original_validate_consistency,
)
from core.langgraph.state import Contradiction, NarrativeState
from core.llm_interface_refactored import llm_service
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
            **state,
            "coherence_score": None,
            "prose_quality_score": None,
            "plot_advancement_score": None,
            "pacing_score": None,
            "tone_consistency_score": None,
            "quality_feedback": "No draft text available for evaluation",
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
        model_name = state.get("extraction_model", config.NARRATIVE_MODEL)

        response, usage = await llm_service.async_call_llm(
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

        # Check if quality is too low and needs revision
        min_quality_threshold = 0.7
        quality_scores = [
            scores.get("coherence_score", 1.0),
            scores.get("prose_quality_score", 1.0),
            scores.get("plot_advancement_score", 1.0),
        ]

        avg_quality = sum(quality_scores) / len(quality_scores)

        # Add quality-based revision trigger
        current_contradictions = state.get("contradictions", [])
        if avg_quality < min_quality_threshold:
            current_contradictions.append(
                Contradiction(
                    type="quality_issue",
                    description=f"Overall quality score ({avg_quality:.2f}) below threshold ({min_quality_threshold})",
                    conflicting_chapters=[state.get("current_chapter", 1)],
                    severity="major",
                    suggested_fix=scores.get("feedback", "Improve prose quality and coherence"),
                )
            )

        return {
            **state,
            "coherence_score": scores.get("coherence_score"),
            "prose_quality_score": scores.get("prose_quality_score"),
            "plot_advancement_score": scores.get("plot_advancement_score"),
            "pacing_score": scores.get("pacing_score"),
            "tone_consistency_score": scores.get("tone_consistency_score"),
            "quality_feedback": scores.get("feedback"),
            "contradictions": current_contradictions,
        }

    except Exception as e:
        logger.error(
            "evaluate_quality: error during evaluation",
            error=str(e),
            exc_info=True,
        )
        # Return state with no scores on error
        return {
            **state,
            "coherence_score": None,
            "prose_quality_score": None,
            "plot_advancement_score": None,
            "pacing_score": None,
            "tone_consistency_score": None,
            "quality_feedback": f"Evaluation failed: {str(e)}",
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
        A dict containing normalized score fields in the range [0.0, 1.0] plus
        a `feedback` string. When parsing fails, returns conservative defaults.
    """
    parsed, _candidates, _parse_errors = try_load_json_from_response(
        response,
        expected_root=dict,
    )
    if isinstance(parsed, dict):
        # Validate and normalize scores
        normalized: dict[str, Any] = {}
        for key in [
            "coherence_score",
            "prose_quality_score",
            "plot_advancement_score",
            "pacing_score",
            "tone_consistency_score",
        ]:
            value = parsed.get(key)
            if isinstance(value, int | float):
                normalized[key] = max(0.0, min(1.0, float(value)))
            else:
                normalized[key] = 0.3

        normalized["feedback"] = parsed.get("feedback", "No feedback provided")
        return normalized

    # Fallback: try to extract scores from text
    fallback_scores = {
        "coherence_score": 0.7,
        "prose_quality_score": 0.7,
        "plot_advancement_score": 0.7,
        "pacing_score": 0.7,
        "tone_consistency_score": 0.7,
        "feedback": "Unable to parse detailed feedback from evaluation response.",
    }

    # Try to extract individual scores using regex
    score_patterns = {
        "coherence_score": r"coherence[^:]*:\s*(\d+\.?\d*)",
        "prose_quality_score": r"prose[^:]*:\s*(\d+\.?\d*)",
        "plot_advancement_score": r"plot[^:]*:\s*(\d+\.?\d*)",
        "pacing_score": r"pacing[^:]*:\s*(\d+\.?\d*)",
        "tone_consistency_score": r"tone[^:]*:\s*(\d+\.?\d*)",
    }

    for key, pattern in score_patterns.items():
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                fallback_scores[key] = max(0.0, min(1.0, value))
            except ValueError as e:
                logger.warning(
                    "Failed to parse quality score from regex match",
                    key=key,
                    matched_text=match.group(1),
                    error=str(e),
                )

    return fallback_scores


async def _fetch_validation_data(current_chapter: int) -> dict[str, Any]:
    """Fetch validation-related data from Neo4j.

    Args:
        current_chapter: Chapter number being validated.

    Returns:
        A dictionary containing:
        - "relationships": Dict mapping (source, target) -> relationship info
    """
    try:
        query = """
            MATCH (c1:Character)-[r]->(c2:Character)
            RETURN c1.name AS source_name,
                   c2.name AS target_name,
                   type(r) AS rel_type,
                   r.chapter_added AS chapter
        """

        results = await neo4j_manager.execute_read_query(query, {"current_chapter": current_chapter})

        validation_data: dict[str, Any] = {
            "relationships": {},
        }

        for row in results:
            source = row.get("source_name", "")
            target = row.get("target_name", "")
            key = (source, target)

            if key not in validation_data["relationships"]:
                validation_data["relationships"][key] = {
                    "rel_type": row.get("rel_type"),
                    "first_chapter": row.get("chapter"),
                }

        logger.debug(
            "_fetch_validation_data: fetched validation data",
            current_chapter=current_chapter,
            relationships_count=len(validation_data["relationships"]),
        )

        return validation_data

    except Exception as e:
        logger.error(
            "_fetch_validation_data: error fetching validation data",
            error=str(e),
            exc_info=True,
        )
        return {
            "relationships": {},
        }


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

    validation_data = await _fetch_validation_data(current_chapter)

    extracted_relationships = get_extracted_relationships(state, content_manager)
    relationship_issues = await _check_relationship_evolution(
        extracted_relationships,
        current_chapter,
        validation_data.get("relationships", {}),
    )
    contradictions.extend(relationship_issues)

    scene_duplication_issues = _check_scene_duplication(state, content_manager)
    contradictions.extend(scene_duplication_issues)

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
        error_msg = f"Validation failed after {max_iterations} iterations. " f"Found {len(critical_issues)} critical and {len(major_issues)} major issues."
        logger.error(
            "detect_contradictions: validation failed on final iteration",
            iteration_count=iteration_count,
            max_iterations=max_iterations,
            critical_issues=len(critical_issues),
            major_issues=len(major_issues),
        )
        return {
            **state,
            "has_fatal_error": True,
            "last_error": error_msg,
            "error_node": "validate",
            "needs_revision": False,
            "contradictions": contradictions,
            "current_node": "detect_contradictions",
        }

    needs_revision = has_issues and not force_continue

    return {
        **state,
        "contradictions": contradictions,
        "needs_revision": needs_revision,
        "current_node": "detect_contradictions",
    }


async def _check_relationship_evolution(
    extracted_relationships: list[Any],
    current_chapter: int,
    existing_relationships: dict[tuple[str, str], dict] | None = None,
) -> list[Contradiction]:
    """Flag abrupt relationship shifts that may require narrative development.

    Args:
        extracted_relationships: Relationships extracted from the current chapter.
        current_chapter: Chapter number being validated.
        existing_relationships: Pre-fetched relationships from Neo4j (optional, for optimization).

    Returns:
        Informational contradictions (typically `minor`) for abrupt transitions.
        Returns an empty list on query errors (best-effort behavior).
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

    try:
        for rel in extracted_relationships:
            # Handle both dict and object types
            if isinstance(rel, dict):
                source = rel.get("source_name", "")
                target = rel.get("target_name", "")
                rel_type = rel.get("relationship_type", "")
            else:
                source = getattr(rel, "source_name", "")
                target = getattr(rel, "target_name", "")
                rel_type = getattr(rel, "relationship_type", "")

            # Use pre-fetched relationships if provided
            if existing_relationships is not None:
                key = (source, target)
                prev_rel_data = existing_relationships.get(key)
                if prev_rel_data:
                    prev_type = prev_rel_data.get("rel_type", "")
                    prev_chapter = prev_rel_data.get("first_chapter", 0)

                    # Check if this is a dramatic shift
                    for (old_rel, new_rel), description in requires_development.items():
                        if prev_type == old_rel and rel_type == new_rel:
                            # Check if enough chapters have passed for development
                            chapters_between = current_chapter - prev_chapter

                            if chapters_between < 3:  # Arbitrary threshold
                                contradictions.append(
                                    Contradiction(
                                        type="relationship",
                                        description=f"{source} and {target}: {description} "
                                        f"from '{prev_type}' to '{rel_type}' without sufficient development "
                                        f"(only {chapters_between} chapters since chapter {prev_chapter})",
                                        conflicting_chapters=[
                                            prev_chapter,
                                            current_chapter,
                                        ],
                                        severity="minor",
                                        suggested_fix=f"Add intermediate scenes showing the {description}",
                                    )
                                )
            else:
                # Fallback: query Neo4j for previous relationship
                query = """
                    MATCH (c1:Character {name: $source})-[r]->(c2:Character {name: $target})
                    RETURN type(r) AS rel_type, r.chapter_added AS first_chapter
                    ORDER BY r.chapter_added DESC
                    LIMIT 1
                """

                result = await neo4j_manager.execute_read_query(query, {"source": source, "target": target})

                if result and len(result) > 0:
                    prev_rel = result[0]
                    prev_type = prev_rel.get("rel_type", "")
                    prev_chapter = prev_rel.get("first_chapter", 0)

                    # Check if this is a dramatic shift
                    for (old_rel, new_rel), description in requires_development.items():
                        if prev_type == old_rel and rel_type == new_rel:
                            # Check if enough chapters have passed for development
                            chapters_between = current_chapter - prev_chapter

                            if chapters_between < 3:  # Arbitrary threshold
                                contradictions.append(
                                    Contradiction(
                                        type="relationship",
                                        description=f"{source} and {target}: {description} "
                                        f"from '{prev_type}' to '{rel_type}' without sufficient development "
                                        f"(only {chapters_between} chapters since chapter {prev_chapter})",
                                        conflicting_chapters=[
                                            prev_chapter,
                                            current_chapter,
                                        ],
                                        severity="minor",
                                        suggested_fix=f"Add intermediate scenes showing the {description}",
                                    )
                                )

        logger.debug(
            "_check_relationship_evolution: relationship evolution check complete",
            relationships_checked=len(extracted_relationships),
            issues_found=len(contradictions),
        )

    except Exception as e:
        logger.error(
            "_check_relationship_evolution: error during relationship check",
            error=str(e),
            exc_info=True,
        )

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

    workflow.add_edge("validate_consistency", "evaluate_quality")
    workflow.add_edge("evaluate_quality", "detect_contradictions")
    workflow.add_edge("detect_contradictions", END)

    return workflow.compile()


__all__ = [
    "validate_consistency",
    "evaluate_quality",
    "detect_contradictions",
    "create_validation_subgraph",
]
