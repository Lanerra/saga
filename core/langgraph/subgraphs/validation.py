# core/langgraph/subgraphs/validation.py
"""
Build the validation subgraph for SAGA's LangGraph workflow.

Migration Reference: docs/langgraph-architecture.md - Section 3.4

This subgraph runs a sequence of checks over a chapter draft and extracted
signals:
- Consistency validation (graph- and heuristic-based).
- LLM-based prose quality evaluation.
- Additional contradiction detection (timeline/world rules/relationship evolution).

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
    get_extracted_entities,
    get_extracted_relationships,
    get_previous_summaries,
)
from core.langgraph.nodes.validation_node import (
    get_extracted_events_for_validation,
)
from core.langgraph.nodes.validation_node import (
    validate_consistency as original_validate_consistency,
)
from core.langgraph.state import Contradiction, NarrativeState
from core.llm_interface_refactored import llm_service
from prompts.prompt_renderer import render_prompt
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

    content_manager = ContentManager(state.get("project_dir", ""))
    draft_text = get_draft_text(state, content_manager)

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
            temperature=0.1,  # Low temperature for consistent evaluation
            max_tokens=16384,
            auto_clean_response=True,
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


async def detect_contradictions(state: NarrativeState) -> NarrativeState:
    """Detect additional narrative contradictions and update revision decision.

    This step augments the contradictions produced by
    [`validate_consistency()`](core/langgraph/subgraphs/validation.py:49) with
    additional checks (timeline violations, world rule violations, relationship evolution).

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

    # Initialize content manager to read externalized content
    content_manager = ContentManager(state.get("project_dir", ""))

    # Check 1: Timeline violations
    # Delegate state-shape assumptions to the validation node's helper so the subgraph
    # doesn't diverge (canonical events live in `world_items` with type == "Event").
    extracted_entities = get_extracted_entities(state, content_manager)
    extracted_events = get_extracted_events_for_validation(extracted_entities)
    timeline_contradictions = await _check_timeline(
        extracted_events,
        current_chapter,
    )
    contradictions.extend(timeline_contradictions)

    # Check 2: World rule violations
    draft_text = get_draft_text(state, content_manager) or ""

    world_rule_contradictions = await _check_world_rules(
        draft_text,
        state.get("current_world_rules", []),
        current_chapter,
    )
    contradictions.extend(world_rule_contradictions)

    # Check 3: Relationship evolution (non-contradictory but noteworthy changes)
    extracted_relationships = get_extracted_relationships(state, content_manager)
    relationship_issues = await _check_relationship_evolution(
        extracted_relationships,
        current_chapter,
    )
    contradictions.extend(relationship_issues)

    logger.info(
        "detect_contradictions: contradiction detection complete",
        chapter=current_chapter,
        total_contradictions=len(contradictions),
        new_contradictions=len(contradictions) - len(state.get("contradictions", [])),
    )

    # Update needs_revision based on all contradictions
    critical_issues = [c for c in contradictions if c.severity == "critical"]
    major_issues = [c for c in contradictions if c.severity == "major"]

    needs_revision = (len(critical_issues) > 0 or len(major_issues) > 0) and not state.get("force_continue", False)

    return {
        **state,
        "contradictions": contradictions,
        "needs_revision": needs_revision,
        "current_node": "detect_contradictions",
    }


async def _check_timeline(
    extracted_events: list[Any],
    current_chapter: int,
) -> list[Contradiction]:
    """Check for timeline violations using extracted events and Neo4j history.

    Args:
        extracted_events: Event-like objects derived from extraction.
        current_chapter: Chapter number being validated.

    Returns:
        Timeline-related contradictions. Returns an empty list on query errors
        (best-effort behavior).
    """
    if not extracted_events:
        return []

    contradictions = []

    try:
        # Query Neo4j for existing events with timestamps
        query = """
            MATCH (e:Event)-[:OCCURRED_IN]->(ch:Chapter)
            WHERE ch.number < $current_chapter
            AND e.timestamp IS NOT NULL
            RETURN e.description AS description,
                   e.timestamp AS timestamp,
                   ch.number AS chapter
            ORDER BY ch.number, e.timestamp
        """

        existing_events = await neo4j_manager.execute_read_query(query, {"current_chapter": current_chapter})

        if not existing_events:
            return []

        # Build timeline of existing events
        timeline = {
            event["description"]: {
                "timestamp": event.get("timestamp"),
                "chapter": event.get("chapter"),
            }
            for event in existing_events
            if event.get("timestamp")
        }

        # Check each extracted event for timeline issues
        for event in extracted_events:
            event_desc = getattr(event, "description", str(event))
            event_attrs = getattr(event, "attributes", {})

            # Check if event references something that should have happened earlier
            if "timestamp" in event_attrs:
                event_time = event_attrs["timestamp"]

                # Look for chronological inconsistencies
                for existing_desc, existing_data in timeline.items():
                    if _events_are_related(event_desc, existing_desc):
                        existing_time = existing_data.get("timestamp")
                        existing_chapter = existing_data.get("chapter")

                        # Check for "before" references to things that happened "after"
                        if isinstance(event_time, str) and isinstance(existing_time, str) and _is_temporal_violation(event_time, existing_time):
                            contradictions.append(
                                Contradiction(
                                    type="event_sequence",
                                    description=f"Timeline issue: '{event_desc}' references time '{event_time}' "
                                    f"which conflicts with '{existing_desc}' at '{existing_time}' "
                                    f"from chapter {existing_chapter}",
                                    conflicting_chapters=[
                                        existing_chapter,
                                        current_chapter,
                                    ],
                                    severity="major",
                                    suggested_fix="Adjust temporal references to maintain consistency",
                                )
                            )

        logger.debug(
            "_check_timeline: timeline validation complete",
            events_checked=len(extracted_events),
            contradictions_found=len(contradictions),
        )

    except Exception as e:
        logger.error(
            "_check_timeline: error during timeline check",
            error=str(e),
            exc_info=True,
        )

    return contradictions


def _events_are_related(event1: str, event2: str) -> bool:
    """Return whether two event descriptions appear related by keyword overlap."""
    # Extract significant words (> 4 chars, not common words)
    common_words = {
        "that",
        "this",
        "with",
        "from",
        "have",
        "were",
        "been",
        "they",
        "their",
    }

    words1 = {w.lower() for w in event1.split() if len(w) > 4 and w.lower() not in common_words}
    words2 = {w.lower() for w in event2.split() if len(w) > 4 and w.lower() not in common_words}

    overlap = words1 & words2
    return len(overlap) >= 2


def _is_temporal_violation(new_time: str, existing_time: str) -> bool:
    """Return whether two time expressions contain an obvious ordering conflict."""
    # Simple keyword-based temporal ordering
    before_keywords = ["before", "earlier", "prior", "yesterday", "last"]
    after_keywords = ["after", "later", "following", "tomorrow", "next"]

    new_lower = str(new_time).lower()
    existing_lower = str(existing_time).lower()

    # Check for obvious contradictions
    for before in before_keywords:
        for after in after_keywords:
            if before in new_lower and after in existing_lower:
                return True
            if after in new_lower and before in existing_lower:
                return True

    return False


async def _check_world_rules(
    draft_text: str,
    world_rules: list[str],
    current_chapter: int,
) -> list[Contradiction]:
    """Check draft text against established world rules (best-effort).

    This function may consult Neo4j for additional `WorldRule` nodes and then
    uses an LLM to identify likely violations.

    Args:
        draft_text: Chapter text to analyze.
        world_rules: Pre-configured rule strings.
        current_chapter: Chapter number being validated.

    Returns:
        World rule violations as contradictions. Returns an empty list on errors.

    Notes:
        This function performs LLM I/O.
    """
    if not world_rules or not draft_text:
        return []

    contradictions = []

    try:
        # Also query Neo4j for any stored world rules
        query = """
            MATCH (r:WorldRule)
            RETURN r.description AS description,
                   r.constraint AS constraint,
                   r.created_chapter AS created_chapter
        """

        db_rules = await neo4j_manager.execute_read_query(query)

        # Combine configured rules with DB rules
        all_rules = list(world_rules)
        if db_rules:
            for rule in db_rules:
                rule_text = rule.get("description") or rule.get("constraint")
                if rule_text and rule_text not in all_rules:
                    all_rules.append(rule_text)

        if not all_rules:
            return []

        # Use LLM to check for rule violations
        rule_check_prompt = _build_rule_check_prompt(draft_text, all_rules)

        model_name = config.NARRATIVE_MODEL
        response, _ = await llm_service.async_call_llm(
            model_name=model_name,
            prompt=rule_check_prompt,
            temperature=0.1,
            max_tokens=16384,
            auto_clean_response=True,
        )

        # Parse violations from response
        violations = _parse_rule_violations(response)

        for violation in violations:
            contradictions.append(
                Contradiction(
                    type="world_rule",
                    description=violation.get("description", "World rule violation detected"),
                    conflicting_chapters=[current_chapter],
                    severity=violation.get("severity", "major"),
                    suggested_fix=violation.get("fix", "Revise to comply with world rules"),
                )
            )

        logger.debug(
            "_check_world_rules: world rule validation complete",
            rules_checked=len(all_rules),
            violations_found=len(contradictions),
        )

    except Exception as e:
        logger.error(
            "_check_world_rules: error during world rule check",
            error=str(e),
            exc_info=True,
        )

    return contradictions


def _build_rule_check_prompt(draft_text: str, rules: list[str]) -> str:
    """Build the world-rule violation evaluation prompt."""
    # Truncate text if too long
    max_length = 6000
    if len(draft_text) > max_length:
        draft_text = draft_text[:max_length] + "\n[... truncated ...]"

    rules_text = "\n".join([f"- {rule}" for rule in rules])

    return f"""Analyze the following text for violations of established world rules.

## World Rules
{rules_text}

## Text to Analyze
{draft_text}

## Task
Identify any instances where the text violates the established world rules.

Return your analysis as a JSON array. If no violations found, return an empty array [].
If violations found, each violation should have:
- "description": Brief description of the violation
- "severity": "minor", "major", or "critical"
- "fix": Suggested way to fix the violation

Example response:
```json
[
    {{
        "description": "Character uses magic without speaking, violating the spoken words requirement",
        "severity": "major",
        "fix": "Add dialogue showing the character speaking the spell"
    }}
]
```

Return only the JSON array:"""


def _parse_rule_violations(response: str) -> list[dict[str, Any]]:
    """Parse a list of rule violations from an LLM response."""
    parsed, _candidates, _parse_errors = try_load_json_from_response(
        response,
        expected_root=list,
    )
    if isinstance(parsed, list):
        return parsed

    return []


async def _check_relationship_evolution(
    extracted_relationships: list[Any],
    current_chapter: int,
) -> list[Contradiction]:
    """Flag abrupt relationship shifts that may require narrative development.

    Args:
        extracted_relationships: Relationships extracted from the current chapter.
        current_chapter: Chapter number being validated.

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

            # Check for previous relationship between these characters
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
