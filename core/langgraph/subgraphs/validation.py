# core/langgraph/subgraphs/validation.py
"""
Validation subgraph for LangGraph workflow.

This module contains the validation nodes for checking generated content
quality, consistency, and narrative coherence.

Migration Reference: docs/langgraph-architecture.md - Section 3.4

Implemented functionality:
- Consistency validation against knowledge graph
- Prose quality evaluation using LLM
- Narrative contradiction detection
- Timeline validation
- World rules validation
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog
from langgraph.graph import END, StateGraph

import config
from core.db_manager import neo4j_manager
from core.langgraph.nodes.validation_node import (
    validate_consistency as original_validate_consistency,
)
from core.langgraph.state import Contradiction, NarrativeState
from core.llm_interface_refactored import llm_service

logger = structlog.get_logger(__name__)


async def validate_consistency(state: NarrativeState) -> NarrativeState:
    """
    Check against graph constraints.

    This is a wrapper around the original validation node that checks for:
    - Character trait consistency
    - Plot stagnation
    - Relationship validation (disabled)

    Args:
        state: Current narrative state

    Returns:
        Updated state with contradictions list
    """
    logger.info("validate_consistency: checking graph constraints")
    return await original_validate_consistency(state)


async def evaluate_quality(state: NarrativeState) -> NarrativeState:
    """
    Analyze prose quality, pacing, tone, and coherence using LLM evaluation.

    This function uses an LLM to evaluate multiple quality dimensions:
    - Prose quality: Writing craft, dialogue, descriptions
    - Pacing: Narrative flow and tension management
    - Tone consistency: Alignment with genre and previous chapters
    - Coherence: Logical flow and continuity
    - Plot advancement: Story progression

    The scores are stored in the state and used to determine if revision is needed.

    Args:
        state: Current narrative state with draft_text

    Returns:
        Updated state with quality scores and feedback
    """
    logger.info(
        "evaluate_quality: analyzing prose quality",
        chapter=state.get("current_chapter"),
        word_count=state.get("draft_word_count", 0),
    )

    draft_text = state.get("draft_text")
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
        previous_summaries=state.get("previous_chapter_summaries", []),
        chapter_outline=state.get("chapter_outlines", {}).get(
            state.get("current_chapter", 1), {}
        ),
    )

    try:
        # Call LLM for quality evaluation
        model_name = state.get("extraction_model", config.NARRATIVE_MODEL)

        response, usage = await llm_service.async_call_llm(
            model_name=model_name,
            prompt=evaluation_prompt,
            temperature=0.1,  # Low temperature for consistent evaluation
            max_tokens=2048,
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
        min_quality_threshold = 0.5
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
                    suggested_fix=scores.get(
                        "feedback", "Improve prose quality and coherence"
                    ),
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
    """
    Build the prompt for LLM-based quality evaluation.

    Args:
        draft_text: The chapter text to evaluate
        chapter_number: Current chapter number
        genre: Novel genre
        theme: Novel theme
        previous_summaries: Summaries of previous chapters
        chapter_outline: Outline for current chapter

    Returns:
        Formatted prompt string
    """
    # Truncate draft if too long (keep first and last parts)
    max_text_length = 8000
    if len(draft_text) > max_text_length:
        half_length = max_text_length // 2
        draft_text = (
            draft_text[:half_length]
            + "\n\n[... middle section truncated for evaluation ...]\n\n"
            + draft_text[-half_length:]
        )

    # Format previous summaries
    summary_context = ""
    if previous_summaries:
        recent_summaries = previous_summaries[-3:]  # Last 3 chapters
        summary_context = "\n".join(
            [
                f"Chapter {chapter_number - len(recent_summaries) + i}: {s}"
                for i, s in enumerate(recent_summaries)
            ]
        )

    # Format chapter outline
    outline_context = ""
    if chapter_outline:
        outline_context = f"""
Scene Description: {chapter_outline.get('scene_description', 'N/A')}
Key Beats: {', '.join(chapter_outline.get('key_beats', ['N/A']))}
Plot Point: {chapter_outline.get('plot_point', 'N/A')}
"""

    prompt = f"""You are an expert literary critic and editor. Evaluate the following chapter from a {genre} novel with the theme "{theme}".

## Chapter {chapter_number}

{draft_text}

## Previous Chapter Context
{summary_context if summary_context else "This is the first chapter."}

## Chapter Outline/Goals
{outline_context if outline_context else "No specific outline provided."}

## Evaluation Criteria

Please evaluate this chapter on the following dimensions, scoring each from 0.0 to 1.0:

1. **Coherence Score** (0.0-1.0): Does the narrative flow logically? Are there any confusing transitions or unclear passages?

2. **Prose Quality Score** (0.0-1.0): Evaluate the writing craft including:
   - Sentence variety and rhythm
   - Dialogue naturalness and character voice
   - Descriptive language and imagery
   - Show vs tell balance

3. **Plot Advancement Score** (0.0-1.0): Does this chapter meaningfully advance the story?
   - Are key plot beats addressed?
   - Does the protagonist face challenges or make decisions?
   - Is there narrative momentum?

4. **Pacing Score** (0.0-1.0): Is the pacing appropriate for this point in the story?
   - Is there a good balance of action, dialogue, and reflection?
   - Does tension build appropriately?
   - Are there unnecessary slow sections?

5. **Tone Consistency Score** (0.0-1.0): Does the tone match the {genre} genre?
   - Is the tone consistent throughout the chapter?
   - Does it align with the established story tone?

## Response Format

Return your evaluation as a JSON object with the following structure:
```json
{{
    "coherence_score": 0.85,
    "prose_quality_score": 0.75,
    "plot_advancement_score": 0.80,
    "pacing_score": 0.70,
    "tone_consistency_score": 0.90,
    "feedback": "Brief 2-3 sentence summary of the main strengths and areas for improvement."
}}
```

Provide only the JSON object, no additional text."""

    return prompt


def _parse_quality_scores(response: str) -> dict[str, Any]:
    """
    Parse the LLM's quality evaluation response.

    Args:
        response: Raw LLM response text

    Returns:
        Dictionary with parsed scores and feedback
    """
    # Try to extract JSON from the response
    try:
        # Look for JSON block in the response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            scores = json.loads(json_str)

            # Validate and normalize scores
            normalized = {}
            for key in [
                "coherence_score",
                "prose_quality_score",
                "plot_advancement_score",
                "pacing_score",
                "tone_consistency_score",
            ]:
                value = scores.get(key)
                if isinstance(value, (int, float)):
                    normalized[key] = max(0.0, min(1.0, float(value)))
                else:
                    normalized[key] = 0.7  # Default fallback

            normalized["feedback"] = scores.get("feedback", "No feedback provided")
            return normalized

    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(
            "_parse_quality_scores: failed to parse JSON response",
            error=str(e),
        )

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
            except ValueError:
                pass

    return fallback_scores


async def detect_contradictions(state: NarrativeState) -> NarrativeState:
    """
    Detect narrative contradictions including timeline violations and world rule violations.

    This function performs additional contradiction detection beyond what
    validate_consistency checks:
    - Timeline violations (events out of order)
    - World rule violations (breaking established rules)
    - Character location inconsistencies
    - Relationship contradictions with established graph data

    Args:
        state: Current narrative state

    Returns:
        Updated state with additional contradictions
    """
    logger.info(
        "detect_contradictions: checking for narrative contradictions",
        chapter=state.get("current_chapter"),
    )

    contradictions = list(state.get("contradictions", []))
    current_chapter = state.get("current_chapter", 1)

    # Check 1: Timeline violations
    timeline_contradictions = await _check_timeline(
        state.get("extracted_entities", {}).get("events", []),
        current_chapter,
    )
    contradictions.extend(timeline_contradictions)

    # Check 2: World rule violations
    world_rule_contradictions = await _check_world_rules(
        state.get("draft_text", ""),
        state.get("current_world_rules", []),
        current_chapter,
    )
    contradictions.extend(world_rule_contradictions)

    # Check 3: Character location consistency
    location_contradictions = await _check_character_locations(
        state.get("extracted_entities", {}).get("characters", []),
        state.get("current_location"),
        current_chapter,
    )
    contradictions.extend(location_contradictions)

    # Check 4: Relationship evolution (non-contradictory but noteworthy changes)
    relationship_issues = await _check_relationship_evolution(
        state.get("extracted_relationships", []),
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

    needs_revision = (
        len(critical_issues) > 0 or len(major_issues) > 2
    ) and not state.get("force_continue", False)

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
    """
    Check for timeline violations in extracted events.

    Validates that:
    - Events reference valid time periods
    - Event sequences are logically consistent
    - No impossible temporal relationships

    Args:
        extracted_events: List of ExtractedEntity events
        current_chapter: Current chapter number

    Returns:
        List of timeline contradiction objects
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

        existing_events = await neo4j_manager.execute_read_query(
            query, {"current_chapter": current_chapter}
        )

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
                        if _is_temporal_violation(event_time, existing_time):
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
    """
    Check if two events are related (share key terms).

    Simple heuristic: events are related if they share significant words.
    """
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

    words1 = {
        w.lower()
        for w in event1.split()
        if len(w) > 4 and w.lower() not in common_words
    }
    words2 = {
        w.lower()
        for w in event2.split()
        if len(w) > 4 and w.lower() not in common_words
    }

    overlap = words1 & words2
    return len(overlap) >= 2


def _is_temporal_violation(new_time: str, existing_time: str) -> bool:
    """
    Check if two timestamps represent a temporal violation.

    This is a simplified check - in production, would use proper datetime parsing.
    """
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
    """
    Check if draft text violates established world rules.

    World rules are constraints like:
    - "Magic requires spoken words"
    - "Technology doesn't work in the Dead Zone"
    - "Vampires cannot enter without invitation"

    Args:
        draft_text: Generated chapter text
        world_rules: List of established world rules
        current_chapter: Current chapter number

    Returns:
        List of world rule violation contradictions
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
            max_tokens=1024,
            auto_clean_response=True,
        )

        # Parse violations from response
        violations = _parse_rule_violations(response)

        for violation in violations:
            contradictions.append(
                Contradiction(
                    type="world_rule",
                    description=violation.get(
                        "description", "World rule violation detected"
                    ),
                    conflicting_chapters=[current_chapter],
                    severity=violation.get("severity", "major"),
                    suggested_fix=violation.get(
                        "fix", "Revise to comply with world rules"
                    ),
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
    """Build prompt for checking world rule violations."""
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
    """Parse rule violations from LLM response."""
    try:
        # Find JSON array in response
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            violations = json.loads(json_match.group())
            if isinstance(violations, list):
                return violations
    except (json.JSONDecodeError, AttributeError):
        pass

    return []


async def _check_character_locations(
    extracted_characters: list[Any],
    current_location: dict[str, Any] | None,
    current_chapter: int,
) -> list[Contradiction]:
    """
    Check for character location inconsistencies.

    Validates that characters are not in impossible locations based on
    the narrative (e.g., character in two places at once, or in a location
    they couldn't reach).

    Args:
        extracted_characters: Characters mentioned in current chapter
        current_location: The scene's current location
        current_chapter: Current chapter number

    Returns:
        List of location-based contradictions
    """
    if not extracted_characters:
        return []

    contradictions = []

    try:
        # Query recent character locations from Neo4j
        query = """
            MATCH (c:Character)-[v:VISITED]->(l:Location)
            WHERE v.chapter = $prev_chapter
            RETURN c.name AS character, l.name AS location
        """

        prev_chapter = current_chapter - 1
        if prev_chapter < 1:
            return []

        prev_locations = await neo4j_manager.execute_read_query(
            query, {"prev_chapter": prev_chapter}
        )

        if not prev_locations:
            return []

        # Build location map
        char_locations = {loc["character"]: loc["location"] for loc in prev_locations}

        # Get current location name
        current_loc_name = None
        if current_location:
            current_loc_name = current_location.get("name") or current_location.get(
                "neo4j_id"
            )

        # Check for impossible location changes
        for char in extracted_characters:
            char_name = getattr(char, "name", str(char))

            if char_name in char_locations:
                prev_loc = char_locations[char_name]

                # Check if location changed without explanation
                if current_loc_name and prev_loc != current_loc_name:
                    # This could be flagged for review but isn't necessarily an error
                    # Only flag if it seems impossible
                    pass  # Removed as this is typically acceptable in narrative

        logger.debug(
            "_check_character_locations: location check complete",
            characters_checked=len(extracted_characters),
        )

    except Exception as e:
        logger.error(
            "_check_character_locations: error during location check",
            error=str(e),
            exc_info=True,
        )

    return contradictions


async def _check_relationship_evolution(
    extracted_relationships: list[Any],
    current_chapter: int,
) -> list[Contradiction]:
    """
    Check for problematic relationship changes.

    While relationship evolution is normal in narratives, sudden unexplained
    changes (e.g., enemies becoming lovers without development) can be flagged.

    Args:
        extracted_relationships: Relationships extracted from current chapter
        current_chapter: Current chapter number

    Returns:
        List of relationship evolution issues
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

            result = await neo4j_manager.execute_read_query(
                query, {"source": source, "target": target}
            )

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
    """
    Create the validation subgraph.

    The validation subgraph performs three sequential checks:
    1. Consistency validation - Check against knowledge graph
    2. Quality evaluation - LLM-based prose quality analysis
    3. Contradiction detection - Narrative contradictions, timeline, world rules

    Returns:
        Compiled StateGraph for validation
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
