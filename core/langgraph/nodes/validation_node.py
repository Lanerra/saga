"""
Consistency validation node for LangGraph workflow.

This module contains the validation logic for checking generated content
against the knowledge graph and established narrative rules.

Migration Reference: docs/langgraph_migration_plan.md - Step 1.4.1

Source Code Referenced:
- core/relationship_validator.py:
  - validate_batch_constraints() (line 416+)
  - should_accept_relationship() (line 479+)
  - ValidationResult class (lines 38-59)
"""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

from core.db_manager import neo4j_manager
from core.langgraph.state import Contradiction, ExtractedEntity, NarrativeState
from core.relationship_validator import validate_batch_constraints

logger = structlog.get_logger(__name__)


async def validate_consistency(state: NarrativeState) -> NarrativeState:
    """
    Check generated content for contradictions against knowledge graph.

    This is the main LangGraph node function that orchestrates consistency
    validation. It checks for:
    1. Invalid relationships (semantic validation)
    2. Character trait contradictions
    3. Plot stagnation
    4. World rule violations (future enhancement)
    5. Event timeline violations (future enhancement)

    PORTED FROM: core/relationship_validator.py

    The validation results determine whether the chapter needs revision.
    If critical or too many major issues are found, needs_revision is set to True.

    Args:
        state: Current narrative state with draft_text and extracted entities

    Returns:
        Updated state with contradictions list and needs_revision flag
    """
    logger.info(
        "validate_consistency",
        chapter=state["current_chapter"],
        relationships=len(state.get("extracted_relationships", [])),
        characters=len(state.get("extracted_entities", {}).get("characters", [])),
    )

    contradictions: List[Contradiction] = []

    # Check 1: Validate all extracted relationships
    # USES: relationship_validator.validate_batch_constraints()
    relationship_contradictions = await _validate_relationships(
        state.get("extracted_relationships", []),
        state["current_chapter"],
    )
    contradictions.extend(relationship_contradictions)

    # Check 2: Character trait consistency
    # NEW FUNCTIONALITY: Checks for contradictory character traits
    trait_contradictions = await _check_character_traits(
        state.get("extracted_entities", {}).get("characters", []),
        state["current_chapter"],
    )
    contradictions.extend(trait_contradictions)

    # Check 3: Plot stagnation detection
    # NEW FUNCTIONALITY: Ensures chapter advances the plot
    if _is_plot_stagnant(state):
        contradictions.append(
            Contradiction(
                type="plot_stagnation",
                description="Chapter does not significantly advance plot",
                conflicting_chapters=[state["current_chapter"]],
                severity="major",
                suggested_fix="Introduce conflict, decision, or revelation",
            )
        )

    # Severity-based decision for revision
    critical_issues = [c for c in contradictions if c.severity == "critical"]
    major_issues = [c for c in contradictions if c.severity == "major"]
    minor_issues = [c for c in contradictions if c.severity == "minor"]

    # Needs revision if:
    # - Any critical issues found, OR
    # - More than 2 major issues found
    # Unless force_continue is set
    needs_revision = (
        len(critical_issues) > 0 or len(major_issues) > 2
    ) and not state.get("force_continue", False)

    logger.info(
        "validate_consistency: validation complete",
        total_issues=len(contradictions),
        critical=len(critical_issues),
        major=len(major_issues),
        minor=len(minor_issues),
        needs_revision=needs_revision,
    )

    return {
        **state,
        "contradictions": contradictions,
        "needs_revision": needs_revision,
        "current_node": "validate_consistency",
        "last_error": None,
    }


async def _validate_relationships(
    relationships: List[Any],
    chapter: int,
) -> List[Contradiction]:
    """
    Validate extracted relationships using relationship constraint system.

    USES: core.relationship_validator.validate_batch_constraints()

    This function converts ExtractedRelationship instances to triple format,
    validates them using the existing constraint system, and converts
    validation failures to Contradiction instances.

    Args:
        relationships: List of ExtractedRelationship instances
        chapter: Current chapter number

    Returns:
        List of Contradiction instances for invalid relationships
    """
    if not relationships:
        return []

    contradictions = []

    try:
        # Convert to triple format for validation
        triples = []
        for rel in relationships:
            triple = {
                "subject": rel.source_name,
                "predicate": rel.relationship_type,
                "object_entity": rel.target_name,
            }
            triples.append(triple)

        # Validate using existing constraint system
        validation_results = validate_batch_constraints(triples)

        # Convert validation failures to contradictions
        for rel, result in zip(relationships, validation_results):
            if not result.is_valid:
                # Determine severity based on whether suggestions exist
                severity = "minor" if result.suggestions else "major"

                # Build suggested fix from validation result
                suggested_fix = None
                if result.suggestions:
                    best_suggestion = result.suggestions[0]
                    suggested_fix = f"Use: {best_suggestion[0]} (reason: {best_suggestion[1]})"
                elif result.validated_relationship != result.original_relationship:
                    suggested_fix = f"Use normalized form: {result.validated_relationship}"

                # Build error description
                error_details = (
                    f"Errors: {', '.join(result.errors)}" if result.errors else ""
                )

                contradictions.append(
                    Contradiction(
                        type="relationship",
                        description=f"Invalid relationship: {rel.source_name} "
                        f"{rel.relationship_type} {rel.target_name}. "
                        f"{error_details}",
                        conflicting_chapters=[chapter],
                        severity=severity,
                        suggested_fix=suggested_fix,
                    )
                )

        logger.debug(
            "_validate_relationships: validation complete",
            total=len(relationships),
            invalid=len(contradictions),
        )

        return contradictions

    except Exception as e:
        logger.error(
            "_validate_relationships: error during validation",
            error=str(e),
            exc_info=True,
        )
        # Return empty list on error to avoid breaking workflow
        return []


async def _check_character_traits(
    extracted_chars: List[ExtractedEntity],
    current_chapter: int,
) -> List[Contradiction]:
    """
    Compare extracted character attributes with established traits.

    NEW FUNCTIONALITY: Not in current SAGA, but specified in LangGraph architecture.

    This function checks for contradictory trait pairs like "brave" vs "cowardly"
    by querying Neo4j for established character traits and comparing them with
    newly extracted attributes.

    Args:
        extracted_chars: List of ExtractedEntity instances for characters
        current_chapter: Current chapter number

    Returns:
        List of Contradiction instances for trait inconsistencies
    """
    if not extracted_chars:
        return []

    contradictions = []

    # Define contradictory trait pairs
    contradictory_pairs = [
        ("introverted", "extroverted"),
        ("brave", "cowardly"),
        ("honest", "deceitful"),
        ("kind", "cruel"),
        ("optimistic", "pessimistic"),
        ("calm", "anxious"),
        ("trusting", "suspicious"),
        ("generous", "selfish"),
        ("patient", "impatient"),
        ("humble", "arrogant"),
    ]

    try:
        for char in extracted_chars:
            # Get established traits from Neo4j
            query = """
                MATCH (c:Character {name: $name})
                RETURN c.traits AS traits,
                       c.created_chapter AS first_chapter,
                       c.description AS description
                LIMIT 1
            """

            result = await neo4j_manager.execute_read_query(query, {"name": char.name})

            if result and len(result) > 0:
                existing = result[0]
                established_traits = set(existing.get("traits", []))

                # Extract traits from new attributes
                # Attributes dict may contain traits as keys
                new_trait_candidates = set(char.attributes.keys())

                # Also check description for trait keywords
                # (Simple keyword matching - could be enhanced with NLP)

                # Check for contradictions
                for trait_a, trait_b in contradictory_pairs:
                    # Check if established trait conflicts with new trait
                    if trait_a in established_traits and trait_b in new_trait_candidates:
                        contradictions.append(
                            Contradiction(
                                type="character_trait",
                                description=f"{char.name} was established as '{trait_a}' "
                                f"in chapter {existing.get('first_chapter', '?')}, "
                                f"but is now described as '{trait_b}'",
                                conflicting_chapters=[
                                    existing.get("first_chapter", 0),
                                    current_chapter,
                                ],
                                severity="major",
                                suggested_fix=f"Remove '{trait_b}' or explain character development",
                            )
                        )
                    # Also check reverse
                    elif trait_b in established_traits and trait_a in new_trait_candidates:
                        contradictions.append(
                            Contradiction(
                                type="character_trait",
                                description=f"{char.name} was established as '{trait_b}' "
                                f"in chapter {existing.get('first_chapter', '?')}, "
                                f"but is now described as '{trait_a}'",
                                conflicting_chapters=[
                                    existing.get("first_chapter", 0),
                                    current_chapter,
                                ],
                                severity="major",
                                suggested_fix=f"Remove '{trait_a}' or explain character development",
                            )
                        )

        logger.debug(
            "_check_character_traits: trait checking complete",
            characters=len(extracted_chars),
            contradictions=len(contradictions),
        )

        return contradictions

    except Exception as e:
        logger.error(
            "_check_character_traits: error during trait checking",
            error=str(e),
            exc_info=True,
        )
        # Return empty list on error to avoid breaking workflow
        return []


def _is_plot_stagnant(state: NarrativeState) -> bool:
    """
    Detect if plot is not advancing sufficiently.

    NEW FUNCTIONALITY: Specified in LangGraph architecture.

    This function checks multiple heuristics to determine if the chapter
    is making sufficient progress:
    - Minimum word count (1500 words)
    - Presence of new events
    - Presence of new relationships

    Args:
        state: Current narrative state

    Returns:
        True if plot appears stagnant, False otherwise
    """
    # Check 1: Minimum word count
    word_count = state.get("draft_word_count", 0)
    if word_count < 1500:
        logger.debug(
            "_is_plot_stagnant: insufficient word count",
            word_count=word_count,
            minimum=1500,
        )
        return True

    # Check 2: Events extracted
    # If no events were extracted, the chapter may not be advancing the plot
    entities = state.get("extracted_entities", {})
    events = entities.get("events", [])
    if len(events) == 0:
        logger.debug("_is_plot_stagnant: no events extracted")
        # Don't fail just for this - relationships might be enough
        # return True

    # Check 3: Relationships changed
    # New relationships indicate character interactions and plot development
    relationships = state.get("extracted_relationships", [])
    if len(relationships) == 0:
        logger.debug("_is_plot_stagnant: no relationships extracted")
        # Combined with no events, this might indicate stagnation
        if len(events) == 0:
            return True

    # Check 4: New characters or world items
    # Adding new elements shows world expansion
    characters = entities.get("characters", [])
    world_items = entities.get("world_items", [])

    total_new_elements = len(characters) + len(world_items)
    if total_new_elements == 0 and len(relationships) == 0:
        logger.debug("_is_plot_stagnant: no new elements or relationships")
        return True

    # If we made it here, the chapter seems to be making progress
    return False


__all__ = ["validate_consistency"]
