# core/langgraph/nodes/validation_node.py
"""
Consistency validation node for LangGraph workflow.

This module contains the validation logic for checking generated content
against the knowledge graph and established narrative rules.

Migration Reference: docs/langgraph_migration_plan.md - Step 1.4.1

Validation Checks:
1. Relationship semantic validation - ensures relationships make sense for entity types
2. Character trait consistency - detects contradictory character traits
3. Plot stagnation - ensures chapters advance the narrative

The relationship validation system uses flexible semantic rules defined in
core/relationship_validation.py to prevent obviously nonsensical relationships
while maintaining creative flexibility.
"""

from __future__ import annotations

from typing import Any

import structlog

from core.db_manager import neo4j_manager
from core.langgraph.state import Contradiction, ExtractedEntity, NarrativeState

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

    NOTE: Relationship validation has been disabled (constraint system removed)

    The validation results determine whether the chapter needs revision.
    If critical or too many major issues are found, needs_revision is set to True.

    Args:
        state: Current narrative state with draft_text and extracted entities

    Returns:
        Updated state with contradictions list and needs_revision flag
    """
    logger.info(
        "validate_consistency",
        chapter=state.get("current_chapter", 1),
        relationships=len(state.get("extracted_relationships", [])),
        characters=len(state.get("extracted_entities", {}).get("characters", [])),
    )

    contradictions: list[Contradiction] = []

    # Check 1: Validate all extracted relationships
    # Validates semantic correctness (e.g., prevents "Character FRIENDS_WITH Location")
    relationship_contradictions = await _validate_relationships(
        state.get("extracted_relationships", []),
        state.get("current_chapter", 1),
        state.get("extracted_entities"),
    )
    contradictions.extend(relationship_contradictions)

    # Check 2: Character trait consistency
    # NEW FUNCTIONALITY: Checks for contradictory character traits
    trait_contradictions = await _check_character_traits(
        state.get("extracted_entities", {}).get("characters", []),
        state.get("current_chapter", 1),
    )
    contradictions.extend(trait_contradictions)

    # Check 3: Plot stagnation detection
    # NEW FUNCTIONALITY: Ensures chapter advances the plot
    if _is_plot_stagnant(state):
        contradictions.append(
            Contradiction(
                type="plot_stagnation",
                description="Chapter does not significantly advance plot",
                conflicting_chapters=[state.get("current_chapter", 1)],
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
    relationships: list[Any],
    chapter: int,
    extracted_entities: dict[str, list[ExtractedEntity]] | None = None,
) -> list[Contradiction]:
    """
    Validate extracted relationships for semantic correctness.

    This function checks that relationships make semantic sense given the
    entity types involved. It validates:
    1. Relationship types are recognized
    2. Entity types are valid
    3. Relationship type is compatible with entity types

    The validation is designed to be flexible for creative writing while
    preventing obviously nonsensical relationships (e.g., "Character FRIENDS_WITH Location").

    Args:
        relationships: List of ExtractedRelationship instances
        chapter: Current chapter number
        extracted_entities: Dict with "characters" and "world_items" lists for type lookup

    Returns:
        List of Contradiction instances for invalid relationships
    """
    from core.relationship_validation import get_relationship_validator

    if not relationships:
        return []

    contradictions = []
    validator = get_relationship_validator()

    # Build entity type lookup from extracted entities
    entity_type_map = {}
    if extracted_entities:
        # Add characters
        for char in extracted_entities.get("characters", []):
            entity_type_map[char.name] = char.type

        # Add world items (locations, objects, events, etc.)
        for item in extracted_entities.get("world_items", []):
            entity_type_map[item.name] = item.type

        # Add events if they're tracked separately
        for event in extracted_entities.get("events", []):
            entity_type_map[event.name] = event.type

    # Validate each relationship
    for rel in relationships:
        # Get entity types
        source_type = entity_type_map.get(
            rel.source_name, "Character"
        )  # Default to Character
        target_type = entity_type_map.get(
            rel.target_name, "Character"
        )  # Default to Character

        # Validate the relationship (use flexible mode for creative writing)
        is_valid, errors, warnings = validator.validate(
            relationship_type=rel.relationship_type,
            source_name=rel.source_name,
            source_type=source_type,
            target_name=rel.target_name,
            target_type=target_type,
            severity_mode="flexible",
        )

        # Only create contradictions for actual errors (not warnings)
        if not is_valid:
            # Determine severity based on error type
            severity = "major"  # Default severity

            # If the relationship type is completely unknown, it's critical
            if "Unknown relationship type" in errors[0] if errors else "":
                severity = "critical"

            contradictions.append(
                Contradiction(
                    type="invalid_relationship",
                    description=f"Invalid relationship: {rel.source_name}({source_type}) "
                    f"-[{rel.relationship_type}]-> {rel.target_name}({target_type}). "
                    f"Errors: {'; '.join(errors)}",
                    conflicting_chapters=[chapter],
                    severity=severity,
                    suggested_fix="Use a semantically appropriate relationship type, or "
                    "verify entity types are correct. See docs/ontology.md for guidance.",
                )
            )

    logger.debug(
        "_validate_relationships: relationship validation complete",
        total_relationships=len(relationships),
        invalid_relationships=len(contradictions),
    )

    return contradictions


async def _check_character_traits(
    extracted_chars: list[ExtractedEntity],
    current_chapter: int,
) -> list[Contradiction]:
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
                    if (
                        trait_a in established_traits
                        and trait_b in new_trait_candidates
                    ):
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
                    elif (
                        trait_b in established_traits
                        and trait_a in new_trait_candidates
                    ):
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

    # Check 2: Get all extracted elements
    entities = state.get("extracted_entities", {})
    events = entities.get("events", [])
    characters = entities.get("characters", [])
    world_items = entities.get("world_items", [])
    relationships = state.get("extracted_relationships", [])

    # Check 3: Count total new content
    total_new_elements = len(characters) + len(world_items) + len(events)
    total_relationships = len(relationships)

    # If we have no new elements AND no relationships, the plot is stagnant
    if total_new_elements == 0 and total_relationships == 0:
        logger.debug(
            "_is_plot_stagnant: no new elements or relationships",
            characters=len(characters),
            world_items=len(world_items),
            events=len(events),
            relationships=total_relationships,
        )
        return True

    # If we made it here, the chapter seems to be making progress
    return False


__all__ = ["validate_consistency"]
