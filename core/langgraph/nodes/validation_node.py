# core/langgraph/nodes/validation_node.py
"""Validate generated narrative for internal consistency.

This module defines the Phase 2 validation node used by the LangGraph workflow.
It checks the current chapter draft and extracted facts for issues that should
trigger a revision loop (for example, contradictory character traits or plot
stagnation).

Migration Reference: docs/langgraph_migration_plan.md - Step 1.4.1

Notes:
    Relationship validation is intentionally permissive by default: it may log
    informational warnings but does not block persistence. Revision decisions are
    based on contradictions such as trait conflicts and plot stagnation.
"""

from __future__ import annotations

from typing import Any

import structlog

from config.settings import settings
from core.db_manager import neo4j_manager
from core.langgraph.state import Contradiction, NarrativeState
from models.kg_constants import CONTRADICTORY_TRAIT_PAIRS

logger = structlog.get_logger(__name__)


def _normalize_trait(value: Any) -> str | None:
    """Normalize a trait value for stable comparisons.

    Traits are treated as values (typically emitted under `attributes["traits"]`).
    Normalization reduces false negatives due to casing and whitespace differences.

    Args:
        value: Candidate trait value.

    Returns:
        Normalized trait string, or `None` when the value is not a non-empty string.
    """
    if not isinstance(value, str):
        return None
    norm = value.strip().lower()
    return norm or None


def _coerce_traits_list(raw: Any) -> list[str]:
    """Coerce a value into a list of normalized trait strings.

    The canonical contract from extraction is `attributes["traits"]` as `list[str]`,
    but this function tolerates legacy shapes to keep validation stable.

    Args:
        raw: Raw trait value from state or persisted objects.

    Returns:
        A list of normalized, non-empty trait strings.
    """
    if raw is None:
        return []

    values: list[Any]
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list | tuple | set):
        values = list(raw)
    else:
        return []

    out: list[str] = []
    for v in values:
        norm = _normalize_trait(v)
        if norm:
            out.append(norm)
    return out


def get_extracted_events_for_validation(extracted_entities: dict[str, Any] | None) -> list[Any]:
    """Derive event entities from extraction state.

    State-shape contract:
    - Canonical: events live in `extracted_entities["world_items"]` with `type == "Event"`
      (case-insensitive).
    - Legacy compatibility: if `extracted_entities["events"]` exists, it is also included.

    Args:
        extracted_entities: Extracted entities bucket from state.

    Returns:
        A de-duplicated list of event-like objects (dicts or `ExtractedEntity` instances)
        suitable for downstream plot/timeline checks.
    """
    if not extracted_entities:
        return []

    world_items = extracted_entities.get("world_items", [])
    legacy_events = extracted_entities.get("events", [])

    candidates: list[Any] = []
    if isinstance(world_items, list):
        candidates.extend(world_items)
    if isinstance(legacy_events, list):
        candidates.extend(legacy_events)

    # Filter by type == Event (supports ExtractedEntity objects or dicts)
    filtered: list[Any] = []
    for item in candidates:
        item_type = None
        if isinstance(item, dict):
            item_type = item.get("type")
        else:
            item_type = getattr(item, "type", None)

        if isinstance(item_type, str) and item_type.strip().lower() == "event":
            filtered.append(item)

    # Deduplicate by (name, description) when available; fall back to id(item)
    seen: set[tuple[str, str] | int] = set()
    deduped: list[Any] = []
    for item in filtered:
        name = ""
        desc = ""
        if isinstance(item, dict):
            name = str(item.get("name") or "")
            desc = str(item.get("description") or "")
        else:
            name = str(getattr(item, "name", "") or "")
            desc = str(getattr(item, "description", "") or "")

        key = (name, desc) if (name or desc) else id(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped


def _get_character_trait_values_for_validation(char: Any) -> set[str]:
    """Extract normalized trait values for a character.

    Args:
        char: Character-like object (typically an `ExtractedEntity` or dict).

    Returns:
        Normalized trait values derived from `attributes["traits"]`. Missing or
        invalid shapes yield an empty set.
    """
    attrs = char.get("attributes") if isinstance(char, dict) else getattr(char, "attributes", None)
    if not isinstance(attrs, dict):
        return set()

    raw_traits = attrs.get("traits", [])
    return set(_coerce_traits_list(raw_traits))


async def validate_consistency(state: NarrativeState) -> NarrativeState:
    """Validate the chapter against extracted facts and narrative rules.

    This node computes `contradictions` and decides whether the workflow should
    enter a revision loop via `needs_revision`.

    Validation inputs:
    - Extracted relationships (validated in permissive mode by default).
    - Extracted character traits (checked for contradictions).
    - Plot stagnation heuristics.

    Args:
        state: Workflow state.

    Returns:
        Updated state containing:
        - contradictions: List of detected contradictions.
        - needs_revision: Whether to route to revision.
        - current_node: `"validate_consistency"`.

    Notes:
        This node performs Neo4j reads and/or semantic checks as part of validation.
        Relationship validation is permissive by default and does not block writes;
        revision decisions are driven by contradiction severity and plot stagnation.
    """
    # Check if validation is disabled
    if not settings.validation.ENABLE_VALIDATION:
        logger.info("validate_consistency: validation disabled, skipping all checks")
        return {
            "contradictions": [],
            "needs_revision": False,
            "current_node": "validate_consistency",
            "last_error": None,
        }

    # Initialize content manager to read externalized content
    from core.langgraph.content_manager import (
        ContentManager,
        get_extracted_entities,
        get_extracted_relationships,
        require_project_dir,
    )

    content_manager = ContentManager(require_project_dir(state))

    # Get extraction results (prefers externalized content)
    extracted_entities = get_extracted_entities(state, content_manager)
    extracted_relationships = get_extracted_relationships(state, content_manager)

    def _summarize_bucket(bucket_name: str, value: Any) -> dict[str, Any]:
        if not isinstance(value, list) or not value:
            return {"bucket": bucket_name, "count": len(value) if isinstance(value, list) else 0, "sample_type": None}

        sample = value[0]
        if isinstance(sample, dict):
            keys = sorted([str(k) for k in sample.keys()])
            return {"bucket": bucket_name, "count": len(value), "sample_type": "dict", "sample_keys": keys[:12]}

        return {"bucket": bucket_name, "count": len(value), "sample_type": type(sample).__name__}

    buckets = [
        _summarize_bucket("characters", extracted_entities.get("characters", [])),
        _summarize_bucket("world_items", extracted_entities.get("world_items", [])),
        _summarize_bucket("events", extracted_entities.get("events", [])),
    ]
    logger.info("validate_consistency: extracted_entities_shape", buckets=buckets)

    rel_sample_type = None
    rel_sample_keys = None
    if extracted_relationships:
        sample_rel = extracted_relationships[0]
        if isinstance(sample_rel, dict):
            rel_sample_type = "dict"
            rel_sample_keys = sorted([str(k) for k in sample_rel.keys()])[:12]
        else:
            rel_sample_type = type(sample_rel).__name__

    logger.info(
        "validate_consistency",
        chapter=state.get("current_chapter", 1),
        relationships=len(extracted_relationships),
        characters=len(extracted_entities.get("characters", [])),
    )
    logger.info(
        "validate_consistency: extracted_relationships_shape",
        count=len(extracted_relationships),
        sample_type=rel_sample_type,
        sample_keys=rel_sample_keys,
    )

    contradictions: list[Contradiction] = []

    # Check 1: Validate all extracted relationships (PERMISSIVE MODE)
    # In permissive mode, this only logs info messages and never blocks.
    # Relationship validation is now informational only to support creative freedom.
    relationship_contradictions = await _validate_relationships(
        extracted_relationships,
        state.get("current_chapter", 1),
        extracted_entities,
    )
    # Note: In permissive mode, relationship_contradictions will be empty
    # since the validator always returns valid=True
    contradictions.extend(relationship_contradictions)

    # Check 2: Character trait consistency
    # NEW FUNCTIONALITY: Checks for contradictory character traits
    trait_contradictions = await _check_character_traits(
        extracted_entities.get("characters", []),
        state.get("current_chapter", 1),
    )
    contradictions.extend(trait_contradictions)

    # Check 3: Plot stagnation detection
    # NEW FUNCTIONALITY: Ensures chapter advances the plot
    if _is_plot_stagnant(state, extracted_entities, extracted_relationships):
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
    # - Any major issues found
    # Unless force_continue is set
    needs_revision = (len(critical_issues) > 0 or len(major_issues) > 0) and not state.get("force_continue", False)

    logger.info(
        "validate_consistency: validation complete",
        total_issues=len(contradictions),
        critical=len(critical_issues),
        major=len(major_issues),
        minor=len(minor_issues),
        needs_revision=needs_revision,
    )

    return {
        "contradictions": contradictions,
        "needs_revision": needs_revision,
        "current_node": "validate_consistency",
        "last_error": None,
    }


async def _validate_relationships(
    relationships: list[Any],
    chapter: int,
    extracted_entities: dict[str, Any] | None = None,
) -> list[Contradiction]:
    """
    Validate extracted relationships for semantic correctness.

    **PERMISSIVE MODE (Default):**
    This function now operates in permissive mode, where all relationships are
    accepted and only informational messages are logged. The validator trusts
    the LLM to create semantically appropriate relationships for the narrative.

    The function logs debug-level info for:
    1. Novel relationship types not in the predefined schema
    2. Novel entity types not in the predefined ontology
    3. Unusual entity type combinations (informational only)

    This supports creative freedom and emergent narrative patterns.

    Args:
        relationships: List of ExtractedRelationship instances
        chapter: Current chapter number
        extracted_entities: Dict with "characters" and "world_items" lists for type lookup

    Returns:
        Empty list in permissive mode (no contradictions generated)
    """
    from core.relationship_validation import get_relationship_validator

    if not relationships:
        return []

    contradictions: list[Contradiction] = []
    validator = get_relationship_validator()  # Defaults to permissive mode

    # Build entity type lookup from extracted entities
    entity_type_map: dict[str, str] = {}
    if extracted_entities:
        # Add characters
        for char in extracted_entities.get("characters", []):
            if isinstance(char, dict):
                name = char.get("name")
                entity_type = char.get("type")
            else:
                name = getattr(char, "name", None)
                entity_type = getattr(char, "type", None)

            if isinstance(name, str) and name and isinstance(entity_type, str) and entity_type:
                entity_type_map[name] = entity_type

        # Add world items (locations, objects, events, etc.)
        for item in extracted_entities.get("world_items", []):
            if isinstance(item, dict):
                name = item.get("name")
                entity_type = item.get("type")
            else:
                name = getattr(item, "name", None)
                entity_type = getattr(item, "type", None)

            if isinstance(name, str) and name and isinstance(entity_type, str) and entity_type:
                entity_type_map[name] = entity_type

        # Add events if they're tracked separately
        for event in extracted_entities.get("events", []):
            if isinstance(event, dict):
                name = event.get("name")
                entity_type = event.get("type")
            else:
                name = getattr(event, "name", None)
                entity_type = getattr(event, "type", None)

            if isinstance(name, str) and name and isinstance(entity_type, str) and entity_type:
                entity_type_map[name] = entity_type

    # Validate each relationship
    for rel in relationships:
        if isinstance(rel, dict):
            relationship_type = rel.get("relationship_type")
            source_name = rel.get("source_name")
            target_name = rel.get("target_name")
        else:
            relationship_type = getattr(rel, "relationship_type", None)
            source_name = getattr(rel, "source_name", None)
            target_name = getattr(rel, "target_name", None)

        if not (isinstance(relationship_type, str) and relationship_type):
            logger.warning("relationship_validation_missing_relationship_type", relationship=type(rel).__name__)
            continue
        if not (isinstance(source_name, str) and source_name):
            logger.warning("relationship_validation_missing_source_name", relationship_type=relationship_type)
            continue
        if not (isinstance(target_name, str) and target_name):
            logger.warning("relationship_validation_missing_target_name", relationship_type=relationship_type)
            continue

        # Get entity types
        source_type = entity_type_map.get(source_name, "Character")
        target_type = entity_type_map.get(target_name, "Character")

        # Validate the relationship (permissive mode - always valid)
        is_valid, errors, info_warnings = validator.validate(
            relationship_type=relationship_type,
            source_name=source_name,
            source_type=source_type,
            target_name=target_name,
            target_type=target_type,
            severity_mode="flexible",
        )

        # In permissive mode (default), is_valid is always True
        # We don't create contradictions, only log info messages
        # This allows the LLM creative freedom in defining relationships
        if not is_valid:
            # This should never happen in permissive mode
            # but we keep this for strict mode compatibility
            logger.warning(
                "relationship_validation_strict_mode_violation",
                relationship=f"{rel.source_name}({source_type}) -{rel.relationship_type}-> {rel.target_name}({target_type})",
                errors=errors,
            )
            # Don't add to contradictions in permissive mode
            # If strict mode is ever enabled, this would create contradictions

    logger.debug(
        "_validate_relationships: relationship validation complete",
        total_relationships=len(relationships),
        invalid_relationships=len(contradictions),
    )

    return contradictions


async def _check_character_traits(
    extracted_chars: list[Any],
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

    try:
        for char in extracted_chars:
            character_name = char.get("name") if isinstance(char, dict) else getattr(char, "name", None)
            if not isinstance(character_name, str) or not character_name:
                continue

            # Get established traits from Neo4j (from traits property)
            query = """
                MATCH (c:Character {name: $name})
                RETURN coalesce(c.traits, []) AS traits,
                       c.created_chapter AS first_chapter,
                       c.description AS description
                LIMIT 1
            """

            result = await neo4j_manager.execute_read_query(query, {"name": character_name})

            if result and len(result) > 0:
                existing = result[0]
                # Normalize established traits defensively (Neo4j may return mixed casing)
                traits_list = existing.get("traits", [])
                established_traits = set(_coerce_traits_list(traits_list))

                # Extract *new* traits from the extraction contract:
                #   ExtractedEntity.attributes["traits"] -> list[str]
                new_trait_candidates = _get_character_trait_values_for_validation(char)

                # Check for contradictions (pair values are already lowercase in our list)
                for trait_a, trait_b in CONTRADICTORY_TRAIT_PAIRS:
                    # Check if established trait conflicts with new trait
                    if trait_a in established_traits and trait_b in new_trait_candidates:
                        contradictions.append(
                            Contradiction(
                                type="character_trait",
                                description=f"{character_name} was established as '{trait_a}' in chapter {existing.get('first_chapter', '?')}, but is now described as '{trait_b}'",
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
                                description=f"{character_name} was established as '{trait_b}' in chapter {existing.get('first_chapter', '?')}, but is now described as '{trait_a}'",
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


def _is_plot_stagnant(
    state: NarrativeState,
    entities: dict[str, Any] | None = None,
    relationships: list[Any] | None = None,
) -> bool:
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
        entities: Extracted entities for the current chapter
        relationships: Extracted relationships for the current chapter

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

    if entities is None:
        entities_raw = state.get("extracted_entities", {}) or {}
        if not isinstance(entities_raw, dict):
            entities = {}
        else:
            entities = entities_raw

    if relationships is None:
        relationships_raw = state.get("extracted_relationships", [])
        if not isinstance(relationships_raw, list):
            relationships = []
        else:
            relationships = relationships_raw

    # Check 2: Get all extracted elements
    characters = entities.get("characters", []) if entities else []
    world_items = entities.get("world_items", []) if entities else []

    # Canonical state-shape: events are stored in world_items with type == "Event".
    # We also accept legacy `extracted_entities["events"]` if present.
    extracted_events = get_extracted_events_for_validation(entities)

    # Avoid double-counting: world_items includes events, but we still want to count
    # events explicitly for readability and future heuristics.
    non_event_world_items: list[Any] = []
    if isinstance(world_items, list):
        for item in world_items:
            item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            if isinstance(item_type, str) and item_type.strip().lower() == "event":
                continue
            non_event_world_items.append(item)

    # Check 3: Count total new content
    total_new_elements = len(characters) + len(non_event_world_items) + len(extracted_events)
    total_relationships = len(relationships) if relationships is not None else 0

    # When entity/relationship extraction is disabled, the entity heuristic has no
    # signal to work with and would incorrectly flag every chapter as stagnant.
    any_extraction_enabled = (
        settings.ENABLE_CHARACTER_EXTRACTION_FROM_NARRATIVE
        or settings.ENABLE_LOCATION_EXTRACTION_FROM_NARRATIVE
        or settings.ENABLE_EVENT_EXTRACTION_FROM_NARRATIVE
        or settings.ENABLE_ITEM_EXTRACTION_FROM_NARRATIVE
        or settings.ENABLE_RELATIONSHIP_EXTRACTION_FROM_NARRATIVE
    )
    if not any_extraction_enabled:
        logger.debug(
            "_is_plot_stagnant: entity/relationship extraction is disabled, skipping element heuristic",
        )
        return False

    if total_new_elements == 0 and total_relationships == 0:
        logger.debug(
            "_is_plot_stagnant: no new elements or relationships",
            characters=len(characters) if isinstance(characters, list) else None,
            world_items=len(non_event_world_items),
            events=len(extracted_events),
            relationships=total_relationships,
        )
        return True

    return False


__all__ = ["validate_consistency"]
