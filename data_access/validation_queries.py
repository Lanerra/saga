"""Read prior canon from graph-accepted, checksum-bound attempt artifacts."""

from __future__ import annotations

import json
from typing import Any, cast

from core.langgraph.chapter_lifecycle import AttemptManifest, ChapterLifecycle, canonical_bytes
from core.langgraph.content_manager import ContentManager, get_extracted_entities, get_extracted_relationships
from core.langgraph.state import ExtractedEntity, ExtractedRelationship, NarrativeState
from core.service_context import get_services

PRIOR_ACCEPTED_ATTEMPTS_QUERY = """
MATCH (attempt:ChapterAttempt {project_id: $project_id, phase: 'accepted'})
WHERE attempt.chapter_number IS :: INTEGER NOT NULL
  AND attempt.chapter_number > 0 AND attempt.chapter_number < $current_chapter
OPTIONAL MATCH (chapter:Chapter {number: attempt.chapter_number})
RETURN attempt.id AS id, attempt.manifest AS manifest, attempt.phase AS phase,
       attempt.acceptance AS acceptance, attempt.chapter_number AS chapter,
       chapter.attempt_id AS chapter_attempt_id,
       chapter.graph_project_id AS chapter_project_id,
       chapter.generation_status AS chapter_status
ORDER BY chapter, id
"""


def parse_relationship_assertions(relationships: list[Any], entities: dict[str, Any]) -> list[dict[str, str]]:
    """Project extraction channels onto sorted unique assertions, not accepted truth."""
    if not isinstance(relationships, list) or not isinstance(entities, dict):
        raise ValueError("Relationship assertions require a list and entity dictionary")
    assertions: set[tuple[str, str, str]] = set()
    for relationship in relationships:
        if isinstance(relationship, ExtractedRelationship):
            relationship = relationship.model_dump()
        if not isinstance(relationship, dict):
            raise ValueError("Relationship assertion must be a dictionary")
        values = tuple(relationship.get(name) for name in ("source_name", "target_name", "relationship_type"))
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("Relationship assertion identity missing")
        assertions.add(cast(tuple[str, str, str], values))
    for channel in ("characters", "world_items"):
        profiles = entities.get(channel, [])
        if not isinstance(profiles, list):
            raise ValueError("Entity profiles must be a list")
        for entity in profiles:
            if isinstance(entity, ExtractedEntity):
                entity = entity.model_dump()
            if not isinstance(entity, dict) or not isinstance(entity.get("name"), str) or not entity["name"].strip():
                raise ValueError("Entity identity missing")
            attributes = entity.get("attributes", {})
            if not isinstance(attributes, dict) or not isinstance(attributes.get("relationships", {}), dict):
                raise ValueError("Profile relationships must be a dictionary")
            for target, assertion in attributes.get("relationships", {}).items():
                if not isinstance(target, str) or not target.strip() or not isinstance(assertion, dict):
                    raise ValueError("Profile relationship identity missing")
                relationship_type = assertion.get("type")
                if not isinstance(relationship_type, str) or not relationship_type.strip():
                    raise ValueError("Profile relationship type missing")
                relationship_type = relationship_type.strip().upper().replace(" ", "_")
                assertions.add((entity["name"], target, relationship_type))
    return [{"source_name": source, "target_name": target, "relationship_type": relationship_type} for source, target, relationship_type in sorted(assertions)]


def get_candidate_relationship_assertions(state: NarrativeState, manager: ContentManager) -> list[dict[str, str]]:
    """Read candidate channels without accepting or retaining them as history."""
    entities_reference = state.get("extracted_entities_ref")
    relationships_reference = state.get("extracted_relationships_ref")
    entities = manager.load_json_strict(entities_reference) if entities_reference else get_extracted_entities(state, manager)
    relationships = manager.load_json_strict(relationships_reference) if relationships_reference else get_extracted_relationships(state, manager)
    return parse_relationship_assertions(relationships, entities)


async def fetch_prior_accepted_facts(state: NarrativeState) -> dict[str, Any]:
    """Never infer acceptance from generated flags or mutable semantic edges.

    Retained extraction is the assertion snapshot. Graph acceptance is the decision;
    its matching local receipt and manuscript bytes must still verify. Publication
    may lag acceptance and is not independently required to establish that decision.
    Every earlier chapter requires verified acceptance, even with empty extraction.
    """
    current = ChapterLifecycle(state)
    if state.get("extraction_status") != "complete" or state.get("extraction_policy") != "fail_closed":
        raise ValueError("Validation requires complete fail-closed extraction")
    if get_services().database.require_project_binding() != current.project_id:
        raise ValueError("Validation/database ownership mismatch")
    if state.get("revision_rollback_failure") is not None or state.get("has_fatal_error"):
        raise ValueError("Validation cannot cross an unresolved failure barrier")
    attempt_id = state.get("attempt_id")
    if attempt_id is not None:
        current.load(attempt_id)
        if current.files.exists(current.phase_path("compensation_required")):
            raise ValueError("Validation cannot cross a compensation barrier")
    rows = await get_services().database.execute_read_query(PRIOR_ACCEPTED_ATTEMPTS_QUERY, {"project_id": current.project_id, "current_chapter": current.chapter_number})
    relationships: dict[tuple[str, str], set[tuple[int, str]]] = {}
    characters: dict[str, set[tuple[int, tuple[str, ...]]]] = {}
    seen: set[int] = set()
    for row in rows:
        chapter = row.get("chapter")
        if row.get("phase") != "accepted":
            continue
        if type(chapter) is not int or chapter <= 0:
            raise ValueError("Invalid accepted chapter identity")
        if chapter >= current.chapter_number:
            continue
        if chapter in seen:
            raise ValueError("Ambiguous accepted chapter history")
        manifest = AttemptManifest.model_validate_json(row["manifest"])
        if row.get("id") != manifest.attempt_id:
            raise ValueError("Accepted attempt identity mismatch")
        prior = ChapterLifecycle({"lifecycle_version": 1, "project_dir": str(current.files.root), "graph_project_id": current.project_id, "current_chapter": chapter, "iteration_count": manifest.iteration_count}).load(manifest.attempt_id)
        prior._validate_rows([row])
        if row.get("chapter_attempt_id") != manifest.attempt_id or row.get("chapter_project_id") != current.project_id or row.get("chapter_status") != "finalized":
            raise ValueError("Accepted chapter projection mismatch")
        if prior.files.exists(prior.phase_path("compensation_required")):
            raise ValueError("Accepted history has a compensation barrier")
        acceptance = json.loads(row["acceptance"])
        if prior.files.read_bytes(prior.phase_path("acceptance")) != canonical_bytes(acceptance):
            raise ValueError("Graph/local acceptance mismatch")
        prior._verify_acceptance(acceptance)
        extracted = json.loads(prior._read_artifact(dict(prior.artifact_ref("extracted_relationships_ref"))))
        if not isinstance(extracted, list):
            raise ValueError("Accepted relationships must be a list")
        for item in extracted:
            relationship = ExtractedRelationship.model_validate(item, strict=True)
            if relationship.chapter != chapter or not all(value.strip() for value in (relationship.source_name, relationship.target_name, relationship.relationship_type)):
                raise ValueError("Accepted relationship identity mismatch")
        entities = json.loads(prior._read_artifact(dict(prior.artifact_ref("extracted_entities_ref"))))
        if not isinstance(entities, dict) or not isinstance(entities.get("characters"), list):
            raise ValueError("Accepted characters must be a list")
        for assertion in parse_relationship_assertions(extracted, entities):
            key = (assertion["source_name"], assertion["target_name"])
            relationships.setdefault(key, set()).add((chapter, assertion["relationship_type"]))
        for character in entities["characters"]:
            if not isinstance(character, dict) or not isinstance(character.get("name"), str) or not character["name"].strip():
                raise ValueError("Accepted character identity missing")
            traits = character.get("attributes", {}).get("traits", [])
            if not isinstance(traits, list) or any(not isinstance(trait, str) or not trait.strip() for trait in traits):
                raise ValueError("Accepted character traits must be nonempty strings")
            if traits:
                characters.setdefault(character["name"], set()).add((chapter, tuple(sorted(set(traits)))))
        seen.add(chapter)
    missing_chapters = [chapter for chapter in range(1, current.chapter_number) if chapter not in seen]
    if missing_chapters:
        raise ValueError(f"Prior accepted canon requires reconciliation: missing verified acceptance for chapters {missing_chapters}")
    return {
        "relationships": {key: [{"first_chapter": chapter, "rel_type": relationship_type} for chapter, relationship_type in sorted(facts)] for key, facts in sorted(relationships.items())},
        "characters": {name: [{"first_chapter": chapter, "traits": list(traits)} for chapter, traits in sorted(facts)] for name, facts in sorted(characters.items())},
    }
