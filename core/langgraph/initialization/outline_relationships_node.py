"""Extract relationships from outline during initialization."""

from __future__ import annotations

import json
from typing import Any

import structlog

import config
from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization.catalog import ProducerEvidence, RelationshipArtifact, select_catalog
from core.langgraph.initialization.snapshot import CatalogRelationship, digest, encoded, require, strict_json
from core.langgraph.initialization.staged_import import retain_producer_selection
from core.langgraph.state import NarrativeState
from core.service_context import get_services
from models.kg_constants import RELATIONSHIP_TYPES
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)


async def extract_outline_relationships(state: NarrativeState) -> NarrativeState:
    """Select exact catalog IDs; retain response provenance before graph planning."""
    catalog = select_catalog(state)
    content_manager = ContentManager(state["project_dir"])
    existing = state.get("outline_relationships_ref")
    if existing is not None:
        artifact = RelationshipArtifact.model_validate_json(content_manager.load_text_strict(existing))
        artifact.verify(catalog)
        return {"outline_relationships_ref": existing, "current_node": "outline_relationships"}
    template = "initialization/extract_outline_relationships.j2"
    prompt = render_prompt(template, {
        "novel_title": state.get("title", ""), "novel_genre": state.get("genre", ""),
        "protagonist": state.get("protagonist_name", ""), "setting": state.get("setting", ""),
        "outline_text": encoded({name: catalog.inputs.source(name) for name in ("global_outline", "act_outlines", "chapter_outlines")}),
        "canonical_relationship_types": sorted(RELATIONSHIP_TYPES),
        "catalog": catalog.model_candidates("Character", "Location", "Item", "Event"),
    })
    async def produce() -> RelationshipArtifact:
        response, _ = await get_services().language_model.async_call_llm(
            model_name=config.NARRATIVE_MODEL, prompt=prompt, temperature=0.5,
            max_tokens=config.MAX_GENERATION_TOKENS, allow_fallback=False,
            auto_clean_response=False, system_prompt=get_system_prompt("knowledge_agent"),
            response_format=catalog.response_format("extract_outline_relationships"),
        )
        data = strict_json(response)
        require(isinstance(data, dict) and set(data) == {"kg_triples"} and isinstance(data["kg_triples"], list), "Expected exactly a kg_triples array")
        require(len(data["kg_triples"]) <= 20, "Outline relationship extraction exceeds 20 assertions")
        artifact = RelationshipArtifact(
            project_id=catalog.inputs.project_id, catalog_identity=catalog.identity,
            relationships=tuple(CatalogRelationship.model_validate(item) for item in data["kg_triples"]),
            evidence=(ProducerEvidence(model=config.NARRATIVE_MODEL, template=template, prompt_checksum=digest(prompt), response=response),),
        )
        artifact.verify(catalog)
        return artifact

    artifact = RelationshipArtifact.model_validate_json(await retain_producer_selection(state["project_dir"], "relationships", catalog.identity, produce))
    artifact.verify(catalog)
    reference = content_manager.save_json(artifact.model_dump(mode="json"), "outline_relationships", catalog.identity, version=2)
    return {
        "outline_relationships_ref": reference,
        "current_node": "outline_relationships",
        "initialization_step": "outline_relationships_extracted",
    }


def _parse_relationships_extraction(response: str) -> list[dict[str, Any]]:
    """Parse LLM response into relationship dictionaries.

    Args:
        response: LLM response (JSON with kg_triples key).

    Returns:
        List of relationship dictionaries.

    Raises:
        ValueError: If the output violates the JSON/schema contract.
        json.JSONDecodeError: If response is not valid JSON.
    """
    raw_text = response.strip()

    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    if raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]

    raw_text = raw_text.strip()

    data = json.loads(raw_text)

    if not isinstance(data, dict):
        raise ValueError("Expected JSON object with kg_triples key")

    if set(data) != {"kg_triples"}:
        raise ValueError("Expected exactly the kg_triples field")
    kg_triples_list = data["kg_triples"]
    if not isinstance(kg_triples_list, list):
        raise ValueError("kg_triples must be a JSON array")

    relationships: list[dict[str, Any]] = []

    for triple in kg_triples_list:
        if not isinstance(triple, dict):
            raise ValueError("Relationship triple must be an object")

        subject = triple.get("subject", "")
        predicate = triple.get("predicate", "")
        object_entity = triple.get("object_entity", "")
        description = triple.get("description", "")

        if isinstance(subject, dict):
            subject = subject.get("name", str(subject))
        if isinstance(object_entity, dict):
            object_entity = object_entity.get("name", str(object_entity))

        subject_text = str(subject).strip() if subject else ""
        target_text = str(object_entity).strip() if object_entity else ""
        predicate_text = str(predicate).strip() if predicate else ""

        if not subject_text or not target_text or not predicate_text:
            raise ValueError("Relationship triple has an incomplete identity")

        relationships.append(
            {
                "source_name": subject_text,
                "target_name": target_text,
                "relationship_type": predicate_text,
                "description": str(description).strip() if description else "",
                "chapter": 0,
                "confidence": 0.8,
            }
        )

    return relationships
