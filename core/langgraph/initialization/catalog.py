"""Selected initialization entities: one materialization, exact IDs, no aliases."""

from __future__ import annotations

import json
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

import config
from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization.snapshot import (
    ARTIFACTS,
    CATALOG_ARTIFACT,
    CatalogRelationship,
    FrozenPayload,
    InitializationSnapshot,
    digest,
    encoded,
    require,
    select_inputs,
    strict_json,
)
from core.langgraph.state import NarrativeState
from core.parsers.act_outline_parser import ActOutlineParser
from core.parsers.chapter_outline_parser import ChapterOutlineParser
from core.parsers.global_outline_parser import GlobalOutlineParser
from core.service_context import get_services
from models.kg_constants import RELATIONSHIP_TYPES
from models.kg_models import ActKeyEvent, Chapter, CharacterProfile, MajorPlotPoint, Scene, SceneEvent, WorldItem
from prompts.prompt_renderer import get_system_prompt, render_prompt
from utils.text_processing import generate_entity_id


class ProducerEvidence(FrozenPayload):
    model: str
    template: str
    prompt_checksum: str
    response: str


class GraphEntity(FrozenPayload):
    label: Literal["Character", "Location", "Item", "Event", "Scene", "Chapter"]
    identity: str
    payload: str


class EntityCatalog(FrozenPayload):
    schema_version: Literal[2] = 2
    inputs: InitializationSnapshot
    entities: tuple[GraphEntity, ...]
    evidence: tuple[ProducerEvidence, ...]

    @property
    def identity(self) -> str:
        return digest(self.model_dump_json())

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        require(tuple(artifact.content_type for artifact in self.inputs.artifacts) == ARTIFACTS[:-1], "Catalog requires exactly selected source artifacts")
        require(self.inputs.schema_version == 1 and not self.inputs.relationships, "Catalog inputs cannot contain relationship assertions")
        require(len({entity.identity for entity in self.entities}) == len(self.entities), "Duplicate catalog identity")
        for entity in self.entities:
            payload = strict_json(entity.payload)
            require(payload.get("id") == entity.identity, "Catalog payload identity mismatch")
            if entity.label == "Character":
                CharacterProfile.model_validate(payload)
            elif entity.label in {"Location", "Item"}:
                item = WorldItem.model_validate(payload)
                require(item.category == ("location" if entity.label == "Location" else "object"), "Catalog world label mismatch")
            elif entity.label == "Chapter":
                Chapter.model_validate({key: value for key, value in payload.items() if key != "generation_status"})
                require(payload.get("generation_status") == "planned", "Catalog chapter must be planned")
            elif entity.label == "Scene":
                Scene.model_validate(payload)
            else:
                event_models: dict[str, type[BaseModel]] = {"MajorPlotPoint": MajorPlotPoint, "ActKeyEvent": ActKeyEvent, "SceneEvent": SceneEvent}
                require(payload.get("event_type") in event_models, "Catalog event type mismatch")
                event_models[payload["event_type"]].model_validate(payload)
        return self

    def verify_inputs(self, inputs: InitializationSnapshot) -> None:
        require(self.inputs == inputs, "Catalog parent/project/checksum binding mismatch; select the matching source artifacts")

    def endpoint(self, identity: Any, label: str) -> str:
        require(isinstance(identity, str), "Catalog ID must be a string")
        require(any(entity.identity == identity and entity.label == label for entity in self.entities), f"Unknown catalog ID or wrong label: {identity} ({label})")
        return str(identity)

    def candidates(self, *labels: str) -> list[dict[str, Any]]:
        return [{"id": entity.identity, "label": entity.label, **json.loads(entity.payload)} for entity in self.entities if entity.label in labels]

    def model_candidates(self, *labels: str) -> list[dict[str, Any]]:
        """Keep semantic fields and exact IDs without empty values or storage bookkeeping."""
        storage_fields = {"created_ts", "updated_ts", "created_chapter", "last_updated_chapter", "is_provisional", "embedding_vector", "embedding_model", "entity_embedding_vector", "entity_embedding_model"}
        return [
            {key: value for key, value in candidate.items() if key not in storage_fields and value not in (None, "", [], {})}
            for candidate in self.candidates(*labels)
        ]

    def response_format(self, name: str) -> dict[str, Any]:
        """Constrain producer syntax and literal choices; application admission remains authoritative."""
        def choices(*labels: str, nullable: bool = False) -> dict[str, Any]:
            identifiers: list[str | None] = [entity.identity for entity in self.entities if entity.label in labels]
            if nullable:
                identifiers.append(None)
            require(bool(identifiers), "Selector requires catalog candidates")
            return {"type": ["string", "null"] if nullable else "string", "enum": identifiers}

        def record(properties: dict[str, Any]) -> dict[str, Any]:
            return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}

        def array(properties: dict[str, Any]) -> dict[str, Any]:
            return {"type": "array", "items": record(properties)}

        role = {"type": ["string", "null"]}
        if name == "extract_outline_relationships":
            labels = ["Character", "Location", "Item", "Event"]
            rows = array({
                "source_id": choices(*labels), "source_label": {"type": "string", "enum": labels},
                "target_id": choices(*labels), "target_label": {"type": "string", "enum": labels},
                "relationship_type": {"type": "string", "enum": sorted(RELATIONSHIP_TYPES)}, "description": {"type": "string"},
            })
            rows["maxItems"] = 20
            schema = record({"kg_triples": rows})
        elif name == "catalog_possessions":
            schema = record({"possessions": array({"character_id": choices("Character"), "item_id": choices("Item")})})
        elif name == "catalog_event_characters":
            schema = array({"character_id": choices("Character"), "role": role})
        elif name == "catalog_event_location":
            schema = record({"location_id": choices("Location", nullable=True)})
        elif name == "catalog_event_items":
            schema = record({"featured_items": array({"item_id": choices("Item"), "role": role})})
        else:
            raise ValueError(f"Unknown catalog producer contract: {name}")
        return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


class RelationshipArtifact(FrozenPayload):
    schema_version: Literal[2] = 2
    project_id: str
    catalog_identity: str = Field(pattern=r"^[a-f0-9]{64}$")
    relationships: tuple[CatalogRelationship, ...]
    evidence: tuple[ProducerEvidence, ...] = ()

    def verify(self, catalog: EntityCatalog) -> None:
        require(self.project_id == catalog.inputs.project_id and self.catalog_identity == catalog.identity, "Relationship catalog/project binding mismatch")
        for relationship in self.relationships:
            catalog.endpoint(relationship.source_id, relationship.source_label)
            catalog.endpoint(relationship.target_id, relationship.target_label)


def materialize_entities(inputs: InitializationSnapshot, world_items: list[WorldItem], evidence: tuple[ProducerEvidence, ...]) -> EntityCatalog:
    entities: list[GraphEntity] = []

    def add(label: Any, model: BaseModel) -> None:
        payload = model.model_dump(mode="json")
        if label == "Chapter":
            payload["generation_status"] = "planned"
        entities.append(GraphEntity(label=label, identity=payload["id"], payload=encoded(payload)))

    for sheet in inputs.characters:
        add("Character", CharacterProfile(
            name=sheet.name, id=generate_entity_id(sheet.name, "character"), personality_description=sheet.description, traits=list(sheet.traits), status=sheet.status,
            created_chapter=0, is_provisional=False, physical_description=sheet.physical_description,
            relationships={}, motivations=sheet.motivations, background=sheet.background, skills=list(sheet.skills),
            internal_conflict=sheet.internal_conflict, is_protagonist=sheet.is_protagonist,
        ))
    for event in GlobalOutlineParser()._parse_major_plot_points(inputs.source("global_outline")):
        add("Event", event)
    act_events = ActOutlineParser()._parse_act_key_events(inputs.source("act_outlines"))
    require({event.act_number for event in act_events} == set(range(1, inputs.total_acts + 1)), "Incomplete act event coverage")
    for act_event in act_events:
        add("Event", act_event)
    for item in world_items:
        add("Location" if item.category == "location" else "Item", item)
    chapter_parser = ChapterOutlineParser()
    names = [sheet.name for sheet in inputs.characters]
    for chapter in inputs.chapters:
        source = chapter.model_dump(mode="json")
        add("Chapter", chapter_parser._parse_chapter(source))
        for scene in chapter_parser._parse_scenes(source, names):
            add("Scene", scene)
        for scene_event in chapter_parser._parse_scene_events(source, names):
            add("Event", scene_event)
    return EntityCatalog(inputs=inputs, entities=tuple(entities), evidence=evidence)


def select_catalog(state: NarrativeState) -> EntityCatalog:
    reference = state.get("initialization_catalog_ref")
    require(isinstance(reference, dict), "Initialization requires selected initialization_catalog_ref; unfrozen legacy v1 initialization cannot resume as v2; preserve it and start a separate project")
    assert reference is not None
    require(reference["content_type"] == CATALOG_ARTIFACT and reference["version"] == 2, "Expected initialization catalog version 2")
    manager = ContentManager(state["project_dir"])
    catalog = EntityCatalog.model_validate_json(manager.load_text_strict(reference))
    catalog.verify_inputs(select_inputs(state))
    return catalog


async def materialize_initialization_catalog(state: NarrativeState) -> NarrativeState:
    if state.get("initialization_catalog_ref") is not None:
        select_catalog(state)
        return {"initialization_catalog_ref": state["initialization_catalog_ref"], "current_node": "initialization_catalog"}
    require(not state.get("initialization_id") and not state.get("initialization_complete") and not state.get("outline_relationships_ref"),
            "Unfrozen legacy v1 initialization cannot resume as v2; replay its frozen plan if available, otherwise preserve it and start a separate project")
    inputs = select_inputs(state)
    from core.langgraph.initialization.commit_init_node import _parse_world_items_extraction
    from core.langgraph.initialization.staged_import import retain_producer_selection

    async def produce() -> EntityCatalog:
        template = "knowledge_agent/extract_world_items_lines.j2"
        prompt = render_prompt(template, {
            "setting": json.loads(inputs.metadata)["setting"],
            "outline_text": encoded({"global_outline": inputs.source("global_outline"), "act_outlines": inputs.source("act_outlines")}),
        })
        response, _ = await get_services().language_model.async_call_llm(
            model_name=config.NARRATIVE_MODEL, prompt=prompt, temperature=0.3,
            max_tokens=config.MAX_GENERATION_TOKENS, allow_fallback=False,
            auto_clean_response=False, system_prompt=get_system_prompt("knowledge_agent"),
        )
        world_items = _parse_world_items_extraction(response)
        return materialize_entities(inputs, world_items, (ProducerEvidence(model=config.NARRATIVE_MODEL, template=template, prompt_checksum=digest(prompt), response=response),))

    catalog = EntityCatalog.model_validate_json(await retain_producer_selection(state["project_dir"], "catalog", inputs.identity, produce))
    catalog.verify_inputs(inputs)
    reference = ContentManager(state["project_dir"]).save_json(catalog.model_dump(mode="json"), CATALOG_ARTIFACT, catalog.inputs.identity, version=2)
    return {"initialization_catalog_ref": reference, "current_node": "initialization_catalog", "initialization_step": "catalog_materialized"}
