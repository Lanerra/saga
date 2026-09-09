"""One pre-transaction domain producer; frozen plans never call a provider."""

from __future__ import annotations

import json
from typing import Any, Literal

import config
from core.langgraph.initialization.catalog import EntityCatalog, RelationshipArtifact
from core.langgraph.initialization.catalog import GraphEntity as GraphEntity
from core.langgraph.initialization.catalog import ProducerEvidence as ProducerEvidence
from core.langgraph.initialization.snapshot import CatalogRelationship, FrozenPayload, InitializationSnapshot, Relationship, digest, encoded, require, strict_json
from core.parsers.chapter_outline_parser import ChapterOutlineParser
from core.service_context import get_services
from data_access.cypher_builders.native_builders import NativeCypherBuilder
from models.kg_models import ActKeyEvent, Chapter, CharacterProfile, MajorPlotPoint, Scene, SceneEvent, WorldItem
from prompts.prompt_renderer import get_system_prompt, render_prompt


class Statement(FrozenPayload):
    query: str
    parameters: str


class RelationshipDescription(FrozenPayload):
    source: Literal["profile", "outline"]
    description: str


class InitializationGraphPlan(FrozenPayload):
    snapshot: InitializationSnapshot
    entities: tuple[GraphEntity, ...]
    statements: tuple[Statement, ...]
    evidence: tuple[ProducerEvidence, ...]
    projections: str = ""

    @property
    def identity(self) -> str:
        return digest(self.model_dump_json())


class CatalogGraphPlan(InitializationGraphPlan):
    schema_version: Literal[2] = 2


def load_plan(text: str) -> InitializationGraphPlan:
    data = strict_json(text)
    if data.get("schema_version") == 2:
        plan = CatalogGraphPlan.model_validate_json(text)
        require(plan.snapshot.schema_version == 2, "Catalog plan requires snapshot version 2")
        return plan
    require("schema_version" not in data and data.get("snapshot", {}).get("schema_version") == 1, "Unsupported initialization plan version; preserve the selected plan and use its compatible runtime")
    return InitializationGraphPlan.model_validate_json(text)


async def produce_plan(snapshot: InitializationSnapshot) -> InitializationGraphPlan:
    require(snapshot.schema_version == 2, "Unfrozen legacy v1 initialization cannot resume as v2; replay its frozen plan if available, otherwise preserve it and start a separate project")
    catalog = EntityCatalog.model_validate_json(encoded(snapshot.source("initialization_catalog")))
    assertions = RelationshipArtifact.model_validate_json(encoded(snapshot.source("outline_relationships")))
    assertions.verify(catalog)
    require(snapshot.relationships == assertions.relationships, "Snapshot relationship assertions differ from selected artifact")
    require(catalog.inputs == snapshot.model_copy(update={"schema_version": 1, "relationships": (), "artifacts": tuple(artifact for artifact in snapshot.artifacts if artifact.content_type not in {"initialization_catalog", "outline_relationships"})}), "Catalog snapshot parent binding mismatch")

    statements: list[Statement] = []
    evidence = [*catalog.evidence, *assertions.evidence]
    names = {json.loads(entity.payload)["name"]: entity.identity for entity in catalog.entities if entity.label == "Character"}
    identifiers = {entity.identity for entity in catalog.entities}
    edge_identities: dict[tuple[str, str, str], int] = {}
    edge_properties: dict[tuple[str, str, str], dict[str, Any]] = {}
    edge_descriptions: dict[tuple[str, str, str], set[RelationshipDescription]] = {}

    def statement(query: str, parameters: dict[str, Any]) -> None:
        statements.append(Statement(query=query, parameters=encoded(parameters)))


    def edge(source: str, target: str, kind: str, properties: dict[str, Any], *, description: RelationshipDescription | None = None) -> None:
        if kind not in {"FEATURES_CHARACTER", "OCCURS_IN_SCENE", "INVOLVES", "PART_OF", "HAPPENS_BEFORE", "FEATURES_ITEM", "OCCURS_AT", "POSSESSES"}:
            Relationship(source_name=source, target_name=target, relationship_type=kind, description="")
        require(source in identifiers and target in identifiers, f"Unknown relationship endpoint: {source} → {target}")
        require(not {"description", "description_assertions", "description_status"} & properties.keys(), "Descriptions require source attribution")
        key = (source, target, kind)
        properties = {"chapter_added": 0, "is_provisional": False, **properties}
        if key in edge_identities:
            existing = edge_properties[key]
            require(all(existing[name] == properties[name] for name in existing.keys() & properties.keys() if name != "confidence"), f"Conflicting duplicate relationship: {key}")
            # Structured metadata must agree; confidence is strongest assertion, not a sum.
            merged = {**existing, **properties}
            if "confidence" in existing and "confidence" in properties:
                merged["confidence"] = max(existing["confidence"], properties["confidence"])
            properties = merged
        edge_properties[key] = properties
        assertions = edge_descriptions.setdefault(key, set())
        if description is not None:
            assertions.add(description)
        if assertions:
            # Profile-first, then lexical display is not a semantic verdict. JSON strings are Neo4j-safe properties.
            ordered = sorted(assertions, key=lambda assertion: (assertion.source != "profile", assertion.description))
            properties = {
                **properties,
                "description": ordered[0].description,
                "description_assertions": [encoded(assertion.model_dump()) for assertion in ordered],
                "description_status": "unresolved" if len({assertion.description for assertion in ordered}) > 1 else "single_description",
            }
        if key in edge_identities:
            index = edge_identities[key]
            statements[index] = Statement(query=statements[index].query, parameters=encoded({"source": source, "target": target, "properties": properties}))
            return
        edge_identities[key] = len(statements)
        statement(
            f"MATCH (source {{id: $source}}), (target {{id: $target}}) CREATE (source)-[relationship:{kind}]->(target) "
            "SET relationship = $properties, relationship.created_ts = timestamp(), relationship.updated_ts = timestamp()",
            {"source": source, "target": target, "properties": properties},
        )

    async def extract(template: str, context: dict[str, Any]) -> Any:
        prompt = render_prompt(template, context)
        response, _ = await get_services().language_model.async_call_llm(
            model_name=config.NARRATIVE_MODEL, prompt=prompt, temperature=0.3,
            max_tokens=config.MAX_GENERATION_TOKENS, allow_fallback=False,
            auto_clean_response=False, system_prompt=get_system_prompt("knowledge_agent"),
        )
        result = strict_json(response)
        evidence.append(ProducerEvidence(model=config.NARRATIVE_MODEL, template=template, prompt_checksum=digest(prompt), response=response))
        return result

    characters = [CharacterProfile.model_validate_json(entity.payload) for entity in catalog.entities if entity.label == "Character"]
    world_items = [WorldItem.model_validate_json(entity.payload) for entity in catalog.entities if entity.label in {"Location", "Item"}]
    major_events = [MajorPlotPoint.model_validate_json(entity.payload) for entity in catalog.entities if entity.label == "Event" and json.loads(entity.payload)["event_type"] == "MajorPlotPoint"]
    act_events = [ActKeyEvent.model_validate_json(entity.payload) for entity in catalog.entities if entity.label == "Event" and json.loads(entity.payload)["event_type"] == "ActKeyEvent"]
    for entity in catalog.entities:
        if entity.label == "Character":
            statement(*NativeCypherBuilder.character_upsert_cypher(next(character for character in characters if character.id == entity.identity), 0))
        elif entity.label in {"Location", "Item"}:
            statement(*NativeCypherBuilder.world_item_upsert_cypher(next(item for item in world_items if item.id == entity.identity), 0))
        else:
            properties = {key: value for key, value in json.loads(entity.payload).items() if value is not None}
            statement(f"CREATE (n:{entity.label}) SET n = $properties, n.created_ts = timestamp(), n.updated_ts = timestamp()", {"properties": properties})
    global_source = catalog.inputs.source("global_outline")
    chapter_parser = ChapterOutlineParser()
    locations = [item for item in world_items if item.category == "location"]
    items = [item for item in world_items if item.category == "object"]
    character_names = [character.name for character in characters]
    for arc in global_source.get("character_arcs", []):
        require(set(arc) == {"character_name", "starting_state", "ending_state", "key_moments"}, "Malformed character arc")
        require(arc["character_name"] in character_names, "Unknown character arc identity")
        statement("MATCH (c:Character {id: $identity}) SET c.arc_start = $start, c.arc_end = $end, c.arc_key_moments = $moments", {
            "identity": names[arc["character_name"]], "start": arc["starting_state"], "end": arc["ending_state"], "moments": arc["key_moments"],
        })
    for sheet in snapshot.characters:
        for target, relationship in json.loads(sheet.relationships).items():
            require(target in character_names, f"Unknown character relationship: {target}")
            require(set(relationship) == {"type", "description"}, "Malformed character relationship")
            edge(names[sheet.name], names[target], relationship["type"], {"type": relationship["type"], "source_profile_managed": True},
                 description=RelationshipDescription(source="profile", description=relationship["description"]))
    for relationship in snapshot.relationships:
        require(isinstance(relationship, CatalogRelationship), "Version 2 requires catalog-ID assertions")
        assert isinstance(relationship, CatalogRelationship)
        edge(relationship.source_id, relationship.target_id, relationship.relationship_type, {"confidence": relationship.confidence},
             description=RelationshipDescription(source="outline", description=relationship.description))

    if items:
        possessions = await extract("initialization/catalog_possessions.j2", {
            "outline_text": encoded(global_source), "known_characters": catalog.candidates("Character"),
            "known_items": catalog.candidates("Item"),
        })
        require(isinstance(possessions, dict) and set(possessions) == {"possessions"} and isinstance(possessions["possessions"], list), "Malformed possessions extraction")
        for possession in possessions["possessions"]:
            require(isinstance(possession, dict) and set(possession) == {"character_id", "item_id"}, "Malformed possession")
            edge(catalog.endpoint(possession["character_id"], "Character"), catalog.endpoint(possession["item_id"], "Item"), "POSSESSES", {})

    for event in act_events:
        # Explicit structural sequence replaces display-text matching of plot-point names.
        major_sequence = 1 if event.act_number == 1 and event.sequence_in_act <= 2 else 2 if event.act_number == 1 else 3 if event.act_number < snapshot.total_acts else 4
        edge(event.id, major_events[major_sequence - 1].id, "PART_OF", {})
        for following in act_events:
            if following.act_number == event.act_number and following.sequence_in_act > event.sequence_in_act:
                edge(event.id, following.id, "HAPPENS_BEFORE", {})
        context = {"event_id": event.id, "event_name": event.name, "event_description": event.description, "event_cause": event.cause, "event_effect": event.effect}
        involved = await extract("initialization/catalog_event_characters.j2", {**context, "known_characters": catalog.candidates("Character")})
        require(isinstance(involved, list), "Character extraction must be an explicit array")
        for involvement in involved:
            require(isinstance(involvement, dict) and set(involvement) == {"character_id", "role"}, "Malformed character involvement")
            require(involvement["role"] is None or isinstance(involvement["role"], str), "Malformed character role")
            # Neo4j represents the producer's explicit unknown role by an absent property.
            edge(event.id, catalog.endpoint(involvement["character_id"], "Character"), "INVOLVES", {} if involvement["role"] is None else {"role": involvement["role"]})
        if locations:
            place = await extract("initialization/catalog_event_location.j2", {**context, "known_locations": catalog.candidates("Location")})
            require(isinstance(place, dict) and set(place) == {"location_id"}, "Malformed location extraction")
            if place["location_id"] is not None:
                edge(event.id, catalog.endpoint(place["location_id"], "Location"), "OCCURS_AT", {})
        if items:
            featured = await extract("initialization/catalog_event_items.j2", {**context, "known_items": catalog.candidates("Item")})
            require(isinstance(featured, dict) and set(featured) == {"featured_items"} and isinstance(featured["featured_items"], list), "Malformed item extraction")
            for featured_item in featured["featured_items"]:
                require(isinstance(featured_item, dict) and set(featured_item) == {"item_id", "role"}, "Malformed featured item")
                require(featured_item["role"] is None or isinstance(featured_item["role"], str), "Malformed item role")
                edge(event.id, catalog.endpoint(featured_item["item_id"], "Item"), "FEATURES_ITEM", {} if featured_item["role"] is None else {"role": featured_item["role"]})

    for chapter_outline in snapshot.chapters:
        number = chapter_outline.chapter_number
        chapter = next(Chapter.model_validate_json(entity.payload) for entity in catalog.entities if entity.label == "Chapter" and json.loads(entity.payload)["number"] == number)
        scenes = [Scene.model_validate_json(entity.payload) for entity in catalog.entities if entity.label == "Scene" and json.loads(entity.payload)["chapter_number"] == number]
        scene_events = [SceneEvent.model_validate_json(entity.payload) for entity in catalog.entities if entity.label == "Event" and json.loads(entity.payload)["event_type"] == "SceneEvent" and json.loads(entity.payload)["chapter_number"] == number]
        for scene in scenes:
            edge(scene.id, chapter.id, "PART_OF", {})
            text = " ".join([scene.setting, scene.plot_point, *scene.beats])
            for character in chapter_parser._extract_characters_from_text(text, character_names):
                edge(scene.id, names[character], "FEATURES_CHARACTER", {"is_pov": character == scene.pov_character})
            for item in items:
                if item.name in chapter_parser._match_items_in_text(text, [item.name]):
                    edge(scene.id, item.id, "FEATURES_ITEM", {})
            for location in locations:
                if chapter_parser._score_location_match(location.name, text) > 0:
                    edge(scene.id, location.id, "OCCURS_AT", {})
        for scene_event in scene_events:
            scene = next(scene for scene in scenes if scene.scene_index == scene_event.scene_index)
            edge(scene_event.id, scene.id, "OCCURS_IN_SCENE", {})
            for character in chapter_parser._extract_characters_from_text(scene_event.name, character_names):
                edge(scene_event.id, names[character], "INVOLVES", {"role": "protagonist" if character == scene_event.pov_character else "participant"})
            match = chapter_parser._find_best_act_key_event(scene_event, [item.model_dump() for item in act_events if item.act_number == scene_event.act_number])
            if match:
                edge(scene_event.id, match["id"], "PART_OF", {})
    if config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE:
        from core.entity_embedding_service import build_entity_embedding_update_statements

        for query, parameters in await build_entity_embedding_update_statements(characters=characters, world_items=world_items):
            statement(query, parameters)
    return CatalogGraphPlan(snapshot=snapshot, entities=catalog.entities, statements=tuple(statements), evidence=tuple(evidence))
