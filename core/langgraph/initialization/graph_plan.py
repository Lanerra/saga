"""One pre-transaction domain producer; frozen plans never call a provider."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel

import config
from core.langgraph.initialization.snapshot import FrozenPayload, InitializationSnapshot, Relationship, digest, encoded, require
from core.parsers.act_outline_parser import ActOutlineParser
from core.parsers.chapter_outline_parser import ChapterOutlineParser
from core.parsers.global_outline_parser import GlobalOutlineParser
from core.service_context import get_services
from data_access.cypher_builders.native_builders import NativeCypherBuilder
from models.kg_models import CharacterProfile
from prompts.prompt_renderer import get_system_prompt, render_prompt
from utils.text_processing import generate_entity_id


class Statement(FrozenPayload):
    query: str
    parameters: str


class ProducerEvidence(FrozenPayload):
    model: str
    template: str
    prompt_checksum: str
    response: str


class GraphEntity(FrozenPayload):
    label: Literal["Character", "Location", "Item", "Event", "Scene", "Chapter"]
    identity: str
    payload: str


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


async def produce_plan(snapshot: InitializationSnapshot) -> InitializationGraphPlan:
    from core.langgraph.initialization.commit_init_node import _parse_world_items_extraction

    statements: list[Statement] = []
    entities: list[GraphEntity] = []
    evidence: list[ProducerEvidence] = []
    names: dict[str, str] = {}
    identifiers: set[str] = set()
    edge_identities: dict[tuple[str, str, str], int] = {}
    edge_properties: dict[tuple[str, str, str], dict[str, Any]] = {}
    edge_descriptions: dict[tuple[str, str, str], set[RelationshipDescription]] = {}

    def statement(query: str, parameters: dict[str, Any]) -> None:
        statements.append(Statement(query=query, parameters=encoded(parameters)))

    def entity(label: Any, identity: str, payload: dict[str, Any], *, named: bool = False) -> None:
        require(identity not in identifiers, f"Duplicate graph identity: {identity}")
        identifiers.add(identity)
        if named:
            name = payload["name"]
            require(name.casefold() not in {key.casefold() for key in names}, f"Ambiguous entity name: {name}")
            names[name] = identity
        entities.append(GraphEntity(label=label, identity=identity, payload=encoded(payload)))

    def node(label: Any, model: BaseModel) -> None:
        properties = model.model_dump(mode="json", exclude_none=True)
        if label == "Chapter":
            properties["generation_status"] = "planned"
        identity = properties["id"]
        entity(label, identity, properties)
        statement(f"CREATE (n:{label}) SET n = $properties, n.created_ts = timestamp(), n.updated_ts = timestamp()", {"properties": properties})

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
        result = json.loads(response)
        evidence.append(ProducerEvidence(model=config.NARRATIVE_MODEL, template=template, prompt_checksum=digest(prompt), response=response))
        return result

    characters = []
    for sheet in snapshot.characters:
        profile = CharacterProfile(
            name=sheet.name, id=generate_entity_id(sheet.name, "character"), personality_description=sheet.description, traits=list(sheet.traits), status=sheet.status,
            created_chapter=0, is_provisional=False, physical_description=sheet.physical_description,
            relationships={}, motivations=sheet.motivations, background=sheet.background, skills=list(sheet.skills),
            internal_conflict=sheet.internal_conflict, is_protagonist=sheet.is_protagonist,
        )
        identity = profile.id
        entity("Character", identity, profile.model_dump(mode="json"), named=True)
        statement(*NativeCypherBuilder.character_upsert_cypher(profile, 0))
        characters.append(profile)

    global_source = snapshot.source("global_outline")
    act_source = snapshot.source("act_outlines")
    global_parser = GlobalOutlineParser()
    act_parser = ActOutlineParser()
    chapter_parser = ChapterOutlineParser()
    major_events = global_parser._parse_major_plot_points(global_source)
    act_events = act_parser._parse_act_key_events(act_source)
    require({event.act_number for event in act_events} == set(range(1, snapshot.total_acts + 1)), "Incomplete act event coverage")
    for event in [*major_events, *act_events]:
        node("Event", event)

    # Global and act prose share one extraction result and one world writer.
    world_result = await extract("knowledge_agent/extract_world_items_lines.j2", {
        "setting": json.loads(snapshot.metadata)["setting"],
        "outline_text": encoded({"global_outline": global_source, "act_outlines": act_source}),
    })
    world_items = _parse_world_items_extraction(encoded(world_result))
    for item in world_items:
        label = "Location" if item.category == "location" else "Item"
        entity(label, item.id, item.model_dump(mode="json"), named=True)
        statement(*NativeCypherBuilder.world_item_upsert_cypher(item, 0))
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
        require(relationship.source_name in names and relationship.target_name in names, "Unknown outline relationship identity")
        edge(names[relationship.source_name], names[relationship.target_name], relationship.relationship_type, {"confidence": relationship.confidence},
             description=RelationshipDescription(source="outline", description=relationship.description))

    if items:
        possessions = await extract("knowledge_agent/extract_item_possession.j2", {
            "outline_text": encoded(global_source), "known_characters": character_names,
            "known_items": [{"name": item.name, "description": item.description} for item in items],
        })
        require(isinstance(possessions, dict) and set(possessions) == {"possessions"} and isinstance(possessions["possessions"], list), "Malformed possessions extraction")
        for possession in possessions["possessions"]:
            require(set(possession) == {"character", "item"}, "Malformed possession")
            require(possession["character"] in character_names and possession["item"] in [item.name for item in items], "Unknown possession endpoint")
            edge(names[possession["character"]], names[possession["item"]], "POSSESSES", {})

    for event in act_events:
        # Explicit structural sequence replaces display-text matching of plot-point names.
        major_sequence = 1 if event.act_number == 1 and event.sequence_in_act <= 2 else 2 if event.act_number == 1 else 3 if event.act_number < snapshot.total_acts else 4
        edge(event.id, major_events[major_sequence - 1].id, "PART_OF", {})
        for following in act_events:
            if following.act_number == event.act_number and following.sequence_in_act > event.sequence_in_act:
                edge(event.id, following.id, "HAPPENS_BEFORE", {})
        context = {"event_name": event.name, "event_description": event.description, "event_cause": event.cause, "event_effect": event.effect}
        involved = await extract("knowledge_agent/extract_event_characters.j2", {**context, "known_characters": character_names})
        require(isinstance(involved, list), "Character extraction must be an explicit array")
        for involvement in involved:
            require(isinstance(involvement, dict) and set(involvement) == {"name", "role"}, "Malformed character involvement")
            require(involvement["name"] in character_names and (involvement["role"] is None or isinstance(involvement["role"], str)), "Unknown character involvement")
            # Neo4j represents the producer's explicit unknown role by an absent property.
            edge(event.id, names[involvement["name"]], "INVOLVES", {} if involvement["role"] is None else {"role": involvement["role"]})
        if locations:
            place = await extract("knowledge_agent/extract_event_location.j2", {**context, "known_locations": [{"name": location.name, "description": location.description} for location in locations]})
            require(isinstance(place, dict) and set(place) == {"location"}, "Malformed location extraction")
            if place["location"] is not None:
                require(place["location"] in [location.name for location in locations], "Unknown event location")
                edge(event.id, names[place["location"]], "OCCURS_AT", {})
        if items:
            featured = await extract("knowledge_agent/extract_event_item.j2", {**context, "known_items": [{"name": item.name, "description": item.description} for item in items]})
            require(isinstance(featured, dict) and set(featured) == {"featured_items"} and isinstance(featured["featured_items"], list), "Malformed item extraction")
            for featured_item in featured["featured_items"]:
                require(set(featured_item) == {"item", "role"} and featured_item["item"] in [item.name for item in items], "Unknown featured item")
                edge(event.id, names[featured_item["item"]], "FEATURES_ITEM", {"role": featured_item["role"]})

    for chapter_outline in snapshot.chapters:
        source = chapter_outline.model_dump(mode="json")
        chapter = chapter_parser._parse_chapter(source)
        scenes = chapter_parser._parse_scenes(source, character_names)
        scene_events = chapter_parser._parse_scene_events(source, character_names)
        node("Chapter", chapter)
        for scene in scenes:
            node("Scene", scene)
            edge(scene.id, chapter.id, "PART_OF", {})
            text = " ".join([scene.setting, scene.plot_point, *scene.beats])
            for character in chapter_parser._extract_characters_from_text(text, character_names):
                edge(scene.id, names[character], "FEATURES_CHARACTER", {"is_pov": character == scene.pov_character})
            for item_name in chapter_parser._match_items_in_text(text, [item.name for item in items]):
                edge(scene.id, names[item_name], "FEATURES_ITEM", {})
            for location in locations:
                if chapter_parser._score_location_match(location.name, text) > 0:
                    edge(scene.id, location.id, "OCCURS_AT", {})
        for event in scene_events:
            node("Event", event)
            scene = next(scene for scene in scenes if scene.scene_index == event.scene_index)
            edge(event.id, scene.id, "OCCURS_IN_SCENE", {})
            for character in chapter_parser._extract_characters_from_text(event.name, character_names):
                edge(event.id, names[character], "INVOLVES", {"role": "protagonist" if character == event.pov_character else "participant"})
            match = chapter_parser._find_best_act_key_event(event, [item.model_dump() for item in act_events if item.act_number == event.act_number])
            if match:
                edge(event.id, match["id"], "PART_OF", {})
    if config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE:
        from core.entity_embedding_service import build_entity_embedding_update_statements

        for query, parameters in await build_entity_embedding_update_statements(characters=characters, world_items=world_items):
            statement(query, parameters)
    return InitializationGraphPlan(snapshot=snapshot, entities=tuple(entities), statements=tuple(statements), evidence=tuple(evidence))
