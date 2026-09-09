"""Character domain ownership and enrichment handoff regressions."""

import json
from pathlib import Path
from typing import Any

import pytest

from core.langgraph.initialization.graph_plan import produce_plan
from core.langgraph.initialization.snapshot import select_snapshot
from core.parsers.narrative_enrichment_parser import NarrativeEnrichmentParser, PhysicalDescriptionExtractionResult
from core.service_context import get_services
from data_access import character_queries
from data_access.cypher_builders.native_builders import NativeCypherBuilder
from models.kg_models import CharacterProfile
from tests.test_staged_initialization import example_state, with_catalog

DOMAIN = {
    "motivations": "Discover", "background": "Harbor", "skills": ["navigation"],
    "internal_conflict": "Duty", "is_protagonist": True, "physical_description": "Silver hair",
}


def test_native_record_readers_retain_identity_and_domain() -> None:
    node = {"name": "Ada", "id": "character_ada", **DOMAIN}
    for profile in (CharacterProfile.from_dict_record({"c": node}), CharacterProfile.from_db_node(node)):
        assert profile.id == "character_ada"
        assert {key: profile.to_dict()[key] for key in DOMAIN} == DOMAIN


def test_native_writer_owns_only_explicit_domain_fields() -> None:
    profile = CharacterProfile.from_dict("Ada", {"id": "character_ada", **DOMAIN, "author_note": {"private": True}})
    _, parameters = NativeCypherBuilder.character_upsert_cypher(profile, 0)
    assert parameters["domain_properties"] == DOMAIN
    _, shallow = NativeCypherBuilder.character_upsert_cypher(CharacterProfile(name="Ada"), 1)
    assert shallow["domain_properties"] == {}
    _, cleared = NativeCypherBuilder.character_upsert_cypher(CharacterProfile(name="Ada", skills=[], is_protagonist=False, physical_description=""), 2)
    assert cleared["domain_properties"] == {"skills": [], "is_protagonist": False, "physical_description": ""}


async def test_frozen_producer_supplies_typed_domain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def provider(**arguments: Any) -> tuple[str, dict[str, Any]]:
        return "[]", {}

    monkeypatch.setattr(get_services().language_model, 'async_call_llm', provider)
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    plan = await produce_plan(select_snapshot(with_catalog(example_state(tmp_path))))
    character = next(entity for entity in plan.entities if entity.label == "Character")
    payload = json.loads(character.payload)
    assert {key: payload[key] for key in DOMAIN if key != "physical_description"} == {key: value for key, value in DOMAIN.items() if key != "physical_description"}
    assert payload["updates"] == {}
    parameters = json.loads(plan.statements[0].parameters)
    assert parameters["id"] == character.identity
    assert parameters["domain_properties"] == {**DOMAIN, "physical_description": ""}


async def test_parser_identical_description_issues_no_write(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = CharacterProfile(name="Ada", id="character_ada", physical_description="Silver hair")
    writes: list[Any] = []

    async def read(query: str, parameters: Any = None) -> list[dict[str, Any]]:
        return [{"c": profile.model_dump(), "traits": [], "relationships": []}]

    async def write(statements: Any) -> None:
        writes.append(statements)

    monkeypatch.setattr(get_services().database, "execute_read_query", read)
    monkeypatch.setattr(get_services().database, "execute_cypher_batch", write)
    monkeypatch.setattr("config.ENABLE_PHYSICAL_DESCRIPTION_VALIDATION", False)
    parser = NarrativeEnrichmentParser("Ada has silver hair.", 1)
    assert await parser.update_character_physical_descriptions([PhysicalDescriptionExtractionResult(character_name="Ada", extracted_description="Silver hair")]) is True
    assert writes == []


async def test_parser_unknown_character_is_not_success(monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[Any] = []

    async def read(query: str, parameters: Any = None) -> list[dict[str, Any]]:
        return [{"c": {"name": "Ada", "id": "character_ada"}, "traits": [], "relationships": []}]

    async def write(statements: Any) -> None:
        writes.append(statements)

    monkeypatch.setattr(get_services().database, "execute_read_query", read)
    monkeypatch.setattr(get_services().database, "execute_cypher_batch", write)
    parser = NarrativeEnrichmentParser("Missing has silver hair.", 1)
    assert await parser.update_character_physical_descriptions([PhysicalDescriptionExtractionResult(character_name="Missing", extracted_description="Silver hair")]) is False
    assert writes == []


async def test_physical_only_sync_requires_stable_identity() -> None:
    with pytest.raises(ValueError, match="stable ID"):
        await character_queries.sync_characters([CharacterProfile(name="Ada", physical_description="Silver hair")], 1, physical_description_only=True)
