"""Offline native-profile regressions for storage-only prompt projection."""
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import config
from core.exceptions import DatabaseError
from data_access import character_queries, world_queries
from data_access.cache_coordinator import clear_all_data_access_caches
from models import CharacterProfile, WorldItem
from prompts import prompt_data_getters
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager


@pytest.fixture(autouse=True)
def isolated_prompt_cache() -> Iterator[None]:
    assert Path(prompt_data_getters.__file__).resolve() == Path(__file__).resolve().parents[1] / "prompts/prompt_data_getters.py"
    original_configuration = {key: value for key, value in vars(config).items() if key.startswith("ENTITY_EMBEDDING_")}
    prompt_data_getters.clear_context_cache()
    yield
    prompt_data_getters.clear_context_cache()
    clear_all_data_access_caches()
    assert {key: value for key, value in vars(config).items() if key.startswith("ENTITY_EMBEDDING_")} == original_configuration


def storage_properties() -> dict[str, Any]:
    return {
        "entity_embedding_vector": [-0.00016391277313232422] * 1024,
        "entity_embedding_model": "qwen3-embedding:0.6b",
        "entity_embedding_model_identity": "641195a47d15318782d483714e63e93237a1d912f330d0357d6a4d81b0d2918d",
        "entity_embedding_text_hash": "53a8d13b23cf061b09470c42674a3517ef77cba1",
    }


def semantic_properties() -> dict[str, Any]:
    return {
        "description": "The bell rings at 12.75 seconds.",
        "id": "entity_literal_08",
        "created_chapter": 0,
        "chapter_last_updated": 2,
        "is_provisional": True,
        "measurements": [12.75, -3, 1024],
        "coordinates": [[1, 2], [3, 4]],
        "embedding_theory": "A character's theory, not index storage.",
        "relationships": {"Iona": [{"type": "TRUSTS", "chapter_added": 2, "confidence": 0.875, "description": "Keeps the repair secret.", "source": "profile"}]},
    }


@pytest.mark.parametrize("container", ["root", "mapping", "list"])
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.run_settings(
    ENTITY_EMBEDDING_VECTOR_PROPERTY="index_coordinates",
    ENTITY_EMBEDDING_TEXT_HASH_PROPERTY="index_input_digest",
    ENTITY_EMBEDDING_MODEL_PROPERTY="index_encoder",
)
def test_formatter_omits_only_storage_properties(container: str, configured: bool) -> None:
    storage = storage_properties()
    if configured:
        storage = {
            config.ENTITY_EMBEDDING_VECTOR_PROPERTY: storage["entity_embedding_vector"],
            config.ENTITY_EMBEDDING_TEXT_HASH_PROPERTY: storage["entity_embedding_text_hash"],
            config.ENTITY_EMBEDDING_MODEL_PROPERTY: storage["entity_embedding_model"],
            f"{config.ENTITY_EMBEDDING_MODEL_PROPERTY}_identity": storage["entity_embedding_model_identity"],
        }
    semantic = semantic_properties()
    data = {**semantic, **storage}
    if container == "mapping":
        data, semantic = {"details": data}, {"details": semantic}
    elif container == "list":
        data, semantic = {"assertions": [data]}, {"assertions": [semantic]}
    original = deepcopy(data)
    expected = prompt_data_getters._format_dict_for_plain_text_prompt(semantic)
    assert prompt_data_getters._format_dict_for_plain_text_prompt(data) == expected
    assert data == original


@pytest.mark.parametrize("kind", ["character", "world"])
async def test_native_profile_read_to_prompt_retains_canon_not_storage(
    kind: str, owned_graph_cache: None, offline_graph_reads: FakeNeo4jManager,
) -> None:
    node = {
        **semantic_properties(), **storage_properties(), "name": "R08W Bell", "category": "Location",
        "traits": ["precise"], "status": "Active",
        "development_in_chapter_2": "Admits the repair.", "development_in_chapter_9": "Future character fact.",
        "source_quality_chapter_2": "provisional_from_unrevised_draft",
    }
    original = deepcopy(node)
    profile: CharacterProfile | WorldItem | None
    if kind == "character":
        offline_graph_reads.configure_response(r"RETURN\s+c,", [{"c": node, "traits": ["precise"], "relationships": []}])
        profile = await character_queries.get_character_profile_by_name(node["name"])
        assert profile is not None
        original_profile = profile.model_dump()
        result = await prompt_data_getters.get_filtered_character_profiles_for_prompt_plain_text([node["name"]], 2)
        assert "Id: entity_literal_08" in result
        assert "Admits the repair." in result
        assert "Future character fact." not in result
        assert offline_graph_reads.executed_queries[0][1] == {"name": node["name"], "include_provisional": False}
    else:
        offline_graph_reads.configure_response(r"RETURN we", [{"we": node}])
        offline_graph_reads.configure_response(r"ELABORATED_IN_CHAPTER", [
            {"chapter": 2, "summary": "Bell repair is visible.", "is_provisional": True},
            {"chapter": 9, "summary": "Future world fact.", "is_provisional": False},
        ])
        profile = await world_queries.get_world_item_by_id(node["id"])
        assert profile is not None
        original_profile = profile.model_dump()
        result = await prompt_data_getters.get_filtered_world_data_for_prompt_plain_text({"Location": [node["id"]]}, 2)
        assert "Bell repair is visible." in result
        assert "Future world fact." not in result
        assert offline_graph_reads.executed_queries[0][1] == {"id": node["id"], "include_provisional": False}
        assert profile.id == node["id"]
    assert node["name"] in result
    assert "The bell rings at 12.75 seconds." in result
    assert "Is provisional: True" in result
    assert "Created chapter: 0" in result
    assert "Chapter last updated: 2" in result
    assert "Data from Chapter 2 may be provisional (from unrevised draft)." in result
    assert "Entity embedding" not in result
    assert "-0.00016391277313232422" not in result
    assert node == original
    assert profile.model_dump() == original_profile
    assert profile.to_dict()["entity_embedding_vector"] == storage_properties()["entity_embedding_vector"]
    assert profile.to_dict()["entity_embedding_model_identity"] == storage_properties()["entity_embedding_model_identity"]


async def test_required_profile_query_failure_remains_failure(
    owned_graph_cache: None, offline_graph_reads: FakeNeo4jManager,
) -> None:
    def fail(parameters: dict[str, Any] | None) -> list[dict[str, Any]]:
        raise DatabaseError("r08w synthetic profile read failure")

    offline_graph_reads.configure_response(r"RETURN\s+c,", fail)
    with pytest.raises(DatabaseError, match="r08w synthetic profile read failure"):
        await prompt_data_getters.get_filtered_character_profiles_for_prompt_plain_text(["R08W Bell"], 2)
