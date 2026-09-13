"""Exact-name query maps and stable-ID reads with SQLite-backed data."""
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import config
from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.relationship_normalization_node import normalize_relationships
from core.langgraph.state import NarrativeState
from core.service_context import get_services
from data_access import character_queries, world_queries
from data_access.cache_coordinator import clear_character_read_caches, clear_world_read_caches
from models import CharacterProfile, WorldItem


@pytest.mark.parametrize("rebuild", [True, False])
def test_name_maps_preserve_articles_case_and_apostrophes(rebuild: bool) -> None:
    names = ["The Doctor", "Doctor", "King's Cross", "Kings Cross", "The Hague", "Hague"]
    characters = [CharacterProfile(name=name, id=f"person-{index}") for index, name in enumerate(names)]
    places = [WorldItem(name=name, category="Location", id=f"place-{index}") for index, name in enumerate(names)]
    character_queries.CHAR_NAME_TO_CANONICAL.clear()
    world_queries.WORLD_NAME_TO_ID.clear()
    try:
        (character_queries.rebuild_character_name_map if rebuild else character_queries.update_character_name_map)(characters)
        (world_queries.rebuild_world_name_map if rebuild else world_queries.update_world_name_map)(places)
        assert [character_queries.resolve_character_name(name) for name in names] == names
        assert [world_queries.resolve_world_name(name) for name in names] == [place.id for place in places]
        assert world_queries.resolve_world_name("the hague") is None
        assert character_queries.resolve_character_name("the doctor") == "the doctor"
    finally:
        character_queries.CHAR_NAME_TO_CANONICAL.clear()
        world_queries.WORLD_NAME_TO_ID.clear()


class SQLiteReadStore:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute("CREATE TABLE entities (lookup_id TEXT PRIMARY KEY, payload TEXT)")
        self.connection.execute("INSERT INTO entities VALUES (?, ?)", ("place-17", json.dumps(payload)))
        self.parameters: list[dict[str, Any]] = []

    async def verify_project_ownership(self) -> None:
        return None

    async def execute_read_query(self, query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        self.parameters.append(parameters)
        if "ELABORATED_IN_CHAPTER" in query:
            return []
        rows = self.connection.execute("SELECT payload FROM entities WHERE lookup_id = ?", (parameters["id"],)).fetchall()
        return [{"we": json.loads(row[0])} for row in rows]


@pytest.mark.parametrize("identifier", [None, "", "other-id"])
async def test_id_reads_do_not_fabricate_or_substitute_identity(monkeypatch: pytest.MonkeyPatch, identifier: str | None) -> None:
    store = SQLiteReadStore({"name": "The Hague", "category": "Location", "id": identifier})
    monkeypatch.setattr(get_services(), "database", store)
    clear_world_read_caches()
    try:
        with pytest.raises(ValueError, match="ID|identity"):
            await world_queries.get_world_item_by_id("place-17")
    finally:
        clear_world_read_caches()
        store.connection.close()


async def test_by_id_does_not_treat_unresolved_id_as_display_name(monkeypatch: pytest.MonkeyPatch) -> None:
    store = SQLiteReadStore({"name": "The Hague", "category": "Location", "id": "place-17"})
    monkeypatch.setattr(get_services(), "database", store)
    clear_world_read_caches()
    world_queries.rebuild_world_name_map([WorldItem(name="The Hague", category="Location", id="place-17")])
    try:
        assert await world_queries.get_world_item_by_id("The Hague") is None
        assert store.parameters == [{"id": "The Hague", "include_provisional": False}]
        result = await world_queries.get_world_item_by_id("place-17")
        assert result is not None and result.id == "place-17" and result.name == "The Hague"
    finally:
        clear_world_read_caches()
        world_queries.WORLD_NAME_TO_ID.clear()
        store.connection.close()


async def test_normalization_cannot_publish_partial_parse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ENABLE_RELATIONSHIP_NORMALIZATION", True)
    manager = ContentManager(str(tmp_path))
    reference = manager.save_json([{"source_name": "Mara"}], "extracted_relationships", "chapter_1", 1)
    state: NarrativeState = {"project_dir": str(tmp_path), "current_chapter": 1, "extracted_relationships_ref": reference}
    with pytest.raises(ValueError):
        await normalize_relationships(state)
    assert manager.get_latest_version("extracted_relationships", "chapter_1") == 1


async def test_character_query_preserves_literal_display_name(monkeypatch: pytest.MonkeyPatch) -> None:
    class CharacterReadStore:
        async def verify_project_ownership(self) -> None:
            return None

        async def execute_read_query(self, query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
            assert parameters == {"name": "The Doctor", "include_provisional": False}
            return [{"c": {"name": "The Doctor", "id": "person-17"}, "traits": [], "relationships": []}]

    character_queries.rebuild_character_name_map([CharacterProfile(name="Doctor", id="person-other"), CharacterProfile(name="The Doctor", id="person-17")])
    monkeypatch.setattr(get_services(), "database", CharacterReadStore())
    clear_character_read_caches()
    try:
        result = await character_queries.get_character_profile_by_name("The Doctor")
        assert result is not None and result.name == "The Doctor" and result.id == "person-17"
    finally:
        clear_character_read_caches()
        character_queries.CHAR_NAME_TO_CANONICAL.clear()
