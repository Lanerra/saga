"""Behavior and live ownership of shared scene parsing and Location persistence."""

from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from core.langgraph.nodes import context_character_retrieval, context_retrieval_node, context_world_retrieval
from core.parsers.chapter_outline_parser import ChapterOutlineParser
from core.parsers.global_outline_parser import GlobalOutlineParser
from core.service_context import get_services
from models.kg_models import Location

CONTEXT_MODULES = (context_character_retrieval, context_retrieval_node, context_world_retrieval)
LOCATION_QUERY = """
MERGE (l:Location {id: $id})
ON CREATE SET
    l.name = $name,
    l.description = $description,
    l.category = $category,
    l.created_chapter = $created_chapter,
    l.is_provisional = $is_provisional,
    l.created_ts = timestamp(),
    l.updated_ts = timestamp()
ON MATCH SET
    l.name = $name,
    l.description = $description,
    l.category = $category,
    l.created_chapter = $created_chapter,
    l.is_provisional = $is_provisional,
    l.updated_ts = timestamp()
"""


@pytest.mark.parametrize("module", CONTEXT_MODULES)
@pytest.mark.parametrize(
    ("scene", "expected"),
    [
        ({}, []),
        ({"characters": []}, []),
        ({"characters": [" Alice ", "Bob", "Alice", "", "\t", "alice", "Bób"]}, ["Alice", "Bob", "alice", "Bób"]),
    ],
)
def test_scene_character_policy_preserves_input(module: Any, scene: dict, expected: list[str]) -> None:
    original = deepcopy(scene)
    assert module.extract_scene_characters(scene) == expected
    assert scene == original


@pytest.mark.parametrize("module", CONTEXT_MODULES)
def test_scene_character_policy_has_one_pure_owner(module: Any) -> None:
    parser = getattr(module, "extract_scene_characters", None)
    assert callable(parser), "Live context consumers must use the shared scene-plan parser"
    assert parser.__module__ == "processing.scene_plan_parser"
    assert Path(inspect.getfile(parser)).resolve() == Path(__file__).resolve().parents[1] / "processing/scene_plan_parser.py"


@pytest.mark.parametrize("parser_class", [GlobalOutlineParser, ChapterOutlineParser])
@pytest.mark.parametrize("fail_at", [0, 1, 2])
async def test_location_persistence_preserves_query_order_and_failure(
    parser_class: Any, fail_at: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    owners: list[str] = []

    async def record_write(query: str, parameters: dict[str, Any]) -> None:
        frame = inspect.currentframe()
        assert frame is not None and frame.f_back is not None
        owners.append(frame.f_back.f_globals["__name__"])
        calls.append((" ".join(query.split()), dict(parameters)))
        if len(calls) == fail_at:
            raise RuntimeError("synthetic location write failure")

    monkeypatch.setattr(get_services().database, "execute_write_query", record_write)
    locations = [
        Location(id="location_gate", name="Gate", description="Stone arch", created_chapter=0, is_provisional=False),
        Location(id="location_cove", name=None, description="Sheltered water", created_chapter=4, is_provisional=True),
    ]
    original = [location.model_dump() for location in locations]
    result = await parser_class(chapter_number=7).create_location_nodes(locations)
    assert result is (fail_at == 0)
    expected = [
        (" ".join(LOCATION_QUERY.split()), {
            "id": "location_gate", "name": "Gate", "description": "Stone arch", "category": "Location", "created_chapter": 0, "is_provisional": False,
        }),
        (" ".join(LOCATION_QUERY.split()), {
            "id": "location_cove", "name": None, "description": "Sheltered water", "category": "Location", "created_chapter": 4, "is_provisional": True,
        }),
    ]
    assert calls == expected[:fail_at or 2]
    assert [location.model_dump() for location in locations] == original
    assert owners == ["data_access.location_queries"] * len(calls)


@pytest.mark.parametrize("parser_class", [GlobalOutlineParser, ChapterOutlineParser])
async def test_empty_locations_do_not_write(parser_class: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    async def unexpected_write(query: str, parameters: dict[str, Any]) -> None:
        raise AssertionError("Empty Location input must not write")

    monkeypatch.setattr(get_services().database, "execute_write_query", unexpected_write)
    assert await parser_class().create_location_nodes([]) is True
