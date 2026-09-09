"""Execute production filter expressions in a SQL three-valued-logic model.

SQL and Cypher share the boolean/null behavior used by these expressions.
This is not an actual Neo4j or vector-index integration test.
"""

import re
import sqlite3
from typing import Any

import numpy as np
import pytest

import config
from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes import context_scene_retrieval
from core.service_context import get_services
from data_access import chapter_queries


@pytest.fixture
async def production_query(monkeypatch: pytest.MonkeyPatch) -> tuple[str, dict[str, Any]]:
    monkeypatch.setattr(config, "EXPECTED_EMBEDDING_DIM", 2)
    captured: list[tuple[str, dict[str, Any]]] = []

    async def read(query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        captured.append((query, parameters))
        return [{"context_chapters": []}]

    monkeypatch.setattr(get_services().database, "execute_read_query", read)
    assert await chapter_queries.find_semantic_context_native(np.array([1.0, 0.0]), current_chapter_number=5, limit=3, embedding_model=config.EMBEDDING_MODEL) == []
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize("branch", ["similar_c", "prev_c"])
@pytest.mark.parametrize("include_provisional", [False, True])
@pytest.mark.parametrize("properties", [{}, {"is_provisional": None}, {"is_provisional": False}, {"is_provisional": True}], ids=["missing", "null", "final", "provisional"])
@pytest.mark.parametrize("temporal", [{}, {"number": None}, {"number": 4}, {"number": 5}, {"number": 6}], ids=["missing-number", "null-number", "past", "current", "future"])
def test_production_predicate_truth_table(
    production_query: tuple[str, dict[str, Any]], branch: str, include_provisional: bool,
    properties: dict[str, Any], temporal: dict[str, Any],
) -> None:
    query, parameters = production_query
    ending = r"\s*//" if branch == "similar_c" else r"\s*AND NOT ANY"
    match = re.search(rf"WHERE ({branch}\.number < .*?){ending}", query, flags=re.DOTALL)
    assert match is not None
    expression = match[1].replace(f"{branch}.number", "$number").replace(f"{branch}.is_provisional", "$is_provisional")
    with sqlite3.connect(":memory:") as database:
        value = database.execute(
            "SELECT " + expression,
            {**parameters, "number": temporal.get("number"), "is_provisional": properties.get("is_provisional"), "include_provisional": include_provisional},
        ).fetchone()[0]
    past = temporal.get("number") is not None and temporal["number"] < parameters["current_chapter"]
    expected = past and (include_provisional or properties.get("is_provisional") is not True)
    assert (value == 1) is expected


async def test_scene_caller_uses_non_provisional_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setattr(config, "EXPECTED_EMBEDDING_DIM", 2)
    captured: list[dict[str, Any]] = []

    async def embedding(text: str) -> np.ndarray:
        return np.array([1.0, 0.0])

    async def read(query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        captured.append(parameters)
        return [{"context_chapters": []}]

    monkeypatch.setattr(get_services().language_model, "async_get_embedding", embedding)
    monkeypatch.setattr(get_services().database, "execute_read_query", read)
    assert await context_scene_retrieval.get_semantic_context({}, "synthetic scene", 5, "synthetic", ContentManager(str(tmp_path))) is None
    assert len(captured) == 1
    assert captured[0]["include_provisional"] is False
    assert captured[0]["current_chapter"] == 5
    assert captured[0]["prev_chapter_num"] == 4
    assert captured[0]["final_limit"] == 3


@pytest.mark.parametrize("include_provisional", [False, True])
async def test_requested_boolean_reaches_query(monkeypatch: pytest.MonkeyPatch, include_provisional: bool) -> None:
    monkeypatch.setattr(config, "EXPECTED_EMBEDDING_DIM", 2)
    captured: list[dict[str, Any]] = []

    async def read(query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        captured.append(parameters)
        return [{"context_chapters": []}]

    monkeypatch.setattr(get_services().database, "execute_read_query", read)
    assert await chapter_queries.find_semantic_context_native(np.array([1.0, 0.0]), 5, include_provisional=include_provisional, embedding_model=config.EMBEDDING_MODEL) == []
    assert captured[0]["include_provisional"] is include_provisional
