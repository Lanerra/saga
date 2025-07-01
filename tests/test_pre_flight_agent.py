# tests/test_pre_flight_agent.py
from typing import Any
from unittest.mock import AsyncMock

import pytest
from agents.pre_flight_check_agent import PreFlightCheckAgent
from core.db_manager import neo4j_manager
from core.llm_interface import llm_service
from data_access import character_queries, world_queries
from kg_maintainer.models import WorldItem


@pytest.mark.asyncio
async def test_preflight_resolves_trait(monkeypatch):
    agent = PreFlightCheckAgent()
    monkeypatch.setattr(
        agent,
        "_identify_contradictory_pairs",
        AsyncMock(return_value=[("Incorporeal", "Corporeal")]),
    )
    monkeypatch.setattr(agent, "_gather_canonical_facts", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        neo4j_manager,
        "execute_read_query",
        AsyncMock(return_value=[{"c": "Saga"}]),
    )

    removed: dict[str, str] = {}

    async def fake_remove(name: str, trait: str) -> bool:
        removed["name"] = name
        removed["trait"] = trait
        return True

    monkeypatch.setattr(
        character_queries,
        "remove_character_trait",
        AsyncMock(side_effect=fake_remove),
    )
    monkeypatch.setattr(
        llm_service,
        "async_call_llm",
        AsyncMock(return_value=('{"trait": "Incorporeal"}', {})),
    )

    await agent.perform_core_checks(
        {"protagonist_name": "Saga"},
        {"Saga": {}},
        {},
    )
    assert removed["trait"] == "Corporeal"
    assert removed["name"] in {"Saga", "S\xe1g\xe1"}


@pytest.mark.asyncio
async def test_preflight_resolves_world_trait(monkeypatch):
    agent = PreFlightCheckAgent()
    monkeypatch.setattr(
        agent,
        "_identify_contradictory_pairs",
        AsyncMock(return_value=[("Incorporeal", "Corporeal")]),
    )
    monkeypatch.setattr(agent, "_gather_canonical_facts", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        neo4j_manager,
        "execute_read_query",
        AsyncMock(return_value=[{"we": "w"}]),
    )

    removed: dict[str, str] = {}

    async def fake_remove(wid: str, trait: str) -> bool:
        removed["id"] = wid
        removed["trait"] = trait
        return True

    monkeypatch.setattr(
        world_queries,
        "remove_world_element_trait_aspect",
        AsyncMock(side_effect=fake_remove),
    )
    monkeypatch.setattr(
        llm_service,
        "async_call_llm",
        AsyncMock(return_value=('{"trait": "Incorporeal"}', {})),
    )

    world_data = {"loc": {"city": WorldItem(id="city", category="loc", name="city")}}
    await agent.perform_core_checks({}, {}, world_data)
    assert removed == {"id": "city", "trait": "Corporeal"}


@pytest.mark.asyncio
async def test_preflight_enforces_canon(monkeypatch):
    agent = PreFlightCheckAgent()
    monkeypatch.setattr(
        agent,
        "_identify_contradictory_pairs",
        AsyncMock(return_value=[("Incorporeal", "Corporeal")]),
    )
    monkeypatch.setattr(
        agent,
        "_gather_canonical_facts",
        AsyncMock(
            return_value=[
                {
                    "name": "Ságá",
                    "trait": "Corporeal",
                    "conflicts_with": "Incorporeal",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        neo4j_manager,
        "execute_read_query",
        AsyncMock(return_value=[{"c": "Saga"}]),
    )

    removed: dict[str, str] = {}

    async def fake_remove(name: str, trait: str) -> bool:
        removed["name"] = name
        removed["trait"] = trait
        return True

    monkeypatch.setattr(
        character_queries,
        "remove_character_trait",
        AsyncMock(side_effect=fake_remove),
    )

    monkeypatch.setattr(
        llm_service,
        "async_call_llm",
        AsyncMock(return_value=('{"trait": "Corporeal"}', {})),
    )

    await agent.perform_core_checks({"protagonist_name": "Saga"}, {"Saga": {}}, {})
    assert removed == {"name": "Ságá", "trait": "Incorporeal"}


@pytest.mark.asyncio
async def test_gather_canonical_facts(monkeypatch):
    agent = PreFlightCheckAgent()

    async def fake_query(query: str, params: dict[str, Any] | None = None):
        assert "is_canonical_truth" in query
        return [{"name": "Saga", "trait": "Corporeal"}]

    monkeypatch.setattr(
        neo4j_manager,
        "execute_read_query",
        AsyncMock(side_effect=fake_query),
    )
    monkeypatch.setattr(
        llm_service,
        "async_call_llm",
        AsyncMock(return_value=('[{"conflicts_with": "Incorporeal"}]', {})),
    )

    facts = await agent._gather_canonical_facts({"title": "T"})
    assert facts == [
        {"name": "Saga", "trait": "Corporeal", "conflicts_with": "Incorporeal"}
    ]
