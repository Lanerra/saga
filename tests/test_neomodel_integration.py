import pytest
from unittest.mock import AsyncMock

from data_access.character_queries import _save_character_ogm
from data_access.models import CharacterNode, TraitNode
from kg_maintainer.models import CharacterProfile


@pytest.mark.asyncio
async def test_save_character_ogm(monkeypatch):
    mock_char = AsyncMock()
    mock_char.traits.connect = AsyncMock()
    nodes_mgr = AsyncMock()
    nodes_mgr.get_or_none = AsyncMock(return_value=None)
    monkeypatch.setattr(CharacterNode, "nodes", nodes_mgr)
    monkeypatch.setattr(CharacterNode, "save", AsyncMock(return_value=mock_char))

    trait_mgr = AsyncMock()
    trait_mgr.get_or_none = AsyncMock(return_value=None)
    trait_instance = AsyncMock()
    monkeypatch.setattr(TraitNode, "nodes", trait_mgr)
    monkeypatch.setattr(TraitNode, "save", AsyncMock(return_value=trait_instance))

    profile = CharacterProfile(name="Alice", description="desc", traits=["brave"])
    await _save_character_ogm(profile)

    nodes_mgr.get_or_none.assert_awaited_with(name="Alice")
    assert CharacterNode.save.await_count
    trait_mgr.get_or_none.assert_awaited()
    mock_char.traits.connect.assert_awaited()
