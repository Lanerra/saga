"""Tests for core/knowledge_graph_service.py"""

from unittest.mock import AsyncMock, patch

import pytest

from core.exceptions import KnowledgeGraphPersistenceError
from core.knowledge_graph_service import KnowledgeGraphService
from data_access.cypher_builders.native_builders import NativeCypherBuilder
from models.kg_models import CharacterProfile, WorldItem


def _make_character(name: str) -> CharacterProfile:
    return CharacterProfile(name=name)


def _make_world_item(category: str, name: str, description: str) -> WorldItem:
    return WorldItem.from_dict(category, name, {"description": description})


@pytest.mark.asyncio
class TestPersistEntitiesEmptyInput:

    async def test_returns_true_without_calling_neo4j(self) -> None:
        with patch("core.knowledge_graph_service.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_cypher_batch = AsyncMock()
            service = KnowledgeGraphService(cypher_builder=NativeCypherBuilder())

            result = await service.persist_entities()

            assert result is True
            fake_neo4j.execute_cypher_batch.assert_not_called()

    async def test_explicit_empty_lists_without_calling_neo4j(self) -> None:
        with patch("core.knowledge_graph_service.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_cypher_batch = AsyncMock()
            service = KnowledgeGraphService(cypher_builder=NativeCypherBuilder())

            result = await service.persist_entities(
                characters=[], world_items=[], extra_statements=[]
            )

            assert result is True
            fake_neo4j.execute_cypher_batch.assert_not_called()


@pytest.mark.asyncio
class TestPersistEntitiesCharacters:

    async def test_single_character_upsert(self) -> None:
        with patch("core.knowledge_graph_service.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_cypher_batch = AsyncMock()
            service = KnowledgeGraphService(cypher_builder=NativeCypherBuilder())

            result = await service.persist_entities(characters=[_make_character("Alice")])

            assert result is True
            fake_neo4j.execute_cypher_batch.assert_called_once()
            statements = fake_neo4j.execute_cypher_batch.call_args[0][0]
            assert len(statements) == 1
            assert statements[0][1]["name"] == "Alice"

    async def test_multiple_characters_upsert(self) -> None:
        with patch("core.knowledge_graph_service.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_cypher_batch = AsyncMock()
            service = KnowledgeGraphService(cypher_builder=NativeCypherBuilder())

            result = await service.persist_entities(
                characters=[_make_character("Alice"), _make_character("Bob")]
            )

            assert result is True
            statements = fake_neo4j.execute_cypher_batch.call_args[0][0]
            assert len(statements) == 2
            assert statements[0][1]["name"] == "Alice"
            assert statements[1][1]["name"] == "Bob"


@pytest.mark.asyncio
class TestPersistEntitiesWorldItems:

    async def test_single_world_item_upsert(self) -> None:
        with patch("core.knowledge_graph_service.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_cypher_batch = AsyncMock()
            service = KnowledgeGraphService(cypher_builder=NativeCypherBuilder())

            item = _make_world_item("Location", "Castle", "A big castle")
            result = await service.persist_entities(world_items=[item])

            assert result is True
            fake_neo4j.execute_cypher_batch.assert_called_once()
            statements = fake_neo4j.execute_cypher_batch.call_args[0][0]
            assert len(statements) == 1
            assert statements[0][1]["name"] == "Castle"
            assert statements[0][1]["description"] == "A big castle"

    async def test_multiple_world_items_upsert(self) -> None:
        with patch("core.knowledge_graph_service.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_cypher_batch = AsyncMock()
            service = KnowledgeGraphService(cypher_builder=NativeCypherBuilder())

            items = [
                _make_world_item("Location", "Castle", "A big castle"),
                _make_world_item("Location", "Forest", "A dark forest"),
            ]
            result = await service.persist_entities(world_items=items)

            assert result is True
            statements = fake_neo4j.execute_cypher_batch.call_args[0][0]
            assert len(statements) == 2
            assert statements[0][1]["name"] == "Castle"
            assert statements[1][1]["name"] == "Forest"


@pytest.mark.asyncio
class TestPersistEntitiesExtraStatements:

    async def test_extra_statements_included(self) -> None:
        with patch("core.knowledge_graph_service.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_cypher_batch = AsyncMock()
            service = KnowledgeGraphService(cypher_builder=NativeCypherBuilder())
            extra = [("MATCH (n) RETURN n LIMIT 1", {"key": "value"})]

            result = await service.persist_entities(extra_statements=extra)

            assert result is True
            fake_neo4j.execute_cypher_batch.assert_called_once()
            statements = fake_neo4j.execute_cypher_batch.call_args[0][0]
            assert len(statements) == 1
            assert statements[0] == ("MATCH (n) RETURN n LIMIT 1", {"key": "value"})


@pytest.mark.asyncio
class TestPersistEntitiesMixedInputs:

    async def test_characters_world_items_and_extra_statements(self) -> None:
        with patch("core.knowledge_graph_service.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_cypher_batch = AsyncMock()
            service = KnowledgeGraphService(cypher_builder=NativeCypherBuilder())

            characters = [_make_character("Alice")]
            world_items = [_make_world_item("Location", "Castle", "A big castle")]
            extra = [("RETURN 1", {})]

            result = await service.persist_entities(
                characters=characters,
                world_items=world_items,
                extra_statements=extra,
            )

            assert result is True
            statements = fake_neo4j.execute_cypher_batch.call_args[0][0]
            assert len(statements) == 3
            assert statements[0][1]["name"] == "Alice"
            assert statements[1][1]["name"] == "Castle"
            assert statements[2] == ("RETURN 1", {})


@pytest.mark.asyncio
class TestPersistEntitiesStrictFailure:

    async def test_raises_persistence_error_on_neo4j_failure(self) -> None:
        with patch("core.knowledge_graph_service.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_cypher_batch = AsyncMock(
                side_effect=RuntimeError("connection lost")
            )
            service = KnowledgeGraphService(cypher_builder=NativeCypherBuilder())

            with pytest.raises(KnowledgeGraphPersistenceError):
                await service.persist_entities(characters=[_make_character("Alice")])

    async def test_non_strict_returns_false_on_neo4j_failure(self) -> None:
        with patch("core.knowledge_graph_service.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_cypher_batch = AsyncMock(
                side_effect=RuntimeError("connection lost")
            )
            service = KnowledgeGraphService(cypher_builder=NativeCypherBuilder())

            result = await service.persist_entities(
                characters=[_make_character("Alice")], strict=False
            )

            assert result is False


@pytest.mark.asyncio
class TestPersistEntitiesChapterNumber:

    async def test_chapter_number_passed_to_character_builder(self) -> None:
        with patch("core.knowledge_graph_service.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_cypher_batch = AsyncMock()
            service = KnowledgeGraphService(cypher_builder=NativeCypherBuilder())

            await service.persist_entities(
                characters=[_make_character("Alice")], chapter_number=7
            )

            statements = fake_neo4j.execute_cypher_batch.call_args[0][0]
            assert statements[0][1]["chapter_number"] == 7

    async def test_chapter_number_passed_to_world_item_builder(self) -> None:
        with patch("core.knowledge_graph_service.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_cypher_batch = AsyncMock()
            service = KnowledgeGraphService(cypher_builder=NativeCypherBuilder())

            await service.persist_entities(
                world_items=[_make_world_item("Location", "Castle", "A big castle")],
                chapter_number=5,
            )

            statements = fake_neo4j.execute_cypher_batch.call_args[0][0]
            assert statements[0][1]["chapter_number"] == 5

    async def test_default_chapter_number_is_zero(self) -> None:
        with patch("core.knowledge_graph_service.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_cypher_batch = AsyncMock()
            service = KnowledgeGraphService(cypher_builder=NativeCypherBuilder())

            await service.persist_entities(characters=[_make_character("Alice")])

            statements = fake_neo4j.execute_cypher_batch.call_args[0][0]
            assert statements[0][1]["chapter_number"] == 0
