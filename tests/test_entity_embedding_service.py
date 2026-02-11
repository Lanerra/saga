import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from core.entity_embedding_service import (
    build_entity_embedding_update_statements,
    compute_entity_embedding_text,
    compute_entity_embedding_text_hash,
)
from models.kg_models import CharacterProfile, WorldItem


class TestComputeEntityEmbeddingText:
    def test_all_fields_populated(self) -> None:
        result = compute_entity_embedding_text(
            name="Aragorn",
            description="Ranger of the North",
            category="Character",
        )
        assert result == "Aragorn\nCharacter\nRanger of the North"

    def test_empty_name_excluded(self) -> None:
        result = compute_entity_embedding_text(
            name="",
            description="A powerful wizard",
            category="Character",
        )
        assert result == "Character\nA powerful wizard"

    def test_empty_category_excluded(self) -> None:
        result = compute_entity_embedding_text(
            name="Gandalf",
            description="A powerful wizard",
            category="",
        )
        assert result == "Gandalf\nA powerful wizard"

    def test_empty_description_excluded(self) -> None:
        result = compute_entity_embedding_text(
            name="Gandalf",
            description="",
            category="Character",
        )
        assert result == "Gandalf\nCharacter"

    def test_all_empty_returns_empty_string(self) -> None:
        result = compute_entity_embedding_text(
            name="",
            description="",
            category="",
        )
        assert result == ""

    def test_whitespace_only_fields_excluded(self) -> None:
        result = compute_entity_embedding_text(
            name="  ",
            description="  \t  ",
            category="  \n  ",
        )
        assert result == ""


class TestComputeEntityEmbeddingTextHash:
    def test_returns_sha1_hex_string(self) -> None:
        text = "Aragorn\nCharacter\nRanger of the North"
        result = compute_entity_embedding_text_hash(text)
        expected = hashlib.sha1(text.encode("utf-8")).hexdigest()
        assert result == expected

    def test_deterministic(self) -> None:
        text = "consistent input"
        first = compute_entity_embedding_text_hash(text)
        second = compute_entity_embedding_text_hash(text)
        assert first == second

    def test_different_text_produces_different_hash(self) -> None:
        hash_a = compute_entity_embedding_text_hash("alpha")
        hash_b = compute_entity_embedding_text_hash("beta")
        assert hash_a != hash_b

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="entity embedding text must be a non-empty string"):
            compute_entity_embedding_text_hash("")

    def test_non_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="entity embedding text must be a non-empty string"):
            compute_entity_embedding_text_hash(42)  # type: ignore[arg-type]


class TestBuildEntityEmbeddingUpdateStatements:
    @pytest.mark.asyncio
    async def test_returns_empty_when_persistence_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)

        fake_character = SimpleNamespace(name="Alice", description="A brave warrior")
        fake_world_item = WorldItem.from_dict("Location", "Castle", {"description": "A big castle"})

        result = await build_entity_embedding_update_statements(
            characters=[fake_character],
            world_items=[fake_world_item],
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_raises_when_config_properties_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", True)
        monkeypatch.setattr("config.ENTITY_EMBEDDING_VECTOR_PROPERTY", "")
        monkeypatch.setattr("config.ENTITY_EMBEDDING_TEXT_HASH_PROPERTY", "")
        monkeypatch.setattr("config.ENTITY_EMBEDDING_MODEL_PROPERTY", "")

        with pytest.raises(ValueError, match="entity embedding property configuration is missing"):
            await build_entity_embedding_update_statements(
                characters=[],
                world_items=[],
            )

    @pytest.mark.asyncio
    async def test_skips_entities_with_unchanged_hashes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", True)
        monkeypatch.setattr("config.ENTITY_EMBEDDING_VECTOR_PROPERTY", "entity_embedding_vector")
        monkeypatch.setattr("config.ENTITY_EMBEDDING_TEXT_HASH_PROPERTY", "entity_embedding_text_hash")
        monkeypatch.setattr("config.ENTITY_EMBEDDING_MODEL_PROPERTY", "entity_embedding_model")
        monkeypatch.setattr("config.EMBEDDING_MODEL", "fake-model")

        fake_character = SimpleNamespace(name="Alice", description="A brave warrior")
        character_embedding_text = compute_entity_embedding_text(
            name="Alice", category="", description="A brave warrior"
        )
        character_hash = compute_entity_embedding_text_hash(character_embedding_text)

        fake_world_item = WorldItem.from_dict("Location", "Castle", {"description": "A big castle"})
        world_embedding_text = compute_entity_embedding_text(
            name="Castle", category="Location", description="A big castle"
        )
        world_hash = compute_entity_embedding_text_hash(world_embedding_text)

        async def fake_execute_read_query(query: str, params: dict) -> list[dict]:
            if "Character" in query:
                return [{"key": "Alice", "existing_hash": character_hash}]
            return [{"key": fake_world_item.id, "existing_hash": world_hash}]

        with patch(
            "core.entity_embedding_service.neo4j_manager.execute_read_query",
            new=AsyncMock(side_effect=fake_execute_read_query),
        ):
            result = await build_entity_embedding_update_statements(
                characters=[fake_character],
                world_items=[fake_world_item],
            )

        assert result == []
