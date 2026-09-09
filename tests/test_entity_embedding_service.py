import hashlib
from unittest.mock import AsyncMock

import pytest

import config
from core.embedding_contract import embedding_identity
from core.entity_embedding_service import (
    build_entity_embedding_update_statements,
    compute_entity_embedding_text,
    compute_entity_embedding_text_hash,
)
from core.service_context import get_services
from models.kg_models import CharacterProfile, WorldItem
from tests.fakes.service_context import patch_service


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
    async def test_returns_empty_when_persistence_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)

        fake_character = CharacterProfile.from_dict("Alice", {"description": "A brave warrior"})
        fake_world_item = WorldItem.from_dict("Location", "Castle", {"description": "A big castle"})

        result = await build_entity_embedding_update_statements(
            characters=[fake_character],
            world_items=[fake_world_item],
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_raises_when_config_properties_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
    async def test_skips_entities_with_unchanged_hashes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", True)
        monkeypatch.setattr("config.ENTITY_EMBEDDING_VECTOR_PROPERTY", "entity_embedding_vector")
        monkeypatch.setattr("config.ENTITY_EMBEDDING_TEXT_HASH_PROPERTY", "entity_embedding_text_hash")
        monkeypatch.setattr("config.ENTITY_EMBEDDING_MODEL_PROPERTY", "entity_embedding_model")
        monkeypatch.setattr("config.EMBEDDING_MODEL", "fake-model")
        monkeypatch.setattr(config, "EXPECTED_EMBEDDING_DIM", 2)

        fake_character = CharacterProfile(name="Alice", id="", personality_description="A brave warrior")
        character_embedding_text = compute_entity_embedding_text(name="Alice", category="", description="A brave warrior")
        character_hash = compute_entity_embedding_text_hash(character_embedding_text)

        fake_world_item = WorldItem.from_dict("Location", "Castle", {"description": "A big castle"})
        world_embedding_text = compute_entity_embedding_text(name="Castle", category="Location", description="A big castle")
        world_hash = compute_entity_embedding_text_hash(world_embedding_text)

        async def fake_execute_read_query(query: str, params: dict) -> list[dict]:
            metadata = {"existing_model": config.EMBEDDING_MODEL, "existing_identity": embedding_identity(), "existing_vector": [0.25, 0.75]}
            return [{"key": 0, "id": "alice-id", "existing_hash": character_hash, **metadata},
                    {"key": 1, "id": fake_world_item.id, "existing_hash": world_hash, **metadata}]

        with patch_service(
            'database.execute_read_query',
            new=AsyncMock(side_effect=fake_execute_read_query),
        ):
            result = await build_entity_embedding_update_statements(
                characters=[fake_character],
                world_items=[fake_world_item],
            )

        assert result == []


@pytest.mark.parametrize("label", ["Character", "Location", "Item", "Event"])
@pytest.mark.asyncio
async def test_embedding_updates_carry_label_and_canonical_id(label: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "EXPECTED_EMBEDDING_DIM", 2)
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", True)
    character = CharacterProfile(name="Alias", id="stable", personality_description="Synthetic")
    item = WorldItem(name="Alias", id="stable", category=label, description="Synthetic")
    reads = AsyncMock(return_value=[])
    provider = AsyncMock(return_value=[[0.25, 0.75]])
    monkeypatch.setattr(get_services().database, 'execute_read_query', reads)
    monkeypatch.setattr(get_services().language_model, 'async_get_embeddings_batch', provider)
    statements = await build_entity_embedding_update_statements(
        characters=[character] if label == "Character" else [],
        world_items=[] if label == "Character" else [item],
    )
    assert len(statements) == 1
    query, parameters = statements[0]
    assert parameters["identity"] == {"label": label, "id": "stable", "name": "Alias"}
    assert parameters["vector"] == [0.25, 0.75]
    assert "size(candidates) <> 1" in query
    assert reads.call_args.args[1]["entities"][0]["label"] == label
    assert reads.call_args.args[1]["entities"][0]["id"] == "stable"


@pytest.mark.parametrize("label", ["Character", "Location", "Item", "Event"])
@pytest.mark.parametrize("identifier", [None, " ", 7, []])
@pytest.mark.asyncio
async def test_embedding_malformed_identity_rejected_before_io(label: str, identifier: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", True)
    character = CharacterProfile(name="Synthetic").model_copy(update={"id": identifier})
    item = WorldItem(name="Synthetic", id="", category=label).model_copy(update={"id": identifier})
    reads = AsyncMock()
    provider = AsyncMock()
    monkeypatch.setattr(get_services().database, 'execute_read_query', reads)
    monkeypatch.setattr(get_services().language_model, 'async_get_embeddings_batch', provider)
    with pytest.raises(ValueError, match="^Invalid canonical entity ID$"):
        await build_entity_embedding_update_statements(
            characters=[character] if label == "Character" else [], world_items=[] if label == "Character" else [item],
        )
    reads.assert_not_called()
    provider.assert_not_called()


@pytest.mark.parametrize("label", ["Character", "Location", "Item", "Event"])
@pytest.mark.asyncio
async def test_embedding_name_lookup_pins_resolved_id(label: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "EXPECTED_EMBEDDING_DIM", 2)
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", True)
    reads = AsyncMock(return_value=[{"key": 0, "id": "resolved-id", "existing_hash": None}])
    monkeypatch.setattr(get_services().database, 'execute_read_query', reads)
    monkeypatch.setattr(get_services().language_model, 'async_get_embeddings_batch', AsyncMock(return_value=[[0.25, 0.75]]))
    statements = await build_entity_embedding_update_statements(
        characters=[CharacterProfile(name="Synthetic")] if label == "Character" else [],
        world_items=[] if label == "Character" else [WorldItem(name="Synthetic", id="", category=label)],
    )
    assert reads.call_args.args[1]["entities"][0]["id"] is None
    assert statements[0][1]["identity"] == {"label": label, "id": "resolved-id", "name": "Synthetic"}
