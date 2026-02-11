import pytest

from core.schema_validator import (
    SchemaValidationService,
    canonicalize_entity_type_for_persistence,
    validate_kg_object,
    validate_node_labels,
)
from models.kg_models import CharacterProfile, WorldItem


class TestValidateEntityType:
    def _make_service(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        enabled: bool = True,
        normalize: bool = True,
        log: bool = False,
    ) -> SchemaValidationService:
        monkeypatch.setattr("core.schema_validator.ENFORCE_SCHEMA_VALIDATION", enabled)
        monkeypatch.setattr("core.schema_validator.NORMALIZE_COMMON_VARIANTS", normalize)
        monkeypatch.setattr("core.schema_validator.LOG_SCHEMA_VIOLATIONS", log)
        return SchemaValidationService()

    def test_exact_canonical_label(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = self._make_service(monkeypatch)
        result = service.validate_entity_type("Character")
        assert result == (True, "Character", None)

    def test_case_insensitive_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = self._make_service(monkeypatch)
        result = service.validate_entity_type("character")
        assert result == (True, "Character", None)

    def test_normalization_map_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = self._make_service(monkeypatch)
        result = service.validate_entity_type("Person")
        assert result == (True, "Character", None)

    def test_unknown_label_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = self._make_service(monkeypatch)
        is_valid, name, error = service.validate_entity_type("Spaceship")
        assert is_valid is False
        assert name == "Spaceship"
        assert error is not None
        assert "Invalid entity type 'Spaceship'" in error

    def test_empty_string_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = self._make_service(monkeypatch)
        result = service.validate_entity_type("")
        assert result == (False, "", "Entity type cannot be empty")

    def test_disabled_returns_original(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = self._make_service(monkeypatch, enabled=False)
        result = service.validate_entity_type("Spaceship")
        assert result == (True, "Spaceship", None)


class TestValidateCategory:
    def _make_service(self, monkeypatch: pytest.MonkeyPatch) -> SchemaValidationService:
        monkeypatch.setattr("core.schema_validator.ENFORCE_SCHEMA_VALIDATION", True)
        monkeypatch.setattr("core.schema_validator.NORMALIZE_COMMON_VARIANTS", True)
        monkeypatch.setattr("core.schema_validator.LOG_SCHEMA_VIOLATIONS", False)
        return SchemaValidationService()

    def test_known_category(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = self._make_service(monkeypatch)
        result = service.validate_category("Character", "Protagonist")
        assert result == (True, None)

    def test_unknown_category(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = self._make_service(monkeypatch)
        is_known, suggestion = service.validate_category("Character", "Alien")
        assert is_known is False
        assert suggestion is not None
        assert "Alien" in suggestion
        assert "Suggested:" in suggestion

    def test_empty_category(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = self._make_service(monkeypatch)
        result = service.validate_category("Character", "")
        assert result == (True, None)

    def test_type_not_in_suggested_categories(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = self._make_service(monkeypatch)
        result = service.validate_category("Scene", "Flashback")
        assert result == (True, None)


class TestCanonicalizeEntityTypeForPersistence:
    def test_exact_canonical(self) -> None:
        assert canonicalize_entity_type_for_persistence("Character") == "Character"

    def test_case_insensitive(self) -> None:
        assert canonicalize_entity_type_for_persistence("location") == "Location"

    def test_alias_normalizes(self) -> None:
        assert canonicalize_entity_type_for_persistence("City") == "Location"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid entity type 'Spaceship'"):
            canonicalize_entity_type_for_persistence("Spaceship")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Entity type cannot be empty"):
            canonicalize_entity_type_for_persistence("")


class TestValidateKgObject:
    def test_valid_character_profile(self) -> None:
        profile = CharacterProfile(name="Aria")
        errors = validate_kg_object(profile)
        assert errors == []

    def test_empty_name_character_profile(self) -> None:
        profile = CharacterProfile(name="")
        errors = validate_kg_object(profile)
        assert errors == ["CharacterProfile name cannot be empty"]

    def test_valid_world_item(self) -> None:
        item = WorldItem.from_dict("Weapon", "Excalibur", {"description": "A legendary sword"})
        errors = validate_kg_object(item)
        assert errors == []

    def test_empty_name_world_item(self) -> None:
        item = WorldItem.from_dict("Weapon", "placeholder", {"description": "X"})
        item.name = ""
        errors = validate_kg_object(item)
        assert "WorldItem name cannot be empty" in errors

    def test_empty_category_world_item(self) -> None:
        item = WorldItem.from_dict("Weapon", "Excalibur", {"description": "X"})
        item.category = ""
        errors = validate_kg_object(item)
        assert "WorldItem category cannot be empty" in errors

    def test_unknown_object_type(self) -> None:
        errors = validate_kg_object(42)
        assert len(errors) == 1
        assert "Unknown object type for validation" in errors[0]


class TestValidateNodeLabels:
    def test_valid_labels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("core.schema_validator.ENFORCE_SCHEMA_VALIDATION", True)
        monkeypatch.setattr("core.schema_validator.NORMALIZE_COMMON_VARIANTS", True)
        monkeypatch.setattr("core.schema_validator.LOG_SCHEMA_VIOLATIONS", False)
        errors = validate_node_labels(["Character", "Location"])
        assert errors == []

    def test_invalid_label(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("core.schema_validator.ENFORCE_SCHEMA_VALIDATION", True)
        monkeypatch.setattr("core.schema_validator.NORMALIZE_COMMON_VARIANTS", True)
        monkeypatch.setattr("core.schema_validator.LOG_SCHEMA_VIOLATIONS", False)
        errors = validate_node_labels(["Spaceship"])
        assert len(errors) == 1
        assert "Invalid label 'Spaceship'" in errors[0]
