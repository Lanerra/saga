# tests/test_validation_utils.py
"""Validate bootstrap content consistency checking against runtime config."""

from typing import Any

import pytest

import config
from models.kg_models import CharacterProfile, WorldItem
from models.validation_utils import (
    BootstrapContentValidator,
    ConfigurationValidationError,
)

GENRE = "cosmic horror"
THEME = "isolation in deep space"
SETTING = "an abandoned station orbiting Neptune"
PROTAGONIST = "Vera Okonkwo"


@pytest.fixture()
def patched_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "CONFIGURED_GENRE", GENRE)
    monkeypatch.setattr(config, "CONFIGURED_THEME", THEME)
    monkeypatch.setattr(config, "CONFIGURED_SETTING_DESCRIPTION", SETTING)
    monkeypatch.setattr(config, "DEFAULT_PROTAGONIST_NAME", PROTAGONIST)


@pytest.fixture()
def validator() -> BootstrapContentValidator:
    return BootstrapContentValidator()


@pytest.fixture()
def matching_plot_outline() -> dict[str, Any]:
    return {
        "genre": GENRE,
        "theme": THEME,
        "setting": SETTING,
        "protagonist_name": PROTAGONIST,
    }


@pytest.fixture()
def matching_character_profiles() -> dict[str, CharacterProfile]:
    return {
        PROTAGONIST: CharacterProfile(name=PROTAGONIST),
        "Side Character": CharacterProfile(name="Side Character"),
    }


@pytest.fixture()
def matching_world_building() -> dict[str, dict[str, WorldItem]]:
    overview = WorldItem.from_dict("_overview_", "_overview_", {"description": SETTING})
    return {"_overview_": {"_overview_": overview}}


class TestConfigurationValidationError:
    """ConfigurationValidationError attribute storage."""

    def test_stores_all_attributes(self) -> None:
        error = ConfigurationValidationError(
            message="genre mismatch",
            field="genre",
            bootstrap_value="fantasy",
            runtime_value="horror",
        )

        assert error.message == "genre mismatch"
        assert error.field == "genre"
        assert error.bootstrap_value == "fantasy"
        assert error.runtime_value == "horror"
        assert str(error) == "genre mismatch"
        assert isinstance(error, Exception)


class TestPlotOutlineConsistency:
    """Plot outline validation against runtime config."""

    def test_all_fields_match(
        self,
        patched_config: None,
        validator: BootstrapContentValidator,
        matching_plot_outline: dict[str, Any],
    ) -> None:
        errors = validator.validate_plot_outline_consistency(matching_plot_outline)
        assert errors == []

    def test_genre_mismatch(
        self,
        patched_config: None,
        validator: BootstrapContentValidator,
        matching_plot_outline: dict[str, Any],
    ) -> None:
        matching_plot_outline["genre"] = "romance"
        errors = validator.validate_plot_outline_consistency(matching_plot_outline)

        assert len(errors) == 1
        assert errors[0].field == "genre"
        assert errors[0].bootstrap_value == "romance"
        assert errors[0].runtime_value == GENRE

    def test_theme_mismatch(
        self,
        patched_config: None,
        validator: BootstrapContentValidator,
        matching_plot_outline: dict[str, Any],
    ) -> None:
        matching_plot_outline["theme"] = "love conquers all"
        errors = validator.validate_plot_outline_consistency(matching_plot_outline)

        assert len(errors) == 1
        assert errors[0].field == "theme"
        assert errors[0].bootstrap_value == "love conquers all"
        assert errors[0].runtime_value == THEME

    def test_setting_mismatch(
        self,
        patched_config: None,
        validator: BootstrapContentValidator,
        matching_plot_outline: dict[str, Any],
    ) -> None:
        matching_plot_outline["setting"] = "a sunny meadow"
        errors = validator.validate_plot_outline_consistency(matching_plot_outline)

        assert len(errors) == 1
        assert errors[0].field == "setting"
        assert errors[0].bootstrap_value == "a sunny meadow"
        assert errors[0].runtime_value == SETTING

    def test_protagonist_name_mismatch(
        self,
        patched_config: None,
        validator: BootstrapContentValidator,
        matching_plot_outline: dict[str, Any],
    ) -> None:
        matching_plot_outline["protagonist_name"] = "John Doe"
        errors = validator.validate_plot_outline_consistency(matching_plot_outline)

        assert len(errors) == 1
        assert errors[0].field == "protagonist_name"
        assert errors[0].bootstrap_value == "John Doe"
        assert errors[0].runtime_value == PROTAGONIST

    def test_missing_fields_do_not_trigger_errors(
        self,
        patched_config: None,
        validator: BootstrapContentValidator,
    ) -> None:
        """None-valued or absent fields are skipped by the validator."""
        sparse_outline: dict[str, Any] = {"title": "My Story"}
        errors = validator.validate_plot_outline_consistency(sparse_outline)
        assert errors == []


class TestCharacterProfilesConsistency:
    """Character profile validation against config protagonist."""

    def test_protagonist_exists_and_matches(
        self,
        patched_config: None,
        validator: BootstrapContentValidator,
        matching_character_profiles: dict[str, CharacterProfile],
        matching_plot_outline: dict[str, Any],
    ) -> None:
        errors = validator.validate_character_profiles_consistency(matching_character_profiles, matching_plot_outline)
        assert errors == []

    def test_protagonist_missing(
        self,
        patched_config: None,
        validator: BootstrapContentValidator,
        matching_plot_outline: dict[str, Any],
    ) -> None:
        profiles: dict[str, CharacterProfile] = {
            "Someone Else": CharacterProfile(name="Someone Else"),
        }
        errors = validator.validate_character_profiles_consistency(profiles, matching_plot_outline)

        assert len(errors) == 1
        assert errors[0].field == "protagonist_name"
        assert errors[0].bootstrap_value is None
        assert errors[0].runtime_value == PROTAGONIST

    def test_protagonist_name_mismatch_in_profile(
        self,
        patched_config: None,
        validator: BootstrapContentValidator,
        matching_plot_outline: dict[str, Any],
    ) -> None:
        """Profile keyed correctly but internal .name differs."""
        wrong_name_profile = CharacterProfile(name="Wrong Name")
        profiles: dict[str, CharacterProfile] = {
            PROTAGONIST: wrong_name_profile,
        }
        errors = validator.validate_character_profiles_consistency(profiles, matching_plot_outline)

        assert len(errors) == 1
        assert errors[0].field == "protagonist_name"
        assert errors[0].bootstrap_value == "Wrong Name"
        assert errors[0].runtime_value == PROTAGONIST


class TestWorldBuildingConsistency:
    """World building overview validation against config setting."""

    def test_overview_matches(
        self,
        patched_config: None,
        validator: BootstrapContentValidator,
        matching_world_building: dict[str, dict[str, WorldItem]],
        matching_plot_outline: dict[str, Any],
    ) -> None:
        errors = validator.validate_world_building_consistency(matching_world_building, matching_plot_outline)
        assert errors == []

    def test_overview_mismatch(
        self,
        patched_config: None,
        validator: BootstrapContentValidator,
        matching_plot_outline: dict[str, Any],
    ) -> None:
        mismatched_overview = WorldItem.from_dict("_overview_", "_overview_", {"description": "a tropical island"})
        world_building: dict[str, dict[str, WorldItem]] = {"_overview_": {"_overview_": mismatched_overview}}
        errors = validator.validate_world_building_consistency(world_building, matching_plot_outline)

        assert len(errors) == 1
        assert errors[0].field == "setting_description"
        assert errors[0].bootstrap_value == "a tropical island"
        assert errors[0].runtime_value == SETTING

    def test_no_overview_section(
        self,
        patched_config: None,
        validator: BootstrapContentValidator,
        matching_plot_outline: dict[str, Any],
    ) -> None:
        world_building: dict[str, dict[str, WorldItem]] = {"locations": {"city": WorldItem.from_dict("locations", "city", {"description": "a big city"})}}
        errors = validator.validate_world_building_consistency(world_building, matching_plot_outline)
        assert errors == []


class TestValidateAllComponents:
    """Aggregate validation across all component types."""

    def test_returns_true_when_all_valid(
        self,
        patched_config: None,
        validator: BootstrapContentValidator,
        matching_plot_outline: dict[str, Any],
        matching_character_profiles: dict[str, CharacterProfile],
        matching_world_building: dict[str, dict[str, WorldItem]],
    ) -> None:
        is_valid, errors = validator.validate_all_components(
            matching_plot_outline,
            matching_character_profiles,
            matching_world_building,
        )

        assert is_valid is True
        assert errors == []

    def test_aggregates_errors_from_all_validators(
        self,
        patched_config: None,
        validator: BootstrapContentValidator,
    ) -> None:
        bad_plot: dict[str, Any] = {
            "genre": "wrong genre",
            "theme": "wrong theme",
        }
        empty_profiles: dict[str, CharacterProfile] = {}
        mismatched_overview = WorldItem.from_dict("_overview_", "_overview_", {"description": "wrong setting"})
        bad_world: dict[str, dict[str, WorldItem]] = {"_overview_": {"_overview_": mismatched_overview}}

        is_valid, errors = validator.validate_all_components(bad_plot, empty_profiles, bad_world)

        assert is_valid is False
        fields = [e.field for e in errors]
        assert "genre" in fields
        assert "theme" in fields
        assert "protagonist_name" in fields
        assert "setting_description" in fields
        assert len(errors) == 4


class TestSuggestCorrections:
    """Correction suggestions derived from runtime config values."""

    def test_maps_field_errors_to_config_values(
        self,
        patched_config: None,
        validator: BootstrapContentValidator,
    ) -> None:
        errors = [
            ConfigurationValidationError("genre off", "genre", "wrong", GENRE),
            ConfigurationValidationError("theme off", "theme", "wrong", THEME),
            ConfigurationValidationError("setting off", "setting", "wrong", SETTING),
            ConfigurationValidationError("protag off", "protagonist_name", "wrong", PROTAGONIST),
            ConfigurationValidationError("world off", "setting_description", "wrong", SETTING),
        ]

        corrections = validator.suggest_corrections(errors)

        assert corrections["plot_outline"]["genre"] == GENRE
        assert corrections["plot_outline"]["theme"] == THEME
        assert corrections["plot_outline"]["setting"] == SETTING
        assert corrections["plot_outline"]["protagonist_name"] == PROTAGONIST
        assert corrections["world_building"]["_overview_"] == SETTING
        assert corrections["config_updates"]["CONFIGURED_GENRE"] == GENRE
        assert corrections["config_updates"]["CONFIGURED_THEME"] == THEME
        assert corrections["config_updates"]["CONFIGURED_SETTING_DESCRIPTION"] == SETTING
        assert corrections["config_updates"]["DEFAULT_PROTAGONIST_NAME"] == PROTAGONIST
        assert corrections["character_profiles"] == {}
