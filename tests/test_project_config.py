# tests/test_project_config.py
"""Tests for core/project_config.py - NarrativeProjectConfig Pydantic model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.project_config import NarrativeProjectConfig

VALID_CONFIG_DATA = {
    "title": "The Great Adventure",
    "genre": "Fantasy",
    "theme": "Redemption",
    "setting": "Medieval Kingdom",
    "protagonist_name": "Aria",
    "narrative_style": "Third person limited",
    "total_chapters": 12,
}


@pytest.fixture()
def valid_config_data() -> dict[str, str | int]:
    return dict(VALID_CONFIG_DATA)


class TestValidCreation:
    """Verify that valid inputs produce correct NarrativeProjectConfig instances."""

    def test_all_required_fields(self, valid_config_data: dict[str, str | int]) -> None:
        """A config with all required fields is created with matching values."""
        config = NarrativeProjectConfig(**valid_config_data)

        assert config.title == "The Great Adventure"
        assert config.genre == "Fantasy"
        assert config.theme == "Redemption"
        assert config.setting == "Medieval Kingdom"
        assert config.protagonist_name == "Aria"
        assert config.narrative_style == "Third person limited"
        assert config.total_chapters == 12

    def test_optional_fields_have_correct_defaults(self, valid_config_data: dict[str, str | int]) -> None:
        """Optional fields default to empty strings when omitted."""
        config = NarrativeProjectConfig(**valid_config_data)

        assert config.created_from == ""
        assert config.original_prompt == ""

    def test_optional_fields_accept_values(self, valid_config_data: dict[str, str | int]) -> None:
        """Optional fields accept explicit string values."""
        valid_config_data["created_from"] = "bootstrap"
        valid_config_data["original_prompt"] = "Write me a fantasy novel"

        config = NarrativeProjectConfig(**valid_config_data)

        assert config.created_from == "bootstrap"
        assert config.original_prompt == "Write me a fantasy novel"


class TestMissingRequiredFields:
    """Verify that omitting required fields raises ValidationError."""

    REQUIRED_FIELDS = [
        "title",
        "genre",
        "theme",
        "setting",
        "protagonist_name",
        "narrative_style",
        "total_chapters",
    ]

    @pytest.mark.parametrize("field", REQUIRED_FIELDS)
    def test_missing_required_field(self, valid_config_data: dict[str, str | int], field: str) -> None:
        """Each required field must be present."""
        del valid_config_data[field]

        with pytest.raises(ValidationError) as exception_info:
            NarrativeProjectConfig(**valid_config_data)

        errors = exception_info.value.errors()
        field_errors = [error for error in errors if field in error["loc"]]
        assert len(field_errors) >= 1


class TestExtraFieldForbidden:
    """Verify that extra fields are rejected (extra='forbid')."""

    def test_extra_field_raises_validation_error(self, valid_config_data: dict[str, str | int]) -> None:
        """An unrecognized field causes a ValidationError."""
        valid_config_data["unexpected_field"] = "surprise"

        with pytest.raises(ValidationError) as exception_info:
            NarrativeProjectConfig(**valid_config_data)

        errors = exception_info.value.errors()
        extra_errors = [error for error in errors if error["type"] == "extra_forbidden"]
        assert len(extra_errors) == 1


class TestEmptyStringRejected:
    """Verify that empty strings are rejected for fields with min_length=1."""

    STRING_FIELDS_WITH_MIN_LENGTH = [
        "title",
        "genre",
        "theme",
        "setting",
        "protagonist_name",
        "narrative_style",
    ]

    @pytest.mark.parametrize("field", STRING_FIELDS_WITH_MIN_LENGTH)
    def test_empty_string_raises_validation_error(self, valid_config_data: dict[str, str | int], field: str) -> None:
        """An empty string violates the min_length=1 constraint."""
        valid_config_data[field] = ""

        with pytest.raises(ValidationError) as exception_info:
            NarrativeProjectConfig(**valid_config_data)

        errors = exception_info.value.errors()
        field_errors = [error for error in errors if field in error["loc"]]
        assert len(field_errors) >= 1


class TestTotalChaptersConstraint:
    """Verify the ge=1 constraint on total_chapters."""

    def test_zero_chapters_raises_validation_error(self, valid_config_data: dict[str, str | int]) -> None:
        """total_chapters=0 violates the ge=1 constraint."""
        valid_config_data["total_chapters"] = 0

        with pytest.raises(ValidationError) as exception_info:
            NarrativeProjectConfig(**valid_config_data)

        errors = exception_info.value.errors()
        field_errors = [error for error in errors if "total_chapters" in error["loc"]]
        assert len(field_errors) == 1

    def test_negative_chapters_raises_validation_error(self, valid_config_data: dict[str, str | int]) -> None:
        """total_chapters=-1 violates the ge=1 constraint."""
        valid_config_data["total_chapters"] = -1

        with pytest.raises(ValidationError) as exception_info:
            NarrativeProjectConfig(**valid_config_data)

        errors = exception_info.value.errors()
        field_errors = [error for error in errors if "total_chapters" in error["loc"]]
        assert len(field_errors) == 1

    def test_one_chapter_is_valid(self, valid_config_data: dict[str, str | int]) -> None:
        """total_chapters=1 is the minimum valid value."""
        valid_config_data["total_chapters"] = 1

        config = NarrativeProjectConfig(**valid_config_data)

        assert config.total_chapters == 1


class TestStrictTypeCoercion:
    """Verify that strict=True rejects implicit type coercion."""

    def test_integer_for_string_field_raises_validation_error(self, valid_config_data: dict[str, str | int]) -> None:
        """Passing an int where a StrictStr is expected is rejected."""
        valid_config_data["title"] = 42

        with pytest.raises(ValidationError) as exception_info:
            NarrativeProjectConfig(**valid_config_data)

        errors = exception_info.value.errors()
        field_errors = [error for error in errors if "title" in error["loc"]]
        assert len(field_errors) >= 1

    def test_float_for_integer_field_raises_validation_error(self, valid_config_data: dict[str, str | int]) -> None:
        """Passing a float where a StrictInt is expected is rejected."""
        valid_config_data["total_chapters"] = 5.0

        with pytest.raises(ValidationError) as exception_info:
            NarrativeProjectConfig(**valid_config_data)

        errors = exception_info.value.errors()
        field_errors = [error for error in errors if "total_chapters" in error["loc"]]
        assert len(field_errors) >= 1

    def test_string_for_integer_field_raises_validation_error(self, valid_config_data: dict[str, str | int]) -> None:
        """Passing a string where a StrictInt is expected is rejected."""
        valid_config_data["total_chapters"] = "10"

        with pytest.raises(ValidationError) as exception_info:
            NarrativeProjectConfig(**valid_config_data)

        errors = exception_info.value.errors()
        field_errors = [error for error in errors if "total_chapters" in error["loc"]]
        assert len(field_errors) >= 1

    def test_boolean_for_string_field_raises_validation_error(self, valid_config_data: dict[str, str | int]) -> None:
        """Passing a bool where a StrictStr is expected is rejected."""
        valid_config_data["genre"] = True

        with pytest.raises(ValidationError) as exception_info:
            NarrativeProjectConfig(**valid_config_data)

        errors = exception_info.value.errors()
        field_errors = [error for error in errors if "genre" in error["loc"]]
        assert len(field_errors) >= 1


class TestModelRoundtrip:
    """Verify model_dump / model_validate produce equivalent instances."""

    def test_dump_and_validate_roundtrip(self, valid_config_data: dict[str, str | int]) -> None:
        """Serializing then deserializing produces an equal model instance."""
        original = NarrativeProjectConfig(**valid_config_data)

        dumped = original.model_dump()
        restored = NarrativeProjectConfig.model_validate(dumped)

        assert restored == original

    def test_dump_contains_all_fields(self, valid_config_data: dict[str, str | int]) -> None:
        """model_dump includes both required and optional fields."""
        config = NarrativeProjectConfig(**valid_config_data)

        dumped = config.model_dump()

        expected_keys = {
            "title",
            "genre",
            "theme",
            "setting",
            "protagonist_name",
            "narrative_style",
            "total_chapters",
            "created_from",
            "original_prompt",
        }
        assert set(dumped.keys()) == expected_keys

    def test_roundtrip_with_optional_fields_populated(self) -> None:
        """Roundtrip preserves explicitly set optional fields."""
        data = {
            **VALID_CONFIG_DATA,
            "created_from": "bootstrap",
            "original_prompt": "Write an epic saga",
        }
        original = NarrativeProjectConfig(**data)

        dumped = original.model_dump()
        restored = NarrativeProjectConfig.model_validate(dumped)

        assert restored == original
        assert restored.created_from == "bootstrap"
        assert restored.original_prompt == "Write an epic saga"
