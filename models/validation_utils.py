# models/validation_utils.py
"""Validate bootstrap-generated content against runtime configuration.

This module compares values embedded in bootstrap artifacts (plot outline, character
profiles, world building) against runtime configuration constants.

Notes:
    Validation is reported as a list of [`ConfigurationValidationError`](models/validation_utils.py:21)
    instances. Callers can decide whether to fail fast, log, or auto-correct.
"""

from __future__ import annotations

from typing import Any

import structlog

import config
from models.kg_models import CharacterProfile, WorldItem

logger = structlog.get_logger(__name__)


class ConfigurationValidationError(Exception):
    """Represent a single configuration mismatch detected during validation."""

    def __init__(self, message: str, field: str, bootstrap_value: Any, runtime_value: Any):
        self.message = message
        self.field = field
        self.bootstrap_value = bootstrap_value
        self.runtime_value = runtime_value
        super().__init__(message)


class BootstrapContentValidator:
    """Validate consistency between bootstrap content and runtime configuration.

    The validator compares values from bootstrap-generated structures against runtime
    constants in the [`config`](config/__init__.py:1) module.

    Notes:
        Methods return a list of [`ConfigurationValidationError`](models/validation_utils.py:21)
        values rather than raising immediately.
    """

    def __init__(self) -> None:
        self.logger = structlog.get_logger(__name__)

    def validate_plot_outline_consistency(self, plot_outline: dict[str, Any], bootstrap_source: str = "bootstrap") -> list[ConfigurationValidationError]:
        """Validate that plot outline content matches runtime configuration.

        Args:
            plot_outline: Plot outline content to validate.
            bootstrap_source: Identifier for the bootstrap source.

        Returns:
            A list of detected mismatches.

        Notes:
            `bootstrap_source` is currently unused by the implementation, but is kept to
            support attribution/logging in call sites.
        """
        errors: list[ConfigurationValidationError] = []

        # Check genre consistency
        plot_genre = plot_outline.get("genre")
        if plot_genre and plot_genre != config.CONFIGURED_GENRE:
            errors.append(
                ConfigurationValidationError(
                    f"Plot outline genre '{plot_genre}' differs from runtime configuration '{config.CONFIGURED_GENRE}'",
                    "genre",
                    plot_genre,
                    config.CONFIGURED_GENRE,
                )
            )

        # Check theme consistency
        plot_theme = plot_outline.get("theme")
        if plot_theme and plot_theme != config.CONFIGURED_THEME:
            errors.append(
                ConfigurationValidationError(
                    f"Plot outline theme '{plot_theme}' differs from runtime configuration '{config.CONFIGURED_THEME}'",
                    "theme",
                    plot_theme,
                    config.CONFIGURED_THEME,
                )
            )

        # Check setting consistency
        plot_setting = plot_outline.get("setting")
        if plot_setting and plot_setting != config.CONFIGURED_SETTING_DESCRIPTION:
            errors.append(
                ConfigurationValidationError(
                    f"Plot outline setting '{plot_setting}' differs from runtime configuration '{config.CONFIGURED_SETTING_DESCRIPTION}'",
                    "setting",
                    plot_setting,
                    config.CONFIGURED_SETTING_DESCRIPTION,
                )
            )

        # Check protagonist name consistency
        plot_protagonist = plot_outline.get("protagonist_name")
        if plot_protagonist and plot_protagonist != config.DEFAULT_PROTAGONIST_NAME:
            errors.append(
                ConfigurationValidationError(
                    f"Plot outline protagonist '{plot_protagonist}' differs from runtime configuration '{config.DEFAULT_PROTAGONIST_NAME}'",
                    "protagonist_name",
                    plot_protagonist,
                    config.DEFAULT_PROTAGONIST_NAME,
                )
            )

        return errors

    def validate_character_profiles_consistency(
        self,
        character_profiles: dict[str, CharacterProfile],
        plot_outline: dict[str, Any],
    ) -> list[ConfigurationValidationError]:
        """Validate that character profiles match configuration-derived expectations.

        This primarily checks for a protagonist profile keyed by
        `config.DEFAULT_PROTAGONIST_NAME`.

        Args:
            character_profiles: Mapping of character name to character profile.
            plot_outline: Plot outline context (currently unused by the implementation).

        Returns:
            A list of detected mismatches.
        """
        errors: list[ConfigurationValidationError] = []

        # Check if protagonist exists and matches expected name
        expected_protagonist = config.DEFAULT_PROTAGONIST_NAME
        if expected_protagonist not in character_profiles:
            errors.append(
                ConfigurationValidationError(
                    f"Expected protagonist '{expected_protagonist}' not found in character profiles",
                    "protagonist_name",
                    None,
                    expected_protagonist,
                )
            )
        else:
            # Validate protagonist profile has expected name
            protagonist_profile = character_profiles[expected_protagonist]
            if protagonist_profile.name != expected_protagonist:
                errors.append(
                    ConfigurationValidationError(
                        f"Protagonist profile name '{protagonist_profile.name}' differs from expected '{expected_protagonist}'",
                        "protagonist_name",
                        protagonist_profile.name,
                        expected_protagonist,
                    )
                )

        return errors

    def validate_world_building_consistency(
        self,
        world_building: dict[str, dict[str, WorldItem]],
        plot_outline: dict[str, Any],
    ) -> list[ConfigurationValidationError]:
        """Validate that world building content matches runtime configuration.

        This currently validates only the setting overview description stored at
        `world_building["_overview_"]["_overview_"]`, when present.

        Args:
            world_building: Mapping of category to mapping of item name to world item.
            plot_outline: Plot outline context (currently unused by the implementation).

        Returns:
            A list of detected mismatches.
        """
        errors: list[ConfigurationValidationError] = []

        # Check if world overview exists and matches setting description
        if "_overview_" in world_building and "_overview_" in world_building["_overview_"]:
            overview_item = world_building["_overview_"]["_overview_"]
            if isinstance(overview_item, WorldItem):
                overview_desc = overview_item.description
                if overview_desc and overview_desc != config.CONFIGURED_SETTING_DESCRIPTION:
                    errors.append(
                        ConfigurationValidationError(
                            "World overview description differs from runtime configuration",
                            "setting_description",
                            overview_desc,
                            config.CONFIGURED_SETTING_DESCRIPTION,
                        )
                    )

        return errors

    def validate_all_components(
        self,
        plot_outline: dict[str, Any],
        character_profiles: dict[str, CharacterProfile],
        world_building: dict[str, dict[str, WorldItem]],
        bootstrap_source: str = "bootstrap",
    ) -> tuple[bool, list[ConfigurationValidationError]]:
        """Validate all components for consistency.

        Args:
            plot_outline: Plot outline content to validate.
            character_profiles: Mapping of character name to character profile.
            world_building: Mapping of category to mapping of item name to world item.
            bootstrap_source: Identifier for the bootstrap source.

        Returns:
            A tuple of `(is_valid, errors)` where `errors` contains all detected mismatches.

        Notes:
            This method logs a summary via structlog when validation passes or fails.
        """
        all_errors: list[ConfigurationValidationError] = []

        # Validate plot outline
        plot_errors = self.validate_plot_outline_consistency(plot_outline, bootstrap_source)
        all_errors.extend(plot_errors)

        # Validate character profiles
        char_errors = self.validate_character_profiles_consistency(character_profiles, plot_outline)
        all_errors.extend(char_errors)

        # Validate world building
        world_errors = self.validate_world_building_consistency(world_building, plot_outline)
        all_errors.extend(world_errors)

        is_valid = len(all_errors) == 0

        if not is_valid:
            self.logger.warning(
                "Configuration validation failed",
                error_count=len(all_errors),
                errors=[str(error) for error in all_errors],
            )
        else:
            self.logger.info("Configuration validation passed")

        return is_valid, all_errors

    def suggest_corrections(self, errors: list[ConfigurationValidationError]) -> dict[str, Any]:
        """Suggest corrections for a set of validation errors.

        Args:
            errors: Validation errors returned by a `validate_*` method.

        Returns:
            A dictionary with suggested values grouped by:
            - `plot_outline`
            - `character_profiles`
            - `world_building`
            - `config_updates`

        Notes:
            Suggestions are derived from runtime configuration constants and do not
            mutate inputs.
        """
        corrections: dict[str, Any] = {
            "plot_outline": {},
            "character_profiles": {},
            "world_building": {},
            "config_updates": {},
        }

        for error in errors:
            if error.field == "genre":
                corrections["plot_outline"]["genre"] = config.CONFIGURED_GENRE
                corrections["config_updates"]["CONFIGURED_GENRE"] = config.CONFIGURED_GENRE
            elif error.field == "theme":
                corrections["plot_outline"]["theme"] = config.CONFIGURED_THEME
                corrections["config_updates"]["CONFIGURED_THEME"] = config.CONFIGURED_THEME
            elif error.field == "setting":
                corrections["plot_outline"]["setting"] = config.CONFIGURED_SETTING_DESCRIPTION
                corrections["config_updates"]["CONFIGURED_SETTING_DESCRIPTION"] = config.CONFIGURED_SETTING_DESCRIPTION
            elif error.field == "protagonist_name":
                corrections["plot_outline"]["protagonist_name"] = config.DEFAULT_PROTAGONIST_NAME
                corrections["config_updates"]["DEFAULT_PROTAGONIST_NAME"] = config.DEFAULT_PROTAGONIST_NAME
            elif error.field == "setting_description":
                corrections["world_building"]["_overview_"] = config.CONFIGURED_SETTING_DESCRIPTION
                corrections["config_updates"]["CONFIGURED_SETTING_DESCRIPTION"] = config.CONFIGURED_SETTING_DESCRIPTION

        return corrections


# Global validator instance
bootstrap_validator = BootstrapContentValidator()
