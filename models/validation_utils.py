"""
Validation utilities for ensuring consistency between bootstrapped content and runtime configuration.

This module provides validation functions to ensure that the novel generation pipeline
maintains consistency between bootstrap-generated content and runtime configuration values.
"""

from __future__ import annotations

import structlog
from typing import Any, Dict, List, Optional, Tuple

import config
from models.kg_models import CharacterProfile, WorldItem

logger = structlog.get_logger(__name__)


class ConfigurationValidationError(Exception):
    """Exception raised when configuration validation fails."""

    def __init__(self, message: str, field: str, bootstrap_value: Any, runtime_value: Any):
        self.message = message
        self.field = field
        self.bootstrap_value = bootstrap_value
        self.runtime_value = runtime_value
        super().__init__(message)


class BootstrapContentValidator:
    """
    Validates consistency between bootstrap-generated content and runtime configuration values.

    This class ensures that the novel generation pipeline maintains consistency between
    the content generated during bootstrap and the configuration values used at runtime.
    """

    def __init__(self):
        self.logger = structlog.get_logger(__name__)

    def validate_plot_outline_consistency(
        self,
        plot_outline: Dict[str, Any],
        bootstrap_source: str = "bootstrap"
    ) -> List[ConfigurationValidationError]:
        """
        Validate that plot outline content is consistent with runtime configuration.

        Args:
            plot_outline: The plot outline dictionary to validate
            bootstrap_source: Source identifier for bootstrap content

        Returns:
            List of validation errors found
        """
        errors: List[ConfigurationValidationError] = []

        # Check genre consistency
        plot_genre = plot_outline.get("genre")
        if plot_genre and plot_genre != config.CONFIGURED_GENRE:
            errors.append(
                ConfigurationValidationError(
                    f"Plot outline genre '{plot_genre}' differs from runtime configuration '{config.CONFIGURED_GENRE}'",
                    "genre",
                    plot_genre,
                    config.CONFIGURED_GENRE
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
                    config.CONFIGURED_THEME
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
                    config.CONFIGURED_SETTING_DESCRIPTION
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
                    config.DEFAULT_PROTAGONIST_NAME
                )
            )

        return errors

    def validate_character_profiles_consistency(
        self,
        character_profiles: Dict[str, CharacterProfile],
        plot_outline: Dict[str, Any]
    ) -> List[ConfigurationValidationError]:
        """
        Validate that character profiles are consistent with plot outline and configuration.

        Args:
            character_profiles: Dictionary of character profiles
            plot_outline: The plot outline for context

        Returns:
            List of validation errors found
        """
        errors: List[ConfigurationValidationError] = []

        # Check if protagonist exists and matches expected name
        expected_protagonist = config.DEFAULT_PROTAGONIST_NAME
        if expected_protagonist not in character_profiles:
            errors.append(
                ConfigurationValidationError(
                    f"Expected protagonist '{expected_protagonist}' not found in character profiles",
                    "protagonist_name",
                    None,
                    expected_protagonist
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
                        expected_protagonist
                    )
                )

        return errors

    def validate_world_building_consistency(
        self,
        world_building: Dict[str, Dict[str, WorldItem]],
        plot_outline: Dict[str, Any]
    ) -> List[ConfigurationValidationError]:
        """
        Validate that world building content is consistent with configuration.

        Args:
            world_building: Dictionary of world building elements
            plot_outline: The plot outline for context

        Returns:
            List of validation errors found
        """
        errors: List[ConfigurationValidationError] = []

        # Check if world overview exists and matches setting description
        if "_overview_" in world_building and "_overview_" in world_building["_overview_"]:
            overview_item = world_building["_overview_"]["_overview_"]
            if isinstance(overview_item, WorldItem):
                overview_desc = overview_item.description
                if overview_desc and overview_desc != config.CONFIGURED_SETTING_DESCRIPTION:
                    errors.append(
                        ConfigurationValidationError(
                            f"World overview description differs from runtime configuration",
                            "setting_description",
                            overview_desc,
                            config.CONFIGURED_SETTING_DESCRIPTION
                        )
                    )

        return errors

    def validate_all_components(
        self,
        plot_outline: Dict[str, Any],
        character_profiles: Dict[str, CharacterProfile],
        world_building: Dict[str, Dict[str, WorldItem]],
        bootstrap_source: str = "bootstrap"
    ) -> Tuple[bool, List[ConfigurationValidationError]]:
        """
        Validate all components for consistency.

        Args:
            plot_outline: The plot outline dictionary
            character_profiles: Dictionary of character profiles
            world_building: Dictionary of world building elements
            bootstrap_source: Source identifier for bootstrap content

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        all_errors: List[ConfigurationValidationError] = []

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
                errors=[str(error) for error in all_errors]
            )
        else:
            self.logger.info("Configuration validation passed")

        return is_valid, all_errors

    def suggest_corrections(
        self,
        errors: List[ConfigurationValidationError]
    ) -> Dict[str, Any]:
        """
        Suggest corrections for validation errors.

        Args:
            errors: List of validation errors

        Returns:
            Dictionary of suggested corrections
        """
        corrections: Dict[str, Any] = {
            "plot_outline": {},
            "character_profiles": {},
            "world_building": {},
            "config_updates": {}
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


def validate_bootstrap_consistency(
    plot_outline: Dict[str, Any],
    character_profiles: Dict[str, CharacterProfile],
    world_building: Dict[str, Dict[str, WorldItem]],
    bootstrap_source: str = "bootstrap"
) -> Tuple[bool, List[ConfigurationValidationError]]:
    """
    Convenience function to validate bootstrap consistency.

    Args:
        plot_outline: The plot outline dictionary
        character_profiles: Dictionary of character profiles
        world_building: Dictionary of world building elements
        bootstrap_source: Source identifier for bootstrap content

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    return bootstrap_validator.validate_all_components(
        plot_outline, character_profiles, world_building, bootstrap_source
    )


def get_validation_corrections(
    errors: List[ConfigurationValidationError]
) -> Dict[str, Any]:
    """
    Get suggested corrections for validation errors.

    Args:
        errors: List of validation errors

    Returns:
        Dictionary of suggested corrections
    """
    return bootstrap_validator.suggest_corrections(errors)
