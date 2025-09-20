"""
Runtime configuration validation for ensuring consistency during novel generation.

This module provides functions to validate runtime configuration values against
bootstrap-generated content and ensure narrative consistency throughout the
generation process.
"""

from __future__ import annotations

import structlog
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import config
from models.kg_models import CharacterProfile, WorldItem
from models.validation_utils import (
    BootstrapContentValidator,
    ConfigurationValidationError
)

logger = structlog.get_logger(__name__)


class RuntimeConfigurationValidator:
    """
    Validates runtime configuration against bootstrap content and tracks consistency.

    This class ensures that runtime configuration values remain consistent with
    bootstrap-generated content throughout the novel generation process.
    """

    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self.validator = BootstrapContentValidator()
        self._validation_history: List[Dict[str, Any]] = []

    def validate_runtime_config_against_bootstrap(
        self,
        plot_outline: Dict[str, Any],
        character_profiles: Dict[str, CharacterProfile],
        world_building: Dict[str, Dict[str, WorldItem]]
    ) -> Tuple[bool, List[ConfigurationValidationError]]:
        """
        Validate runtime configuration against bootstrap content.

        Args:
            plot_outline: Current plot outline from database/runtime
            character_profiles: Current character profiles
            world_building: Current world building elements

        Returns:
            Tuple of (is_valid, validation_errors)
        """
        self.logger.info("Validating runtime configuration against bootstrap content")

        # Use the bootstrap validator to check consistency
        is_valid, validation_errors = self.validator.validate_all_components(
            plot_outline, character_profiles, world_building, "runtime"
        )

        # Record validation in history
        self._record_validation(
            is_valid,
            validation_errors,
            {
                "plot_outline_keys": list(plot_outline.keys()),
                "character_count": len(character_profiles),
                "world_categories": list(world_building.keys())
            }
        )

        return is_valid, validation_errors

    def validate_config_value_changes(
        self,
        old_values: Dict[str, Any],
        new_values: Dict[str, Any]
    ) -> List[ConfigurationValidationError]:
        """
        Validate changes to configuration values for potential impact.

        Args:
            old_values: Previous configuration values
            new_values: New configuration values

        Returns:
            List of validation errors for problematic changes
        """
        errors: List[ConfigurationValidationError] = []

        # Check for changes to critical narrative configuration
        critical_fields = [
            "CONFIGURED_GENRE",
            "CONFIGURED_THEME",
            "CONFIGURED_SETTING_DESCRIPTION",
            "DEFAULT_PROTAGONIST_NAME"
        ]

        for field in critical_fields:
            old_value = old_values.get(field)
            new_value = new_values.get(field)

            if old_value != new_value and old_value is not None:
                errors.append(
                    ConfigurationValidationError(
                        f"Critical configuration field '{field}' changed from '{old_value}' to '{new_value}' during runtime",
                        field.lower(),
                        old_value,
                        new_value
                    )
                )

        return errors

    def check_bootstrap_content_drift(
        self,
        initial_bootstrap: Dict[str, Any],
        current_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check for drift between initial bootstrap content and current content.

        Args:
            initial_bootstrap: Original bootstrap result
            current_content: Current content from database/runtime

        Returns:
            Dictionary describing any drift detected
        """
        drift_report = {
            "drift_detected": False,
            "drift_details": {},
            "recommendations": []
        }

        # Extract content for comparison
        initial_plot = initial_bootstrap.get("plot_outline", {})
        current_plot = current_content.get("plot_outline", {})

        # Check plot outline drift
        plot_drift = self._compare_dict_content(initial_plot, current_plot, "plot_outline")
        if plot_drift["differences_found"]:
            drift_report["drift_detected"] = True
            drift_report["drift_details"]["plot_outline"] = plot_drift

        # Check character profiles drift
        initial_chars = initial_bootstrap.get("character_profiles", {})
        current_chars = current_content.get("character_profiles", {})

        char_drift = self._compare_character_profiles(initial_chars, current_chars)
        if char_drift["differences_found"]:
            drift_report["drift_detected"] = True
            drift_report["drift_details"]["character_profiles"] = char_drift

        # Check world building drift
        initial_world = initial_bootstrap.get("world_building", {})
        current_world = current_content.get("world_building", {})

        world_drift = self._compare_world_building(initial_world, current_world)
        if world_drift["differences_found"]:
            drift_report["drift_detected"] = True
            drift_report["drift_details"]["world_building"] = world_drift

        # Generate recommendations
        if drift_report["drift_detected"]:
            drift_report["recommendations"] = self._generate_drift_recommendations(drift_report)

        return drift_report

    def _compare_dict_content(
        self,
        initial: Dict[str, Any],
        current: Dict[str, Any],
        content_type: str
    ) -> Dict[str, Any]:
        """
        Compare two dictionaries for differences.

        Args:
            initial: Initial content dictionary
            current: Current content dictionary
            content_type: Type of content being compared

        Returns:
            Dictionary describing differences
        """
        differences = {
            "differences_found": False,
            "missing_keys": [],
            "extra_keys": [],
            "changed_values": {},
            "content_type": content_type
        }

        # Check for missing keys
        for key in initial.keys():
            if key not in current:
                differences["missing_keys"].append(key)

        # Check for extra keys
        for key in current.keys():
            if key not in initial:
                differences["extra_keys"].append(key)

        # Check for changed values
        for key in initial.keys():
            if key in current and initial[key] != current[key]:
                differences["changed_values"][key] = {
                    "initial": initial[key],
                    "current": current[key]
                }

        differences["differences_found"] = (
            len(differences["missing_keys"]) > 0 or
            len(differences["extra_keys"]) > 0 or
            len(differences["changed_values"]) > 0
        )

        return differences

    def _compare_character_profiles(
        self,
        initial: Dict[str, CharacterProfile],
        current: Dict[str, CharacterProfile]
    ) -> Dict[str, Any]:
        """
        Compare character profiles for differences.

        Args:
            initial: Initial character profiles
            current: Current character profiles

        Returns:
            Dictionary describing differences
        """
        differences = {
            "differences_found": False,
            "missing_characters": [],
            "extra_characters": [],
            "changed_characters": {},
            "content_type": "character_profiles"
        }

        initial_names = set(initial.keys())
        current_names = set(current.keys())

        # Check for missing characters
        differences["missing_characters"] = list(initial_names - current_names)

        # Check for extra characters
        differences["extra_characters"] = list(current_names - initial_names)

        # Check for changed character properties
        for name in initial_names.intersection(current_names):
            initial_profile = initial[name]
            current_profile = current[name]

            if isinstance(initial_profile, CharacterProfile) and isinstance(current_profile, CharacterProfile):
                if initial_profile.to_dict() != current_profile.to_dict():
                    differences["changed_characters"][name] = {
                        "initial": initial_profile.to_dict(),
                        "current": current_profile.to_dict()
                    }

        differences["differences_found"] = (
            len(differences["missing_characters"]) > 0 or
            len(differences["extra_characters"]) > 0 or
            len(differences["changed_characters"]) > 0
        )

        return differences

    def _compare_world_building(
        self,
        initial: Dict[str, Dict[str, WorldItem]],
        current: Dict[str, Dict[str, WorldItem]]
    ) -> Dict[str, Any]:
        """
        Compare world building content for differences.

        Args:
            initial: Initial world building
            current: Current world building

        Returns:
            Dictionary describing differences
        """
        differences = {
            "differences_found": False,
            "missing_categories": [],
            "extra_categories": [],
            "changed_items": {},
            "content_type": "world_building"
        }

        initial_categories = set(initial.keys())
        current_categories = set(current.keys())

        # Check for missing categories
        differences["missing_categories"] = list(initial_categories - current_categories)

        # Check for extra categories
        differences["extra_categories"] = list(current_categories - initial_categories)

        # Check for changed items within categories
        for category in initial_categories.intersection(current_categories):
            initial_items = initial[category]
            current_items = current[category]

            initial_item_names = set(initial_items.keys())
            current_item_names = set(current_items.keys())

            # Check for missing items
            missing_items = initial_item_names - current_item_names
            if missing_items:
                if category not in differences["changed_items"]:
                    differences["changed_items"][category] = {}
                differences["changed_items"][category]["missing_items"] = list(missing_items)

            # Check for extra items
            extra_items = current_item_names - initial_item_names
            if extra_items:
                if category not in differences["changed_items"]:
                    differences["changed_items"][category] = {}
                differences["changed_items"][category]["extra_items"] = list(extra_items)

            # Check for changed items
            for item_name in initial_item_names.intersection(current_item_names):
                initial_item = initial_items[item_name]
                current_item = current_items[item_name]

                if isinstance(initial_item, WorldItem) and isinstance(current_item, WorldItem):
                    if initial_item.to_dict() != current_item.to_dict():
                        if category not in differences["changed_items"]:
                            differences["changed_items"][category] = {}
                        if "changed_items" not in differences["changed_items"][category]:
                            differences["changed_items"][category]["changed_items"] = {}
                        differences["changed_items"][category]["changed_items"][item_name] = {
                            "initial": initial_item.to_dict(),
                            "current": current_item.to_dict()
                        }

        differences["differences_found"] = (
            len(differences["missing_categories"]) > 0 or
            len(differences["extra_categories"]) > 0 or
            len(differences["changed_items"]) > 0
        )

        return differences

    def _generate_drift_recommendations(self, drift_report: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations for addressing content drift.

        Args:
            drift_report: Report describing detected drift

        Returns:
            List of recommendation strings
        """
        recommendations = []

        drift_details = drift_report.get("drift_details", {})

        if "plot_outline" in drift_details:
            plot_drift = drift_details["plot_outline"]
            if plot_drift.get("differences_found"):
                recommendations.append(
                    "Plot outline has drifted from bootstrap values. Consider re-running bootstrap or updating configuration."
                )

        if "character_profiles" in drift_details:
            char_drift = drift_details["character_profiles"]
            if char_drift.get("differences_found"):
                recommendations.append(
                    "Character profiles have changed since bootstrap. Validate character consistency."
                )

        if "world_building" in drift_details:
            world_drift = drift_details["world_building"]
            if world_drift.get("differences_found"):
                recommendations.append(
                    "World building content has drifted. Check world consistency and consider re-bootstrap."
                )

        if not recommendations:
            recommendations.append("No specific recommendations available for detected drift.")

        return recommendations

    def _record_validation(
        self,
        is_valid: bool,
        validation_errors: List[ConfigurationValidationError],
        context: Dict[str, Any]
    ) -> None:
        """
        Record validation result in history.

        Args:
            is_valid: Whether validation passed
            validation_errors: List of validation errors
            context: Additional context information
        """
        validation_record = {
            "timestamp": datetime.now().isoformat(),
            "is_valid": is_valid,
            "error_count": len(validation_errors),
            "context": context
        }

        self._validation_history.append(validation_record)

        # Keep only last 10 validations to prevent memory issues
        if len(self._validation_history) > 10:
            self._validation_history = self._validation_history[-10:]

    def get_validation_history(self) -> List[Dict[str, Any]]:
        """
        Get validation history.

        Returns:
            List of validation records
        """
        return self._validation_history.copy()

    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of validation history.

        Returns:
            Dictionary with validation summary statistics
        """
        if not self._validation_history:
            return {"total_validations": 0, "success_rate": 0.0}

        total_validations = len(self._validation_history)
        successful_validations = sum(1 for record in self._validation_history if record["is_valid"])

        return {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "failed_validations": total_validations - successful_validations,
            "success_rate": successful_validations / total_validations,
            "last_validation": self._validation_history[-1] if self._validation_history else None
        }


# Global validator instance
runtime_config_validator = RuntimeConfigurationValidator()


def validate_runtime_configuration(
    plot_outline: Dict[str, Any],
    character_profiles: Dict[str, CharacterProfile],
    world_building: Dict[str, Dict[str, WorldItem]]
) -> Tuple[bool, List[ConfigurationValidationError]]:
    """
    Convenience function to validate runtime configuration.

    Args:
        plot_outline: Current plot outline from database/runtime
        character_profiles: Current character profiles
        world_building: Current world building elements

    Returns:
        Tuple of (is_valid, validation_errors)
    """
    return runtime_config_validator.validate_runtime_config_against_bootstrap(
        plot_outline, character_profiles, world_building
    )


def check_bootstrap_drift(
    initial_bootstrap: Dict[str, Any],
    current_content: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check for drift between bootstrap and current content.

    Args:
        initial_bootstrap: Original bootstrap result
        current_content: Current content from database/runtime

    Returns:
        Dictionary describing any drift detected
    """
    return runtime_config_validator.check_bootstrap_content_drift(
        initial_bootstrap, current_content
    )
