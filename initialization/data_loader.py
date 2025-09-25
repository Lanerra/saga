# initialization/data_loader.py
from typing import Any

import structlog
from pydantic import ValidationError

import config
from models import CharacterProfile, WorldItem
from models.user_input_models import UserStoryInputModel, user_story_to_objects
from utils.common import load_yaml_file

from .error_handling import ErrorSeverity, handle_bootstrap_error

logger = structlog.get_logger(__name__)


def _apply_user_story_config_overrides(model: UserStoryInputModel) -> None:
    """Update runtime config to align critical fields with user-supplied story."""

    overrides: dict[str, str] = {}

    concept = model.novel_concept
    if concept:
        if concept.genre:
            overrides["CONFIGURED_GENRE"] = concept.genre
        if concept.theme:
            overrides["CONFIGURED_THEME"] = concept.theme

    # Prefer explicit protagonist field, fall back to grouped characters
    protagonist = model.protagonist
    if not protagonist and model.characters:
        protagonist = model.characters.protagonist
    if protagonist and protagonist.name:
        overrides["DEFAULT_PROTAGONIST_NAME"] = protagonist.name

    setting_description: str | None = None
    if model.setting and model.setting.primary_setting_overview:
        setting_description = model.setting.primary_setting_overview
    if concept and concept.setting:
        setting_description = concept.setting

    if setting_description:
        overrides["CONFIGURED_SETTING_DESCRIPTION"] = setting_description

        # Keep concept/setting models consistent with override so downstream validation passes
        if concept and concept.setting != setting_description:
            concept.setting = setting_description
        if model.setting and model.setting.primary_setting_overview != setting_description:
            model.setting.primary_setting_overview = setting_description

    if not overrides:
        return

    for key, value in overrides.items():
        config.set(key, value)
        setattr(config, key, value)

    logger.info(
        "Applied user story overrides to runtime configuration.",
        overrides=overrides,
    )


def load_user_supplied_model() -> UserStoryInputModel | None:
    """Load user story YAML into a validated model."""
    data = load_yaml_file(config.USER_STORY_ELEMENTS_FILE_PATH)
    if not data:
        return None

    try:
        model = UserStoryInputModel(**data)
        _apply_user_story_config_overrides(model)
        logger.info(
            "Loaded user story elements from file.",
            file_path=config.USER_STORY_ELEMENTS_FILE_PATH,
            novel_title=model.novel_concept.title if model.novel_concept else None,
        )
        return model
    except ValidationError as exc:
        # Pydantic validation errors - these are expected and should be handled gracefully
        handle_bootstrap_error(
            exc,
            "User story YAML validation",
            ErrorSeverity.ERROR,
            {
                "file_path": config.USER_STORY_ELEMENTS_FILE_PATH,
                "validation_errors": exc.errors(),
            },
        )
        return None
    except (TypeError, ValueError) as exc:
        # Data type or value errors - more specific than generic Exception
        handle_bootstrap_error(
            exc,
            "User story YAML data parsing",
            ErrorSeverity.ERROR,
            {"file_path": config.USER_STORY_ELEMENTS_FILE_PATH},
        )
        return None
    except Exception as exc:
        # Unexpected errors - these should be investigated
        handle_bootstrap_error(
            exc,
            "User story YAML loading",
            ErrorSeverity.CRITICAL,
            {"file_path": config.USER_STORY_ELEMENTS_FILE_PATH},
        )
        return None


def convert_model_to_objects(
    model: UserStoryInputModel,
) -> tuple[
    dict[str, Any],
    dict[str, CharacterProfile],
    dict[str, dict[str, WorldItem]],
]:
    """Convert a validated model into internal objects."""
    return user_story_to_objects(model)
