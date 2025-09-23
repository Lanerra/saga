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


def load_user_supplied_model() -> UserStoryInputModel | None:
    """Load user story YAML into a validated model."""
    data = load_yaml_file(config.USER_STORY_ELEMENTS_FILE_PATH)
    if not data:
        return None

    try:
        return UserStoryInputModel(**data)
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
