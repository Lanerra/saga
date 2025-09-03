# initialization/data_loader.py
from typing import Any

import structlog

import config
from models import CharacterProfile, WorldItem
from models.user_input_models import UserStoryInputModel, user_story_to_objects
from utils.yaml_parser import load_yaml_file

logger = structlog.get_logger(__name__)


def load_user_supplied_model() -> UserStoryInputModel | None:
    """Load user story YAML into a validated model."""
    data = load_yaml_file(config.USER_STORY_ELEMENTS_FILE_PATH)
    if not data:
        return None
    try:
        return UserStoryInputModel(**data)
    except Exception as exc:  # pragma: no cover
        logger.error(f"Failed to parse user story YAML: {exc}")
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
