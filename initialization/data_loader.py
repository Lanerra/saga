from typing import Any, Dict, Optional, Tuple

import structlog

from config import settings
from kg_maintainer.models import CharacterProfile, WorldItem
from models.user_input_models import UserStoryInputModel, user_story_to_objects
from yaml_parser import load_yaml_file

logger = structlog.get_logger(__name__)


def load_user_supplied_model() -> Optional[UserStoryInputModel]:
    """Load user story YAML into a validated model."""
    data = load_yaml_file(settings.USER_STORY_ELEMENTS_FILE_PATH)
    if not data:
        return None
    try:
        return UserStoryInputModel(**data)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to parse user story YAML: %s", exc)
        return None


def convert_model_to_objects(
    model: UserStoryInputModel,
) -> Tuple[
    Dict[str, Any],
    Dict[str, CharacterProfile],
    Dict[str, Dict[str, WorldItem]],
]:
    """Convert a validated model into internal objects."""
    return user_story_to_objects(model)
