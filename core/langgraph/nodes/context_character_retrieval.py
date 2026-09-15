"""Character profile retrieval for scene-based context.

Provides character context building filtered by characters appearing in each scene.
"""

import structlog

import config
from core.langgraph.content_manager import ContentManager
from core.langgraph.state import NarrativeState
from core.text_processing_service import truncate_text_by_tokens
from processing.scene_plan_parser import extract_scene_characters
from prompts.prompt_data_getters import (
    get_filtered_character_profiles_for_prompt_plain_text,
)

logger = structlog.get_logger(__name__)

CHARACTER_PROFILES_TOKEN_BUDGET = config.MAX_GENERATION_TOKENS


async def get_scene_character_context(
    state: NarrativeState,
    current_scene: dict,
    chapter_number: int,
    model_name: str,
    content_manager: ContentManager,
) -> str | None:
    """Build a token-budgeted character profile block for the current scene.

    Args:
        state: Workflow state.
        current_scene: Scene plan entry that may include a character list.
        chapter_number: Current chapter number for limiting profile retrieval.
        model_name: Model name used for token counting.
        content_manager: Content manager instance.

    Returns:
        Formatted character profiles block, or `None` when the scene does not specify
        characters or successful queries find no profiles.

    Raises:
        Exception: Propagates required profile read failures to retrieve_context's
            fatal-error policy; a failed query is not an empty character context.
    """
    scene_characters = extract_scene_characters(current_scene)

    if not scene_characters:
        logger.debug("context_character: no characters specified in scene")
        return None

    logger.debug(
        "context_character: filtering profiles for scene characters",
        characters=scene_characters,
    )

    character_profiles_text = await get_filtered_character_profiles_for_prompt_plain_text(
        character_names=scene_characters,
        up_to_chapter_inclusive=chapter_number - 1 if chapter_number > 1 else config.KG_PREPOPULATION_CHAPTER_NUM,
    )

    if character_profiles_text and character_profiles_text != "No character profiles available.":
        truncated = truncate_text_by_tokens(
            text=character_profiles_text,
            model_name=model_name,
            max_tokens=CHARACTER_PROFILES_TOKEN_BUDGET,
            truncation_marker="\n... (character profiles truncated for context budget)",
        )
        return f"**Scene Character Profiles:**\n{truncated}"

    return None

