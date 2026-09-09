"""KG facts retrieval for scene-based context.

Provides knowledge-graph facts filtered by scene-specific entities (characters, location, related events).
"""

from typing import Any, cast

import structlog

import config
from core.langgraph.content_manager import ContentManager
from core.langgraph.state import NarrativeState
from core.text_processing_service import truncate_text_by_tokens
from models.agent_models import SceneDetail
from processing.scene_plan_parser import extract_scene_characters
from prompts.prompt_data_getters import get_reliable_kg_facts_for_drafting_prompt

logger = structlog.get_logger(__name__)

KG_FACTS_TOKEN_BUDGET = config.MAX_GENERATION_TOKENS


async def get_scene_kg_facts(
    state: NarrativeState,
    current_scene: dict,
    chapter_number: int,
    chapter_outlines: dict[int, dict],
    chapter_plan: list,
    model_name: str,
    content_manager: ContentManager,
) -> str | None:
    """Get KG facts filtered by scene-specific entities.

    Uses targeted queries focusing on:
    - Characters in the scene
    - Location of the scene
    - Related events and relationships

    Args:
        state: Workflow state.
        current_scene: Current scene definition.
        chapter_number: Current chapter number.
        chapter_outlines: Chapter outlines dictionary from content manager.
        chapter_plan: Full chapter plan (list of scenes).
        model_name: Model name for token counting.
        content_manager: Content manager instance.

    Returns:
        Formatted KG facts string or None.
    """
    scene_characters = extract_scene_characters(current_scene)

    scene_detail_untyped: dict[str, Any] = {
        "characters_involved": scene_characters,
        **current_scene,
    }
    scene_detail = cast(SceneDetail, scene_detail_untyped)

    protagonist_name = getattr(config, "DEFAULT_PROTAGONIST_NAME", "Protagonist")

    kg_facts_block = await get_reliable_kg_facts_for_drafting_prompt(
        chapter_outlines=chapter_outlines,
        chapter_number=chapter_number,
        chapter_plan=[scene_detail],
        protagonist_name=protagonist_name,
    )

    if kg_facts_block and "No specific reliable KG facts" not in kg_facts_block:
        truncated = truncate_text_by_tokens(
            text=kg_facts_block,
            model_name=model_name,
            max_tokens=KG_FACTS_TOKEN_BUDGET,
            truncation_marker="\n... (KG facts truncated for context budget)",
        )
        return truncated

    return None

