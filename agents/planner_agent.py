# planner_agent.py
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import config
from core.llm_interface import llm_service
from data_access import chapter_queries
from models import CharacterProfile, SceneDetail, WorldItem
from prompt_data_getters import (
    get_character_state_snippet_for_prompt,
    get_reliable_kg_facts_for_drafting_prompt,
    get_world_state_snippet_for_prompt,
)
from prompt_renderer import render_prompt

logger = logging.getLogger(__name__)


class PlannerAgent:
    def __init__(self, model_name: str = config.PLANNING_MODEL):
        self.model_name = model_name
        logger.info(f"PlannerAgent initialized with model: {self.model_name}")


    async def plan_chapter_scenes(
        self,
        plot_outline: Dict[str, Any],
        character_profiles: Dict[str, CharacterProfile],
        world_building: Dict[str, Dict[str, WorldItem]],
        chapter_number: int,
        plot_point_focus: Optional[str],
        plot_point_index: int,
    ) -> Tuple[Optional[List[SceneDetail]], Optional[Dict[str, int]]]:
        """
        DEPRECATED: This method has been consolidated into NarrativeAgent._plan_chapter_scenes().
        Use NarrativeAgent.generate_chapter() instead.
        
        Generates a detailed scene plan for the chapter.
        Returns the plan and LLM usage data.
        """
        logger.warning(
            "PlannerAgent.plan_chapter_scenes() is deprecated. Use NarrativeAgent.generate_chapter() instead."
        )
        return None, None

    async def plan_continuation(
        self, summary_text: str, num_points: int = 5
    ) -> Tuple[Optional[List[str]], Optional[Dict[str, int]]]:
        """
        DEPRECATED: This method has been consolidated into NarrativeAgent.plan_continuation().
        Use NarrativeAgent.plan_continuation() instead.
        
        Generate future plot points from a story summary.
        """
        logger.warning(
            "PlannerAgent.plan_continuation() is deprecated. Use NarrativeAgent.plan_continuation() instead."
        )
        return None, None
