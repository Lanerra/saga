# world_continuity_agent.py
# from parsing_utils import split_text_into_blocks,
# parse_key_value_block  # Removed
import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import config
import utils  # MODIFIED: For spaCy functions
from core.llm_interface import llm_service  # MODIFIED
from data_access import character_queries, kg_queries, world_queries
from models import ProblemDetail, SceneDetail
from processing.problem_parser import parse_problem_list
from prompt_data_getters import (
    get_filtered_character_profiles_for_prompt_plain_text,
    get_filtered_world_data_for_prompt_plain_text,
)
from prompt_renderer import render_prompt

logger = logging.getLogger(__name__)

PROBLEM_DETAIL_KEY_MAP = {
    "issue_category": "issue_category",
    "problem_description": "problem_description",
    "quote_from_original": "quote_from_original_text",  # Maps to text field
    "suggested_fix_focus": "suggested_fix_focus",
}


class WorldContinuityAgent:
    def __init__(self, model_name: str = config.EVALUATION_MODEL):
        self.model_name = model_name
        logger.info(f"WorldContinuityAgent initialized with model: {self.model_name}")
        utils.load_spacy_model_if_needed()  # Ensure spaCy model is available

    async def _parse_llm_consistency_output(
        self, json_text: str, chapter_number: int, original_draft_text: str
    ) -> List[ProblemDetail]:
        """Parse LLM JSON output for consistency problems."""
        problems = parse_problem_list(json_text, category="consistency")
        if not problems:
            logger.info(
                f"Consistency check for Ch {chapter_number} yielded no problems."
            )
            return []

        for i, prob in enumerate(problems):
            quote_text = prob["quote_from_original_text"]
            if (
                "N/A - General Issue" in quote_text
                or not quote_text.strip()
                or quote_text == "N/A"
            ):
                prob["quote_from_original_text"] = "N/A - General Issue"
            elif utils.spacy_manager.nlp is not None and original_draft_text.strip():
                offsets = await utils.find_quote_and_sentence_offsets_with_spacy(
                    original_draft_text, quote_text
                )
                if offsets:
                    q_start, q_end, s_start, s_end = offsets
                    prob["quote_char_start"] = q_start
                    prob["quote_char_end"] = q_end
                    prob["sentence_char_start"] = s_start
                    prob["sentence_char_end"] = s_end
                else:
                    logger.warning(
                        f"Ch {chapter_number} consistency problem {i + 1}: Could not find quote via spaCy: '{quote_text[:50]}...'"
                    )
            elif not original_draft_text.strip():
                logger.warning(
                    f"Ch {chapter_number} consistency problem {i + 1}: Original draft text is empty. Cannot find offsets for quote: '{quote_text[:50]}...'"
                )
            else:
                logger.info(
                    f"Ch {chapter_number} consistency problem {i + 1}: spaCy not available, quote offsets not determined for: '{quote_text[:50]}...'"
                )

        return problems

    async def check_consistency(
        self,
        plot_outline: Dict[str, Any],
        draft_text: str,
        chapter_number: int,
        previous_chapters_context: str,
        ignore_spans: Optional[List[Tuple[int, int]]] | None = None,
    ) -> Tuple[List[ProblemDetail], Optional[Dict[str, int]]]:
        if not draft_text:
            logger.warning(
                f"WorldContinuityAgent: Consistency check skipped for Ch {chapter_number}: empty draft text."
            )
            return [], None

        processed_text = utils.remove_spans_from_text(draft_text, ignore_spans or [])
        logger.info(
            f"WorldContinuityAgent performing focused consistency check for Chapter {chapter_number}..."
        )

        protagonist_name_str = plot_outline.get("protagonist_name", "The Protagonist")
        characters = await character_queries.get_character_profiles_from_db()
        world_item_ids_by_category = (
            await world_queries.get_all_world_item_ids_by_category()
        )
        char_profiles_plain_text = (
            await get_filtered_character_profiles_for_prompt_plain_text(
                characters,
                chapter_number - 1,
            )
        )
        world_building_plain_text = await get_filtered_world_data_for_prompt_plain_text(
            world_item_ids_by_category,
            chapter_number - 1,
        )

        plot_points_summary_lines = (
            [
                f"- PP {i + 1}: {pp[:100]}..."
                for i, pp in enumerate(plot_outline.get("plot_points", []))
            ]
            if plot_outline.get("plot_points")
            else ["  - Not available"]
        )
        plot_points_summary_str = "\n".join(plot_points_summary_lines)

        # This whole block will replace the existing few_shot_consistency_example_str
        few_shot_consistency_example_str = """
**Ignore the narrative details in this example. It shows the required format only.**
[
  {
    "issue_category": "consistency",
    "problem_description": "The 'Sunstone' is described as glowing blue in this"
    " chapter, but the world building notes explicitly state all Sunstones are"
    " crimson red.",
    "quote_from_original_text": "She admired the brilliant blue glow of the"
    " Sunstone clutched in her hand.",
    "suggested_fix_focus": "Change the Sunstone's color to 'crimson red' to"
    " align with established world canon."
  },
  {
    "issue_category": "consistency",
    "problem_description": "Character Kael claims to have never met Elara before,"
    " but Previous Chapter Context (KG Fact) states \"Kael | mentored | Elara"
    " (Ch: 3)\".",
    "quote_from_original_text": "\"I do not believe we have crossed paths"
    " before, young one,\" Kael said, peering at Elara.",
    "suggested_fix_focus": "Adjust Kael's dialogue to acknowledge his prior"
    " mentorship of Elara, or introduce a reason for his feigned ignorance"
    " (e.g., memory loss, testing her)."
  }
]
"""
        prompt = render_prompt(
            "world_continuity_agent/consistency_check.j2",
            {
                "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
                "chapter_number": chapter_number,
                "novel_title": plot_outline.get("title", "Untitled Novel"),
                "protagonist_name_str": protagonist_name_str,
                "novel_genre": plot_outline.get("genre", "N/A"),
                "novel_theme": plot_outline.get("theme", "N/A"),
                "novel_protagonist": plot_outline.get("protagonist_name", "N/A"),
                "protagonist_arc": plot_outline.get("character_arc", "N/A"),
                "logline": plot_outline.get("logline", "N/A"),
                "plot_points_summary_str": plot_points_summary_str,
                "char_profiles_plain_text": char_profiles_plain_text,
                "world_building_plain_text": world_building_plain_text,
                "previous_chapters_context": previous_chapters_context,
                "draft_text": processed_text,
                "few_shot_consistency_example_str": few_shot_consistency_example_str,
            },
        )

        logger.info(
            f"Calling LLM ({self.model_name}) for World/Continuity consistency"
            f" check of chapter {chapter_number} (expecting JSON)..."
        )
        (
            cleaned_consistency_text,
            usage_data,
        ) = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=config.Temperatures.CONSISTENCY_CHECK,
            allow_fallback=True,
            stream_to_disk=False,
            auto_clean_response=True,
        )

        consistency_problems = await self._parse_llm_consistency_output(
            cleaned_consistency_text, chapter_number, draft_text
        )

        logger.info(
            f"World/Continuity consistency check for Ch {chapter_number} found"
            f" {len(consistency_problems)} problems."
        )
        return consistency_problems, usage_data

    async def check_scene_plan_consistency(
        self,
        plot_outline: Dict[str, Any],
        scene_plan: List[SceneDetail],
        chapter_number: int,
    ) -> Tuple[List[ProblemDetail], Optional[Dict[str, int]]]:
        """Validate a scene plan before drafting begins."""

        if not scene_plan:
            logger.warning(
                "WorldContinuityAgent: Scene plan consistency check skipped for Ch %s: empty plan.",
                chapter_number,
            )
            return [], None

        protagonist_name_str = plot_outline.get("protagonist_name", "The Protagonist")
        characters = await character_queries.get_character_profiles_from_db()
        world_item_ids_by_category = (
            await world_queries.get_all_world_item_ids_by_category()
        )
        char_profiles_plain_text = (
            await get_filtered_character_profiles_for_prompt_plain_text(
                characters,
                chapter_number - 1,
            )
        )
        world_building_plain_text = await get_filtered_world_data_for_prompt_plain_text(
            world_item_ids_by_category,
            chapter_number - 1,
        )

        plot_points_summary_lines = (
            [
                f"- PP {i + 1}: {pp[:100]}..."
                for i, pp in enumerate(plot_outline.get("plot_points", []))
            ]
            if plot_outline.get("plot_points")
            else ["  - Not available"]
        )
        plot_points_summary_str = "\n".join(plot_points_summary_lines)

        few_shot_consistency_example_str = """
**Ignore the narrative details in this example. It shows the required format only.**
[
  {
    "issue_category": "consistency",
    "problem_description": "The 'Sunstone' is described as glowing blue in this"
    " chapter, but the world building notes explicitly state all Sunstones are"
    " crimson red.",
    "quote_from_original_text": "She admired the brilliant blue glow of the"
    " Sunstone clutched in her hand.",
    "suggested_fix_focus": "Change the Sunstone's color to 'crimson red' to"
    " align with established world canon."
  },
  {
    "issue_category": "consistency",
    "problem_description": "Character Kael claims to have never met Elara before",
    " but Previous Chapter Context (KG Fact) states \"Kael | mentored | Elara"
    " (Ch: 3)\".",
    "quote_from_original_text": "\"I do not believe we have crossed paths"
    " before, young one,\" Kael said, peering at Elara.",
    "suggested_fix_focus": "Adjust Kael's dialogue to acknowledge his prior"
    " mentorship of Elara, or introduce a reason for his feigned ignorance"
    " (e.g., memory loss, testing her)."
  }
]
"""

        prompt = render_prompt(
            "world_continuity_agent/plan_consistency_check.j2",
            {
                "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
                "chapter_number": chapter_number,
                "novel_title": plot_outline.get("title", "Untitled Novel"),
                "protagonist_name_str": protagonist_name_str,
                "novel_genre": plot_outline.get("genre", "N/A"),
                "novel_theme": plot_outline.get("theme", "N/A"),
                "novel_protagonist": plot_outline.get("protagonist_name", "N/A"),
                "protagonist_arc": plot_outline.get("character_arc", "N/A"),
                "logline": plot_outline.get("logline", "N/A"),
                "plot_points_summary_str": plot_points_summary_str,
                "char_profiles_plain_text": char_profiles_plain_text,
                "world_building_plain_text": world_building_plain_text,
                "scene_plan_json": json.dumps(scene_plan, ensure_ascii=False, indent=2),
                "few_shot_consistency_example_str": few_shot_consistency_example_str,
            },
        )

        logger.info(
            "Calling LLM (%s) for scene plan consistency check of chapter %s (expecting JSON)...",
            self.model_name,
            chapter_number,
        )
        cleaned_consistency_text, usage_data = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=config.Temperatures.CONSISTENCY_CHECK,
            allow_fallback=True,
            stream_to_disk=False,
            auto_clean_response=True,
        )

        consistency_problems = await self._parse_llm_consistency_output(
            cleaned_consistency_text, chapter_number, json.dumps(scene_plan)
        )

        logger.info(
            "Scene plan consistency check for Ch %s found %s problems.",
            chapter_number,
            len(consistency_problems),
        )
        return consistency_problems, usage_data

    async def suggest_canon_corrections(
        self, problems: List[ProblemDetail]
    ) -> List[str]:
        """Provide actionable suggestions using canonical information.

        Args:
            problems: Detected consistency issues.

        Returns:
            A list of correction messages referencing known canon.
        """

        suggestions: List[str] = []
        if not problems:
            return suggestions

        utils.load_spacy_model_if_needed()
        for problem in problems:
            if problem.get("issue_category") != "consistency":
                continue

            base_suggestion = problem.get("suggested_fix_focus", "")
            text_for_entities = " ".join(
                [problem.get("problem_description", ""), base_suggestion]
            )
            entities: Set[str] = set()
            if utils.spacy_manager.nlp:
                doc = utils.spacy_manager.nlp(text_for_entities)
                entities.update(ent.text for ent in doc.ents)
            else:
                entities.update(
                    word for word in text_for_entities.split() if word.istitle()
                )

            canonical_facts: List[str] = []
            for entity in entities:
                facts = await self.query_kg_for_contradiction(entity)
                for fact in facts[:3]:
                    canonical_facts.append(
                        f"{fact.get('subject')} -{fact.get('predicate')}-> {fact.get('object')} (Ch: {fact.get('chapter_added')})"
                    )
                if canonical_facts:
                    break

            suggestion_text = base_suggestion
            if canonical_facts:
                suggestion_text += f" Reference canon: {'; '.join(canonical_facts)}."

            suggestions.append(suggestion_text)

        return suggestions

    async def query_kg_for_contradiction(
        self,
        entity1: str,
        entity2: Optional[str] = None,
        relation: Optional[str] = None,
        chapter_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        logger.info(
            f"WorldContinuityAgent: Querying KG regarding potential"
            f" contradiction around '{entity1}'..."
        )
        facts = []
        facts.extend(
            await kg_queries.query_kg_from_db(
                subject=entity1,
                chapter_limit=chapter_limit,
                include_provisional=False,
            )
        )
        facts.extend(
            await kg_queries.query_kg_from_db(
                obj_val=entity1,
                chapter_limit=chapter_limit,
                include_provisional=False,
            )
        )

        if entity2 and relation:
            facts.extend(
                await kg_queries.query_kg_from_db(
                    subject=entity1,
                    predicate=relation,
                    obj_val=entity2,
                    chapter_limit=chapter_limit,
                    include_provisional=False,
                )
            )

        unique_facts_strs = set()
        formatted_facts = []
        for fact_dict in facts:
            fact_str = (
                f"{fact_dict.get('subject')} -{fact_dict.get('predicate')}->"
                f" {fact_dict.get('object')} (Ch: {fact_dict.get('chapter_added')}"
                f", Prov: {fact_dict.get('is_provisional')})"
            )
            if fact_str not in unique_facts_strs:
                formatted_facts.append(fact_dict)
                unique_facts_strs.add(fact_str)
        logger.info(
            f"Found {len(formatted_facts)} non-provisional KG facts related to"
            f" '{entity1}' (up to chapter {chapter_limit})."
        )
        return formatted_facts
