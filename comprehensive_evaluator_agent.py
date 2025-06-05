# comprehensive_evaluator_agent.py
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple

import config
from llm_interface import llm_service  # MODIFIED
import utils  # MODIFIED: For spaCy functions
from type import EvaluationResult, ProblemDetail
from data_access import chapter_queries
from prompt_data_getters import (
    get_filtered_character_profiles_for_prompt_plain_text,
    get_filtered_world_data_for_prompt_plain_text,
    get_reliable_kg_facts_for_drafting_prompt,
)

# from parsing_utils import split_text_into_blocks # No longer needed if parsing JSON list
# from parsing_utils import parse_key_value_block # Removed
import json  # Added for JSON parsing

logger = logging.getLogger(__name__)

PROBLEM_DETAIL_KEY_MAP = {
    "issue_category": "issue_category",
    "problem_description": "problem_description",
    "quote_from_original": "quote_from_original_text",  # Maps to text field
    "suggested_fix_focus": "suggested_fix_focus",
}

# Standardized internal category names
INTERNAL_VALID_CATEGORIES = {
    "consistency",
    "plot_arc",
    "thematic_alignment",  # Standardized internal name
    "narrative_depth_and_length",  # Standardized internal name
    "meta",
}


def _normalize_llm_category_to_internal(llm_category_str: str) -> str:
    """Normalizes category names from LLM output to internal standard names."""
    normalized = llm_category_str.lower().strip().replace(" ", "_")
    if "thematic" in normalized:  # Catches "thematic_alignment" or "thematic"
        return "thematic_alignment"
    if (
        "narrative_depth" in normalized
    ):  # Catches "narrative_depth_and_length" or "narrative_depth"
        return "narrative_depth_and_length"
    # For exact matches or other categories like consistency, plot_arc, meta
    if normalized in INTERNAL_VALID_CATEGORIES:
        return normalized
    return "meta"  # Default if no clear mapping


class ComprehensiveEvaluatorAgent:
    def __init__(self, model_name: str = config.EVALUATION_MODEL):
        self.model_name = model_name
        logger.info(
            f"ComprehensiveEvaluatorAgent initialized with model: {self.model_name}"
        )
        utils.load_spacy_model_if_needed()  # Ensure spaCy model is available

    async def _parse_llm_evaluation_output(
        self, json_text: str, chapter_number: int, original_draft_text: str
    ) -> List[ProblemDetail]:
        """
        Parses LLM JSON output for chapter evaluation problems.
        Populates character offsets for the quote and its containing sentence using spaCy.
        """
        final_problems: List[ProblemDetail] = []

        if not json_text or not json_text.strip():
            logger.info(
                f"JSON evaluation for Ch {chapter_number} is empty. No problems parsed."
            )
            return []

        try:
            # LLM is expected to output a JSON list of problem objects.
            # Each object should have keys like: issue_category, problem_description,
            # quote_from_original_text, suggested_fix_focus
            parsed_data = json.loads(json_text)
            if not isinstance(parsed_data, list):
                # Handle cases where LLM might return a single JSON object or a dict with a status
                if (
                    isinstance(parsed_data, dict)
                    and "status" in parsed_data
                    and "no significant problems found" in parsed_data["status"].lower()
                ):
                    logger.info(
                        f"JSON evaluation for Ch {chapter_number} indicates no problems: {parsed_data}"
                    )
                    return []
                if (
                    isinstance(parsed_data, dict)
                    and "problems" in parsed_data
                    and isinstance(parsed_data["problems"], list)
                ):
                    logger.info(
                        f"JSON evaluation for Ch {chapter_number} has problems nested under 'problems' key."
                    )
                    parsed_data = parsed_data[
                        "problems"
                    ]  # Process the list of problems
                else:
                    logger.error(
                        f"LLM evaluation output was not a JSON list of problems as expected. Received type: {type(parsed_data)}. Content: {json_text[:300]}"
                    )
                    final_problems.append(
                        {
                            "issue_category": "meta",
                            "problem_description": "LLM output was not a list of problems.",
                            "quote_from_original_text": "N/A - LLM Output Format Error",
                            "quote_char_start": None,
                            "quote_char_end": None,
                            "sentence_char_start": None,
                            "sentence_char_end": None,
                            "suggested_fix_focus": "Ensure LLM outputs a JSON list of problem objects.",
                        }
                    )
                    return final_problems

            if not parsed_data:  # Empty list from JSON
                logger.info(
                    f"JSON evaluation for Ch {chapter_number} was an empty list. No problems parsed."
                )
                return []

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode JSON from LLM evaluation output for Ch {chapter_number}: {e}. Text: {json_text[:500]}..."
            )
            # Check for common "no problems" text even if JSON is malformed around it
            if "no significant problems found" in json_text.lower():
                logger.info(
                    "JSON decode error, but text indicates no significant problems."
                )
                return []
            final_problems.append(
                {
                    "issue_category": "meta",
                    "problem_description": f"Invalid JSON from LLM: {e}",
                    "quote_from_original_text": "N/A - Invalid JSON",
                    "quote_char_start": None,
                    "quote_char_end": None,
                    "sentence_char_start": None,
                    "sentence_char_end": None,
                    "suggested_fix_focus": "Review LLM output for JSON validity.",
                }
            )
            return final_problems

        for i, problem_dict in enumerate(parsed_data):
            if not isinstance(problem_dict, dict):
                logger.warning(
                    f"Problem item {i + 1} in JSON list for Ch {chapter_number} is not a dictionary. Skipping. Item: {problem_dict}"
                )
                continue

            problem_meta: ProblemDetail = {
                "issue_category": _normalize_llm_category_to_internal(
                    problem_dict.get("issue_category", "meta")
                ),
                "problem_description": problem_dict.get(
                    "problem_description", "N/A - Missing description from LLM"
                ),
                "quote_from_original_text": problem_dict.get(
                    "quote_from_original_text", "N/A - General Issue"
                ),
                "quote_char_start": None,
                "quote_char_end": None,
                "sentence_char_start": None,
                "sentence_char_end": None,
                "suggested_fix_focus": problem_dict.get(
                    "suggested_fix_focus", "N/A - Missing suggestion from LLM"
                ),
            }

            if (
                problem_meta["issue_category"] == "meta"
                and str(problem_dict.get("issue_category", "")).lower().strip()
                != "meta"
            ):
                logger.warning(
                    f"LLM provided category '{problem_dict.get('issue_category')}' in problem {i + 1} for Ch {chapter_number}, which normalized to 'meta'."
                )

            quote_text_from_llm = problem_meta["quote_from_original_text"]
            if (
                "N/A - General Issue" in quote_text_from_llm
                or not quote_text_from_llm.strip()
                or quote_text_from_llm == "N/A"
            ):
                problem_meta["quote_from_original_text"] = "N/A - General Issue"
            elif utils.spacy_manager.nlp is not None and original_draft_text.strip():
                offsets_tuple = await utils.find_quote_and_sentence_offsets_with_spacy(
                    original_draft_text, quote_text_from_llm
                )
                if offsets_tuple:
                    q_start, q_end, s_start, s_end = offsets_tuple
                    problem_meta["quote_char_start"] = q_start
                    problem_meta["quote_char_end"] = q_end
                    problem_meta["sentence_char_start"] = s_start
                    problem_meta["sentence_char_end"] = s_end
                else:
                    logger.warning(
                        f"Ch {chapter_number} problem {i + 1}: Could not find quote via spaCy: '{quote_text_from_llm[:50]}...'"
                    )
            elif not original_draft_text.strip():
                logger.warning(
                    f"Ch {chapter_number} problem {i + 1}: Original draft text is empty. Cannot find offsets for quote: '{quote_text_from_llm[:50]}...'"
                )
            else:  # spaCy not loaded
                logger.info(
                    f"Ch {chapter_number} problem {i + 1}: spaCy not available, quote offsets not determined for: '{quote_text_from_llm[:50]}...'"
                )

            final_problems.append(problem_meta)
        return final_problems

    async def _perform_llm_comprehensive_evaluation(
        self,
        novel_props: Dict[str, Any],
        draft_text: str,
        chapter_number: int,
        plot_point_focus: Optional[str],
        plot_point_index: int,
        previous_chapters_context: str,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
        if not draft_text:
            logger.warning(
                f"Comprehensive evaluation skipped for Ch {chapter_number}: empty draft text."
            )
            return {
                "problems_found_text_output": "Draft is empty.",
                "legacy_consistency_issues": "Skipped (empty draft)",
                "legacy_plot_arc_deviation": "Skipped (empty draft)",
                "legacy_thematic_issues": "Skipped (empty draft)",
                "legacy_narrative_depth_issues": "Skipped (empty draft)",
            }, None

        plot_point_focus_str = (
            plot_point_focus
            if plot_point_focus is not None
            else "Not available for this chapter."
        )
        if plot_point_focus is None:
            logger.warning(
                f"Plot point focus not available for Ch {chapter_number} during comprehensive evaluation."
            )

        novel_theme_str = novel_props.get("theme", "Not specified")
        novel_genre_str = novel_props.get("genre", "Not specified")
        protagonist_arc_str = novel_props.get("character_arc", "Not specified")
        protagonist_name_str = novel_props.get("protagonist_name", "The Protagonist")

        char_profiles_plain_text = (
            await get_filtered_character_profiles_for_prompt_plain_text(
                novel_props, chapter_number - 1
            )
        )
        world_building_plain_text = await get_filtered_world_data_for_prompt_plain_text(
            novel_props, chapter_number - 1
        )
        kg_check_results_text = await get_reliable_kg_facts_for_drafting_prompt(
            novel_props, chapter_number, None
        )

        plot_points_summary_lines = (
            [
                f"- PP {i + 1}: {pp[:100]}..."
                for i, pp in enumerate(novel_props.get("plot_points", []))
            ]
            if novel_props.get("plot_points")
            else ["  - Not available"]
        )
        plot_points_summary_str = "\n".join(plot_points_summary_lines)

        # This whole block will replace the existing few_shot_eval_example_str
        few_shot_eval_example_str = """
[
  {
    "issue_category": "CONSISTENCY",
    "problem_description": "Character Elara states she has never left her village, but her profile mentions she trained at the Royal Academy in the Capital.",
    "quote_from_original_text": "I've never seen anything beyond these village walls,\\" Elara sighed, gazing at the distant mountains.",
    "suggested_fix_focus": "Adjust Elara's dialogue to align with her established backstory of training in the Capital, or reconcile this statement with her past (e.g., she's being metaphorical or hiding her past)."
  },
  {
    "issue_category": "PLOT_ARC",
    "problem_description": "The chapter focuses heavily on a minor side character's backstory, which doesn't significantly advance the intended plot point about finding the Sunstone.",
    "quote_from_original_text": "The old merchant then spent a long while recounting his youthful adventures in the spice trade, detailing three different voyages.",
    "suggested_fix_focus": "Reduce the side character's backstory significantly or tie it directly into how it helps or hinders the search for the Sunstone. Ensure the main plot point progression is central."
  },
  {
    "issue_category": "NARRATIVE_DEPTH_AND_LENGTH",
    "problem_description": "The confrontation with the antagonist feels rushed and lacks emotional impact. The protagonist's internal reaction to the antagonist's reveal is minimal.",
    "quote_from_original_text": "\\"It was you all along!\\" John exclaimed. The Baron merely smiled. Then they fought.",
    "suggested_fix_focus": "Expand on John's internal thoughts and feelings upon discovering the Baron's betrayal. Show, don't just tell, the emotional weight of this moment. Describe the fight with more detail and tension."
  }
]
"""
        # Note: The user/developer needs to update the actual LLM prompt to request JSON.
        # This tool only changes the agent's parsing logic and the example string.
        prompt_lines = []
        if config.ENABLE_LLM_NO_THINK_DIRECTIVE:
            prompt_lines.append("/no_think")

        prompt_lines.extend(
            [
                f'You are a Master Editor evaluating Chapter {chapter_number} of a novel titled "{novel_props.get("title", "Untitled Novel")}" (Protagonist: {protagonist_name_str}).',
                "Analyze the **Complete Chapter Text** provided below.",
                "Your task is to identify specific issues related to these EXACT categories:",
                "1.  **CONSISTENCY**",
                "2.  **PLOT_ARC**",
                "3.  **THEMATIC_ALIGNMENT**",
                f"4.  **NARRATIVE_DEPTH_AND_LENGTH** (target: at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters)",
                "",
                "**Reference Information for CONSISTENCY Check (Summary Format):**",
                "  **Plot Outline Summary:**",
                "  ```text",
                f"  Title: {novel_props.get('title', 'N/A')}",
                f"  Genre: {novel_props.get('genre', 'N/A')}",
                f"  Theme: {novel_props.get('theme', 'N/A')}",
                f"  Protagonist: {novel_props.get('protagonist_name', 'N/A')} ({novel_props.get('character_arc', 'N/A')})",
                f"  Logline: {novel_props.get('logline', 'N/A')}",
                "  Key Plot Points (summary):",
                plot_points_summary_str,
                "  ```",
                "  **Character Profiles (Key Info - check 'prompt_notes' for provisional status):**",
                "  ```text",
                char_profiles_plain_text,
                "  ```",
                "  **World Building Notes (Key Info - check 'prompt_notes' for provisional status):**",
                "  ```text",
                world_building_plain_text,
                "  ```",
                kg_check_results_text,
                "  **Previous Chapters Context (Semantic Flow & KG Facts for Canon):**",
                "  --- PREVIOUS CONTEXT ---",
                previous_chapters_context
                if previous_chapters_context.strip()
                else "N/A (e.g., Chapter 1 or context retrieval failed).",
                "  --- END PREVIOUS CONTEXT ---",
                "",
                f"**Complete Chapter {chapter_number} Text (to analyze):**",
                "--- BEGIN COMPLETE CHAPTER TEXT ---",
                draft_text,
                "--- END COMPLETE CHAPTER TEXT ---",
                "",
                "**Output Format (CRITICAL - JSON ONLY):**",
                'Provide your evaluation as a JSON array of problem objects. Each object must have these keys: "issue_category", "problem_description", "quote_from_original_text", "suggested_fix_focus".',
                "For `issue_category`, use one of the EXACT category names: CONSISTENCY, PLOT_ARC, THEMATIC_ALIGNMENT, NARRATIVE_DEPTH_AND_LENGTH.",
                'The `quote_from_original_text` must be a VERBATIM quote (10-50 words) from the chapter text. If general or no quote applies, use "N/A - General Issue".',
                'If NO problems are found, output an empty JSON array `[]` or a JSON object like `{"status": "No significant problems found"}`.',
                "",
                "**Follow this example structure for your JSON output precisely:**",
                "```json",
                few_shot_eval_example_str.strip(),
                "```",
                "",
                "Begin your JSON output now:",
            ]
        )
        prompt = "\n".join(prompt_lines)

        logger.info(
            f"Calling LLM ({self.model_name}) for comprehensive evaluation of chapter {chapter_number} (expecting JSON)..."
        )
        cleaned_evaluation_text, usage_data = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=config.TEMPERATURE_EVALUATION,
            allow_fallback=True,
            stream_to_disk=False,
            frequency_penalty=config.FREQUENCY_PENALTY_EVALUATION,
            presence_penalty=config.PRESENCE_PENALTY_EVALUATION,
            auto_clean_response=True,
        )

        no_issues_keywords = [
            "no significant problems found",
            "no issues found",
            "no problems found",
            "no revision needed",
            "no changes needed",
            "all clear",
            "looks good",
            "is fine",
            "is acceptable",
            "passes evaluation",
            "meets criteria",
            "therefore, no revision is needed",
        ]
        is_likely_no_issues_text = False
        if cleaned_evaluation_text.strip():
            normalized_eval_text = (
                cleaned_evaluation_text.lower().strip().replace(".", "")
            )
            for keyword in no_issues_keywords:
                normalized_keyword = keyword.lower().strip().replace(".", "")
                if normalized_keyword == normalized_eval_text or (
                    len(normalized_eval_text) < len(normalized_keyword) + 20
                    and normalized_keyword in normalized_eval_text
                ):
                    is_likely_no_issues_text = True
                    break
        eval_output_dict: Dict[str, Any]
        if is_likely_no_issues_text:
            logger.info(
                f"Heuristic: Evaluation for Ch {chapter_number} appears to indicate 'no issues': '{cleaned_evaluation_text[:100]}...'"
            )
            eval_output_dict = {
                "problems_found_text_output": cleaned_evaluation_text,
                "legacy_consistency_issues": None,
                "legacy_plot_arc_deviation": None,
                "legacy_thematic_issues": None,
                "legacy_narrative_depth_issues": None,
            }
        elif not cleaned_evaluation_text.strip():
            logger.error(
                f"Comprehensive evaluation LLM for Ch {chapter_number} returned empty text."
            )
            eval_output_dict = {
                "problems_found_text_output": "Evaluation LLM call failed or returned empty.",
                "legacy_consistency_issues": "LLM call failed.",
                "legacy_plot_arc_deviation": "LLM call failed.",
                "legacy_thematic_issues": "LLM call failed.",
                "legacy_narrative_depth_issues": "LLM call failed.",
            }
        else:
            legacy_consistency = (
                "Potential consistency issues."
                if "consistency" in cleaned_evaluation_text.lower()
                else None
            )
            legacy_plot = (
                "Potential plot arc issues."
                if "plot_arc" in cleaned_evaluation_text.lower()
                else None
            )
            legacy_theme = (
                "Potential thematic issues."
                if "thematic" in cleaned_evaluation_text.lower()
                else None
            )
            legacy_depth = (
                "Potential narrative depth/length issues."
                if "narrative_depth" in cleaned_evaluation_text.lower()
                else None
            )
            logger.info(
                f"Comprehensive evaluation for Ch {chapter_number} complete. LLM output (first 200 chars): '{cleaned_evaluation_text[:200]}...'"
            )
            eval_output_dict = {
                "problems_found_text_output": cleaned_evaluation_text,
                "legacy_consistency_issues": legacy_consistency,
                "legacy_plot_arc_deviation": legacy_plot,
                "legacy_thematic_issues": legacy_theme,
                "legacy_narrative_depth_issues": legacy_depth,
            }
        return eval_output_dict, usage_data

    async def evaluate_chapter_draft(
        self,
        novel_props: Dict[str, Any],
        draft_text: str,
        chapter_number: int,
        plot_point_focus: Optional[str],
        plot_point_index: int,
        previous_chapters_context: str,
    ) -> Tuple[EvaluationResult, Optional[Dict[str, int]]]:
        logger.info(
            f"ComprehensiveEvaluatorAgent evaluating chapter {chapter_number} draft (length: {len(draft_text)} chars)..."
        )
        reasons_for_revision_summary: list[str] = []
        problem_details_list: List[ProblemDetail] = []
        needs_revision = False
        coherence_score: Optional[float] = None
        total_usage_data: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        if not draft_text:
            needs_revision = True
            problem_details_list.append(
                {
                    "issue_category": "meta",
                    "problem_description": "Draft is empty.",
                    "quote_from_original_text": "N/A - General Issue",
                    "quote_char_start": None,
                    "quote_char_end": None,
                    "sentence_char_start": None,
                    "sentence_char_end": None,
                    "suggested_fix_focus": "Generate content for the chapter.",
                }
            )
            reasons_for_revision_summary.append("Draft is empty.")
        elif len(draft_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
            needs_revision = True
            problem_details_list.append(
                {
                    "issue_category": "narrative_depth_and_length",
                    "problem_description": f"Draft is too short ({len(draft_text)} chars). Minimum required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.",
                    "quote_from_original_text": "N/A - General Issue",
                    "quote_char_start": None,
                    "quote_char_end": None,
                    "sentence_char_start": None,
                    "sentence_char_end": None,
                    "suggested_fix_focus": f"Expand content significantly across multiple scenes/sections to meet the {config.MIN_ACCEPTABLE_DRAFT_LENGTH} character target. Focus on adding descriptive detail, character introspection, and dialogue.",
                }
            )
            reasons_for_revision_summary.append(
                f"Draft is too short ({len(draft_text)} chars). Minimum required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}."
            )

        current_embedding_task = llm_service.async_get_embedding(draft_text)  # MODIFIED
        if chapter_number > 1:
            prev_embedding = await chapter_queries.get_embedding_from_db(
                chapter_number - 1
            )
            current_embedding = await current_embedding_task
            if current_embedding is not None and prev_embedding is not None:
                coherence_score = utils.numpy_cosine_similarity(
                    current_embedding, prev_embedding
                )
                logger.info(
                    f"Coherence score with previous chapter ({chapter_number - 1}): {coherence_score:.4f}"
                )
                if coherence_score < config.REVISION_COHERENCE_THRESHOLD:
                    needs_revision = True
                    problem_details_list.append(
                        {
                            "issue_category": "consistency",
                            "problem_description": f"Low coherence with previous chapter (Score: {coherence_score:.4f}, Threshold: {config.REVISION_COHERENCE_THRESHOLD}). The narrative flow or tone may be disjointed.",
                            "quote_from_original_text": "N/A - General Issue",
                            "quote_char_start": None,
                            "quote_char_end": None,
                            "sentence_char_start": None,
                            "sentence_char_end": None,
                            "suggested_fix_focus": "Review the transition from the previous chapter. Ensure stylistic, tonal, and narrative continuity. This might involve adjusting opening scenes or overall pacing.",
                        }
                    )
                    reasons_for_revision_summary.append(
                        f"Low coherence with previous chapter (Score: {coherence_score:.4f}, Threshold: {config.REVISION_COHERENCE_THRESHOLD})."
                    )
            else:
                logger.warning(
                    f"Could not perform coherence check for ch {chapter_number} (missing current or previous embedding)."
                )
        else:
            logger.info("Skipping coherence check for Chapter 1.")
            await current_embedding_task

        (
            llm_eval_output_dict,
            llm_usage,
        ) = await self._perform_llm_comprehensive_evaluation(
            novel_props,
            draft_text,
            chapter_number,
            plot_point_focus,
            plot_point_index,
            previous_chapters_context,
        )
        if llm_usage:
            total_usage_data["prompt_tokens"] += llm_usage.get("prompt_tokens", 0)
            total_usage_data["completion_tokens"] += llm_usage.get(
                "completion_tokens", 0
            )
            total_usage_data["total_tokens"] += llm_usage.get("total_tokens", 0)

        llm_eval_text_output = llm_eval_output_dict.get(
            "problems_found_text_output", ""
        )
        parsed_problems_from_llm = await self._parse_llm_evaluation_output(
            llm_eval_text_output, chapter_number, draft_text
        )

        if parsed_problems_from_llm:
            problem_details_list.extend(parsed_problems_from_llm)
            needs_revision = True
            category_map_to_reason = {
                "consistency": "Consistency issues identified by LLM.",
                "plot_arc": "Plot Arc deviation identified by LLM.",
                "thematic_alignment": "Thematic Alignment issues identified by LLM.",
                "narrative_depth_and_length": "Narrative Depth/Length issues identified by LLM.",
                "meta": "Meta/Uncategorized issues identified by LLM.",
            }
            for prob in parsed_problems_from_llm:
                reason = category_map_to_reason.get(prob["issue_category"])
                if reason and reason not in reasons_for_revision_summary:
                    reasons_for_revision_summary.append(reason)
        elif (
            llm_eval_text_output.strip()
            and "no significant problems found" not in llm_eval_text_output.lower()
        ):
            logger.warning(
                f"LLM evaluation for Ch {chapter_number} provided text, but no problems were parsed. Text: '{llm_eval_text_output[:200]}...'"
            )
            problem_details_list.append(
                {
                    "issue_category": "meta",
                    "problem_description": "LLM evaluation output was non-empty but could not be parsed into specific problems.",
                    "quote_from_original_text": "N/A - LLM Output Parsing",
                    "quote_char_start": None,
                    "quote_char_end": None,
                    "sentence_char_start": None,
                    "sentence_char_end": None,
                    "suggested_fix_focus": "Review LLM evaluation output and parsing logic. The output might not conform to the expected problem format.",
                }
            )
            if "LLM evaluation output unparsable." not in reasons_for_revision_summary:
                reasons_for_revision_summary.append("LLM evaluation output unparsable.")
            needs_revision = True

        unique_reasons_summary = sorted(list(set(reasons_for_revision_summary)))
        validated_problem_details_list: List[ProblemDetail] = []
        for prob_item in problem_details_list:
            if (
                prob_item["quote_from_original_text"]
                not in [
                    "N/A - General Issue",
                    "N/A - LLM Output Parsing",
                    "N/A - Malformed LLM Output",
                ]
                and prob_item["quote_char_start"] is None
                and prob_item["quote_from_original_text"].strip()
                and draft_text.strip()
            ):
                logger.warning(
                    f"CompEvaluator: Problem quote TEXT for Ch {chapter_number} ('{prob_item['quote_from_original_text'][:50]}...') present, "
                    f"but its offsets were NOT found by spaCy utils. Problem desc: {prob_item['problem_description']}"
                )
            validated_problem_details_list.append(prob_item)

        logger.info(
            f"Evaluation for Ch {chapter_number} complete. Needs revision: {needs_revision}. Summary of reasons: {'; '.join(unique_reasons_summary) if unique_reasons_summary else 'None'}. Detailed problems found: {len(validated_problem_details_list)}"
        )

        final_eval_result: EvaluationResult = {
            "needs_revision": needs_revision,
            "reasons": unique_reasons_summary,
            "problems_found": validated_problem_details_list,
            "coherence_score": coherence_score,
            "consistency_issues": llm_eval_output_dict.get("legacy_consistency_issues"),
            "plot_deviation_reason": llm_eval_output_dict.get(
                "legacy_plot_arc_deviation"
            ),
            "thematic_issues": llm_eval_output_dict.get("legacy_thematic_issues"),
            "narrative_depth_issues": llm_eval_output_dict.get(
                "legacy_narrative_depth_issues"
            ),
        }
        return final_eval_result, total_usage_data if total_usage_data[
            "total_tokens"
        ] > 0 else None
