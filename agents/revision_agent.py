# agents/revision_agent.py
from typing import Any

import structlog

import utils
from config import NARRATIVE_MODEL, REVISION_EVALUATION_THRESHOLD
from core.llm_interface_refactored import llm_service
from data_access import chapter_queries, world_queries

# Import native versions for performance optimization
from data_access.character_queries import get_character_profiles
from models import ProblemDetail
from processing.problem_parser import parse_problem_list
from prompts.prompt_data_getters import (
    get_filtered_character_profiles_for_prompt_plain_text,
    get_filtered_world_data_for_prompt_plain_text,
    get_reliable_kg_facts_for_drafting_prompt,
)
from prompts.prompt_renderer import render_prompt, get_system_prompt

logger = structlog.get_logger()


class RevisionAgent:
    def __init__(self, config: dict, model_name: str = NARRATIVE_MODEL):
        self.model_name = model_name
        self.config = config
        self.threshold = REVISION_EVALUATION_THRESHOLD
        logger.info(f"RevisionAgent initialized with model: {self.model_name}")
        utils.load_spacy_model_if_needed()

    async def validate_revision(
        self, chapter_text: str, previous_chapter_text: str, world_state: dict
    ) -> tuple[bool, list[str]]:
        """Main public method that orchestrates all revision validation.

        Args:
            chapter_text: The current chapter text to validate
            previous_chapter_text: The previous chapter text for context
            world_state: Dictionary containing plot_outline, chapter_number, and other context

        Returns:
            Tuple of (is_valid, list_of_issue_descriptions)
        """
        logger.info("Validating revision", threshold=self.threshold)

        # Extract required information from world_state
        plot_outline = world_state.get("plot_outline", {})
        chapter_number = world_state.get("chapter_number", 1)
        previous_chapters_context = world_state.get("previous_chapters_context", "")

        # Step 1: Check continuity (from WorldContinuityAgent)
        continuity_problems = await self._check_continuity(chapter_text, world_state)

        # Step 2: Evaluate quality (from ComprehensiveEvaluatorAgent)
        has_quality_issues, quality_issue_descriptions = await self._evaluate_quality(
            chapter_text, world_state
        )

        # Combine all problems
        all_problems = continuity_problems
        if has_quality_issues:
            # Convert quality issues to ProblemDetail format for consistency
            for desc in quality_issue_descriptions:
                all_problems.append(
                    {
                        "issue_category": "quality",
                        "problem_description": desc,
                        "quote_from_original_text": "N/A - General Issue",
                        "quote_char_start": None,
                        "quote_char_end": None,
                        "sentence_char_start": None,
                        "sentence_char_end": None,
                        "suggested_fix_focus": "Address quality issues identified by evaluator.",
                    }
                )

        # If no problems found, chapter is valid
        if not all_problems:
            logger.info("Revision validation passed - no issues found")
            return True, ["Revision validation passed - no issues found"]

        # Step 3: Validate patch (from PatchValidationAgent)
        # For now, we'll assume the patch validation is handled elsewhere
        # as it requires specific patch instructions

        # Generate issue descriptions for the orchestrator
        issue_descriptions = []
        for problem in all_problems:
            issue_descriptions.append(problem["problem_description"])

        # Determine if revision is needed based on problem count and severity
        needs_revision = len(all_problems) > 0

        logger.info(
            "Revision validation complete",
            needs_revision=needs_revision,
            problem_count=len(all_problems),
        )

        return not needs_revision, issue_descriptions

    async def _check_continuity(
        self, chapter_text: str, world_state: dict
    ) -> list[ProblemDetail]:
        """Internal method for continuity checking (from WorldContinuityAgent)."""
        plot_outline = world_state.get("plot_outline", {})
        chapter_number = world_state.get("chapter_number", 1)
        previous_chapters_context = world_state.get("previous_chapters_context", "")

        if not chapter_text:
            logger.warning(
                f"Continuity check skipped for Ch {chapter_number}: empty chapter text."
            )
            return []

        logger.info(
            f"RevisionAgent performing continuity check for Chapter {chapter_number}..."
        )

        protagonist_name_str = plot_outline.get("protagonist_name", "The Protagonist")
        characters = await get_character_profiles()
        world_item_ids_by_category = (
            await world_queries.get_all_world_item_ids_by_category()
        )
        char_profiles_plain_text = (
            await get_filtered_character_profiles_for_prompt_plain_text(
                [
                    char.name for char in characters
                ],  # Extract names from CharacterProfile objects
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

        prompt = render_prompt(
            "revision_agent/consistency_check.j2",
            {
                "no_think": False,  # Using no_think directive from config
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
                "draft_text": chapter_text,
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
            temperature=0.2,  # CONSISTENCY_CHECK temperature
            allow_fallback=True,
            stream_to_disk=False,
            auto_clean_response=True,
            system_prompt=get_system_prompt("revision_agent"),
        )

        continuity_problems = await self._parse_llm_continuity_output(
            cleaned_consistency_text, chapter_number, chapter_text
        )

        logger.info(
            f"World/Continuity consistency check for Ch {chapter_number} found"
            f" {len(continuity_problems)} problems."
        )
        return continuity_problems

    async def _evaluate_quality(
        self, chapter_text: str, world_state: dict
    ) -> tuple[bool, list[str]]:
        """Internal method for quality evaluation (from ComprehensiveEvaluatorAgent).

        Returns: (has_quality_issues, list_of_quality_issue_descriptions)
        """
        plot_outline = world_state.get("plot_outline", {})
        chapter_number = world_state.get("chapter_number", 1)
        previous_chapters_context = world_state.get("previous_chapters_context", "")

        processed_text = chapter_text
        logger.info(
            f"RevisionAgent evaluating chapter {chapter_number} draft (length: {len(processed_text)} chars)..."
        )

        reasons_for_revision_summary: list[str] = []
        needs_revision = False

        if not chapter_text:
            needs_revision = True
            reasons_for_revision_summary.append("Draft is empty")
            return needs_revision, reasons_for_revision_summary
        elif len(chapter_text) < self.config.MIN_ACCEPTABLE_DRAFT_LENGTH:
            needs_revision = True
            reasons_for_revision_summary.append(
                f"Draft is too short ({len(chapter_text)} chars). Minimum required: {self.config.MIN_ACCEPTABLE_DRAFT_LENGTH}."
            )

        # Check coherence with previous chapter if available
        if chapter_number > 1 and previous_chapters_context:
            try:
                current_embedding_task = llm_service.async_get_embedding(chapter_text)
                prev_embedding = await chapter_queries.get_embedding_from_db(
                    chapter_number - 1
                )
                current_embedding = await current_embedding_task

                if current_embedding is not None and prev_embedding is not None:
                    try:
                        coherence_score = utils.numpy_cosine_similarity(
                            current_embedding, prev_embedding
                        )
                    except ValueError:
                        logger.warning(
                            "Cosine similarity shape mismatch handled: setting to 0.0 for coherence check compatibility."
                        )
                        coherence_score = 0.0
                    logger.info(
                        f"Coherence score with previous chapter ({chapter_number - 1}): {coherence_score:.4f}"
                    )
                    if coherence_score < 0.60:  # REVISION_COHERENCE_THRESHOLD
                        needs_revision = True
                        reasons_for_revision_summary.append(
                            f"Low coherence with previous chapter (Score: {coherence_score:.4f}, Threshold: 0.60)."
                        )
            except Exception as e:
                logger.warning(
                    f"Could not perform coherence check for ch {chapter_number}: {e}"
                )
        else:
            logger.info("Skipping coherence check for Chapter 1.")

        # Perform LLM evaluation
        llm_eval_output_dict, _ = await self._perform_llm_comprehensive_evaluation(
            plot_outline,
            [],  # character_names - will be fetched in the method
            {},  # world_item_ids_by_category - will be fetched in the method
            processed_text,
            chapter_number,
            None,  # plot_point_focus
            0,  # plot_point_index
            previous_chapters_context,
        )

        llm_eval_text_output = llm_eval_output_dict.get(
            "problems_found_text_output", ""
        )

        # Check if LLM indicates issues
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
        ]

        is_likely_no_issues_text = False
        if llm_eval_text_output.strip():
            normalized_eval_text = llm_eval_text_output.lower().strip().replace(".", "")
            for keyword in no_issues_keywords:
                normalized_keyword = keyword.lower().strip().replace(".", "")
                if normalized_keyword == normalized_eval_text or (
                    len(normalized_eval_text) < len(normalized_keyword) + 20
                    and normalized_keyword in normalized_eval_text
                ):
                    is_likely_no_issues_text = True
                    break

        if not is_likely_no_issues_text and llm_eval_text_output.strip():
            # Extract quality issues from LLM output
            quality_categories = [
                "consistency issues",
                "plot arc issues",
                "thematic issues",
                "narrative depth issues",
                "repetition issues",
            ]

            normalized_output = llm_eval_text_output.lower()
            for category in quality_categories:
                if category in normalized_output:
                    reasons_for_revision_summary.append(
                        f"Potential {category} identified by LLM."
                    )

        if not reasons_for_revision_summary and not is_likely_no_issues_text:
            reasons_for_revision_summary.append(
                "LLM evaluation identified potential quality issues."
            )

        logger.info(
            f"Quality evaluation for Ch {chapter_number} complete. Needs revision: {needs_revision}. "
            f"Summary of reasons: {'; '.join(reasons_for_revision_summary) if reasons_for_revision_summary else 'None'}."
        )

        return needs_revision, reasons_for_revision_summary

    async def _validate_patch(
        self, chapter_text: str, problems: list[ProblemDetail]
    ) -> bool:
        """Internal method for patch validation (from PatchValidationAgent).

        Validates if chapter_text adequately addresses all identified problems.
        """
        if not problems:
            return True

        # For now, return True as patch validation requires specific patch instructions
        # In a full implementation, this would use the patch validation prompt
        # and check if the current text addresses the identified problems

        logger.info("Patch validation placeholder - always returning True for now")
        return True

    async def _parse_llm_continuity_output(
        self, json_text: str, chapter_number: int, original_draft_text: str
    ) -> list[ProblemDetail]:
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

    async def _perform_llm_comprehensive_evaluation(
        self,
        plot_outline: dict[str, Any],
        character_names: list[str],
        world_item_ids_by_category: dict[str, list[str]],
        draft_text: str,
        chapter_number: int,
        plot_point_focus: str | None,
        plot_point_index: int,
        previous_chapters_context: str,
    ) -> tuple[dict[str, Any], dict[str, int] | None]:
        """Perform comprehensive evaluation using LLM."""
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

        if plot_point_focus is None:
            logger.warning(
                f"Plot point focus not available for Ch {chapter_number} during comprehensive evaluation."
            )

        protagonist_name_str = plot_outline.get("protagonist_name", "The Protagonist")

        # Fetch character and world data if not provided (empty parameters)
        if not character_names:
            logger.info("Fetching character profiles from database for evaluation...")
            character_profiles_dict = await get_character_profiles()
            character_names = [profile.name for profile in character_profiles_dict]
            logger.info(
                f"Found {len(character_names)} characters for evaluation: {character_names}"
            )

        # character_names is now always a List[str] as required by the function

        if not world_item_ids_by_category:
            logger.info("Fetching world item IDs from database for evaluation...")
            world_item_ids_by_category = (
                await world_queries.get_all_world_item_ids_by_category()
            )

        char_profiles_plain_text = (
            await get_filtered_character_profiles_for_prompt_plain_text(
                character_names,
                chapter_number - 1,
            )
        )
        world_building_plain_text = await get_filtered_world_data_for_prompt_plain_text(
            world_item_ids_by_category,
            chapter_number - 1,
        )
        kg_check_results_text = await get_reliable_kg_facts_for_drafting_prompt(
            plot_outline, chapter_number, None
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

        prompt = render_prompt(
            "revision_agent/evaluate_chapter.j2",
            {
                "no_think": False,  # Using no_think directive from config
                "chapter_number": chapter_number,
                "novel_title": plot_outline.get("title", "Untitled Novel"),
                "protagonist_name_str": protagonist_name_str,
                "min_length": self.config.MIN_ACCEPTABLE_DRAFT_LENGTH,
                "novel_genre": plot_outline.get("genre", "N/A"),
                "novel_theme": plot_outline.get("theme", "N/A"),
                "novel_protagonist": plot_outline.get("protagonist_name", "N/A"),
                "protagonist_arc": plot_outline.get("character_arc", "N/A"),
                "logline": plot_outline.get("logline", "N/A"),
                "plot_points_summary_str": plot_points_summary_str,
                "char_profiles_plain_text": char_profiles_plain_text,
                "world_building_plain_text": world_building_plain_text,
                "kg_check_results_text": kg_check_results_text,
                "previous_chapters_context": previous_chapters_context,
                "draft_text": draft_text,
            },
        )

        logger.info(
            f"Calling LLM ({self.model_name}) for comprehensive evaluation of chapter {chapter_number} (expecting JSON)..."
        )
        cleaned_evaluation_text, usage_data = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=self.config.TEMPERATURE_EVALUATION,  # EVALUATION temperature
            allow_fallback=True,
            stream_to_disk=False,
            frequency_penalty=self.config.FREQUENCY_PENALTY_EVALUATION,
            presence_penalty=self.config.PRESENCE_PENALTY_EVALUATION,
            auto_clean_response=True,
            system_prompt=get_system_prompt("revision_agent"),
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
        eval_output_dict: dict[str, Any]
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
