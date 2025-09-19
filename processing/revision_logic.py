# processing/revision_logic.py
"""
Handles the revision of chapter drafts based on evaluation feedback for the SAGA system.
Supports both full rewrite and targeted patch-based revisions.
Context data for prompts is now formatted as plain text.
"""

import asyncio
import hashlib
from typing import Any

import structlog

import config
import utils  # For numpy_cosine_similarity, find_semantically_closest_segment, AND find_quote_and_sentence_offsets_with_spacy, format_scene_plan_for_prompt
from agents.revision_agent import RevisionAgent
from core.llm_interface_refactored import count_tokens, llm_service
from core.text_processing_service import truncate_text_by_tokens
from models import (
    CharacterProfile,
    EvaluationResult,
    PatchInstruction,
    ProblemDetail,
    SceneDetail,
    WorldItem,
)
from prompts.prompt_renderer import render_prompt, get_system_prompt

logger = structlog.get_logger(__name__)


async def _process_patch_group(
    group_idx: int,
    group_problem: ProblemDetail,
    group_members: list[ProblemDetail],
    plot_outline: dict[str, Any],
    original_text: str,
    chapter_number: int,
    hybrid_context_for_revision: str,
    chapter_plan: list[SceneDetail] | None,
    validator: Any,  # RevisionAgent or _BypassValidator
) -> PatchInstruction | None:
    """
    Process a single group of problems and generate a patch instruction.
    Extracted from _generate_patch_instructions_logic for better modularity.
    """
    context_snippet = await utils._get_context_window_for_patch_llm(
        original_text,
        group_problem,
        config.MAX_CHARS_FOR_PATCH_CONTEXT_WINDOW,
    )

    patch_instr: PatchInstruction | None = None

    for _ in range(config.PATCH_GENERATION_ATTEMPTS):
        patch_instr_tmp, _ = await _generate_single_patch_instruction_llm(
            plot_outline,
            context_snippet,
            group_problem,
            chapter_number,
            hybrid_context_for_revision,
            chapter_plan,
        )
        if not patch_instr_tmp:
            continue
        if not config.AGENT_ENABLE_PATCH_VALIDATION:
            patch_instr = patch_instr_tmp
            break
        valid, _ = await validator.validate_patch(
            context_snippet, patch_instr_tmp, group_members
        )
        if valid:
            patch_instr = patch_instr_tmp
            break

    if not patch_instr:
        logger.warning(
            f"Failed to generate valid patch for group {group_idx} in Ch {chapter_number}."
        )
    return patch_instr


def _validate_revision_inputs(
    original_text: str, chapter_number: int
) -> tuple[bool, tuple[str, str, list[tuple[int, int]]] | None]:
    """
    Validate inputs for chapter revision process.

    Returns:
        tuple: (is_valid, early_return_value_or_None)
    """
    if not original_text:
        logger.error(
            f"Revision for ch {chapter_number} aborted: missing original text."
        )
        return False, (None, None, [])
    return True, None


def _prepare_problems_for_revision(
    evaluation_result: EvaluationResult, chapter_number: int
) -> tuple[list[ProblemDetail], str, bool]:
    """
    Process and prepare problems from evaluation result.

    Returns:
        tuple: (problems_to_fix, revision_reason_str, should_continue)
    """
    problems_to_fix: list[ProblemDetail] = evaluation_result.get("problems_found", [])
    problems_to_fix = _deduplicate_problems(
        _consolidate_overlapping_problems(problems_to_fix)
    )

    if not problems_to_fix and evaluation_result.get("needs_revision"):
        logger.warning(
            f"Revision for ch {chapter_number} explicitly requested, but no specific problems were itemized. This might lead to a full rewrite attempt if general reasons exist."
        )
    elif not problems_to_fix:
        logger.info(
            f"No specific problems found for ch {chapter_number}, and not marked for revision. No revision performed."
        )
        return [], "", False

    revision_reason_str_list = evaluation_result.get("reasons", [])
    revision_reason_str = (
        "\n- ".join(revision_reason_str_list)
        if revision_reason_str_list
        else "General unspecified issues."
    )
    logger.info(
        f"Attempting revision for chapter {chapter_number}. Reason(s):\n- {revision_reason_str}"
    )

    return problems_to_fix, revision_reason_str, True


async def _attempt_patch_based_revision(
    plot_outline: dict[str, Any],
    original_text: str,
    problems_to_fix: list[ProblemDetail],
    chapter_number: int,
    hybrid_context_for_revision: str,
    chapter_plan: list[SceneDetail] | None,
    already_patched_spans: list[tuple[int, int]],
) -> tuple[str | None, list[tuple[int, int]]]:
    """
    Attempt patch-based revision of the chapter.

    Returns:
        tuple: (patched_text_or_None, updated_spans)
    """
    if not config.ENABLE_PATCH_BASED_REVISION:
        return None, already_patched_spans

    logger.info(
        f"Attempting patch-based revision for Ch {chapter_number} with {len(problems_to_fix)} problem(s)."
    )

    sentence_embeddings = await _get_sentence_embeddings(original_text)
    if config.AGENT_ENABLE_PATCH_VALIDATION:
        validator: RevisionAgent | Any = RevisionAgent(config)
    else:

        class _BypassValidator:
            async def validate_patch(
                self, *_args: Any, **_kwargs: Any
            ) -> tuple[bool, None]:
                return True, None

        validator = _BypassValidator()

    patch_instructions = await _generate_patch_instructions_logic(
        plot_outline,
        original_text,
        problems_to_fix,
        chapter_number,
        hybrid_context_for_revision,
        chapter_plan,
        validator,
    )

    if patch_instructions:
        patched_text, updated_spans = await _apply_patches_to_text(
            original_text,
            patch_instructions,
            already_patched_spans,
            sentence_embeddings,
        )
        logger.info(
            f"Patch process for Ch {chapter_number}: Generated {len(patch_instructions)} patch instructions and applied them. "
            f"Original len: {len(original_text)}, Patched text len: {len(patched_text if patched_text else '')}."
        )
        return patched_text, updated_spans
    else:
        logger.warning(
            f"Patch-based revision for Ch {chapter_number}: No valid patch instructions were generated. Will consider full rewrite if needed."
        )
        return None, already_patched_spans


async def _evaluate_patched_text(
    plot_outline: dict[str, Any],
    character_profiles: dict[str, CharacterProfile],
    world_building: dict[str, dict[str, WorldItem]],
    patched_text: str,
    chapter_number: int,
    hybrid_context_for_revision: str,
) -> bool:
    """
    Evaluate whether patched text is good enough to use as final result.

    Returns:
        bool: True if patched text should be used as final result
    """
    if patched_text is None:
        return False

    evaluator = RevisionAgent(config)
    world_ids = {
        cat: [item.id for item in items.values() if isinstance(item, WorldItem)]
        for cat, items in world_building.items()
        if isinstance(items, dict)
    }
    plot_focus, plot_idx = _get_plot_point_info(plot_outline, chapter_number)
    post_eval, _ = await evaluator.evaluate_chapter_draft(
        plot_outline,
        list(character_profiles.keys()),
        world_ids,
        patched_text,
        chapter_number,
        plot_focus,
        plot_idx,
        hybrid_context_for_revision,
    )
    remaining = len(post_eval.get("problems_found", []))
    return remaining <= config.POST_PATCH_PROBLEM_THRESHOLD


async def _perform_full_rewrite(
    plot_outline: dict[str, Any],
    original_text: str,
    problems_to_fix: list[ProblemDetail],
    chapter_number: int,
    hybrid_context_for_revision: str,
    chapter_plan: list[SceneDetail] | None,
    revision_reason_str: str,
    is_from_flawed_source: bool,
) -> tuple[str, str]:
    """
    Perform a full chapter rewrite using LLM.

    Returns:
        tuple: (final_revised_text, raw_llm_output)
    """
    logger.info(
        f"Proceeding with full chapter rewrite for Ch {chapter_number} as patching was ineffective or disabled."
    )

    # Prepare original snippet
    max_original_snippet_tokens = config.MAX_CONTEXT_TOKENS
    original_snippet = truncate_text_by_tokens(
        original_text,
        config.MEDIUM_MODEL,
        max_original_snippet_tokens,
        truncation_marker="\n... (original draft snippet truncated for brevity in rewrite prompt)",
    )

    # Prepare plan focus section
    plan_focus_section_parts: list[str] = []
    plot_point_focus, _ = _get_plot_point_info(plot_outline, chapter_number)
    max_plan_tokens_for_full_rewrite = config.MAX_CONTEXT_TOKENS // 2

    if config.ENABLE_AGENTIC_PLANNING and chapter_plan:
        formatted_plan_fr = _get_formatted_scene_plan_from_agent_or_fallback(
            chapter_plan,
            config.MEDIUM_MODEL,
            max_plan_tokens_for_full_rewrite,
        )
        plan_focus_section_parts.append(formatted_plan_fr)
        if "plan truncated" in formatted_plan_fr:
            logger.warning(
                f"Scene plan token-truncated for Ch {chapter_number} full rewrite prompt."
            )
    else:
        plan_focus_section_parts.append(
            f"**Original Chapter Focus (Target):**\n{plot_point_focus or 'Not specified.'}\n"
        )
    plan_focus_section_str = "".join(plan_focus_section_parts)

    # Prepare length expansion instructions
    length_issue_explicit_instruction_parts: list[str] = []
    needs_expansion_from_problems = any(
        (
            p["issue_category"] == "narrative_depth_and_length"
            and (
                "short" in p["problem_description"].lower()
                or "length" in p["problem_description"].lower()
                or "expand" in p["suggested_fix_focus"].lower()
                or "depth" in p["problem_description"].lower()
            )
        )
        for p in problems_to_fix
    )
    if needs_expansion_from_problems:
        length_issue_explicit_instruction_parts.extend(
            [
                "\n**Specific Focus on Expansion:** A key critique involves insufficient length and/or narrative depth. ",
                "Your revision MUST substantially expand the narrative by incorporating more descriptive details, character thoughts/introspection, dialogue, actions, and sensory information. ",
                f"Aim for a chapter length of at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters.",
            ]
        )
    length_issue_explicit_instruction_str = "".join(
        length_issue_explicit_instruction_parts
    )

    # Prepare problem descriptions
    all_problem_descriptions_parts: list[str] = []
    if problems_to_fix:
        all_problem_descriptions_parts.append(
            "**Detailed Issues to Address (from evaluation):**\n"
        )
        for prob_idx, prob_item in enumerate(problems_to_fix):
            all_problem_descriptions_parts.extend(
                [
                    f"  {prob_idx + 1}. Category: {prob_item['issue_category']}",
                    f"     Description: {prob_item['problem_description']}",
                    f'     Quote Ref: "{prob_item["quote_from_original_text"][:100].replace(chr(10), " ")}..."',
                    f"     Fix Focus: {prob_item['suggested_fix_focus']}\n",
                ]
            )
        all_problem_descriptions_parts.append("---\n")
    all_problem_descriptions_str = "".join(all_problem_descriptions_parts)

    # Prepare deduplication note
    deduplication_note = ""
    if is_from_flawed_source:
        deduplication_note = (
            "\n**(Note: The 'Original Draft Snippet' below may have had repetitive content removed "
            "prior to evaluation, or other flaws were present. Ensure your rewrite is cohesive "
            "and addresses any resulting narrative gaps or inconsistencies.)**\n"
        )

    # Render prompt
    protagonist_name = plot_outline.get(
        "protagonist_name", config.DEFAULT_PROTAGONIST_NAME
    )
    prompt_full_rewrite = render_prompt(
        "revision_agent/full_chapter_rewrite.j2",
        {
            "config": config,
            "chapter_number": chapter_number,
            "protagonist_name": protagonist_name,
            "revision_reason": llm_service.clean_model_response(
                revision_reason_str
            ).strip(),
            "all_problem_descriptions": all_problem_descriptions_str,
            "deduplication_note": deduplication_note,
            "length_issue_explicit_instruction": length_issue_explicit_instruction_str,
            "plan_focus_section": plan_focus_section_str,
            "hybrid_context_for_revision": hybrid_context_for_revision,
            "original_snippet": original_snippet,
            "genre": plot_outline.get("genre", "story"),
            "min_acceptable_draft_length": config.MIN_ACCEPTABLE_DRAFT_LENGTH,
        },
    )

    # Call LLM
    logger.info(
        f"Calling LLM ({config.MEDIUM_MODEL}) for Ch {chapter_number} full rewrite. Min length: {config.MIN_ACCEPTABLE_DRAFT_LENGTH} chars."
    )

    raw_revised_llm_output, _ = await llm_service.async_call_llm(
        model_name=config.MEDIUM_MODEL,
        prompt=prompt_full_rewrite,
        temperature=config.Temperatures.REVISION,
        max_tokens=None,
        allow_fallback=True,
        stream_to_disk=True,
        frequency_penalty=config.FREQUENCY_PENALTY_REVISION,
        presence_penalty=config.PRESENCE_PENALTY_REVISION,
        auto_clean_response=False,
        system_prompt=get_system_prompt("revision_agent"),
    )

    final_revised_text = llm_service.clean_model_response(raw_revised_llm_output)

    logger.info(
        f"Full rewrite for Ch {chapter_number} generated text of length {len(final_revised_text)}."
    )

    return final_revised_text, raw_revised_llm_output


def _validate_final_result(
    final_revised_text: str | None, chapter_number: int
) -> tuple[bool, tuple[str, str, list[tuple[int, int]]] | None]:
    """
    Validate the final revision result.

    Returns:
        tuple: (is_valid, early_return_value_or_None)
    """
    if not final_revised_text:
        logger.error(
            f"Revision process for ch {chapter_number} resulted in no usable content."
        )
        return False, None

    if len(final_revised_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
        logger.warning(
            f"Final revised draft for ch {chapter_number} is short ({len(final_revised_text)} chars). Min target: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}."
        )

    logger.info(
        f"Revision process for ch {chapter_number} produced a candidate text (Length: {len(final_revised_text)} chars)."
    )

    return True, None


def _get_formatted_scene_plan_from_agent_or_fallback(
    chapter_plan: list[SceneDetail],
    model_name_for_tokens: str,
    max_tokens_budget: int,
) -> str:
    """Formats a chapter plan into plain text for LLM prompts using the central utility."""
    return utils.format_scene_plan_for_prompt(
        chapter_plan, model_name_for_tokens, max_tokens_budget
    )


def _get_plot_point_info(
    plot_outline: dict[str, Any], chapter_number: int
) -> tuple[str | None, int]:
    plot_points = plot_outline.get("plot_points", [])
    if not isinstance(plot_points, list) or not plot_points or chapter_number <= 0:
        return None, -1
    plot_point_index = min(chapter_number - 1, len(plot_points) - 1)
    if 0 <= plot_point_index < len(plot_points):
        plot_point = plot_points[plot_point_index]
        return str(plot_point) if plot_point is not None else None, plot_point_index
    return None, -1


_sentence_embedding_cache: dict[str, list[tuple[int, int, Any]]] = {}


async def _get_sentence_embeddings(
    text: str, cache: dict[str, list[tuple[int, int, Any]]] | None = None
) -> list[tuple[int, int, Any]]:
    """Return a list of (start, end, embedding) for each sentence."""
    if cache is None:
        cache = _sentence_embedding_cache
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if text_hash in cache:
        return cache[text_hash]

    # Lazy-load spaCy only if needed by downstream text segmentation utilities
    try:
        utils.load_spacy_model_if_needed()
    except Exception:
        # Do not fail hard; allow utils.get_text_segments to use any fallback
        pass

    segments = utils.get_text_segments(text, "sentence")
    if not segments:
        return []
    tasks = [llm_service.async_get_embedding(seg[0]) for seg in segments]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    embeddings: list[tuple[int, int, Any]] = []
    for (_, start, end), res in zip(segments, results, strict=False):
        if isinstance(res, Exception) or res is None:
            continue
        embeddings.append((start, end, res))
    cache[text_hash] = embeddings
    return embeddings


async def _find_sentence_via_embeddings(
    quote_text: str, embeddings: list[tuple[int, int, Any]]
) -> tuple[int, int] | None:
    if not embeddings or not quote_text.strip():
        return None
    q_emb = await llm_service.async_get_embedding(quote_text)
    if q_emb is None:
        return None
    best_sim = -1.0
    best_span: tuple[int, int] | None = None
    for start, end, emb in embeddings:
        sim = utils.numpy_cosine_similarity(q_emb, emb)
        if sim > best_sim:
            best_sim = sim
            best_span = (start, end)
    return best_span


async def _generate_single_patch_instruction_llm(
    plot_outline: dict[str, Any],
    original_chapter_text_snippet_for_llm: str,
    problem: ProblemDetail,
    chapter_number: int,
    hybrid_context_for_revision: str,
    chapter_plan: list[SceneDetail] | None,
) -> tuple[PatchInstruction | None, dict[str, int] | None]:
    """
    Generates a single patch instruction. The PatchInstruction will store target_char_start/end
    referring to the SENTENCE containing the problem quote if available.
    """
    plan_focus_section_parts: list[str] = []
    plot_point_focus, _ = _get_plot_point_info(plot_outline, chapter_number)
    max_plan_tokens_for_patch_prompt = config.MAX_CONTEXT_TOKENS // 2

    if config.ENABLE_AGENTIC_PLANNING and chapter_plan:
        formatted_plan = _get_formatted_scene_plan_from_agent_or_fallback(
            chapter_plan,
            config.MEDIUM_MODEL,
            max_plan_tokens_for_patch_prompt,
        )
        plan_focus_section_parts.append(formatted_plan)
        if "plan truncated" in formatted_plan:
            logger.warning(
                f"Scene plan token-truncated for Ch {chapter_number} patch generation prompt."
            )
    else:
        plan_focus_section_parts.append(
            f"**Original Chapter Focus (Reference for overall chapter direction):**\n{plot_point_focus or 'Not specified.'}\n"
        )
    plan_focus_section_str = "".join(plan_focus_section_parts)

    is_general_expansion_task = False
    length_expansion_instruction_header_parts: list[str] = []
    original_quote_text_from_problem = problem["quote_from_original_text"]

    if problem["issue_category"] == "narrative_depth_and_length" and (
        "short" in problem["problem_description"].lower()
        or "length" in problem["problem_description"].lower()
        or "expand" in problem["suggested_fix_focus"].lower()
        or "depth" in problem["problem_description"].lower()
        or original_quote_text_from_problem == "N/A - General Issue"
    ):
        length_expansion_instruction_header_parts.append(
            "\n**Critical: SUBSTANTIAL EXPANSION REQUIRED FOR THIS SEGMENT/PASSAGE.** "
        )
        length_expansion_instruction_header_parts.append(
            "The 'replace_with' text MUST be significantly longer and more detailed. "
        )
        length_expansion_instruction_header_parts.append(
            "Add descriptive details, character thoughts, dialogue, actions, and sensory information. "
        )
        if original_quote_text_from_problem == "N/A - General Issue":
            is_general_expansion_task = True
            length_expansion_instruction_header_parts.append(
                "Since the original quote is 'N/A - General Issue', your 'replace_with' text should be a **new, expanded passage** "
                "that addresses the 'Problem Description' and 'Suggested Fix Focus' within the broader 'Text Snippet' context. "
                "This generated text is intended as a candidate for insertion or to inform a broader rewrite of a section."
            )
        else:
            length_expansion_instruction_header_parts.append(
                "Aim for a notable increase in length and detail for the conceptual segment related to the original quote."
            )
    length_expansion_instruction_header_str = "".join(
        length_expansion_instruction_header_parts
    )

    prompt_instruction_for_replacement_scope_parts: list[str] = []
    max_patch_output_tokens = 0

    if is_general_expansion_task:
        prompt_instruction_for_replacement_scope_parts.append(
            "    - The 'Original Quote Illustrating Problem' is \"N/A - General Issue\". Therefore, your `replace_with` text should be a **new, self-contained, and substantially expanded passage** "
            'that addresses the "Problem Description" and "Suggested Fix Focus" as guided by the `length_expansion_instruction_header_str`. '
            "This new passage is intended for potential insertion into the chapter, not to replace a specific quote."
        )
        max_patch_output_tokens = config.MAX_GENERATION_TOKENS // 2
        max_patch_output_tokens = max(max_patch_output_tokens, 750)
        logger.info(
            f"Patch (Ch {chapter_number}, general expansion): Max output tokens set to {max_patch_output_tokens}."
        )
    else:
        prompt_instruction_for_replacement_scope_parts.append(
            "    - The 'Original Quote Illustrating Problem' is specific. Your `replace_with` text should be a revised version "
            "of the **entire conceptual sentence or short paragraph** within the 'ORIGINAL TEXT SNIPPET' that best corresponds to that quote. Your output will replace that whole segment.\n"
            "    - **Crucially, for this specific fix, your replacement text should primarily focus on correcting the identified issue. "
        )
        if length_expansion_instruction_header_str:
            prompt_instruction_for_replacement_scope_parts.append(
                "If `length_expansion_instruction_header_str` is present, apply its guidance to *this specific segment*. "
            )
        prompt_instruction_for_replacement_scope_parts.append(
            "Otherwise, aim for a length comparable to the original segment, plus necessary additions for the fix. "
            "Avoid excessive unrelated expansion beyond the scope of the problem for this segment.**"
        )
        original_snippet_tokens = count_tokens(
            original_chapter_text_snippet_for_llm,
            config.MEDIUM_MODEL,
        )
        expansion_factor = 2.5 if length_expansion_instruction_header_str else 1.5
        max_patch_output_tokens = int(original_snippet_tokens * expansion_factor)
        max_patch_output_tokens = min(
            max_patch_output_tokens, config.MAX_GENERATION_TOKENS // 2
        )
        max_patch_output_tokens = max(max_patch_output_tokens, 200)
        logger.info(
            f"Patch (Ch {chapter_number}, specific fix): Original snippet tokens: {original_snippet_tokens}. Max output tokens set to {max_patch_output_tokens}."
        )
    prompt_instruction_for_replacement_scope_str = "".join(
        prompt_instruction_for_replacement_scope_parts
    )

    protagonist_name = plot_outline.get(
        "protagonist_name", config.DEFAULT_PROTAGONIST_NAME
    )

    prompt = render_prompt(
        "revision_agent/patch_generation.j2",
        {
            "config": config,
            "chapter_number": chapter_number,
            "novel_title": plot_outline.get("title", "Untitled Novel"),
            "protagonist_name": protagonist_name,
            "genre": plot_outline.get("genre", "N/A"),
            "theme": plot_outline.get("theme", "N/A"),
            "character_arc": plot_outline.get("character_arc", "N/A"),
            "plan_focus_section": plan_focus_section_str,
            "hybrid_context_for_revision": hybrid_context_for_revision,
            "problem": problem,
            "original_quote_text_from_problem": original_quote_text_from_problem,
            "original_chapter_text_snippet_for_llm": original_chapter_text_snippet_for_llm,
            "length_expansion_instruction_header_str": length_expansion_instruction_header_str,
            "prompt_instruction_for_replacement_scope_str": prompt_instruction_for_replacement_scope_str,
        },
    )

    logger.info(
        f"Calling LLM ({config.MEDIUM_MODEL}) for patch in Ch {chapter_number}. Problem: '{problem['problem_description'][:60].replace(chr(10), ' ')}...' Quote Text: '{original_quote_text_from_problem[:50].replace(chr(10), ' ')}...' Max Output Tokens: {max_patch_output_tokens}"
    )

    (
        replace_with_text_cleaned,
        usage_data,
    ) = await llm_service.async_call_llm(
        model_name=config.MEDIUM_MODEL,
        prompt=prompt,
        temperature=config.Temperatures.PATCH,
        max_tokens=max_patch_output_tokens,
        allow_fallback=True,
        stream_to_disk=False,
        frequency_penalty=config.FREQUENCY_PENALTY_PATCH,
        presence_penalty=config.PRESENCE_PENALTY_PATCH,
        auto_clean_response=True,
        system_prompt=get_system_prompt("revision_agent"),
    )

    # MODIFICATION: No longer check if the cleaned text is empty here, as an empty string is now a valid "deletion" instruction.
    # The check for a failed LLM call (returning None) is implicitly handled by the structure below.
    if replace_with_text_cleaned is None:
        logger.error(
            f"Patch LLM call failed and returned None for Ch {chapter_number} problem: {problem['problem_description']}"
        )
        return None, usage_data

    # Log if a deletion is being suggested
    if not replace_with_text_cleaned.strip():
        logger.info(
            f"Patch LLM suggested DELETION (empty output) for Ch {chapter_number} problem: {problem['problem_description']}"
        )

    if length_expansion_instruction_header_str:
        if not is_general_expansion_task:
            if (
                len(original_chapter_text_snippet_for_llm) > 100
                and len(replace_with_text_cleaned)
                < len(original_chapter_text_snippet_for_llm) * 1.2
            ):
                logger.warning(
                    f"Patch for Ch {chapter_number} (specific quote, segment expansion requested) output length ({len(replace_with_text_cleaned)}) "
                    f"is not significantly larger than context snippet ({len(original_chapter_text_snippet_for_llm)}). "
                    f"Problem: {problem['problem_description'][:60]}"
                )
        elif is_general_expansion_task and len(replace_with_text_cleaned) < 500:
            logger.warning(
                f"Patch for Ch {chapter_number} ('N/A - General Issue' expansion) produced a relatively short new passage (len: {len(replace_with_text_cleaned)}). "
                f"Problem: {problem['problem_description'][:60]}"
            )

    target_start_for_patch: int | None = problem.get("sentence_char_start")
    target_end_for_patch: int | None = problem.get("sentence_char_end")

    if (
        original_quote_text_from_problem != "N/A - General Issue"
        and (target_start_for_patch is None or target_end_for_patch is None)
        and (
            problem.get("quote_char_start") is not None
            and problem.get("quote_char_end") is not None
        )
    ):
        logger.warning(
            f"Patch for Ch {chapter_number}: Problem '{original_quote_text_from_problem[:50]}' had specific text but no sentence offsets. "
            f"PatchInstruction will use quote offsets ({problem.get('quote_char_start')}-{problem.get('quote_char_end')}). Application will use semantic search."
        )
        target_start_for_patch = problem.get("quote_char_start")
        target_end_for_patch = problem.get("quote_char_end")
    elif original_quote_text_from_problem != "N/A - General Issue" and (
        target_start_for_patch is None or target_end_for_patch is None
    ):
        logger.error(
            f"Patch for Ch {chapter_number}: Problem '{original_quote_text_from_problem[:50]}' specific text but NO OFFSETS (sentence or quote). Patch will likely fail to apply precisely."
        )

    patch_instruction: PatchInstruction = {
        "original_problem_quote_text": original_quote_text_from_problem,
        "target_char_start": target_start_for_patch,
        "target_char_end": target_end_for_patch,
        "replace_with": replace_with_text_cleaned,  # This can now be ""
        "reason_for_change": f"Fixing '{problem['issue_category']}': {problem['problem_description']}",
    }
    return patch_instruction, usage_data


def _consolidate_overlapping_problems(
    problems: list[ProblemDetail],
) -> list[ProblemDetail]:
    """
    Groups problems by their overlapping text spans and consolidates them.
    This prevents generating multiple patches for the same or overlapping sentences.
    """
    if not problems:
        return []

    # Separate problems that have a specific sentence span from those that are general.
    span_problems = [
        p
        for p in problems
        if p.get("sentence_char_start") is not None
        and p.get("sentence_char_end") is not None
    ]
    general_problems = [
        p
        for p in problems
        if p.get("sentence_char_start") is None or p.get("sentence_char_end") is None
    ]

    if not span_problems:
        return general_problems

    # Sort problems by their start offset to enable linear merging
    span_problems.sort(key=lambda p: p["sentence_char_start"])  # type: ignore

    merged_groups: list[list[ProblemDetail]] = []
    if span_problems:
        current_group = [span_problems[0]]
        current_group_end = span_problems[0]["sentence_char_end"]

        for i in range(1, len(span_problems)):
            next_problem = span_problems[i]
            next_start = next_problem["sentence_char_start"]
            next_end = next_problem["sentence_char_end"]

            # If the next problem starts before the current group's span ends, it overlaps.
            if next_start < current_group_end:  # type: ignore
                current_group.append(next_problem)
                current_group_end = max(current_group_end, next_end)  # type: ignore
            else:
                # The next problem does not overlap, so finalize the current group and start a new one.
                merged_groups.append(current_group)
                current_group = [next_problem]
                current_group_end = next_end

        merged_groups.append(current_group)  # Add the last group

    consolidated_problems: list[ProblemDetail] = []
    for group in merged_groups:
        if len(group) == 1:
            consolidated_problems.append(group[0])
            continue

        # Consolidate the group into a single new ProblemDetail
        first_problem = group[0]
        # Calculate the union of all spans in the group
        group_start_offset = min(p["sentence_char_start"] for p in group)  # type: ignore
        group_end_offset = max(p["sentence_char_end"] for p in group)  # type: ignore

        # Combine all details from the problems in the group
        all_categories = sorted(list(set(p["issue_category"] for p in group)))
        all_descriptions = "; ".join(
            f"({p['issue_category']}) {p['problem_description']}" for p in group
        )
        all_fix_foci = "; ".join(
            f"({p['issue_category']}) {p['suggested_fix_focus']}" for p in group
        )
        # Use the quote from the first problem in the span as a representative
        representative_quote = first_problem["quote_from_original_text"]

        consolidated_problem: ProblemDetail = {
            "issue_category": ", ".join(all_categories),
            "problem_description": f"Multiple issues in one segment: {all_descriptions}",
            "quote_from_original_text": representative_quote,
            "quote_char_start": first_problem.get("quote_char_start"),
            "quote_char_end": first_problem.get("quote_char_end"),
            "sentence_char_start": group_start_offset,
            "sentence_char_end": group_end_offset,
            "suggested_fix_focus": f"Holistically revise the segment to address all points: {all_fix_foci}",
        }
        consolidated_problems.append(consolidated_problem)
        logger.info(
            f"Consolidated {len(group)} overlapping problems into one targeting span {group_start_offset}-{group_end_offset}."
        )

    consolidated_problems.extend(general_problems)
    return consolidated_problems


def _deduplicate_problems(problems: list[ProblemDetail]) -> list[ProblemDetail]:
    """Remove exact duplicates based on span and quote text."""
    unique: list[ProblemDetail] = []
    seen: set[tuple[int | None, int | None, str]] = set()
    for prob in problems:
        key = (
            prob.get("sentence_char_start"),
            prob.get("sentence_char_end"),
            prob.get("quote_from_original_text", ""),
        )
        if key in seen:
            logger.info(
                "Deduplicating problem at span %s-%s with quote '%s'.",
                prob.get("sentence_char_start"),
                prob.get("sentence_char_end"),
                prob.get("quote_from_original_text", "")[:30],
            )
            continue
        seen.add(key)
        unique.append(prob)
    return unique


def _group_problems_for_patch_generation(
    problems: list[ProblemDetail],
) -> list[tuple[ProblemDetail, list[ProblemDetail]]]:
    """Return consolidated problem with list of original problems."""
    if not problems:
        return []

    span_problems = [
        p
        for p in problems
        if p.get("sentence_char_start") is not None
        and p.get("sentence_char_end") is not None
    ]
    general_problems = [
        p
        for p in problems
        if p.get("sentence_char_start") is None or p.get("sentence_char_end") is None
    ]

    span_problems.sort(key=lambda p: p["sentence_char_start"])
    merged_groups: list[list[ProblemDetail]] = []
    if span_problems:
        current_group = [span_problems[0]]
        current_end = span_problems[0]["sentence_char_end"]
        for prob in span_problems[1:]:
            start = prob["sentence_char_start"]
            end = prob["sentence_char_end"]
            if start < current_end:
                current_group.append(prob)
                current_end = max(current_end, end)
            else:
                merged_groups.append(current_group)
                current_group = [prob]
                current_end = end
        merged_groups.append(current_group)

    result: list[tuple[ProblemDetail, list[ProblemDetail]]] = []

    for group in merged_groups:
        first = group[0]
        group_start = min(p["sentence_char_start"] for p in group)  # type: ignore
        group_end = max(p["sentence_char_end"] for p in group)  # type: ignore
        all_cats = sorted(list(set(p["issue_category"] for p in group)))
        all_desc = "; ".join(
            f"({p['issue_category']}) {p['problem_description']}" for p in group
        )
        all_fix = "; ".join(
            f"({p['issue_category']}) {p['suggested_fix_focus']}" for p in group
        )
        rep_quote = first["quote_from_original_text"]
        consolidated: ProblemDetail = {
            "issue_category": ", ".join(all_cats),
            "problem_description": f"Multiple issues in one segment: {all_desc}",
            "quote_from_original_text": rep_quote,
            "quote_char_start": first.get("quote_char_start"),
            "quote_char_end": first.get("quote_char_end"),
            "sentence_char_start": group_start,
            "sentence_char_end": group_end,
            "suggested_fix_focus": f"Holistically revise the segment to address all points: {all_fix}",
        }
        result.append((consolidated, group))

    for p in general_problems:
        result.append((p, [p]))

    return result


async def _generate_patch_instructions_logic(
    plot_outline: dict[str, Any],
    original_text: str,
    problems_to_fix: list[ProblemDetail],
    chapter_number: int,
    hybrid_context_for_revision: str,
    chapter_plan: list[SceneDetail] | None,
    validator: RevisionAgent,
) -> list[PatchInstruction]:
    patch_instructions: list[PatchInstruction] = []

    grouped = _group_problems_for_patch_generation(problems_to_fix)

    groups_to_process = grouped[: config.MAX_PATCH_INSTRUCTIONS_TO_GENERATE]
    if len(grouped) > len(groups_to_process):
        logger.warning(
            f"Found {len(grouped)} patch groups for Ch {chapter_number}. "
            f"Processing only the first {len(groups_to_process)} groups."
        )
    if not groups_to_process:
        return []

    tasks = [
        _process_patch_group(
            idx,
            gp,
            gm,
            plot_outline,
            original_text,
            chapter_number,
            hybrid_context_for_revision,
            chapter_plan,
            validator,
        )
        for idx, (gp, gm) in enumerate(groups_to_process, start=1)
    ]

    results = await asyncio.gather(*tasks)
    for patch_instr in results:
        if patch_instr:
            patch_instructions.append(patch_instr)

    logger.info(
        f"Generated {len(patch_instructions)} patch instructions for Ch {chapter_number}."
    )
    return patch_instructions


async def _apply_patches_to_text(
    original_text: str,
    patch_instructions: list[PatchInstruction],
    already_patched_spans: list[tuple[int, int]] | None = None,
    sentence_embeddings: list[tuple[int, int, Any]] | None = None,
) -> tuple[str, list[tuple[int, int]]]:
    """
    Applies patch instructions to the original text and returns the new text and a
    comprehensive, re-mapped list of all patched spans (old and new).
    """
    if already_patched_spans is None:
        already_patched_spans = []

    if not patch_instructions:
        return original_text, already_patched_spans

    # 1. Prepare new replacements, filtering out overlaps with existing patched spans.
    replacements: list[tuple[int, int, str]] = []
    for patch_idx, patch in enumerate(patch_instructions):
        # MODIFICATION START: Handle empty replace_with as a valid deletion instruction.
        # An empty or whitespace-only replace_with string is now a valid patch.
        replacement_text = patch.get("replace_with", "")
        if replacement_text is None:  # handle case where key is missing
            replacement_text = ""
        # MODIFICATION END

        segment_start: int | None = patch.get("target_char_start")
        segment_end: int | None = patch.get("target_char_end")
        method_used = "direct offsets"

        if segment_start is None or segment_end is None:
            quote_text = patch["original_problem_quote_text"]
            if quote_text != "N/A - General Issue" and quote_text.strip():
                logger.info(
                    f"Patch {patch_idx + 1}: Missing direct offsets for '{quote_text[:50]}...'. Using semantic search."
                )
                method_used = "semantic search"
                if sentence_embeddings:
                    found = await _find_sentence_via_embeddings(
                        quote_text, sentence_embeddings
                    )
                    if found:
                        segment_start, segment_end = found
                if segment_start is None or segment_end is None:
                    match = await utils.find_semantically_closest_segment(
                        original_text, quote_text, "sentence"
                    )
                    if match:
                        segment_start, segment_end, _ = match
            else:
                logger.warning(
                    f"Patch {patch_idx + 1}: Cannot apply, no quote text for search and no offsets."
                )
                continue

        if segment_start is None or segment_end is None:
            logger.warning(
                f"Patch {patch_idx + 1}: Failed to find target segment via {method_used}."
            )
            continue

        # Check for overlaps with already patched spans and other new patches in this batch
        is_overlapping = any(
            max(segment_start, old_start) < min(segment_end, old_end)
            for old_start, old_end in already_patched_spans
        ) or any(
            max(segment_start, r_start) < min(segment_end, r_end)
            for r_start, r_end, _ in replacements
        )

        if is_overlapping:
            logger.warning(
                f"Patch {patch_idx + 1} for segment {segment_start}-{segment_end} overlaps with a previously patched area or another new patch. Skipping."
            )
            continue

        original_segment = original_text[segment_start:segment_end]
        if replacement_text.strip() == original_segment.strip():
            logger.info(
                f"Patch {patch_idx + 1}: replacement identical to original segment {segment_start}-{segment_end}. Skipping."
            )
            continue
        if replacement_text.strip() and original_segment.strip():
            orig_emb, repl_emb = await asyncio.gather(
                llm_service.async_get_embedding(original_segment),
                llm_service.async_get_embedding(replacement_text),
            )
            if orig_emb is not None and repl_emb is not None:
                try:
                    similarity = utils.numpy_cosine_similarity(orig_emb, repl_emb)
                except ValueError:
                    logger.warning(
                        "Cosine similarity shape mismatch handled: setting to 0.0 for patch similarity check."
                    )
                    similarity = 0.0
                if similarity >= config.REVISION_SIMILARITY_ACCEPTANCE:
                    logger.info(
                        f"Patch {patch_idx + 1}: replacement highly similar to original segment {segment_start}-{segment_end}. Skipping."
                    )
                    continue

        replacements.append((segment_start, segment_end, replacement_text))
        log_action = "DELETION" if not replacement_text.strip() else "REPLACEMENT"
        logger.info(
            f"Patch {patch_idx + 1}: Queued {log_action} for {segment_start}-{segment_end} via {method_used}."
        )

    if not replacements:
        logger.info("No non-overlapping patches to apply in this cycle.")
        return original_text, already_patched_spans

    # 2. Build the new text and remap all spans in a single pass.
    # Create a unified list of all operations (old spans to copy, new spans to insert).
    all_ops: list[dict[str, Any]] = []
    for start, end in already_patched_spans:
        all_ops.append(
            {
                "type": "old",
                "start": start,
                "end": end,
                "text": original_text[start:end],
            }
        )
    for start, end, text in replacements:
        all_ops.append({"type": "new", "start": start, "end": end, "text": text})

    all_ops.sort(key=lambda x: x["start"])

    result_parts = []
    all_spans_in_new_text = []
    last_original_end = 0

    for op in all_ops:
        # Copy the text from the end of the last operation to the start of this one
        result_parts.append(original_text[last_original_end : op["start"]])

        # Calculate the starting position of the new span in the constructed text
        new_span_start = len("".join(result_parts))

        # Append the operation's text (either old text or new replacement text)
        result_parts.append(op["text"])

        # Calculate the end position and add the span to our list
        new_span_end = len("".join(result_parts))

        # MODIFICATION: Only add a protected span if the replacement was not a deletion.
        # A deleted segment should not be protected from future patches.
        if new_span_end > new_span_start:
            all_spans_in_new_text.append((new_span_start, new_span_end))

        # Update the pointer for the next iteration
        last_original_end = op["end"]

    # Append any remaining text after the last operation
    result_parts.append(original_text[last_original_end:])

    patched_text = "".join(result_parts)
    final_spans = sorted(all_spans_in_new_text)

    num_deletions = sum(1 for _, _, txt in replacements if not txt.strip())
    num_replacements = len(replacements) - num_deletions
    logger.info(
        f"Applied {num_replacements} replacements and {num_deletions} deletions. Total protected spans in new text: {len(final_spans)}."
    )

    return patched_text, final_spans


async def revise_chapter_draft_logic(
    plot_outline: dict[str, Any],
    character_profiles: dict[str, CharacterProfile],
    world_building: dict[str, dict[str, WorldItem]],
    original_text: str,
    chapter_number: int,
    evaluation_result: EvaluationResult,
    hybrid_context_for_revision: str,
    chapter_plan: list[SceneDetail] | None,
    is_from_flawed_source: bool = False,
    already_patched_spans: list[tuple[int, int]] | None = None,
) -> tuple[str, str, list[tuple[int, int]]] | None:
    """
    Orchestrates the chapter revision process with patch-based and full rewrite options.

    This function has been refactored into smaller, focused sub-functions for better
    maintainability and reduced complexity.
    """
    if already_patched_spans is None:
        already_patched_spans = []

    # Phase 1: Validate inputs
    is_valid, early_return = _validate_revision_inputs(original_text, chapter_number)
    if not is_valid:
        return early_return

    # Phase 2: Prepare problems for revision
    problems_to_fix, revision_reason_str, should_continue = (
        _prepare_problems_for_revision(evaluation_result, chapter_number)
    )
    if not should_continue:
        return (original_text, "No revision performed.", [])

    # Phase 3: Attempt patch-based revision
    patched_text, updated_spans = await _attempt_patch_based_revision(
        plot_outline,
        original_text,
        problems_to_fix,
        chapter_number,
        hybrid_context_for_revision,
        chapter_plan,
        already_patched_spans,
    )

    # Phase 4: Evaluate patched text quality
    final_revised_text: str | None = None
    final_raw_llm_output: str | None = (
        f"Chapter revised using {len(updated_spans) - len(already_patched_spans)} new patches."
    )
    final_spans_for_next_cycle = updated_spans

    use_patched_text_as_final = False
    if patched_text is not None and patched_text != original_text:
        use_patched_text_as_final = await _evaluate_patched_text(
            plot_outline,
            character_profiles,
            world_building,
            patched_text,
            chapter_number,
            hybrid_context_for_revision,
        )

    if use_patched_text_as_final:
        final_revised_text = patched_text
        logger.info(f"Ch {chapter_number}: Using patched text as the revised version.")

    # Phase 5: Perform full rewrite if needed
    if not use_patched_text_as_final and evaluation_result.get("needs_revision"):
        final_revised_text, final_raw_llm_output = await _perform_full_rewrite(
            plot_outline,
            original_text,
            problems_to_fix,
            chapter_number,
            hybrid_context_for_revision,
            chapter_plan,
            revision_reason_str,
            is_from_flawed_source,
        )
        final_spans_for_next_cycle = []  # A full rewrite resets the patched spans.

    # Phase 6: Final validation
    is_valid, early_return = _validate_final_result(final_revised_text, chapter_number)
    if not is_valid:
        return early_return

    return (
        final_revised_text,
        final_raw_llm_output,
        final_spans_for_next_cycle,
    )
