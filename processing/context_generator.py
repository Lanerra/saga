# context_generation_logic.py
"""
Handles the generation of contextual information for chapter writing in the SAGA system.
Now includes a hybrid approach combining semantic context and Knowledge Graph facts.
"""

import asyncio
from typing import Any

import structlog
from config import settings
from core.llm_interface import (
    count_tokens,
    llm_service,
    truncate_text_by_tokens,
)  # MODIFIED
from data_access import (
    chapter_queries,
)  # For chapter data and similarity search
from kg_maintainer.models import SceneDetail
from prompt_data_getters import get_reliable_kg_facts_for_drafting_prompt

logger = structlog.get_logger(__name__)


def _get_nested_prop_from_agent_or_props(
    agent_or_props: Any,
    primary_key: str,
    secondary_key: str,
    default: Any = None,
) -> Any:
    """Helper to get a nested property from an agent-like object or a dictionary."""
    primary_data = (
        getattr(agent_or_props, primary_key, None)
        if not isinstance(agent_or_props, dict)
        else agent_or_props.get(primary_key, {})
    )
    if isinstance(primary_data, dict):
        return primary_data.get(secondary_key, default)
    return getattr(primary_data, secondary_key, default)


async def _generate_semantic_chapter_context_logic(
    agent_or_props: Any, current_chapter_number: int
) -> str:
    """
    Constructs SEMANTIC context for the current chapter from previous summaries/text,
    using Neo4j vector similarity search.
    'agent_or_props' can be the NANA_Orchestrator instance or a novel_props dictionary.
    This function will now be mindful of token limits for its output.
    """
    if current_chapter_number <= 1:
        return ""
    logger.debug(
        f"Retrieving and constructing SEMANTIC context for Chapter {current_chapter_number} via Neo4j vector search..."
    )

    if isinstance(agent_or_props, dict):
        plot_outline_data = agent_or_props.get(
            "plot_outline_full", agent_or_props.get("plot_outline", {})
        )
    else:
        plot_outline_data = getattr(agent_or_props, "plot_outline_full", None)
        if not plot_outline_data:
            plot_outline_data = getattr(agent_or_props, "plot_outline", {})

    plot_points = []
    if isinstance(plot_outline_data, dict):
        plot_points = plot_outline_data.get("plot_points", [])
    else:
        plot_points = getattr(plot_outline_data, "plot_points", [])

    plot_point_focus = None
    if plot_points and isinstance(plot_points, list) and current_chapter_number > 0:
        idx = current_chapter_number - 1
        if 0 <= idx < len(plot_points):
            plot_point_focus = (
                str(plot_points[idx]) if plot_points[idx] is not None else None
            )
        else:
            logger.warning(
                f"Cannot determine plot point focus for chapter {current_chapter_number}: index {idx} out of bounds for {len(plot_points)} plot points."
            )
    else:
        logger.warning(
            f"Cannot determine plot point focus for chapter {current_chapter_number}: plot_points list is empty or invalid."
        )

    context_query_text = (
        plot_point_focus
        if plot_point_focus
        else f"Narrative context relevant to events leading up to chapter {current_chapter_number}."
    )

    if plot_point_focus:
        plot_point_index_display = (
            current_chapter_number - 1
        ) + 1  # 1-based for logging
        logger.info(
            f"Semantic context query for ch {current_chapter_number} based on Plot Point {plot_point_index_display}: '{context_query_text[:100]}...'"
        )
    else:
        logger.warning(
            f"No specific plot point found for ch {current_chapter_number}. Using generic semantic context query."
        )

    max_semantic_tokens = (settings.MAX_CONTEXT_TOKENS * 2) // 3

    # Include the summaries of the immediate previous chapters before searching
    immediate_context_limit = 2
    immediate_start = max(1, current_chapter_number - immediate_context_limit)
    immediate_parts: list[str] = []
    total_tokens_accumulated = 0

    for i in range(immediate_start, current_chapter_number):
        if total_tokens_accumulated >= max_semantic_tokens:
            break
        chap_data = await chapter_queries.get_chapter_data_from_db(i)
        if chap_data:
            content = (chap_data.get("summary") or chap_data.get("text", "")).strip()
            is_prov = chap_data.get("is_provisional", False)
            ctype = (
                "Provisional Summary"
                if chap_data.get("summary") and is_prov
                else "Summary"
                if chap_data.get("summary")
                else "Provisional Text Snippet"
                if is_prov
                else "Text Snippet"
            )
            if content:
                prefix = f"[Immediate Context from Chapter {i} ({ctype})]:\n"
                suffix = "\n---\n"
                full_content_part = f"{prefix}{content}{suffix}"
                part_tokens = count_tokens(full_content_part, settings.DRAFTING_MODEL)
                if total_tokens_accumulated + part_tokens <= max_semantic_tokens:
                    immediate_parts.append(full_content_part)
                    total_tokens_accumulated += part_tokens
                else:
                    remaining_tokens = max_semantic_tokens - total_tokens_accumulated
                    if (
                        remaining_tokens
                        > count_tokens(prefix + suffix, settings.DRAFTING_MODEL) + 10
                    ):
                        truncated_content_part = truncate_text_by_tokens(
                            full_content_part,
                            settings.DRAFTING_MODEL,
                            remaining_tokens,
                        )
                        immediate_parts.append(truncated_content_part)
                        total_tokens_accumulated += remaining_tokens
                    break

    if total_tokens_accumulated >= max_semantic_tokens:
        final_semantic_context = "\n".join(reversed(immediate_parts)).strip()
        final_tokens_count = count_tokens(
            final_semantic_context, settings.DRAFTING_MODEL
        )
        logger.info(
            f"Constructed semantic context solely from immediate chapters: {final_tokens_count} tokens."
        )
        return final_semantic_context

    query_embedding_np = await llm_service.async_get_embedding(context_query_text)

    if query_embedding_np is None:
        logger.warning(
            "Failed to generate embedding for semantic context query. Falling back to sequential previous chapter summaries/text."
        )
        context_parts_list: list[str] = []
        fallback_chapter_limit = settings.CONTEXT_CHAPTER_COUNT
        for i in range(
            max(1, current_chapter_number - fallback_chapter_limit),
            immediate_start,
        ):
            if total_tokens_accumulated >= max_semantic_tokens:
                break
            chap_data = await chapter_queries.get_chapter_data_from_db(i)
            if chap_data:
                content = (
                    chap_data.get("summary") or chap_data.get("text", "")
                ).strip()
                is_prov = chap_data.get("is_provisional", False)
                ctype = (
                    "Provisional Summary"
                    if chap_data.get("summary") and is_prov
                    else "Summary"
                    if chap_data.get("summary")
                    else "Provisional Text Snippet"
                    if is_prov
                    else "Text Snippet"
                )
                if content:
                    prefix = (
                        f"[Fallback Semantic Context from Chapter {i} ({ctype})]:\n"
                    )
                    suffix = "\n---\n"
                    full_content_part = f"{prefix}{content}{suffix}"
                    part_tokens = count_tokens(
                        full_content_part, settings.DRAFTING_MODEL
                    )
                    if total_tokens_accumulated + part_tokens <= max_semantic_tokens:
                        context_parts_list.append(full_content_part)
                        total_tokens_accumulated += part_tokens
                    else:
                        remaining_tokens = (
                            max_semantic_tokens - total_tokens_accumulated
                        )
                        if (
                            remaining_tokens
                            > count_tokens(prefix + suffix, settings.DRAFTING_MODEL)
                            + 10
                        ):
                            truncated_content_part = truncate_text_by_tokens(
                                full_content_part,
                                settings.DRAFTING_MODEL,
                                remaining_tokens,
                            )
                            context_parts_list.append(truncated_content_part)
                            total_tokens_accumulated += remaining_tokens
                        break
        final_semantic_context = "\n".join(
            immediate_parts + list(reversed(context_parts_list))
        ).strip()
        final_tokens_count = count_tokens(
            final_semantic_context, settings.DRAFTING_MODEL
        )
        logger.info(
            f"Constructed fallback semantic context: {final_tokens_count} tokens."
        )
        return final_semantic_context

    similar_chapters_data = await chapter_queries.find_similar_chapters_in_db(
        query_embedding_np,
        settings.CONTEXT_CHAPTER_COUNT,
        current_chapter_number,
    )

    if not similar_chapters_data:
        logger.info(
            "No similar past chapters found via Neo4j vector search for semantic context."
        )
        return ""

    # Exclude immediate context chapters and apply decay to similarity scores
    excluded_chapters = set(range(immediate_start, current_chapter_number))
    filtered_chapters: list[dict[str, Any]] = []
    for ch in similar_chapters_data:
        if ch.get("chapter_number") in excluded_chapters:
            continue
        score = float(ch.get("score", 0.0))
        distance = current_chapter_number - int(ch.get("chapter_number", 0))
        adjusted = score * (0.95 ** max(distance, 0))
        ch["adjusted_score"] = adjusted
        filtered_chapters.append(ch)

    sorted_chapters_for_context = sorted(
        filtered_chapters,
        key=lambda x: x.get("adjusted_score", 0.0),
        reverse=True,
    )

    context_parts_list: list[str] = []

    for chap_data in sorted_chapters_for_context:
        if total_tokens_accumulated >= max_semantic_tokens:
            break

        chap_num = chap_data["chapter_number"]
        content = (chap_data.get("summary") or chap_data.get("text", "")).strip()
        is_prov = chap_data.get("is_provisional", False)
        score = chap_data.get("score", "N/A")
        score_str = f"{score:.3f}" if isinstance(score, float) else str(score)

        ctype = (
            "Provisional Summary"
            if chap_data.get("summary") and is_prov
            else "Summary"
            if chap_data.get("summary")
            else "Provisional Text Snippet"
            if is_prov
            else "Text Snippet"
        )

        if content:
            prefix = f"[Semantic Context from Chapter {chap_num} (Similarity: {score_str}, Type: {ctype})]:\n"
            suffix = "\n---\n"
            full_content_part = f"{prefix}{content}{suffix}"
            part_tokens = count_tokens(
                full_content_part, settings.DRAFTING_MODEL
            )  # MODIFIED

            if total_tokens_accumulated + part_tokens <= max_semantic_tokens:
                context_parts_list.append(full_content_part)
                total_tokens_accumulated += part_tokens
            else:
                remaining_tokens = max_semantic_tokens - total_tokens_accumulated
                if (
                    remaining_tokens
                    > count_tokens(prefix + suffix, settings.DRAFTING_MODEL) + 10
                ):  # MODIFIED
                    truncated_content_part = truncate_text_by_tokens(  # MODIFIED
                        full_content_part,
                        settings.DRAFTING_MODEL,
                        remaining_tokens,
                    )
                    context_parts_list.append(truncated_content_part)
                    total_tokens_accumulated += remaining_tokens
                break
            logger.debug(
                f"Added SEMANTIC context from ch {chap_num} ({ctype}, Sim: {score_str}), {part_tokens} tokens. Total: {total_tokens_accumulated}."
            )
        else:
            logger.warning(
                f"Chapter {chap_num} (Sim: {score_str}) from vector search had no content (summary/text). Skipping."
            )

    final_semantic_context = "\n".join(
        list(reversed(immediate_parts)) + context_parts_list
    ).strip()
    final_tokens_count = count_tokens(final_semantic_context, settings.DRAFTING_MODEL)
    logger.info(
        f"Constructed final SEMANTIC context: {final_tokens_count} tokens from {len(immediate_parts) + len(context_parts_list)} chapter snippets (via Neo4j vector search)."
    )
    return final_semantic_context


async def generate_hybrid_chapter_context_logic(
    agent_or_props: Any,
    current_chapter_number: int,
    chapter_plan: list[SceneDetail] | None,
) -> str:
    """
    Constructs HYBRID context for the current chapter.
    'agent_or_props' can be the NANA_Orchestrator instance or a novel_props dictionary.
    MODIFIED: Ensures agent_or_props is passed to helpers.
    """
    if current_chapter_number <= 0:
        return ""
    logger.info(f"Generating HYBRID context for Chapter {current_chapter_number}...")

    semantic_context_task = _generate_semantic_chapter_context_logic(
        agent_or_props, current_chapter_number
    )
    if isinstance(agent_or_props, dict):
        plot_outline_data = agent_or_props.get(
            "plot_outline_full", agent_or_props.get("plot_outline", {})
        )
    else:
        plot_outline_data = getattr(agent_or_props, "plot_outline_full", None)
        if not plot_outline_data:
            plot_outline_data = getattr(agent_or_props, "plot_outline", {})

    kg_facts_task = get_reliable_kg_facts_for_drafting_prompt(
        plot_outline_data, current_chapter_number, chapter_plan
    )

    semantic_context_str, kg_facts_str = await asyncio.gather(
        semantic_context_task, kg_facts_task
    )
    hybrid_context_parts: list[str] = []
    if semantic_context_str and semantic_context_str.strip():
        hybrid_context_parts.append(
            "--- SEMANTIC CONTEXT FROM PAST CHAPTERS (FOR NARRATIVE FLOW & TONE) ---"
        )
        hybrid_context_parts.append(semantic_context_str)
        hybrid_context_parts.append("--- END SEMANTIC CONTEXT ---")
    else:
        hybrid_context_parts.append(
            "--- SEMANTIC CONTEXT FROM PAST CHAPTERS (FOR NARRATIVE FLOW & TONE) ---"
        )
        hybrid_context_parts.append("No relevant semantic context could be retrieved.")
        hybrid_context_parts.append("--- END SEMANTIC CONTEXT ---")
    if kg_facts_str and kg_facts_str.strip():
        hybrid_context_parts.append(
            "\n\n--- KEY RELIABLE KG FACTS (FOR ESTABLISHED CANON & CONTINUITY) ---"
        )
        cleaned_kg_facts = (
            kg_facts_str.split("\n", 1)[-1]
            if kg_facts_str.startswith("**Key Reliable KG Facts")
            else kg_facts_str
        )
        if not cleaned_kg_facts.strip() or cleaned_kg_facts.lower().startswith(
            "no specific reliable kg facts"
        ):
            hybrid_context_parts.append(
                "No specific reliable KG facts were identified as highly relevant for this chapter's focus."
            )
        else:
            hybrid_context_parts.append(cleaned_kg_facts)
        hybrid_context_parts.append("--- END KEY RELIABLE KG FACTS ---")
    else:
        hybrid_context_parts.append(
            "\n\n--- KEY RELIABLE KG FACTS (FOR ESTABLISHED CANON & CONTINUITY) ---"
        )
        hybrid_context_parts.append(
            "Knowledge Graph fact retrieval did not yield specific results for this chapter."
        )
        hybrid_context_parts.append("--- END KEY RELIABLE KG FACTS ---")

    final_hybrid_context = "\n".join(hybrid_context_parts).strip()
    num_hybrid_tokens = count_tokens(
        final_hybrid_context, settings.DRAFTING_MODEL
    )  # MODIFIED
    if num_hybrid_tokens > settings.MAX_CONTEXT_TOKENS:
        logger.warning(
            f"Hybrid context token count ({num_hybrid_tokens}) exceeds MAX_CONTEXT_TOKENS ({settings.MAX_CONTEXT_TOKENS}). Truncating."
        )
        final_hybrid_context = truncate_text_by_tokens(  # MODIFIED
            final_hybrid_context,
            settings.DRAFTING_MODEL,
            settings.MAX_CONTEXT_TOKENS,
            truncation_marker="\n... (Hybrid context truncated due to token limit)",
        )
        num_hybrid_tokens = count_tokens(
            final_hybrid_context, settings.DRAFTING_MODEL
        )  # MODIFIED
    logger.info(
        f"Generated HYBRID context for Chapter {current_chapter_number}, Tokens (est.): {num_hybrid_tokens}."
    )
    return final_hybrid_context
