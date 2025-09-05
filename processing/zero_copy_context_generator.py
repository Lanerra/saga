# processing/zero_copy_context_generator.py
"""
Zero-copy context generation that eliminates serialization overhead.
Uses single comprehensive Neo4j queries and direct field access.
"""

import asyncio
import logging
from typing import Any

import numpy as np

import config
from core.db_manager import neo4j_manager
from core.llm_interface_refactored import count_tokens, llm_service
from core.text_processing_service import truncate_text_by_tokens
from models import SceneDetail
from prompts.prompt_data_getters import get_reliable_kg_facts_for_drafting_prompt

logger = logging.getLogger(__name__)


class ZeroCopyContextGenerator:
    """Generate context with minimal data copying and serialization."""

    @staticmethod
    async def generate_hybrid_context_native(
        plot_outline: dict[str, Any],
        current_chapter_number: int,
        chapter_plan: list[SceneDetail] | None = None,
        semantic_limit: int | None = None,
        kg_limit: int | None = None,
    ) -> str:
        """
        Generate hybrid context without serialization overhead.

        Args:
            plot_outline: Plot outline data
            current_chapter_number: Current chapter being processed
            chapter_plan: Optional scene details for KG facts
            semantic_limit: Max chapters for semantic context
            kg_limit: Max facts for KG context

        Returns:
            Formatted hybrid context string
        """
        if current_chapter_number <= 0:
            return ""

        logger.info(
            f"Generating NATIVE hybrid context for Chapter {current_chapter_number}..."
        )

        # Run semantic and KG context generation in parallel
        semantic_task = ZeroCopyContextGenerator._generate_semantic_context_native(
            plot_outline, current_chapter_number, semantic_limit
        )
        kg_facts_task = get_reliable_kg_facts_for_drafting_prompt(
            plot_outline, current_chapter_number, chapter_plan
        )

        semantic_context_str, kg_facts_str = await asyncio.gather(
            semantic_task, kg_facts_task
        )

        # Build hybrid context with minimal string operations
        context_parts = []

        # Add plot point focus section - THIS IS THE CRITICAL FIX
        plot_point_focus = ZeroCopyContextGenerator._get_plot_point_for_chapter(
            plot_outline, current_chapter_number
        )
        if plot_point_focus and plot_point_focus.strip():
            # Add enhanced guidance for narrative focus
            focus_directive = ZeroCopyContextGenerator._get_narrative_focus_directive(
                plot_point_focus, current_chapter_number
            )
            context_parts.extend(
                [
                    f"--- PLOT POINT FOR CHAPTER {current_chapter_number} (PRIMARY FOCUS) ---",
                    f"**Chapter {current_chapter_number} Plot Point:** {plot_point_focus}",
                    "",
                    f"**Narrative Focus Directive:** {focus_directive}",
                    "--- END PLOT POINT ---",
                ]
            )
        else:
            context_parts.extend(
                [
                    f"--- PLOT POINT FOR CHAPTER {current_chapter_number} (PRIMARY FOCUS) ---",
                    f"No specific plot point found for chapter {current_chapter_number}. Follow general narrative progression.",
                    "--- END PLOT POINT ---",
                ]
            )

        # Add semantic context section
        if semantic_context_str and semantic_context_str.strip():
            context_parts.extend(
                [
                    "",
                    "--- SEMANTIC CONTEXT FROM PAST CHAPTERS (FOR NARRATIVE FLOW & TONE) ---",
                    semantic_context_str,
                    "--- END SEMANTIC CONTEXT ---",
                ]
            )
        else:
            context_parts.extend(
                [
                    "",
                    "--- SEMANTIC CONTEXT FROM PAST CHAPTERS (FOR NARRATIVE FLOW & TONE) ---",
                    "No relevant semantic context could be retrieved.",
                    "--- END SEMANTIC CONTEXT ---",
                ]
            )

        # Add KG facts section
        if kg_facts_str and kg_facts_str.strip():
            context_parts.extend(
                [
                    "",
                    "--- KEY RELIABLE KG FACTS (FOR ESTABLISHED CANON & CONTINUITY) ---",
                    kg_facts_str,
                    "--- END KEY RELIABLE KG FACTS ---",
                ]
            )
        else:
            context_parts.extend(
                [
                    "",
                    "--- KEY RELIABLE KG FACTS (FOR ESTABLISHED CANON & CONTINUITY) ---",
                    "No reliable KG facts could be retrieved for this context.",
                    "--- END KEY RELIABLE KG FACTS ---",
                ]
            )

        return "\n".join(context_parts)

    @staticmethod
    def _get_plot_point_for_chapter(
        plot_outline: dict[str, Any], current_chapter_number: int
    ) -> str | None:
        """
        Extract the plot point description for the specified chapter.

        Args:
            plot_outline: Plot outline data
            current_chapter_number: Chapter number (1-based)

        Returns:
            Plot point description string, or None if not found
        """
        plot_points = plot_outline.get("plot_points", [])
        if not plot_points or current_chapter_number <= 0:
            return None

        plot_point_index = current_chapter_number - 1  # Convert to 0-based index

        if 0 <= plot_point_index < len(plot_points):
            plot_point_item = plot_points[plot_point_index]
            # Handle both string and dict formats
            if isinstance(plot_point_item, dict):
                return plot_point_item.get("description", "").strip()
            elif isinstance(plot_point_item, str):
                return plot_point_item.strip()

        return None

    @staticmethod
    def _get_narrative_focus_directive(
        plot_point_text: str, chapter_number: int
    ) -> str:
        """
        Generate a narrative focus directive based on plot point content and chapter progression.

        Args:
            plot_point_text: The plot point description
            chapter_number: Current chapter number

        Returns:
            A directive to guide narrative focus and differentiate from previous chapters
        """
        plot_lower = plot_point_text.lower()

        # Early chapters (1-3): Focus on progression from internal to external
        if chapter_number == 1:
            if any(
                word in plot_lower
                for word in ["discover", "finds", "uncover", "reveal"]
            ):
                return "Focus on INTERNAL DISCOVERY and personal revelation. Emphasize quiet moments of realization, the weight of truth, and the protagonist's emotional response to learning something that changes their worldview."
            else:
                return "Focus on ATMOSPHERE and CHARACTER ESTABLISHMENT. Build the world and introduce the protagonist's situation through immersive details."

        elif chapter_number == 2:
            if any(
                word in plot_lower
                for word in ["frame", "flee", "escape", "hunt", "chase", "forced"]
            ):
                return "Focus on EXTERNAL PRESSURE and ACTION. Emphasize movement, urgency, pursuit, and the protagonist's transition from passive to active. This chapter should feel dynamic and show escalating stakes."
            elif any(
                word in plot_lower for word in ["decrypt", "transmission", "uncover"]
            ):
                return "Focus on ACTIVE INVESTIGATION with building tension. Show the protagonist taking deliberate action while external forces close in. Balance discovery with mounting pressure."
            else:
                return "Focus on ESCALATION and STAKES RAISING. Build on the previous chapter's revelations with increased tension and external conflict."

        elif chapter_number == 3:
            if any(
                word in plot_lower
                for word in ["leader", "reveals", "network", "organization"]
            ):
                return "Focus on WORLD EXPANSION and new character dynamics. Introduce broader scope and shift from personal crisis to larger conflict."
            else:
                return "Focus on SCOPE EXPANSION. Broaden the conflict beyond the protagonist's personal struggle."

        elif chapter_number == 4:
            if any(
                word in plot_lower
                for word in ["genetic", "valuable", "special", "abilities"]
            ):
                return "Focus on PROTAGONIST EMPOWERMENT and personal significance. Show the protagonist's unique value and growing agency."
            else:
                return "Focus on PROTAGONIST AGENCY. Show the protagonist taking control of their destiny."

        # Later chapters: Focus on progression and climax building
        else:
            if chapter_number <= 8:
                return "Focus on SKILL DEVELOPMENT and relationship building. Show the protagonist becoming more capable and connected."
            elif chapter_number <= 12:
                return "Focus on STRATEGIC ACTION and alliance building. Show coordinated efforts and growing resistance."
            elif chapter_number <= 16:
                return "Focus on ESCALATING CONFLICT and high-stakes confrontations. Build toward climax."
            else:
                return "Focus on RESOLUTION and CONSEQUENCES. Show the ultimate confrontation and its aftermath."

    @staticmethod
    async def _generate_semantic_context_native(
        plot_outline: dict[str, Any],
        current_chapter_number: int,
        semantic_limit: int | None = None,
    ) -> str:
        """
        Generate semantic context using single optimized Neo4j query.
        Eliminates multiple DB calls and dict conversion overhead.
        """
        if current_chapter_number <= 1:
            return ""

        # Get plot point focus for query
        plot_points = plot_outline.get("plot_points", [])
        plot_point_focus = None

        if plot_points and current_chapter_number > 0:
            idx = current_chapter_number - 1
            if 0 <= idx < len(plot_points):
                plot_point_focus = str(plot_points[idx]) if plot_points[idx] else None

        context_query_text = (
            plot_point_focus
            if plot_point_focus
            else f"Narrative context relevant to events leading up to chapter {current_chapter_number}."
        )

        # Get embedding for similarity search
        query_embedding_np = await llm_service.async_get_embedding(context_query_text)
        if query_embedding_np is None:
            logger.warning(
                "Failed to generate embedding for semantic context query. Using fallback."
            )
            return await ZeroCopyContextGenerator._fallback_sequential_context(
                current_chapter_number
            )

        # Single comprehensive Neo4j query for all context data
        return await ZeroCopyContextGenerator._execute_semantic_context_query(
            query_embedding_np, current_chapter_number, semantic_limit
        )

    @staticmethod
    async def _execute_semantic_context_query(
        query_embedding: np.ndarray,
        current_chapter_number: int,
        semantic_limit: int | None = None,
    ) -> str:
        """
        Execute optimized Neo4j query for semantic context.
        Returns formatted context directly without intermediate conversions.
        """
        query_embedding_list = neo4j_manager.embedding_to_list(query_embedding)
        if not query_embedding_list:
            return ""

        limit = semantic_limit or config.CONTEXT_CHAPTER_COUNT
        max_semantic_tokens = (config.MAX_CONTEXT_TOKENS * 2) // 3

        # Single query to get similar chapters AND immediate previous chapter
        cypher_query = """
        // Get similar chapters via vector search
        CALL db.index.vector.queryNodes($index_name, $limit, $query_vector)
        YIELD node AS similar_c, score
        WHERE similar_c.number < $current_chapter
        
        WITH collect({
            chapter_number: similar_c.number,
            summary: similar_c.summary,
            text: similar_c.text,
            is_provisional: COALESCE(similar_c.is_provisional, false),
            score: score,
            source: 'similarity'
        }) AS similar_chapters
        
        // Also get immediate previous chapter if not in similarity results
        OPTIONAL MATCH (prev_c:Chapter) 
        WHERE prev_c.number = $prev_chapter_num
        
        WITH similar_chapters, prev_c,
             CASE 
                WHEN prev_c IS NOT NULL AND 
                     NOT ANY(sc IN similar_chapters WHERE sc.chapter_number = $prev_chapter_num)
                THEN [{
                    chapter_number: prev_c.number,
                    summary: prev_c.summary,
                    text: prev_c.text,
                    is_provisional: COALESCE(prev_c.is_provisional, false),
                    score: 0.999,
                    source: 'immediate_previous'
                }]
                ELSE []
             END AS prev_chapter_data
        
        RETURN similar_chapters + prev_chapter_data AS all_context_chapters
        """

        try:
            results = await neo4j_manager.execute_read_query(
                cypher_query,
                {
                    "index_name": config.NEO4J_VECTOR_INDEX_NAME,
                    "limit": limit + 1,  # Extra for potential duplicates
                    "query_vector": query_embedding_list,
                    "current_chapter": current_chapter_number,
                    "prev_chapter_num": current_chapter_number - 1,
                },
            )

            if not results or not results[0]:
                logger.info(
                    "No semantic context chapters found via Neo4j vector search."
                )
                return ""

            context_chapters = results[0]["all_context_chapters"]
            if not context_chapters:
                return ""

            # Sort by score and chapter number (highest score first)
            sorted_chapters = sorted(
                context_chapters,
                key=lambda x: (x.get("score", 0.0), x.get("chapter_number", 0)),
                reverse=True,
            )

            # Build context with token tracking
            return ZeroCopyContextGenerator._build_context_from_chapters(
                sorted_chapters, max_semantic_tokens
            )

        except Exception as e:
            logger.error(f"Error executing semantic context query: {e}", exc_info=True)
            return await ZeroCopyContextGenerator._fallback_sequential_context(
                current_chapter_number
            )

    @staticmethod
    def _build_context_from_chapters(
        chapters: list[dict[str, Any]], max_tokens: int
    ) -> str:
        """
        Build context string from chapter data with token limit enforcement.
        Optimized for minimal string operations.
        """
        context_parts = []
        total_tokens = 0

        for chap_data in chapters:
            if total_tokens >= max_tokens:
                break

            chap_num = chap_data.get("chapter_number")
            content = (chap_data.get("summary") or chap_data.get("text", "")).strip()
            is_provisional = chap_data.get("is_provisional", False)
            score = chap_data.get("score", "N/A")

            if not content:
                continue

            # Format content type
            content_type = (
                "Provisional Summary"
                if chap_data.get("summary") and is_provisional
                else "Summary"
                if chap_data.get("summary")
                else "Provisional Text Snippet"
                if is_provisional
                else "Text Snippet"
            )

            score_str = f"{score:.3f}" if isinstance(score, float) else str(score)

            # Build formatted content
            prefix = f"[Semantic Context from Chapter {chap_num} (Similarity: {score_str}, Type: {content_type})]:\n"
            suffix = "\n---\n"
            full_content = f"{prefix}{content}{suffix}"

            # Check token limit
            content_tokens = count_tokens(full_content, config.DRAFTING_MODEL)

            if total_tokens + content_tokens <= max_tokens:
                context_parts.append(full_content)
                total_tokens += content_tokens
            else:
                # Try to fit truncated version
                remaining_tokens = max_tokens - total_tokens
                prefix_suffix_tokens = count_tokens(
                    prefix + suffix, config.DRAFTING_MODEL
                )

                if remaining_tokens > prefix_suffix_tokens + 10:
                    truncated_content = truncate_text_by_tokens(
                        full_content, config.DRAFTING_MODEL, remaining_tokens
                    )
                    context_parts.append(truncated_content)
                    total_tokens += remaining_tokens
                break

            logger.debug(
                f"Added semantic context from ch {chap_num} ({content_type}, Sim: {score_str}), "
                f"{content_tokens} tokens. Total: {total_tokens}."
            )

        final_context = "\n".join(reversed(context_parts)).strip()
        final_tokens = count_tokens(final_context, config.DRAFTING_MODEL)

        logger.info(
            f"Built semantic context: {final_tokens} tokens from {len(context_parts)} chapters."
        )
        return final_context

    @staticmethod
    async def _fallback_sequential_context(current_chapter_number: int) -> str:
        """Fallback context generation using sequential chapter access."""
        max_semantic_tokens = (config.MAX_CONTEXT_TOKENS * 2) // 3
        context_parts = []
        total_tokens = 0

        fallback_limit = config.CONTEXT_CHAPTER_COUNT
        start_chapter = max(1, current_chapter_number - fallback_limit)

        for chapter_num in range(start_chapter, current_chapter_number):
            if total_tokens >= max_semantic_tokens:
                break

            try:
                # Simple direct query for fallback
                query = "MATCH (c:Chapter {number: $chapter_num}) RETURN c.summary AS summary, c.text AS text"
                results = await neo4j_manager.execute_read_query(
                    query, {"chapter_num": chapter_num}
                )

                if results and results[0]:
                    content = (
                        results[0].get("summary") or results[0].get("text", "")
                    ).strip()
                    if content:
                        formatted = f"[Fallback Context from Chapter {chapter_num}]:\n{content}\n---\n"
                        content_tokens = count_tokens(formatted, config.DRAFTING_MODEL)

                        if total_tokens + content_tokens <= max_semantic_tokens:
                            context_parts.append(formatted)
                            total_tokens += content_tokens
                        else:
                            break

            except Exception as e:
                logger.debug(
                    f"Error getting fallback context for chapter {chapter_num}: {e}"
                )
                continue

        final_context = "\n".join(reversed(context_parts)).strip()
        logger.info(
            f"Built fallback semantic context: {count_tokens(final_context, config.DRAFTING_MODEL)} tokens."
        )
        return final_context
