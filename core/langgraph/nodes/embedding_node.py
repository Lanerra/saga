# core/langgraph/nodes/embedding_node.py
"""
Embedding generation node for LangGraph workflow.

This module creates embeddings for generated chapter text to enable
semantic search and context retrieval.
"""

import numpy as np
import structlog

from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service

logger = structlog.get_logger(__name__)


async def generate_embedding(state: NarrativeState) -> NarrativeState:
    """
    Generate embedding for the current chapter draft text.

    This node should run after generation and before/parallel to extraction.
    It updates the state with the vector representation of the chapter text.

    Args:
        state: Current narrative state containing draft_text

    Returns:
        Updated state with generated_embedding
    """
    draft_text = state.get("draft_text")
    chapter_num = state.get("current_chapter")

    logger.info(
        "generate_embedding: starting embedding generation", chapter=chapter_num
    )

    if not draft_text or not draft_text.strip():
        logger.warning(
            "generate_embedding: no draft text found, skipping embedding generation",
            chapter=chapter_num,
        )
        return {
            **state,
            "generated_embedding": None,
            "current_node": "generate_embedding",
        }

    try:
        # Generate embedding using the refactored LLM service
        embedding = await llm_service.async_get_embedding(draft_text)

        if embedding is None:
            logger.error(
                "generate_embedding: failed to generate embedding (returned None)",
                chapter=chapter_num,
            )
            return {
                **state,
                "generated_embedding": None,
                "current_node": "generate_embedding",
            }

        # Convert numpy array to list for JSON serialization in state
        if isinstance(embedding, np.ndarray):
            embedding_list = embedding.tolist()
        else:
            embedding_list = list(embedding)

        logger.info(
            "generate_embedding: successfully generated embedding",
            chapter=chapter_num,
            dimensions=len(embedding_list),
        )

        return {
            **state,
            "generated_embedding": embedding_list,
            "current_node": "generate_embedding",
        }

    except Exception as e:
        logger.error(
            "generate_embedding: error during embedding generation",
            error=str(e),
            chapter=chapter_num,
            exc_info=True,
        )
        # We don't want to fail the whole workflow for a missing embedding,
        # but we should log it as a non-fatal error
        return {
            **state,
            "generated_embedding": None,
            "current_node": "generate_embedding",
            "last_error": f"Embedding generation failed: {e}",
            # Note: Not setting has_fatal_error=True as this is optional
        }
