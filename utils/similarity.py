import asyncio
import structlog
from typing import Optional, Tuple

import numpy as np

from core.llm_interface import llm_service

from .text_processing import get_text_segments

logger = structlog.get_logger(__name__)


def numpy_cosine_similarity(
    vec1: Optional[np.ndarray], vec2: Optional[np.ndarray]
) -> float:
    """Calculate cosine similarity between two numpy vectors."""
    if vec1 is None or vec2 is None:
        logger.debug("Cosine similarity: one or both vectors are None. Returning 0.0.")
        return 0.0
    try:
        v1 = np.asarray(vec1, dtype=np.float32).flatten()
        v2 = np.asarray(vec2, dtype=np.float32).flatten()
    except ValueError as e:
        logger.warning(
            "Cosine similarity: Could not convert input to numpy array: %s. Returning 0.0.",
            e,
        )
        return 0.0
    if v1.shape != v2.shape:
        logger.warning(
            "Cosine similarity: shape mismatch %s vs %s. Returning 0.0.",
            v1.shape,
            v2.shape,
        )
        return 0.0
    if v1.size == 0:
        logger.debug("Cosine similarity: input vector(s) are empty. Returning 0.0.")
        return 0.0
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0.0 or norm_v2 == 0.0:
        logger.debug("Cosine similarity: at least one zero-norm vector. Returning 0.0.")
        return 0.0
    dot_product = np.dot(v1, v2)
    similarity = dot_product / (norm_v1 * norm_v2)
    return float(np.clip(similarity, -1.0, 1.0))


async def find_semantically_closest_segment(
    original_doc: str,
    query_text: str,
    segment_type: str = "paragraph",
    min_similarity_threshold: float = 0.65,
) -> Optional[Tuple[int, int, float]]:
    """Find the document segment most semantically similar to ``query_text``."""
    if not original_doc or not query_text:
        logger.debug(
            "find_semantically_closest_segment: original_doc or query_text is empty."
        )
        return None

    query_embedding = await llm_service.async_get_embedding(query_text)
    if query_embedding is None:
        logger.warning(
            "Could not get embedding for semantic search query: '%s...'",
            query_text[:60].replace(chr(10), " "),
        )
        return None

    segments_with_indices = get_text_segments(original_doc, segment_type)
    if not segments_with_indices:
        logger.warning(
            "No segments found in original document for semantic search (Type: %s).",
            segment_type,
        )
        return None

    best_match_info: Optional[Tuple[int, int, float]] = None
    highest_similarity = -2.0

    segment_texts = [s[0] for s in segments_with_indices]
    segment_embeddings_tasks = [
        llm_service.async_get_embedding(seg_text) for seg_text in segment_texts
    ]
    segment_embeddings_results = await asyncio.gather(
        *segment_embeddings_tasks, return_exceptions=True
    )

    for i, seg_embedding_or_exc in enumerate(segment_embeddings_results):
        if isinstance(seg_embedding_or_exc, Exception) or seg_embedding_or_exc is None:
            logger.warning(
                "Could not get embedding for segment: '%s...' Error: %s",
                segment_texts[i][:60].replace(chr(10), " "),
                seg_embedding_or_exc,
            )
            continue

        seg_embedding = seg_embedding_or_exc
        similarity = numpy_cosine_similarity(query_embedding, seg_embedding)

        if similarity > highest_similarity:
            highest_similarity = similarity
            _, start_char, end_char = segments_with_indices[i]
            best_match_info = (start_char, end_char, highest_similarity)

    if best_match_info and best_match_info[2] < min_similarity_threshold:
        logger.info(
            "Semantic match found for query '%s...' but similarity (%.2f) is below threshold (%.2f).",
            query_text[:60].replace(chr(10), " "),
            best_match_info[2],
            min_similarity_threshold,
        )
        return None

    if best_match_info:
        logger.info(
            "Best semantic match for query '%s...' has similarity %.2f at span %d-%d.",
            query_text[:60].replace(chr(10), " "),
            best_match_info[2],
            best_match_info[0],
            best_match_info[1],
        )
    else:
        logger.info(
            "No suitable semantic match found (above threshold %.2f) for query '%s...'. Highest sim was %.2f.",
            min_similarity_threshold,
            query_text[:60].replace(chr(10), " "),
            highest_similarity,
        )

    return best_match_info
