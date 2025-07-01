# data_access/chapter_queries.py
import json
from typing import Any

import numpy as np
import structlog
from config import settings
from core.db_manager import neo4j_manager
from core.llm_interface import llm_service

logger = structlog.get_logger(__name__)


async def load_chapter_count_from_db() -> int:
    """Return the number of chapters stored in the database."""

    query = (
        f"MATCH (c:{settings.NEO4J_VECTOR_NODE_LABEL}) RETURN count(c) AS chapter_count"
    )
    try:
        result = await neo4j_manager.execute_read_query(query)
        count = result[0]["chapter_count"] if result and result[0] else 0
        logger.info(f"Neo4j loaded chapter count: {count}")
        return count
    except Exception as e:
        logger.error(f"Failed to load chapter count from Neo4j: {e}", exc_info=True)
        return 0


async def save_chapter_data_to_db(
    chapter_number: int,
    text: str,
    raw_llm_output: str,
    summary: str | None,
    embedding_array: np.ndarray | None,
    is_provisional: bool = False,
    end_state: dict[str, Any] | None = None,
):
    """Save chapter text, raw output, and embedding to Neo4j.

    Args:
        chapter_number: Index of the chapter being stored.
        text: Final chapter text.
        raw_llm_output: Unprocessed LLM response for auditing.
        summary: Optional chapter summary text.
        embedding_array: Precomputed embedding for ``text``.
        is_provisional: Whether the chapter is a provisional draft.

    Returns:
        ``None``. Data is persisted directly to the database.
    """
    if chapter_number < 0:
        logger.error(
            f"Neo4j: Cannot save chapter data for invalid chapter_number: {chapter_number}."
        )
        return

    if embedding_array is None:
        embedding_array = await llm_service.async_get_embedding(text)
    embedding_list = neo4j_manager.embedding_to_list(embedding_array)

    query = f"""
    MERGE (c:{settings.NEO4J_VECTOR_NODE_LABEL} {{number: $chapter_number_param}})
    SET c.text = $text_param,
        c.raw_llm_output = $raw_llm_output_param,
        c.summary = $summary_param,
        c.is_provisional = $is_provisional_param,
        c.end_state_json = $end_state_param,
        c.{settings.NEO4J_VECTOR_PROPERTY_NAME} = $text_embedding_param,
        c.last_updated = timestamp()
    """
    parameters = {
        "chapter_number_param": chapter_number,
        "text_param": text,
        "raw_llm_output_param": raw_llm_output,
        "summary_param": summary if summary is not None else "",
        "is_provisional_param": is_provisional,
        "end_state_param": json.dumps(end_state) if end_state is not None else "",
        "text_embedding_param": embedding_list,
    }
    try:
        await neo4j_manager.execute_write_query(query, parameters)
        logger.info(
            f"Neo4j: Successfully saved chapter data for chapter {chapter_number}."
        )
    except Exception as e:
        logger.error(
            f"Neo4j: Error saving chapter data for chapter {chapter_number}: {e}",
            exc_info=True,
        )


async def get_chapter_data_from_db(chapter_number: int) -> dict[str, Any] | None:
    """Retrieve stored chapter data for the given chapter number."""

    if chapter_number < 0:
        return None
    query = f"""
    MATCH (c:{settings.NEO4J_VECTOR_NODE_LABEL} {{number: $chapter_number_param}})
    RETURN c.text AS text,
           c.raw_llm_output AS raw_llm_output,
           c.summary AS summary,
           c.is_provisional AS is_provisional,
           c.end_state_json AS end_state_json
    """
    try:
        result = await neo4j_manager.execute_read_query(
            query, {"chapter_number_param": chapter_number}
        )
        if result and result[0]:
            logger.debug(f"Neo4j: Data found for chapter {chapter_number}.")
            return {
                "text": result[0].get("text"),
                "summary": result[0].get("summary"),
                "is_provisional": result[0].get("is_provisional", False),
                "raw_llm_output": result[0].get("raw_llm_output"),
                "end_state_json": result[0].get("end_state_json"),
            }
        logger.debug(f"Neo4j: No data found for chapter {chapter_number}.")
        return None
    except Exception as e:
        logger.error(
            f"Neo4j: Error getting chapter data for {chapter_number}: {e}",
            exc_info=True,
        )
        return None


async def get_embedding_from_db(chapter_number: int) -> np.ndarray | None:
    """Return the text embedding for a chapter if present."""

    if chapter_number < 0:
        return None
    query = f"""
    MATCH (c:{settings.NEO4J_VECTOR_NODE_LABEL} {{number: $chapter_number_param}})
    WHERE c.{settings.NEO4J_VECTOR_PROPERTY_NAME} IS NOT NULL
    RETURN c.{settings.NEO4J_VECTOR_PROPERTY_NAME} AS text_embedding
    """
    try:
        result = await neo4j_manager.execute_read_query(
            query, {"chapter_number_param": chapter_number}
        )
        if result and result[0] and result[0].get("text_embedding"):
            embedding_list = result[0]["text_embedding"]
            return neo4j_manager.list_to_embedding(embedding_list)
        logger.debug(
            f"Neo4j: No embedding vector found on chapter node {chapter_number}."
        )
        return None
    except Exception as e:
        logger.error(
            f"Neo4j: Error getting embedding for {chapter_number}: {e}", exc_info=True
        )
        return None


async def find_similar_chapters_in_db(
    query_embedding: np.ndarray,
    limit: int,
    current_chapter_to_exclude: int | None = None,
) -> list[dict[str, Any]]:
    """Return chapters with embeddings most similar to the query vector.

    Args:
        query_embedding: Embedding representing the search text.
        limit: Maximum number of chapters to return.
        current_chapter_to_exclude: Optional chapter to omit from results.

    Returns:
        A list of chapter metadata dictionaries sorted by similarity.
    """
    if query_embedding is None or query_embedding.size == 0:
        logger.warning(
            "Neo4j: find_similar_chapters_in_db called with empty query_embedding."
        )
        return []

    query_embedding_list = neo4j_manager.embedding_to_list(query_embedding)
    if query_embedding_list is None:
        logger.error(
            "Neo4j: Failed to convert query_embedding to list for similarity search."
        )
        return []

    exclude_clause = ""
    params_dict: dict[str, Any] = {
        "index_name_param": settings.NEO4J_VECTOR_INDEX_NAME,
        "limit_param": limit
        + (0),  # 1 if current_chapter_to_exclude is not None else 0),
        "queryVector_param": query_embedding_list,
    }
    if current_chapter_to_exclude is not None:
        exclude_clause = "WHERE c.number <> $current_chapter_to_exclude_param "
        params_dict["current_chapter_to_exclude_param"] = current_chapter_to_exclude

    similarity_query = f"""
    CALL db.index.vector.queryNodes($index_name_param, $limit_param, $queryVector_param)
    YIELD node AS c, score
    {exclude_clause}
    RETURN c.number AS chapter_number,
           c.summary AS summary,
           c.text AS text,
           c.is_provisional AS is_provisional,
           score
    ORDER BY score DESC
    """
    similar_chapters_data: list[dict[str, Any]] = []
    try:
        results = await neo4j_manager.execute_read_query(similarity_query, params_dict)
        if results:
            for record in results:
                if len(similar_chapters_data) < limit:
                    similar_chapters_data.append(
                        {
                            "chapter_number": record.get("chapter_number"),
                            "summary": record.get("summary"),
                            "text": record.get("text"),
                            "is_provisional": record.get("is_provisional", False),
                            "score": record.get("score"),
                        }
                    )
                else:
                    break
        logger.info(
            f"Neo4j: Vector search found {len(similar_chapters_data)} similar chapters (limit {limit})."
        )
    except Exception as e:
        logger.error(
            f"Neo4j: Error during vector similarity search: {e}", exc_info=True
        )

    return similar_chapters_data
