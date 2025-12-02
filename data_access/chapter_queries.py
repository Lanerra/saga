# data_access/chapter_queries.py
from typing import Any

import numpy as np
import structlog

import config
from core.db_manager import neo4j_manager

logger = structlog.get_logger(__name__)


async def load_chapter_count_from_db() -> int:
    query = "MATCH (c:Chapter) RETURN count(c) AS chapter_count"
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
) -> None:
    if chapter_number <= 0:
        logger.error(
            f"Neo4j: Cannot save chapter data for invalid chapter_number: {chapter_number}."
        )
        return

    embedding_list = neo4j_manager.embedding_to_list(embedding_array)

    query = """
    MERGE (c:Chapter {number: $chapter_number_param})
    SET c.text = $text_param,
        c.raw_llm_output = $raw_llm_output_param,
        c.summary = $summary_param,
        c.is_provisional = $is_provisional_param,
        c.embedding_vector = $embedding_vector_param,
        c.last_updated = timestamp()
    """
    parameters = {
        "chapter_number_param": chapter_number,
        "text_param": text,
        "raw_llm_output_param": raw_llm_output,
        "summary_param": summary if summary is not None else "",
        "is_provisional_param": is_provisional,
        "embedding_vector_param": embedding_list,
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
    if chapter_number <= 0:
        return None
    query = """
    MATCH (c:Chapter {number: $chapter_number_param})
    RETURN c.text AS text, c.raw_llm_output AS raw_llm_output, c.summary AS summary, c.is_provisional AS is_provisional
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
    if chapter_number <= 0:
        return None
    query = """
    MATCH (c:Chapter {number: $chapter_number_param})
    WHERE c.embedding_vector IS NOT NULL
    RETURN c.embedding_vector AS embedding_vector
    """
    try:
        result = await neo4j_manager.execute_read_query(
            query, {"chapter_number_param": chapter_number}
        )
        if result and result[0] and result[0].get("embedding_vector"):
            embedding_list = result[0]["embedding_vector"]
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

    # We want to exclude the current chapter being processed and only look at past chapters
    # current_chapter_to_exclude: the chapter number to exclude (e.g., the current chapter being generated)

    exclude_clause = ""
    params_dict: dict[str, Any] = {
        "index_name_param": config.NEO4J_VECTOR_INDEX_NAME,
        "limit_param": limit + (1 if current_chapter_to_exclude is not None else 0),
        "queryVector_param": query_embedding_list,
    }

    if current_chapter_to_exclude is not None:
        # Exclude the specific chapter and only consider chapters before it
        exclude_clause = "WHERE c.number <> $current_chapter_to_exclude_param AND c.number < $current_chapter_to_exclude_param"
        params_dict["current_chapter_to_exclude_param"] = current_chapter_to_exclude
    else:
        # No specific chapter to exclude, but still only consider past chapters
        # Using a large number as default to include all existing chapters
        exclude_clause = "WHERE c.number < $current_chapter_upper_bound_param"
        params_dict["current_chapter_upper_bound_param"] = 1000000

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
                if (
                    current_chapter_to_exclude is not None
                    and record.get("chapter_number") == current_chapter_to_exclude
                ):
                    continue
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


# Native context functions for performance optimization
async def find_semantic_context_native(
    query_embedding: np.ndarray,
    current_chapter_number: int,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Native version of semantic context retrieval using single optimized query.
    Eliminates multiple DB calls and reduces serialization overhead.

    Args:
        query_embedding: NumPy embedding for similarity search
        current_chapter_number: Current chapter to exclude from results
        limit: Maximum number of chapters to return

    Returns:
        List of chapter context data including similar + immediate previous
    """
    if query_embedding is None or query_embedding.size == 0:
        logger.warning("Native context search called with empty query embedding")
        return []

    query_embedding_list = neo4j_manager.embedding_to_list(query_embedding)
    if not query_embedding_list:
        logger.error("Failed to convert query embedding for native context search")
        return []

    search_limit = limit or config.CONTEXT_CHAPTER_COUNT
    prev_chapter_num = current_chapter_number - 1

    # Single comprehensive query combining similarity search + immediate previous
    cypher_query = """
    // Vector similarity search for semantic context
    CALL db.index.vector.queryNodes($index_name, $search_limit, $query_vector)
    YIELD node AS similar_c, score
    WHERE similar_c.number < $current_chapter
    
    WITH collect({
        chapter_number: similar_c.number,
        summary: similar_c.summary,
        text: similar_c.text,
        is_provisional: COALESCE(similar_c.is_provisional, false),
        score: score,
        context_type: 'similarity'
    }) AS similar_results
    
    // Get immediate previous chapter if not in similarity results
    OPTIONAL MATCH (prev_c:Chapter {number: $prev_chapter_num})
    WHERE prev_c.number < $current_chapter
      AND NOT ANY(sr IN similar_results WHERE sr.chapter_number = $prev_chapter_num)
    
    WITH similar_results,
         CASE 
           WHEN prev_c IS NOT NULL 
           THEN [{
               chapter_number: prev_c.number,
               summary: prev_c.summary,
               text: prev_c.text,
               is_provisional: COALESCE(prev_c.is_provisional, false),
               score: 0.999,
               context_type: 'immediate_previous'
           }]
           ELSE []
         END AS prev_result
    
    RETURN similar_results + prev_result AS context_chapters
    """

    try:
        results = await neo4j_manager.execute_read_query(
            cypher_query,
            {
                "index_name": config.NEO4J_VECTOR_INDEX_NAME,
                "search_limit": search_limit + 1,  # Buffer for potential duplicates
                "query_vector": query_embedding_list,
                "current_chapter": current_chapter_number,
                "prev_chapter_num": prev_chapter_num,
            },
        )

        if not results or not results[0]:
            logger.info(
                f"No semantic context found for chapter {current_chapter_number}"
            )
            return []

        context_chapters = results[0]["context_chapters"]

        logger.info(
            f"Native context search found {len(context_chapters)} chapters for chapter {current_chapter_number}"
        )

        return context_chapters

    except Exception as e:
        logger.error(f"Error in native semantic context search: {e}", exc_info=True)
        return []


async def get_chapter_content_batch_native(
    chapter_numbers: list[int],
) -> dict[int, dict[str, Any]]:
    """
    Native batch retrieval of chapter content.
    Optimized for minimal serialization with single query.

    Args:
        chapter_numbers: List of chapter numbers to retrieve

    Returns:
        Dict mapping chapter numbers to their content data
    """
    if not chapter_numbers:
        return {}

    # Single query to get all requested chapters
    cypher_query = """
    MATCH (c:Chapter)
    WHERE c.number IN $chapter_numbers
    RETURN c.number AS chapter_number,
           c.summary AS summary,
           c.text AS text,
           c.is_provisional AS is_provisional
    ORDER BY c.number
    """

    try:
        results = await neo4j_manager.execute_read_query(
            cypher_query, {"chapter_numbers": chapter_numbers}
        )

        chapter_data = {}
        for record in results:
            chapter_num = record["chapter_number"]
            chapter_data[chapter_num] = {
                "summary": record.get("summary"),
                "text": record.get("text"),
                "is_provisional": record.get("is_provisional", False),
            }

        logger.debug(
            f"Native batch retrieval got {len(chapter_data)} chapters of {len(chapter_numbers)} requested"
        )

        return chapter_data

    except Exception as e:
        logger.error(f"Error in native chapter batch retrieval: {e}", exc_info=True)
        return {}
