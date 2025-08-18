# data_access/chapter_queries.py
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from core.db_manager import neo4j_manager

logger = logging.getLogger(__name__)


async def load_chapter_count_from_db() -> int:
    query = (
        f"MATCH (c:{config.NEO4J_VECTOR_NODE_LABEL}) RETURN count(c) AS chapter_count"
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
    summary: Optional[str],
    embedding_array: Optional[np.ndarray],
    is_provisional: bool = False,
):
    if chapter_number <= 0:
        logger.error(
            f"Neo4j: Cannot save chapter data for invalid chapter_number: {chapter_number}."
        )
        return

    embedding_list = neo4j_manager.embedding_to_list(embedding_array)

    query = f"""
    MERGE (c:{config.NEO4J_VECTOR_NODE_LABEL} {{number: $chapter_number_param}})
    SET c.text = $text_param,
        c.raw_llm_output = $raw_llm_output_param,
        c.summary = $summary_param,
        c.is_provisional = $is_provisional_param,
        c.{config.NEO4J_VECTOR_PROPERTY_NAME} = $embedding_vector_param,
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


async def get_chapter_data_from_db(chapter_number: int) -> Optional[Dict[str, Any]]:
    if chapter_number <= 0:
        return None
    query = f"""
    MATCH (c:{config.NEO4J_VECTOR_NODE_LABEL} {{number: $chapter_number_param}})
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


async def get_embedding_from_db(chapter_number: int) -> Optional[np.ndarray]:
    if chapter_number <= 0:
        return None
    query = f"""
    MATCH (c:{config.NEO4J_VECTOR_NODE_LABEL} {{number: $chapter_number_param}})
    WHERE c.{config.NEO4J_VECTOR_PROPERTY_NAME} IS NOT NULL
    RETURN c.{config.NEO4J_VECTOR_PROPERTY_NAME} AS embedding_vector
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
    current_chapter_to_exclude: Optional[int] = None,
) -> List[Dict[str, Any]]:
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
    params_dict: Dict[str, Any] = {
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
    similar_chapters_data: List[Dict[str, Any]] = []
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


async def get_all_past_embeddings_from_db(
    current_chapter_number: int,
) -> List[Tuple[int, np.ndarray]]:
    logger.warning(
        "get_all_past_embeddings_from_db is deprecated. Use find_similar_chapters_in_db for semantic context."
    )
    embeddings_list: List[Tuple[int, np.ndarray]] = []
    query = f"""
    MATCH (c:{config.NEO4J_VECTOR_NODE_LABEL})
    WHERE c.number < $current_chapter_number_param AND c.number > 0
      AND c.{config.NEO4J_VECTOR_PROPERTY_NAME} IS NOT NULL
    RETURN c.number AS chapter_number, c.{config.NEO4J_VECTOR_PROPERTY_NAME} AS embedding_vector
    ORDER BY c.number DESC
    """
    try:
        results = await neo4j_manager.execute_read_query(
            query, {"current_chapter_number_param": current_chapter_number}
        )
        if results:
            for record in results:
                if record.get("embedding_vector"):
                    deserialized_emb = neo4j_manager.list_to_embedding(
                        record["embedding_vector"]
                    )
                    if deserialized_emb is not None:
                        embeddings_list.append(
                            (record["chapter_number"], deserialized_emb)
                        )
        logger.info(
            f"Neo4j (Deprecated Call): Retrieved {len(embeddings_list)} past embeddings for context before chapter {current_chapter_number}."
        )
        return embeddings_list
    except Exception as e:
        logger.error(
            f"Neo4j (Deprecated Call): Error getting all past embeddings: {e}",
            exc_info=True,
        )
        return []
