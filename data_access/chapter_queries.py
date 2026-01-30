# data_access/chapter_queries.py
from typing import Any

import numpy as np
import structlog
from neo4j.exceptions import Neo4jError

import config
from core.db_manager import neo4j_manager
from core.exceptions import handle_database_error

logger = structlog.get_logger(__name__)


def compute_chapter_id(chapter_number: int, *, novel_id: str | None = None) -> str:
    """Compute the canonical, deterministic `Chapter.id` value.

    Args:
        chapter_number: 1-indexed chapter number.
        novel_id: Optional novel identity namespace. Defaults to
            `config.MAIN_NOVEL_INFO_NODE_ID`.

    Returns:
        The deterministic `Chapter.id` string.

    Notes:
        Canonical identity contract:
        - Chapter nodes must have an `id` to satisfy schema constraints (see the unique
          constraint in [`core/db_manager.py`](core/db_manager.py:413)).
        - The id must be stable so different persistence paths converge on the same identity
          semantics.
    """
    novel_id_val = novel_id or config.MAIN_NOVEL_INFO_NODE_ID
    return f"chapter_{novel_id_val}_{int(chapter_number)}"


def build_chapter_upsert_statement(
    *,
    chapter_number: int,
    title: str | None = None,
    act_number: int | None = None,
    summary: str | None = None,
    embedding_vector: list[float] | None = None,
    is_provisional: bool | None = None,
    novel_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build the canonical Chapter upsert Cypher statement and parameter map.

    Args:
        chapter_number: Chapter number (unique).
        title: Chapter title to set. When None, the title field is not modified.
        act_number: Act number to set. When None, the act_number field is not modified.
        summary: Summary to set. When None, the summary field is not modified.
        embedding_vector: Embedding vector to set. When None, the embedding field is not
            modified.
        is_provisional: Provisional flag to set. When None, the provisional field is not
            modified.
        novel_id: Optional novel identity namespace used for deterministic chapter id.

    Returns:
        A `(cypher_query, parameters)` tuple.

    Notes:
        This is the single source of truth for Chapter persistence semantics and must:
        - MERGE Chapter by `number`,
        - always ensure `c.id` is populated (coalesce for existing nodes),
        - avoid clobbering fields when the caller does not provide a value.
    """
    chapter_id = compute_chapter_id(chapter_number, novel_id=novel_id)

    query = """
    MERGE (c:Chapter {number: $chapter_number_param})
    ON CREATE SET
        c.created_ts = timestamp()
    SET
        c.id = coalesce(c.id, $chapter_id_param),
        c.title = coalesce(c.title, $title_param),
        c.act_number = coalesce(c.act_number, $act_number_param),
        c.created_chapter = $chapter_number_param,
        c.updated_ts = timestamp()

    FOREACH (_ IN CASE WHEN $summary_param IS NULL THEN [] ELSE [1] END |
        SET c.summary = $summary_param
    )

    FOREACH (_ IN CASE WHEN $is_provisional_param IS NULL THEN [] ELSE [1] END |
        SET c.is_provisional = $is_provisional_param
    )

    FOREACH (_ IN CASE WHEN $embedding_vector_param IS NULL THEN [] ELSE [1] END |
        SET c.embedding_vector = $embedding_vector_param
    )
    """

    parameters = {
        "chapter_number_param": int(chapter_number),
        "chapter_id_param": chapter_id,
        "title_param": title,
        "act_number_param": act_number,
        "summary_param": summary,
        "is_provisional_param": is_provisional,
        "embedding_vector_param": embedding_vector,
    }

    return query, parameters


async def load_chapter_count_from_db() -> int:
    """Return the number of `:Chapter` nodes in Neo4j.

    Returns:
        The chapter count.

    Raises:
        DatabaseError: When a database error occurs.
    """
    query = "MATCH (c:Chapter) RETURN count(c) AS chapter_count"
    try:
        result = await neo4j_manager.execute_read_query(query)
        count = result[0]["chapter_count"] if result and result[0] else 0
        logger.info(f"Neo4j loaded chapter count: {count}")
        return count
    except (Neo4jError, KeyError, ValueError) as e:
        logger.error(f"Failed to load chapter count from Neo4j: {e}", exc_info=True)
        raise handle_database_error("load chapter count", e)


async def save_chapter_data_to_db(
    chapter_number: int,
    title: str | None = None,
    act_number: int | None = None,
    summary: str | None = None,
    embedding_array: np.ndarray | None = None,
    is_provisional: bool = False,
) -> None:
    """Persist Chapter metadata using canonical Chapter persistence semantics.

    Args:
        chapter_number: Chapter number to persist.
        title: Chapter title to set. When None, title is not modified.
        act_number: Act number to set. When None, act_number is not modified.
        summary: Summary to set. When None, summary is not modified.
        embedding_array: Embedding vector to set. When None, embedding is not modified.
        is_provisional: Whether the chapter should be marked provisional.

    Returns:
        None.

    Notes:
        This function delegates to [`build_chapter_upsert_statement()`](data_access/chapter_queries.py:39)
        so all Chapter writes consistently satisfy schema constraints (including `Chapter.id`).

        Error behavior:
            Invalid `chapter_number` returns early without raising. Neo4j write failures are
            raised as DatabaseError.
    """
    if chapter_number <= 0:
        logger.error(f"Neo4j: Cannot save chapter data for invalid chapter_number: {chapter_number}.")
        return

    embedding_list = neo4j_manager.embedding_to_list(embedding_array)

    query, parameters = build_chapter_upsert_statement(
        chapter_number=chapter_number,
        title=title,
        act_number=act_number,
        summary=summary,
        embedding_vector=embedding_list,
        is_provisional=is_provisional,
    )

    try:
        await neo4j_manager.execute_write_query(query, parameters)
        logger.info(f"Neo4j: Successfully saved chapter data for chapter {chapter_number}.")

        from data_access.cache_coordinator import clear_chapter_read_caches

        clear_chapter_read_caches()

    except (Neo4jError, KeyError, ValueError) as e:
        logger.error(
            f"Neo4j: Error saving chapter data for chapter {chapter_number}: {e}",
            exc_info=True,
        )
        raise handle_database_error("save chapter data", e, chapter_number=chapter_number)


async def get_chapter_data_from_db(chapter_number: int) -> dict[str, Any] | None:
    """Return chapter summary metadata for a single chapter.

    Args:
        chapter_number: Chapter number to fetch.

    Returns:
        A dict with keys:
        - `summary`
        - `is_provisional`

        Returns None when `chapter_number` is invalid or when no chapter exists.

    Raises:
        Exception: Standardized database exceptions via
            [`handle_database_error()`](core/exceptions.py:1) on Neo4j failures.
    """
    if chapter_number <= 0:
        return None
    query = """
    MATCH (c:Chapter {number: $chapter_number_param})
    RETURN c.summary AS summary, c.is_provisional AS is_provisional
    """
    try:
        result = await neo4j_manager.execute_read_query(query, {"chapter_number_param": chapter_number})
        if result and result[0]:
            logger.debug(f"Neo4j: Data found for chapter {chapter_number}.")
            return {
                "summary": result[0].get("summary"),
                "is_provisional": result[0].get("is_provisional", False),
            }
        logger.debug(f"Neo4j: No data found for chapter {chapter_number}.")
        return None
    except (Neo4jError, KeyError, ValueError) as e:
        logger.error(
            f"Neo4j: Error getting chapter data for {chapter_number}: {e}",
            exc_info=True,
        )
        # P1.9: Standardize error handling.
        # Returning None is reserved for "not found" / invalid inputs; DB/runtime errors should
        # be raised so callers can distinguish operational failures from missing data.
        raise handle_database_error(
            "get_chapter_data_from_db",
            e,
            chapter_number=chapter_number,
        ) from e


async def get_embedding_from_db(chapter_number: int) -> np.ndarray | None:
    """Return the embedding vector for a chapter when present.

    Args:
        chapter_number: Chapter number to fetch.

    Returns:
        The embedding as a NumPy array when present. Returns None when:
        - `chapter_number` is invalid
        - the chapter has no stored embedding vector

    Raises:
        DatabaseError: On database errors
    """
    if chapter_number <= 0:
        return None
    query = """
    MATCH (c:Chapter {number: $chapter_number_param})
    WHERE c.embedding_vector IS NOT NULL
    RETURN c.embedding_vector AS embedding_vector
    """
    try:
        result = await neo4j_manager.execute_read_query(query, {"chapter_number_param": chapter_number})
        if result and result[0] and result[0].get("embedding_vector"):
            embedding_list = result[0]["embedding_vector"]
            return neo4j_manager.list_to_embedding(embedding_list)
        logger.debug(f"Neo4j: No embedding vector found on chapter node {chapter_number}.")
        return None
    except (Neo4jError, KeyError, ValueError) as e:
        raise handle_database_error(
            "get_embedding_from_db",
            e,
            chapter_number=chapter_number,
        ) from e


async def find_semantic_context_native(
    query_embedding: np.ndarray,
    current_chapter_number: int,
    limit: int | None = None,
    *,
    include_provisional: bool = False,
) -> list[dict[str, Any]]:
    """Return semantic context chapters using a single optimized vector query.

    Args:
        query_embedding: Query embedding used for similarity search.
        current_chapter_number: Current chapter to exclude from results.
        limit: Maximum number of chapters to return (defaults to `config.CONTEXT_CHAPTER_COUNT`).
        include_provisional: Whether provisional chapters may be returned.

    Returns:
        A list of dictionaries with keys:
        - `chapter_number`
        - `summary`
        - `is_provisional`
        - `score`
        - `context_type`

    Raises:
        DatabaseError: When a database error occurs.

    Returns:
        An empty list when no context exists.

    Notes:
        Ordering and bounds:
            This query enforces deterministic ordering by score descending and applies a
            strict limit. It attempts to include the immediate previous chapter when
            available by boosting its score.

        Error behavior:
            This function logs exceptions and returns an empty list rather than raising.
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
    #
    # P1.8:
    # - Enforce deterministic ordering (score DESC)
    # - Enforce strict limit (<= search_limit)
    # - Still include immediate previous chapter when available (prioritized via boosted score)
    cypher_query = """
    // Vector similarity search for semantic context
    CALL db.index.vector.queryNodes($index_name, $search_limit, $query_vector)
    YIELD node AS similar_c, score
    WHERE similar_c.number < $current_chapter
      AND (
            $include_provisional = false
            OR COALESCE(similar_c.is_provisional, false) = false
          )

    // Ensure deterministic ordering before we collect results.
    WITH similar_c, score
    ORDER BY score DESC

    WITH collect({
        chapter_number: similar_c.number,
        summary: similar_c.summary,
        is_provisional: COALESCE(similar_c.is_provisional, false),
        score: score,
        context_type: 'similarity'
    }) AS similar_results

    // Get immediate previous chapter if not in similarity results
    OPTIONAL MATCH (prev_c:Chapter {number: $prev_chapter_num})
    WHERE prev_c.number < $current_chapter
      AND (
            $include_provisional = false
            OR COALESCE(prev_c.is_provisional, false) = false
          )
      AND NOT ANY(sr IN similar_results WHERE sr.chapter_number = $prev_chapter_num)

    WITH similar_results,
         CASE
           WHEN prev_c IS NOT NULL
           THEN [{
               chapter_number: prev_c.number,
               summary: prev_c.summary,
               is_provisional: COALESCE(prev_c.is_provisional, false),
               // Boost so it stays in the final limited set.
               score: 2.0,
               context_type: 'immediate_previous'
           }]
           ELSE []
         END AS prev_result

    WITH (similar_results + prev_result) AS combined

    UNWIND combined AS item
    WITH item
    ORDER BY item.score DESC

    RETURN collect(item)[..$final_limit] AS context_chapters
    """

    try:
        results = await neo4j_manager.execute_read_query(
            cypher_query,
            {
                "index_name": config.NEO4J_VECTOR_INDEX_NAME,
                # Use a small buffer, then apply strict limiting in Cypher.
                "search_limit": search_limit + 5,
                "final_limit": search_limit,
                "query_vector": query_embedding_list,
                "current_chapter": current_chapter_number,
                "prev_chapter_num": prev_chapter_num,
                "include_provisional": include_provisional,
            },
        )

        if not results or not results[0]:
            logger.info(f"No semantic context found for chapter {current_chapter_number}")
            return []

        context_chapters = results[0]["context_chapters"]

        # P1.8: Defensive contract enforcement (even if Cypher returns an overage).
        # Keep ordering deterministic (score DESC) and enforce strict limit (<= search_limit).
        if isinstance(context_chapters, list):
            context_chapters = sorted(
                context_chapters,
                key=lambda c: float(c.get("score", 0.0)) if isinstance(c, dict) else 0.0,
                reverse=True,
            )[:search_limit]

        logger.info(f"Native context search found {len(context_chapters)} chapters for chapter {current_chapter_number}")

        return context_chapters

    except (Neo4jError, KeyError, ValueError) as e:
        logger.error(f"Error in native semantic context search: {e}", exc_info=True)
        raise handle_database_error("find semantic context", e, current_chapter_number=current_chapter_number, limit=limit)


async def get_chapter_content_batch_native(
    chapter_numbers: list[int],
) -> dict[int, dict[str, Any]]:
    """Return chapter content for a list of chapter numbers in a single query.

    Args:
        chapter_numbers: Chapter numbers to retrieve.

    Returns:
        A mapping from chapter number to a dict with keys:
        - `summary`
        - `is_provisional`

    Raises:
        DatabaseError: When a database error occurs.

    Returns:
        An empty dict when no chapter numbers are provided.
    """
    if not chapter_numbers:
        return {}

    # Single query to get all requested chapters
    cypher_query = """
    MATCH (c:Chapter)
    WHERE c.number IN $chapter_numbers
    RETURN c.number AS chapter_number,
           c.summary AS summary,
           c.is_provisional AS is_provisional
    ORDER BY c.number
    """

    try:
        results = await neo4j_manager.execute_read_query(cypher_query, {"chapter_numbers": chapter_numbers})

        chapter_data = {}
        for record in results:
            chapter_num = record["chapter_number"]
            chapter_data[chapter_num] = {
                "summary": record.get("summary"),
                "is_provisional": record.get("is_provisional", False),
            }

        logger.debug(f"Native batch retrieval got {len(chapter_data)} chapters of {len(chapter_numbers)} requested")

        return chapter_data

    except (Neo4jError, KeyError, ValueError) as e:
        logger.error(f"Error in native chapter batch retrieval: {e}", exc_info=True)
        raise handle_database_error("get chapter content batch", e, chapter_numbers=chapter_numbers)
