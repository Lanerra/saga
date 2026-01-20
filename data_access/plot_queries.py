# data_access/plot_queries.py
from typing import Any

import structlog
from async_lru import alru_cache  # type: ignore[import-untyped]
from neo4j.exceptions import Neo4jError

import config
from core.db_manager import neo4j_manager
from core.exceptions import handle_database_error

logger = structlog.get_logger(__name__)


async def save_plot_outline_to_db(plot_data: dict[str, Any]) -> bool:
    """Synchronize the novel plot outline to Neo4j (destructive).

    This function performs a synchronization, not an append-only update. When `plot_data`
    includes a `plot_points` list, it deletes existing `:PlotPoint` nodes that are linked to
    the active novel but are missing from the input set.

    Args:
        plot_data: Plot outline payload. Keys other than `id` and `plot_points` are stored on
            the active `:NovelInfo` node.
            - Primitive values are stored as properties.
            - Lists/dicts are stored as JSON strings under `<key>_json`.

    Returns:
        True when the sync completed successfully, or when `plot_data` is empty (no-op).

    Raises:
        DatabaseError: When a database read/write failure occurs.

    Notes:
        Destructive sync semantics:
            - If `plot_data["plot_points"]` is missing, plot point sync is skipped entirely
              (including deletes) to avoid destructive behavior from partial input.
            - If `plot_data["plot_points"]` is present but not a list, plot point sync is
              also skipped.

        Identity semantics:
            Plot point ids are deterministic and sequence-derived:
            `pp_{novel_id}_{sequence}` where sequence is 1-indexed within the input list.

        Security:
            This function does not interpolate caller-provided strings into Cypher identifiers.
            Dynamic property updates use `SET ni += $primitive_props` and JSON encoding via
            `apoc.convert.toJson(...)`, avoiding unsafe property-key interpolation.
    """
    # NOTE: Despite earlier messaging, this function performs a *synchronization* that may
    # delete PlotPoint nodes missing from the input list. This is destructive by design.
    logger.info("Synchronizing plot outline to Neo4j (destructive sync)...")
    if not plot_data:
        logger.warning("save_plot_outline_to_db: plot_data is empty. No changes will be made.")
        return True

    novel_id = config.MAIN_NOVEL_INFO_NODE_ID
    statements: list[tuple[str, dict[str, Any]]] = []

    primitive_props: dict[str, Any] = {}
    structured_props: dict[str, Any] = {}

    for key, value in plot_data.items():
        if value is None or key == "id" or key == "plot_points":
            continue

        if isinstance(value, list | dict):
            structured_props[key] = value
        else:
            primitive_props[key] = value

    statements.append(
        (
            """
        MERGE (ni:NovelInfo {id: $id_val})
        ON CREATE SET ni.created_ts = timestamp()
        SET ni.updated_ts = timestamp()

        SET ni += $primitive_props

        WITH ni, $structured_props AS structured
        WITH
            ni,
            apoc.map.fromPairs(
                [k IN keys(structured) | [k + "_json", apoc.convert.toJson(structured[k])]]
            ) AS json_props
        SET ni += json_props
        """,
            {"id_val": novel_id, "primitive_props": primitive_props, "structured_props": structured_props},
        )
    )

    input_plot_points_list = plot_data.get("plot_points", None)

    # If plot_points is missing or invalid, we must still persist NovelInfo properties, but
    # we should skip plot point synchronization entirely (including any deletes) to avoid
    # surprising destructive behavior from partial/malformed input.
    if input_plot_points_list is None:
        logger.info("plot_data.plot_points not provided. Skipping plot point sync (NovelInfo properties will still be saved).")
    elif not isinstance(input_plot_points_list, list):
        logger.warning("plot_data.plot_points is not a list. Skipping plot point sync (NovelInfo properties will still be saved).")
        input_plot_points_list = None

    if input_plot_points_list is not None:
        all_input_pp_ids: set[str] = {f"pp_{novel_id}_{i + 1}" for i in range(len(input_plot_points_list))}

        try:
            existing_pp_records = await neo4j_manager.execute_read_query(
                "MATCH (:NovelInfo {id: $novel_id_param})-[:HAS_PLOT_POINT]->(pp:PlotPoint) RETURN pp.id AS id",
                {"novel_id_param": novel_id},
            )
            existing_db_pp_ids: set[str] = {record["id"] for record in existing_pp_records if record and record["id"]}
        except (Neo4jError, KeyError, ValueError) as e:
            logger.error(
                f"Failed to retrieve existing PlotPoint IDs for novel {novel_id}: {e}",
                exc_info=True,
            )
            raise handle_database_error("retrieve existing PlotPoint IDs", e, novel_id=novel_id)

        pp_to_delete = existing_db_pp_ids - all_input_pp_ids
        if pp_to_delete:
            statements.append(
                (
                    """
                MATCH (pp:PlotPoint)
                WHERE pp.id IN $pp_ids_to_delete
                DETACH DELETE pp
                """,
                    {"pp_ids_to_delete": list(pp_to_delete)},
                )
            )

        plot_points_data = []
        for i, point_desc_str_or_dict in enumerate(input_plot_points_list):
            pp_id = f"pp_{novel_id}_{i + 1}"

            pp_props = {
                "id": pp_id,
                "sequence": i + 1,
                "status": "pending",
            }
            if isinstance(point_desc_str_or_dict, str):
                pp_props["description"] = point_desc_str_or_dict
            elif isinstance(point_desc_str_or_dict, dict):
                pp_props["description"] = str(point_desc_str_or_dict.get("description", ""))
                pp_props["status"] = str(point_desc_str_or_dict.get("status", "pending"))
                for k_pp, v_pp in point_desc_str_or_dict.items():
                    if isinstance(v_pp, str | int | float | bool) and k_pp not in pp_props:
                        pp_props[k_pp] = v_pp
            else:
                logger.warning(f"Skipping invalid plot point item at index {i}: {point_desc_str_or_dict}")
                continue

            plot_points_data.append(pp_props)

        if plot_points_data:
            statements.append(
                (
                    """
                UNWIND $plot_points AS pp_data
                MERGE (pp:PlotPoint {id: pp_data.id})
                ON CREATE SET pp = pp_data, pp.created_ts = timestamp()
                ON MATCH SET  pp = pp_data, pp.updated_ts = timestamp()
                """,
                    {"plot_points": plot_points_data},
                )
            )

            statements.append(
                (
                    """
                MATCH (ni:NovelInfo {id: $novel_id})
                UNWIND $plot_point_ids AS pp_id
                MATCH (pp:PlotPoint {id: pp_id})
                MERGE (ni)-[:HAS_PLOT_POINT]->(pp)
                """,
                    {
                        "novel_id": novel_id,
                        "plot_point_ids": [pp["id"] for pp in plot_points_data],
                    },
                )
            )

            sequential_links = [
                {
                    "prev_id": plot_points_data[i - 1]["id"],
                    "curr_id": plot_points_data[i]["id"],
                }
                for i in range(1, len(plot_points_data))
            ]
            if sequential_links:
                statements.append(
                    (
                        """
                    UNWIND $links AS link
                    MATCH (prev_pp:PlotPoint {id: link.prev_id})
                    MATCH (curr_pp:PlotPoint {id: link.curr_id})
                    OPTIONAL MATCH (prev_pp)-[old_next_rel:NEXT_PLOT_POINT]->(:PlotPoint)
                    DELETE old_next_rel
                    MERGE (prev_pp)-[:NEXT_PLOT_POINT]->(curr_pp)
                    """,
                        {"links": sequential_links},
                    )
                )

    try:
        if statements:
            await neo4j_manager.execute_cypher_batch(statements)
        logger.info(f"Successfully synchronized plot outline for novel '{novel_id}' to Neo4j.")

        from data_access.cache_coordinator import clear_plot_read_caches

        clear_plot_read_caches()

        return True
    except (Neo4jError, KeyError, ValueError) as e:
        logger.error(
            f"Error synchronizing plot outline for novel '{novel_id}': {e}",
            exc_info=True,
        )
        raise handle_database_error("synchronize plot outline", e, novel_id=novel_id)


@alru_cache(maxsize=128)
async def get_plot_outline_from_db() -> dict[str, Any]:
    """Return the plot outline for the active novel.

    Returns:
        A dictionary of NovelInfo plot properties with an additional `plot_points` key whose
        value is a list of PlotPoint dicts, ordered by sequence.

        Returns an empty dict when no `:NovelInfo` node exists, when the NovelInfo payload
        is not a map, or when required data is missing.

    Notes:
        Cache semantics:
            This function is cached (read-through). Callers should treat returned data as
            immutable to avoid leaking mutations across cache hits. Write paths should
            invalidate via [`clear_plot_read_caches()`](data_access/cache_coordinator.py).

        JSON decoding contract:
            Properties stored under `<key>_json` are decoded using APOC JSON helpers and
            returned as structured list/dict values. The `_json` keys are removed from the
            returned structure.
    """
    logger.info("Loading decomposed plot outline from Neo4j...")
    novel_id = config.MAIN_NOVEL_INFO_NODE_ID
    plot_data: dict[str, Any] = {}

    novel_info_query = """
    MATCH (ni:NovelInfo {id: $novel_id_param})
    WITH apoc.map.removeKeys(properties(ni), ["id", "created_ts", "updated_ts"]) AS base
    WITH base, [k IN keys(base) WHERE k ENDS WITH "_json"] AS json_keys
    WITH
        apoc.map.removeKeys(base, json_keys) AS primitives,
        apoc.map.fromPairs(
            [
                k IN json_keys |
                [
                    substring(k, 0, size(k) - 5),
                    CASE
                        WHEN base[k] STARTS WITH "[" THEN apoc.convert.fromJsonList(base[k])
                        ELSE apoc.convert.fromJsonMap(base[k])
                    END
                ]
            ]
        ) AS decoded
    RETURN apoc.map.merge(primitives, decoded) AS plot_data
    """
    result_list = await neo4j_manager.execute_read_query(novel_info_query, {"novel_id_param": novel_id})

    if not result_list or not result_list[0] or not result_list[0].get("plot_data"):
        logger.warning(f"No NovelInfo node found with id '{novel_id}'. Returning empty plot outline.")
        return {}

    plot_data = result_list[0]["plot_data"]
    if not isinstance(plot_data, dict):
        logger.warning(f"NovelInfo query returned non-map plot_data for id '{novel_id}'. Returning empty plot outline.")
        return {}

    # Fetch PlotPoints linked to this NovelInfo, ordered by sequence
    plot_points_query = """
    MATCH (ni:NovelInfo {id: $novel_id_param})-[:HAS_PLOT_POINT]->(pp:PlotPoint)
    RETURN pp
    ORDER BY pp.sequence ASC
    """
    pp_results = await neo4j_manager.execute_read_query(plot_points_query, {"novel_id_param": novel_id})

    # Return plot points as structured dicts end-to-end.
    fetched_plot_points: list[dict[str, Any]] = []
    if pp_results:
        for record in pp_results:
            pp_node = record.get("pp")
            if not pp_node:
                continue
            pp_dict = dict(pp_node)
            pp_dict.pop("created_ts", None)
            pp_dict.pop("updated_ts", None)
            fetched_plot_points.append(pp_dict)

    plot_data["plot_points"] = fetched_plot_points

    logger.info(f"Successfully loaded plot outline for novel '{novel_id}'. Plot points: {len(plot_data.get('plot_points', []))}")
    return plot_data


async def append_plot_point(description: str, prev_plot_point_id: str) -> str:
    """Append a new plot point after the most recent plot point.

    Args:
        description: Plot point description text.
        prev_plot_point_id: Plot point id to link from via `NEXT_PLOT_POINT`. When falsy,
            no previous link is created.

    Returns:
        The newly created plot point id.

    Raises:
        DatabaseError: When a database error occurs or when the write query does not return an id.

    Notes:
        Concurrency:
            Sequence/id assignment is performed atomically inside a single write query by
            incrementing a NovelInfo-scoped counter property (`ni.last_plot_point_seq`).
            Neo4j takes a write lock on the NovelInfo node for the transaction, serializing
            concurrent increments and preventing duplicate ids.
    """
    novel_id = config.MAIN_NOVEL_INFO_NODE_ID
    prev_id = prev_plot_point_id or None

    query = """
    MATCH (ni:NovelInfo {id: $novel_id})
    SET ni.last_plot_point_seq = coalesce(ni.last_plot_point_seq, 0) + 1
    WITH ni, ni.last_plot_point_seq AS next_seq
    WITH ni, next_seq, ("pp_" + $novel_id + "_" + toString(next_seq)) AS pp_id
    CREATE (pp:PlotPoint {
        id: pp_id,
        sequence: next_seq,
        description: $desc,
        status: 'pending',
        created_ts: timestamp()
    })
    MERGE (ni)-[:HAS_PLOT_POINT]->(pp)
    WITH ni, pp, pp_id
    OPTIONAL MATCH (ni)-[:HAS_PLOT_POINT]->(prev:PlotPoint {id: $prev_id})
    FOREACH (_ IN CASE WHEN prev IS NULL THEN [] ELSE [1] END |
        OPTIONAL MATCH (prev)-[r:NEXT_PLOT_POINT]->(:PlotPoint)
        DELETE r
        MERGE (prev)-[:NEXT_PLOT_POINT]->(pp)
    )
    RETURN pp_id AS id
    """

    result = await neo4j_manager.execute_write_query(query, {"novel_id": novel_id, "desc": description, "prev_id": prev_id})
    if not result or not result[0] or not result[0].get("id"):
        raise handle_database_error("append plot point",
                                   Exception("No ID returned from plot point creation"),
                                   novel_id=novel_id, description=description)

    from data_access.cache_coordinator import clear_plot_read_caches

    clear_plot_read_caches()

    return result[0]["id"]


async def plot_point_exists(description: str) -> bool:
    """Return whether a plot point with the given description exists.

    Args:
        description: Plot point description to match (case-insensitive).

    Returns:
        True when any plot point under the active novel matches the description.
    """
    novel_id = config.MAIN_NOVEL_INFO_NODE_ID
    query = """
    MATCH (ni:NovelInfo {id: $novel_id})-[:HAS_PLOT_POINT]->(pp:PlotPoint)
    WHERE toLower(pp.description) = toLower($desc)
    RETURN count(pp) AS cnt
    """
    result = await neo4j_manager.execute_read_query(query, {"novel_id": novel_id, "desc": description})
    return bool(result and result[0] and result[0].get("cnt", 0) > 0)


async def get_last_plot_point_id() -> str | None:
    """Return the id of the most recent plot point for the active novel.

    Returns:
        The most recent plot point id by `sequence`, or None when no plot points exist.
    """
    novel_id = config.MAIN_NOVEL_INFO_NODE_ID
    query = """
    MATCH (ni:NovelInfo {id: $novel_id})-[:HAS_PLOT_POINT]->(pp:PlotPoint)
    RETURN pp.id AS id
    ORDER BY pp.sequence DESC
    LIMIT 1
    """
    result = await neo4j_manager.execute_read_query(query, {"novel_id": novel_id})
    return result[0].get("id") if result and result[0] and result[0].get("id") else None
