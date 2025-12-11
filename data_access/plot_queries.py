# data_access/plot_queries.py
from typing import Any

import structlog

import config
from core.db_manager import neo4j_manager

logger = structlog.get_logger(__name__)


async def save_plot_outline_to_db(plot_data: dict[str, Any]) -> bool:
    logger.info("Synchronizing plot outline to Neo4j (non-destructive)...")
    if not plot_data:
        logger.warning(
            "save_plot_outline_to_db: plot_data is empty. No changes will be made."
        )
        return True

    novel_id = config.MAIN_NOVEL_INFO_NODE_ID
    statements: list[tuple[str, dict[str, Any]]] = []

    novel_props_for_set = {
        k: v
        for k, v in plot_data.items()
        if not isinstance(v, list | dict) and v is not None and k != "id"
    }
    novel_props_for_set["id"] = novel_id

    statements.append(
        (
            """
        MERGE (ni:NovelInfo {id: $id_val})
        ON CREATE SET ni = $props, ni.created_ts = timestamp()
        ON MATCH SET  ni = $props, ni.updated_ts = timestamp()
        """,
            {"id_val": novel_id, "props": novel_props_for_set},
        )
    )

    input_plot_points_list = plot_data.get("plot_points", [])
    if not isinstance(input_plot_points_list, list):
        logger.warning("plot_data.plot_points is not a list. Skipping plot point sync.")
        return True

    all_input_pp_ids: set[str] = {
        f"pp_{novel_id}_{i + 1}" for i in range(len(input_plot_points_list))
    }

    try:
        existing_pp_records = await neo4j_manager.execute_read_query(
            "MATCH (:NovelInfo {id: $novel_id_param})-[:HAS_PLOT_POINT]->(pp:PlotPoint) RETURN pp.id AS id",
            {"novel_id_param": novel_id},
        )
        existing_db_pp_ids: set[str] = {
            record["id"] for record in existing_pp_records if record and record["id"]
        }
    except Exception as e:
        logger.error(
            f"Failed to retrieve existing PlotPoint IDs for novel {novel_id}: {e}",
            exc_info=True,
        )
        return False

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
            pp_props["description"] = str(
                point_desc_str_or_dict.get("description", "")
            )
            pp_props["status"] = str(
                point_desc_str_or_dict.get("status", "pending")
            )
            for k_pp, v_pp in point_desc_str_or_dict.items():
                if (
                    isinstance(v_pp, str | int | float | bool)
                    and k_pp not in pp_props
                ):
                    pp_props[k_pp] = v_pp
        else:
            logger.warning(
                f"Skipping invalid plot point item at index {i}: {point_desc_str_or_dict}"
            )
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
            {"prev_id": plot_points_data[i - 1]["id"], "curr_id": plot_points_data[i]["id"]}
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
        logger.info(
            f"Successfully synchronized plot outline for novel '{novel_id}' to Neo4j."
        )
        return True
    except Exception as e:
        logger.error(
            f"Error synchronizing plot outline for novel '{novel_id}': {e}",
            exc_info=True,
        )
        return False


async def get_plot_outline_from_db() -> dict[str, Any]:
    logger.info("Loading decomposed plot outline from Neo4j...")
    novel_id = config.MAIN_NOVEL_INFO_NODE_ID
    plot_data: dict[str, Any] = {}

    # Fetch NovelInfo node properties
    novel_info_query = "MATCH (ni:NovelInfo {id: $novel_id_param}) RETURN ni"
    result_list = await neo4j_manager.execute_read_query(
        novel_info_query, {"novel_id_param": novel_id}
    )

    if not result_list or not result_list[0] or not result_list[0].get("ni"):
        logger.warning(
            f"No NovelInfo node found with id '{novel_id}'. Returning empty plot outline."
        )
        return {}

    novel_node = result_list[0]["ni"]
    plot_data.update(dict(novel_node))
    plot_data.pop("id", None)  # Remove internal DB ID from returned dict
    plot_data.pop("created_ts", None)
    plot_data.pop("updated_ts", None)

    # Fetch PlotPoints linked to this NovelInfo, ordered by sequence
    plot_points_query = """
    MATCH (ni:NovelInfo {id: $novel_id_param})-[:HAS_PLOT_POINT]->(pp:PlotPoint)
    RETURN pp
    ORDER BY pp.sequence ASC
    """
    pp_results = await neo4j_manager.execute_read_query(
        plot_points_query, {"novel_id_param": novel_id}
    )

    fetched_plot_points = []
    if pp_results:
        for record in pp_results:
            pp_node = record.get("pp")
            if pp_node:
                fetched_plot_points.append(
                    pp_node.get(
                        "description",
                        f"Plot Point {pp_node.get('sequence')}: Desc missing",
                    )
                )
    plot_data["plot_points"] = fetched_plot_points

    logger.info(
        f"Successfully loaded plot outline for novel '{novel_id}'. Plot points: {len(plot_data.get('plot_points', []))}"
    )
    return plot_data


async def append_plot_point(description: str, prev_plot_point_id: str) -> str:
    """Append a new PlotPoint node linked to NovelInfo and previous PlotPoint."""
    novel_id = config.MAIN_NOVEL_INFO_NODE_ID
    # Determine next sequence number
    query = (
        "MATCH (:NovelInfo {id: $novel_id})-[:HAS_PLOT_POINT]->(pp:PlotPoint) "
        "RETURN coalesce(max(pp.sequence), 0) AS max_seq"
    )
    result = await neo4j_manager.execute_read_query(query, {"novel_id": novel_id})
    max_seq = result[0].get("max_seq") if result else 0
    next_seq = (max_seq or 0) + 1
    pp_id = f"pp_{novel_id}_{next_seq}"

    statements: list[tuple[str, dict[str, Any]]] = [
        (
            """
        MERGE (pp:PlotPoint {id: $pp_id})
        ON CREATE SET pp.description = $desc, pp.sequence = $seq, pp.status = 'pending', pp.created_ts = timestamp()
        ON MATCH SET  pp.description = $desc, pp.sequence = $seq, pp.updated_ts = timestamp()
        """,
            {"pp_id": pp_id, "desc": description, "seq": next_seq},
        ),
        (
            """
        MATCH (ni:NovelInfo {id: $novel_id})
        MATCH (pp:PlotPoint {id: $pp_id})
        MERGE (ni)-[:HAS_PLOT_POINT]->(pp)
        """,
            {"novel_id": novel_id, "pp_id": pp_id},
        ),
    ]

    if prev_plot_point_id:
        statements.append(
            (
                """
            MATCH (prev:PlotPoint {id: $prev_id})
            MATCH (curr:PlotPoint {id: $pp_id})
            OPTIONAL MATCH (prev)-[r:NEXT_PLOT_POINT]->(:PlotPoint)
            DELETE r
            MERGE (prev)-[:NEXT_PLOT_POINT]->(curr)
            """,
                {"prev_id": prev_plot_point_id, "pp_id": pp_id},
            )
        )

    await neo4j_manager.execute_cypher_batch(statements)
    return pp_id


async def plot_point_exists(description: str) -> bool:
    """Check if a plot point with the given description exists."""
    query = """
    MATCH (pp:PlotPoint)
    WHERE toLower(pp.description) = toLower($desc)
    RETURN count(pp) AS cnt
    """
    result = await neo4j_manager.execute_read_query(query, {"desc": description})
    return bool(result and result[0] and result[0].get("cnt", 0) > 0)


async def get_last_plot_point_id() -> str | None:
    """Return the ID of the most recent PlotPoint."""
    query = """
    MATCH (pp:PlotPoint)
    RETURN pp.id AS id
    ORDER BY pp.sequence DESC
    LIMIT 1
    """
    result = await neo4j_manager.execute_read_query(query)
    return result[0].get("id") if result and result[0] else None
