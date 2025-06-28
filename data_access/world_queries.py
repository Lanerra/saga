# data_access/world_queries.py
from typing import Any

import structlog
import utils
from async_lru import alru_cache  # type: ignore
from config import settings
from core.db_manager import neo4j_manager
from kg_constants import (
    KG_IS_PROVISIONAL,
    KG_NODE_CHAPTER_UPDATED,
    KG_NODE_CREATED_CHAPTER,
)
from kg_maintainer.models import WorldItem
from utils import kg_property_keys as kg_keys

from .cypher_builders.world_cypher import generate_world_element_node_cypher

logger = structlog.get_logger(__name__)

# Mapping from normalized world item names to canonical IDs
WORLD_NAME_TO_ID: dict[str, str] = {}


def resolve_world_name(name: str) -> str | None:
    """Return canonical world item ID for a display name if known."""
    if not name:
        return None
    return WORLD_NAME_TO_ID.get(utils._normalize_for_id(name))


def get_world_item_by_name(
    world_data: dict[str, dict[str, WorldItem]], name: str
) -> WorldItem | None:
    """Retrieve a WorldItem from cached data using a fuzzy name lookup."""
    item_id = resolve_world_name(name)
    if not item_id:
        return None
    for items in world_data.values():
        if not isinstance(items, dict):
            continue
        for item in items.values():
            if isinstance(item, WorldItem) and item.id == item_id:
                return item
    return None


async def sync_world_items(
    world_items: dict[str, dict[str, WorldItem]],
    chapter_number: int,
    full_sync: bool = False,
) -> bool:
    """Persist world element data to Neo4j."""
    WORLD_NAME_TO_ID.clear()
    for _cat, items in world_items.items():
        if not isinstance(items, dict):
            continue
        for item in items.values():
            if isinstance(item, WorldItem):
                WORLD_NAME_TO_ID[utils._normalize_for_id(item.name)] = item.id
    if full_sync:
        world_dict = {
            cat: {name: item.to_dict() for name, item in items.items()}
            for cat, items in world_items.items()
        }
        return await sync_full_state_from_object_to_db(world_dict)

    statements: list[tuple[str, dict[str, Any]]] = []
    count = 0
    for category_items in world_items.values():
        if not isinstance(category_items, dict):
            continue
        for item_obj in category_items.values():
            statements.extend(
                generate_world_element_node_cypher(item_obj, chapter_number)
            )
            count += 1

    try:
        if statements:
            await neo4j_manager.execute_cypher_batch(statements)
        logger.info(
            "Persisted %d world element updates for chapter %d.",
            count,
            chapter_number,
        )
        return True
    except Exception as exc:  # pragma: no cover - log and return failure
        logger.error(
            "Error persisting world element updates for chapter %d: %s",
            chapter_number,
            exc,
            exc_info=True,
        )
        return False


async def sync_full_state_from_object_to_db(world_data: dict[str, Any]) -> bool:
    """Persist the entire world-building state to Neo4j."""

    logger.info("Synchronizing world building data to Neo4j (non-destructive)...")

    novel_id_param = settings.MAIN_NOVEL_INFO_NODE_ID
    wc_id_param = (
        settings.MAIN_WORLD_CONTAINER_NODE_ID
    )  # Unique ID for the WorldContainer
    statements: list[tuple[str, dict[str, Any]]] = []

    # 1. Synchronize WorldContainer (_overview_)
    overview_details = world_data.get("_overview_", {})
    if isinstance(overview_details, dict):
        wc_props = {
            "id": wc_id_param,  # Ensure ID is part of props for SET
            "overview_description": str(overview_details.get("description", "")),
            KG_IS_PROVISIONAL: overview_details.get(
                kg_keys.source_quality_key(settings.KG_PREPOPULATION_CHAPTER_NUM)
            )
            == "provisional_from_unrevised_draft",
        }
        # Add other direct properties from overview_details if any
        for k_overview, v_overview in overview_details.items():
            if (
                isinstance(v_overview, str | int | float | bool)
                and k_overview not in wc_props
            ):
                wc_props[k_overview] = v_overview

        statements.append(
            (
                """
            MERGE (wc:Entity {id: $id_val})
            ON CREATE SET wc:WorldContainer, wc = $props, wc.created_ts = timestamp()
            ON MATCH SET  wc:WorldContainer, wc = $props, wc.updated_ts = timestamp()
            """,
                {"id_val": wc_id_param, "props": wc_props},
            )
        )
        # Link WorldContainer to NovelInfo
        statements.append(
            (
                """
            MATCH (ni:NovelInfo:Entity {id: $novel_id_val})
            MATCH (wc:WorldContainer:Entity {id: $wc_id_val})
            MERGE (ni)-[:HAS_WORLD_META]->(wc)
            """,
                {"novel_id_val": novel_id_param, "wc_id_val": wc_id_param},
            )
        )

    # 2. Collect all WorldElement IDs from input data
    all_input_we_ids: set[str] = set()
    for category_str, items_dict_value in world_data.items():
        if category_str == "_overview_" or not isinstance(items_dict_value, dict):
            continue
        for (
            item_name_str,
            item_details_value,
        ) in items_dict_value.items():  # Iterate through items in the category
            # Ensure item_name_str itself is not a reserved key
            if item_name_str.startswith(
                ("_", kg_keys.SOURCE_QUALITY_PREFIX, "category_updated_in_chapter_")
            ):
                continue

            # Use the 'id' from item_details_value if present and valid, otherwise generate.
            # This aligns with how WorldItem.from_dict handles ID.
            we_id_str = ""
            if (
                isinstance(item_details_value, dict)
                and isinstance(item_details_value.get("id"), str)
                and item_details_value.get("id").strip()
            ):
                we_id_str = item_details_value.get("id")
            else:  # Fallback to old generation for consistency if 'id' isn't in the dict from DB.
                # WorldItem.from_dict ensures 'id' is always there for objects from parsing.
                # For data from DB via get_world_building_from_db, 'id' is popped but needs to be reconstructed for this check.
                norm_cat = utils._normalize_for_id(category_str)
                norm_name = utils._normalize_for_id(item_name_str)
                if not norm_cat:
                    norm_cat = "unknown_category"
                if not norm_name:
                    norm_name = "unknown_name"
                we_id_str = f"{norm_cat}_{norm_name}"

            if we_id_str:
                all_input_we_ids.add(we_id_str)

    # 3. Get existing WorldElement IDs from DB to find orphans
    try:
        existing_we_records = await neo4j_manager.execute_read_query(
            "MATCH (we:WorldElement:Entity)"
            " WHERE we.is_deleted IS NULL OR we.is_deleted = FALSE"
            " RETURN we.id AS id"
        )
        existing_db_we_ids: set[str] = {
            record["id"] for record in existing_we_records if record and record["id"]
        }
    except Exception as e:
        logger.error(
            f"Failed to retrieve existing WorldElement IDs from DB: {e}", exc_info=True
        )
        return False

    # WorldElements to delete (in DB but not in input world_data)
    we_to_delete = existing_db_we_ids - all_input_we_ids
    if we_to_delete:
        statements.append(
            (
                """
            MATCH (we:WorldElement:Entity)
            WHERE we.id IN $we_ids_to_delete
            SET we.is_deleted = TRUE
            """,
                {"we_ids_to_delete": list(we_to_delete)},
            )
        )

    # 4. Process each WorldElement from input data
    for category_str, items_category_dict in world_data.items():
        if category_str == "_overview_" or not isinstance(items_category_dict, dict):
            continue

        for item_name_str, details_dict in items_category_dict.items():
            if not isinstance(details_dict, dict) or item_name_str.startswith(
                ("_", kg_keys.SOURCE_QUALITY_PREFIX, "category_updated_in_chapter_")
            ):
                continue

            # ID should be taken from details_dict if present, otherwise generated.
            # This aligns with WorldItem.from_dict's ID handling.
            we_id_str = ""
            if (
                isinstance(details_dict.get("id"), str)
                and details_dict.get("id").strip()
            ):
                we_id_str = details_dict.get("id")
            else:  # Fallback for safety or if 'id' was somehow removed before this point
                norm_cat = utils._normalize_for_id(category_str)
                norm_name = utils._normalize_for_id(item_name_str)
                if not norm_cat:
                    norm_cat = "unknown_category"
                if not norm_name:
                    norm_name = "unknown_name"
                we_id_str = f"{norm_cat}_{norm_name}"

            # Prepare WorldElement properties
            we_node_props = {
                "id": we_id_str,
                "name": item_name_str,  # This is the display name
                "category": category_str,  # This is the display category
            }

            created_chap_num = details_dict.get(
                KG_NODE_CREATED_CHAPTER,  # Check direct KG constant key first
                details_dict.get(
                    "created_chapter",
                    settings.KG_PREPOPULATION_CHAPTER_NUM,
                ),
            )  # Fallback

            we_node_props[KG_NODE_CREATED_CHAPTER] = int(created_chap_num)

            # Provisional status: check specific source_quality_chapter_X, then KG_IS_PROVISIONAL, then 'is_provisional'
            is_prov = False
            sq_key_for_created_chap = kg_keys.source_quality_key(
                we_node_props[KG_NODE_CREATED_CHAPTER]
            )
            if (
                details_dict.get(sq_key_for_created_chap)
                == "provisional_from_unrevised_draft"
            ):
                is_prov = True
            elif (
                details_dict.get(KG_IS_PROVISIONAL) is True
            ):  # Check direct KG constant key
                is_prov = True
            elif (
                details_dict.get("is_provisional") is True
            ):  # Fallback to 'is_provisional'
                is_prov = True
            we_node_props[KG_IS_PROVISIONAL] = is_prov
            we_node_props["is_deleted"] = False

            # Add other direct properties
            for k_detail, v_detail in details_dict.items():
                if (
                    isinstance(v_detail, str | int | float | bool)
                    and k_detail not in we_node_props
                    and not k_detail.startswith(kg_keys.ELABORATION_PREFIX)
                    and not k_detail.startswith(kg_keys.ADDED_PREFIX)
                    and not k_detail.startswith(kg_keys.SOURCE_QUALITY_PREFIX)
                    and k_detail
                    not in [
                        "goals",
                        "rules",
                        "key_elements",
                        "traits",
                        "id",
                        "name",
                        "category",
                        "created_chapter",
                        "is_provisional",
                    ]
                ):  # Exclude already handled
                    we_node_props[k_detail] = v_detail

            # MERGE WorldElement node
            statements.append(
                (
                    """
                MERGE (we:Entity {id: $id_val})
                ON CREATE SET we:WorldElement, we = $props, we.created_ts = timestamp()
                ON MATCH SET  we:WorldElement, we += $props, we.updated_ts = timestamp()
                """,
                    {"id_val": we_id_str, "props": we_node_props},
                )
            )
            # Link WorldElement to WorldContainer
            statements.append(
                (
                    """
                MATCH (wc:WorldContainer:Entity {id: $wc_id_val})
                MATCH (we:WorldElement:Entity {id: $we_id_val})
                MERGE (wc)-[:CONTAINS_ELEMENT]->(we)
                """,
                    {"wc_id_val": wc_id_param, "we_id_val": we_id_str},
                )
            )

            # Reconcile list properties (goals, rules, key_elements, traits) as ValueNode relationships
            list_prop_map = {
                "goals": "HAS_GOAL",
                "rules": "HAS_RULE",
                "key_elements": "HAS_KEY_ELEMENT",
                "traits": "HAS_TRAIT_ASPECT",
            }
            for list_prop_key, rel_name_internal in list_prop_map.items():
                current_prop_values: set[str] = {
                    str(v).strip()
                    for v in details_dict.get(list_prop_key, [])
                    if isinstance(v, str) and str(v).strip()
                }

                # Delete relationships to ValueNodes no longer in the list
                statements.append(
                    (
                        f"""
                    MATCH (we:WorldElement:Entity {{id: $we_id_val}})-[r:{rel_name_internal}]->(v:ValueNode:Entity {{type: $value_node_type}})
                    WHERE NOT v.value IN $current_values_list
                    DELETE r
                    """,
                        {
                            "we_id_val": we_id_str,
                            "value_node_type": list_prop_key,
                            "current_values_list": list(current_prop_values),
                        },
                    )
                )
                # Add/Ensure relationships for current values
                if current_prop_values:
                    statements.append(
                        (
                            f"""
                        MATCH (we:WorldElement:Entity {{id: $we_id_val}})
                        UNWIND $current_values_list AS item_value_str
                        MERGE (v:Entity:ValueNode {{value: item_value_str, type: $value_node_type}})
                           ON CREATE SET v.created_ts = timestamp()
                        MERGE (we)-[:{rel_name_internal}]->(v)
                        """,
                            {
                                "we_id_val": we_id_str,
                                "value_node_type": list_prop_key,
                                "current_values_list": list(current_prop_values),
                            },
                        )
                    )

            # Reconcile WorldElaborationEvents
            statements.append(
                (
                    """
                MATCH (we:WorldElement:Entity {id: $we_id_val})-[r:ELABORATED_IN_CHAPTER]->(elab:WorldElaborationEvent:Entity)
                DETACH DELETE elab, r
                """,
                    {"we_id_val": we_id_str},
                )
            )
            for key_str, value_val in details_dict.items():
                if (
                    key_str.startswith(kg_keys.ELABORATION_PREFIX)
                    and isinstance(value_val, str)
                    and value_val.strip()
                ):
                    try:
                        chap_num_val = kg_keys.parse_elaboration_key(key_str)
                        if chap_num_val is None:
                            logger.warning(
                                f"Could not parse chapter number from world elab key: {key_str} for item {item_name_str}"
                            )
                            continue

                        elab_summary = value_val.strip()
                        elab_event_id = (
                            f"elab_{we_id_str}_ch{chap_num_val}_{hash(elab_summary)}"
                        )

                        elab_is_provisional = False
                        sq_key_for_elab_chap = kg_keys.source_quality_key(chap_num_val)
                        if (
                            details_dict.get(sq_key_for_elab_chap)
                            == "provisional_from_unrevised_draft"
                        ):
                            elab_is_provisional = True

                        elab_props = {
                            "id": elab_event_id,
                            "summary": elab_summary,
                            KG_NODE_CHAPTER_UPDATED: chap_num_val,
                            KG_IS_PROVISIONAL: elab_is_provisional,
                        }
                        statements.append(
                            (
                                """
                            MATCH (we:WorldElement:Entity {id: $we_id_val})
                            CREATE (elab:Entity:WorldElaborationEvent)
                            SET elab = $props, elab.created_ts = timestamp()
                            CREATE (we)-[:ELABORATED_IN_CHAPTER]->(elab)
                            """,
                                {"we_id_val": we_id_str, "props": elab_props},
                            )
                        )
                    except ValueError:
                        logger.warning(
                            f"Could not parse chapter for world elab key: {key_str} for item {item_name_str}"
                        )

    # 5. Cleanup orphaned ValueNodes (those not connected to any WorldElement after reconciliation)
    statements.append(
        (
            """
        MATCH (v:ValueNode:Entity)
        WHERE NOT EXISTS((:WorldElement:Entity)-[]->(v)) AND NOT EXISTS((:Entity)-[:DYNAMIC_REL]->(v))
        DETACH DELETE v
        """,
            {},
        )
    )

    try:
        if statements:
            await neo4j_manager.execute_cypher_batch(statements)
        logger.info("Successfully synchronized world building data to Neo4j.")
        return True
    except Exception as e:
        logger.error(f"Error synchronizing world building data: {e}", exc_info=True)
        return False


@alru_cache(maxsize=128)
async def get_world_item_by_id(item_id: str) -> WorldItem | None:
    """Retrieve a single ``WorldItem`` from Neo4j by its ID or fall back to name."""
    logger.info("Loading world item '%s' from Neo4j...", item_id)

    query = (
        "MATCH (we:WorldElement:Entity {id: $id})"
        " WHERE we.is_deleted IS NULL OR we.is_deleted = FALSE"
        " RETURN we"
    )
    results = await neo4j_manager.execute_read_query(query, {"id": item_id})
    if not results or not results[0].get("we"):
        alt_id = resolve_world_name(item_id)
        if alt_id and alt_id != item_id:
            results = await neo4j_manager.execute_read_query(query, {"id": alt_id})

    if not results or not results[0].get("we"):
        logger.info("No world item found for id '%s'.", item_id)
        return None

    we_node = results[0]["we"]
    category = we_node.get("category")
    item_name = we_node.get("name")
    if not category or not item_name:
        logger.warning("WorldElement missing category or name for id '%s'.", item_id)
        return None

    item_detail: dict[str, Any] = dict(we_node)
    item_detail.pop("created_ts", None)
    item_detail.pop("updated_ts", None)

    created_chapter_num = item_detail.pop(
        KG_NODE_CREATED_CHAPTER, settings.KG_PREPOPULATION_CHAPTER_NUM
    )
    item_detail["created_chapter"] = int(created_chapter_num)
    item_detail[kg_keys.added_key(created_chapter_num)] = True

    if item_detail.pop(KG_IS_PROVISIONAL, False):
        item_detail["is_provisional"] = True
        item_detail[kg_keys.source_quality_key(created_chapter_num)] = (
            "provisional_from_unrevised_draft"
        )
    else:
        item_detail["is_provisional"] = False

    list_prop_map = {
        "goals": "HAS_GOAL",
        "rules": "HAS_RULE",
        "key_elements": "HAS_KEY_ELEMENT",
        "traits": "HAS_TRAIT_ASPECT",
    }
    for list_prop_key, rel_name_internal in list_prop_map.items():
        list_values_query = f"""
        MATCH (:WorldElement:Entity {{id: $we_id_param}})-[:{rel_name_internal}]->(v:ValueNode:Entity {{type: $value_node_type_param}})
        RETURN v.value AS item_value
        ORDER BY v.value ASC
        """
        list_val_res = await neo4j_manager.execute_read_query(
            list_values_query,
            {"we_id_param": item_id, "value_node_type_param": list_prop_key},
        )
        item_detail[list_prop_key] = sorted(
            [
                res_item["item_value"]
                for res_item in list_val_res
                if res_item and res_item.get("item_value") is not None
            ]
        )

    elab_query = f"""
    MATCH (:WorldElement:Entity {{id: $we_id_param}})-[:ELABORATED_IN_CHAPTER]->(elab:WorldElaborationEvent:Entity)
    RETURN elab.summary AS summary, elab.{KG_NODE_CHAPTER_UPDATED} AS chapter, elab.{KG_IS_PROVISIONAL} AS is_provisional
    ORDER BY elab.chapter_updated ASC
    """
    elab_results = await neo4j_manager.execute_read_query(
        elab_query, {"we_id_param": item_id}
    )
    if elab_results:
        for elab_rec in elab_results:
            chapter_val = elab_rec.get("chapter")
            summary_val = elab_rec.get("summary")
            if chapter_val is not None and summary_val is not None:
                elab_key = kg_keys.elaboration_key(chapter_val)
                item_detail[elab_key] = summary_val
                if elab_rec.get(KG_IS_PROVISIONAL):
                    item_detail[kg_keys.source_quality_key(chapter_val)] = (
                        "provisional_from_unrevised_draft"
                    )

    item_detail["id"] = item_id
    return WorldItem.from_dict(category, item_name, item_detail)


@alru_cache(maxsize=128)
async def get_all_world_item_ids_by_category() -> dict[str, list[str]]:
    """Return all world item IDs grouped by category."""
    query = (
        "MATCH (we:WorldElement:Entity) "
        "WHERE we.is_deleted IS NULL OR we.is_deleted = FALSE "
        "RETURN we.category AS category, we.id AS id"
    )
    results = await neo4j_manager.execute_read_query(query)
    mapping: dict[str, list[str]] = {}
    for record in results:
        category = record.get("category")
        item_id = record.get("id")
        if category and item_id:
            mapping.setdefault(category, []).append(item_id)
    return mapping


async def get_world_building_from_db() -> dict[str, dict[str, WorldItem]]:
    """Load all world elements grouped by category from Neo4j."""

    logger.info("Loading decomposed world building data from Neo4j...")
    world_data: dict[str, dict[str, WorldItem]] = {}
    wc_id_param = settings.MAIN_WORLD_CONTAINER_NODE_ID

    await fix_missing_world_element_core_fields()

    WORLD_NAME_TO_ID.clear()

    # Load WorldContainer (_overview_)
    overview_query = "MATCH (wc:WorldContainer:Entity {id: $wc_id_param}) RETURN wc"
    overview_res_list = await neo4j_manager.execute_read_query(
        overview_query, {"wc_id_param": wc_id_param}
    )
    if overview_res_list and overview_res_list[0] and overview_res_list[0].get("wc"):
        wc_node = overview_res_list[0]["wc"]
        overview_data = dict(wc_node)
        overview_data.pop("created_ts", None)
        overview_data.pop("updated_ts", None)
        if overview_data.get(KG_IS_PROVISIONAL):
            overview_data[
                kg_keys.source_quality_key(settings.KG_PREPOPULATION_CHAPTER_NUM)
            ] = "provisional_from_unrevised_draft"
        world_data.setdefault("_overview_", {})["_overview_"] = WorldItem.from_dict(
            "_overview_",
            "_overview_",
            overview_data,
        )
        WORLD_NAME_TO_ID[utils._normalize_for_id("_overview_")] = (
            utils._normalize_for_id("_overview_")
        )

    # Load WorldElements and their details
    we_query = (
        "MATCH (we:WorldElement:Entity)"
        " WHERE we.is_deleted IS NULL OR we.is_deleted = FALSE"
        " RETURN we"
    )
    we_results = await neo4j_manager.execute_read_query(we_query)

    if not we_results:
        logger.info("No WorldElements found in Neo4j.")
        standard_categories = [
            "locations",
            "society",
            "systems",
            "lore",
            "history",
            "factions",
        ]
        for cat_key in standard_categories:
            world_data.setdefault(cat_key, {})
        return world_data

    for record in we_results:
        we_node = record.get("we")
        if not we_node:
            continue

        # These are the display/canonical versions from the node
        category = we_node.get("category")
        item_name = we_node.get("name")
        we_id = we_node.get("id")

        if not all([category, item_name, we_id]):
            logger.warning(
                f"Skipping WorldElement with missing core fields (id, name, or category): {we_node}"
            )
            continue

        world_data.setdefault(category, {})

        item_detail = dict(we_node)
        item_detail.pop("created_ts", None)
        item_detail.pop("updated_ts", None)

        created_chapter_num = item_detail.pop(
            KG_NODE_CREATED_CHAPTER,
            settings.KG_PREPOPULATION_CHAPTER_NUM,
        )
        item_detail["created_chapter"] = int(
            created_chapter_num
        )  # Ensure it's int and under standard key
        item_detail[kg_keys.added_key(created_chapter_num)] = True

        if item_detail.pop(KG_IS_PROVISIONAL, False):
            item_detail["is_provisional"] = True  # Ensure under standard key
            item_detail[kg_keys.source_quality_key(created_chapter_num)] = (
                "provisional_from_unrevised_draft"
            )
        else:
            item_detail["is_provisional"] = False

        list_prop_map = {
            "goals": "HAS_GOAL",
            "rules": "HAS_RULE",
            "key_elements": "HAS_KEY_ELEMENT",
            "traits": "HAS_TRAIT_ASPECT",
        }
        for list_prop_key, rel_name_internal in list_prop_map.items():
            list_values_query = f"""
            MATCH (:WorldElement:Entity {{id: $we_id_param}})-[:{rel_name_internal}]->(v:ValueNode:Entity {{type: $value_node_type_param}})
            RETURN v.value AS item_value
            ORDER BY v.value ASC
            """
            list_val_res = await neo4j_manager.execute_read_query(
                list_values_query,
                {"we_id_param": we_id, "value_node_type_param": list_prop_key},
            )
            item_detail[list_prop_key] = sorted(
                [
                    res_item["item_value"]
                    for res_item in list_val_res
                    if res_item and res_item["item_value"] is not None
                ]
            )

        elab_query = f"""
        MATCH (:WorldElement:Entity {{id: $we_id_param}})-[:ELABORATED_IN_CHAPTER]->(elab:WorldElaborationEvent:Entity)
        RETURN elab.summary AS summary, elab.{KG_NODE_CHAPTER_UPDATED} AS chapter, elab.{KG_IS_PROVISIONAL} AS is_provisional
        ORDER BY elab.chapter_updated ASC
        """
        elab_results = await neo4j_manager.execute_read_query(
            elab_query, {"we_id_param": we_id}
        )
        if elab_results:
            for elab_rec in elab_results:
                chapter_val = elab_rec.get("chapter")
                summary_val = elab_rec.get("summary")
                if chapter_val is not None and summary_val is not None:
                    elab_key = kg_keys.elaboration_key(chapter_val)
                    item_detail[elab_key] = summary_val
                    if elab_rec.get(KG_IS_PROVISIONAL):
                        item_detail[kg_keys.source_quality_key(chapter_val)] = (
                            "provisional_from_unrevised_draft"
                        )

        item_detail["id"] = we_id  # Add the canonical ID from the DB
        world_data.setdefault(category, {})[item_name] = WorldItem.from_dict(
            category,
            item_name,
            item_detail,
        )
        WORLD_NAME_TO_ID[utils._normalize_for_id(item_name)] = we_id

    logger.info(
        f"Successfully loaded and recomposed world building data ({len(we_results)} elements) from Neo4j."
    )
    return world_data


async def get_world_elements_for_snippet_from_db(
    category: str, chapter_limit: int, item_limit: int
) -> list[dict[str, Any]]:
    """Return a subset of world elements for prompt context."""

    query = f"""
    MATCH (we:WorldElement:Entity {{category: $category_param}})
    WHERE (we.{KG_NODE_CREATED_CHAPTER} IS NULL OR we.{KG_NODE_CREATED_CHAPTER} <= $chapter_limit_param)

    OPTIONAL MATCH (we)-[:ELABORATED_IN_CHAPTER]->(elab:WorldElaborationEvent:Entity)
    WHERE elab.{KG_NODE_CHAPTER_UPDATED} <= $chapter_limit_param AND elab.{KG_IS_PROVISIONAL} = TRUE
    
    WITH we, COLLECT(DISTINCT elab) AS provisional_elaborations_found
    WITH we, ( we.{KG_IS_PROVISIONAL} = TRUE OR size(provisional_elaborations_found) > 0 ) AS is_item_provisional_overall
    
    RETURN we.name AS name,
           we.description AS description, 
           is_item_provisional_overall AS is_provisional
    ORDER BY we.name ASC 
    LIMIT $item_limit_param
    """
    params = {
        "category_param": category,
        "chapter_limit_param": chapter_limit,
        "item_limit_param": item_limit,
    }
    items = []
    try:
        results = await neo4j_manager.execute_read_query(query, params)
        if results:
            for record in results:
                desc_val = record.get("description")
                desc = (
                    str(desc_val) if desc_val is not None else ""
                )  # Ensure desc is a string

                items.append(
                    {
                        "name": record.get("name"),
                        "description_snippet": (
                            desc[:50].strip() + "..."
                            if len(desc) > 50
                            else desc.strip()
                        ),
                        "is_provisional": record.get("is_provisional", False),
                    }
                )
    except Exception as e:
        logger.error(
            f"Error fetching world elements for snippet (cat {category}): {e}",
            exc_info=True,
        )
    return items


async def find_thin_world_elements_for_enrichment() -> list[dict[str, Any]]:
    """Finds WorldElement nodes that are considered 'thin' (e.g., missing description)."""
    query = """
    MATCH (we:WorldElement)
    WHERE (we.description IS NULL OR we.description = '') AND (we.is_deleted IS NULL OR we.is_deleted = FALSE)
    RETURN we.id AS id, we.name AS name, we.category as category
    LIMIT 20 // Limit to avoid overwhelming the LLM in one cycle
    """
    try:
        results = await neo4j_manager.execute_read_query(query)
        return results if results else []
    except Exception as e:
        logger.error(f"Error finding thin world elements: {e}", exc_info=True)
        return []


async def fix_missing_world_element_core_fields() -> int:
    """Populate missing ``id``, ``name``, or ``category`` on WorldElements."""

    query = """
    MATCH (we:WorldElement)
    WHERE (we.is_deleted IS NULL OR we.is_deleted = FALSE)
      AND (
        we.id IS NULL OR trim(we.id) = "" OR
        we.name IS NULL OR trim(we.name) = "" OR
        we.category IS NULL OR trim(we.category) = ""
      )
    RETURN id(we) AS nid, we.id AS id, we.name AS name, we.category AS category
    """

    try:
        results = await neo4j_manager.execute_read_query(query)
    except Exception as exc:  # pragma: no cover - narrow DB errors
        logger.error(
            "Error fetching WorldElements missing core fields: %s",
            exc,
            exc_info=True,
        )
        return 0

    if not results:
        return 0

    statements: list[tuple[str, dict[str, Any]]] = []

    for rec in results:
        neo_id = rec.get("nid")
        if neo_id is None:
            continue

        w_id = rec.get("id")
        name = rec.get("name")
        category = rec.get("category")

        if isinstance(w_id, str):
            w_id = w_id.strip() or None
        if isinstance(name, str):
            name = name.strip() or None
        if isinstance(category, str):
            category = category.strip() or None

        props: dict[str, Any] = {}

        if not name and isinstance(w_id, str):
            name_part = w_id.split("_", 1)[-1]
            props["name"] = name_part.replace("_", " ").title()
            name = props["name"]
        elif not name:
            props["name"] = "Unnamed Element"
            name = props["name"]

        if not category:
            if isinstance(w_id, str):
                props["category"] = w_id.split("_")[0]
            else:
                props["category"] = "unknown_category"
            category = props["category"]

        if not w_id and name:
            props["id"] = (
                f"{utils._normalize_for_id(category)}_{utils._normalize_for_id(name)}"
            )

        if props:
            statements.append(
                (
                    "MATCH (we:WorldElement) WHERE id(we) = $nid SET we += $props",
                    {"nid": neo_id, "props": props},
                )
            )

    if not statements:
        return 0

    try:
        await neo4j_manager.execute_cypher_batch(statements)
        logger.info("Filled missing core fields for %d WorldElements.", len(statements))
    except Exception as exc:  # pragma: no cover - narrow DB errors
        logger.error(
            "Error updating WorldElements with missing core fields: %s",
            exc,
            exc_info=True,
        )
        return 0

    return len(statements)


async def remove_world_element_trait_aspect(element_id: str, trait_value: str) -> bool:
    """Remove a trait aspect from a world element."""
    query = (
        "MATCH (:WorldElement {id: $we_id})-"
        "[r:HAS_TRAIT_ASPECT]->(v:ValueNode {type: 'traits', value: $trait})"
        " DELETE r"
    )
    cleanup_query = (
        "MATCH (v:ValueNode {type: 'traits', value: $trait})"
        " WHERE NOT EXISTS((:WorldElement)-[:HAS_TRAIT_ASPECT]->(v))"
        " DETACH DELETE v"
    )
    try:
        await neo4j_manager.execute_write_query(
            query, {"we_id": element_id, "trait": trait_value}
        )
        await neo4j_manager.execute_write_query(cleanup_query, {"trait": trait_value})
        return True
    except Exception as exc:  # pragma: no cover - log but return False
        logger.error(
            "Error removing trait '%s' from world element %s: %s",
            trait_value,
            element_id,
            exc,
        )
        return False
