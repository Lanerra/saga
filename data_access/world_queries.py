# data_access/world_queries.py
from typing import Any

import structlog
from async_lru import alru_cache  # type: ignore[import-untyped]

import config
import utils
from core.db_manager import neo4j_manager
from core.exceptions import handle_database_error
from core.schema_validator import validate_kg_object
from models import WorldItem
from models.kg_constants import (
    KG_IS_PROVISIONAL,
    KG_NODE_CHAPTER_UPDATED,
    KG_NODE_CREATED_CHAPTER,
    WORLD_ITEM_CANONICAL_LABELS,
    WORLD_ITEM_LEGACY_LABELS,
)

from .cypher_builders.native_builders import NativeCypherBuilder

# Legacy world cypher builder removed; native builder is the single path.

logger = structlog.get_logger(__name__)

# Mapping from normalized world item names to canonical IDs
#
# Lifecycle contract (P1):
# - `resolve_world_name()` is best-effort ONLY (purely in-memory; no DB IO).
# - The authoritative map population happens in explicit "populate" flows:
#   - write-path: `sync_world_items()` (from provided models)
#   - read-path:  `get_world_building()` (from fetched models)
# - Callers/tests may clear/reset explicitly via the helpers below.
WORLD_NAME_TO_ID: dict[str, str] = {}


def clear_world_name_map() -> None:
    """Clear the in-process world name→id map."""
    WORLD_NAME_TO_ID.clear()


def rebuild_world_name_map(world_items: list["WorldItem"]) -> None:
    """Rebuild the world name→id map from a list of WorldItem models.

    This clears existing entries to avoid stale accumulation across runs/tests.
    """
    WORLD_NAME_TO_ID.clear()
    for item in world_items:
        if isinstance(item, WorldItem) and item.name and item.id:
            WORLD_NAME_TO_ID[utils._normalize_for_id(item.name)] = item.id


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


@alru_cache(maxsize=128)
async def get_world_item_by_id(
    item_id: str, *, include_provisional: bool = False
) -> WorldItem | None:
    """Retrieve a single ``WorldItem`` from Neo4j by its ID or fall back to name.

    Provisional contract (P0):
    - Default excludes provisional world items (node-level) and provisional elaboration events
      unless include_provisional=True.

    Notes:
        This function may be called with either a canonical KG id (e.g. ``locations_castle``)
        or a display name (e.g. ``Castle``). If name→id fallback succeeds, all subsequent
        enrichment queries must use the resolved canonical id ("effective id") consistently.
    """
    logger.info(f"Loading world item '{item_id}' from Neo4j...")

    requested_id = item_id
    effective_id: str = item_id

    world_item_labels = WORLD_ITEM_CANONICAL_LABELS + WORLD_ITEM_LEGACY_LABELS
    label_predicate = "(" + " OR ".join([f"we:{label}" for label in world_item_labels]) + ")"

    query = (
        f"MATCH (we {{id: $id}}) WHERE {label_predicate}"
        " AND (we.is_deleted IS NULL OR we.is_deleted = FALSE)"
        " AND ($include_provisional = TRUE OR coalesce(we.is_provisional, FALSE) = FALSE)"
        " RETURN we"
    )

    results = await neo4j_manager.execute_read_query(
        query, {"id": requested_id, "include_provisional": include_provisional}
    )
    if not results or not results[0].get("we"):
        alt_id = resolve_world_name(requested_id)
        if alt_id and alt_id != requested_id:
            effective_id = alt_id
            results = await neo4j_manager.execute_read_query(
                query, {"id": effective_id, "include_provisional": include_provisional}
            )

    if not results or not results[0].get("we"):
        logger.info(f"No world item found for id '{requested_id}'.")
        return None

    we_node = results[0]["we"]
    category = we_node.get("category")
    item_name = we_node.get("name")
    we_id = we_node.get("id")

    # Validate and normalize core fields for world item
    # This ensures that all world items have valid id, category, and name
    try:
        category, item_name, we_id = utils.validate_world_item_fields(
            category, item_name, we_id
        )
    except Exception as e:
        logger.error(
            f"Error validating world item core fields: Category='{category}', Name='{item_name}', ID='{we_id}': {e}",
            exc_info=True,
        )
        return None

    # Prefer the fetched/validated node id as the single effective id for enrichment + identity.
    effective_id = we_id

    # Check if any fields were missing and log a warning if so
    missing_fields = []
    if not we_node.get("category"):
        missing_fields.append("category")
    if not we_node.get("name"):
        missing_fields.append("name")
    if not we_node.get("id"):
        missing_fields.append("id")

    if missing_fields:
        logger.warning(
            f"Corrected world item with missing core fields ({', '.join(missing_fields)}) for id '{item_id}': {we_node}"
        )
        # Update the we_node dict with corrected values for subsequent processing
        we_node["category"] = category
        we_node["name"] = item_name
        we_node["id"] = we_id

    item_detail: dict[str, Any] = dict(we_node)
    item_detail.pop("created_ts", None)
    item_detail.pop("updated_ts", None)

    created_chapter_num = item_detail.pop(
        KG_NODE_CREATED_CHAPTER, config.KG_PREPOPULATION_CHAPTER_NUM
    )
    item_detail["created_chapter"] = int(created_chapter_num)
    item_detail[f"added_in_chapter_{created_chapter_num}"] = True

    if item_detail.pop(KG_IS_PROVISIONAL, False):
        item_detail["is_provisional"] = True
        item_detail[f"source_quality_chapter_{created_chapter_num}"] = (
            "provisional_from_unrevised_draft"
        )
    else:
        item_detail["is_provisional"] = False

    # Fetch traits from Trait nodes
    traits_query = """
    MATCH ({id: $we_id_param})-[:HAS_TRAIT]->(t:Trait)
    RETURN t.name AS trait_name
    ORDER BY t.name ASC
    """
    traits_res = await neo4j_manager.execute_read_query(
        traits_query,
        {"we_id_param": effective_id},
    )
    item_detail["traits"] = sorted(
        [
            res_item["trait_name"]
            for res_item in traits_res
            if res_item and res_item.get("trait_name") is not None
        ]
    )

    elab_query = f"""
    MATCH ({{id: $we_id_param}})-[:ELABORATED_IN_CHAPTER]->(elab:WorldElaborationEvent)
    WHERE $include_provisional = TRUE OR coalesce(elab.{KG_IS_PROVISIONAL}, FALSE) = FALSE
    RETURN elab.summary AS summary, elab.{KG_NODE_CHAPTER_UPDATED} AS chapter, elab.{KG_IS_PROVISIONAL} AS is_provisional
    ORDER BY elab.chapter_updated ASC
    """
    elab_results = await neo4j_manager.execute_read_query(
        elab_query,
        {"we_id_param": effective_id, "include_provisional": include_provisional},
    )
    if elab_results:
        for elab_rec in elab_results:
            chapter_val = elab_rec.get("chapter")
            summary_val = elab_rec.get("summary")
            if chapter_val is not None and summary_val is not None:
                elab_key = f"elaboration_in_chapter_{chapter_val}"
                item_detail[elab_key] = summary_val
                if elab_rec.get(KG_IS_PROVISIONAL):
                    item_detail[f"source_quality_chapter_{chapter_val}"] = (
                        "provisional_from_unrevised_draft"
                    )

    # Do not overwrite identity with the caller-provided value; callers must see the
    # canonical/effective id that actually exists in the graph.
    item_detail["id"] = effective_id
    return WorldItem.from_dict(category, item_name, item_detail)


async def get_world_elements_for_snippet_from_db(
    category: str, chapter_limit: int, item_limit: int, *, include_provisional: bool = False
) -> list[dict[str, Any]]:
    query = f"""
    MATCH (we {{category: $category_param}})
    WHERE (we:Object OR we:Artifact OR we:Location OR we:Document OR we:Item OR we:Relic)
      AND (we.is_deleted IS NULL OR we.is_deleted = FALSE)
      AND (we.{KG_NODE_CREATED_CHAPTER} IS NULL OR we.{KG_NODE_CREATED_CHAPTER} <= $chapter_limit_param)

    OPTIONAL MATCH (we)-[:ELABORATED_IN_CHAPTER]->(elab:WorldElaborationEvent)
    WHERE elab.{KG_NODE_CHAPTER_UPDATED} <= $chapter_limit_param
      AND coalesce(elab.{KG_IS_PROVISIONAL}, FALSE) = TRUE

    WITH we, COLLECT(DISTINCT elab) AS provisional_elaborations_found
    WITH
      we,
      ( coalesce(we.{KG_IS_PROVISIONAL}, FALSE) = TRUE OR size(provisional_elaborations_found) > 0 )
        AS is_item_provisional_overall
    WHERE $include_provisional = TRUE OR is_item_provisional_overall = FALSE

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
        "include_provisional": include_provisional,
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
        # P1.9: Raise standardized DB errors (do not silently return partial/empty results).
        raise handle_database_error(
            "get_world_elements_for_snippet_from_db",
            e,
            category=category,
            chapter_limit=chapter_limit,
            item_limit=item_limit,
        )
    return items


async def find_thin_world_elements_for_enrichment() -> list[dict[str, Any]]:
    """Find typed world items that are considered 'thin' (e.g., missing description)."""
    query = """
    MATCH (we)
    WHERE (we:Object OR we:Artifact OR we:Location OR we:Document OR we:Item OR we:Relic)
      AND toString(we.description) = ''
      AND (we.is_deleted IS NULL OR we.is_deleted = FALSE)
    RETURN we.id AS id, we.name AS name, we.category as category
    LIMIT 20
    """
    try:
        results = await neo4j_manager.execute_read_query(query)
        return results if results else []
    except Exception as e:
        logger.error(f"Error finding thin world elements: {e}", exc_info=True)
        return []


# Native model functions for performance optimization
async def sync_world_items(
    world_items: list[WorldItem],
    chapter_number: int,
) -> bool:
    """Persist world element data to Neo4j using native models."""

    # Validate all world items before syncing
    for item in world_items:
        if isinstance(item, WorldItem):
            errors = validate_kg_object(item)
            if errors:
                logger.warning(f"Invalid WorldItem '{item.name}': {errors}")

    # Update name mapping deterministically (avoid stale accumulation).
    rebuild_world_name_map(world_items)

    try:
        cypher_builder = NativeCypherBuilder()
        statements = cypher_builder.batch_world_item_upsert_cypher(
            world_items, chapter_number
        )

        if statements:
            await neo4j_manager.execute_cypher_batch(statements)

        logger.info(
            "Persisted %d world item updates for chapter %d using native models.",
            len(world_items),
            chapter_number,
        )

        # P1.6: Post-write cache invalidation
        # Local import avoids circular import / eager import side effects.
        from data_access.cache_coordinator import clear_world_read_caches

        clear_world_read_caches()

        return True

    except Exception as exc:
        logger.error(
            "Error persisting world item updates for chapter %d: %s",
            chapter_number,
            exc,
            exc_info=True,
        )
        return False


async def get_world_building(*, include_provisional: bool = False) -> list[WorldItem]:
    """
    Native model version of get_world_building_from_db.
    Returns world items as model instances without dict conversion.

    Returns:
        List of WorldItem models
    """
    try:
        cypher_builder = NativeCypherBuilder()
        query, params = cypher_builder.world_item_fetch_cypher()

        results = await neo4j_manager.execute_read_query(query, params)
        world_items = []

        for record in results:
            if record and record.get("w"):
                item = WorldItem.from_dict_record(record)
                world_items.append(item)

        # Update name-to-id mapping for compatibility with callers
        rebuild_world_name_map(world_items)

        if not include_provisional:
            world_items = [w for w in world_items if not getattr(w, "is_provisional", False)]

        logger.info("Fetched %d world items using native models", len(world_items))
        return world_items

    except Exception as exc:
        logger.error(f"Error fetching world building: {exc}", exc_info=True)
        return []


async def get_world_items_for_chapter_context_native(
    chapter_number: int, limit: int = 10, *, include_provisional: bool = False
) -> list[WorldItem]:
    """
    Get world items relevant for chapter context using native models.

    Args:
        chapter_number: Current chapter being processed
        limit: Maximum number of world items to return

    Returns:
        List of WorldItem models relevant to the chapter
    """
    try:
        query = """
        MATCH (w)-[:REFERENCED_IN]->(ch:Chapter)
        WHERE ch.number < $chapter_number
          AND (w.is_deleted IS NULL OR w.is_deleted = FALSE)
          AND ($include_provisional = TRUE OR coalesce(w.is_provisional, FALSE) = FALSE)
        WITH w, max(ch.number) as last_reference
        ORDER BY last_reference DESC
        LIMIT $limit
        RETURN w
        """

        results = await neo4j_manager.execute_read_query(
            query,
            {
                "chapter_number": chapter_number,
                "limit": limit,
                "include_provisional": include_provisional,
            },
        )

        world_items = []
        for record in results:
            if record and record.get("w"):
                item = WorldItem.from_dict_record(record)
                world_items.append(item)

        logger.debug(
            "Fetched %d world items for chapter %d context using native models",
            len(world_items),
            chapter_number,
        )

        return world_items

    except Exception as exc:
        logger.error(
            "Error fetching world items for chapter %d context: %s",
            chapter_number,
            exc,
            exc_info=True,
        )
        return []


# Phase 1.2: Bootstrap Element Injection - New functions for bootstrap element discovery
async def get_bootstrap_world_elements() -> list[WorldItem]:
    """
    Get world elements created during bootstrap phase for early chapter injection.

    Returns world elements that were created during the bootstrap/genesis phase
    and should be injected into early chapter contexts to establish their narrative presence.

    Returns:
        List of WorldItem models from bootstrap phase, sorted by category then name
    """
    # More efficient query that filters out elements without meaningful descriptions earlier
    query = """
    MATCH (we)
    WHERE (we:Object OR we:Artifact OR we:Location OR we:Document OR we:Item OR we:Relic)
      AND (we.is_deleted IS NULL OR we.is_deleted = FALSE)
      AND (toString(we.source) CONTAINS 'bootstrap' OR we.created_chapter = 0 OR we.created_chapter = $prepop_chapter)
      AND we.description IS NOT NULL
      AND trim(toString(we.description)) <> ''
      AND NOT (toString(we.description) CONTAINS $fill_in_marker)
    RETURN we
    ORDER BY we.category ASC, we.name ASC
    LIMIT 20
    """

    params = {
        "prepop_chapter": config.KG_PREPOPULATION_CHAPTER_NUM,
        "fill_in_marker": config.FILL_IN,
    }

    try:
        records = await neo4j_manager.execute_read_query(query, params)

        bootstrap_elements = []
        for record in records:
            # Some tests/mocks historically used "w" instead of "we"; accept both.
            we_node = None
            if isinstance(record, dict):
                we_node = record.get("we") or record.get("w")

            if not we_node:
                continue

            # Convert Neo4j node to WorldItem
            try:
                world_item = WorldItem.from_db_node(we_node)
                # Additional validation: ensure the description is meaningful after conversion
                if (
                    world_item.description
                    and world_item.description.strip()
                    and config.FILL_IN not in world_item.description
                ):
                    bootstrap_elements.append(world_item)

            except Exception as e:
                logger.warning(
                    f"Failed to convert bootstrap element node to WorldItem: {e}. "
                    f"Node: {dict(we_node)}"
                )
                continue

        logger.info(
            f"Retrieved {len(bootstrap_elements)} bootstrap world elements for early chapter injection"
        )

        return bootstrap_elements

    except Exception as e:
        logger.error(
            f"Failed to retrieve bootstrap world elements: {e}. "
            f"Error type: {type(e).__name__}. "
            f"Check Neo4j connection and query syntax."
        )
        return []
