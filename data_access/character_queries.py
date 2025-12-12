# data_access/character_queries.py
from typing import Any

import structlog
from async_lru import alru_cache  # type: ignore[import-untyped]
from neo4j.exceptions import ServiceUnavailable

import utils
from core.db_manager import neo4j_manager
from core.schema_validator import validate_kg_object
from models import CharacterProfile

from .cypher_builders.native_builders import NativeCypherBuilder

# Mapping from normalized character names to canonical display names
#
# Lifecycle contract (P1):
# - `resolve_character_name()` is best-effort ONLY (purely in-memory; no DB IO).
# - The authoritative map population happens in explicit "populate" flows:
#   - write-path: `sync_characters()` (from provided models)
#   - read-path:  `get_character_profiles()` (from fetched models)
# - Callers/tests may clear/reset explicitly via the helpers below.
CHAR_NAME_TO_CANONICAL: dict[str, str] = {}


def clear_character_name_map() -> None:
    """Clear the in-process character name canonicalization map."""
    CHAR_NAME_TO_CANONICAL.clear()


def rebuild_character_name_map(characters: list["CharacterProfile"]) -> None:
    """Rebuild the character name canonicalization map from a list of profiles.

    This clears existing entries to avoid stale accumulation across runs/tests.
    """
    CHAR_NAME_TO_CANONICAL.clear()
    for char in characters:
        if isinstance(char, CharacterProfile) and char.name:
            CHAR_NAME_TO_CANONICAL[utils._normalize_for_id(char.name)] = char.name


def resolve_character_name(name: str) -> str:
    """Return canonical character name for a display variant."""
    if not name:
        return name
    return CHAR_NAME_TO_CANONICAL.get(utils._normalize_for_id(name), name)


logger = structlog.get_logger(__name__)


@alru_cache(maxsize=128)
async def get_character_profile_by_name(
    name: str, *, include_provisional: bool = False
) -> CharacterProfile | None:
    """
    Retrieve a single CharacterProfile from Neo4j by character name.

    Provisional contract (P0):
    - Default excludes provisional relationship + event data unless include_provisional=True.
    - Node-level provisional status is preserved on the returned profile (callers can decide).
    """
    canonical_name = resolve_character_name(name)
    logger.info(f"Loading character profile '{canonical_name}' from Neo4j...")

    query = """
        MATCH (c:Character {name: $name})
        WHERE c.is_deleted IS NULL OR c.is_deleted = FALSE

        OPTIONAL MATCH (c)-[:HAS_TRAIT]->(t:Trait)

        // Do NOT add a WHERE clause after OPTIONAL MATCH; it will null-drop the row.
        OPTIONAL MATCH (c)-[r]->(target)

        OPTIONAL MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent)

        WITH
            c,
            collect(DISTINCT t.name) AS traits,
            collect(
                DISTINCT CASE
                    WHEN coalesce(r.source_profile_managed, false) = true
                     AND (
                          $include_provisional = true
                          OR coalesce(r.is_provisional, FALSE) = FALSE
                     )
                    THEN {
                        target_name: target.name,
                        rel_type: type(r),
                        rel_props: properties(r)
                    }
                END
            ) AS relationships_raw,
            collect(
                DISTINCT CASE
                    WHEN dev IS NOT NULL
                     AND (
                          $include_provisional = true
                          OR coalesce(dev.is_provisional, FALSE) = FALSE
                     )
                    THEN {
                        summary: dev.summary,
                        chapter: dev.chapter_updated,
                        is_provisional: coalesce(dev.is_provisional, FALSE)
                    }
                END
            ) AS dev_events_raw

        RETURN
            c,
            traits,
            [rel IN relationships_raw WHERE rel IS NOT NULL] AS relationships,
            [e IN dev_events_raw WHERE e IS NOT NULL] AS dev_events
    """

    results = await neo4j_manager.execute_read_query(
        query, {"name": canonical_name, "include_provisional": include_provisional}
    )
    if not results or not results[0].get("c"):
        logger.info(f"No character profile found for '{canonical_name}'.")
        return None

    record = results[0]
    char_node = record["c"]
    profile: dict[str, Any] = dict(char_node)
    profile.pop("name", None)
    profile.pop("created_ts", None)
    profile.pop("updated_ts", None)

    profile["traits"] = sorted([t for t in record["traits"] if t])

    relationships: dict[str, Any] = {}
    for rel_rec in record["relationships"]:
        if not rel_rec or not rel_rec.get("target_name"):
            continue
        target_name = rel_rec["target_name"]
        rel_props_full = rel_rec.get("rel_props", {})
        rel_props_cleaned = {}
        if isinstance(rel_props_full, dict):
            rel_props_cleaned = {
                k: v
                for k, v in rel_props_full.items()
                if k not in ["created_ts", "updated_ts", "source_profile_managed", "chapter_added"]
            }
        # P1.7: Canonical relationship typing = type(r) from Cypher (`rel_type`).
        # Fall back to legacy property-based typing if present.
        rel_type = rel_rec.get("rel_type")
        if rel_type:
            rel_props_cleaned["type"] = rel_type
        elif isinstance(rel_props_full, dict) and "type" in rel_props_full:
            rel_props_cleaned["type"] = rel_props_full["type"]

        if "chapter_added" in rel_props_full:
            rel_props_cleaned["chapter_added"] = rel_props_full["chapter_added"]
        relationships[target_name] = rel_props_cleaned
    profile["relationships"] = relationships

    for dev_rec in record["dev_events"]:
        if not dev_rec or dev_rec.get("summary") is None:
            continue
        chapter_num = dev_rec.get("chapter")
        summary = dev_rec.get("summary")
        if chapter_num is not None and summary is not None:
            dev_key = f"development_in_chapter_{chapter_num}"
            profile[dev_key] = summary
            if dev_rec.get("is_provisional"):
                profile[f"source_quality_chapter_{chapter_num}"] = "provisional_from_unrevised_draft"

    return CharacterProfile.from_dict(name, profile)


@alru_cache(maxsize=128)
async def get_character_profile_by_id(
    character_id: str, *, include_provisional: bool = False
) -> CharacterProfile | None:
    """
    Retrieve a single CharacterProfile from Neo4j by character ID.

    Provisional contract (P0):
    - Default excludes provisional relationship + event data unless include_provisional=True.
    - Node-level provisional status is preserved on the returned profile (callers can decide).
    """
    logger.info(f"Loading character profile with ID '{character_id}' from Neo4j...")

    query = """
        MATCH (c:Character {id: $character_id})
        WHERE c.is_deleted IS NULL OR c.is_deleted = FALSE

        OPTIONAL MATCH (c)-[:HAS_TRAIT]->(t:Trait)

        // Do NOT add a WHERE clause after OPTIONAL MATCH; it will null-drop the row.
        OPTIONAL MATCH (c)-[r]->(target)

        OPTIONAL MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent)

        WITH
            c,
            collect(DISTINCT t.name) AS traits,
            collect(
                DISTINCT CASE
                    WHEN coalesce(r.source_profile_managed, false) = true
                     AND (
                          $include_provisional = true
                          OR coalesce(r.is_provisional, FALSE) = FALSE
                     )
                    THEN {
                        target_name: target.name,
                        rel_type: type(r),
                        rel_props: properties(r)
                    }
                END
            ) AS relationships_raw,
            collect(
                DISTINCT CASE
                    WHEN dev IS NOT NULL
                     AND (
                          $include_provisional = true
                          OR coalesce(dev.is_provisional, FALSE) = FALSE
                     )
                    THEN {
                        summary: dev.summary,
                        chapter: dev.chapter_updated,
                        is_provisional: coalesce(dev.is_provisional, FALSE)
                    }
                END
            ) AS dev_events_raw

        RETURN
            c,
            traits,
            [rel IN relationships_raw WHERE rel IS NOT NULL] AS relationships,
            [e IN dev_events_raw WHERE e IS NOT NULL] AS dev_events
    """

    results = await neo4j_manager.execute_read_query(
        query,
        {"character_id": character_id, "include_provisional": include_provisional},
    )
    if not results or not results[0].get("c"):
        logger.info(f"No character profile found with ID '{character_id}'.")
        return None

    record = results[0]
    char_node = record["c"]
    name = char_node.get("name", "")

    profile: dict[str, Any] = dict(char_node)
    profile.pop("name", None)
    profile.pop("created_ts", None)
    profile.pop("updated_ts", None)

    profile["traits"] = sorted([t for t in record["traits"] if t])

    relationships: dict[str, Any] = {}
    for rel_rec in record["relationships"]:
        if not rel_rec or not rel_rec.get("target_name"):
            continue
        target_name = rel_rec["target_name"]
        rel_props_full = rel_rec.get("rel_props", {})
        rel_props_cleaned = {}
        if isinstance(rel_props_full, dict):
            rel_props_cleaned = {
                k: v
                for k, v in rel_props_full.items()
                if k not in ["created_ts", "updated_ts", "source_profile_managed", "chapter_added"]
            }
        # P1.7: Canonical relationship typing = type(r) from Cypher (`rel_type`).
        # Fall back to legacy property-based typing if present.
        rel_type = rel_rec.get("rel_type")
        if rel_type:
            rel_props_cleaned["type"] = rel_type
        elif isinstance(rel_props_full, dict) and "type" in rel_props_full:
            rel_props_cleaned["type"] = rel_props_full["type"]

        if "chapter_added" in rel_props_full:
            rel_props_cleaned["chapter_added"] = rel_props_full["chapter_added"]
        relationships[target_name] = rel_props_cleaned
    profile["relationships"] = relationships

    for dev_rec in record["dev_events"]:
        if not dev_rec or dev_rec.get("summary") is None:
            continue
        chapter_num = dev_rec.get("chapter")
        summary = dev_rec.get("summary")
        if chapter_num is not None and summary is not None:
            dev_key = f"development_in_chapter_{chapter_num}"
            profile[dev_key] = summary
            if dev_rec.get("is_provisional"):
                profile[f"source_quality_chapter_{chapter_num}"] = "provisional_from_unrevised_draft"

    return CharacterProfile.from_dict(name, profile)


async def get_all_character_names() -> list[str]:
    """Return a list of all character names from Neo4j."""
    query = (
        "MATCH (c:Character) "
        "WHERE c.is_deleted IS NULL OR c.is_deleted = FALSE "
        "RETURN c.name AS name ORDER BY c.name"
    )
    results = await neo4j_manager.execute_read_query(query)
    return [record["name"] for record in results if record.get("name")]


def _process_snippet_result(
    record: dict[str, Any], *, include_provisional: bool
) -> dict[str, Any]:
    """Process snippet query result into standardized format.

    Provisional contract (P0):
    - Default behavior (include_provisional=False) should NOT surface provisional development
      notes as the "most recent" signal.
    - Still compute and return `is_provisional_overall` so callers can understand data quality.
    """
    dev_events = record.get("dev_events", [])

    valid_events = [e for e in dev_events if isinstance(e, dict) and e.get("summary")]

    non_provisional = [e for e in valid_events if not e.get("is_provisional")]
    provisional = [e for e in valid_events if e.get("is_provisional")]

    most_recent_non_prov = (
        max(non_provisional, key=lambda e: e["chapter"]) if non_provisional else None
    )
    most_recent_prov = (
        max(provisional, key=lambda e: e["chapter"]) if provisional else None
    )

    if include_provisional:
        if most_recent_prov and (
            not most_recent_non_prov
            or most_recent_prov["chapter"] >= most_recent_non_prov["chapter"]
        ):
            most_current = most_recent_prov
        else:
            most_current = most_recent_non_prov
        dev_note = most_current.get("summary", "N/A") if most_current else "N/A"
    else:
        # Exclude provisional notes by default.
        dev_note = most_recent_non_prov.get("summary", "N/A") if most_recent_non_prov else "N/A"

    char_is_provisional = record.get("char_is_provisional", False)
    has_provisional_relationships = record.get("provisional_rel_count", 0) > 0
    has_provisional_events = len(provisional) > 0
    is_provisional_overall = (
        char_is_provisional or has_provisional_relationships or has_provisional_events
    )

    return {
        "description": record.get("description"),
        "current_status": record.get("current_status"),
        "most_recent_development_note": dev_note,
        "is_provisional_overall": is_provisional_overall,
    }


async def get_character_info_for_snippet_from_db(
    char_name: str, chapter_limit: int, *, include_provisional: bool = False
) -> dict[str, Any] | None:
    """Get character info for snippet with chapter limit.

    Provisional contract (P0):
    - Default excludes provisional development notes from the returned "most recent" view.
    - Still computes and returns `is_provisional_overall` based on underlying graph data.
    """
    canonical_name = resolve_character_name(char_name)

    query = """
    MATCH (c:Character {name: $char_name_param})
    WHERE c.is_deleted IS NULL OR c.is_deleted = FALSE

    // Do NOT add a WHERE clause after OPTIONAL MATCH; it will null-drop the row.
    OPTIONAL MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent)

    // Do NOT add a WHERE clause after OPTIONAL MATCH; it will null-drop the row.
    OPTIONAL MATCH (c)-[r]-()

    WITH
        c,
        collect(
            DISTINCT CASE
                WHEN dev.chapter_updated <= $chapter_limit_param THEN {
                    summary: dev.summary,
                    chapter: dev.chapter_updated,
                    is_provisional: coalesce(dev.is_provisional, FALSE)
                }
            END
        ) AS dev_events_raw,
        count(
            DISTINCT CASE
                WHEN coalesce(r.is_provisional, FALSE) = TRUE
                 AND coalesce(r.chapter_added, -1) <= $chapter_limit_param
                THEN r
            END
        ) AS provisional_rel_count

    WITH
        c,
        [e IN dev_events_raw WHERE e IS NOT NULL] AS dev_events,
        provisional_rel_count

    RETURN c.description AS description,
           c.status AS current_status,
           c.is_provisional AS char_is_provisional,
           dev_events,
           provisional_rel_count
    """

    params = {"char_name_param": canonical_name, "chapter_limit_param": chapter_limit}

    try:
        result = await neo4j_manager.execute_read_query(query, params)
    except ServiceUnavailable as e:
        logger.warning(
            "Neo4j service unavailable when fetching snippet for '%s': %s. Attempting single reconnect.",
            char_name,
            e,
        )
        try:
            await neo4j_manager.connect()
            result = await neo4j_manager.execute_read_query(query, params)
        except Exception as retry_exc:
            logger.error(
                "Retry after reconnect failed for character '%s': %s",
                char_name,
                retry_exc,
                exc_info=True,
            )
            return None
    except Exception as e:
        logger.error(
            f"Error fetching character info for snippet ({char_name}): {e}",
            exc_info=True,
        )
        return None

    if not result or not result[0]:
        logger.debug(
            "No detailed snippet info found for character '%s' up to chapter %d.",
            char_name,
            chapter_limit,
        )
        return None

    return _process_snippet_result(result[0], include_provisional=include_provisional)


async def find_thin_characters_for_enrichment() -> list[dict[str, Any]]:
    """Finds character nodes that are considered 'thin' (e.g., auto-created stubs)."""
    query = """
    MATCH (c:Character)
    WHERE c.description STARTS WITH 'Auto-created via relationship'
       OR c.description IS NULL
       OR c.description = ''
    RETURN c.name AS name
    LIMIT 20 // Limit to avoid overwhelming the LLM in one cycle
    """
    try:
        results = await neo4j_manager.execute_read_query(query)
        return results if results else []
    except Exception as e:
        logger.error(f"Error finding thin characters: {e}", exc_info=True)
        return []


# Native model functions for performance optimization
async def sync_characters(
    characters: list[CharacterProfile],
    chapter_number: int,
) -> bool:
    """
    Native model version of sync_characters.
    Persist character data directly from models without dict conversion.

    Args:
        characters: List of CharacterProfile models
        chapter_number: Current chapter for tracking updates

    Returns:
        True if successful, False otherwise
    """
    if not characters:
        logger.info("No characters to sync")
        return True

    # Validate all characters before syncing
    for char in characters:
        errors = validate_kg_object(char)
        if errors:
            logger.warning(f"Invalid CharacterProfile for '{char.name}': {errors}")

    try:
        cypher_builder = NativeCypherBuilder()
        statements = cypher_builder.batch_character_upsert_cypher(
            characters, chapter_number
        )

        if statements:
            await neo4j_manager.execute_cypher_batch(statements)

        logger.info(
            "Persisted %d character updates for chapter %d using native models.",
            len(characters),
            chapter_number,
        )

        # Update canonical name mapping deterministically (avoid stale accumulation).
        rebuild_character_name_map(characters)

        # P1.6: Post-write cache invalidation
        # Local import avoids circular import / eager import side effects.
        from data_access.cache_coordinator import clear_character_read_caches

        clear_character_read_caches()

        return True

    except Exception as exc:
        logger.error(
            "Error persisting character updates for chapter %d: %s",
            chapter_number,
            exc,
            exc_info=True,
        )
        return False


async def get_character_profiles() -> list[CharacterProfile]:
    """
    Native model version of get_character_profiles_from_db.
    Returns characters as model instances without dict conversion.

    Returns:
        List of CharacterProfile models
    """
    try:
        cypher_builder = NativeCypherBuilder()
        query, params = cypher_builder.character_fetch_cypher()

        results = await neo4j_manager.execute_read_query(query, params)
        characters = []

        for record in results:
            if record and record.get("c"):
                char = CharacterProfile.from_dict_record(record)
                characters.append(char)

        # Populate canonical name map for best-effort resolve helpers.
        rebuild_character_name_map(characters)

        logger.info("Fetched %d characters using native models", len(characters))
        return characters

    except Exception as exc:
        logger.error(f"Error fetching character profiles: {exc}", exc_info=True)
        return []


async def get_characters_for_chapter_context_native(
    chapter_number: int, limit: int = 10
) -> list[CharacterProfile]:
    """
    Get characters relevant for chapter context using native models.

    Args:
        chapter_number: Current chapter being processed
        limit: Maximum number of characters to return

    Returns:
        List of CharacterProfile models relevant to the chapter
    """
    try:
        query = """
        MATCH (c:Character)-[:APPEARS_IN]->(ch:Chapter)
        WHERE ch.number < $chapter_number
        WITH c, max(ch.number) as last_appearance
        ORDER BY last_appearance DESC
        LIMIT $limit

        OPTIONAL MATCH (c)-[r]->(other)
        RETURN c,
               collect({
                   target_name: other.name,
                   type: CASE
                       WHEN type(r) = 'RELATIONSHIP' THEN coalesce(r.type, type(r))
                       ELSE type(r)
                   END,
                   description: r.description
               }) as relationships
        """

        results = await neo4j_manager.execute_read_query(
            query, {"chapter_number": chapter_number, "limit": limit}
        )

        characters = []
        for record in results:
            if record and record.get("c"):
                char = CharacterProfile.from_dict_record(record)
                characters.append(char)

        logger.debug(
            "Fetched %d characters for chapter %d context using native models",
            len(characters),
            chapter_number,
        )

        return characters

    except Exception as exc:
        logger.error(
            "Error fetching characters for chapter %d context: %s",
            chapter_number,
            exc,
            exc_info=True,
        )
        return []
