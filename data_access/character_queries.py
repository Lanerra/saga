# data_access/character_queries.py
import hashlib
from typing import Any

import structlog
from async_lru import alru_cache  # type: ignore[import-untyped]
from neo4j.exceptions import ServiceUnavailable

import config
import utils
from core.db_manager import neo4j_manager
from core.schema_validator import validate_kg_object
from models import CharacterProfile
from models.kg_constants import KG_IS_PROVISIONAL, KG_NODE_CHAPTER_UPDATED

from .cypher_builders.native_builders import NativeCypherBuilder

# Mapping from normalized character names to canonical display names
CHAR_NAME_TO_CANONICAL: dict[str, str] = {}

# Mapping from normalized trait names to canonical display names
TRAIT_NAME_TO_CANONICAL: dict[str, str] = {}


def resolve_character_name(name: str) -> str:
    """Return canonical character name for a display variant."""
    if not name:
        return name
    return CHAR_NAME_TO_CANONICAL.get(utils._normalize_for_id(name), name)


logger = structlog.get_logger(__name__)


@alru_cache(maxsize=128)
async def get_character_profile_by_name(name: str) -> CharacterProfile | None:
    """Retrieve a single ``CharacterProfile`` from Neo4j by character name."""
    canonical_name = resolve_character_name(name)
    logger.info(f"Loading character profile '{canonical_name}' from Neo4j...")

    query = (
        "MATCH (c:Character {name: $name})"
        " WHERE c.is_deleted IS NULL OR c.is_deleted = FALSE"
        " RETURN c"
    )
    results = await neo4j_manager.execute_read_query(query, {"name": canonical_name})
    if not results or not results[0].get("c"):
        logger.info(f"No character profile found for '{canonical_name}'.")
        return None

    char_node = results[0]["c"]
    profile: dict[str, Any] = dict(char_node)
    profile.pop("name", None)
    profile.pop("created_ts", None)
    profile.pop("updated_ts", None)

    traits_query = (
        "MATCH (:Character {name: $char_name})-[:HAS_TRAIT]->(t:Trait)"
        " RETURN t.name AS trait_name"
    )
    trait_results = await neo4j_manager.execute_read_query(
        traits_query, {"char_name": name}
    )
    profile["traits"] = sorted(
        [tr["trait_name"] for tr in trait_results if tr and tr.get("trait_name")]
    )

    rels_query = """
        MATCH (:Character {name: $char_name})-[r]->(target)
        WHERE coalesce(r.source_profile_managed, false) = true
        RETURN target.name AS target_name, type(r) AS rel_type, properties(r) AS rel_props
    """
    rel_results = await neo4j_manager.execute_read_query(
        rels_query, {"char_name": name}
    )
    relationships: dict[str, Any] = {}
    if rel_results:
        for rel_rec in rel_results:
            target_name = rel_rec.get("target_name")
            rel_props_full = rel_rec.get("rel_props", {})
            rel_props_cleaned = {}
            if isinstance(rel_props_full, dict):
                rel_props_cleaned = {
                    k: v
                    for k, v in rel_props_full.items()
                    if k
                    not in [
                        "created_ts",
                        "updated_ts",
                        "source_profile_managed",
                        "chapter_added",
                    ]
                }
            if "type" in rel_props_full:
                rel_props_cleaned["type"] = rel_props_full["type"]
            if "chapter_added" in rel_props_full:
                rel_props_cleaned["chapter_added"] = rel_props_full["chapter_added"]
            if target_name:
                relationships[target_name] = rel_props_cleaned
    profile["relationships"] = relationships

    dev_query = (
        "MATCH (:Character {name: $char_name})-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent)\n"
        f"RETURN dev.summary AS summary, dev.{KG_NODE_CHAPTER_UPDATED} AS chapter, dev.{KG_IS_PROVISIONAL} AS is_provisional, dev.id as dev_id\n"
        "ORDER BY dev.chapter_updated ASC"
    )
    dev_results = await neo4j_manager.execute_read_query(dev_query, {"char_name": name})
    if dev_results:
        for dev_rec in dev_results:
            chapter_num = dev_rec.get("chapter")
            summary = dev_rec.get("summary")
            if chapter_num is not None and summary is not None:
                dev_key = f"development_in_chapter_{chapter_num}"
                profile[dev_key] = summary
                if dev_rec.get(KG_IS_PROVISIONAL):
                    profile[f"source_quality_chapter_{chapter_num}"] = (
                        "provisional_from_unrevised_draft"
                    )

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


@alru_cache(maxsize=128)
# NOTE: Legacy get_character_profiles_from_db() removed. Use get_character_profiles().


async def get_character_info_for_snippet_from_db(
    char_name: str, chapter_limit: int
) -> dict[str, Any] | None:
    canonical_name = resolve_character_name(char_name)
    query = """
    MATCH (c:Character {name: $char_name_param})
    WHERE c.is_deleted IS NULL OR c.is_deleted = FALSE

    // Subquery to get the most recent non-provisional development event
    CALL (c) {
        OPTIONAL MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent)
        WHERE dev.chapter_updated <= $chapter_limit_param
          AND (dev.is_provisional IS NULL OR dev.is_provisional = FALSE)
        RETURN dev AS dev_np
        ORDER BY dev.chapter_updated DESC
        LIMIT 1
    }

    // Subquery to get the most recent provisional development event
    CALL (c) {
        OPTIONAL MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent)
        WHERE dev.chapter_updated <= $chapter_limit_param
          AND dev.is_provisional = TRUE
        RETURN dev AS dev_p
        ORDER BY dev.chapter_updated DESC
        LIMIT 1
    }

    // Subquery to check for the existence of any provisional data related to the character
    CALL (c) {
        RETURN (
            c.is_provisional = TRUE OR
            EXISTS {
                MATCH (c)-[r]-()
                WHERE coalesce(r.is_provisional, FALSE) = TRUE AND coalesce(r.chapter_added, -1) <= $chapter_limit_param
            } OR
            EXISTS {
                MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent)
                WHERE coalesce(dev.is_provisional, FALSE) = TRUE AND coalesce(dev.chapter_updated, -1) <= $chapter_limit_param
            }
        ) AS is_provisional_flag
    }

    WITH c, dev_np, dev_p, is_provisional_flag

    // Determine the single most current development event
    WITH c, is_provisional_flag,
         CASE
           WHEN dev_p IS NOT NULL AND (dev_np IS NULL OR dev_p.chapter_updated >= dev_np.chapter_updated)
           THEN dev_p
           ELSE dev_np
         END AS most_current_dev_event

    RETURN c.description AS description,
           c.status AS current_status,
           most_current_dev_event,
           is_provisional_flag AS is_provisional_overall
    """
    params = {"char_name_param": canonical_name, "chapter_limit_param": chapter_limit}
    try:
        result = await neo4j_manager.execute_read_query(query, params)
    except ServiceUnavailable as e:
        logger.warning(
            "Neo4j service unavailable when fetching snippet for '%s': %s."
            " Attempting single reconnect.",
            char_name,
            e,
        )
        try:
            await neo4j_manager.connect()
            result = await neo4j_manager.execute_read_query(query, params)
            if result and result[0]:
                record = result[0]
                most_current_dev_event_node = record.get("most_current_dev_event")
                dev_note = (
                    most_current_dev_event_node.get("summary", "N/A")
                    if most_current_dev_event_node
                    else "N/A"
                )

                return {
                    "description": record.get("description"),
                    "current_status": record.get("current_status"),
                    "most_recent_development_note": dev_note,
                    "is_provisional_overall": record.get(
                        "is_provisional_overall", False
                    ),
                }
            logger.debug(
                f"No detailed snippet info found for character '{char_name}' up to chapter {chapter_limit}."
            )
        except Exception as retry_exc:  # pragma: no cover - log and return
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

    if result and result[0]:
        record = result[0]
        most_current_dev_event_node = record.get("most_current_dev_event")
        dev_note = (
            most_current_dev_event_node.get("summary", "N/A")
            if most_current_dev_event_node
            else "N/A"
        )

        return {
            "description": record.get("description"),
            "current_status": record.get("current_status"),
            "most_recent_development_note": dev_note,
            "is_provisional_overall": record.get("is_provisional_overall", False),
        }
    logger.debug(
        "No detailed snippet info found for character '%s' up to chapter %d.",
        char_name,
        chapter_limit,
    )
    return None


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

        # Update canonical name mapping
        for char in characters:
            CHAR_NAME_TO_CANONICAL[utils._normalize_for_id(char.name)] = char.name

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
                   type: r.type,
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
