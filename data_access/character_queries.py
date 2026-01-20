# data_access/character_queries.py
from collections import defaultdict
from typing import Any

import structlog
from async_lru import alru_cache  # type: ignore[import-untyped]
from neo4j.exceptions import Neo4jError, ServiceUnavailable

import utils
from core.db_manager import neo4j_manager
from core.exceptions import handle_database_error
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
    """Clear the in-process character name canonicalization map.

    Notes:
        This only clears in-memory state used by [`resolve_character_name()`](data_access/character_queries.py:43).
        It does not modify Neo4j.
    """
    CHAR_NAME_TO_CANONICAL.clear()


def rebuild_character_name_map(characters: list["CharacterProfile"]) -> None:
    """Rebuild the character name canonicalization map from a list of profiles.

    Args:
        characters: Character profiles to use as the authoritative source of canonical
            display names.

    Returns:
        None.

    Notes:
        This clears existing entries to avoid stale accumulation across runs/tests. This is
        an in-process cache and is populated as a side effect of:
        - [`sync_characters()`](data_access/character_queries.py:484) (write path), and
        - [`get_character_profiles()`](data_access/character_queries.py:543) (read path).
    """
    CHAR_NAME_TO_CANONICAL.clear()
    for char in characters:
        if isinstance(char, CharacterProfile) and char.name:
            CHAR_NAME_TO_CANONICAL[utils._normalize_for_id(char.name)] = char.name


def resolve_character_name(name: str) -> str:
    """Return a canonical character display name when a mapping is known.

    Args:
        name: A character name variant.

    Returns:
        The canonical display name when the in-memory mapping has an entry. Otherwise
        returns `name` unchanged.

    Notes:
        This is best-effort and does not perform any database IO. Callers that require
        authoritative resolution must query Neo4j.
    """
    if not name:
        return name
    return CHAR_NAME_TO_CANONICAL.get(utils._normalize_for_id(name), name)


logger = structlog.get_logger(__name__)


@alru_cache(maxsize=128)
async def get_character_profile_by_name(name: str, *, include_provisional: bool = False) -> CharacterProfile | None:
    """Return a character profile by name.

    Args:
        name: Character display name. This is passed through
            [`resolve_character_name()`](data_access/character_queries.py:43) before querying.
        include_provisional: Whether to include provisional relationship and event data in the
            returned profile.

    Returns:
        A `CharacterProfile` when a matching character exists and is not deleted. Returns
        None when the character does not exist.

        The returned profile includes:
        - `traits`: sorted list of trait names
        - `relationships`: mapping keyed by `target_name`
            - if there is exactly one relationship to a target, the value is a dict with at
              least `type` and optional `description` / `chapter_added`
            - if there are multiple relationships to a target, the value is a list of those
              dicts (stable-sorted)

    Notes:
        Provisional semantics:
            - By default, provisional relationships and development events are excluded unless
              `include_provisional=True`.
            - Node-level provisional status on the character is preserved on the returned
              profile so callers can make data-quality decisions.

        Cache semantics:
            This function is cached (read-through). Callers should treat returned model
            instances as immutable to avoid leaking mutations across cache hits. Write paths
            should invalidate via [`clear_character_read_caches()`](data_access/cache_coordinator.py:31).

        Query contract:
            This read path filters to relationships with `source_profile_managed=true` to
            avoid surfacing unmanaged edges.
    """
    canonical_name = resolve_character_name(name)
    logger.info(f"Loading character profile '{canonical_name}' from Neo4j...")

    query = """
        MATCH (c:Character {name: $name})
        WHERE c.is_deleted IS NULL OR c.is_deleted = FALSE

        // Do NOT add a WHERE clause after OPTIONAL MATCH; it will null-drop the row.
        OPTIONAL MATCH (c)-[r]->(target)

        WITH
            c,
            collect(
                DISTINCT CASE
                    WHEN coalesce(r.source_profile_managed, false) = true
                     AND (
                          $include_provisional = false
                          OR coalesce(r.is_provisional, FALSE) = FALSE
                     )
                    THEN {
                        target_name: target.name,
                        rel_type: type(r),
                        rel_props: properties(r)
                    }
                END
            ) AS relationships_raw

        RETURN
            c,
            coalesce(c.traits, []) AS traits,
            [rel IN relationships_raw WHERE rel IS NOT NULL] AS relationships
    """

    results = await neo4j_manager.execute_read_query(query, {"name": canonical_name, "include_provisional": include_provisional})
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

    # Relationships are represented as a dict keyed by target_name.
    #
    # Remediation: relationship projection must NOT be lossy when there are multiple
    # relationships to the same target (e.g., FRIEND_OF + ENEMY_OF to "Bob").
    #
    # Representation choice (least-breaking):
    # - If there is exactly one relationship to a target, keep the legacy shape:
    #     relationships[target_name] -> { ...props..., "type": "REL_TYPE" }
    # - If there are multiple relationships to a target, store a list:
    #     relationships[target_name] -> [{...}, {...}, ...]
    #
    # This preserves existing callers/tests for the common single-relationship case,
    # while making collisions explicit rather than overwriting.
    rels_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for rel_rec in record["relationships"]:
        if not rel_rec or not rel_rec.get("target_name"):
            continue

        target_name = rel_rec["target_name"]

        rel_props_full = rel_rec.get("rel_props", {})
        rel_props_cleaned: dict[str, Any] = {}

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

        # P1.7: Canonical relationship typing = type(r) from Cypher (`rel_type`).
        # Fall back to legacy property-based typing if present.
        rel_type = rel_rec.get("rel_type")
        if rel_type:
            rel_props_cleaned["type"] = rel_type
        elif isinstance(rel_props_full, dict) and "type" in rel_props_full:
            rel_props_cleaned["type"] = rel_props_full["type"]

        # Preserve chapter_added when present (callers sometimes use it).
        if isinstance(rel_props_full, dict) and "chapter_added" in rel_props_full:
            rel_props_cleaned["chapter_added"] = rel_props_full["chapter_added"]

        rels_by_target[target_name].append(rel_props_cleaned)

    # Deterministic/stable output:
    # - sort targets
    # - sort multi-relationship lists within each target
    relationships: dict[str, Any] = {}
    for target_name in sorted(rels_by_target.keys()):
        rel_list = rels_by_target[target_name]
        rel_list_sorted = sorted(
            rel_list,
            key=lambda r: (
                str(r.get("type", "")),
                str(r.get("description", "")),
                str(r.get("chapter_added", "")),
            ),
        )
        relationships[target_name] = rel_list_sorted[0] if len(rel_list_sorted) == 1 else rel_list_sorted

    profile["relationships"] = relationships

    return CharacterProfile.from_dict(name, profile)


@alru_cache(maxsize=128)
async def get_character_profile_by_id(character_id: str, *, include_provisional: bool = False) -> CharacterProfile | None:
    """Return a character profile by id.

    Args:
        character_id: Canonical character id stored on the `:Character` node.
        include_provisional: Whether to include provisional relationship and event data in the
            returned profile.

    Returns:
        A `CharacterProfile` when a matching character exists and is not deleted. Returns
        None when the character does not exist.

        The returned profile includes:
        - `traits`: sorted list of trait names
        - `relationships`: mapping keyed by `target_name` whose values are dicts containing at
          least `type` and optional `description` / `chapter_added`.

    Notes:
        Provisional semantics:
            - By default, provisional relationships and development events are excluded unless
              `include_provisional=True`.
            - Node-level provisional status on the character is preserved on the returned
              profile so callers can make data-quality decisions.

        Relationship shape:
            The relationships mapping preserves multiple relationship types to the same
            target. If there is exactly one relationship to a target, it's returned as a dict.
            If there are multiple relationships to the same target, they're returned as a list
            sorted by type, description, and chapter_added.

        Cache semantics:
            This function is cached (read-through). Callers should treat returned model
            instances as immutable. Write paths should invalidate via
            [`clear_character_read_caches()`](data_access/cache_coordinator.py:31).
    """
    logger.info(f"Loading character profile with ID '{character_id}' from Neo4j...")

    query = """
        MATCH (c:Character {id: $character_id})
        WHERE c.is_deleted IS NULL OR c.is_deleted = FALSE

        // Do NOT add a WHERE clause after OPTIONAL MATCH; it will null-drop the row.
        OPTIONAL MATCH (c)-[r]->(target)

        WITH
            c,
            collect(
                DISTINCT CASE
                    WHEN coalesce(r.source_profile_managed, false) = true
                     AND (
                          $include_provisional = false
                          OR coalesce(r.is_provisional, FALSE) = FALSE
                     )
                    THEN {
                        target_name: target.name,
                        rel_type: type(r),
                        rel_props: properties(r)
                    }
                END
            ) AS relationships_raw

        RETURN
            c,
            coalesce(c.traits, []) AS traits,
            [rel IN relationships_raw WHERE rel IS NOT NULL] AS relationships
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

    # Collect all relationships, grouping by target_name to preserve multiple relationship
    # types to the same target (matching get_character_profile_by_name behavior)
    from collections import defaultdict
    rels_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)

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
        
        rels_by_target[target_name].append(rel_props_cleaned)

    # Build final relationships dict with consistent shape:
    # - Single relationship: dict
    # - Multiple relationships: list
    relationships: dict[str, Any] = {}
    for target_name in sorted(rels_by_target.keys()):
        rel_list = rels_by_target[target_name]
        rel_list_sorted = sorted(
            rel_list,
            key=lambda r: (
                str(r.get("type", "")),
                str(r.get("description", "")),
                str(r.get("chapter_added", "")),
            ),
        )
        relationships[target_name] = rel_list_sorted[0] if len(rel_list_sorted) == 1 else rel_list_sorted
    profile["relationships"] = relationships

    return CharacterProfile.from_dict(name, profile)


async def get_all_character_names() -> list[str]:
    """Return all character names.

    Returns:
        A list of character names, ordered alphabetically.

    Notes:
        This function does not currently filter provisional characters.
    """
    query = "MATCH (c:Character) " "WHERE c.is_deleted IS NULL OR c.is_deleted = FALSE " "RETURN c.name AS name ORDER BY c.name"
    results = await neo4j_manager.execute_read_query(query)
    return [record["name"] for record in results if record.get("name")]


def _process_snippet_result(record: dict[str, Any], *, include_provisional: bool) -> dict[str, Any]:
    """Normalize a snippet query record into a stable response shape.

    Args:
        record: A single Neo4j record dict produced by
            [`get_character_info_for_snippet_from_db()`](data_access/character_queries.py:376).
        include_provisional: Unused, kept for backward compatibility.

    Returns:
        A dictionary with keys:
        - `description`
        - `current_status`
        - `is_provisional_overall`

    Notes:
        Provisional semantics:
            `is_provisional_overall` reflects whether the character node or any
            relationships are provisional.
    """
    char_is_provisional = record.get("char_is_provisional", False)
    has_provisional_relationships = record.get("provisional_rel_count", 0) > 0
    is_provisional_overall = char_is_provisional or has_provisional_relationships

    return {
        "description": record.get("description"),
        "current_status": record.get("current_status"),
        "is_provisional_overall": is_provisional_overall,
    }


async def get_character_info_for_snippet_from_db(
    char_name: str,
    chapter_limit: int,
    *,
    include_provisional: bool = False,
) -> dict[str, Any] | None:
    """Return condensed character info suitable for snippet context.

    Args:
        char_name: Character name. This is passed through
            [`resolve_character_name()`](data_access/character_queries.py:43) before querying.
        chapter_limit: Upper bound on `chapter_added` used to constrain the view.
        include_provisional: Unused, kept for backward compatibility.

    Returns:
        A dictionary with keys:
        - `description`
        - `current_status`
        - `is_provisional_overall`

        Returns None when the character is not found.

    Raises:
        DatabaseConnectionError: On connection failures
        DatabaseError: On other database errors

    Notes:
        Provisional semantics:
            `is_provisional_overall` reflects whether the character node or any
            relationships up to `chapter_limit` are provisional.

        Error behavior:
            This function retries once on `neo4j.exceptions.ServiceUnavailable` by reconnecting
            and reissuing the query. Other failures raise DatabaseError.
    """
    canonical_name = resolve_character_name(char_name)

    query = """
    MATCH (c:Character {name: $char_name_param})
    WHERE c.is_deleted IS NULL OR c.is_deleted = FALSE

    // Do NOT add a WHERE clause after OPTIONAL MATCH; it will null-drop the row.
    OPTIONAL MATCH (c)-[r]-()

    WITH
        c,
        count(
            DISTINCT CASE
                WHEN coalesce(r.is_provisional, FALSE) = TRUE
                 AND coalesce(r.chapter_added, -1) <= $chapter_limit_param
                THEN r
            END
        ) AS provisional_rel_count

    RETURN c.description AS description,
           c.status AS current_status,
           c.is_provisional AS char_is_provisional,
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
        await neo4j_manager.connect()
        try:
            result = await neo4j_manager.execute_read_query(query, params)
        except (ServiceUnavailable, Neo4jError, KeyError, ValueError, TypeError) as retry_error:
            raise handle_database_error(
                "get_character_info_for_snippet_from_db (retry)",
                retry_error,
                character=char_name,
                chapter_limit=chapter_limit,
            ) from retry_error
    except (Neo4jError, KeyError, ValueError, TypeError) as e:
        raise handle_database_error(
            "get_character_info_for_snippet_from_db",
            e,
            character=char_name,
            chapter_limit=chapter_limit,
        ) from e

    if not result or not result[0]:
        logger.debug(
            "No detailed snippet info found for character '%s' up to chapter %d.",
            char_name,
            chapter_limit,
        )
        return None

    return _process_snippet_result(result[0], include_provisional=include_provisional)


async def find_thin_characters_for_enrichment() -> list[dict[str, Any]]:
    """Find thin character nodes suitable for enrichment.

    Returns:
        A list of dictionaries with at least `name` for up to 20 characters considered thin.

    Notes:
        This is a diagnostic discovery query intended to seed enrichment workflows. It is not
        a strict completeness guarantee.
    """
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
    except (Neo4jError, KeyError, ValueError, TypeError) as e:
        logger.error(f"Error finding thin characters: {e}", exc_info=True)
        return []


# Native model functions for performance optimization
async def sync_characters(
    characters: list[CharacterProfile],
    chapter_number: int,
) -> bool:
    """Persist character profiles to Neo4j using the native Cypher builder.

    Args:
        characters: Character profiles to upsert.
        chapter_number: Chapter number used for provenance and update tracking.

    Returns:
        True when the batch write completed successfully. False when a write error occurred.

    Notes:
        Cache semantics:
            This is a write path. On success it invalidates character read caches via
            [`clear_character_read_caches()`](data_access/cache_coordinator.py:31).

        In-memory name resolution:
            On success it rebuilds the canonical display-name mapping used by
            [`resolve_character_name()`](data_access/character_queries.py:43).
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
        statements = cypher_builder.batch_character_upsert_cypher(characters, chapter_number)

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

    except (Neo4jError, KeyError, ValueError, TypeError) as exc:
        logger.error(
            "Error persisting character updates for chapter %d: %s",
            chapter_number,
            exc,
            exc_info=True,
        )
        return False


async def get_character_profiles() -> list[CharacterProfile]:
    """Return all character profiles.

    Returns:
        A list of `CharacterProfile` instances. Returns an empty list on query failures.

    Notes:
        In-memory name resolution:
            This call rebuilds the canonical display-name mapping used by
            [`resolve_character_name()`](data_access/character_queries.py:43).

        Error behavior:
            This function logs exceptions and returns an empty list rather than raising.
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

    except (Neo4jError, KeyError, ValueError, TypeError) as exc:
        logger.error(f"Error fetching character profiles: {exc}", exc_info=True)
        return []


async def get_characters_for_chapter_context_native(chapter_number: int, limit: int = 5) -> list[CharacterProfile]:
    """Return characters relevant for chapter context.

    Args:
        chapter_number: Current chapter being processed. Only earlier chapters are considered.
        limit: Maximum number of characters to return.

    Returns:
        A list of `CharacterProfile` instances, ordered by most recent appearance.

    Notes:
        Error behavior:
            This function logs exceptions and returns an empty list rather than raising.
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

        results = await neo4j_manager.execute_read_query(query, {"chapter_number": chapter_number, "limit": limit})

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

    except (Neo4jError, KeyError, ValueError, TypeError) as exc:
        logger.error(
            "Error fetching characters for chapter %d context: %s",
            chapter_number,
            exc,
            exc_info=True,
        )
        return []
