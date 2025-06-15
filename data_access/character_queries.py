# data_access/character_queries.py
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from async_lru import alru_cache  # type: ignore

from neo4j.exceptions import ServiceUnavailable  # type: ignore

import config
from core_db.base_db_manager import neo4j_manager
from kg_constants import (
    KG_IS_PROVISIONAL,
    KG_NODE_CHAPTER_UPDATED,
    KG_REL_CHAPTER_ADDED,
)
from kg_maintainer.models import CharacterProfile

logger = logging.getLogger(__name__)


def generate_character_node_cypher(
    profile: CharacterProfile, chapter_number_for_delta: int = 0
) -> List[Tuple[str, Dict[str, Any]]]:
    """Create Cypher statements for a character update."""
    statements: List[Tuple[str, Dict[str, Any]]] = []

    props_from_profile = profile.to_dict()
    # Create a clean property dictionary for the node, excluding complex types.
    basic_props = {
        k: v
        for k, v in props_from_profile.items()
        if isinstance(v, (str, int, float, bool))
        and k not in ["name", "traits", "relationships"]
        and not k.startswith("development_in_chapter_")
        and not k.startswith("source_quality_chapter_")
    }

    # Add any updates from the profile's 'updates' field.
    if isinstance(profile.updates, dict):
        for k_update, v_update in profile.updates.items():
            if (
                isinstance(v_update, (str, int, float, bool))
                and not k_update.startswith("development_in_chapter_")
                and not k_update.startswith("source_quality_chapter_")
            ):
                if k_update not in basic_props:
                    basic_props[k_update] = v_update

    # Determine provisional status based on the current chapter's update source.
    current_chapter_source_quality_key = (
        f"source_quality_chapter_{chapter_number_for_delta}"
    )
    if (
        isinstance(profile.updates, dict)
        and profile.updates.get(current_chapter_source_quality_key)
        == "provisional_from_unrevised_draft"
    ):
        basic_props[KG_IS_PROVISIONAL] = True
    elif KG_IS_PROVISIONAL not in basic_props:
        # Default to False if not explicitly set by the update.
        basic_props[KG_IS_PROVISIONAL] = False

    # Ensure is_deleted is explicitly set to False for active characters.
    basic_props["is_deleted"] = False

    statements.append(
        (
            """
            MERGE (c:Character:Entity {name: $name})
            ON CREATE SET
                c += $props,
                c.created_ts = timestamp()
            ON MATCH SET
                c += $props,
                c.updated_ts = timestamp()
            """,
            {"name": profile.name, "props": basic_props},
        )
    )

    # Ensure the character is linked to the main novel node.
    statements.append(
        (
            """
            MATCH (ni:NovelInfo:Entity {id: $novel_id})
            MATCH (c:Character:Entity {name: $name})
            MERGE (ni)-[:HAS_CHARACTER]->(c)
            """,
            {"novel_id": config.MAIN_NOVEL_INFO_NODE_ID, "name": profile.name},
        )
    )

    # Process and link traits.
    if profile.traits:
        for trait_name in profile.traits:
            if isinstance(trait_name, str) and trait_name.strip():
                statements.append(
                    (
                        """
                        MATCH (c:Character:Entity {name: $name})
                        MERGE (t:Entity {name: $trait_name})
                            ON CREATE SET t:Trait, t.created_ts = timestamp()
                            ON MATCH SET t:Trait
                        MERGE (c)-[:HAS_TRAIT]->(t)
                        """,
                        {
                            "name": profile.name,
                            "trait_name": trait_name.strip(),
                        },
                    )
                )

    # Process and link development events for the current chapter.
    dev_event_key = f"development_in_chapter_{chapter_number_for_delta}"
    if isinstance(profile.updates, dict) and dev_event_key in profile.updates:
        dev_event_summary = profile.updates[dev_event_key]
        if isinstance(dev_event_summary, str) and dev_event_summary.strip():
            dev_event_id = (
                f"dev_{profile.name}_ch{chapter_number_for_delta}_"
                f"{hash(dev_event_summary)}"
            )
            dev_event_props = {
                "id": dev_event_id,
                "summary": dev_event_summary,
                "chapter_updated": chapter_number_for_delta,
                KG_IS_PROVISIONAL: basic_props.get(KG_IS_PROVISIONAL, False),
            }
            statements.append(
                (
                    """
                    MATCH (c:Character:Entity {name: $name})
                    MERGE (dev:Entity {id: $dev_event_id})
                        ON CREATE SET
                            dev:DevelopmentEvent,
                            dev = $props,
                            dev.created_ts = timestamp()
                        ON MATCH SET
                            dev:DevelopmentEvent,
                            dev = $props,
                            dev.updated_ts = timestamp()
                    MERGE (c)-[:DEVELOPED_IN_CHAPTER]->(dev)
                    """,
                    {
                        "name": profile.name,
                        "dev_event_id": dev_event_id,
                        "props": dev_event_props,
                    },
                )
            )

    # Process and link relationships.
    if profile.relationships:
        for target_char_name, rel_detail in profile.relationships.items():
            if isinstance(target_char_name, str) and target_char_name.strip():
                rel_type_str = "RELATED_TO"
                rel_cypher_props: Dict[str, Any] = {}
                if isinstance(rel_detail, str) and rel_detail.strip():
                    rel_cypher_props["description"] = rel_detail.strip()
                    if rel_detail.isupper() and " " not in rel_detail:
                        rel_type_str = rel_detail
                elif isinstance(rel_detail, dict):
                    rel_type_str = (
                        str(rel_detail.get("type", rel_type_str))
                        .upper()
                        .replace(" ", "_")
                    )
                    for k_rel, v_rel in rel_detail.items():
                        if isinstance(v_rel, (str, int, float, bool)):
                            rel_cypher_props[k_rel] = v_rel
                    rel_cypher_props.pop("type", None)

                rel_cypher_props[KG_REL_CHAPTER_ADDED] = chapter_number_for_delta
                rel_cypher_props[KG_IS_PROVISIONAL] = basic_props.get(
                    KG_IS_PROVISIONAL, False
                )
                rel_cypher_props["source_profile_managed"] = True

                statements.append(
                    (
                        """
                        MATCH (c1:Character:Entity {name: $source_name})
                        MERGE (c2:Entity {name: $target_name})
                            ON CREATE SET
                                c2.description = (
                                    'Auto-created via relationship from '
                                    + $source_name
                                ),
                                c2.created_ts = timestamp()
                        // CORRECTED: Removed "SET c2:Character" and "ON MATCH SET c2:Character"
                        // Now, it only creates a generic :Entity stub, which the Healer agent can enrich later.
                        // This prevents wrongly labeling a location like "Slums" as a "Character".

                        MERGE (
                            c1
                        )-[
                            r:DYNAMIC_REL {
                                type: $rel_type_str,
                                chapter_added: $chapter_num_delta
                            }
                        ]->(
                            c2
                        )
                            ON CREATE SET
                                r = $rel_props,
                                r.created_ts = timestamp()
                            ON MATCH SET
                                r += $rel_props,
                                r.updated_ts = timestamp()
                        """,
                        {
                            "source_name": profile.name,
                            "target_name": target_char_name.strip(),
                            "rel_type_str": rel_type_str,
                            "chapter_num_delta": chapter_number_for_delta,
                            "rel_props": rel_cypher_props,
                        },
                    )
                )

    return statements


async def sync_characters(
    profiles: Dict[str, CharacterProfile],
    chapter_number: int,
    full_sync: bool = False,
) -> bool:
    """Persist character data to Neo4j."""
    if full_sync:
        profile_dicts = {k: v.to_dict() for k, v in profiles.items()}
        return await sync_full_state_from_object_to_db(profile_dicts)

    statements: List[Tuple[str, Dict[str, Any]]] = []
    for profile in profiles.values():
        statements.extend(generate_character_node_cypher(profile, chapter_number))

    try:
        if statements:
            await neo4j_manager.execute_cypher_batch(statements)
        logger.info(
            "Persisted %d character updates for chapter %d.",
            len(profiles),
            chapter_number,
        )
        return True
    except Exception as exc:  # pragma: no cover - log and return failure
        logger.error(
            "Error persisting character updates for chapter %d: %s",
            chapter_number,
            exc,
            exc_info=True,
        )
        return False


async def sync_full_state_from_object_to_db(profiles_data: Dict[str, Any]) -> bool:
    logger.info("Synchronizing character profiles to Neo4j (non-destructive)...")

    novel_id_param = config.MAIN_NOVEL_INFO_NODE_ID
    statements: List[Tuple[str, Dict[str, Any]]] = []

    all_input_char_names: Set[str] = set(profiles_data.keys())

    try:
        existing_char_records = await neo4j_manager.execute_read_query(
            "MATCH (c:Character:Entity)"
            " WHERE c.is_deleted IS NULL OR c.is_deleted = FALSE"
            " RETURN c.name AS name"
        )
        existing_db_char_names: Set[str] = {
            record["name"]
            for record in existing_char_records
            if record and record["name"]
        }
    except Exception as e:
        logger.error(
            f"Failed to retrieve existing character names from DB: {e}", exc_info=True
        )
        return False

    chars_to_delete = existing_db_char_names - all_input_char_names
    if chars_to_delete:
        statements.append(
            (
                """
            MATCH (c:Character:Entity)
            WHERE c.name IN $char_names_to_delete
            SET c.is_deleted = TRUE
            """,
                {"char_names_to_delete": list(chars_to_delete)},
            )
        )

    for char_name, profile_dict in profiles_data.items():
        if not isinstance(profile_dict, dict):
            logger.warning(f"Skipping invalid profile for '{char_name}' (not a dict).")
            continue

        char_direct_props = {
            k: v
            for k, v in profile_dict.items()
            if isinstance(v, (str, int, float, bool)) and k != "name"
        }
        char_direct_props["is_deleted"] = False

        statements.append(
            (
                """
            MERGE (c:Character:Entity {name: $char_name_val})
            ON CREATE SET
                c += $props,
                c.created_ts = timestamp()
            ON MATCH SET
                c += $props,
                c.updated_ts = timestamp()
            """,
                {"char_name_val": char_name, "props": char_direct_props},
            )
        )

        statements.append(
            (
                """
            MATCH (ni:NovelInfo:Entity {id: $novel_id_param})
            MATCH (c:Character:Entity {name: $char_name_val})
            MERGE (ni)-[:HAS_CHARACTER]->(c)
            """,
                {"novel_id_param": novel_id_param, "char_name_val": char_name},
            )
        )

        current_profile_traits: Set[str] = {
            str(t).strip()
            for t in profile_dict.get("traits", [])
            if isinstance(t, str) and str(t).strip()
        }

        statements.append(
            (
                """
            MATCH (c:Character:Entity {name: $char_name_val})-[r:HAS_TRAIT]->(t:Trait:Entity)
            WHERE NOT t.name IN $current_traits_list
            DELETE r
            """,
                {
                    "char_name_val": char_name,
                    "current_traits_list": list(current_profile_traits),
                },
            )
        )

        if current_profile_traits:
            statements.append(
                (
                    """
                MATCH (c:Character:Entity {name: $char_name_val})
                UNWIND $current_traits_list AS trait_name_val
                MERGE (t:Entity {name: trait_name_val})
                    ON CREATE SET t:Trait, t.created_ts = timestamp()
                    ON MATCH SET t:Trait
                MERGE (c)-[:HAS_TRAIT]->(t)
                """,
                    {
                        "char_name_val": char_name,
                        "current_traits_list": list(current_profile_traits),
                    },
                )
            )

        statements.append(
            (
                """
            MATCH (c:Character:Entity {name: $char_name_val})-[r:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent:Entity)
            DETACH DELETE dev, r
            """,
                {"char_name_val": char_name},
            )
        )

        for key, value_str in profile_dict.items():
            if (
                key.startswith("development_in_chapter_")
                and isinstance(value_str, str)
                and value_str.strip()
            ):
                try:
                    chap_num_int = int(key.split("_")[-1])
                    dev_event_summary = value_str.strip()
                    dev_event_id = (
                        f"dev_{char_name}_ch{chap_num_int}_{hash(dev_event_summary)}"
                    )

                    dev_event_props = {
                        "id": dev_event_id,
                        "summary": dev_event_summary,
                        KG_NODE_CHAPTER_UPDATED: chap_num_int,
                        KG_IS_PROVISIONAL: profile_dict.get(
                            f"source_quality_chapter_{chap_num_int}"
                        )
                        == "provisional_from_unrevised_draft",
                    }

                    statements.append(
                        (
                            """
                        MATCH (c:Character:Entity {name: $char_name_val})
                        CREATE (dev:Entity:DevelopmentEvent)
                        SET dev = $props, dev.created_ts = timestamp()
                        CREATE (c)-[:DEVELOPED_IN_CHAPTER]->(dev)
                        """,
                            {"char_name_val": char_name, "props": dev_event_props},
                        )
                    )
                except ValueError:
                    logger.warning(
                        f"Could not parse chapter number from dev key: {key} for char {char_name}"
                    )

        profile_defined_rels = profile_dict.get("relationships", {})
        target_chars_in_profile_rels: Set[str] = set()
        if isinstance(profile_defined_rels, dict):
            target_chars_in_profile_rels = {
                str(k).strip() for k in profile_defined_rels.keys() if str(k).strip()
            }

        # Corrected DYNAMIC_REL deletion: Only delete profile-managed relationships
        # whose targets are no longer in the current profile's relationship list.
        statements.append(
            (
                """
            MATCH (c1:Character:Entity {name: $char_name_val})-[r:DYNAMIC_REL]->(c2:Entity)
            WHERE r.source_profile_managed = TRUE AND NOT c2.name IN $target_chars_list
            DELETE r
            """,
                # CORRECTED: Changed `c2:Character:Entity` to `c2:Entity` to catch relationships to non-character entities.
                {
                    "char_name_val": char_name,
                    "target_chars_list": list(target_chars_in_profile_rels),
                },
            )
        )

        if isinstance(profile_defined_rels, dict):
            for target_char_name_str, rel_detail in profile_defined_rels.items():
                target_char_name = str(target_char_name_str).strip()
                if not target_char_name:
                    continue

                rel_type_str = "RELATED_TO"
                # Ensure rel_cypher_props is re-initialized for each relationship
                rel_cypher_props = {
                    "source_profile_managed": True,
                    "confidence": 1.0,
                }

                chapter_added_val = config.KG_PREPOPULATION_CHAPTER_NUM
                if isinstance(rel_detail, dict) and "chapter_added" in rel_detail:
                    try:
                        chapter_added_val = int(rel_detail["chapter_added"])
                    except ValueError:
                        pass
                rel_cypher_props["chapter_added"] = chapter_added_val

                if isinstance(rel_detail, str) and rel_detail.strip():
                    rel_cypher_props["description"] = rel_detail.strip()
                    if rel_detail.isupper() and " " not in rel_detail:
                        rel_type_str = rel_detail
                elif isinstance(rel_detail, dict):
                    rel_type_str = (
                        str(rel_detail.get("type", rel_type_str))
                        .upper()
                        .replace(" ", "_")
                    )
                    for k_rel, v_rel in rel_detail.items():
                        if (
                            isinstance(v_rel, (str, int, float, bool))
                            and k_rel != "type"
                            and k_rel != "chapter_added"
                        ):  # chapter_added handled above
                            rel_cypher_props[k_rel] = v_rel

                rel_cypher_props[KG_IS_PROVISIONAL] = (
                    profile_dict.get(f"source_quality_chapter_{chapter_added_val}")
                    == "provisional_from_unrevised_draft"
                )

                statements.append(
                    (
                        """
                    MATCH (s:Character:Entity {name: $subject_param})
                    MERGE (o:Entity {name: $object_param})
                    MERGE (s)-[r:DYNAMIC_REL {type: $predicate_param, chapter_added: $chapter_added_val }]->(o)
                    ON CREATE SET r = $props_param, r.created_ts = timestamp()
                    ON MATCH SET  r += $props_param, r.updated_ts = timestamp()
                    """,
                        # CORRECTED: Changed `MATCH (o:Character...)` to `MERGE (o:Entity...)` to handle any entity type.
                        # Also changed `r = $props_param` on MATCH to `r += $props_param` to be non-destructive.
                        {
                            "subject_param": char_name,
                            "object_param": target_char_name,
                            "predicate_param": rel_type_str,
                            "chapter_added_val": chapter_added_val,
                            "props_param": rel_cypher_props,
                        },
                    )
                )

    statements.append(
        (
            """
        MATCH (t:Trait:Entity)
        WHERE NOT EXISTS((:Character:Entity)-[:HAS_TRAIT]->(t))
        DETACH DELETE t
        """,
            {},
        )
    )

    try:
        if statements:
            await neo4j_manager.execute_cypher_batch(statements)
        logger.info(
            f"Successfully synchronized {len(all_input_char_names)} character profiles to Neo4j."
        )
        return True
    except Exception as e:
        logger.error(f"Error synchronizing character profiles: {e}", exc_info=True)
        return False


@alru_cache(maxsize=128)
async def get_character_profile_by_name(name: str) -> Optional[CharacterProfile]:
    """Retrieve a single ``CharacterProfile`` from Neo4j by character name."""
    logger.info("Loading character profile '%s' from Neo4j...", name)

    query = (
        "MATCH (c:Character:Entity {name: $name})"
        " WHERE c.is_deleted IS NULL OR c.is_deleted = FALSE"
        " RETURN c"
    )
    results = await neo4j_manager.execute_read_query(query, {"name": name})
    if not results or not results[0].get("c"):
        logger.info("No character profile found for '%s'.", name)
        return None

    char_node = results[0]["c"]
    profile: Dict[str, Any] = dict(char_node)
    profile.pop("name", None)
    profile.pop("created_ts", None)
    profile.pop("updated_ts", None)

    traits_query = (
        "MATCH (:Character:Entity {name: $char_name})-[:HAS_TRAIT]->(t:Trait:Entity)"
        " RETURN t.name AS trait_name"
    )
    trait_results = await neo4j_manager.execute_read_query(
        traits_query, {"char_name": name}
    )
    profile["traits"] = sorted(
        [tr["trait_name"] for tr in trait_results if tr and tr.get("trait_name")]
    )

    rels_query = """
        MATCH (:Character:Entity {name: $char_name})-[r:DYNAMIC_REL]->(target:Entity)
        WHERE r.source_profile_managed = TRUE
        RETURN target.name AS target_name, properties(r) AS rel_props
    """
    rel_results = await neo4j_manager.execute_read_query(
        rels_query, {"char_name": name}
    )
    relationships: Dict[str, Any] = {}
    if rel_results:
        for rel_rec in rel_results:
            target_name = rel_rec.get("target_name")
            rel_props_full = rel_rec.get("rel_props", {})
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
        "MATCH (:Character:Entity {name: $char_name})-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent:Entity)\n"
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


async def get_character_profiles_from_db() -> Dict[str, CharacterProfile]:
    logger.info("Loading decomposed character profiles from Neo4j...")
    profiles_data: Dict[str, CharacterProfile] = {}

    char_query = (
        "MATCH (c:Character:Entity)"
        " WHERE c.is_deleted IS NULL OR c.is_deleted = FALSE"
        " RETURN c"
    )
    char_results = await neo4j_manager.execute_read_query(char_query)

    if not char_results:
        logger.info("No character profiles found in Neo4j.")
        return {}

    for record in char_results:
        char_node = record.get("c")
        if not char_node:
            continue
        char_name = char_node.get("name")
        if not char_name:
            continue

        profile = dict(char_node)
        profile.pop("name", None)
        profile.pop("created_ts", None)
        profile.pop("updated_ts", None)

        traits_query = "MATCH (:Character:Entity {name: $char_name})-[:HAS_TRAIT]->(t:Trait:Entity) RETURN t.name AS trait_name"
        trait_results = await neo4j_manager.execute_read_query(
            traits_query, {"char_name": char_name}
        )
        profile["traits"] = sorted(
            [tr["trait_name"] for tr in trait_results if tr and tr["trait_name"]]
        )

        rels_query = """
        MATCH (:Character:Entity {name: $char_name})-[r:DYNAMIC_REL]->(target:Entity)
        WHERE r.source_profile_managed = TRUE
        RETURN target.name AS target_name, properties(r) AS rel_props
        """
        # CORRECTED: `target:Character:Entity` changed to `target:Entity` to fetch all relationship types.
        rel_results = await neo4j_manager.execute_read_query(
            rels_query, {"char_name": char_name}
        )
        relationships = {}
        if rel_results:
            for rel_rec in rel_results:
                target_name = rel_rec.get("target_name")
                rel_props_full = rel_rec.get("rel_props", {})
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
                # Restore type if it was part of the key props for merge
                if "type" in rel_props_full:
                    rel_props_cleaned["type"] = rel_props_full["type"]
                # Restore chapter_added from the rel_props_full if it was there
                if "chapter_added" in rel_props_full:
                    rel_props_cleaned["chapter_added"] = rel_props_full["chapter_added"]

                if target_name:
                    relationships[target_name] = rel_props_cleaned
        profile["relationships"] = relationships

        dev_query = (
            "MATCH (:Character:Entity {name: $char_name})-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent:Entity)\n"
            f"RETURN dev.summary AS summary, dev.{KG_NODE_CHAPTER_UPDATED} AS chapter, dev.{KG_IS_PROVISIONAL} AS is_provisional, dev.id as dev_id\n"
            "ORDER BY dev.chapter_updated ASC"
        )
        dev_results = await neo4j_manager.execute_read_query(
            dev_query, {"char_name": char_name}
        )
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

        profiles_data[char_name] = CharacterProfile.from_dict(char_name, profile)

    logger.info(
        f"Successfully loaded and recomposed {len(profiles_data)} character profiles from Neo4j."
    )
    return profiles_data


async def get_character_info_for_snippet_from_db(
    char_name: str, chapter_limit: int
) -> Optional[Dict[str, Any]]:
    query = """
    MATCH (c:Character:Entity {name: $char_name_param})
    WHERE c.is_deleted IS NULL OR c.is_deleted = FALSE

    // Subquery to get the most recent non-provisional development event
    CALL (c) {
        OPTIONAL MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent:Entity)
        WHERE dev.chapter_updated <= $chapter_limit_param
          AND (dev.is_provisional IS NULL OR dev.is_provisional = FALSE)
        RETURN dev AS dev_np
        ORDER BY dev.chapter_updated DESC
        LIMIT 1
    }

    // Subquery to get the most recent provisional development event
    CALL (c) {
        OPTIONAL MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent:Entity)
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
                MATCH (c)-[r:DYNAMIC_REL]-(:Entity)
                WHERE r.is_provisional = TRUE AND r.chapter_added <= $chapter_limit_param
            } OR
            EXISTS {
                MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent:Entity)
                WHERE dev.is_provisional = TRUE AND dev.chapter_updated <= $chapter_limit_param
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
    params = {"char_name_param": char_name, "chapter_limit_param": chapter_limit}
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


async def find_thin_characters_for_enrichment() -> List[Dict[str, Any]]:
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
