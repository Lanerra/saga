# data_access/cypher_builders/character_cypher.py
from typing import Any

import structlog

import config
import utils
from data_access.kg_queries import validate_relationship_type
from models import CharacterProfile
from models.kg_constants import KG_IS_PROVISIONAL, KG_REL_CHAPTER_ADDED

TRAIT_NAME_TO_CANONICAL: dict[str, str] = {}

logger = structlog.get_logger(__name__)


def generate_character_node_cypher(
    profile: CharacterProfile, chapter_number_for_delta: int = 0
) -> list[tuple[str, dict[str, Any]]]:
    """Create Cypher statements for a character update."""
    statements: list[tuple[str, dict[str, Any]]] = []

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
                canonical = utils.normalize_trait_name(trait_name)
                if canonical:
                    TRAIT_NAME_TO_CANONICAL[canonical] = trait_name
                    statements.append(
                        (
                            """
                            MATCH (c:Character:Entity {name: $name})
                            MERGE (t:Trait:Entity {name: $trait_name})
                                ON CREATE SET t.created_ts = timestamp()
                            MERGE (c)-[:HAS_TRAIT]->(t)
                            """,
                            {
                                "name": profile.name,
                                "trait_name": canonical,
                            },
                        )
                    )

    # Process and link development events for the current chapter.
    dev_event_key = f"development_in_chapter_{chapter_number_for_delta}"
    if isinstance(profile.updates, dict) and dev_event_key in profile.updates:
        dev_event_summary = profile.updates[dev_event_key]
        if isinstance(dev_event_summary, str) and dev_event_summary.strip():
            stable_hash = hashlib.sha1(
                f"{profile.name}|{chapter_number_for_delta}|{dev_event_summary}".encode()
            ).hexdigest()[:16]
            dev_event_id = (
                f"dev_{utils._normalize_for_id(profile.name)}_ch{chapter_number_for_delta}_"
                f"{stable_hash}"
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
                rel_cypher_props: dict[str, Any] = {}
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

                # Normalize/validate relationship type before using it in Cypher
                rel_type_str = validate_relationship_type(rel_type_str)

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
                                c2:Entity,
                                c2.description = (
                                    'Auto-created via relationship from '
                                    + $source_name
                                ),
                                c2.created_ts = timestamp()

                        MERGE (
                        c1
                    )-[
                        r:`$rel_type_str`
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


import hashlib
