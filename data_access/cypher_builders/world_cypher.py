import json
from typing import Any

import structlog  # MODIFIED
from config import settings  # MODIFIED
from kg_constants import (
    KG_IS_PROVISIONAL,
    KG_NODE_CREATED_CHAPTER,
)
from kg_maintainer.models import WorldItem
from utils import kg_property_keys as kg_keys

logger = structlog.get_logger(__name__)  # MODIFIED


def generate_world_element_node_cypher(
    item: WorldItem, chapter_number_for_delta: int = 0
) -> list[tuple[str, dict[str, Any]]]:
    """Create Cypher statements for a world element update."""

    statements: list[tuple[str, dict[str, Any]]] = []

    node_props = {
        "name": item.name,
        "category": item.category,
        KG_NODE_CREATED_CHAPTER: item.created_chapter,
    }
    node_props["is_deleted"] = False

    current_chapter_source_quality_key = kg_keys.source_quality_key(
        chapter_number_for_delta
    )
    if (
        isinstance(item.properties, dict)
        and item.properties.get(current_chapter_source_quality_key)
        == "provisional_from_unrevised_draft"
    ):
        node_props[KG_IS_PROVISIONAL] = True
    elif KG_IS_PROVISIONAL not in node_props:
        node_props[KG_IS_PROVISIONAL] = item.is_provisional

    if isinstance(item.properties, dict):
        for key, value in item.properties.items():
            if (
                isinstance(value, (str, int, float, bool))
                and key not in node_props
                and not key.startswith(kg_keys.ELABORATION_PREFIX)
                and not key.startswith(kg_keys.SOURCE_QUALITY_PREFIX)
                and not key.startswith(kg_keys.ADDED_PREFIX)
                and key not in ["goals", "rules", "key_elements", "traits"]
            ):
                node_props[key] = value
            elif isinstance(value, (list, dict)) and key not in [
                "goals",
                "rules",
                "key_elements",
                "traits",
            ]:
                try:
                    node_props[key] = json.dumps(value, ensure_ascii=False)
                except TypeError:
                    logger.warning(
                        "Could not JSON serialize property '%s' for WorldElement '%s'. Skipping.",
                        key,
                        item.id,
                    )

    statements.append(
        (
            """
            MERGE (we:Entity {id: $id})
            ON CREATE SET
                we:WorldElement,
                we += $props,
                we.created_ts = timestamp()
            ON MATCH SET
                we:WorldElement,
                we += $props,
                we.updated_ts = timestamp()
            """,
            {"id": item.id, "props": node_props},
        )
    )

    statements.append(
        (
            """
            MATCH (wc:WorldContainer:Entity {id: $wc_id})
            MATCH (we:WorldElement:Entity {id: $we_id})
            MERGE (wc)-[:CONTAINS_ELEMENT]->(we)
            """,
            {
                "wc_id": settings.MAIN_WORLD_CONTAINER_NODE_ID,
                "we_id": item.id,
            },  # MODIFIED
        )
    )

    list_properties_to_relate = {
        "goals": "HAS_GOAL",
        "rules": "HAS_RULE",
        "key_elements": "HAS_KEY_ELEMENT",
        "traits": "HAS_TRAIT_ASPECT",
    }

    for prop_key, rel_type in list_properties_to_relate.items():
        list_value = []
        if isinstance(item.properties, dict):
            list_value = item.properties.get(prop_key, [])

        if isinstance(list_value, list):
            for value_str_unstripped in list_value:
                if isinstance(value_str_unstripped, str):
                    value_str = value_str_unstripped.strip()
                    if value_str:
                        statements.append(
                            (
                                f"""
                                MATCH (
                                    we:WorldElement:Entity {{id: $we_id}}
                                )
                                MERGE (
                                    v:Entity:ValueNode {{
                                        value: $value_str,
                                        type: $prop_key
                                    }}
                                )
                                  ON CREATE SET v.created_ts = timestamp()
                                MERGE (we)-[:{rel_type}]->(v)
                                """,
                                {
                                    "we_id": item.id,
                                    "value_str": value_str,
                                    "prop_key": prop_key,
                                },
                            )
                        )

    elab_event_key = kg_keys.elaboration_key(chapter_number_for_delta)
    if isinstance(item.properties, dict) and elab_event_key in item.properties:
        elab_summary = item.properties[elab_event_key]
        if isinstance(elab_summary, str) and elab_summary.strip():
            elab_event_id = (
                f"elab_{item.id}_ch{chapter_number_for_delta}_{hash(elab_summary)}"
            )
            elab_props = {
                "id": elab_event_id,
                "summary": elab_summary,
                "chapter_updated": chapter_number_for_delta,
                KG_IS_PROVISIONAL: node_props.get(KG_IS_PROVISIONAL, False),
            }
            statements.append(
                (
                    """
                    MATCH (we:WorldElement:Entity {id: $we_id})
                    MERGE (elab:Entity {id: $elab_event_id})
                        ON CREATE SET
                            elab:WorldElaborationEvent,
                            elab = $props,
                            elab.created_ts = timestamp()
                        ON MATCH SET
                            elab:WorldElaborationEvent,
                            elab = $props,
                            elab.updated_ts = timestamp()
                    MERGE (we)-[:ELABORATED_IN_CHAPTER]->(elab)
                    """,
                    {
                        "we_id": item.id,
                        "elab_event_id": elab_event_id,
                        "props": elab_props,
                    },
                )
            )

    return statements
