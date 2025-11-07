"""
Knowledge graph commit node with deduplication for LangGraph workflow.

This module contains the deduplication and Neo4j commit logic ported from
processing/entity_deduplication.py and various data_access modules for use
in the LangGraph-based narrative generation workflow.

Migration Reference: docs/langgraph_migration_plan.md - Step 1.2.1

Source Code Ported From:
- processing/entity_deduplication.py:
  - check_entity_similarity() (lines 39-116)
  - should_merge_entities() (lines 119-160)
  - prevent_character_duplication() (lines 163-200)
  - prevent_world_item_duplication() (lines 203-244)
  - generate_entity_id() (lines 18-36)
- data_access/kg_queries.py:
  - add_kg_triples_batch_to_db() (lines 1144+)
- data_access/chapter_queries.py:
  - save_chapter_data_to_db() (lines 25-67)
- core/knowledge_graph_service.py:
  - persist_entities() (lines 24-74)
"""

from __future__ import annotations

import structlog

import config
from core.knowledge_graph_service import knowledge_graph_service
from core.langgraph.state import ExtractedEntity, ExtractedRelationship, NarrativeState
from data_access import chapter_queries, kg_queries
from models.kg_models import CharacterProfile, WorldItem
from processing.entity_deduplication import (
    check_entity_similarity,
    generate_entity_id,
    should_merge_entities,
)

logger = structlog.get_logger(__name__)


async def commit_to_graph(state: NarrativeState) -> NarrativeState:
    """
    Deduplicate entities and commit to Neo4j knowledge graph.

    This is the main LangGraph node function that orchestrates the commit process.
    It handles character deduplication, world item deduplication, entity persistence,
    relationship creation, and chapter node creation.

    PORTED FROM: Multiple sources
    - processing/entity_deduplication.py (deduplication logic)
    - core/knowledge_graph_service.py (persistence)
    - data_access/kg_queries.py (relationship creation)
    - data_access/chapter_queries.py (chapter node creation)

    Process Flow:
    1. Deduplicate extracted characters (name matching, similarity)
    2. Deduplicate extracted world items (category + name matching)
    3. Convert ExtractedEntity models to CharacterProfile/WorldItem models
    4. Persist entities to Neo4j using existing query layer
    5. Create relationships between entities (with validation)
    6. Create chapter node with metadata
    7. Update state

    Args:
        state: Current narrative state with extracted_entities and extracted_relationships

    Returns:
        Updated state with current_node set to "commit_to_graph"
    """
    logger.info(
        "commit_to_graph",
        chapter=state["current_chapter"],
        characters=len(state.get("extracted_entities", {}).get("characters", [])),
        world_items=len(state.get("extracted_entities", {}).get("world_items", [])),
        relationships=len(state.get("extracted_relationships", [])),
    )

    extracted = state.get("extracted_entities", {})
    char_entities = extracted.get("characters", [])
    world_entities = extracted.get("world_items", [])
    relationships = state.get("extracted_relationships", [])

    # Track mappings for deduplication
    char_mappings: dict[str, str] = {}  # new_name -> existing_name (or same)
    world_mappings: dict[str, str] = {}  # new_name -> existing_id (or new_id)

    try:
        # Step 1: Deduplicate characters
        for char in char_entities:
            deduplicated_name = await _deduplicate_character(
                char.name, char.description, state["current_chapter"]
            )
            char_mappings[char.name] = deduplicated_name

        # Step 2: Deduplicate world items
        for item in world_entities:
            deduplicated_id = await _deduplicate_world_item(
                item.name,
                item.attributes.get("category", ""),
                item.description,
                state["current_chapter"],
            )
            world_mappings[item.name] = deduplicated_id

        # Step 3: Convert ExtractedEntity to CharacterProfile/WorldItem models
        character_models = _convert_to_character_profiles(
            char_entities, char_mappings, state["current_chapter"]
        )
        world_item_models = _convert_to_world_items(
            world_entities, world_mappings, state["current_chapter"]
        )

        # Step 4: Persist entities to Neo4j
        if character_models or world_item_models:
            success = await knowledge_graph_service.persist_entities(
                character_models, world_item_models, state["current_chapter"]
            )
            if not success:
                logger.warning(
                    "commit_to_graph: entity persistence returned failure flag",
                    chapter=state["current_chapter"],
                )

        # Step 5: Create relationships (using deduplicated names/IDs)
        if relationships:
            await _create_relationships(
                relationships,
                char_mappings,
                world_mappings,
                state["current_chapter"],
                is_from_flawed_draft=False,  # Assume finalized draft
            )

        # Step 6: Create chapter node
        await _create_chapter_node(
            chapter_number=state["current_chapter"],
            text=state.get("draft_text", ""),
            word_count=state.get("draft_word_count", 0),
            summary=None,  # Summary would be generated by separate node
        )

        logger.info(
            "commit_to_graph: successfully committed to knowledge graph",
            chapter=state["current_chapter"],
            characters=len(character_models),
            world_items=len(world_item_models),
            relationships=len(relationships),
        )

        return {
            **state,
            "current_node": "commit_to_graph",
            "last_error": None,
        }

    except Exception as e:
        logger.error(
            "commit_to_graph: error during commit",
            error=str(e),
            chapter=state["current_chapter"],
            exc_info=True,
        )
        return {
            **state,
            "current_node": "commit_to_graph",
            "last_error": f"Commit to graph failed: {e}",
        }


async def _deduplicate_character(name: str, description: str, chapter: int) -> str:
    """
    Check for duplicate characters and return the name to use.

    PORTED FROM: processing/entity_deduplication.py
    - prevent_character_duplication() (lines 163-200)
    - check_entity_similarity() (lines 39-116)
    - should_merge_entities() (lines 119-160)

    Args:
        name: Character name from extraction
        description: Character description
        chapter: Current chapter number

    Returns:
        Name to use (either existing character name or original name)
    """
    # Check if duplicate prevention is enabled in config
    if (
        not config.ENABLE_DUPLICATE_PREVENTION
        or not config.DUPLICATE_PREVENTION_CHARACTER_ENABLED
    ):
        return name

    # Check for similar existing character
    similar_entity = await check_entity_similarity(name, "character")

    if similar_entity:
        # Determine if we should merge based on similarity
        should_merge = await should_merge_entities(
            name,
            description,
            similar_entity,
            similarity_threshold=config.DUPLICATE_PREVENTION_SIMILARITY_THRESHOLD,
        )

        if should_merge:
            existing_name = similar_entity["existing_name"]
            logger.info(
                "commit_to_graph: merged character",
                new_name=name,
                existing_name=existing_name,
                similarity=similar_entity.get("similarity", 0.0),
            )
            return existing_name

    # No merge - use original name
    return name


async def _deduplicate_world_item(
    name: str, category: str, description: str, chapter: int
) -> str:
    """
    Check for duplicate world items and return the ID to use.

    PORTED FROM: processing/entity_deduplication.py
    - prevent_world_item_duplication() (lines 203-244)
    - check_entity_similarity() (lines 39-116)
    - should_merge_entities() (lines 119-160)
    - generate_entity_id() (lines 18-36)

    Args:
        name: World item name from extraction
        category: World item category
        description: World item description
        chapter: Current chapter number

    Returns:
        ID to use (either existing world item ID or new deterministic ID)
    """
    # Check if duplicate prevention is enabled in config
    if (
        not config.ENABLE_DUPLICATE_PREVENTION
        or not config.DUPLICATE_PREVENTION_WORLD_ITEM_ENABLED
    ):
        # Generate new ID
        return generate_entity_id(name, category, chapter)

    # Check for similar existing world item
    similar_entity = await check_entity_similarity(name, "world_element", category)

    if similar_entity:
        # Determine if we should merge based on similarity
        should_merge = await should_merge_entities(
            name,
            description,
            similar_entity,
            similarity_threshold=config.DUPLICATE_PREVENTION_SIMILARITY_THRESHOLD,
        )

        if should_merge:
            existing_id = similar_entity.get("existing_id")
            logger.info(
                "commit_to_graph: merged world item",
                new_name=name,
                existing_id=existing_id,
                category=category,
                similarity=similar_entity.get("similarity", 0.0),
            )
            return existing_id

    # No merge - generate new deterministic ID
    return generate_entity_id(name, category, chapter)


def _convert_to_character_profiles(
    entities: list[ExtractedEntity],
    name_mappings: dict[str, str],
    chapter: int,
) -> list[CharacterProfile]:
    """
    Convert ExtractedEntity instances to CharacterProfile models.

    This function bridges the LangGraph state model (ExtractedEntity) with
    the existing SAGA model (CharacterProfile) for persistence.

    Args:
        entities: List of character ExtractedEntity instances
        name_mappings: Dict mapping extracted names to deduplicated names
        chapter: Current chapter number

    Returns:
        List of CharacterProfile models ready for persistence
    """
    profiles = []

    for entity in entities:
        # Use deduplicated name
        final_name = name_mappings.get(entity.name, entity.name)

        # Extract traits from attributes
        traits = [
            k for k, v in entity.attributes.items() if isinstance(v, str) and not v
        ]
        if not traits:
            # Try to extract from description or use empty list
            traits = []

        # Extract status
        status = entity.attributes.get("status", "Unknown")

        # Extract relationships
        relationships = {
            k: v
            for k, v in entity.attributes.items()
            if isinstance(v, str) and v and k != "status"
        }

        profiles.append(
            CharacterProfile(
                name=final_name,
                description=entity.description,
                traits=traits,
                status=status if isinstance(status, str) else "Unknown",
                relationships=relationships,
                created_chapter=entity.first_appearance_chapter,
                is_provisional=False,  # Entities from finalized draft are not provisional
                updates={},  # Empty updates for new extraction
            )
        )

    return profiles


def _convert_to_world_items(
    entities: list[ExtractedEntity],
    id_mappings: dict[str, str],
    chapter: int,
) -> list[WorldItem]:
    """
    Convert ExtractedEntity instances to WorldItem models.

    This function bridges the LangGraph state model (ExtractedEntity) with
    the existing SAGA model (WorldItem) for persistence.

    Args:
        entities: List of world item ExtractedEntity instances
        id_mappings: Dict mapping extracted names to deduplicated IDs
        chapter: Current chapter number

    Returns:
        List of WorldItem models ready for persistence
    """
    items = []

    for entity in entities:
        # Use deduplicated ID
        final_id = id_mappings.get(entity.name, entity.name)

        # Extract category from attributes
        category = entity.attributes.get("category", "object")

        # Extract structured fields
        goals = entity.attributes.get("goals", [])
        rules = entity.attributes.get("rules", [])
        key_elements = entity.attributes.get("key_elements", [])

        # Ensure these are lists
        if not isinstance(goals, list):
            goals = [str(goals)] if goals else []
        if not isinstance(rules, list):
            rules = [str(rules)] if rules else []
        if not isinstance(key_elements, list):
            key_elements = [str(key_elements)] if key_elements else []

        # Collect additional properties
        additional_properties = {
            k: v
            for k, v in entity.attributes.items()
            if k not in {"category", "id", "goals", "rules", "key_elements"}
        }

        items.append(
            WorldItem(
                id=final_id,
                category=category,
                name=entity.name,
                description=entity.description,
                goals=goals,
                rules=rules,
                key_elements=key_elements,
                traits=[],  # Traits typically not used for world items
                created_chapter=entity.first_appearance_chapter,
                is_provisional=False,
                additional_properties=additional_properties,
            )
        )

    return items


async def _create_relationships(
    relationships: list[ExtractedRelationship],
    char_mappings: dict[str, str],
    world_mappings: dict[str, str],
    chapter: int,
    is_from_flawed_draft: bool,
) -> None:
    """
    Create relationship edges in Neo4j knowledge graph.

    PORTED FROM: data_access/kg_queries.py
    - add_kg_triples_batch_to_db() (lines 1144+)

    This function converts ExtractedRelationship instances to the triple format
    expected by the existing KG queries infrastructure, applying deduplication
    mappings to ensure relationships reference the correct entity names/IDs.

    Args:
        relationships: List of ExtractedRelationship instances
        char_mappings: Character name mappings (old -> deduplicated)
        world_mappings: World item name to ID mappings
        chapter: Current chapter number
        is_from_flawed_draft: Whether relationships are from unrevised draft
    """
    if not relationships:
        return

    # Convert to triple format expected by add_kg_triples_batch_to_db
    structured_triples = []

    for rel in relationships:
        # Apply deduplication mappings
        source_name = char_mappings.get(rel.source_name, rel.source_name)
        target_name = char_mappings.get(rel.target_name, rel.target_name)

        # Check if target is a world item
        if rel.target_name in world_mappings:
            target_name = world_mappings[rel.target_name]

        # Build triple in the format expected by kg_queries
        triple = {
            "subject": source_name,
            "predicate": rel.relationship_type,
            "object_entity": target_name,
            "is_literal_object": False,
            "description": rel.description,
            "confidence": rel.confidence,
            "chapter_added": chapter,
        }

        structured_triples.append(triple)

    # Use existing infrastructure to persist relationships
    try:
        await kg_queries.add_kg_triples_batch_to_db(
            structured_triples, chapter, is_from_flawed_draft
        )

        logger.info(
            "commit_to_graph: created relationships",
            count=len(structured_triples),
            chapter=chapter,
        )

    except Exception as e:
        logger.error(
            "commit_to_graph: failed to create relationships",
            error=str(e),
            chapter=chapter,
            exc_info=True,
        )


async def _create_chapter_node(
    chapter_number: int,
    text: str,
    word_count: int,
    summary: str | None,
) -> None:
    """
    Create chapter node in Neo4j with metadata.

    PORTED FROM: data_access/chapter_queries.py
    - save_chapter_data_to_db() (lines 25-67)

    Args:
        chapter_number: Chapter number
        text: Chapter text content
        word_count: Word count for metadata
        summary: Optional chapter summary
    """
    try:
        # For now, we'll create a basic chapter node without embedding
        # Embedding generation would be handled by a separate node
        await chapter_queries.save_chapter_data_to_db(
            chapter_number=chapter_number,
            text=text,
            raw_llm_output=text,  # Same as text in this context
            summary=summary,
            embedding_array=None,  # Embeddings generated separately
            is_provisional=False,  # Finalized chapter
        )

        logger.info(
            "commit_to_graph: created chapter node",
            chapter=chapter_number,
            word_count=word_count,
        )

    except Exception as e:
        logger.error(
            "commit_to_graph: failed to create chapter node",
            error=str(e),
            chapter=chapter_number,
            exc_info=True,
        )


__all__ = ["commit_to_graph"]
