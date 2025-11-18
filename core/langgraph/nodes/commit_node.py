# core/langgraph/nodes/commit_node.py
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
        # Step 1: Deduplicate characters (READ operations)
        for char in char_entities:
            deduplicated_name = await _deduplicate_character(
                char.name, char.description, state["current_chapter"]
            )
            char_mappings[char.name] = deduplicated_name

        # Step 2: Deduplicate world items (READ operations)
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

        # Step 4-6: Collect ALL Cypher statements for single transaction
        # This ensures atomicity - either all succeed or all are rolled back
        all_statements: list[tuple[str, dict]] = []

        # Step 4a: Collect entity persistence statements
        if character_models or world_item_models:
            entity_statements = await _build_entity_persistence_statements(
                character_models, world_item_models, state["current_chapter"]
            )
            all_statements.extend(entity_statements)

        # Step 4b: Collect relationship statements
        if relationships:
            relationship_statements = await _build_relationship_statements(
                relationships,
                char_entities,
                world_entities,
                char_mappings,
                world_mappings,
                state["current_chapter"],
                is_from_flawed_draft=state.get("is_from_flawed_draft", False),
            )
            all_statements.extend(relationship_statements)

        # Step 4c: Collect chapter node statement
        chapter_statement = _build_chapter_node_statement(
            chapter_number=state["current_chapter"],
            text=state.get("draft_text", ""),
            word_count=state.get("draft_word_count", 0),
            summary=None,
        )
        all_statements.append(chapter_statement)

        # Step 5: Execute ALL statements in a SINGLE transaction
        # If any statement fails, all are rolled back
        if all_statements:
            from core.db_manager import neo4j_manager

            await neo4j_manager.execute_cypher_batch(all_statements)

            logger.info(
                "commit_to_graph: successfully committed to knowledge graph in single transaction",
                chapter=state["current_chapter"],
                characters=len(character_models),
                world_items=len(world_item_models),
                relationships=len(relationships),
                total_statements=len(all_statements),
            )

        return {
            **state,
            "current_node": "commit_to_graph",
            "last_error": None,
            "has_fatal_error": False,
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
    char_entities: list[ExtractedEntity],
    world_entities: list[ExtractedEntity],
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
        char_entities: List of character ExtractedEntity instances
        world_entities: List of world item ExtractedEntity instances
        char_mappings: Character name mappings (old -> deduplicated)
        world_mappings: World item name to ID mappings
        chapter: Current chapter number
        is_from_flawed_draft: Whether relationships are from unrevised draft
    """
    if not relationships:
        return

    # Build entity lookup maps for type resolution
    entity_type_map = {}
    entity_category_map = {}

    for entity in char_entities:
        entity_type_map[entity.name] = entity.type
        entity_category_map[entity.name] = entity.attributes.get("category", "")

    for entity in world_entities:
        entity_type_map[entity.name] = entity.type
        entity_category_map[entity.name] = entity.attributes.get("category", "")

    # Helper to create subject/object dict with type info
    def _make_entity_dict(name: str, original_name: str) -> dict:
        """Create entity dict with name, type, and category."""
        entity_type = entity_type_map.get(original_name, "object")  # Default to object
        entity_category = entity_category_map.get(original_name, "")

        # Map extraction types to Neo4j node types
        type_mapping = {
            "character": "Character",
            "location": "Location",
            "event": "Event",
            "object": "Object",
        }
        neo4j_type = type_mapping.get(entity_type, "Object")

        return {
            "name": name,
            "type": neo4j_type,
            "category": entity_category,
        }

    # Convert to triple format expected by add_kg_triples_batch_to_db
    structured_triples = []

    for rel in relationships:
        # Apply deduplication mappings
        source_name = char_mappings.get(rel.source_name, rel.source_name)
        target_name = char_mappings.get(rel.target_name, rel.target_name)

        # Check if target is a world item
        if rel.target_name in world_mappings:
            target_name = world_mappings[rel.target_name]

        # Build triple in the format expected by kg_queries (dict format for subject/object)
        triple = {
            "subject": _make_entity_dict(source_name, rel.source_name),
            "predicate": rel.relationship_type,
            "object_entity": _make_entity_dict(target_name, rel.target_name),
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


async def _build_entity_persistence_statements(
    characters: list[CharacterProfile],
    world_items: list[WorldItem],
    chapter_number: int,
) -> list[tuple[str, dict]]:
    """
    Build Cypher statements for entity persistence without executing them.

    This extracts the statement-building logic from knowledge_graph_service
    so statements can be batched into a single transaction.

    Args:
        characters: List of CharacterProfile models
        world_items: List of WorldItem models
        chapter_number: Current chapter for tracking

    Returns:
        List of (cypher_query, parameters) tuples
    """
    statements: list[tuple[str, dict]] = []

    # Use the cypher builder from knowledge_graph_service
    from data_access.cypher_builders.native_builders import NativeCypherBuilder

    cypher_builder = NativeCypherBuilder()

    # Generate Cypher for characters
    for char in characters:
        cypher, params = cypher_builder.character_upsert_cypher(char, chapter_number)
        statements.append((cypher, params))

    # Generate Cypher for world items
    for item in world_items:
        cypher, params = cypher_builder.world_item_upsert_cypher(item, chapter_number)
        statements.append((cypher, params))

    logger.info(
        "_build_entity_persistence_statements: built statements",
        characters=len(characters),
        world_items=len(world_items),
        total_statements=len(statements),
    )

    return statements


async def _build_relationship_statements(
    relationships: list[ExtractedRelationship],
    char_entities: list[ExtractedEntity],
    world_entities: list[ExtractedEntity],
    char_mappings: dict[str, str],
    world_mappings: dict[str, str],
    chapter: int,
    is_from_flawed_draft: bool,
) -> list[tuple[str, dict]]:
    """
    Build Cypher statements for relationships without executing them.

    This builds the same triples as _create_relationships but returns
    Cypher statements instead of executing them.

    Args:
        relationships: List of ExtractedRelationship instances
        char_entities: List of character ExtractedEntity instances
        world_entities: List of world item ExtractedEntity instances
        char_mappings: Character name mappings (old -> deduplicated)
        world_mappings: World item name to ID mappings
        chapter: Current chapter number
        is_from_flawed_draft: Whether relationships are from unrevised draft

    Returns:
        List of (cypher_query, parameters) tuples
    """
    if not relationships:
        return []

    # Build entity lookup maps for type resolution (same as _create_relationships)
    entity_type_map = {}
    entity_category_map = {}

    for entity in char_entities:
        entity_type_map[entity.name] = entity.type
        entity_category_map[entity.name] = entity.attributes.get("category", "")

    for entity in world_entities:
        entity_type_map[entity.name] = entity.type
        entity_category_map[entity.name] = entity.attributes.get("category", "")

    # Helper to create subject/object dict with type info
    def _make_entity_dict(name: str, original_name: str) -> dict:
        entity_type = entity_type_map.get(original_name, "object")
        entity_category = entity_category_map.get(original_name, "")

        type_mapping = {
            "character": "Character",
            "location": "Location",
            "event": "Event",
            "object": "Object",
        }
        neo4j_type = type_mapping.get(entity_type, "Object")

        return {
            "name": name,
            "type": neo4j_type,
            "category": entity_category,
        }

    # Convert to triple format
    structured_triples = []

    for rel in relationships:
        source_name = char_mappings.get(rel.source_name, rel.source_name)
        target_name = char_mappings.get(rel.target_name, rel.target_name)

        if rel.target_name in world_mappings:
            target_name = world_mappings[rel.target_name]

        triple = {
            "subject": _make_entity_dict(source_name, rel.source_name),
            "predicate": rel.relationship_type,
            "object_entity": _make_entity_dict(target_name, rel.target_name),
            "is_literal_object": False,
            "description": rel.description,
            "confidence": rel.confidence,
            "chapter_added": chapter,
        }

        structured_triples.append(triple)

    # Build Cypher statements from triples
    # This creates basic relationship statements without full constraint validation
    # (Full validation logic from kg_queries is too complex to inline here)
    statements: list[tuple[str, dict]] = []

    for triple in structured_triples:
        try:
            subject = triple["subject"]
            predicate = triple["predicate"]
            obj = triple["object_entity"]

            subject_name = subject["name"]
            subject_type = subject["type"]
            predicate_clean = predicate.strip().upper().replace(" ", "_")
            object_name = obj["name"]
            object_type = obj["type"]

            # Get Cypher labels for nodes
            from data_access.kg_queries import _get_cypher_labels

            subject_labels = _get_cypher_labels(subject_type)
            object_labels = _get_cypher_labels(object_type)

            # Build relationship Cypher
            query = f"""
            MERGE (subj{subject_labels} {{name: $subject_name}})
            MERGE (obj{object_labels} {{name: $object_name}})
            MERGE (subj)-[r:{predicate_clean}]->(obj)
            SET r.type = $rel_type,
                r.chapter_added = $chapter,
                r.is_provisional = $is_provisional,
                r.confidence = $confidence,
                r.description = $description,
                r.last_updated = timestamp()
            """

            params = {
                "subject_name": subject_name,
                "object_name": object_name,
                "rel_type": predicate_clean,
                "chapter": chapter,
                "is_provisional": is_from_flawed_draft,
                "confidence": triple.get("confidence", 1.0),
                "description": triple.get("description", ""),
            }

            statements.append((query, params))

        except Exception as e:
            logger.warning(
                "_build_relationship_statements: failed to build statement for triple",
                error=str(e),
                triple=triple,
            )
            continue

    logger.info(
        "_build_relationship_statements: built statements",
        relationships=len(relationships),
        statements=len(statements),
    )

    return statements


def _build_chapter_node_statement(
    chapter_number: int,
    text: str,
    word_count: int,
    summary: str | None,
) -> tuple[str, dict]:
    """
    Build Cypher statement for chapter node creation.

    Args:
        chapter_number: Chapter number
        text: Chapter text content
        word_count: Word count for metadata
        summary: Optional chapter summary

    Returns:
        Tuple of (cypher_query, parameters)
    """

    query = """
    MERGE (c:Chapter {number: $chapter_number_param})
    SET c.text = $text_param,
        c.raw_llm_output = $raw_llm_output_param,
        c.summary = $summary_param,
        c.is_provisional = $is_provisional_param,
        c.embedding_vector = $embedding_vector_param,
        c.last_updated = timestamp()
    """

    parameters = {
        "chapter_number_param": chapter_number,
        "text_param": text,
        "raw_llm_output_param": text,  # Same as text
        "summary_param": summary if summary is not None else "",
        "is_provisional_param": False,  # Finalized chapter
        "embedding_vector_param": None,  # Embeddings generated separately
    }

    logger.debug(
        "_build_chapter_node_statement: built statement",
        chapter=chapter_number,
    )

    return (query, parameters)


__all__ = ["commit_to_graph"]
