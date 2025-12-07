# core/langgraph/nodes/commit_node.py
"""
Knowledge graph commit node with two-phase deduplication for LangGraph workflow.

This module contains the deduplication and Neo4j commit logic ported from
processing/entity_deduplication.py and various data_access modules for use
in the LangGraph-based narrative generation workflow.

## Two-Phase Deduplication Architecture

This module implements a two-phase deduplication strategy to prevent duplicate
entities in the knowledge graph:

### Phase 1: Name-Based Deduplication (BEFORE Relationships)
- Runs during entity extraction
- Uses Levenshtein similarity on entity names and descriptions
- Catches obvious duplicates with high name similarity (threshold: 0.8+)
- Fast and efficient for clear matches

### Phase 2: Relationship-Based Deduplication (AFTER Relationships)
- Runs AFTER relationships are committed to Neo4j
- Uses relationship pattern similarity to identify duplicates
- Catches borderline cases that Phase 1 missed
- Example: "Alice" vs "Alice Chen" might not merge in Phase 1, but if both
  have identical relationships with "Bob" and "Central Lab", they're clearly
  the same person

### Why Two Phases?

The critical issue is that deduplication needs relationship context, but
relationships can't be extracted until entities exist. The two-phase approach
solves this by:
1. Quick deduplication before relationships (Phase 1)
2. Relationship-aware deduplication after relationships (Phase 2)

This prevents knowledge graph bloat from false negatives (duplicates not merged)
in long-form narratives where entity references may vary across chapters.

Migration Reference: docs/langgraph_migration_plan.md - Step 1.2.1

Source Code Ported From:
- processing/entity_deduplication.py:
  - check_entity_similarity() (lines 39-116)
  - should_merge_entities() (lines 119-160)
  - prevent_character_duplication() (lines 163-200)
  - prevent_world_item_duplication() (lines 203-244)
  - generate_entity_id() (lines 18-36)
  - [NEW] check_relationship_pattern_similarity()
  - [NEW] find_relationship_based_duplicates()
  - [NEW] merge_duplicate_entities()
- data_access/kg_queries.py:
  - add_kg_triples_batch_to_db() (lines 1144+)
- data_access/chapter_queries.py:
  - save_chapter_data_to_db() (lines 25-67)
- core/knowledge_graph_service.py:
  - persist_entities() (lines 24-74)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

import config
from core.langgraph.content_manager import (
    ContentManager,
    get_draft_text,
    get_extracted_entities,
    get_extracted_relationships,
    load_embedding,
)
from core.langgraph.state import ExtractedEntity, ExtractedRelationship, NarrativeState
from core.schema_validator import schema_validator
from data_access import chapter_queries, kg_queries
from models.kg_models import CharacterProfile, WorldItem
from processing.entity_deduplication import (
    check_entity_similarity,
    generate_entity_id,
    should_merge_entities,
)
from utils.text_processing import validate_and_filter_traits

logger = structlog.get_logger(__name__)


async def commit_to_graph(state: NarrativeState) -> NarrativeState:
    """
    Deduplicate entities and commit to Neo4j knowledge graph using two-phase deduplication.

    This is the main LangGraph node function that orchestrates the commit process.
    It handles character deduplication, world item deduplication, entity persistence,
    relationship creation, chapter node creation, and relationship-based deduplication.

    PORTED FROM: Multiple sources
    - processing/entity_deduplication.py (deduplication logic)
    - core/knowledge_graph_service.py (persistence)
    - data_access/kg_queries.py (relationship creation)
    - data_access/chapter_queries.py (chapter node creation)

    Process Flow:
    1. Phase 1 Deduplication: Deduplicate extracted characters (name matching, similarity)
    2. Phase 1 Deduplication: Deduplicate extracted world items (category + name matching)
    3. Convert ExtractedEntity models to CharacterProfile/WorldItem models
    4. Persist entities to Neo4j using existing query layer
    5. Create relationships between entities (with validation)
    6. Create chapter node with metadata
    7. Phase 2 Deduplication: Find and merge duplicates based on relationship patterns
    8. Update state with merge statistics

    ## Two-Phase Deduplication

    Phase 1 (Steps 1-2): Name-based deduplication before relationships
    - Fast, catches obvious duplicates
    - Uses Levenshtein similarity on names

    Phase 2 (Step 7): Relationship-based deduplication after relationships
    - Catches borderline cases that Phase 1 missed
    - Uses relationship pattern similarity (Jaccard similarity)
    - Example: "Alice" and "Alice Chen" with identical relationships to
      "Bob" and "Central Lab" are clearly the same person

    Args:
        state: Current narrative state with extracted_entities and extracted_relationships

    Returns:
        Updated state with:
        - current_node set to "commit_to_graph"
        - phase2_deduplication_merges dict with merge counts
    """
    # Initialize content manager to read externalized content
    content_manager = ContentManager(state.get("project_dir", ""))

    # Get extraction results from externalized content
    extracted = get_extracted_entities(state, content_manager)

    # Convert dicts to ExtractedEntity objects if needed
    char_entities_raw = extracted.get("characters", [])
    char_entities = [
        ExtractedEntity(**e) if isinstance(e, dict) else e for e in char_entities_raw
    ]

    world_entities_raw = extracted.get("world_items", [])
    world_entities = [
        ExtractedEntity(**e) if isinstance(e, dict) else e for e in world_entities_raw
    ]

    # Convert dicts to ExtractedRelationship objects if needed
    relationships_raw = get_extracted_relationships(state, content_manager)
    relationships = [
        ExtractedRelationship(**r) if isinstance(r, dict) else r
        for r in relationships_raw
    ]

    logger.info(
        "commit_to_graph",
        chapter=state.get("current_chapter", 1),
        characters=len(char_entities),
        world_items=len(world_entities),
        relationships=len(relationships),
    )

    # Track mappings for deduplication
    char_mappings: dict[str, str] = {}  # new_name -> existing_name (or same)
    world_mappings: dict[str, str] = {}  # new_name -> existing_id (or new_id)

    try:
        # Step 1: Deduplicate characters (READ operations)
        for char in char_entities:
            deduplicated_name = await _deduplicate_character(
                char.name, char.description, state.get("current_chapter", 1)
            )
            char_mappings[char.name] = deduplicated_name

        # Step 2: Deduplicate world items (READ operations)
        for item in world_entities:
            deduplicated_id = await _deduplicate_world_item(
                item.name,
                item.attributes.get("category", ""),
                item.description,
                state.get("current_chapter", 1),
            )
            world_mappings[item.name] = deduplicated_id

        # Step 3: Convert ExtractedEntity to CharacterProfile/WorldItem models
        character_models = _convert_to_character_profiles(
            char_entities, char_mappings, state.get("current_chapter", 1)
        )
        world_item_models = _convert_to_world_items(
            world_entities, world_mappings, state.get("current_chapter", 1)
        )

        # Step 4-6: Collect ALL Cypher statements for single transaction
        # This ensures atomicity - either all succeed or all are rolled back
        all_statements: list[tuple[str, dict]] = []

        # Step 4a: Collect entity persistence statements
        if character_models or world_item_models:
            entity_statements = await _build_entity_persistence_statements(
                character_models, world_item_models, state.get("current_chapter", 1)
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
                state.get("current_chapter", 1),
                is_from_flawed_draft=state.get("is_from_flawed_draft", False),
            )
            all_statements.extend(relationship_statements)

        # Step 4c: Collect chapter node statement
        content_manager = ContentManager(state.get("project_dir", ""))
        draft_text = get_draft_text(state, content_manager) or ""

        # Get embedding from ref if available
        embedding = None
        if state.get("embedding_ref"):
            try:
                embedding = load_embedding(
                    content_manager, state.get("embedding_ref", None)
                )
            except Exception as e:
                logger.warning(
                    "commit_to_graph: failed to load embedding", error=str(e)
                )
        elif state.get("generated_embedding"):
            # Fallback for backward compatibility or if not externalized yet
            embedding = state.get("generated_embedding")

        chapter_statement = _build_chapter_node_statement(
            chapter_number=state.get("current_chapter", 1),
            text=draft_text,
            word_count=state.get("draft_word_count", 0),
            summary=None,
            embedding=embedding,
        )
        all_statements.append(chapter_statement)

        # Step 5: Execute ALL statements in a SINGLE transaction
        # If any statement fails, all are rolled back
        if all_statements:
            from core.db_manager import neo4j_manager

            await neo4j_manager.execute_cypher_batch(all_statements)

            logger.info(
                "commit_to_graph: successfully committed to knowledge graph in single transaction",
                chapter=state.get("current_chapter", 1),
                characters=len(character_models),
                world_items=len(world_item_models),
                relationships=len(relationships),
                total_statements=len(all_statements),
            )

        # Step 6: Phase 2 Deduplication - Relationship-based duplicate detection
        # This runs AFTER relationships are committed, so relationship context is available
        # to help identify duplicates that were missed in Phase 1 (name-based deduplication)
        phase2_merges = await _run_phase2_deduplication(state.get("current_chapter", 1))

        return {
            **state,
            "current_node": "commit_to_graph",
            "last_error": None,
            "has_fatal_error": False,
            "phase2_deduplication_merges": phase2_merges,
        }

    except Exception as e:
        logger.error(
            "commit_to_graph: fatal error during commit",
            error=str(e),
            chapter=state.get("current_chapter", 1),
            exc_info=True,
        )
        return {
            **state,
            "current_node": "commit_to_graph",
            "last_error": f"Commit to graph failed: {e}",
            "has_fatal_error": True,
            "error_node": "commit",
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

        # Extract and validate traits from attributes
        raw_traits = entity.attributes.get("traits", [])
        traits = validate_and_filter_traits(raw_traits)

        if len(traits) != len(raw_traits):
            logger.warning(
                "_extract_character_profiles_from_entities: filtered invalid traits",
                character=final_name,
                original_count=len(raw_traits),
                filtered_count=len(traits),
            )

        # Extract status
        status = entity.attributes.get("status", "Unknown")

        # Extract relationships
        relationships = entity.attributes.get("relationships", {})

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
        entity_type = entity_type_map.get(original_name, "Object")  # Default to Object
        entity_category = entity_category_map.get(original_name, "")

        # Map extraction types to Neo4j node types
        # The LLM now provides proper node type names (e.g., "DevelopmentEvent", "PlotPoint")
        # so we check if it's already capitalized. If not, apply legacy mapping.
        neo4j_type = "Object"

        if entity_type:
            # Validate and normalize
            is_valid, normalized, _ = schema_validator.validate_entity_type(entity_type)
            if is_valid:
                neo4j_type = normalized
            elif entity_type[0].isupper():
                # Already a proper node type from the ontology
                neo4j_type = entity_type
            else:
                # Legacy lowercase type, apply mapping
                type_mapping = {
                    "character": "Character",
                    "location": "Location",
                    "event": "Event",
                    "object": "Object",
                }
                neo4j_type = type_mapping.get(entity_type.lower(), "Object")

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
    embedding: list[float] | None = None,
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
        embedding: Optional embedding vector
    """
    try:
        # Convert embedding to numpy array if present
        embedding_array = np.array(embedding) if embedding else None

        await chapter_queries.save_chapter_data_to_db(
            chapter_number=chapter_number,
            text=text,
            raw_llm_output=text,  # Same as text in this context
            summary=summary,
            embedding_array=embedding_array,
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

    logger.debug(
        "_build_relationship_statements: entity type map",
        entity_count=len(entity_type_map),
        entity_types=list(entity_type_map.items())[:10],  # Log first 10 for debugging
    )

    # Helper to create subject/object dict with type info
    def _make_entity_dict(
        name: str, original_name: str, explicit_type: str | None = None
    ) -> dict[str, str]:
        # Use explicit type if provided (from ExtractedRelationship.source_type/target_type)
        # Otherwise fall back to entity_type_map lookup
        if explicit_type:
            entity_type = explicit_type
        else:
            entity_type = entity_type_map.get(original_name, "Object")

        entity_category = entity_category_map.get(original_name, "")

        # Map extraction types to Neo4j node types
        # The LLM now provides proper node type names (e.g., "DevelopmentEvent", "PlotPoint")
        # so we check if it's already capitalized. If not, apply legacy mapping.
        neo4j_type = "Object"

        if entity_type:
            # Validate and normalize
            is_valid, normalized, _ = schema_validator.validate_entity_type(entity_type)
            if is_valid:
                neo4j_type = normalized
            elif entity_type[0].isupper():
                # Already a proper node type from the ontology
                neo4j_type = entity_type
            else:
                # Legacy lowercase type, apply mapping
                type_mapping = {
                    "character": "Character",
                    "location": "Location",
                    "event": "Event",
                    "object": "Object",
                }
                neo4j_type = type_mapping.get(entity_type.lower(), "Object")

        return {
            "name": name,
            "type": neo4j_type,
            "category": entity_category,
        }

    # Convert to triple format
    structured_triples: list[dict[str, Any]] = []

    for rel in relationships:
        source_name = char_mappings.get(rel.source_name, rel.source_name)
        target_name = char_mappings.get(rel.target_name, rel.target_name)

        if rel.target_name in world_mappings:
            target_name = world_mappings[rel.target_name]

        # Use explicit types from relationship if available (from parsing "Type:Name" format)
        # Otherwise _make_entity_dict will fall back to entity_type_map
        source_type = getattr(rel, "source_type", None)
        target_type = getattr(rel, "target_type", None)

        triple = {
            "subject": _make_entity_dict(source_name, rel.source_name, source_type),
            "predicate": rel.relationship_type,
            "object_entity": _make_entity_dict(
                target_name, rel.target_name, target_type
            ),
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

            # Explicit casting for mypy
            if not isinstance(subject, dict) or not isinstance(obj, dict):
                continue

            subject_name = subject["name"]
            subject_type = subject["type"]
            if not isinstance(predicate, str):
                predicate = str(predicate)
            predicate_clean = predicate.strip().upper().replace(" ", "_")

            if not predicate_clean:
                logger.warning(
                    "_build_relationship_statements: skipping relationship with empty predicate",
                    triple=triple,
                )
                continue

            object_name = obj["name"]
            object_type = obj["type"]

            # Get Cypher labels for nodes
            from data_access.kg_queries import _get_cypher_labels

            subject_labels = _get_cypher_labels(subject_type)
            object_labels = _get_cypher_labels(object_type)

            # Build relationship Cypher with proper ON CREATE SET for provisional nodes
            # NOTE: We must use backticks around the relationship type to handle special characters
            query = f"""
            MERGE (subj{subject_labels} {{name: $subject_name}})
            ON CREATE SET
                subj.is_provisional = true,
                subj.created_chapter = $chapter,
                subj.description = 'Entity created from relationship extraction. Details to be developed.',
                subj.created_at = timestamp()
            MERGE (obj{object_labels} {{name: $object_name}})
            ON CREATE SET
                obj.is_provisional = true,
                obj.created_chapter = $chapter,
                obj.description = 'Entity created from relationship extraction. Details to be developed.',
                obj.created_at = timestamp()
            MERGE (subj)-[r:`{predicate_clean}`]->(obj)
            SET r.chapter_added = $chapter,
                r.is_provisional = $is_provisional,
                r.confidence = $confidence,
                r.description = $description,
                r.last_updated = timestamp()

            // Link newly created provisional nodes to their chapter
            WITH subj, obj
            OPTIONAL MATCH (c:Chapter {{number: $chapter}})
            FOREACH (_ IN CASE WHEN subj.is_provisional = true AND c IS NOT NULL THEN [1] ELSE [] END |
                MERGE (subj)-[:MENTIONED_IN]->(c)
            )
            FOREACH (_ IN CASE WHEN obj.is_provisional = true AND c IS NOT NULL THEN [1] ELSE [] END |
                MERGE (obj)-[:MENTIONED_IN]->(c)
            )
            """

            params = {
                "subject_name": subject_name,
                "object_name": object_name,
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
    embedding: list[float] | None = None,
) -> tuple[str, dict]:
    """
    Build Cypher statement for chapter node creation.

    Args:
        chapter_number: Chapter number
        text: Chapter text content
        word_count: Word count for metadata
        summary: Optional chapter summary
        embedding: Optional embedding vector

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
        "embedding_vector_param": embedding,
    }

    logger.debug(
        "_build_chapter_node_statement: built statement",
        chapter=chapter_number,
    )

    return (query, parameters)


async def _run_phase2_deduplication(chapter: int) -> dict[str, int]:
    """
    Run Phase 2 deduplication using relationship patterns.

    This function runs AFTER relationships are committed to Neo4j, allowing us to use
    relationship context to identify duplicates that were missed in Phase 1 (name-based
    deduplication).

    Example failure case this addresses:
    - Chapter 5 extracts "Alice" (young woman) and "Alice Chen" (protagonist)
    - Phase 1 deduplication: Names are similar but not identical, borderline similarity
    - Phase 2 deduplication: Both have relationships with "Bob" and "Central Lab",
      so they're clearly the same person -> merge them

    Args:
        chapter: Current chapter number for logging

    Returns:
        Dict with merge counts: {"characters": N, "world_items": M}
    """
    try:
        import config

        # Check if Phase 2 deduplication is enabled
        if not getattr(config, "ENABLE_PHASE2_DEDUPLICATION", False):
            logger.debug(
                "_run_phase2_deduplication: Phase 2 deduplication disabled in config",
                chapter=chapter,
            )
            return {"characters": 0, "world_items": 0}

        # Import Phase 2 functions
        from processing.entity_deduplication import (
            find_relationship_based_duplicates,
            merge_duplicate_entities,
        )

        # Get configuration thresholds
        name_threshold = getattr(config, "PHASE2_NAME_SIMILARITY_THRESHOLD", 0.6)
        rel_threshold = getattr(config, "PHASE2_RELATIONSHIP_SIMILARITY_THRESHOLD", 0.7)

        logger.info(
            "_run_phase2_deduplication: starting Phase 2 deduplication",
            chapter=chapter,
            name_threshold=name_threshold,
            rel_threshold=rel_threshold,
        )

        merge_counts = {"characters": 0, "world_items": 0}

        # Phase 2 for characters
        char_duplicates = await find_relationship_based_duplicates(
            entity_type="character",
            name_similarity_threshold=name_threshold,
            relationship_similarity_threshold=rel_threshold,
        )

        for entity1, entity2, name_sim, rel_sim in char_duplicates:
            success = await merge_duplicate_entities(
                entity1, entity2, entity_type="character"
            )
            if success:
                merge_counts["characters"] += 1
                logger.info(
                    "_run_phase2_deduplication: merged character duplicates",
                    entity1=entity1,
                    entity2=entity2,
                    name_similarity=name_sim,
                    relationship_similarity=rel_sim,
                    chapter=chapter,
                )

        # Phase 2 for world items
        world_duplicates = await find_relationship_based_duplicates(
            entity_type="world_element",
            name_similarity_threshold=name_threshold,
            relationship_similarity_threshold=rel_threshold,
        )

        for entity1, entity2, name_sim, rel_sim in world_duplicates:
            success = await merge_duplicate_entities(
                entity1, entity2, entity_type="world_element"
            )
            if success:
                merge_counts["world_items"] += 1
                logger.info(
                    "_run_phase2_deduplication: merged world item duplicates",
                    entity1=entity1,
                    entity2=entity2,
                    name_similarity=name_sim,
                    relationship_similarity=rel_sim,
                    chapter=chapter,
                )

        logger.info(
            "_run_phase2_deduplication: completed Phase 2 deduplication",
            chapter=chapter,
            character_merges=merge_counts["characters"],
            world_item_merges=merge_counts["world_items"],
        )

        return merge_counts

    except Exception as e:
        logger.error(
            "_run_phase2_deduplication: error during Phase 2 deduplication",
            error=str(e),
            chapter=chapter,
            exc_info=True,
        )
        # Don't fail the commit if Phase 2 deduplication fails
        return {"characters": 0, "world_items": 0}


__all__ = ["commit_to_graph"]
