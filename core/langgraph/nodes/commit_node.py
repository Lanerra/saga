# core/langgraph/nodes/commit_node.py
"""
Commit extracted entities and relationships to Neo4j.

This module implements the Phase 2 persistence boundary for the LangGraph
workflow:
- Deduplicate extracted entities (name-based pre-commit).
- Build Cypher statements for entity upserts + relationship creation + chapter node.
- Execute all statements in a single transaction for atomicity.
- Optionally run post-commit, relationship-aware deduplication.

Migration Reference: docs/langgraph_migration_plan.md - Step 1.2.1

Notes:
    This node performs Neo4j I/O and cache invalidation for `data_access` reads.
"""

from __future__ import annotations

import hashlib
from typing import Any, cast

import numpy as np
import structlog

import config
from core.langgraph.content_manager import (
    ContentManager,
    ContentRef,
    get_draft_text,
    get_extracted_entities,
    get_extracted_relationships,
    load_embedding,
    load_scene_embeddings,
    require_project_dir,
)
from core.langgraph.state import ExtractedEntity, ExtractedRelationship, NarrativeState
from core.schema_validator import canonicalize_entity_type_for_persistence
from data_access import chapter_queries, kg_queries
from data_access.kg_queries import (
    _get_cypher_labels as _get_cypher_labels,
)
from data_access.kg_queries import (
    validate_relationship_type,
    validate_relationship_type_for_cypher_interpolation,
)
from models.kg_models import CharacterProfile, WorldItem
from processing.entity_deduplication import (
    check_entity_similarity,
    generate_entity_id,
    should_merge_entities,
)
from utils.text_processing import validate_and_filter_traits

logger = structlog.get_logger(__name__)


async def commit_to_graph(state: NarrativeState) -> NarrativeState:
    """Deduplicate extracted entities and commit the chapter to Neo4j.

    This node treats Neo4j writes as an atomic batch: entity upserts,
    relationships, and the chapter node are executed in one transaction.

    Args:
        state: Workflow state. Reads extracted entities/relationships (preferring
            externalized refs via
            [`get_extracted_entities()`](core/langgraph/content_manager.py:914) and
            [`get_extracted_relationships()`](core/langgraph/content_manager.py:939)).

    Returns:
        Updated state with:
        - current_node: "commit_to_graph"
        - phase2_deduplication_merges: Relationship-aware merge statistics

        On errors, returns a state with `has_fatal_error` set and `last_error`
        populated.

    Notes:
        - This node performs Neo4j I/O via
          [`neo4j_manager.execute_cypher_batch()`](core/db_manager.py:310).
        - After successful writes it clears `data_access` read caches to prevent
          stale reads within the same process.
    """
    # Initialize content manager to read externalized content
    content_manager = ContentManager(require_project_dir(state))

    # Get extraction results from externalized content
    extracted = get_extracted_entities(state, content_manager)

    # Convert dicts to ExtractedEntity objects if needed
    char_entities_raw = extracted.get("characters", [])
    char_entities = [ExtractedEntity(**e) if isinstance(e, dict) else e for e in char_entities_raw]

    world_entities_raw = extracted.get("world_items", [])
    world_entities = [ExtractedEntity(**e) if isinstance(e, dict) else e for e in world_entities_raw]

    # Convert dicts to ExtractedRelationship objects if needed
    relationships_raw = get_extracted_relationships(state, content_manager)
    relationships = [ExtractedRelationship(**r) if isinstance(r, dict) else r for r in relationships_raw]

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
            deduplicated_name = await _deduplicate_character(char.name, char.description, state.get("current_chapter", 1))
            char_mappings[char.name] = deduplicated_name

        # Step 2: Deduplicate world items (READ operations)
        # First pass: deduplicate within batch (same name = same id)
        seen_names: dict[str, str] = {}  # name -> first assigned id

        for item in world_entities:
            # Check if we've already seen this name in the batch
            if item.name in seen_names:
                # Reuse the id from the first occurrence
                world_mappings[item.name] = seen_names[item.name]
                logger.debug(
                    "commit_to_graph: within-batch duplicate detected",
                    name=item.name,
                    reusing_id=seen_names[item.name],
                )
                continue

            # First time seeing this name, check database for duplicates
            deduplicated_id = await _deduplicate_world_item(
                item.name,
                item.attributes.get("category", ""),
                item.description,
                state.get("current_chapter", 1),
            )
            world_mappings[item.name] = deduplicated_id
            seen_names[item.name] = deduplicated_id

        # Step 3: Convert ExtractedEntity to CharacterProfile/WorldItem models
        # Deduplicate entity lists to prevent creating duplicate models
        unique_char_entities = _deduplicate_entity_list(char_entities)
        unique_world_entities = _deduplicate_entity_list(world_entities)

        character_models = _convert_to_character_profiles(unique_char_entities, char_mappings, state.get("current_chapter", 1))
        world_item_models = _convert_to_world_items(unique_world_entities, world_mappings, state.get("current_chapter", 1))

        # Step 4-6: Collect ALL Cypher statements for single transaction
        # This ensures atomicity - either all succeed or all are rolled back
        all_statements: list[tuple[str, dict]] = []

        # Step 4a: Collect entity persistence statements
        if character_models or world_item_models:
            entity_statements = await _build_entity_persistence_statements(character_models, world_item_models, state.get("current_chapter", 1))
            all_statements.extend(entity_statements)

        # Step 4b: Collect relationship statements
        #
        # Contract: relationship writes are chapter-idempotent.
        # Every commit replaces the chapter's relationship set (including "no relationships").
        relationship_statements = await _build_relationship_statements(
            relationships,
            char_entities,
            world_entities,
            char_mappings,
            world_mappings,
            state.get("current_chapter", 1),
            is_from_flawed_draft=False,
        )
        if relationship_statements:
            all_statements.extend(relationship_statements)

        # Step 4c: Collect chapter node statement
        content_manager = ContentManager(require_project_dir(state))

        from core.exceptions import MissingDraftReferenceError

        try:
            draft_text = get_draft_text(state, content_manager)
        except MissingDraftReferenceError as error:
            return {
                "current_node": "commit_to_graph",
                "last_error": str(error),
                "has_fatal_error": True,
                "error_node": "commit",
            }

        # Get embedding from scene embeddings (preferred) or fallback to chapter embedding
        embedding = None

        # Try to load and aggregate scene embeddings
        scene_embeddings_ref_obj = state.get("scene_embeddings_ref")
        if scene_embeddings_ref_obj is not None:
            try:
                scene_embeddings_ref = cast(ContentRef, scene_embeddings_ref_obj)
                scene_embeddings = load_scene_embeddings(content_manager, scene_embeddings_ref)
                embedding = _aggregate_scene_embeddings_to_chapter(scene_embeddings)
                logger.info(
                    "commit_to_graph: aggregated scene embeddings into chapter embedding",
                    num_scenes=len(scene_embeddings),
                    embedding_dimensions=len(embedding) if embedding else 0,
                )
            except Exception as e:
                logger.warning("commit_to_graph: failed to load/aggregate scene embeddings", error=str(e))

        # Fallback for backward compatibility (should rarely be needed)
        elif state.get("embedding_ref"):
            try:
                embedding_ref = cast(ContentRef, state.get("embedding_ref"))
                embedding = load_embedding(content_manager, embedding_ref)
            except Exception as e:
                logger.warning("commit_to_graph: failed to load chapter embedding", error=str(e))
        elif state.get("generated_embedding"):
            # Fallback for backward compatibility or if not externalized yet
            generated_embedding = state.get("generated_embedding")
            if isinstance(generated_embedding, list):
                embedding = cast(list[float], generated_embedding)
            else:
                embedding = None

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

            # Cache invalidation after Neo4j writes
            #
            # Rationale:
            # - `data_access.*_queries` read functions are cached via async_lru.
            # - This node is the primary persistence boundary for Phase 2, so failing to clear
            #   caches here can cause stale reads (e.g., recently created characters/world items
            #   not visible; KG triple reads not reflecting new relationships).
            #
            # Local import avoids eager import side effects / circular deps.
            from data_access.cache_coordinator import (
                clear_character_read_caches,
                clear_kg_read_caches,
                clear_world_read_caches,
            )

            cleared_character = clear_character_read_caches()
            cleared_world = clear_world_read_caches()
            cleared_kg = clear_kg_read_caches()

            logger.info(
                "commit_to_graph: successfully committed to knowledge graph in single transaction",
                chapter=state.get("current_chapter", 1),
                characters=len(character_models),
                world_items=len(world_item_models),
                relationships=len(relationships),
                total_statements=len(all_statements),
                cache_cleared={
                    "character": cleared_character,
                    "world": cleared_world,
                    "kg": cleared_kg,
                },
            )

        # Step 6: Phase 2 Deduplication - Relationship-based duplicate detection
        # This runs AFTER relationships are committed, so relationship context is available
        # to help identify duplicates that were missed in Phase 1 (name-based deduplication)
        phase2_merges = await _run_phase2_deduplication(state.get("current_chapter", 1))

        return {
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
            "current_node": "commit_to_graph",
            "last_error": f"Commit to graph failed: {e}",
            "has_fatal_error": True,
            "error_node": "commit",
        }


def _deduplicate_entity_list(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    """Remove within-batch duplicate entities by name.

    Args:
        entities: Extracted entities for a single commit batch.

    Returns:
        A list with duplicate names removed (keeping the first occurrence).
    """
    seen_names: set[str] = set()
    unique_entities: list[ExtractedEntity] = []

    for entity in entities:
        if entity.name not in seen_names:
            unique_entities.append(entity)
            seen_names.add(entity.name)
        else:
            logger.debug(
                "_deduplicate_entity_list: skipping duplicate",
                name=entity.name,
                type=entity.type,
            )

    if len(unique_entities) < len(entities):
        logger.info(
            "_deduplicate_entity_list: removed duplicates",
            original_count=len(entities),
            unique_count=len(unique_entities),
            duplicates_removed=len(entities) - len(unique_entities),
        )

    return unique_entities


async def _deduplicate_character(name: str, description: str, chapter: int) -> str:
    """Resolve a character name to an existing character when a likely duplicate exists.

    Args:
        name: Extracted character name.
        description: Extracted character description used for similarity checks.
        chapter: Chapter number used for logging/provenance.

    Returns:
        The name to use for persistence. This may be an existing character name when
        deduplication decides a merge is appropriate, otherwise the original `name`.

    Notes:
        This helper performs similarity checks (which may involve I/O) when duplicate
        prevention is enabled. If duplicate prevention is disabled, it returns `name`
        unchanged.
    """
    # Check if duplicate prevention is enabled in config
    if not config.ENABLE_DUPLICATE_PREVENTION or not config.DUPLICATE_PREVENTION_CHARACTER_ENABLED:
        return name

    # Check for similar existing character
    similar_entity = await check_entity_similarity(
        name,
        "character",
        description=description,
    )

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


async def _deduplicate_world_item(name: str, category: str, description: str, chapter: int) -> str:
    """Resolve a world item to a stable identifier suitable for persistence.

    Args:
        name: Extracted world item name.
        category: Extracted world item category (used in deterministic ID generation).
        description: Extracted world item description used for similarity checks.
        chapter: Chapter number used for deterministic IDs and provenance.

    Returns:
        Stable world-item identifier to use for persistence. When duplicate prevention
        is disabled, this is a deterministic ID derived from `name`, `category`, and
        `chapter`. When enabled, this may instead be an existing item ID.

    Notes:
        This helper may perform I/O for similarity checks when duplicate prevention
        is enabled.
    """
    # Check if duplicate prevention is enabled in config
    if not config.ENABLE_DUPLICATE_PREVENTION or not config.DUPLICATE_PREVENTION_WORLD_ITEM_ENABLED:
        # Generate new ID
        return generate_entity_id(name, category, chapter)

    # Check for similar existing world item
    similar_entity = await check_entity_similarity(
        name,
        "world_element",
        category,
        description=description,
    )

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
            if isinstance(existing_id, str) and existing_id:
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

        # Use the category from attributes (preserves specific type like "artifact", "document")
        # The ExtractedEntity validator automatically stores the original type here before normalization
        category = entity.attributes.get("category", entity.type.lower() if entity.type else "")

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
        additional_properties = {k: v for k, v in entity.attributes.items() if k not in {"category", "id", "goals", "rules", "key_elements"}}

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
        """Create entity dict with name, type, and category.

        CORE-011 contract (canonical-label-first at persistence boundaries):
        - Node labels written to Neo4j MUST be one of the canonical domain labels
          [`VALID_NODE_LABELS`](models/kg_constants.py:66).
        - Subtypes / legacy aliases (e.g., "Structure", "Guild", "DevelopmentEvent") are
          permitted as intake but MUST be canonicalized before persistence.
        - Unknown / unmappable types are rejected with a clear error (no silent fallback).

        When type is missing, we fall back to canonical "Item" and preserve semantics via `category`.
        """
        entity_type = entity_type_map.get(original_name, None)
        entity_category = entity_category_map.get(original_name, "")

        if not entity_type or not str(entity_type).strip():
            neo4j_type = "Item"
        else:
            neo4j_type = canonicalize_entity_type_for_persistence(entity_type)

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
        await kg_queries.add_kg_triples_batch_to_db(structured_triples, chapter, is_from_flawed_draft)

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
    """Create or update the Chapter node with metadata.

    Args:
        chapter_number: Chapter number to persist.
        text: Chapter text content (used for provenance; may not be persisted directly).
        word_count: Word count metadata.
        summary: Optional chapter summary.
        embedding: Optional embedding vector.

    Notes:
        Failures are logged and swallowed; chapter node creation is treated as
        best-effort within the commit flow.
    """
    try:
        # Convert embedding to numpy array if present
        embedding_array = np.array(embedding) if embedding else None

        await chapter_queries.save_chapter_data_to_db(
            chapter_number=chapter_number,
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
    """Build Cypher statements to persist entities.

    Args:
        characters: Character profiles to upsert.
        world_items: World items to upsert.
        chapter_number: Chapter number used for provenance.

    Returns:
        List of `(cypher_query, parameters)` tuples suitable for batched execution.
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

    embedding_statements_count = 0
    if config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE:
        from core.entity_embedding_service import build_entity_embedding_update_statements

        embedding_statements = await build_entity_embedding_update_statements(
            characters=characters,
            world_items=world_items,
        )
        embedding_statements_count = len(embedding_statements)
        statements.extend(embedding_statements)

    logger.info(
        "_build_entity_persistence_statements: built statements",
        characters=len(characters),
        world_items=len(world_items),
        embedding_statements=embedding_statements_count,
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
    """Build Cypher statements to persist extracted relationships.

    Args:
        relationships: Extracted relationships to persist.
        char_entities: Extracted character entities used for type resolution.
        world_entities: Extracted world entities used for type resolution.
        char_mappings: Name deduplication mappings for characters.
        world_mappings: Name/identifier mappings for world items used by persistence.
        chapter: Chapter number used for provenance.
        is_from_flawed_draft: Whether relationships originate from a draft that had
            de-duplication applied.

    Returns:
        List of `(cypher_query, parameters)` tuples suitable for batched execution.
    """
    statements: list[tuple[str, dict]] = []

    # Idempotency: Delete any existing relationships for this chapter before writing the new set.
    # This ensures that revisions or re-runs do not accumulate stale edges.
    delete_query = """
    MATCH ()-[r]->()
    WHERE coalesce(r.chapter_added, -1) = $chapter
    DELETE r
    """
    statements.append((delete_query, {"chapter": chapter}))

    if not relationships:
        logger.info(
            "_build_relationship_statements: no extracted relationships; clearing chapter relationship set",
            chapter=chapter,
        )
        return statements

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

    # Pre-fetch existing entity IDs from database to avoid constraint violations
    # This prevents creating duplicate nodes when deduplication fails or entities appear only in relationships
    entity_id_cache: dict[str, str] = {}

    # Collect ALL unique entity names from relationships (not just those missing from entity_type_map)
    # This catches cases where deduplication failed or didn't find an exact name match
    entity_names_to_check: set[tuple[str, str]] = set()  # (name, type) tuples
    for rel in relationships:
        source_name = char_mappings.get(rel.source_name, rel.source_name)
        target_name = char_mappings.get(rel.target_name, rel.target_name)

        source_type = getattr(rel, "source_type", None) or entity_type_map.get(rel.source_name)
        target_type = getattr(rel, "target_type", None) or entity_type_map.get(rel.target_name)

        # Check all non-Character entities (Characters merge by name, not ID)
        if source_type != "Character":
            entity_names_to_check.add((source_name, source_type or "Item"))
        if target_type != "Character":
            entity_names_to_check.add((target_name, target_type or "Item"))

    # Batch query for all entity IDs
    if entity_names_to_check:
        from core.db_manager import neo4j_manager

        for name, entity_type_raw in entity_names_to_check:
            neo4j_type = canonicalize_entity_type_for_persistence(entity_type_raw or "Item")
            label = _get_cypher_labels(neo4j_type).lstrip(":")

            query = f"""
            MATCH (n:{label} {{name: $name}})
            RETURN n.id as id
            LIMIT 1
            """
            try:
                results = await neo4j_manager.execute_read_query(query, {"name": name})
                if results and results[0].get("id"):
                    cache_key = f"{name}:{neo4j_type}"
                    entity_id_cache[cache_key] = str(results[0]["id"])
                    logger.debug(
                        "_build_relationship_statements: found existing entity in database",
                        name=name,
                        type=neo4j_type,
                        id=entity_id_cache[cache_key],
                    )
            except Exception as e:
                logger.warning(
                    "_build_relationship_statements: failed to lookup existing entity",
                    name=name,
                    type=neo4j_type,
                    error=str(e),
                )

    # Helper to create subject/object dict with type + optional stable id.
    def _make_entity_dict(
        *,
        name: str,
        original_name: str,
        explicit_type: str | None = None,
        stable_id: str | None = None,
        relationship_type: str | None = None,
        role: str | None = None,
    ) -> dict[str, Any]:
        """Build an entity dictionary for relationship persistence.

        Contract:
        - `name` remains human-readable.
        - `id` is a stable identifier used for identity matching when available.
        - Missing or empty types are inferred from relationship semantics when possible,
          otherwise canonicalized to `"Item"`.

        Notes:
            Relationship persistence may need to create provisional nodes for entities that were
            not part of the extracted entity lists. For non-Character nodes, we generate a
            deterministic id to avoid casing-based duplicates (for example "Crew" vs "crew")
            and to prevent leaking deterministic ids into the `name` field.

        Args:
            name: Persisted entity name.
            original_name: Original extracted name used for type/category lookup.
            explicit_type: Explicit entity type override.
            stable_id: Stable identifier used for matching.
            relationship_type: Relationship type (for type inference when type is unknown).
            role: Entity role in relationship ("source" or "target") for type inference.

        Returns:
            Entity dictionary used by relationship persistence.
        """
        from core.relationship_validation import infer_entity_type_from_relationship

        entity_type = explicit_type if explicit_type is not None else entity_type_map.get(original_name, None)
        entity_category = entity_category_map.get(original_name, "")

        if not entity_type or not str(entity_type).strip():
            inferred_type = None
            if relationship_type and role:
                inferred_type = infer_entity_type_from_relationship(name, relationship_type, role)

            if inferred_type:
                neo4j_type = inferred_type
                logger.info(
                    "Inferred entity type from relationship semantics",
                    entity=name,
                    relationship=relationship_type,
                    role=role,
                    inferred_type=inferred_type,
                )
            else:
                neo4j_type = "Item"
        else:
            neo4j_type = canonicalize_entity_type_for_persistence(entity_type)

        resolved_stable_id = stable_id
        if resolved_stable_id is None and neo4j_type != "Character":
            mapped_id = world_mappings.get(original_name)
            if isinstance(mapped_id, str) and mapped_id:
                resolved_stable_id = mapped_id
            else:
                cache_key = f"{name}:{neo4j_type}"
                cached_id = entity_id_cache.get(cache_key)
                if cached_id:
                    resolved_stable_id = cached_id
                else:
                    logger.debug(
                        "_make_entity_dict: no existing entity found, will merge on name",
                        name=name,
                        type=neo4j_type,
                        cache_key=cache_key,
                    )
                    resolved_stable_id = None

        return {
            "name": name,
            "id": resolved_stable_id,
            "type": neo4j_type,
            "category": entity_category,
        }

    # Convert to triple format
    structured_triples: list[dict[str, Any]] = []

    for rel in relationships:
        # `char_mappings` canonicalizes character names for consistent relationship endpoints.
        source_name = char_mappings.get(rel.source_name, rel.source_name)
        target_name = char_mappings.get(rel.target_name, rel.target_name)

        # Use explicit types from relationship if available (from parsing "Type:Name" format)
        # Otherwise _make_entity_dict will fall back to entity_type_map
        source_type = getattr(rel, "source_type", None)
        target_type = getattr(rel, "target_type", None)

        triple = {
            "subject": _make_entity_dict(
                name=source_name,
                original_name=rel.source_name,
                explicit_type=source_type,
                stable_id=None,
                relationship_type=rel.relationship_type,
                role="source",
            ),
            "predicate": rel.relationship_type,
            "object_entity": _make_entity_dict(
                name=target_name,
                original_name=rel.target_name,
                explicit_type=target_type,
                stable_id=None,
                relationship_type=rel.relationship_type,
                role="target",
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
    for triple in structured_triples:
        try:
            subject = triple["subject"]
            predicate = triple["predicate"]
            obj = triple["object_entity"]

            if not isinstance(subject, dict) or not isinstance(obj, dict):
                continue

            subject_name = subject["name"]
            subject_type = subject["type"]
            subject_id = subject.get("id")

            if not isinstance(predicate, str):
                predicate = str(predicate)

            predicate_normalized = validate_relationship_type(predicate)
            predicate_clean = validate_relationship_type_for_cypher_interpolation(predicate_normalized)

            if not predicate_clean:
                logger.warning(
                    "_build_relationship_statements: skipping relationship with empty predicate",
                    triple=triple,
                )
                continue

            object_name = obj["name"]
            object_type = obj["type"]
            object_id = obj.get("id")

            from core.relationship_validation import validate_relationship_semantics_strict

            is_valid, error_message = validate_relationship_semantics_strict(
                predicate_clean,
                subject_type,
                object_type,
            )

            if not is_valid:
                logger.warning(
                    "_build_relationship_statements: skipping semantically invalid relationship",
                    source=subject_name,
                    source_type=subject_type,
                    predicate=predicate_clean,
                    target=object_name,
                    target_type=object_type,
                    reason=error_message,
                )
                continue

            subject_label = _get_cypher_labels(subject_type).lstrip(":")
            object_label = _get_cypher_labels(object_type).lstrip(":")

            rel_id_source = f"{predicate_clean}|{subject_name.strip().lower()}|{object_name.strip().lower()}|{chapter}"
            rel_id = hashlib.sha1(rel_id_source.encode("utf-8")).hexdigest()[:16]

            query = """
            CALL apoc.merge.node(
                [$subject_label],
                {name: $subject_name},
                {
                    created_ts: timestamp(),
                    updated_ts: timestamp(),
                    created_chapter: $chapter,
                    name: $subject_name,
                    is_provisional: true,
                    description: 'Entity created from relationship extraction. Details to be developed.'
                },
                {updated_ts: timestamp()}
            ) YIELD node AS s

            SET s.id = coalesce(s.id, $subject_id, randomUUID())
            WITH s

            CALL apoc.merge.node(
                [$object_label],
                {name: $object_name},
                {
                    created_ts: timestamp(),
                    updated_ts: timestamp(),
                    created_chapter: $chapter,
                    name: $object_name,
                    is_provisional: true,
                    description: 'Entity created from relationship extraction. Details to be developed.'
                },
                {updated_ts: timestamp()}
            ) YIELD node AS o

            SET o.id = coalesce(o.id, $object_id, randomUUID())
            WITH s, o

            CALL apoc.merge.relationship(
                s,
                $predicate_clean,
                {id: $rel_id},
                apoc.map.merge(
                    {
                        chapter_added: $chapter,
                        is_provisional: $is_provisional,
                        confidence: $confidence,
                        description: $description,
                        last_updated: timestamp()
                    },
                    {created_ts: timestamp(), updated_ts: timestamp()}
                ),
                o,
                apoc.map.merge(
                    {
                        chapter_added: $chapter,
                        is_provisional: $is_provisional,
                        confidence: $confidence,
                        description: $description,
                        last_updated: timestamp()
                    },
                    {updated_ts: timestamp()}
                )
            ) YIELD rel
            RETURN rel
            """

            params = {
                "subject_label": subject_label,
                "subject_name": subject_name,
                "subject_id": subject_id,
                "object_label": object_label,
                "object_name": object_name,
                "object_id": object_id,
                "predicate_clean": predicate_clean,
                "rel_id": rel_id,
                "chapter": chapter,
                "is_provisional": is_from_flawed_draft,
                "confidence": triple.get("confidence", 1.0),
                "description": triple.get("description", ""),
            }

            logger.debug(
                "_build_relationship_statements: relationship query preview",
                query_preview=query.strip()[:350],
                subject_label=subject_label,
                object_label=object_label,
                predicate=predicate_clean,
                chapter=chapter,
            )

            statements.append((query, params))

        except ValueError as e:
            # CORE-011: persistence boundary contract violation (canonical labels / safe rel types).
            # Do NOT silently drop relationships; fail the commit path with a clear error.
            raise ValueError(f"Persistence boundary validation failed for relationship triple: {e}") from e
        except Exception as e:
            # Non-contract build errors are treated as best-effort (skip this triple) to avoid
            # failing the entire commit for incidental formatting issues.
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

    NOTE:
    This MUST use the canonical Chapter persistence semantics so Chapter nodes always
    have schema-required identity (`Chapter.id`) and we never create "number-only"
    Chapter nodes.

    Args:
        chapter_number: Chapter number
        text: Chapter text content
        word_count: Word count for metadata
        summary: Optional chapter summary
        embedding: Optional embedding vector

    Returns:
        Tuple of (cypher_query, parameters)
    """
    # Delegate to the authoritative Chapter persistence helper.
    query, parameters = chapter_queries.build_chapter_upsert_statement(
        chapter_number=chapter_number,
        summary=summary,
        embedding_vector=embedding,
        is_provisional=False,  # Chapter draft is validated before commit in the workflow
    )

    logger.debug(
        "_build_chapter_node_statement: built statement (canonical chapter upsert)",
        chapter=chapter_number,
        chapter_id=parameters.get("chapter_id_param"),
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
            success = await merge_duplicate_entities(entity1, entity2, entity_type="character")
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
            success = await merge_duplicate_entities(entity1, entity2, entity_type="world_element")
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

        if merge_counts["characters"] > 0 or merge_counts["world_items"] > 0:
            # Cache invalidation after Phase 2 merges
            #
            # Rationale:
            # - Phase 2 merges mutate Neo4j state *after* the main commit transaction.
            # - data_access read APIs are cached (async_lru), so failing to clear caches
            #   here can cause stale reads of now-merged/deleted entities.
            #
            # Local import avoids eager import side effects / circular deps.
            from data_access.cache_coordinator import (
                clear_character_read_caches,
                clear_kg_read_caches,
                clear_world_read_caches,
            )

            cleared_character = clear_character_read_caches()
            cleared_world = clear_world_read_caches()
            cleared_kg = clear_kg_read_caches()

            logger.debug(
                "_run_phase2_deduplication: invalidated caches after merges",
                chapter=chapter,
                cache_cleared={
                    "character": cleared_character,
                    "world": cleared_world,
                    "kg": cleared_kg,
                },
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


def _aggregate_scene_embeddings_to_chapter(scene_embeddings: list[list[float]] | dict[str, list[float]]) -> list[float]:
    """
    Aggregate scene-level embeddings into a single chapter embedding.

    Strategy: Average all scene embeddings to create a representative chapter embedding.
    This provides semantic coverage of the entire chapter while being computationally efficient.

    Args:
        scene_embeddings: List or dict of scene embedding vectors

    Returns:
        Single chapter embedding vector (averaged from all scenes)
    """
    if not scene_embeddings:
        return []

    # Handle both list and dict formats
    if isinstance(scene_embeddings, dict):
        embeddings_list = list(scene_embeddings.values())
    else:
        embeddings_list = scene_embeddings

    if not embeddings_list:
        return []

    # Convert to numpy array for efficient computation
    embeddings_array = np.array(embeddings_list)

    # Average across scenes (axis=0)
    chapter_embedding = np.mean(embeddings_array, axis=0).tolist()

    return chapter_embedding


__all__ = ["commit_to_graph"]
