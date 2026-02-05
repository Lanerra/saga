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
from utils.text_processing import generate_entity_id, validate_and_filter_traits

logger = structlog.get_logger(__name__)


async def _validate_entities_before_commit(
    extracted_entities: dict[str, list[ExtractedEntity]],
    extracted_relationships: list[ExtractedRelationship],
    chapter: int,
) -> tuple[bool, list[str]]:
    """Validate entities and relationships before committing to Neo4j.

    This function performs pre-commit validation to catch issues before
    writing to the database. It checks for:
    - Invalid entity types
    - Invalid relationship types
    - Missing required fields
    - Semantic validation issues

    Args:
        extracted_entities: Dictionary of extracted entities by type
        extracted_relationships: List of extracted relationships
        chapter: Current chapter number

    Returns:
        Tuple of (is_valid, errors) where is_valid is True if validation passes,
        and errors is a list of error messages.
    """
    errors: list[str] = []

    # Validate extracted entities
    for entity_type, entities in extracted_entities.items():
        if not isinstance(entities, list):
            errors.append(f"Invalid {entity_type} entities: expected list, got {type(entities)}")
            continue

        for i, entity in enumerate(entities):
            if not isinstance(entity, dict | ExtractedEntity):
                errors.append(f"Invalid {entity_type} entity at index {i}: expected ExtractedEntity or dict, got {type(entity)}")
                continue

            # Convert to ExtractedEntity if needed for consistent validation
            if isinstance(entity, dict):
                entity = ExtractedEntity(**entity)

            # Check required fields
            if not entity.name or not isinstance(entity.name, str):
                errors.append(f"Entity {entity_type} at index {i} is missing valid name")

            if not entity.type or not isinstance(entity.type, str):
                errors.append(f"Entity {entity_type} '{entity.name}' is missing valid type")

    # Validate extracted relationships
    for i, relationship in enumerate(extracted_relationships):
        if not isinstance(relationship, dict | ExtractedRelationship):
            errors.append(f"Invalid relationship at index {i}: expected ExtractedRelationship or dict, got {type(relationship)}")
            continue

        # Convert to ExtractedRelationship if needed
        if isinstance(relationship, dict):
            relationship = ExtractedRelationship(**relationship)

        # Check required fields
        if not relationship.source_name or not isinstance(relationship.source_name, str):
            errors.append(f"Relationship at index {i} is missing valid source_name")

        if not relationship.target_name or not isinstance(relationship.target_name, str):
            errors.append(f"Relationship at index {i} is missing valid target_name")

        if not relationship.relationship_type or not isinstance(relationship.relationship_type, str):
            errors.append(f"Relationship between {relationship.source_name} and {relationship.target_name} is missing valid relationship_type")

    # Validate relationship semantics
    if extracted_relationships:
        try:
            from core.relationship_validation import get_relationship_validator

            validator = get_relationship_validator()

            # Build entity type lookup
            entity_type_map: dict[str, str] = {}
            for entities in extracted_entities.values():
                for entity in entities:
                    if isinstance(entity, dict):
                        name = entity.get("name")
                        entity_type = entity.get("type")
                    else:
                        name = getattr(entity, "name", None)
                        entity_type = getattr(entity, "type", None)

                    if isinstance(name, str) and name and isinstance(entity_type, str) and entity_type:
                        entity_type_map[name] = entity_type

            # Validate each relationship
            for _, rel in enumerate(extracted_relationships):
                if isinstance(rel, dict):
                    relationship_type = rel.get("relationship_type")
                    source_name = rel.get("source_name")
                    target_name = rel.get("target_name")
                else:
                    relationship_type = getattr(rel, "relationship_type", None)
                    source_name = getattr(rel, "source_name", None)
                    target_name = getattr(rel, "target_name", None)

                if not (isinstance(relationship_type, str) and relationship_type):
                    continue
                if not (isinstance(source_name, str) and source_name):
                    continue
                if not (isinstance(target_name, str) and target_name):
                    continue

                source_type = entity_type_map.get(source_name, "Character")
                target_type = entity_type_map.get(target_name, "Character")

                # Validate relationship (permissive mode - log warnings but don't fail)
                is_valid, errors_list, info_warnings = validator.validate(
                    relationship_type=relationship_type,
                    source_name=source_name,
                    source_type=source_type,
                    target_name=target_name,
                    target_type=target_type,
                    severity_mode="flexible",
                )

                # Log warnings but don't fail validation
                if info_warnings:
                    logger.warning(
                        "pre_commit_validation: relationship validation warnings",
                        relationship=f"{source_name}({source_type}) -{relationship_type}-> {target_name}({target_type})",
                        warnings=info_warnings,
                        chapter=chapter,
                    )
        except Exception as e:
            logger.warning(
                "pre_commit_validation: relationship validation failed",
                error=str(e),
                chapter=chapter,
            )

    return (len(errors) == 0, errors)


async def _rollback_commit(chapter: int) -> bool:
    """Rollback the commit for the specified chapter.

    This function removes all data committed for a specific chapter,
    including entities, relationships, and chapter nodes created in that commit.

    Args:
        chapter: Chapter number to rollback

    Returns:
        True if rollback was successful, False otherwise.
    """
    try:
        from core.db_manager import neo4j_manager

        # Build rollback statements
        statements = []

        # 1. Delete relationships added in this chapter
        delete_rels_query = """
        MATCH ()-[r]->()
        WHERE coalesce(r.chapter_added, -1) = $chapter
        DELETE r
        """
        statements.append((delete_rels_query, {"chapter": chapter}))

        # 2. Delete entities created in this chapter (that aren't referenced by other chapters)
        # Only delete if created_chapter equals chapter and no other relationships exist
        delete_entities_query = """
        MATCH (n)
        WHERE n.created_chapter = $chapter
        AND NOT (n)--()
        DELETE n
        """
        statements.append((delete_entities_query, {"chapter": chapter}))

        # 3. Delete chapter node
        delete_chapter_query = """
        MATCH (c:Chapter {chapter_number: $chapter})
        DELETE c
        """
        statements.append((delete_chapter_query, {"chapter": chapter}))

        # Execute rollback in a transaction
        await neo4j_manager.execute_cypher_batch(statements)

        logger.info(
            "_rollback_commit: successfully rolled back chapter",
            chapter=chapter,
        )

        return True

    except Exception as e:
        logger.error(
            "_rollback_commit: rollback failed",
            error=str(e),
            chapter=chapter,
            exc_info=True,
        )
        return False


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
        - If validation fails after commit, the node will attempt to rollback
          the changes.
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

    # Step 0: Pre-commit validation
    # Validate entities and relationships before committing to database
    chapter = state.get("current_chapter", 1)
    extracted_entities_dict = {
        "characters": char_entities,
        "world_items": world_entities,
    }

    is_valid, validation_errors = await _validate_entities_before_commit(
        extracted_entities_dict,
        relationships,
        chapter,
    )

    if not is_valid:
        error_msg = f"Pre-commit validation failed: {', '.join(validation_errors[:5])}"
        if len(validation_errors) > 5:
            error_msg += f" (and {len(validation_errors) - 5} more errors)"
        logger.error(
            "commit_to_graph: pre-commit validation failed",
            error=error_msg,
            chapter=chapter,
            error_count=len(validation_errors),
        )
        return {
            "current_node": "commit_to_graph",
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "commit",
        }

    # Track mappings for deduplication (kept for backward compatibility)
    char_mappings: dict[str, str] = {}  # new_name -> existing_name (or same)
    world_mappings: dict[str, str] = {}  # new_name -> existing_id (or new_id)

    try:
        # Step 1: Deduplicate characters (READ operations)
        # Since entities are canonical from Stage 1, no deduplication is needed
        for char in char_entities:
            char_mappings[char.name] = char.name

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

            # First time seeing this name, generate deterministic ID
            deduplicated_id = generate_entity_id(
                item.name,
                item.attributes.get("category", ""),
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

        # Filter out invalid abstract-concept relationships
        relationships = _filter_invalid_relationships(relationships)

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

        return {
            "current_node": "commit_to_graph",
            "last_error": None,
            "has_fatal_error": False,
        }

    except Exception as e:
        logger.error(
            "commit_to_graph: fatal error during commit",
            error=str(e),
            chapter=state.get("current_chapter", 1),
            exc_info=True,
        )

        # Attempt rollback on commit failure
        chapter = state.get("current_chapter", 1)
        rollback_success = await _rollback_commit(chapter)

        if rollback_success:
            logger.info(
                "commit_to_graph: rollback successful after commit failure",
                chapter=chapter,
            )
        else:
            logger.warning(
                "commit_to_graph: rollback failed after commit failure",
                chapter=chapter,
            )

        return {
            "current_node": "commit_to_graph",
            "last_error": f"Commit to graph failed: {e}",
            "has_fatal_error": True,
            "error_node": "commit",
        }


def _filter_invalid_relationships(
    relationships: list[ExtractedRelationship],
) -> list[ExtractedRelationship]:
    """Filter out relationships with abstract/invalid entities.

    Rejects relationships where the target is:
    - Generic concepts (truth, knowledge, secrets, understanding)
    - Descriptive phrases (unknown dangers, the bloom)
    - Relationship pairs (Elias and Caleb)
    - Goals/actions (to understand the bloom)

    Args:
        relationships: Extracted relationships to validate.

    Returns:
        Filtered list containing only relationships with valid, concrete entities.
    """
    import re

    invalid_patterns = [
        r"^(the|a|an)\s",  # Articles: "the bloom", "a secret"
        r"\s+and\s+",  # Pairs: "Elias and Caleb"
        r"^to\s",  # Goals: "to understand"
        r"(truth|knowledge|secrets?|understanding|consequences)",  # Abstract concepts
        r"(unknown|mysterious)\s",  # Descriptive adjectives
        r"('s|')\s",  # Possessives: "Elias's decision"
    ]

    combined_pattern = "|".join(f"({p})" for p in invalid_patterns)
    pattern = re.compile(combined_pattern, re.IGNORECASE)

    valid = []
    filtered_count = 0

    for rel in relationships:
        target = rel.target_name.strip()
        source = rel.source_name.strip()

        # Reject if target matches invalid pattern
        if pattern.search(target):
            logger.debug(
                "_filter_invalid_relationships: rejected abstract target",
                source=source,
                predicate=rel.relationship_type,
                target=target,
            )
            filtered_count += 1
            continue

        # Reject if source matches invalid pattern
        if pattern.search(source):
            logger.debug(
                "_filter_invalid_relationships: rejected abstract source",
                source=source,
                predicate=rel.relationship_type,
                target=target,
            )
            filtered_count += 1
            continue

        # Reject single-word lowercase concepts (except proper names)
        if " " not in target and target.islower() and target not in ["bayou", "plantation"]:
            logger.debug(
                "_filter_invalid_relationships: rejected lowercase concept",
                target=target,
            )
            filtered_count += 1
            continue

        valid.append(rel)

    if filtered_count > 0:
        logger.info(
            "_filter_invalid_relationships: filtered abstract concepts",
            original_count=len(relationships),
            valid_count=len(valid),
            filtered_count=filtered_count,
        )

    return valid


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
                personality_description=entity.description,
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

            # Generate stable relationship ID WITHOUT chapter number to prevent duplicates
            # This ensures "Elias TRUSTS Caleb" creates ONE edge, not one per chapter
            rel_id_source = f"{predicate_clean}|{subject_name.strip().lower()}|{object_name.strip().lower()}"
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
                {
                    chapter_added: $chapter,
                    is_provisional: $is_provisional,
                    confidence: $confidence,
                    description: $description,
                    created_ts: timestamp(),
                    updated_ts: timestamp()
                },
                o,
                {
                    is_provisional: $is_provisional,
                    confidence: $confidence,
                    description: $description,
                    updated_ts: timestamp()
                }
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
