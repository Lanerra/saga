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

Module layout (I4 refactor):
    commit_node.py           — main entry: commit_to_graph (~250 lines)
    commit_validation.py     — filtering + dedup logic
    commit_entity_conversion.py — CharacterProfile/WorldItem conversion
    commit_graph_ops.py      — chapter node + embedding aggregation
"""

from __future__ import annotations

from typing import Any, Literal, cast

import structlog

import config
from core.langgraph.chapter_lifecycle import ChapterLifecycle
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
from core.langgraph.nodes.commit_entity_conversion import (
    _convert_to_character_profiles,
    _convert_to_world_items,
)
from core.langgraph.nodes.commit_graph_ops import (
    _aggregate_scene_embeddings_to_chapter,
    _build_chapter_node_statement,
)
from core.langgraph.nodes.commit_validation import (
    _deduplicate_entity_list,
    _filter_invalid_relationships,
)
from core.langgraph.state import ExtractedEntity, ExtractedRelationship, NarrativeState
from core.schema_validator import canonicalize_entity_type_for_persistence
from data_access.cypher_builders.native_builders import chapter_assertion_delete_statement, relationship_statement
from data_access.kg_queries import (
    _get_cypher_labels as _get_cypher_labels,
)
from data_access.kg_queries import (
    validate_relationship_type,
    validate_relationship_type_for_cypher_interpolation,
)
from models.kg_models import CharacterProfile, WorldItem
from utils import classify_category_label

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
                        entity_name = entity.get("name")
                        resolved_type = entity.get("type")
                    else:
                        entity_name = getattr(entity, "name", None)
                        resolved_type = getattr(entity, "type", None)

                    if isinstance(entity_name, str) and entity_name and isinstance(resolved_type, str) and resolved_type:
                        entity_type_map[entity_name] = resolved_type

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


async def _get_existing_entity_names() -> set[str]:
    """Query Neo4j for names of all existing Character, Location, Event, and Item nodes.

    Returns:
        A set of lowercased entity names already persisted in the knowledge graph.
    """
    query = """
    MATCH (n)
    WHERE n:Character OR n:Location OR n:Event OR n:Item
    RETURN DISTINCT toLower(n.name) AS name
    """
    from core.service_context import get_services

    results = await get_services().database.execute_read_query(query, {})
    return {row["name"] for row in results if row.get("name")}


async def _prepare_explicit_entity_admission(
    entities: list[ExtractedEntity],
) -> tuple[list[tuple[str, dict[str, Any]]], frozenset[tuple[str, str]]]:
    """Validate explicit identities before protective name filtering.

    An absent ID retains legacy name-only admission. A supplied ID must be a
    nonblank string and cannot compete with another entity's normalized name,
    even across labels. Existing profiles never receive scene-level semantic
    updates; an otherwise admitted alias may receive embeddings and assertions.
    Recheck identity/existence in the write transaction before chapter deletion.
    """
    from core.service_context import get_services

    candidates: list[dict[str, Any]] = []
    identities: dict[tuple[str, str], ExtractedEntity] = {}
    for entity in entities:
        if "id" not in entity.attributes:
            continue
        identifier = entity.attributes["id"]
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Invalid canonical entity ID")
        label = canonicalize_entity_type_for_persistence(entity.type)
        if label != "Character":
            category = entity.attributes.get("category", entity.type.lower())
            if classify_category_label(category) != label:
                raise ValueError("Explicit canonical entity label conflicts with category")
        key = (label, identifier)
        if key in identities:
            if entity != identities[key]:
                raise ValueError("Conflicting explicit canonical entity inputs")
            continue
        identities[key] = entity
        for other in entities:
            if other.name.strip().lower() == entity.name.strip().lower() and (canonicalize_entity_type_for_persistence(other.type), other.attributes.get("id")) != key:
                raise ValueError("Explicit canonical entity ID/name conflict in batch")
        candidates.append({"index": len(candidates), "label": label, "id": identifier, "name": entity.name})
    if not candidates:
        return [], frozenset()

    query = """
        UNWIND $admission_entities AS entity
        OPTIONAL MATCH (candidate)
        WHERE entity.label IN labels(candidate) AND candidate.id = entity.id
        WITH entity, collect(candidate) AS candidates
        CALL apoc.util.validate(size(candidates) > 1, 'Ambiguous canonical entity ID', [])
        WITH entity, head(candidates) AS found
        OPTIONAL MATCH (named)
        WHERE (named:Character OR named:Location OR named:Item OR named:Event)
          AND toLower(trim(named.name)) = toLower(trim(entity.name))
        WITH entity, found, collect(named) AS names
        CALL apoc.util.validate(
            any(named IN names WHERE found IS NULL OR named <> found),
            'Explicit canonical entity ID/name conflict', [])
        CALL apoc.util.validate(
            entity.existing IS NOT NULL AND entity.existing <> (found IS NOT NULL),
            'Canonical entity changed during admission', [])
        RETURN entity.index AS index, found IS NOT NULL AS existing
    """
    rows = await get_services().database.execute_read_query(query, {"admission_entities": candidates})
    if {row["index"] for row in rows} != set(range(len(candidates))) or len(rows) != len(candidates):
        raise ValueError("Incomplete canonical entity admission")
    for row in rows:
        if type(row["existing"]) is not bool:
            raise ValueError("Invalid canonical entity admission result")
        candidates[row["index"]]["existing"] = row["existing"]
    protected = frozenset((entity["label"], entity["id"]) for entity in candidates if entity["existing"])
    return [(query, {"admission_entities": candidates})], protected


def _invalidate_postcommit_caches(chapter: int) -> None:
    """Attempt each cache group without changing the durable commit outcome."""
    try:
        from data_access.cache_coordinator import (
            clear_character_read_caches,
            clear_kg_read_caches,
            clear_world_read_caches,
        )
    except Exception as error:
        logger.warning("commit_to_graph: postcommit cache invalidation failed", chapter=chapter, error=str(error))
        return

    cleared: dict[str, dict[str, bool]] = {}
    for name, invalidate in (
        ("character", clear_character_read_caches),
        ("world", clear_world_read_caches),
        ("kg", clear_kg_read_caches),
    ):
        try:
            result = invalidate()
        except Exception as error:
            logger.warning("commit_to_graph: postcommit cache invalidation failed", chapter=chapter, cache=name, error=str(error))
            continue
        if not all(result.values()):
            logger.warning("commit_to_graph: postcommit cache invalidation failed", chapter=chapter, cache=name, cache_cleared=result)
        else:
            cleared[name] = result

    if len(cleared) == 3:
        logger.info("commit_to_graph: postcommit caches invalidated", chapter=chapter, cache_cleared=cleared)


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

        On errors, returns a state with `has_fatal_error` set and `last_error`
        populated.

    Notes:
        - This node performs Neo4j I/O via
          [`get_services().database.execute_cypher_batch()`](core/db_manager.py:310).
        - After successful writes it clears `data_access` read caches to prevent
          stale reads within the same process.
        - The database manager owns transaction rollback. Preparation or batch
          failures never trigger chapter-wide compensating deletes.
        - Cache failures after a successful batch are operational warnings, not
          failed graph commits. A batch exception is not proof of rollback.
    """
    lifecycle = None
    if "lifecycle_version" in state:
        try:
            lifecycle = ChapterLifecycle(state).stage()
            state = {**state, **lifecycle.state_update("staged")}
            receipt = await lifecycle.graph_receipt()
            if lifecycle.files.exists(lifecycle.phase_path("compensation_required")):
                raise ValueError("Rejected attempt requires revision reconciliation")
            if receipt is not None:
                if receipt["phase"] != "committed":
                    raise ValueError("Attempt is not eligible for commit replay")
                lifecycle.observe("committed")
                _invalidate_postcommit_caches(lifecycle.chapter_number)
                return {**lifecycle.state_update("committed"), "current_node": "commit_to_graph", "has_fatal_error": False, "last_error": None}
        except Exception as error:
            return {"current_node": "commit_to_graph", "last_error": str(error), "has_fatal_error": True, "error_node": "commit"}

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

    # Reject invalid extraction before constructing graph writes.
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

    relationship_characters = char_entities
    relationship_world_items = world_entities

    # Track mappings for deduplication
    char_mappings: dict[str, str] = {}
    world_mappings: dict[str, str] = {}

    try:
        admission_statements, protected_identities = await _prepare_explicit_entity_admission([*char_entities, *world_entities])
        # Name-only and same-name profiles retain legacy protection, while explicit
        # ID/name conflicts must fail before any candidate can be silently filtered.
        existing_names = await _get_existing_entity_names()
        char_entities = [entity for entity in char_entities if entity.name.lower() not in existing_names]
        world_entities = [entity for entity in world_entities if entity.name.lower() not in existing_names]

        for char in char_entities:
            char_mappings[char.name] = char.name

        # Missing IDs use the same canonical graph resolver as characters.
        # Python punctuation normalization must not collapse distinct named places.
        for item in world_entities:
            if "id" in item.attributes:
                world_mappings[item.name] = item.attributes["id"]

        # Deduplicate entity lists to prevent creating duplicate models
        unique_char_entities = _deduplicate_entity_list(char_entities)
        unique_world_entities = _deduplicate_entity_list(world_entities)

        character_models = _convert_to_character_profiles(unique_char_entities, char_mappings, state.get("current_chapter", 1))
        world_item_models = _convert_to_world_items(unique_world_entities, world_mappings, state.get("current_chapter", 1))

        # All chapter writes share one transaction.
        all_statements: list[tuple[str, dict]] = [*admission_statements, chapter_assertion_delete_statement(chapter)]

        # Step 4a: Collect entity persistence statements
        if character_models or world_item_models:
            entity_statements = await _build_entity_persistence_statements(
                character_models,
                world_item_models,
                state.get("current_chapter", 1),
                protected_identities=protected_identities,
            )
            all_statements.extend(entity_statements)

        # Step 4b: Collect relationship statements
        #
        # Contract: relationship writes are chapter-idempotent.
        # Every commit replaces the chapter's relationship set (including "no relationships").

        # Validate the entire relationship batch before any graph write.
        relationships = _filter_invalid_relationships(relationships)

        relationship_statements = await _build_relationship_statements(
            relationships,
            relationship_characters,
            relationship_world_items,
            char_mappings,
            world_mappings,
            state.get("current_chapter", 1),
            is_from_flawed_draft=False,
        )
        if relationship_statements:
            all_statements.extend(relationship_statements[1:])

        # Step 4c: Collect chapter node statement
        content_manager = ContentManager(require_project_dir(state))

        from core.exceptions import MissingDraftReferenceError

        try:
            get_draft_text(state, content_manager)
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
            raise ValueError("Legacy generated_embedding has no producer identity; regenerate an identified embedding artifact")

        chapter_statement = _build_chapter_node_statement(
            chapter_number=state.get("current_chapter", 1),
            summary=None,
            embedding=embedding,
        )
        all_statements.append(chapter_statement)

        # If any statement fails, all are rolled back
        if all_statements:
            from core.service_context import get_services

            if lifecycle is None:
                await get_services().database.execute_cypher_batch(all_statements)
            else:
                await lifecycle.commit(all_statements)

            logger.info(
                "commit_to_graph: successfully committed to knowledge graph in single transaction",
                chapter=state.get("current_chapter", 1),
                characters=len(character_models),
                world_items=len(world_item_models),
                relationships=len(relationships),
                total_statements=len(all_statements),
            )

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

    _invalidate_postcommit_caches(chapter)

    return {
        **(lifecycle.state_update("committed") if lifecycle is not None else {}),
        "current_node": "commit_to_graph",
        "last_error": None,
        "has_fatal_error": False,
    }


async def _build_entity_persistence_statements(
    characters: list[CharacterProfile],
    world_items: list[WorldItem],
    chapter_number: int,
    *,
    protected_identities: frozenset[tuple[str, str]] = frozenset(),
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

    from data_access.cypher_builders.native_builders import NativeCypherBuilder

    cypher_builder = NativeCypherBuilder()

    # Generate Cypher for characters
    for char in characters:
        if ("Character", char.id) in protected_identities:
            continue
        cypher, params = cypher_builder.character_upsert_cypher(char.model_copy(update={"relationships": {}}), chapter_number, assertion_origin="chapter_profile")
        statements.append((cypher, params))

    # Generate Cypher for world items
    for item in world_items:
        if (classify_category_label(item.category), item.id) in protected_identities:
            continue
        cypher, params = cypher_builder.world_item_upsert_cypher(item.model_copy(update={"relationships": {}}), chapter_number, assertion_origin="chapter_profile")
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

    statements.append(chapter_assertion_delete_statement(chapter))

    # Build entity lookup maps for type resolution
    entity_type_map = {}
    entity_category_map = {}

    for entity in char_entities:
        entity_type_map[entity.name] = entity.type
        entity_category_map[entity.name] = entity.attributes.get("category", "")

    for entity in world_entities:
        if entity.name in entity_type_map and entity_type_map[entity.name] != entity.type:
            raise ValueError("Ambiguous relationship endpoint identity across entity types")
        entity_type_map[entity.name] = entity.type
        entity_category_map[entity.name] = entity.attributes.get("category", "")

    logger.debug(
        "_build_relationship_statements: entity type map",
        entity_count=len(entity_type_map),
        entity_types=list(entity_type_map.items())[:10],  # Log first 10 for debugging
    )

    entity_identity_map = {(canonicalize_entity_type_for_persistence(entity.type), entity.name): entity.attributes.get("id") for entity in [*char_entities, *world_entities]}

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
        known_type = entity_type_map.get(original_name)
        if explicit_type is not None and known_type is not None and (canonicalize_entity_type_for_persistence(explicit_type) != canonicalize_entity_type_for_persistence(known_type)):
            raise ValueError("Relationship endpoint type conflicts with extracted entity identity")

        if not entity_type or not str(entity_type).strip():
            inferred_type = None
            if relationship_type and role in ("source", "target"):
                inferred_type = infer_entity_type_from_relationship(
                    name,
                    relationship_type,
                    cast(Literal["source", "target"], role),
                )

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

        resolved_stable_id = stable_id if stable_id is not None else entity_identity_map.get((neo4j_type, original_name))
        known_id = entity_identity_map.get((neo4j_type, original_name))
        if stable_id is not None and known_id is not None and stable_id != known_id:
            raise ValueError("Relationship endpoint ID conflicts with extracted entity identity")
        if resolved_stable_id is None and neo4j_type != "Character":
            resolved_stable_id = world_mappings.get(original_name)

        return {
            "name": name,
            "id": resolved_stable_id,
            "type": neo4j_type,
            "category": entity_category,
        }

    # Convert to triple format
    structured_triples: list[dict[str, Any]] = []

    for entity in [*char_entities, *world_entities]:
        source = _make_entity_dict(name=entity.name, original_name=entity.name, explicit_type=entity.type)
        for target_name, information in entity.attributes.get("relationships", {}).items():
            if not isinstance(information, dict):
                raise ValueError("Profile relationship must be a dictionary")
            target_type = information.get("target_label", "Character" if source["type"] == "Character" else "Item")
            target = _make_entity_dict(name=target_name, original_name=target_name, explicit_type=target_type, stable_id=information.get("target_id"))
            structured_triples.append(
                {
                    "subject": source,
                    "predicate": information["type"],
                    "object_entity": target,
                    "description": information.get("description", ""),
                    "confidence": 1.0,
                    "assertion_origin": "chapter_profile",
                }
            )

    for rel in relationships:
        # `char_mappings` canonicalizes character names for consistent relationship endpoints.
        source_name = char_mappings.get(rel.source_name, rel.source_name)
        target_name = char_mappings.get(rel.target_name, rel.target_name)
        if (rel.source_id is not None and source_name != rel.source_name) or (rel.target_id is not None and target_name != rel.target_name):
            raise ValueError("Explicit relationship identity cannot authorize a name alias")
        if rel.chapter != chapter:
            raise ValueError("Relationship chapter conflicts with commit chapter")

        # Use explicit types from relationship if available (from parsing "Type:Name" format)
        # Otherwise _make_entity_dict will fall back to entity_type_map
        source_type = getattr(rel, "source_type", None)
        target_type = getattr(rel, "target_type", None)

        triple: dict[str, Any] = {
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
            "scene_index": rel.scene_index,
            "scene_assertions": rel.scene_assertions,
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
                raise ValueError("Relationship endpoints must be typed entity dictionaries")

            subject_name = subject["name"]
            subject_type = subject["type"]
            subject_id = subject.get("id")

            if not isinstance(predicate, str):
                predicate = str(predicate)

            predicate_normalized = validate_relationship_type(predicate)
            predicate_clean = validate_relationship_type_for_cypher_interpolation(predicate_normalized)

            if not predicate_clean:
                raise ValueError("Relationship predicate must not be empty")

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

            query, params = relationship_statement(
                {"name": subject_name, "type": subject_label, "id": subject_id},
                predicate_clean,
                {"name": object_name, "type": object_label, "id": object_id},
                chapter,
                origin=triple.get("assertion_origin", "chapter_extraction"),
                provisional=is_from_flawed_draft,
                confidence=triple.get("confidence", 1.0),
                description=triple.get("description", ""),
                scene_index=triple.get("scene_index"),
                scene_assertions=triple.get("scene_assertions"),
            )

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

    logger.info(
        "_build_relationship_statements: built statements",
        relationships=len(relationships),
        statements=len(statements),
    )

    return statements


__all__ = ["commit_to_graph"]
