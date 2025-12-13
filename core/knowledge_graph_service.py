# core/knowledge_graph_service.py
"""
Knowledge Graph persistence service (compatibility layer).

Why this exists:
- Some LangGraph code/tests patch `core.knowledge_graph_service.knowledge_graph_service`.
- During the P0 refactors, the old module appears to have been removed/relocated.
- `unittest.mock.patch()` resolves dotted names by attribute-walking the package, so
  we must provide both:
  1) a real submodule `core.knowledge_graph_service`, and
  2) a `core.__init__` attribute pointing to it (handled separately).

This module provides a minimal, safe persistence facade around the new Cypher builder
approach (NativeCypherBuilder + execute_cypher_batch).

It intentionally does NOT perform any dynamic Cypher interpolation beyond what the
builder already guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from core.db_manager import neo4j_manager
from core.exceptions import KnowledgeGraphPersistenceError, create_error_context
from data_access.cypher_builders.native_builders import NativeCypherBuilder
from models.kg_models import CharacterProfile, WorldItem

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class KnowledgeGraphService:
    """
    Minimal persistence service used by legacy call-sites and tests.

    The primary method, persist_entities(), builds upsert Cypher statements for
    characters and world items and executes them in a single batch transaction.
    """

    cypher_builder: NativeCypherBuilder

    async def persist_entities(
        self,
        *,
        characters: list[CharacterProfile] | None = None,
        world_items: list[WorldItem] | None = None,
        chapter_number: int = 0,
        extra_statements: list[tuple[str, dict[str, Any]]] | None = None,
        strict: bool = True,
    ) -> bool:
        """
        Persist entity nodes into Neo4j.

        Args:
            characters: CharacterProfile models to upsert.
            world_items: WorldItem models to upsert.
            chapter_number: Chapter number for metadata tracking.
            extra_statements: Optional additional (query, params) statements to include
                in the same batch transaction.

        Returns:
            True on success.

            Compatibility behavior:
            - If `strict=False`, returns False on failure instead of raising.
            - CORE-007: `strict=True` is the default and raises a typed exception on failure.
        """
        characters = characters or []
        world_items = world_items or []
        extra_statements = extra_statements or []

        statements: list[tuple[str, dict[str, Any]]] = []

        try:
            for char in characters:
                cypher, params = self.cypher_builder.character_upsert_cypher(char, chapter_number)
                statements.append((cypher, params))

            for item in world_items:
                cypher, params = self.cypher_builder.world_item_upsert_cypher(item, chapter_number)
                statements.append((cypher, params))

            statements.extend(extra_statements)

            if not statements:
                logger.debug(
                    "knowledge_graph_service.persist_entities: nothing to persist",
                    chapter=chapter_number,
                )
                return True

            await neo4j_manager.execute_cypher_batch(statements)

            logger.info(
                "knowledge_graph_service.persist_entities: persisted entities",
                chapter=chapter_number,
                characters=len(characters),
                world_items=len(world_items),
                statements=len(statements),
            )
            return True

        except Exception as exc:
            error_details = create_error_context(
                chapter=chapter_number,
                characters=len(characters),
                world_items=len(world_items),
                statements=len(statements),
                error=str(exc),
                error_type=type(exc).__name__,
            )

            logger.error(
                "knowledge_graph_service.persist_entities: failed",
                **error_details,
                exc_info=True,
            )

            if strict:
                raise KnowledgeGraphPersistenceError(
                    "Failed to persist entities to knowledge graph",
                    details=error_details,
                ) from exc

            # Compatibility: explicit non-strict mode preserves legacy boolean failure signal.
            return False


# Singleton service instance (as expected by tests patching this symbol)
knowledge_graph_service = KnowledgeGraphService(cypher_builder=NativeCypherBuilder())

__all__ = ["KnowledgeGraphService", "knowledge_graph_service"]
