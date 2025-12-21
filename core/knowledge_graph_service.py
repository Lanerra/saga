# core/knowledge_graph_service.py
"""Provide a knowledge graph persistence compatibility layer.

This module exists to preserve legacy import paths that are patched by tests and older
LangGraph call sites (for example, patching
[`core.knowledge_graph_service.knowledge_graph_service`](core/knowledge_graph_service.py:140)).

Notes:
    Persistence is implemented using [`NativeCypherBuilder`](data_access/cypher_builders/native_builders.py:1)
    and [`neo4j_manager.execute_cypher_batch()`](core/db_manager.py:1). This module does not perform
    dynamic Cypher interpolation beyond what the builder guarantees.
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
    """Persist entity nodes to Neo4j for legacy call sites."""

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
        """Persist entity nodes into Neo4j.

        Args:
            characters: Characters to upsert.
            world_items: World items to upsert.
            chapter_number: Chapter number recorded on persisted entities.
            extra_statements: Additional `(query, params)` statements to run in the same
                batch transaction.
            strict: Whether to raise a typed exception on failure.

        Returns:
            True on success. When `strict=False`, returns False on failure.

        Raises:
            KnowledgeGraphPersistenceError: When `strict=True` and batch persistence fails.

        Notes:
            CORE-007: `strict=True` is the default and raises a typed exception rather
            than returning ambiguous sentinels.
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
