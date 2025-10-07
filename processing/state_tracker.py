# processing/state_tracker.py
"""StateTracker for managing entity metadata during bootstrap generation.

Tracks entity names, types, descriptions, and timestamps to detect and prevent
conflicts during generation. While the pipeline is sequential for SAGA, this
remains async for simplicity and future-proofing.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import structlog

import config

logger = structlog.get_logger(__name__)


class StateTracker:
    """Tracks entity metadata during bootstrap generation to prevent duplication."""

    def __init__(self) -> None:
        """Initialize the StateTracker with empty state."""
        self._entities: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def reserve(self, name: str, entity_type: str, description: str) -> bool:
        """Reserve an entity name, storing metadata.

        Args:
            name: The entity name to reserve
            entity_type: Type of entity (e.g., "character", "world_item")
            description: Description of the entity

        Returns:
            True if reservation successful, False if name already reserved
        """
        if not config.STATE_TRACKER_ENABLED:
            return True

        async with self._lock:
            if name in self._entities:
                logger.debug(
                    "Entity name already reserved",
                    name=name,
                    existing_type=self._entities[name]["type"],
                )
                return False

            self._entities[name] = {
                "name": name,
                "type": entity_type,
                "description": description,
                "timestamp": datetime.now().isoformat(),
            }

            logger.debug("Entity name reserved", name=name, type=entity_type)
            return True

    async def check(self, name: str) -> dict[str, Any] | None:
        """Check if a name is reserved, returning metadata if exists.

        Args:
            name: The entity name to check

        Returns:
            Dictionary with metadata if name is reserved, None otherwise
        """
        if not config.STATE_TRACKER_ENABLED:
            return None

        async with self._lock:
            return self._entities.get(name)

    async def release(self, name: str) -> None:
        """Release a reserved name, removing its metadata.

        Args:
            name: The entity name to release
        """
        if not config.STATE_TRACKER_ENABLED:
            return

        async with self._lock:
            if name in self._entities:
                del self._entities[name]
                logger.debug("Entity name released", name=name)

    async def rename(self, old_name: str, new_name: str) -> bool:
        """Atomically rename a tracked entity key from old_name to new_name.

        Preserves metadata and updates the embedded "name" field and timestamp.

        Returns True on success, False if old_name is not tracked or new_name
        already exists.
        """
        if not config.STATE_TRACKER_ENABLED:
            return True

        async with self._lock:
            if old_name not in self._entities:
                return False
            if new_name in self._entities and new_name != old_name:
                # Do not overwrite existing entries
                return False
            metadata = self._entities.pop(old_name)
            metadata["name"] = new_name
            metadata["timestamp"] = datetime.now().isoformat()
            self._entities[new_name] = metadata
            logger.debug("Entity name renamed", old_name=old_name, new_name=new_name)
            return True

    async def get_all(self) -> dict[str, dict[str, Any]]:
        """Return all tracked entities.

        Returns:
            Dictionary mapping entity names to their metadata
        """
        async with self._lock:
            return self._entities.copy()

    async def clear(self) -> None:
        """Clear all tracked entities. Used for testing and cleanup."""
        async with self._lock:
            self._entities.clear()
            logger.debug("StateTracker cleared")

    async def has_similar_description(
        self, description: str, entity_type: str | None = None
    ) -> str | None:
        """Check if any tracked entity has a similar description.

        Args:
            description: Description to check for similarity
            entity_type: Optional entity type to filter by

        Returns:
            Name of entity with similar description, None if no match
        """
        if not config.STATE_TRACKER_ENABLED:
            return None

        # Simple similarity check - could be enhanced with semantic similarity
        description_lower = description.lower().strip()

        async with self._lock:
            for name, metadata in self._entities.items():
                if entity_type and metadata["type"] != entity_type:
                    continue

                existing_desc = metadata["description"].lower().strip()

                # Basic similarity heuristics
                if len(description_lower) > 20 and len(existing_desc) > 20:
                    # Check for substantial overlap in longer descriptions
                    desc_words = set(description_lower.split())
                    existing_words = set(existing_desc.split())

                    if desc_words and existing_words:
                        similarity = len(desc_words.intersection(existing_words)) / len(
                            desc_words.union(existing_words)
                        )

                        if similarity >= config.STATE_TRACKER_SIMILARITY_THRESHOLD:
                            logger.debug(
                                "Similar description found",
                                name=name,
                                similarity=similarity,
                                threshold=config.STATE_TRACKER_SIMILARITY_THRESHOLD,
                            )
                            return name

        return None

    async def get_entities_by_type(self, entity_type: str) -> dict[str, dict[str, Any]]:
        """Get all entities of a specific type.

        Args:
            entity_type: The type of entities to retrieve

        Returns:
            Dictionary mapping entity names to metadata for given type
        """
        async with self._lock:
            return {
                name: metadata
                for name, metadata in self._entities.items()
                if metadata["type"] == entity_type
            }

    async def get_recent_entities(self, hours: int = 24) -> dict[str, dict[str, Any]]:
        """Get entities created within the specified time window.

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary mapping entity names to metadata for recent entities
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(hours=hours)

        async with self._lock:
            recent_entities = {}

            for name, metadata in self._entities.items():
                try:
                    entity_time = datetime.fromisoformat(metadata["timestamp"])
                    if entity_time >= cutoff:
                        recent_entities[name] = metadata
                except ValueError:
                    # Skip entities with invalid timestamps
                    continue

            return recent_entities

    async def get_name_diversity_score(self, new_name: str) -> float:
        """Calculate diversity score for a new name compared to existing character names.

        Returns a value in [0,1], where higher means more diverse (less similar).
        """
        async with self._lock:
            if not self._entities:
                return 1.0

            diversity_scores: list[float] = []
            for name, metadata in self._entities.items():
                if metadata["type"] == "character":
                    score = _st_calculate_name_distance(new_name, name)
                    diversity_scores.append(score)

            if not diversity_scores:
                return 1.0

            avg_similarity = sum(diversity_scores) / len(diversity_scores)
            return 1.0 - avg_similarity

    async def check_name_diversity(
        self, new_name: str, min_diversity_threshold: float = 0.4
    ) -> bool:
        """Check if a new name meets diversity requirements.

        Returns True if diversity is acceptable (>= threshold).
        """
        diversity_score = await self.get_name_diversity_score(new_name)
        return diversity_score >= min_diversity_threshold


def _st_calculate_name_distance(name1: str, name2: str) -> float:
    """Lightweight similarity ratio for names used by StateTracker."""
    from difflib import SequenceMatcher

    return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
