# core/langgraph/nodes/scene_extraction_validation.py
"""Entity validation and normalization helpers for scene extraction.

This module is separated from scene_extraction.py to keep spaCy-dependent logic
isolated, avoiding eager spaCy model loading at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

import config

if TYPE_CHECKING:
    from core.text_processing_service import TextProcessingService

logger = structlog.get_logger(__name__)

_text_processing_service: TextProcessingService | None = None


def _get_text_processing_service() -> TextProcessingService:
    """Lazily initialize the TextProcessingService to avoid loading spaCy at import time."""
    global _text_processing_service
    if _text_processing_service is None:
        # Deferred import to avoid circular references
        from core.text_processing_service import TextProcessingService

        _text_processing_service = TextProcessingService()
    return _text_processing_service


def _validate_entity_with_spacy(scene_text: str, entity_name: str) -> bool:
    """Validate that an extracted entity is actually present in the scene text.

    Args:
        scene_text: The source scene text.
        entity_name: The entity name to validate.

    Returns:
        True if entity is validated (present or validation disabled), False if not found.
    """
    if not config.settings.ENABLE_ENTITY_VALIDATION:
        logger.debug("_validate_entity_with_spacy: entity validation disabled by config")
        return True

    if not _get_text_processing_service().spacy_service.is_loaded():
        logger.warning("_validate_entity_with_spacy: spaCy model not loaded, skipping validation")
        return True

    try:
        is_present = _get_text_processing_service().spacy_service.verify_entity_presence(
            scene_text, entity_name, threshold=0.7
        )

        if not is_present:
            logger.warning(
                "_validate_entity_with_spacy: entity not found in text",
                entity_name=entity_name,
                entity_length=len(entity_name),
                scene_text_length=len(scene_text),
            )

        return is_present
    except Exception as e:
        logger.error("_validate_entity_with_spacy: validation failed, using fallback", error=str(e))
        # Fallback to simple substring matching
        return entity_name.lower() in scene_text.lower()


def _get_normalized_entity_key(name: str) -> str:
    """Get a normalized key for entity deduplication using spaCy.

    Args:
        name: The entity name to normalize.

    Returns:
        Normalized key for deduplication.
    """
    if (
        config.settings.ENABLE_ENTITY_VALIDATION
        and _get_text_processing_service().spacy_service.is_loaded()
    ):
        try:
            return _get_text_processing_service().spacy_service.normalize_entity_name(name)
        except Exception as e:
            logger.warning(
                "_get_normalized_entity_key: spaCy normalization failed, using fallback",
                error=str(e),
            )

    # Fallback to simple case-insensitive normalization
    return name.lower()


def load_spacy_model_if_enabled() -> None:
    """Load the spaCy model for entity validation when enabled."""
    if config.settings.ENABLE_ENTITY_VALIDATION:
        _get_text_processing_service().load_spacy_model()
