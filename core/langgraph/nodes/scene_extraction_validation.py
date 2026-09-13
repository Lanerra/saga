# core/langgraph/nodes/scene_extraction_validation.py
"""Entity validation and normalization helpers for scene extraction.

This module is separated from scene_extraction.py to keep spaCy-dependent logic
isolated, avoiding eager spaCy model loading at import time.
"""

from __future__ import annotations

import re
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


def validate_named_entity(entity_name: str) -> str:
    """Reject explicit unnamed/abstract descriptors without rewriting identity.

    These are lexical exclusions, not a classifier or proof of narrative meaning.
    Articles, conjunctions and possessives inside proper names are not exclusions.
    Source grounding is a separate, mandatory check at extraction.
    """
    if not isinstance(entity_name, str) or not entity_name or entity_name != entity_name.strip():
        raise ValueError("Entity name must be an exact nonblank name")
    words = entity_name.split()
    content = words[1:] if words[0].casefold() in {"the", "a", "an"} else words
    descriptor = " ".join(content).casefold()
    excluded = {
        "truth", "knowledge", "secret", "secrets", "understanding", "consequences",
        "fear", "presence", "power", "meaning", "escape", "guilt", "communication",
        "protection", "survival", "group", "entity", "survivors", "force", "hunters",
        "villagers", "unknown dangers", "mysterious entity",
    }
    if not content or descriptor in excluded or words[0].casefold() == "to":
        raise ValueError("Entity name is an unnamed or abstract descriptor")
    if not any(character.isupper() for word in content for character in word):
        raise ValueError("Entity name must identify a named entity, not a lowercase descriptor")
    possessive = re.search(r"(?:['’]s|s['’])\s+(.+)$", entity_name)
    if possessive:
        attribute = possessive.group(1)
        if attribute.casefold() in {"father", "mother", "parent", "parents", "brother", "sister", "family"} or attribute[0].islower():
            raise ValueError("Entity name is an unnamed possessive descriptor")
    return entity_name


def _validate_entity_with_spacy(scene_text: str, entity_name: str) -> bool:
    """Require an exact named span; optional NLP must not authorize fuzzy aliases.

    Keep the historical entry point for callers. This identity check does not
    depend on statistical model availability or ENABLE_ENTITY_VALIDATION.
    """
    try:
        validate_named_entity(entity_name)
    except ValueError:
        return False
    return re.search(r"(?<!\w)" + re.escape(entity_name) + r"(?!\w)", scene_text) is not None


def _get_normalized_entity_key(name: str) -> str:
    """Use literal identity; linguistic similarity is not an alias contract."""
    return name


def load_spacy_model_if_enabled() -> None:
    """Load the spaCy model for entity validation when enabled."""
    if config.settings.ENABLE_ENTITY_VALIDATION:
        _get_text_processing_service().load_spacy_model()
