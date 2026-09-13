# core/langgraph/nodes/scene_extraction_validation.py
"""Entity validation and normalization helpers for scene extraction.

This module is separated from scene_extraction.py to keep spaCy-dependent logic
isolated, avoiding eager spaCy model loading at import time.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import structlog

import config

if TYPE_CHECKING:
    from core.langgraph.initialization.catalog import EntityCatalog
    from core.text_processing_service import TextProcessingService

logger = structlog.get_logger(__name__)

_text_processing_service: TextProcessingService | None = None
_authorized_scene_names: ContextVar[frozenset[str]] = ContextVar("authorized_scene_names", default=frozenset())


@contextmanager
def scene_name_authority(eligible_names: Collection[str]) -> Iterator[None]:
    """Scope parser lexical admission to names selected from verified scene candidates."""
    token = _authorized_scene_names.set(frozenset(eligible_names))
    try:
        yield
    finally:
        _authorized_scene_names.reset(token)


def _get_text_processing_service() -> TextProcessingService:
    """Lazily initialize the TextProcessingService to avoid loading spaCy at import time."""
    global _text_processing_service
    if _text_processing_service is None:
        # Deferred import to avoid circular references
        from core.text_processing_service import TextProcessingService

        _text_processing_service = TextProcessingService()
    return _text_processing_service


def validate_lexical_entity_name(entity_name: str) -> str:
    """Reject explicit unnamed/abstract descriptors without rewriting identity.

    These are lexical exclusions, not a classifier or proof of narrative meaning.
    Articles, conjunctions and possessives inside proper names are not exclusions.
    Source grounding is a separate, mandatory check at extraction.
    """
    if not isinstance(entity_name, str) or not entity_name or entity_name != entity_name.strip():
        raise ValueError("Entity name must be an exact nonblank name")
    if entity_name in _authorized_scene_names.get():
        return entity_name
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


def validate_named_entity(entity_name: str, eligible_names: Collection[str] = ()) -> str:
    """Require explicit eligibility; capitalization is not named-identity evidence.

    Production eligibility comes from the selected catalog, not from a relationship
    response or statistical name guessing. Missing context authorizes no names.
    """
    if not isinstance(entity_name, str) or not entity_name or entity_name != entity_name.strip():
        raise ValueError("Entity name must be an exact nonblank name")
    if entity_name not in eligible_names:
        raise ValueError("Entity name is not an eligible scene identity")
    return entity_name


def _validate_entity_with_spacy(scene_text: str, entity_name: str, eligible_names: Collection[str] = ()) -> bool:
    """Require an eligible exact span; optional NLP cannot authorize identities.

    Keep the historical entry point for callers. This identity check does not
    depend on statistical model availability or ENABLE_ENTITY_VALIDATION.
    """
    try:
        validate_named_entity(entity_name, eligible_names)
    except ValueError:
        return False
    return re.search(r"(?<!\w)" + re.escape(entity_name) + r"(?!\w)", scene_text) is not None


def scene_identity_candidates(catalog: EntityCatalog | None, scene_text: str) -> dict[str, dict[str, Any]]:
    """Project a validated catalog onto literal scene names, failing on ambiguity.

    The catalog is an explicit identity authority, not proof that arbitrary prose
    is named. Novel scene identities require upstream catalog admission; entity
    and relationship responses cannot enlarge this closed set. Catalog selection
    and project/checksum verification belong to select_catalog at the entrypoint.
    """
    if catalog is None:
        raise ValueError("Scene extraction requires an eligible identity catalog")
    candidates: dict[str, dict[str, Any]] = {}
    for candidate in catalog.candidates("Character", "Location", "Item", "Event"):
        name = candidate["name"]
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("Catalog candidate requires an exact nonblank name")
        if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", scene_text) is None:
            continue
        if name in candidates:
            raise ValueError("Ambiguous eligible scene identity; name selects multiple catalog IDs")
        candidates[name] = {"name": name, "id": candidate["id"], "label": candidate["label"]}
    return candidates


def _get_normalized_entity_key(name: str) -> str:
    """Use literal identity; linguistic similarity is not an alias contract."""
    return name


def load_spacy_model_if_enabled() -> None:
    """Load the spaCy model for entity validation when enabled."""
    if config.settings.ENABLE_ENTITY_VALIDATION:
        _get_text_processing_service().load_spacy_model()
