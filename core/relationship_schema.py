"""Finite producer choices derived from the unchanged strict admission rules."""

from core.relationship_validation import validate_relationship_semantics_strict
from models.kg_constants import RELATIONSHIP_TYPES


def allowed_relationship_types(source_label: str, target_label: str) -> list[str]:
    """Preserve all canonical predicates accepted by strict admission, including unruled ones."""
    return sorted(
        predicate for predicate in RELATIONSHIP_TYPES
        if validate_relationship_semantics_strict(predicate, source_label, target_label)[0]
    )
