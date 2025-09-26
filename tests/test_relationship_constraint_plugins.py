# tests/test_relationship_constraint_plugins.py
import core.relationship_constraints as rc
from core.relationship_constraints.constraints import CATEGORY_CONSTRAINTS


def test_constraint_category_loading():
    """Each constraint category should be loaded and registered."""
    for name, constraints in CATEGORY_CONSTRAINTS.items():
        assert constraints, f"{name} provides no constraints"
        first_rel = next(iter(constraints.keys()))
        assert first_rel in rc.RELATIONSHIP_CONSTRAINTS


def test_constraint_example_valid():
    """A simple validation check for each category's first relationship."""
    for name, constraints in CATEGORY_CONSTRAINTS.items():
        if not constraints:
            continue
        rel, detail = next(iter(constraints.items()))
        # Choose the first subject/object types that are valid node labels in the current schema.
        subj_candidates = [
            t for t in detail["valid_subject_types"] if t in rc.NODE_LABELS
        ]
        obj_candidates = [
            t for t in detail["valid_object_types"] if t in rc.NODE_LABELS
        ]
        if not subj_candidates or not obj_candidates:
            # Skip categories whose examples rely on legacy/disabled labels
            continue
        subj = subj_candidates[0]
        obj = obj_candidates[0]
        is_valid, errors = rc.validate_relationship_semantics(subj, rel, obj)
        assert is_valid, f"{name}:{rel} invalid for {subj}->{obj}: {errors}"
