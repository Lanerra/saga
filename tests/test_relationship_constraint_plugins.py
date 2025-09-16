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
        subj = next(iter(detail["valid_subject_types"]))
        obj = next(iter(detail["valid_object_types"]))
        is_valid, errors = rc.validate_relationship_semantics(subj, rel, obj)
        assert is_valid, f"{name}:{rel} invalid for {subj}->{obj}: {errors}"
