import importlib
import pkgutil

import core.relationship_constraints as rc


def test_plugin_category_loading():
    """Each relationship constraint plugin should be loaded and registered."""
    import core.relationship_constraints.plugins as plugins_pkg

    for _, name, _ in pkgutil.iter_modules(plugins_pkg.__path__):
        module = importlib.import_module(
            f"core.relationship_constraints.plugins.{name}"
        )
        constraints = getattr(module, "RELATIONSHIP_CONSTRAINTS", {})
        assert constraints, f"{name} provides no constraints"
        first_rel = next(iter(constraints.keys()))
        assert first_rel in rc.RELATIONSHIP_CONSTRAINTS


def test_plugin_example_valid():
    """A simple validation check for each plugin's first relationship."""
    import core.relationship_constraints.plugins as plugins_pkg

    for _, name, _ in pkgutil.iter_modules(plugins_pkg.__path__):
        module = importlib.import_module(
            f"core.relationship_constraints.plugins.{name}"
        )
        constraints = getattr(module, "RELATIONSHIP_CONSTRAINTS", {})
        if not constraints:
            continue
        rel, detail = next(iter(constraints.items()))
        subj = next(iter(detail["valid_subject_types"]))
        obj = next(iter(detail["valid_object_types"]))
        is_valid, errors = rc.validate_relationship_semantics(subj, rel, obj)
        assert is_valid, f"{name}:{rel} invalid for {subj}->{obj}: {errors}"
