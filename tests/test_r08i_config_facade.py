"""Prevent dynamic module attributes being frozen by monkeypatch teardown."""
import ast
from pathlib import Path

import pytest

import config


@pytest.mark.parametrize("field,value", [("LARGE_MODEL", "synthetic"), ("TOTAL_CHAPTERS", 12), ("ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)])
def test_explicit_facade_override_restores_exact_dictionary(field: str, value: object) -> None:
    before = dict(vars(config))
    bound = config.snapshot_settings()
    with pytest.MonkeyPatch.context() as overrides:
        overrides.setitem(vars(config), field, value)
        assert getattr(config, field) == value
        assert config.snapshot_settings() is bound
    assert vars(config) == before
    assert config.snapshot_settings() is bound


def test_test_config_overrides_do_not_use_dynamic_setattr() -> None:
    offenders = []
    for path in Path(__file__).parent.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "setattr" or not node.args:
                continue
            target = node.args[0]
            module = isinstance(target, ast.Name) and target.id == "config"
            dotted = isinstance(target, ast.Constant) and isinstance(target.value, str) and target.value.startswith("config.") and target.value.count(".") == 1
            if module or dotted:
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], "Use run_settings, or setitem(vars(config), ...) for an intentional facade override"
