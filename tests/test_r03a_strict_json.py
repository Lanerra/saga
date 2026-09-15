"""Finite-number admission at the shared strict JSON boundary."""

import inspect
import json
import math
from pathlib import Path

import pytest

from utils.common import load_strict_json


@pytest.mark.parametrize("number", ["1e999", "-1e999", "1.8e308", "-1.8e308", "NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("container", ["%s", '{"value":%s}', "[%s]", '{"outer":[{"inner":[0,%s]}]}'])
def test_rejects_nonfinite_at_every_depth(number: str, container: str) -> None:
    with pytest.raises(ValueError, match="Nonfinite JSON number"):
        load_strict_json(container % number)


@pytest.mark.parametrize("text", ["0", "-1", "123456789012345678901234567890", "1.0", "-0.0", "1e308", "-1e308", "1.7976931348623157e308", "5e-324", "1e-999", "-1e-999", '"1e999"', '"-1e999"', '"NaN"', "true", "null"])
def test_preserves_finite_numbers_and_non_numeric_values(text: str) -> None:
    expected = json.loads(text)
    actual = load_strict_json(text)
    assert type(actual) is type(expected)
    assert actual == expected
    assert load_strict_json('{"nested":[' + text + "]}") == {"nested": [expected]}
    if isinstance(actual, float):
        assert math.isfinite(actual)
        assert math.copysign(1, actual) == math.copysign(1, expected)


def test_strict_duplicate_and_single_value_contracts_remain() -> None:
    with pytest.raises(ValueError, match="Duplicate JSON object key"):
        load_strict_json('{"nested":[{"a":1,"a":2}]}')
    with pytest.raises(json.JSONDecodeError):
        load_strict_json("1 2")
    assert Path(inspect.getfile(load_strict_json)).resolve() == Path(__file__).resolve().parents[1] / "utils/common.py"
