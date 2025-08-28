# tests/test_helpers_misc.py
import config
from utils.helpers import _is_fill_in


def test_is_fill_in_helper():
    assert not _is_fill_in("abc")
    assert _is_fill_in(config.FILL_IN)
