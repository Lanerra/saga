# tests/test_langgraph/test_content_manager_act_outlines_dual_read.py
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from core.langgraph.content_manager import ContentManager, get_act_outlines


def test_get_act_outlines_v1_int_keys(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))
    manager.load_json = Mock(
        return_value={
            2: {"act_number": 2, "act_role": "confrontation"},
            1: {"act_number": 1, "act_role": "setup"},
        }
    )

    state = {"act_outlines_ref": {"path": "ignored.json"}}
    act_outlines = get_act_outlines(state, manager)

    assert list(act_outlines.keys()) == [1, 2]
    assert act_outlines[1]["act_role"] == "setup"
    assert act_outlines[2]["act_role"] == "confrontation"


def test_get_act_outlines_v1_string_keys(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))
    manager.load_json = Mock(
        return_value={
            "2": {"act_number": 2, "act_role": "confrontation"},
            "1": {"act_number": 1, "act_role": "setup"},
        }
    )

    state = {"act_outlines_ref": {"path": "ignored.json"}}
    act_outlines = get_act_outlines(state, manager)

    assert list(act_outlines.keys()) == [1, 2]
    assert act_outlines[1]["act_role"] == "setup"
    assert act_outlines[2]["act_role"] == "confrontation"


def test_get_act_outlines_v2_list(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))
    manager.load_json = Mock(
        return_value=[
            {"act_number": 2, "act_role": "confrontation"},
            {"act_number": 1, "act_role": "setup"},
        ]
    )

    state = {"act_outlines_ref": {"path": "ignored.json"}}
    act_outlines = get_act_outlines(state, manager)

    assert list(act_outlines.keys()) == [1, 2]
    assert act_outlines[1]["act_role"] == "setup"
    assert act_outlines[2]["act_role"] == "confrontation"


def test_get_act_outlines_v2_container(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))
    manager.load_json = Mock(
        return_value={
            "format_version": 2,
            "acts": [
                {"act_number": 2, "act_role": "confrontation"},
                {"act_number": 1, "act_role": "setup"},
            ],
        }
    )

    state = {"act_outlines_ref": {"path": "ignored.json"}}
    act_outlines = get_act_outlines(state, manager)

    assert list(act_outlines.keys()) == [1, 2]
    assert act_outlines[1]["act_role"] == "setup"
    assert act_outlines[2]["act_role"] == "confrontation"


def test_get_act_outlines_rejects_duplicate_act_numbers_in_v2(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))
    manager.load_json = Mock(
        return_value=[
            {"act_number": 1, "act_role": "setup"},
            {"act_number": 1, "act_role": "setup"},
        ]
    )

    state = {"act_outlines_ref": {"path": "ignored.json"}}
    with pytest.raises(ValueError, match="Duplicate act_number"):
        get_act_outlines(state, manager)
