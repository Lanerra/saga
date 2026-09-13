"""Selected geometry must not be repaired into a different partition."""
from typing import Any

import pytest

from core.langgraph.initialization import commit_init_node
from core.langgraph.initialization.chapter_allocation import choose_act_ranges, determine_act_for_chapter
from tests.test_r05_initialization_contracts import global_response


@pytest.mark.parametrize("case", ["missing_range", "duplicate", "reversed", "overlap", "out_of_bounds", "wrong_type"])
def test_selected_ranges_are_not_rebalanced(case: str) -> None:
    outline: dict[str, Any] = global_response()
    if case == "missing_range":
        del outline["acts"][0]["chapters_start"]
    elif case == "duplicate":
        outline["acts"].append(outline["acts"][0])
    elif case == "reversed":
        outline["acts"].reverse()
    elif case == "overlap":
        outline["acts"][1]["chapters_start"] = 1
    elif case == "out_of_bounds":
        outline["acts"][-1]["chapters_end"] = 4
    else:
        outline["acts"][0]["chapters_start"] = True
    with pytest.raises(ValueError):
        choose_act_ranges(outline, 3)


@pytest.mark.parametrize("number", [0, 4, True])
def test_requested_chapter_is_not_clamped(number: int) -> None:
    with pytest.raises(ValueError):
        determine_act_for_chapter(global_response(), 3, number)


def test_abandoned_direct_writer_is_not_an_initialization_entrypoint() -> None:
    assert not hasattr(commit_init_node, "_legacy_commit_initialization_to_graph")
