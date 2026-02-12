"""Tests for chapter allocation logic: act ranges, chapter distribution, and lookup."""

from __future__ import annotations

import pytest

from core.langgraph.initialization.chapter_allocation import (
    ActRange,
    _is_int,
    choose_act_ranges,
    compute_balanced_act_ranges,
    determine_act_for_chapter,
    extract_explicit_act_ranges,
)


class TestActRange:
    """Validate ActRange dataclass properties and methods."""

    def test_chapters_in_act_single_chapter(self) -> None:
        act_range = ActRange(act_number=1, chapters_start=3, chapters_end=3)
        assert act_range.chapters_in_act == 1

    def test_chapters_in_act_multiple_chapters(self) -> None:
        act_range = ActRange(act_number=1, chapters_start=1, chapters_end=5)
        assert act_range.chapters_in_act == 5

    def test_chapters_in_act_empty_range(self) -> None:
        act_range = ActRange(act_number=1, chapters_start=5, chapters_end=4)
        assert act_range.chapters_in_act == 0

    def test_contains_within_range(self) -> None:
        act_range = ActRange(act_number=1, chapters_start=3, chapters_end=7)
        assert act_range.contains(5) is True

    def test_contains_at_start_boundary(self) -> None:
        act_range = ActRange(act_number=1, chapters_start=3, chapters_end=7)
        assert act_range.contains(3) is True

    def test_contains_at_end_boundary(self) -> None:
        act_range = ActRange(act_number=1, chapters_start=3, chapters_end=7)
        assert act_range.contains(7) is True

    def test_contains_below_range(self) -> None:
        act_range = ActRange(act_number=1, chapters_start=3, chapters_end=7)
        assert act_range.contains(2) is False

    def test_contains_above_range(self) -> None:
        act_range = ActRange(act_number=1, chapters_start=3, chapters_end=7)
        assert act_range.contains(8) is False

    def test_frozen_dataclass(self) -> None:
        act_range = ActRange(act_number=1, chapters_start=1, chapters_end=5)
        with pytest.raises(AttributeError):
            act_range.act_number = 2  # type: ignore[misc]


class TestIsInt:
    """Validate _is_int type guard excludes booleans and non-int types."""

    def test_positive_integer(self) -> None:
        assert _is_int(42) is True

    def test_zero(self) -> None:
        assert _is_int(0) is True

    def test_negative_integer(self) -> None:
        assert _is_int(-1) is True

    def test_bool_true_excluded(self) -> None:
        assert _is_int(True) is False

    def test_bool_false_excluded(self) -> None:
        assert _is_int(False) is False

    def test_float_excluded(self) -> None:
        assert _is_int(3.0) is False

    def test_string_excluded(self) -> None:
        assert _is_int("5") is False

    def test_none_excluded(self) -> None:
        assert _is_int(None) is False


class TestExtractExplicitActRanges:
    """Validate extraction of act ranges from global outline dicts."""

    def test_valid_acts(self) -> None:
        outline = {
            "acts": [
                {"act_number": 1, "chapters_start": 1, "chapters_end": 4},
                {"act_number": 2, "chapters_start": 5, "chapters_end": 8},
                {"act_number": 3, "chapters_start": 9, "chapters_end": 12},
            ]
        }
        result = extract_explicit_act_ranges(outline)
        assert len(result) == 3
        assert result[1] == ActRange(act_number=1, chapters_start=1, chapters_end=4)
        assert result[2] == ActRange(act_number=2, chapters_start=5, chapters_end=8)
        assert result[3] == ActRange(act_number=3, chapters_start=9, chapters_end=12)

    def test_missing_acts_key(self) -> None:
        result = extract_explicit_act_ranges({})
        assert result == {}

    def test_acts_is_not_a_list(self) -> None:
        result = extract_explicit_act_ranges({"acts": "not a list"})
        assert result == {}

    def test_acts_is_none(self) -> None:
        result = extract_explicit_act_ranges({"acts": None})
        assert result == {}

    def test_act_missing_chapters_start(self) -> None:
        outline = {
            "acts": [
                {"act_number": 1, "chapters_end": 4},
            ]
        }
        result = extract_explicit_act_ranges(outline)
        assert result == {}

    def test_act_missing_chapters_end(self) -> None:
        outline = {
            "acts": [
                {"act_number": 1, "chapters_start": 1},
            ]
        }
        result = extract_explicit_act_ranges(outline)
        assert result == {}

    def test_act_missing_act_number(self) -> None:
        outline = {
            "acts": [
                {"chapters_start": 1, "chapters_end": 4},
            ]
        }
        result = extract_explicit_act_ranges(outline)
        assert result == {}

    def test_act_with_bool_fields_excluded(self) -> None:
        outline = {
            "acts": [
                {"act_number": True, "chapters_start": 1, "chapters_end": 4},
            ]
        }
        result = extract_explicit_act_ranges(outline)
        assert result == {}

    def test_non_dict_act_entries_skipped(self) -> None:
        outline = {
            "acts": [
                "not a dict",
                {"act_number": 1, "chapters_start": 1, "chapters_end": 4},
            ]
        }
        result = extract_explicit_act_ranges(outline)
        assert len(result) == 1
        assert result[1] == ActRange(act_number=1, chapters_start=1, chapters_end=4)

    def test_partial_valid_acts(self) -> None:
        outline = {
            "acts": [
                {"act_number": 1, "chapters_start": 1, "chapters_end": 4},
                {"act_number": 2, "chapters_start": "five", "chapters_end": 8},
            ]
        }
        result = extract_explicit_act_ranges(outline)
        assert len(result) == 1
        assert 1 in result
        assert 2 not in result


class TestComputeBalancedActRanges:
    """Validate balanced chapter distribution across acts."""

    def test_even_distribution(self) -> None:
        result = compute_balanced_act_ranges(total_chapters=12, act_count=3)
        assert len(result) == 3
        assert result[1] == ActRange(act_number=1, chapters_start=1, chapters_end=4)
        assert result[2] == ActRange(act_number=2, chapters_start=5, chapters_end=8)
        assert result[3] == ActRange(act_number=3, chapters_start=9, chapters_end=12)

    def test_remainder_distributed_to_earlier_acts(self) -> None:
        result = compute_balanced_act_ranges(total_chapters=10, act_count=3)
        assert result[1].chapters_in_act == 4
        assert result[2].chapters_in_act == 3
        assert result[3].chapters_in_act == 3

    def test_remainder_of_two_with_three_acts(self) -> None:
        result = compute_balanced_act_ranges(total_chapters=11, act_count=3)
        assert result[1].chapters_in_act == 4
        assert result[2].chapters_in_act == 4
        assert result[3].chapters_in_act == 3

    def test_single_act(self) -> None:
        result = compute_balanced_act_ranges(total_chapters=10, act_count=1)
        assert len(result) == 1
        assert result[1] == ActRange(act_number=1, chapters_start=1, chapters_end=10)

    def test_more_acts_than_chapters(self) -> None:
        result = compute_balanced_act_ranges(total_chapters=2, act_count=5)
        assert len(result) == 5
        assert result[1].chapters_in_act == 1
        assert result[2].chapters_in_act == 1
        assert result[3].chapters_in_act == 0
        assert result[4].chapters_in_act == 0
        assert result[5].chapters_in_act == 0

    def test_zero_chapters(self) -> None:
        result = compute_balanced_act_ranges(total_chapters=0, act_count=3)
        assert len(result) == 3
        for act_range in result.values():
            assert act_range.chapters_in_act == 0

    def test_zero_act_count_returns_empty(self) -> None:
        result = compute_balanced_act_ranges(total_chapters=10, act_count=0)
        assert result == {}

    def test_negative_act_count_returns_empty(self) -> None:
        result = compute_balanced_act_ranges(total_chapters=10, act_count=-1)
        assert result == {}

    def test_contiguous_ranges(self) -> None:
        result = compute_balanced_act_ranges(total_chapters=10, act_count=3)
        all_chapters = set()
        for act_range in result.values():
            for chapter in range(act_range.chapters_start, act_range.chapters_end + 1):
                assert chapter not in all_chapters
                all_chapters.add(chapter)
        assert all_chapters == set(range(1, 11))

    def test_one_chapter_one_act(self) -> None:
        result = compute_balanced_act_ranges(total_chapters=1, act_count=1)
        assert result[1] == ActRange(act_number=1, chapters_start=1, chapters_end=1)


class TestChooseActRanges:
    """Validate preference for explicit ranges over balanced fallback."""

    def test_prefers_explicit_when_complete(self) -> None:
        outline = {
            "act_count": 2,
            "acts": [
                {"act_number": 1, "chapters_start": 1, "chapters_end": 6},
                {"act_number": 2, "chapters_start": 7, "chapters_end": 10},
            ],
        }
        result = choose_act_ranges(outline, total_chapters=10)
        assert result[1] == ActRange(act_number=1, chapters_start=1, chapters_end=6)
        assert result[2] == ActRange(act_number=2, chapters_start=7, chapters_end=10)

    def test_falls_back_to_balanced_when_explicit_incomplete(self) -> None:
        outline = {
            "act_count": 3,
            "acts": [
                {"act_number": 1, "chapters_start": 1, "chapters_end": 4},
            ],
        }
        result = choose_act_ranges(outline, total_chapters=12)
        assert len(result) == 3
        assert result[1] == ActRange(act_number=1, chapters_start=1, chapters_end=4)
        assert result[2] == ActRange(act_number=2, chapters_start=5, chapters_end=8)
        assert result[3] == ActRange(act_number=3, chapters_start=9, chapters_end=12)

    def test_falls_back_to_balanced_when_no_acts(self) -> None:
        outline: dict = {"act_count": 3}
        result = choose_act_ranges(outline, total_chapters=9)
        assert len(result) == 3
        assert result[1] == ActRange(act_number=1, chapters_start=1, chapters_end=3)

    def test_defaults_to_three_acts_when_act_count_missing(self) -> None:
        result = choose_act_ranges({}, total_chapters=9)
        assert len(result) == 3

    def test_defaults_to_three_acts_when_act_count_invalid(self) -> None:
        result = choose_act_ranges({"act_count": "many"}, total_chapters=9)
        assert len(result) == 3

    def test_defaults_to_three_acts_when_act_count_zero(self) -> None:
        result = choose_act_ranges({"act_count": 0}, total_chapters=9)
        assert len(result) == 3

    def test_defaults_to_three_acts_when_act_count_negative(self) -> None:
        result = choose_act_ranges({"act_count": -2}, total_chapters=9)
        assert len(result) == 3

    def test_defaults_to_three_acts_when_act_count_is_bool(self) -> None:
        result = choose_act_ranges({"act_count": True}, total_chapters=9)
        assert len(result) == 3


class TestDetermineActForChapter:
    """Validate act lookup for a given chapter number."""

    def test_chapter_in_first_act(self) -> None:
        outline = {"act_count": 3}
        result = determine_act_for_chapter(outline, total_chapters=9, chapter_number=2)
        assert result == 1

    def test_chapter_in_second_act(self) -> None:
        outline = {"act_count": 3}
        result = determine_act_for_chapter(outline, total_chapters=9, chapter_number=5)
        assert result == 2

    def test_chapter_in_last_act(self) -> None:
        outline = {"act_count": 3}
        result = determine_act_for_chapter(outline, total_chapters=9, chapter_number=9)
        assert result == 3

    def test_chapter_beyond_range_clamps_to_last_act(self) -> None:
        outline = {"act_count": 3}
        result = determine_act_for_chapter(outline, total_chapters=9, chapter_number=100)
        assert result == 3

    def test_chapter_zero_clamps_to_first_act(self) -> None:
        outline = {"act_count": 3}
        result = determine_act_for_chapter(outline, total_chapters=9, chapter_number=0)
        assert result == 1

    def test_negative_chapter_clamps_to_first_act(self) -> None:
        outline = {"act_count": 3}
        result = determine_act_for_chapter(outline, total_chapters=9, chapter_number=-5)
        assert result == 1

    def test_with_explicit_ranges(self) -> None:
        outline = {
            "act_count": 2,
            "acts": [
                {"act_number": 1, "chapters_start": 1, "chapters_end": 3},
                {"act_number": 2, "chapters_start": 4, "chapters_end": 8},
            ],
        }
        assert determine_act_for_chapter(outline, total_chapters=8, chapter_number=1) == 1
        assert determine_act_for_chapter(outline, total_chapters=8, chapter_number=3) == 1
        assert determine_act_for_chapter(outline, total_chapters=8, chapter_number=4) == 2
        assert determine_act_for_chapter(outline, total_chapters=8, chapter_number=8) == 2

    def test_with_single_act(self) -> None:
        outline = {"act_count": 1}
        result = determine_act_for_chapter(outline, total_chapters=5, chapter_number=3)
        assert result == 1
