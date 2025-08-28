# tests/test_problem_parser.py
import json

from processing.problem_parser import parse_problem_list


def test_parse_problem_list_valid():
    data = json.dumps(
        [
            {
                "issue_category": "consistency",
                "problem_description": "desc",
                "quote_from_original_text": "q",
                "suggested_fix_focus": "fix",
            }
        ]
    )
    result = parse_problem_list(data)
    assert len(result) == 1
    assert result[0]["issue_category"] == "consistency"


def test_parse_problem_list_invalid_json():
    result = parse_problem_list("notjson", category="plot")
    assert result[0]["issue_category"] == "plot"
    assert "Invalid JSON" in result[0]["problem_description"]


def test_parse_problem_list_empty():
    assert parse_problem_list("") == []
