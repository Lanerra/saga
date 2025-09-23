# tests/test_evaluation_parsing.py
import pytest
import config
from processing.problem_parser import parse_problem_list
from agents.revision_agent import RevisionAgent


def test_problem_parser_valid_list():
    data = '[{"issue_category": "plot_arc", "problem_description": "d", "quote_from_original_text": "q", "suggested_fix_focus": "f"}]'
    problems = parse_problem_list(data)
    assert problems and problems[0]["issue_category"] == "plot_arc"


def test_problem_parser_empty_returns_empty():
    assert parse_problem_list("") == []


def test_problem_parser_invalid_returns_meta_error():
    problems = parse_problem_list("notjson", category="consistency")
    assert problems and problems[0]["issue_category"] == "consistency"


@pytest.mark.asyncio
async def test_revision_agent_consistency_parser_handles_invalid():
    agent = RevisionAgent(config)
    problems = await agent._parse_llm_continuity_output("notjson", 1, "text")
    assert problems and problems[0]["issue_category"] == "consistency"
