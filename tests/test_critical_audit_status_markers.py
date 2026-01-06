# tests/test_critical_audit_status_markers.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_STATUS_PATTERN = re.compile(r"^\*\*Status:\*\* (✅ Completed|🟡 Incomplete|🔴 Missing|🟣 Ambiguous \(decision required\))$")
_DECISION_NEEDED_PATTERN = re.compile(r"^\*\*Decision needed:\*\* .+$")
_ACTIONABLE_HEADING_PATTERN = re.compile(r"^###\s+(?:\d+\)|[A-Z]\))\s+")
_ACTIONABLE_LIST_ITEM_PATTERN = re.compile(r"^\d+\)\s+")


@dataclass(frozen=True)
class ActionableItem:
    line_number: int
    line_text: str


@dataclass(frozen=True)
class StatusProblem:
    line_number: int
    line_text: str
    message: str


def _iter_nonempty_lines(
    lines: list[str],
    start_index: int,
    limit: int,
) -> list[tuple[int, str]]:
    results: list[tuple[int, str]] = []
    for index in range(start_index, len(lines)):
        if len(results) >= limit:
            break
        text = lines[index].strip()
        if text == "":
            continue
        results.append((index, lines[index].rstrip("\n")))
    return results


def _find_status_line(
    lines: list[str],
    item_index: int,
) -> tuple[int, str] | None:
    for index, line in _iter_nonempty_lines(lines, start_index=item_index + 1, limit=3):
        if _STATUS_PATTERN.fullmatch(line.strip()) is not None:
            return index, line
    return None


def _find_decision_needed_line(
    lines: list[str],
    status_index: int,
) -> tuple[int, str] | None:
    for index, line in _iter_nonempty_lines(lines, start_index=status_index + 1, limit=3):
        if _DECISION_NEEDED_PATTERN.fullmatch(line.strip()) is not None:
            return index, line
    return None


def _extract_actionable_items(markdown: str) -> list[ActionableItem]:
    lines = markdown.splitlines()
    actionable_items: list[ActionableItem] = []

    inside_code_fence = False
    for index, raw_line in enumerate(lines):
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            inside_code_fence = not inside_code_fence
            continue

        if inside_code_fence:
            continue

        if _ACTIONABLE_HEADING_PATTERN.match(stripped) is not None:
            actionable_items.append(ActionableItem(line_number=index + 1, line_text=line))
            continue

        if _ACTIONABLE_LIST_ITEM_PATTERN.match(stripped) is not None:
            actionable_items.append(ActionableItem(line_number=index + 1, line_text=line))
            continue

    return actionable_items


def _validate_audit_markdown(markdown: str) -> list[StatusProblem]:
    lines = markdown.splitlines()
    problems: list[StatusProblem] = []

    actionable_items = _extract_actionable_items(markdown)
    for item in actionable_items:
        item_index = item.line_number - 1
        status_result = _find_status_line(lines, item_index=item_index)
        if status_result is None:
            problems.append(
                StatusProblem(
                    line_number=item.line_number,
                    line_text=item.line_text,
                    message="Missing **Status:** line within the next 1–3 non-empty lines.",
                )
            )
            continue

        status_index, status_line = status_result
        if status_line.strip() == "**Status:** 🟣 Ambiguous (decision required)":
            decision_result = _find_decision_needed_line(lines, status_index=status_index)
            if decision_result is None:
                problems.append(
                    StatusProblem(
                        line_number=status_index + 1,
                        line_text=status_line,
                        message="Missing **Decision needed:** line within the next 1–3 non-empty lines after an 🟣 status.",
                    )
                )

    return problems


def _format_problems(problems: list[StatusProblem]) -> str:
    lines: list[str] = []
    for problem in problems:
        lines.append(f"- Line {problem.line_number}: {problem.message}")
        lines.append(f"  Item: {problem.line_text}")
    return "\n".join(lines)


def test_critical_audit_all_actionable_items_have_status_markers() -> None:
    markdown_path = Path("docs/critical-audit.md")
    markdown = markdown_path.read_text(encoding="utf-8")

    problems = _validate_audit_markdown(markdown)
    assert problems == [], _format_problems(problems)


def test_missing_status_marker_is_reported() -> None:
    markdown = "\n".join(
        [
            "# Title",
            "",
            "### 1) Actionable heading",
            "",
            "**Symptoms**",
        ]
    )

    problems = _validate_audit_markdown(markdown)
    assert problems != []
    assert problems[0].message == "Missing **Status:** line within the next 1–3 non-empty lines."


def test_blank_lines_between_item_and_status_are_allowed_and_title_is_ignored() -> None:
    markdown = "\n".join(
        [
            "# Document title",
            "",
            "### 1) Actionable heading",
            "",
            "",
            "**Status:** ✅ Completed",
            "",
            "1) Another actionable item",
            "",
            "",
            "**Status:** 🟣 Ambiguous (decision required)",
            "**Decision needed:** Decide something important.",
        ]
    )

    problems = _validate_audit_markdown(markdown)
    assert problems == [], _format_problems(problems)
