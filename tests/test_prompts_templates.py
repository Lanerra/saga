# tests/test_prompts_templates.py
import re

import pytest
from jinja2 import (
    DictLoader,
    Environment,
    StrictUndefined,
    TemplateError,
    UndefinedError,
)

import prompts.prompt_renderer as pr


def test_narrative_system_prompt_allows_json_only_when_explicitly_requested() -> None:
    """
    Prompt contract guard for audit item 5.1 (narrative system prompt vs scene planning JSON).

    Contract:
    - Narrative agent defaults to prose-first drafting behavior.
    - If the user/template explicitly requests JSON, the system prompt must permit
      JSON-only output (no surrounding prose).
    """
    # Ensure we read current on-disk content (get_system_prompt is cached).
    pr.get_system_prompt.cache_clear()

    system_prompt = pr.get_system_prompt("narrative_agent")

    # New hierarchy: follow requested format.
    assert "Follow the output format explicitly requested" in system_prompt

    # Must explicitly permit JSON-only when requested.
    assert "Output **valid JSON only**" in system_prompt or "Output valid JSON only" in system_prompt

    # Must retain default prose guidance for drafting tasks.
    assert "continuous prose" in system_prompt
    assert "Do not wrap the story in code fences" in system_prompt


def test_initialization_system_prompt_prioritizes_json_for_structured_output() -> None:
    """
    Prompt contract guard for audit item 5.3 (initialization system prompt vs structured JSON output).

    Contract:
    - Initialization agent may have prose-oriented defaults for non-structured tasks.
    - If a task requests JSON / structured output and the runtime parses JSON,
      the system prompt must unambiguously prioritize JSON-only output (no markdown, no fences, no commentary).
    - Must NOT regress to the older ambiguous phrasing that suggests markdown is the default unless JSON is
      "specifically requested".
    """
    # Ensure we read current on-disk content (get_system_prompt is cached).
    pr.get_system_prompt.cache_clear()

    system_prompt = pr.get_system_prompt("initialization")

    # Must explicitly describe the structured-output override and JSON-only requirement.
    assert "Structured-output override" in system_prompt
    assert "valid JSON only" in system_prompt or "valid JSON" in system_prompt

    # Must explicitly forbid markdown/code fences/extraneous text for structured tasks.
    assert "No markdown" in system_prompt
    assert "No code fences" in system_prompt
    assert "No extra commentary" in system_prompt

    # Must not include the older ambiguous prose-default language.
    assert "unless a structured format like JSON is specifically requested" not in system_prompt


def test_renderer_uses_strict_undefined(monkeypatch: pytest.MonkeyPatch) -> None:
    # Use the module's env and verify missing var raises
    env = Environment(
        loader=DictLoader({"tpl.j2": "Hello {{ missing }}"}),
        undefined=StrictUndefined,
    )
    monkeypatch.setattr(pr, "_env", env)
    with pytest.raises((UndefinedError, TemplateError)):
        pr.render_prompt("tpl.j2", {})


def test_draft_scene_outputs_text_only(monkeypatch: pytest.MonkeyPatch) -> None:
    env = Environment(
        loader=DictLoader(
            {
                "narrative_agent/draft_scene.j2": ("4. Output ONLY the scene text.\n" "{{ scene_text }}\n"),
            }
        ),
        undefined=StrictUndefined,
    )
    monkeypatch.setattr(pr, "_env", env)
    out = pr.render_prompt(
        "narrative_agent/draft_scene.j2",
        {
            "scene_text": "Some prose.",
        },
    )
    assert "BEGIN SCENE" not in out


def test_extract_relationships_prompt_contract_requires_wrapper_object() -> None:
    """
    Regression guard for audit item 4.2.3 (relationship extraction wrapper drift).

    This test is intentionally a *prompt contract* test (not parser behavior): it
    renders the real template and asserts the strict output contract is present.
    """
    rendered = pr.render_prompt(
        "knowledge_agent/extract_relationships.j2",
        {
            "novel_title": "Test Novel",
            "novel_genre": "Fantasy",
            "protagonist": "Hero",
            "chapter_number": 1,
            "chapter_text": "Hero meets Bob in the Castle.",
        },
    )

    # Wrapper key must be explicitly required.
    assert '"kg_triples"' in rendered or "kg_triples" in rendered

    # JSON-only output contract must be explicit.
    assert "Return ONLY valid JSON" in rendered

    # Must forbid returning a bare list (the drift regression we want to catch).
    assert "Do NOT return a bare list" in rendered


def test_relationship_disambiguation_prompt_contract_requires_decision_object() -> None:
    """
    Prompt contract guard: relationship disambiguation MUST request strict JSON output
    matching the runtime JSON parsing contract.

    Contract:
    - JSON only (no markdown/fences/commentary).
    - Root object with EXACTLY one key: "decision".
    - "decision" MUST be EXACTLY "NORMALIZE" or "DISTINCT".
    """
    rendered = pr.render_prompt(
        "knowledge_agent/relationship_disambiguate_normalize_or_distinct.j2",
        {
            "new_type": "WORKS_WITH",
            "new_description": "Two characters collaborate on a task.",
            "existing_type": "COLLABORATES_WITH",
            "existing_usage_count": 12,
            "examples_str": "Alice COLLABORATES_WITH Bob",
        },
    )

    assert "Output MUST be valid JSON only." in rendered
    assert "No markdown." in rendered
    assert "No code fences." in rendered
    assert "No commentary." in rendered

    assert 'EXACTLY one key: "decision"' in rendered
    assert '"decision" MUST be EXACTLY one of: "NORMALIZE" or "DISTINCT".' in rendered
    assert '{"decision":"NORMALIZE"}' in rendered
    assert '{"decision":"DISTINCT"}' in rendered


def test_extract_characters_prompt_contract_requires_json_only_and_canonical_root() -> None:
    rendered = pr.render_prompt(
        "knowledge_agent/extract_characters.j2",
        {
            "novel_title": "Test Novel",
            "novel_genre": "Fantasy",
            "protagonist": "Hero",
            "chapter_number": 1,
            "chapter_text": "Hero meets Bob in the Castle.",
        },
    )

    assert re.search(r"Return\s+valid\s+JSON\s+only", rendered, flags=re.IGNORECASE)
    assert re.search(r"no\s+markdown", rendered, flags=re.IGNORECASE)
    assert re.search(r"no\s+code\s+fences", rendered, flags=re.IGNORECASE)
    assert re.search(r"no\s+commentary", rendered, flags=re.IGNORECASE)

    assert "character_updates" in rendered
    assert re.search(
        r'\{\s*"character_updates"\s*:\s*\{\s*\.\.\.\s*\}\s*\}',
        rendered,
    )


def test_extract_events_prompt_contract_requires_json_only_and_canonical_root() -> None:
    rendered = pr.render_prompt(
        "knowledge_agent/extract_events.j2",
        {
            "novel_title": "Test Novel",
            "novel_genre": "Fantasy",
            "protagonist": "Hero",
            "chapter_number": 1,
            "chapter_text": "Hero meets Bob in the Castle.",
        },
    )

    assert re.search(r"Return\s+valid\s+JSON\s+only", rendered, flags=re.IGNORECASE)
    assert re.search(r"no\s+markdown", rendered, flags=re.IGNORECASE)
    assert re.search(r"no\s+code\s+fences", rendered, flags=re.IGNORECASE)
    assert re.search(r"no\s+commentary", rendered, flags=re.IGNORECASE)

    assert "world_updates" in rendered
    assert "Event" in rendered
    assert re.search(
        r'\{\s*"world_updates"\s*:\s*\{\s*"Event"\s*:\s*\{\s*\.\.\.\s*\}\s*\}\s*\}',
        rendered,
    )


def test_extract_locations_prompt_contract_requires_json_only_and_canonical_root() -> None:
    rendered = pr.render_prompt(
        "knowledge_agent/extract_locations.j2",
        {
            "novel_title": "Test Novel",
            "novel_genre": "Fantasy",
            "protagonist": "Hero",
            "chapter_number": 1,
            "chapter_text": "Hero meets Bob in the Castle.",
        },
    )

    assert re.search(r"Return\s+valid\s+JSON\s+only", rendered, flags=re.IGNORECASE)
    assert re.search(r"no\s+markdown", rendered, flags=re.IGNORECASE)
    assert re.search(r"no\s+code\s+fences", rendered, flags=re.IGNORECASE)
    assert re.search(r"no\s+commentary", rendered, flags=re.IGNORECASE)

    assert "world_updates" in rendered
    assert "Location" in rendered
    assert re.search(
        r'\{\s*"world_updates"\s*:\s*\{\s*"Location"\s*:\s*\{\s*\.\.\.\s*\}\s*\}\s*\}',
        rendered,
    )


def test_generate_act_outline_prompt_contract_requires_json_schema() -> None:
    rendered = pr.render_prompt(
        "initialization/generate_act_outline.j2",
        {
            "title": "Test Novel",
            "genre": "Fantasy",
            "theme": "Adventure",
            "setting": "Medieval world",
            "protagonist_name": "Hero",
            "act_number": 1,
            "total_acts": 3,
            "act_role": "Setup/Introduction",
            "chapters_in_act": 7,
            "global_outline": "Global outline context",
            "character_context": "Hero: A brave warrior",
        },
    )

    assert "Return valid JSON only" in rendered
    assert "no markdown" in rendered or "No markdown" in rendered
    assert "no code fences" in rendered or "No code fences" in rendered
    assert "no commentary" in rendered or "No commentary" in rendered

    for required in [
        '"act_number"',
        '"total_acts"',
        '"act_role"',
        '"chapters_in_act"',
        '"sections"',
        '"act_summary"',
        '"opening_situation"',
        '"key_events"',
        '"character_development"',
        '"stakes_and_tension"',
        '"act_ending_turn"',
        '"thematic_thread"',
        '"pacing_notes"',
        '"sequence"',
        '"event"',
        '"cause"',
        '"effect"',
    ]:
        assert required in rendered
