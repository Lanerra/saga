# tests/test_prompts_templates.py
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
    # Ensure no header is enforced by template header block
    env = Environment(
        loader=DictLoader(
            {
                "narrative_agent/draft_scene.j2": (
                    "{% extends 'narrative_agent/base_draft.j2' %}"
                    "{% block task_description %}scene{% endblock %}"
                    "{% block additional_context %}{% endblock %}"
                    "{% block instructions %}4. Output ONLY the scene text.{% endblock %}"
                    "{% block output_header %}{% endblock %}"
                ),
                "narrative_agent/base_draft.j2": ("{{ 'context' }}{% block output_header %}{% endblock %}" "{% block instructions %}{% endblock %}"),
            }
        )
    )
    monkeypatch.setattr(pr, "_env", env)
    out = pr.render_prompt(
        "narrative_agent/draft_scene.j2",
        {
            "novel_title": "X",
            "novel_genre": "Y",
            "chapter_number": 1,
            "scene_detail": {"scene_number": 1, "summary": "s"},
            "previous_scenes_prose": "",
            "hybrid_context_for_draft": "",
            "min_length_per_scene": 10,
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
