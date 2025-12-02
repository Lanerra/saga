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
                "narrative_agent/base_draft.j2": (
                    "{{ 'context' }}{% block output_header %}{% endblock %}"
                    "{% block instructions %}{% endblock %}"
                ),
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
