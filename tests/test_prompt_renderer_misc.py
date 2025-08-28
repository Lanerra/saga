# tests/test_prompt_renderer_misc.py
from jinja2 import DictLoader, Environment

import prompts.prompt_renderer


def test_render_prompt_with_custom_env(monkeypatch):
    env = Environment(
        loader=DictLoader({"greet.j2": "Hello {{ name }}"}), autoescape=False
    )
    monkeypatch.setattr(prompts.prompt_renderer, "_env", env)
    result = prompts.prompt_renderer.render_prompt("greet.j2", {"name": "Bob"})
    assert result == "Hello Bob"
