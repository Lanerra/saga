# prompts/prompt_renderer.py
"""Utilities for rendering LLM prompts using Jinja2 templates."""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

import config

PROMPTS_PATH = Path(__file__).parent
_env = Environment(
    loader=FileSystemLoader(PROMPTS_PATH),
    autoescape=False,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_prompt(template_name: str, context: dict[str, Any]) -> str:
    """Render a Jinja2 template from the prompts directory."""
    template = _env.get_template(template_name)
    # Always include config in template context
    template_context = {"config": config, **context}
    return template.render(**template_context)
