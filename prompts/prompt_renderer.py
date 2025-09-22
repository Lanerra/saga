# prompts/prompt_renderer.py
"""Utilities for rendering LLM prompts using Jinja2 templates and system prompts."""

from functools import lru_cache
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


@lru_cache(maxsize=16)
def get_system_prompt(agent_name: str) -> str:
    """Load a per-agent system prompt from prompts/<agent_name>/system.md.

    Returns an empty string if no system prompt exists, allowing callers
    to omit the system message gracefully.
    """
    system_path = PROMPTS_PATH / agent_name / "system.md"
    try:
        if system_path.exists():
            return system_path.read_text(encoding="utf-8").strip()
    except Exception:
        # Fail open: return empty string if read fails
        return ""
    return ""
